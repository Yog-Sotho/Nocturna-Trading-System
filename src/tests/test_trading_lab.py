"""
Tests for the strategy lab: where its bars come from, the runs it computes,
the queue of runs and the process that executes them.
"""

import math
import sys
import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from src.tests.trading_fixtures import BIL, IEF, SPY, daily_bars
from src.trading import lab
from src.trading.alerts import AlertDispatcher
from src.trading.domain import AssetClass
from src.trading.profile import parse
from src.trading.runtime import TradingNotConfiguredError
from src.trading.validation import history, market, runner, sources

T0 = datetime(2026, 10, 23, 12, tzinfo=UTC)
TODAY = T0.date()
EQUITY = AssetClass.EQUITY
BARS = {"SPY": daily_bars(SPY, date(2014, 1, 2), date(2025, 12, 31), 100.0, seed=1),
        "IEF": daily_bars(IEF, date(2014, 1, 2), date(2025, 12, 31), 80.0, seed=2, drift=0.0001, vol=0.004),
        "BIL": daily_bars(BIL, date(2014, 1, 2), date(2025, 12, 31), 91.0, seed=5, vol=0.0001)}
BENCHMARK = {"mix": [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "1"}]}


@pytest.fixture
def synthetic(monkeypatch):
    """Alpaca's bars, made up: the symbols asked for, from 2014 to 2025."""
    calls = []

    def load(settings, symbols, start, end):
        calls.append((tuple(s for s, _ in symbols), start, end))
        return {"SPY": SPY, "IEF": IEF, "BIL": BIL}, {s: BARS[s] for s, _ in symbols}

    monkeypatch.setattr(market, "load", load)
    return calls


@pytest.fixture
def settings(app):
    return app.extensions["settings"]


class Collect:
    name = "collect"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)


# -- where the bars come from -------------------------------------------------------------


class TestSources:
    def test_alpaca(self, settings, synthetic):
        start, end = datetime(2016, 1, 4, tzinfo=UTC), datetime(2017, 1, 3, tzinfo=UTC)
        data = sources.load(settings, [("SPY", EQUITY), ("BIL", EQUITY)], start, end)
        assert set(data.bars) == {"SPY", "BIL"} and data.sources == []
        assert (data.data_label, data.hurdle_label) == ("Alpaca daily bars", "T-bills (BIL)")
        assert synthetic == [(("SPY", "BIL"), start, end)]
        assert all(-0.01 < r < 0.01 for r in data.hurdle().values())

    def test_t_bills_are_added_once(self):
        assert sources.with_tbills([("SPY", EQUITY)]) == (("SPY", EQUITY), ("BIL", EQUITY))
        assert sources.with_tbills([("BIL", EQUITY), ("SPY", EQUITY)]) == (("BIL", EQUITY), ("SPY", EQUITY))

    def test_an_unknown_source(self, settings):
        with pytest.raises(sources.DataError, match="unknown data source 'yahoo'; known: alpaca, history"):
            sources.load(settings, [("SPY", EQUITY)], T0 - timedelta(days=30), T0, "yahoo")

    def test_the_long_history_keeps_the_symbols_asked_for(self, settings, monkeypatch):
        source = history.Source("https://example.test/french.zip", "ab" * 32, date(1926, 7, 1), date(2026, 8, 31))
        monkeypatch.setattr(history, "load", lambda start, end: ({"SPY": SPY, "IEF": IEF, "BIL": BIL}, BARS,
                                                                 [source]))
        data = sources.load(settings, [("SPY", EQUITY), ("BIL", EQUITY)], T0 - timedelta(days=30), T0,
                            sources.HISTORY)
        assert set(data.instruments) == set(data.bars) == {"SPY", "BIL"} and data.sources == [source]
        assert data.hurdle_label == "T-bills (French's 1-month rate, as BIL)"

    def test_the_long_history_refuses_other_symbols_before_downloading(self, settings, monkeypatch):
        monkeypatch.setattr(history, "load", lambda start, end: pytest.fail("downloaded"))
        with pytest.raises(sources.DataError, match="proxies for SPY, IEF, BIL only, not BTC/USD"):
            sources.load(settings, [("SPY", EQUITY), ("BTC/USD", AssetClass.CRYPTO)], T0 - timedelta(days=9), T0,
                         sources.HISTORY)

    @pytest.mark.parametrize("error", [ValueError("ends on 2026-08-31"), OSError("connection reset")])
    def test_long_history_failures(self, settings, monkeypatch, error):
        def load(start, end):
            raise error

        monkeypatch.setattr(history, "load", load)
        with pytest.raises(sources.DataError, match=f"long-history data: {error}"):
            sources.load(settings, [("SPY", EQUITY)], T0 - timedelta(days=30), T0, sources.HISTORY)


# -- what a run computes ------------------------------------------------------------------


class TestRunner:
    def test_thin_keeps_the_last_mark(self):
        curve = [(T0 + timedelta(days=i), Decimal(100 + i)) for i in range(1500)]
        thinned = runner.thin(curve, 600)
        assert len(thinned) <= 601 and thinned[0] == {"timestamp": T0.isoformat(), "equity": "100"}
        assert thinned[-1]["equity"] == "1599"
        assert runner.thin(curve[:3]) == [{"timestamp": (T0 + timedelta(days=i)).isoformat(),
                                           "equity": str(100 + i)} for i in range(3)]
        assert runner.thin([]) == []

    def test_a_mix_is_backtested_with_its_risk_limits(self, settings, synthetic, monkeypatch):
        seen = {}
        run = runner.run_backtest

        def spy(allocations, bars, config):
            seen["config"], seen["names"] = config, [a.strategy.name for a in allocations]
            return run(allocations, bars, config)

        monkeypatch.setattr(runner, "run_backtest", spy)
        profile = parse({**BENCHMARK, "risk": {"risk_max_drawdown": "0.2", "sizing_target_volatility": "0"}})
        start, end = datetime(2016, 1, 4, tzinfo=UTC), datetime(2018, 1, 2, tzinfo=UTC)
        result = runner.backtest_mix(settings, profile, start, end)
        [(symbols, loaded_from, _)] = synthetic
        assert symbols == ("SPY", "IEF", "BIL") and loaded_from == start - timedelta(days=550)
        config = seen["config"]
        assert config.risk_limits.max_drawdown == Decimal("0.2") and config.sizing.target_volatility is None
        assert config.cash_sleeve == BIL and config.resume_after_kill == timedelta(days=30)
        assert seen["names"] == ["benchmark"]
        assert result.summary.periods > 400 and result.kill_switch_trips == 0
        assert [y["year"] for y in result.years()] == [2016, 2017]
        growth = math.prod(1 + y["return"] for y in result.years())
        assert growth == pytest.approx(1 + result.summary.total_return)

    def test_a_validation_keeps_the_out_of_sample_curve(self, settings, synthetic):
        report, data = runner.validate_candidate(settings, "benchmark", datetime(2016, 1, 4, tzinfo=UTC),
                                                 datetime(2021, 1, 1, tzinfo=UTC))
        assert report.candidate == "benchmark" and data.data_label == "Alpaca daily bars"
        assert report.out_of_sample_curve and report.out_of_sample_curve[0][0] >= datetime(2019, 1, 1, tzinfo=UTC)
        assert "out_of_sample_curve" not in report.to_dict()


# -- what can be run ----------------------------------------------------------------------


class TestParseRequest:
    def test_a_validation(self, settings):
        kind, spec = lab.parse_request(settings, {"kind": "validate", "strategy": "trend-filter",
                                                  "start": "1990-01-02", "data": "history"}, TODAY)
        assert kind == "validate"
        assert spec == {"start": "1990-01-02", "end": "2026-10-23", "data": "history", "strategy": "trend-filter"}
        assert lab.title(kind, spec) == "Validate trend-filter (history, 1990-01-02 to 2026-10-23)"

    def test_a_backtest(self, settings):
        body = {"kind": "backtest", "start": "2017-01-03", "end": "2020-01-02", "strategy": None,
                "mix": [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "0.5"},
                        {"strategy": "tsmom", "params": {"rule": "sign", "long_only": True}, "weight": "0.25"}],
                "risk": {"risk_max_drawdown": "0.15"}}
        kind, spec = lab.parse_request(settings, body, TODAY)
        assert kind == "backtest" and spec["data"] == "alpaca" and spec["end"] == "2020-01-02"
        assert spec["profile"] == {"mix": [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "0.5000"},
                                           {"strategy": "tsmom", "params": {"rule": "sign", "long_only": True},
                                            "weight": "0.2500"}],
                                   "risk": {"risk_max_drawdown": "0.15"}}
        assert lab.title(kind, spec) == "Backtest 50% benchmark, 25% tsmom (alpaca, 2017-01-03 to 2020-01-02)"
        assert lab.parse(spec["profile"]).mix == lab.parse(body).mix  # what is stored is what runs

    @pytest.mark.parametrize("body, message", [
        ([], "the body must be a JSON object"),
        ({"kind": "optimise"}, "kind must be one of: backtest, validate"),
        ({"kind": "backtest", "strategy": "tsmom", "start": "2017-01-03"}, "unknown fields for a backtest: strategy"),
        ({"kind": "validate", "mix": [], "start": "2017-01-03"}, "unknown fields for a validate: mix"),
        ({"kind": "validate", "strategy": "tsmom"}, "start must be a date"),
        ({"kind": "validate", "strategy": "tsmom", "start": "3 Jan 2017"}, "start must be a date (YYYY-MM-DD), not"),
        ({"kind": "validate", "strategy": "tsmom", "start": "1969-12-31"}, "start must not be before 1970-01-01"),
        ({"kind": "validate", "strategy": "tsmom", "start": "2017-01-03", "end": "2026-10-24"},
         "end must not be after today"),
        ({"kind": "validate", "strategy": "tsmom", "start": "2017-01-03", "end": "2017-01-03"},
         "end must be after start"),
        ({"kind": "validate", "strategy": "tsmom", "start": "2017-01-03", "data": "yahoo"},
         "data must be one of: alpaca, history"),
        ({"kind": "validate", "strategy": "martingale", "start": "2017-01-03"}, "unknown strategy 'martingale'"),
        ({"kind": "validate", "strategy": "tsmom", "start": "2023-01-03"}, "a validation needs at least 4 years"),
        ({"kind": "backtest", **BENCHMARK, "start": "2026-10-01"}, "a backtest needs at least 30 days"),
        ({"kind": "backtest", "mix": [], "start": "2017-01-03"}, "the mix needs at least one strategy"),
        ({"kind": "backtest", **BENCHMARK, "risk": {"risk_max_drawdown": "0.04"}, "start": "2017-01-03"},
         "start shrinking before the kill switch"),
        ({"kind": "validate", "strategy": "crypto-trend", "start": "1990-01-02", "data": "history"},
         "proxies for SPY, IEF, BIL only, not BTC/USD, ETH/USD, SOL/USD"),
    ])
    def test_what_cannot_be_run(self, settings, body, message):
        with pytest.raises(lab.LabError) as raised:
            lab.parse_request(settings, body, TODAY)
        assert message in str(raised.value)


# -- the queue ----------------------------------------------------------------------------


def _submit(app, kind="validate", spec=None, at=T0):
    spec = spec or {"start": "2017-01-03", "end": "2026-10-23", "data": "alpaca", "strategy": "tsmom"}
    return lab.submit(app, kind, spec, created_by="adminuser", now=at)


class FakeLauncher:
    def __init__(self):
        self.launches = 0

    def launch(self):
        self.launches += 1


class Clock:
    def __init__(self, start=T0):
        self.t = start

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


class TestQueue:
    def test_runs_are_queued_until_too_many_wait(self, app):
        first = _submit(app)
        assert first["status"] == "queued" and first["title"].startswith("Validate tsmom")
        assert (first["created_by"], first["headline"], first["started_at"]) == ("adminuser", None, None)
        _submit(app)
        _submit(app)
        with pytest.raises(lab.LabBusyError, match="3 lab runs are already waiting or running"):
            _submit(app)
        assert [r["id"] for r in lab.runs(app)] == [first["id"] + 2, first["id"] + 1, first["id"]]
        assert len(lab.runs(app, limit=2)) == 2

    def test_drain_runs_them_oldest_first_and_alerts(self, app, settings, monkeypatch):
        done = []

        def execute(settings, kind, spec):
            done.append(spec["strategy"])
            if spec["strategy"] == "low-vol":
                raise sources.DataError("long-history data: ends on 2026-08-31")
            if spec["strategy"] == "month-end":
                raise KeyError("boom")
            return {"headline": "rejected: fails out-of-sample Sharpe", "passed": False}, "# Validation: tsmom"

        monkeypatch.setattr(lab, "execute", execute)
        ids = [_submit(app, spec={"start": "2017-01-03", "end": "2026-10-23", "data": "alpaca", "strategy": s})["id"]
               for s in ("tsmom", "low-vol", "month-end")]
        sink = Collect()
        alerts = AlertDispatcher([sink], retry_delays=())
        assert lab.drain(app, settings, alerts, now=Clock()) == 3
        assert done == ["tsmom", "low-vol", "month-end"]
        ok, data_error, crash = (lab.get(app, i) for i in ids)
        assert ok["status"] == "done" and ok["headline"] == "rejected: fails out-of-sample Sharpe"
        assert ok["report"] == "# Validation: tsmom" and ok["result"]["passed"] is False and ok["error"] is None
        assert ok["started_at"] < ok["finished_at"]
        assert data_error["status"] == "failed" and data_error["error"] == "long-history data: ends on 2026-08-31"
        assert crash["status"] == "failed" and crash["error"] == "KeyError: 'boom'" and crash["result"] is None
        assert alerts.flush(5)
        assert [a.title for a in sink.alerts] == [f"Lab run #{ids[0]} finished", f"Lab run #{ids[1]} failed",
                                                  f"Lab run #{ids[2]} failed"]
        assert all(a.category == "lab" for a in sink.alerts)
        assert sink.alerts[0].message.endswith(": rejected: fails out-of-sample Sharpe")
        assert lab.drain(app, settings, alerts) == 0

    def test_a_long_run_sends_its_heartbeat(self, app, settings, monkeypatch):
        def execute(settings, kind, spec):
            time.sleep(0.3)
            return {"headline": "x"}, None

        monkeypatch.setattr(lab, "execute", execute)
        run_id = _submit(app)["id"]
        beats = []
        real = lab._beat
        monkeypatch.setattr(lab, "_beat", lambda app, run_id, now: (beats.append(now), real(app, run_id, now)))
        lab.drain(app, settings, now=Clock(), heartbeat_every=0.05)
        assert len(beats) >= 2 and lab.get(app, run_id)["status"] == "done"

    def test_a_failing_heartbeat_does_not_stop_the_run(self, app, settings, monkeypatch, caplog):
        def execute(settings, kind, spec):
            time.sleep(0.15)
            return {"headline": "x"}, None

        def broken(app, run_id, now):
            raise RuntimeError("database gone")

        monkeypatch.setattr(lab, "execute", execute)
        monkeypatch.setattr(lab, "_beat", broken)
        run_id = _submit(app)["id"]
        lab.drain(app, settings, heartbeat_every=0.03)
        assert lab.get(app, run_id)["status"] == "done" and "Lab heartbeat failed" in caplog.text

    def test_a_run_is_claimed_once(self, app):
        first, second = _submit(app)["id"], _submit(app)["id"]
        assert lab._claim(app, T0)[0] == first
        assert lab._claim(app, T0)[0] == second
        assert lab._claim(app, T0) is None

    def test_a_run_whose_process_died_is_failed(self, app):
        stale, fresh = _submit(app)["id"], _submit(app)["id"]
        lab._claim(app, T0)
        lab._claim(app, T0 + timedelta(minutes=4))
        assert lab.expire_stale(app, T0 + timedelta(minutes=6)) == 1
        died, alive = lab.get(app, stale), lab.get(app, fresh)
        assert died["status"] == "failed" and "lab process stopped before the run finished" in died["error"]
        assert alive["status"] == "running"
        assert lab.running_fresh(app, T0 + timedelta(minutes=6))
        assert not lab.running_fresh(app, T0 + timedelta(minutes=10))

    def test_kick_starts_a_process_only_when_one_is_needed(self, app, monkeypatch):
        launcher = FakeLauncher()
        monkeypatch.setitem(app.extensions, "lab", launcher)
        lab.kick(app, T0)
        assert launcher.launches == 0  # nothing queued
        _submit(app)
        lab.kick(app, T0)
        assert launcher.launches == 1
        lab._claim(app, T0)
        _submit(app)
        lab.kick(app, T0 + timedelta(minutes=1))
        assert launcher.launches == 1  # the running process takes it next
        lab.kick(app, T0 + timedelta(minutes=6))
        assert launcher.launches == 2  # that process died: its run failed, a new one is started

    def test_the_newest_verdict_of_each_strategy(self, app):
        def finished(strategy, passed, status="done", kind="validate"):
            spec = {"start": "2017-01-03", "end": "2026-10-23", "data": "alpaca", "strategy": strategy}
            if kind == "backtest":
                spec = {"start": "2017-01-03", "end": "2026-10-23", "data": "alpaca", "profile": BENCHMARK}
            run_id = lab.submit(app, kind, spec, "adminuser", T0)["id"]
            lab._claim(app, T0)
            lab._finish(app, run_id, T0, result={"headline": f"{strategy} {passed}", "passed": passed},
                        error=None if status == "done" else "failed")
            return run_id

        finished("tsmom", False)
        newest = finished("tsmom", True)
        finished("low-vol", True, status="failed")
        assert lab.latest_verdicts(app) == {"tsmom": {
            "run_id": newest, "passed": True, "verdict": "tsmom True", "data": "alpaca", "start": "2017-01-03",
            "end": "2026-10-23", "finished_at": T0.isoformat()}}
        finished("x", True, kind="backtest")
        assert set(lab.latest_verdicts(app)) == {"tsmom"}

    def test_an_unknown_run(self, app):
        assert lab.get(app, 999) is None
        lab._finish(app, 999, T0, error="x")  # nothing to finish


class TestExecute:
    def test_a_backtest(self, settings, synthetic):
        _, spec = lab.parse_request(settings, {"kind": "backtest", **BENCHMARK, "start": "2016-01-04",
                                               "end": "2017-06-01"}, TODAY)
        result, report = lab.execute(settings, "backtest", spec)
        assert report is None
        assert result["headline"].startswith(f"{result['summary']['cagr']:.2%} a year, Sharpe ")
        assert "T-bills (BIL) returned" in result["headline"]
        assert result["data"] == "Alpaca daily bars" and result["starting_cash"] == "100000"
        assert 0 < len(result["curve"]) <= 601 and [y["year"] for y in result["years"]] == [2016, 2017]
        assert result["final_equity"] == result["curve"][-1]["equity"] and result["fills"] > 0

    def test_a_validation(self, settings, synthetic):
        _, spec = lab.parse_request(settings, {"kind": "validate", "strategy": "benchmark", "start": "2016-01-04",
                                               "end": "2021-01-01"}, TODAY)
        result, report = lab.execute(settings, "validate", spec)
        assert report.startswith("# Validation: benchmark")
        assert result["headline"] == result["report"]["verdict"] and isinstance(result["passed"], bool)
        assert result["curve"] and result["report"]["candidate"] == "benchmark"

    def test_without_alpaca_keys(self, settings, monkeypatch):
        keyless = settings.model_copy(update={"alpaca_api_key": None, "alpaca_secret_key": None})
        _, spec = lab.parse_request(keyless, {"kind": "backtest", **BENCHMARK, "start": "2016-01-04"}, TODAY)
        with pytest.raises(TradingNotConfiguredError):
            lab.execute(keyless, "backtest", spec)

    def test_results_are_finite_json(self):
        assert lab._finite({"a": [1.0, float("nan"), (float("inf"), "x")], "b": {"c": -float("inf")}}) == \
            {"a": [1.0, None, [None, "x"]], "b": {"c": None}}


class TestProcess:
    def test_one_process_at_a_time(self, tmp_path):
        marker = tmp_path / "ran"
        launcher = lab.ProcessLauncher([sys.executable, "-c",
                                        f"import pathlib, time; pathlib.Path({str(marker)!r}).touch(); "
                                        "time.sleep(0.5)"])
        launcher.launch()
        first = launcher._process
        launcher.launch()
        assert launcher._process is first
        first.wait(10)
        assert marker.exists()
        launcher.launch()
        assert launcher._process is not first
        launcher._process.wait(10)

    def test_the_default_command(self):
        launcher = lab.ProcessLauncher()
        assert launcher._command[1:] == ["-m", "flask", "--app", "src.main", "lab", "run"]
        assert (lab.ROOT / "src" / "main.py").exists()

    def test_concurrent_launches_start_one(self, monkeypatch):
        started = []

        class Process:
            def poll(self):
                return None

        def popen(*args, **kwargs):
            started.append(args)
            time.sleep(0.05)
            return Process()

        monkeypatch.setattr(lab.subprocess, "Popen", popen)
        launcher = lab.ProcessLauncher(["true"])
        threads = [threading.Thread(target=launcher.launch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(started) == 1


class TestCommand:
    def test_flask_lab_run(self, app, monkeypatch):
        monkeypatch.setattr(lab, "execute", lambda settings, kind, spec: ({"headline": "done"}, None))
        run_id = _submit(app)["id"]
        result = app.test_cli_runner().invoke(args=["lab", "run"])
        assert result.exit_code == 0, result.output
        assert "Lab: 1 run(s) executed." in result.output and lab.get(app, run_id)["status"] == "done"
