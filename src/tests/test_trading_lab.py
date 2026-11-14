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
from src.trading.catalog import STRATEGIES
from src.trading.domain import AssetClass, Instrument
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
CANDIDATE_DEFAULTS = {name: info.defaults for name, info in STRATEGIES.items()}


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
def long_history(monkeypatch):
    """The long history, made up: like French's file, it ends before today (on the last day of 2025)."""
    source = history.Source("https://example.test/french.zip", "ab" * 32, date(2014, 1, 2), date(2025, 12, 31))

    def load(start, end, shorten=False):
        limit = history.covered_until([source])
        if end > limit and not shorten:
            raise ValueError(f"{source.url} ends on {source.last}: pass an end no later than {limit:%Y-%m-%d}")
        return ({"SPY": SPY, "IEF": IEF, "BIL": BIL},
                {s: [b for b in bars if start <= b.start < min(end, limit)] for s, bars in BARS.items()}, [source])

    monkeypatch.setattr(history, "load", load)


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
        monkeypatch.setattr(history, "load", lambda start, end, shorten=False: (
            {"SPY": SPY, "IEF": IEF, "BIL": BIL}, BARS, [source]))
        data = sources.load(settings, [("SPY", EQUITY), ("BIL", EQUITY)], T0 - timedelta(days=30), T0,
                            sources.HISTORY)
        assert set(data.instruments) == set(data.bars) == {"SPY", "BIL"} and data.sources == [source]
        assert data.hurdle_label == "T-bills (French's 1-month rate, as BIL)"
        assert data.end == T0 and data.name == "The long history"

    def test_the_long_history_shortened_ends_where_its_files_do(self, settings, long_history):
        asked = (settings, [("SPY", EQUITY), ("BIL", EQUITY)], datetime(2020, 1, 2, tzinfo=UTC), T0, sources.HISTORY)
        with pytest.raises(sources.DataError, match="long-history data: .* ends on 2025-12-31"):
            sources.load(*asked)
        data = sources.load(*asked, shorten=True)
        # A day after the next session starts (midnight in New York): the last bar is final by then
        assert data.end == datetime(2026, 1, 3, 5, tzinfo=UTC)
        assert data.bars["SPY"][-1].start.date() == date(2025, 12, 31)
        assert data.notes(datetime(2020, 1, 2, tzinfo=UTC), T0) == [
            "The long history ends on 31 December 2025: the run stops there."]

    def test_notes_say_where_bars_start_late(self):
        def data(**bars):
            return sources.MarketData({}, bars, "made up", "T-bills", datetime(2026, 1, 1, tzinfo=UTC))

        start, end = datetime(2012, 1, 3, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)
        recent = [b for b in BARS["IEF"] if b.start.year >= 2021]
        assert data(**BARS).notes(datetime(2014, 1, 2, tzinfo=UTC), end) == []  # a holiday or a weekend is not late
        assert data(**BARS).late(start) == {date(2014, 1, 2): ["BIL", "IEF", "SPY"]}
        assert data(**BARS).notes(start, end) == [
            "Alpaca has no bars for BIL, IEF, SPY before 2 January 2014: the run starts there."]
        assert data(SPY=BARS["SPY"], IEF=recent, BIL=[]).notes(datetime(2014, 1, 2, tzinfo=UTC), end) == [
            "Alpaca has no bars for BIL in this period.",
            "Alpaca has no bars for IEF before 4 January 2021: until then the run cannot trade it."]

    def test_the_long_history_refuses_other_symbols_before_downloading(self, settings, monkeypatch):
        monkeypatch.setattr(history, "load", lambda start, end, shorten=False: pytest.fail("downloaded"))
        with pytest.raises(sources.DataError, match="proxies for SPY, IEF, BIL only, not BTC/USD"):
            sources.load(settings, [("SPY", EQUITY), ("BTC/USD", AssetClass.CRYPTO)], T0 - timedelta(days=9), T0,
                         sources.HISTORY)

    @pytest.mark.parametrize("error", [ValueError("ends on 2026-08-31"), OSError("connection reset")])
    def test_long_history_failures(self, settings, monkeypatch, error):
        def load(start, end, shorten=False):
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

    def test_a_queued_run_can_be_cancelled_and_never_starts(self, app, settings, monkeypatch):
        monkeypatch.setattr(lab, "execute", lambda *args: pytest.fail("a cancelled run was executed"))
        run_id = _submit(app)["id"]
        cancelled = lab.cancel(app, run_id, T0)
        assert cancelled["status"] == "cancelled" and cancelled["error"] == "cancelled"
        assert lab.drain(app, settings) == 0
        with pytest.raises(lab.LabStateError, match=f"run #{run_id} has already finished \\(cancelled\\)"):
            lab.cancel(app, run_id, T0)
        assert lab.cancel(app, 9999, T0) is None

    def test_a_cancelled_run_frees_its_place_in_the_queue(self, app):
        first = _submit(app)["id"]
        _submit(app)
        _submit(app)
        lab.cancel(app, first, T0)
        _submit(app)  # three wait again, not four

    def test_a_running_run_is_interrupted_when_cancelled(self, app, settings, monkeypatch):
        started = threading.Event()

        def execute(settings, kind, spec):
            started.set()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:  # plain Python, as a backtest is
                sum(range(1000))
            pytest.fail("not interrupted")

        monkeypatch.setattr(lab, "execute", execute)
        run_id = _submit(app)["id"]
        after = _submit(app, spec={"start": "2017-01-03", "end": "2026-10-23", "data": "alpaca",
                                   "strategy": "benchmark"})["id"]
        sink = Collect()
        alerts = AlertDispatcher([sink], retry_delays=())
        canceller = threading.Thread(target=lambda: started.wait(5) and lab.cancel(app, run_id, T0))
        canceller.start()
        began = time.monotonic()
        monkeypatch.setattr(lab, "execute", lambda settings, kind, spec: execute(settings, kind, spec)
                            if spec["strategy"] == "tsmom" else ({"headline": "the next one ran"}, None))
        assert lab.drain(app, settings, alerts, heartbeat_every=0.05) == 2
        canceller.join()
        assert time.monotonic() - began < 5
        assert lab.get(app, run_id)["status"] == "cancelled" and lab.get(app, run_id)["result"] is None
        assert lab.get(app, after)["status"] == "done"  # the queue carries on
        assert alerts.flush(5) and [a.title for a in sink.alerts] == [f"Lab run #{after} finished"]

    def test_a_run_cancelled_where_it_cannot_be_interrupted_drops_its_result(self, app, settings, monkeypatch):
        # In a thread (the simulator) nothing interrupts the computation: it finishes, and nothing is stored
        def execute(settings, kind, spec):
            lab.cancel(app, run_id, T0)
            time.sleep(0.2)
            return {"headline": "too late"}, "# report"

        monkeypatch.setattr(lab, "execute", execute)
        run_id = _submit(app)["id"]
        sink = Collect()
        alerts = AlertDispatcher([sink], retry_delays=())
        assert lab.drain(app, settings, alerts, heartbeat_every=0.05, interrupt=lambda: None) == 1
        run = lab.get(app, run_id)
        assert (run["status"], run["result"], run["report"]) == ("cancelled", None, None)
        assert alerts.flush(5) and sink.alerts == []

    def test_only_finished_runs_are_deleted(self, app, settings, monkeypatch):
        monkeypatch.setattr(lab, "execute", lambda *args: ({"headline": "x"}, None))
        run_id = _submit(app)["id"]
        with pytest.raises(lab.LabStateError, match=f"run #{run_id} is queued: cancel it first"):
            lab.remove(app, run_id)
        lab.drain(app, settings)
        assert lab.remove(app, run_id) is True
        assert lab.get(app, run_id) is None and lab.remove(app, run_id) is False

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
        assert result["notes"] == []

    def test_a_mix_is_compared_with_the_60_40_on_the_same_bars(self, settings, synthetic):
        mix = {"mix": [{"strategy": "trend-filter", "params": CANDIDATE_DEFAULTS["trend-filter"], "weight": "0.5"}],
               "risk": {"risk_max_drawdown": "0.08"}}
        _, spec = lab.parse_request(settings, {"kind": "backtest", **mix, "start": "2016-01-04", "end": "2018-01-02"},
                                    TODAY)
        result, _ = lab.execute(settings, "backtest", spec)
        benchmark = result["benchmark"]
        assert benchmark["title"] == "60/40 stocks and bonds" and benchmark["summary"] != result["summary"]
        assert [y["year"] for y in benchmark["years"]] == [y["year"] for y in result["years"]] == [2016, 2017]
        assert benchmark["daily"][0][0] == result["daily"][0][0] and len(benchmark["curve"]) > 1
        assert len(synthetic) == 1  # the 60/40 reused the bars already loaded
        # The plain 60/40 is not compared with itself
        _, spec = lab.parse_request(settings, {"kind": "backtest", **BENCHMARK, "start": "2016-01-04",
                                               "end": "2017-01-03"}, TODAY)
        assert lab.execute(settings, "backtest", spec)[0]["benchmark"] is None

    def test_a_crypto_mix_loads_the_60_40_bars_it_lacks(self, settings, synthetic, monkeypatch):
        crypto = daily_bars(Instrument("BTC/USD", AssetClass.CRYPTO), date(2014, 1, 2), date(2025, 12, 31), 400.0,
                            seed=9)

        def load(settings, symbols, start, end):
            synthetic.append((tuple(s for s, _ in symbols), start, end))
            known = {"SPY": SPY, "IEF": IEF, "BIL": BIL, "BTC/USD": Instrument("BTC/USD", AssetClass.CRYPTO),
                     "ETH/USD": Instrument("ETH/USD", AssetClass.CRYPTO),
                     "SOL/USD": Instrument("SOL/USD", AssetClass.CRYPTO)}
            return ({s: known[s] for s, _ in symbols},
                    {s: BARS.get(s, crypto) for s, _ in symbols})

        monkeypatch.setattr(market, "load", load)
        mix = {"mix": [{"strategy": "crypto-trend", "params": {}, "weight": "0.1"}]}
        mix["mix"][0]["params"] = {k: v for k, v in CANDIDATE_DEFAULTS["crypto-trend"].items()}
        _, spec = lab.parse_request(settings, {"kind": "backtest", **mix, "start": "2016-01-04", "end": "2017-01-03"},
                                    TODAY)
        result, _ = lab.execute(settings, "backtest", spec)
        assert result["benchmark"] is not None
        assert [set(call[0]) for call in synthetic] == [{"BTC/USD", "ETH/USD", "SOL/USD", "BIL"},
                                                         {"SPY", "IEF", "BIL"}]

    def test_the_daily_equity_is_downloaded_not_shown(self, app, settings, synthetic, monkeypatch):
        mix = {"mix": [{"strategy": "trend-filter", "params": CANDIDATE_DEFAULTS["trend-filter"], "weight": "0.5"}]}
        _, spec = lab.parse_request(settings, {"kind": "backtest", **mix, "start": "2016-01-04", "end": "2017-01-03"},
                                    TODAY)
        run_id = _submit(app, "backtest", spec)["id"]
        with pytest.raises(lab.LabStateError, match="only a finished run has equity to download"):
            lab.equity_csv(app, run_id)
        lab.drain(app, settings)
        shown = lab.get(app, run_id)["result"]
        assert "daily" not in shown and "daily" not in shown["benchmark"] and shown["curve"]
        name, text = lab.equity_csv(app, run_id)
        lines = text.splitlines()
        assert name == f"lab-{run_id}-backtest.csv" and lines[0] == "date,the mix,60/40 stocks and bonds"
        assert lines[1].startswith("2016-01-0") and len(lines) > 240 and all(len(r.split(",")) == 3 for r in lines)
        assert lines[-1].startswith("2016-12-30,")  # the last session of 2016, dated by the session it covers
        assert lab.equity_csv(app, 9999) is None

    def test_a_validation_downloads_its_out_of_sample_equity(self, app, settings, synthetic):
        _, spec = lab.parse_request(settings, {"kind": "validate", "strategy": "benchmark", "start": "2016-01-04",
                                               "end": "2021-01-01"}, TODAY)
        run_id = _submit(app, "validate", spec)["id"]
        lab.drain(app, settings)
        name, text = lab.equity_csv(app, run_id)
        assert name == f"lab-{run_id}-benchmark-out-of-sample.csv"
        assert text.splitlines()[0] == "date,benchmark out of sample" and text.splitlines()[1].startswith("2019-")

    def test_older_runs_have_nothing_to_download(self, app):
        run_id = _submit(app)["id"]
        lab._claim(app, T0)
        lab._finish(app, run_id, T0, result={"headline": "x", "curve": []})  # stored before downloads existed
        with pytest.raises(lab.LabStateError, match="made before downloads existed: run it again"):
            lab.equity_csv(app, run_id)

    def test_a_validation(self, settings, synthetic):
        _, spec = lab.parse_request(settings, {"kind": "validate", "strategy": "benchmark", "start": "2016-01-04",
                                               "end": "2021-01-01"}, TODAY)
        result, report = lab.execute(settings, "validate", spec)
        assert report.startswith("# Validation: benchmark")
        assert result["headline"] == result["report"]["verdict"] and isinstance(result["passed"], bool)
        assert result["curve"] and result["report"]["candidate"] == "benchmark"

    def test_a_long_history_run_to_today_stops_where_the_data_does(self, settings, long_history):
        # The form's dates: to today, which the long history never reaches
        _, spec = lab.parse_request(settings, {"kind": "backtest", **BENCHMARK, "start": "2020-01-02",
                                               "data": "history"}, TODAY)
        assert spec["end"] == TODAY.isoformat()
        result, _ = lab.execute(settings, "backtest", spec)
        assert result["notes"] == ["The long history ends on 31 December 2025: the run stops there."]
        assert "2025-12-31" <= result["curve"][-1]["timestamp"] < "2026-01-02"  # not ten months of flat equity
        # 31 December 2025 became final on 1 January: it is the last day of 2025, not a year of its own
        assert [y["year"] for y in result["years"]] == [2020, 2021, 2022, 2023, 2024, 2025]

    def test_a_run_says_where_its_bars_start_late(self, settings, synthetic):
        early = {"start": "2012-01-03", "end": "2019-01-02"}  # the bars start in 2014
        _, spec = lab.parse_request(settings, {"kind": "backtest", **BENCHMARK, **early}, TODAY)
        result, _ = lab.execute(settings, "backtest", spec)
        late = "Alpaca has no bars for BIL, IEF, SPY before 2 January 2014: the run starts there."
        assert result["notes"] == [late] and result["curve"][0]["timestamp"].startswith("2014-01-0")
        _, spec = lab.parse_request(settings, {"kind": "validate", "strategy": "benchmark", **early}, TODAY)
        result, _ = lab.execute(settings, "validate", spec)
        assert result["notes"][0] == late and "Start the run where the bars start" in result["notes"][1]

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
