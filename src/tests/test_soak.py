"""
Tests for the paper soak report (src/trading/soak.py): matching days, the
comparison with a backtest of the same days, explaining reconciliations with
the broker's entries, the verdict, and reading what paper trading recorded.
"""

import json
import random
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
from src.models.user import AuditLog
from src.tests.trading_fixtures import IEF, SPY, daily_bars, ny
from src.trading import soak
from src.trading.backtest import BacktestConfig, run_backtest
from src.trading.calendar import TradingCalendar
from src.trading.domain import AssetClass, Side
from src.trading.engine import Allocation
from src.trading.strategies.benchmark import BuyAndHold

D = Decimal
CALENDAR = TradingCalendar(AssetClass.EQUITY)


def lineup():
    return [Allocation(BuyAndHold({SPY: 0.6, IEF: 0.4}), D("1"))]


BARS = {"SPY": daily_bars(SPY, date(2024, 6, 3), date(2026, 6, 30), 500.0, seed=3),
        "IEF": daily_bars(IEF, date(2024, 6, 3), date(2026, 6, 30), 95.0, seed=4, vol=0.004)}


def paper_run(start=ny(2026, 1, 5, 15), end=ny(2026, 5, 1, 1)):
    """
    A stand-in for paper trading: the engine on the simulated broker, recorded
    as the live engine records it. The simulated broker stamps a fill when its
    bar is final; Alpaca stamps it at the open it traded at.
    """
    result = run_backtest(lineup(), BARS, BacktestConfig(start=start, end=end, starting_cash=D("100000")))
    snapshots = tuple(soak.Snapshot(t, e) for t, e in result.equity_curve)
    fills = tuple(soak.PaperFill(at_open(soak.session_of(f.timestamp, CALENDAR)), f.symbol, f.side, f.qty, f.price,
                                 True) for f in result.fills)
    started = min((o.created_at for o in result.orders), default=None)
    return soak.PaperRecord(snapshots, fills, (), started)


def at_open(day):
    return ny(day.year, day.month, day.day, 9, 30)


def replay(record):
    start = soak.backtest_start(record.started, CALENDAR)
    end = record.snapshots[-1].timestamp + timedelta(minutes=1)
    return run_backtest(lineup(), BARS, BacktestConfig(start=start, end=end,
                                                       starting_cash=record.snapshots[0].equity))


# -- matching days ---------------------------------------------------------------------------

@pytest.mark.parametrize("when, session", [
    (ny(2026, 3, 3, 0, 15), date(2026, 3, 2)),    # Tuesday 00:15: Monday's bar just became final
    (ny(2026, 3, 3, 9, 0), date(2026, 3, 2)),     # still Monday's until the next midnight
    (ny(2026, 3, 7, 0, 15), date(2026, 3, 6)),    # Saturday: Friday's
    (ny(2026, 11, 27, 0, 15), date(2026, 11, 25)),  # the Friday after Thanksgiving: Wednesday's
    (ny(2026, 3, 3, 0, 10), date(2026, 2, 27)),   # before Monday's bar is final: the previous Friday's
])
def test_each_value_belongs_to_the_session_whose_bar_was_final(when, session):
    assert soak.session_of(when, CALENDAR) == session


def test_the_backtest_starts_on_the_bar_the_engine_first_acted_on():
    # Started on Monday afternoon: the newest final bar is Friday's, final at Saturday 00:15
    assert soak.backtest_start(ny(2026, 3, 2, 15), CALENDAR) == ny(2026, 2, 28, 0, 15) - timedelta(seconds=1)


# -- the comparison ------------------------------------------------------------------------------

def _series(start, values):
    days = CALENDAR.sessions(start, start + timedelta(days=len(values) * 2))[:len(values)]
    return dict(zip(days, (D(str(v)) for v in values), strict=True))


def test_identical_series_track_perfectly():
    series = _series(date(2026, 3, 2), [100, 101, 100.5, 102, 101.7])
    tracking = soak.compare(series, series, date(2026, 3, 2))
    assert tracking.returns == 4 and tracking.tracking_error == 0 and tracking.return_gap == 0
    assert tracking.correlation == pytest.approx(1.0)


def test_tracking_error_is_the_annualized_spread_of_daily_differences():
    backtest = _series(date(2026, 3, 2), [100, 101, 102, 103, 104])
    paper = _series(date(2026, 3, 2), [100, 101.1, 101.9, 103.2, 103.9])
    tracking = soak.compare(paper, backtest, date(2026, 3, 2))
    days = sorted(paper)
    differences = [float(paper[b] / paper[a]) - float(backtest[b] / backtest[a])
                   for a, b in zip(days, days[1:], strict=False)]
    mean = sum(differences) / len(differences)
    spread = (sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)) ** 0.5
    assert tracking.tracking_error == pytest.approx(spread * 252 ** 0.5)
    assert tracking.return_gap == pytest.approx(1.039 - 1.04)
    assert tracking.worst_day == days[3] and tracking.worst_difference == pytest.approx(differences[2])


def test_the_comparison_starts_at_the_base_and_skips_days_missing_on_either_side():
    backtest = _series(date(2026, 3, 2), [100, 50, 101, 102, 103])
    paper = dict(backtest)
    paper[sorted(paper)[2]] = D("999")  # a day before the base does not count
    del paper[sorted(paper)[4]]
    tracking = soak.compare(paper, backtest, sorted(backtest)[3])
    assert tracking.base == sorted(backtest)[3] and tracking.returns == 0 and tracking.tracking_error is None


# -- reconciliations ---------------------------------------------------------------------------

def _reconciled(when, detail):
    return soak.AuditEntry(when, "ENGINE_RECONCILED", detail)


def test_a_cash_difference_matching_a_dividend_is_explained():
    start = ny(2026, 3, 2)
    audit = [_reconciled(ny(2026, 3, 10, 9), "cash: engine 1000.00 -> broker 1123.45"),
             _reconciled(ny(2026, 3, 20, 9), "cash: engine 2000 -> broker 2050")]
    activities = [soak.Activity(ny(2026, 3, 10, 7), "DIV", D("123.45"), "SPY"),
                  soak.Activity(ny(2026, 3, 20, 7), "DIV", D("30"), "IEF"),
                  soak.Activity(ny(2026, 3, 20, 8), "INT", D("20"))]
    first, second = soak.explain_reconciliations(audit, activities, start)
    assert first.explanation == "matches the broker's DIV SPY +123.45"
    assert second.explanation == "matches the broker's DIV IEF +30, INT +20"


@pytest.mark.parametrize("detail, activities", [
    ("cash: engine 1000 -> broker 1100", [soak.Activity(ny(2026, 3, 10, 7), "DIV", D("99.50"), "SPY")]),
    ("cash: engine 1000 -> broker 1100", []),
    ("SPY: engine 10 -> broker 11", [soak.Activity(ny(2026, 3, 10, 7), "DIV", D("100"), "SPY")]),
    ("cash: engine 1000 -> broker 1100; adopted orders x1", [soak.Activity(ny(2026, 3, 10, 7), "DIV", D("100"))]),
    ("closed orders x2", []),
])
def test_what_the_broker_does_not_account_for_stays_to_review(detail, activities):
    [found] = soak.explain_reconciliations([_reconciled(ny(2026, 3, 10, 9), detail)], activities, ny(2026, 3, 2))
    assert found.explanation == ""


def test_entries_before_the_soak_or_a_previous_difference_are_not_counted_again():
    audit = [_reconciled(ny(2026, 3, 10, 9), "cash: engine 0 -> broker 50"),
             _reconciled(ny(2026, 3, 12, 9), "cash: engine 50 -> broker 100")]
    activities = [soak.Activity(ny(2026, 3, 1), "DIV", D("50")), soak.Activity(ny(2026, 3, 10, 7), "DIV", D("50"))]
    first, second = soak.explain_reconciliations(audit, activities, ny(2026, 3, 2))
    assert first.explanation and not second.explanation


# -- fills at the open ------------------------------------------------------------------------

def test_opening_fills_are_measured_against_the_open_as_traded():
    day = BARS["SPY"][300]
    opening = day.open
    fills = [soak.PaperFill(day.start, "SPY", Side.BUY, D("1"), D(str(opening * 1.0005)), True),
             soak.PaperFill(day.start, "SPY", Side.SELL, D("1"), D(str(opening * 0.9997)), True),
             soak.PaperFill(day.start, "SPY", Side.BUY, D("1"), D(str(opening * 1.01)), False)]  # not at the open
    slippage = soak.opening_slippage(fills, BARS)
    assert slippage.fills == 2 and slippage.mean_bps == pytest.approx(4.0, abs=0.01)
    assert slippage.worst_bps == pytest.approx(5.0, abs=0.01) and slippage.model_bps == 5
    assert soak.opening_slippage([], BARS).mean_bps is None


# -- the verdict, end to end -----------------------------------------------------------------------

def test_paper_trading_that_matches_its_backtest_passes():
    record = paper_run()
    report = soak.build_report(record, replay(record), [], BARS, CALENDAR)
    assert report.recorded >= 60 and report.missing == ()
    assert report.tracking.base == soak.fill_session(record.fills[0].timestamp, CALENDAR)
    assert report.tracking.tracking_error == pytest.approx(0, abs=1e-12)
    assert report.tracking.return_gap == pytest.approx(0, abs=1e-12)
    assert report.slippage.mean_bps == pytest.approx(5, abs=0.01)  # the simulated broker's 5 bps
    assert report.verdict == soak.PASS, report.to_markdown()


def test_paper_trading_that_drifts_from_its_backtest_fails():
    record = paper_run()
    noise = random.Random(1)
    noisy = tuple(soak.Snapshot(s.timestamp, s.equity * D(str(1 + noise.gauss(0, 0.003)))) for s in record.snapshots)
    report = soak.build_report(soak.PaperRecord(noisy, record.fills, (), record.started), replay(record), [], BARS,
                               CALENDAR)
    tracking = next(c for c in report.criteria if c.name == "Tracking error")
    assert tracking.status == soak.FAIL and report.verdict == soak.FAIL


def test_a_short_soak_is_not_judged_yet():
    record = paper_run(end=ny(2026, 2, 3, 1))
    report = soak.build_report(record, replay(record), [], BARS, CALENDAR)
    days = next(c for c in report.criteria if c.name == "Trading days recorded")
    assert days.status == soak.NOT_YET and days.detail.startswith(f"{report.recorded} of 60")
    assert report.verdict == soak.NOT_YET


def test_events_and_unexplained_reconciliations_are_for_review():
    record = paper_run()
    audit = (soak.AuditEntry(ny(2026, 2, 10, 9), "ENGINE_BROKER_ERROR", "Alpaca timed out"),
             _reconciled(ny(2026, 3, 10, 9), "cash: engine 1000 -> broker 1100"),
             soak.AuditEntry(ny(2026, 3, 11), "ENGINE_RESUME", "after a drill"))
    record = soak.PaperRecord(record.snapshots, record.fills, audit, record.started)
    report = soak.build_report(record, replay(record), [], BARS, CALENDAR)
    statuses = {c.name: c.status for c in report.criteria}
    assert statuses["Errors, halts and the kill switch"] == soak.REVIEW
    assert statuses["Reconciliation differences"] == soak.REVIEW
    assert report.verdict == soak.REVIEW
    markdown = report.to_markdown()
    assert "| broker error | Alpaca timed out |" in markdown and "not explained: review" in markdown


def test_days_the_engine_missed_are_listed():
    record = paper_run()
    gap = soak.session_of(record.snapshots[10].timestamp, CALENDAR)
    kept = tuple(s for s in record.snapshots if soak.session_of(s.timestamp, CALENDAR) != gap)
    report = soak.build_report(soak.PaperRecord(kept, record.fills, (), record.started), replay(record), [], BARS,
                               CALENDAR)
    assert report.missing == (gap,) and f"{gap}" in report.to_markdown()


def test_nothing_recorded_yet():
    report = soak.build_report(soak.PaperRecord((), (), (), None), None, [], {}, CALENDAR)
    assert report.verdict == soak.NOT_YET and "recorded nothing yet" in report.to_markdown()


def test_the_report_as_json():
    record = paper_run()
    report = soak.build_report(record, replay(record), [], BARS, CALENDAR, notes=["a note"])
    data = json.loads(json.dumps(report.to_dict()))
    assert data["verdict"] == "PASS" and data["tracking"]["return_gap"] == 0 and data["notes"] == ["a note"]
    assert data["tolerance"] == {"min_days": 60, "max_tracking_error": 0.02, "max_return_gap": 0.01}


# -- the broker's account entries -----------------------------------------------------------------

def test_broker_entries_are_read_page_by_page():
    pages = [[{"id": "a", "activity_type": "DIV", "net_amount": "12.5", "symbol": "SPY", "status": "executed",
               "created_at": "2026-03-10T11:00:00Z"},
              {"id": "b", "activity_type": "INT", "net_amount": "1.00", "status": "canceled",
               "created_at": "2026-03-11T11:00:00Z"}],
             [{"id": "c", "activity_type": "FEE", "net_amount": "-0.5", "status": "executed",
               "date": "2026-03-12T00:00:00Z"}]]
    calls = []

    def get(url, headers):
        calls.append((url, headers))
        return pages[len(calls) - 1]

    found = soak.broker_activities("key", "secret", datetime(2026, 3, 1, tzinfo=UTC), get=get, page_size=2)
    assert [(a.activity_type, a.net_amount, a.symbol) for a in found] == [("DIV", D("12.5"), "SPY"),
                                                                          ("FEE", D("-0.5"), "")]
    assert calls[0][0].startswith("https://paper-api.alpaca.markets/v2/account/activities?")
    assert "category=non_trade_activity" in calls[0][0] and "after=2026-03-01T00%3A00%3A00Z" in calls[0][0]
    assert "page_token=b" in calls[1][0] and calls[0][1]["APCA-API-KEY-ID"] == "key"


# -- what paper trading recorded --------------------------------------------------------------------

def test_the_record_is_read_from_the_database(app, db):
    t0 = ny(2026, 3, 2, 15)
    with app.app_context():
        db.session.add_all([
            TradingOrder(client_order_id="o1", symbol="SPY", side="buy", qty=D("10"), order_type="market",
                         time_in_force="opg", status="filled", filled_qty=D("10"), created_at=t0, updated_at=t0),
            TradingOrder(client_order_id="o2", symbol="IEF", side="buy", qty=D("5"), order_type="market",
                         time_in_force="day", status="filled", filled_qty=D("5"), created_at=t0, updated_at=t0),
            TradingFill(fill_id="f1", client_order_id="o1", symbol="SPY", side="buy", qty=D("10"), price=D("500"),
                        fee=D("0"), fee_qty=D("0"), timestamp=ny(2026, 3, 3, 9, 30)),
            TradingFill(fill_id="f2", client_order_id="o2", symbol="IEF", side="buy", qty=D("5"), price=D("95"),
                        fee=D("0"), fee_qty=D("0"), timestamp=t0 + timedelta(seconds=2)),
            EquitySnapshot(timestamp=ny(2026, 3, 4, 0, 15), equity=D("100010"), cash=D("94525"),
                           gross_exposure=D("5485"), realized_pnl=D("0"), fees=D("0"), halted=False),
            AuditLog(event_type="ENGINE_RECONCILED", success=True, event_data={"detail": "cash: engine 1 -> broker 2"},
                     timestamp=ny(2026, 3, 5)),
            AuditLog(event_type="LOGIN_SUCCESS", success=True, event_data={}, timestamp=ny(2026, 3, 5)),
        ])
        db.session.commit()
        record = soak.load_record()
    assert record.started == t0
    assert [(f.symbol, f.at_open) for f in record.fills] == [("IEF", False), ("SPY", True)]
    assert record.snapshots == (soak.Snapshot(ny(2026, 3, 4, 0, 15), D("100010"), False),)
    assert record.audit == (soak.AuditEntry(ny(2026, 3, 5), "ENGINE_RECONCILED", "cash: engine 1 -> broker 2"),)


def test_the_command_before_anything_was_recorded(app, db):
    result = app.test_cli_runner().invoke(args=["soak", "report"])
    assert result.exit_code == 0, result.output
    assert "# Paper soak report: NOT YET" in result.output and "recorded nothing yet" in result.output


def test_the_command_fails_on_a_failed_soak(app, monkeypatch):
    failed = soak.SoakReport(soak.TOLERANCE, None, None, 0, (), 0, soak.compare({}, {}, None), (), (),
                             soak.opening_slippage([], {}), 0, 0, (soak.Criterion("Tracking error", soak.FAIL, "x"),))
    monkeypatch.setattr(soak, "run_report", lambda settings, since=None: failed)
    result = app.test_cli_runner().invoke(args=["soak", "report", "--json", "--since", "2026-03-02"])
    assert result.exit_code == 1 and json.loads(result.output)["verdict"] == "FAIL"


def test_the_report_runs_a_backtest_of_the_recorded_days(app, monkeypatch):
    from pydantic import SecretStr

    from src.trading import data
    from src.trading.validation import market

    record = paper_run()
    loads, sources = [], []

    def load(settings, symbols, start, end):
        loads.append((symbols, start, end))
        return {"SPY": SPY, "IEF": IEF}, BARS

    class RawBars:
        def __init__(self, clock, api_key, secret_key, raw=False):
            sources.append((api_key, secret_key, raw))

        def bars(self, instruments, timeframe, start, end):
            return BARS

    def unreachable(key, secret, after):
        raise OSError("no route")

    monkeypatch.setattr(soak, "load_record", lambda since=None: record)
    monkeypatch.setattr(market, "load", load)
    monkeypatch.setattr(data, "AlpacaBarSource", RawBars)
    monkeypatch.setattr(soak, "broker_activities", unreachable)
    settings = app.extensions["settings"].model_copy(
        update={"alpaca_api_key": SecretStr("key"), "alpaca_secret_key": SecretStr("secret")})
    report = soak.run_report(settings)
    [(symbols, start, end)] = loads
    assert [s for s, _ in symbols] == ["SPY", "IEF"]
    assert start == soak.backtest_start(record.started, CALENDAR) - soak.WARM_UP
    assert end == record.snapshots[-1].timestamp + timedelta(minutes=1)
    assert sources == [("key", "secret", True)]
    assert report.backtest_fills > 0 and report.tracking.returns > 0
    assert any("could not read the account's dividends" in n for n in report.notes)


def test_the_report_needs_the_alpaca_keys(app, monkeypatch):
    from src.trading.runtime import TradingNotConfiguredError

    monkeypatch.setattr(soak, "load_record", lambda since=None: paper_run(end=ny(2026, 1, 20)))
    with pytest.raises(TradingNotConfiguredError):
        soak.run_report(app.extensions["settings"].model_copy(update={"alpaca_api_key": None}))
