"""
NOCTURNA Trading System - The paper soak report (Phase 9).

Does the engine, trading the Alpaca paper account unattended, do what the
backtest says it would? The report compares what paper trading recorded
(equity snapshots, fills, the audit trail) with a backtest of the same days,
on the same Alpaca bars and with the same settings, and checks the soak's
pass criteria (docs/ROADMAP.md):

* at least 60 trading days recorded;
* no engine error, halt or kill-switch trip left unreviewed;
* every reconciliation difference explained: a cash difference is explained
  when the broker's own entries (dividends, interest, fees) add up to it;
* annualized tracking error of daily returns under 2%;
* total return within 1 percentage point of the backtest's.

Days are matched by the trading session whose close each equity value marks.
The engine snapshots its equity when a daily bar becomes final, 15 minutes
after the New York midnight that ends it, which is also when the backtest
records its equity. Returns are compared from the first session at whose
close both had traded, so a different first fill time does not count. The
backtest reads split- and dividend-adjusted bars and the paper account gets
its dividends as cash on the pay date, so around a dividend the two differ
for a few days; the tolerance allows for it.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from .backtest import BacktestResult
from .broker import DEFAULT_COSTS
from .calendar import TradingCalendar
from .data import DEFAULT_FEED_DELAY
from .domain import NEW_YORK, AssetClass, Bar, Side

TRADING_DAYS_PER_YEAR = 252
EQUITY_DELAY = DEFAULT_FEED_DELAY[AssetClass.EQUITY]
ALPACA_PAPER_API = "https://paper-api.alpaca.markets"
#: A cash difference matches the broker's entries if they agree to the cent
CASH_MATCH = Decimal("0.01")
#: Broker entries can be booked a little after the engine saw the cash change
BOOKING_SLACK = timedelta(days=1)


@dataclass(frozen=True)
class Tolerance:
    min_days: int = 60
    max_tracking_error: float = 0.02  # annualized standard deviation of daily return differences
    max_return_gap: float = 0.01  # absolute difference of total returns


TOLERANCE = Tolerance()

#: Audit events the operator reviews, and what each means
REVIEW_EVENTS = {
    "ENGINE_STRATEGY_ERROR": "strategy error",
    "ENGINE_BROKER_ERROR": "broker error",
    "ENGINE_UNKNOWN_FILL": "a fill the engine did not expect",
    "ENGINE_REJECTED": "an order the risk gate refused",
    "ENGINE_HALT": "trading halted",
    "ENGINE_KILL_SWITCH": "the drawdown kill switch",
}


@dataclass(frozen=True)
class Snapshot:
    timestamp: datetime
    equity: Decimal
    halted: bool = False


@dataclass(frozen=True)
class PaperFill:
    timestamp: datetime
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    at_open: bool  # from an opening-auction (OPG) order: comparable with the session's open


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    event_type: str
    detail: str


@dataclass(frozen=True)
class Activity:
    """A non-trade entry on the broker account: a dividend, interest, a fee."""

    timestamp: datetime
    activity_type: str
    net_amount: Decimal
    symbol: str = ""


@dataclass(frozen=True)
class PaperRecord:
    snapshots: tuple[Snapshot, ...]
    fills: tuple[PaperFill, ...]
    audit: tuple[AuditEntry, ...]
    #: When the engine first acted: its first order, else its first snapshot
    started: datetime | None


# -- matching days -------------------------------------------------------------------------

def session_of(timestamp: datetime, calendar: TradingCalendar, delay: timedelta = EQUITY_DELAY) -> date:
    """The session whose daily bar had most recently become final at ``timestamp``."""
    day = (timestamp - delay).astimezone(NEW_YORK).date() - timedelta(days=1)
    return day if calendar.is_session(day) else calendar.previous_session(day)


def fill_session(timestamp: datetime, calendar: TradingCalendar) -> date:
    day = timestamp.astimezone(NEW_YORK).date()
    return day if calendar.is_session(day) else calendar.previous_session(day)


def by_session(points: Iterable[tuple[datetime, Decimal]], calendar: TradingCalendar) -> dict[date, Decimal]:
    """Equity at each session's close; a later value for the same session replaces an earlier one."""
    series: dict[date, Decimal] = {}
    for timestamp, value in sorted(points):
        series[session_of(timestamp, calendar)] = value
    return series


def backtest_start(started: datetime, calendar: TradingCalendar, delay: timedelta = EQUITY_DELAY) -> datetime:
    """
    Just before the newest daily bar that was final when paper trading started
    became final: the backtest then makes its first decision on the same bar
    the engine did.
    """
    session = session_of(started, calendar, delay)
    midnight = datetime.combine(session + timedelta(days=1), datetime.min.time(), NEW_YORK)
    return (midnight + delay - timedelta(seconds=1)).astimezone(started.tzinfo)


# -- the comparison --------------------------------------------------------------------------

@dataclass(frozen=True)
class Tracking:
    base: date | None  # the session the comparison starts from
    returns: int  # daily returns compared
    paper_return: float
    backtest_return: float
    tracking_error: float | None
    correlation: float | None
    worst_day: date | None
    worst_difference: float | None  # paper minus backtest, on the worst day

    @property
    def return_gap(self) -> float:
        return self.paper_return - self.backtest_return


def _stdev(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def compare(paper: Mapping[date, Decimal], backtest: Mapping[date, Decimal], base: date | None) -> Tracking:
    """Daily returns of both from ``base`` on, over the sessions both recorded."""
    common = [d for d in sorted(paper) if d in backtest and base is not None and d >= base
              and paper[d] > 0 and backtest[d] > 0]
    if len(common) < 2:
        return Tracking(common[0] if common else None, 0, 0.0, 0.0, None, None, None, None)
    paper_returns, backtest_returns = [], []
    for previous, day in zip(common, common[1:], strict=False):
        paper_returns.append(float(paper[day] / paper[previous]) - 1)
        backtest_returns.append(float(backtest[day] / backtest[previous]) - 1)
    differences = [p - b for p, b in zip(paper_returns, backtest_returns, strict=True)]
    worst = max(range(len(differences)), key=lambda i: abs(differences[i]))
    tracking_error = correlation = None
    if len(differences) >= 2:
        tracking_error = _stdev(differences) * math.sqrt(TRADING_DAYS_PER_YEAR)
        spread_p, spread_b = _stdev(paper_returns), _stdev(backtest_returns)
        if spread_p > 0 and spread_b > 0:
            mean_p = sum(paper_returns) / len(paper_returns)
            mean_b = sum(backtest_returns) / len(backtest_returns)
            covariance = sum((p - mean_p) * (b - mean_b) for p, b in zip(paper_returns, backtest_returns,
                                                                          strict=True)) / (len(differences) - 1)
            correlation = covariance / (spread_p * spread_b)
    return Tracking(
        base=common[0], returns=len(differences),
        paper_return=float(paper[common[-1]] / paper[common[0]]) - 1,
        backtest_return=float(backtest[common[-1]] / backtest[common[0]]) - 1,
        tracking_error=tracking_error, correlation=correlation,
        worst_day=common[worst + 1], worst_difference=differences[worst])


_DIFFERENCE = re.compile(r"(?P<item>[^\s:;]+): engine (?P<engine>[^\s;]+) -> broker (?P<broker>[^\s;]+)")


@dataclass(frozen=True)
class Reconciliation:
    timestamp: datetime
    detail: str
    explanation: str = ""  # empty: not explained


def explain_reconciliations(audit: Sequence[AuditEntry], activities: Sequence[Activity],
                            since: datetime) -> list[Reconciliation]:
    """
    Every reconciliation the engine made, explained where the broker's own
    entries account for it: a cash-only difference that the dividends,
    interest and fees booked since the previous cash difference add up to.
    """
    found = []
    previous = since
    for entry in sorted((e for e in audit if e.event_type == "ENGINE_RECONCILED"), key=lambda e: e.timestamp):
        differences = list(_DIFFERENCE.finditer(entry.detail))
        orders = "adopted orders" in entry.detail or "closed orders" in entry.detail
        explanation = ""
        if differences and not orders and all(m["item"] == "cash" for m in differences):
            try:
                delta = sum((Decimal(m["broker"]) - Decimal(m["engine"]) for m in differences), Decimal(0))
            except InvalidOperation:
                delta = None
            booked = [a for a in activities if previous < a.timestamp <= entry.timestamp + BOOKING_SLACK]
            total = sum((a.net_amount for a in booked), Decimal(0))
            if delta is not None and booked and abs(total - delta) <= CASH_MATCH:
                explanation = "matches the broker's " + ", ".join(
                    " ".join(filter(None, (a.activity_type, a.symbol, f"{a.net_amount:+}"))) for a in booked)
            previous = entry.timestamp
        found.append(Reconciliation(entry.timestamp, entry.detail, explanation))
    return found


@dataclass(frozen=True)
class Slippage:
    fills: int  # opening-auction fills compared
    mean_bps: float | None  # adverse is positive
    worst_bps: float | None
    model_bps: float


def opening_slippage(fills: Sequence[PaperFill], raw_bars: Mapping[str, Sequence[Bar]]) -> Slippage:
    """Opening-auction fills against the session's opening price as traded."""
    opens = {(bar.symbol, bar.start.astimezone(NEW_YORK).date()): bar.open
             for series in raw_bars.values() for bar in series}
    values = []
    for fill in fills:
        reference = opens.get((fill.symbol, fill.timestamp.astimezone(NEW_YORK).date()))
        if fill.at_open and reference:
            values.append(fill.side.sign * (float(fill.price) / reference - 1) * 10_000)
    model = float(DEFAULT_COSTS[AssetClass.EQUITY].slippage_bps)
    if not values:
        return Slippage(0, None, None, model)
    return Slippage(len(values), sum(values) / len(values), max(values), model)


# -- the verdict ---------------------------------------------------------------------------

PASS, NOT_YET, REVIEW, FAIL = "PASS", "NOT YET", "REVIEW", "FAIL"


@dataclass(frozen=True)
class Criterion:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class SoakReport:
    tolerance: Tolerance
    first_session: date | None
    last_session: date | None
    recorded: int  # sessions with an equity snapshot
    missing: tuple[date, ...]  # sessions in between without one
    halted_days: int
    tracking: Tracking
    reconciliations: tuple[Reconciliation, ...]
    events: tuple[AuditEntry, ...]
    slippage: Slippage
    paper_fills: int
    backtest_fills: int
    criteria: tuple[Criterion, ...] = field(default=())
    notes: tuple[str, ...] = field(default=())

    @property
    def verdict(self) -> str:
        statuses = {c.status for c in self.criteria}
        for status in (FAIL, NOT_YET, REVIEW):
            if status in statuses:
                return status
        return PASS

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, (date, datetime)):
                return value.isoformat()
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        result: dict[str, Any] = plain(asdict(self))
        result["verdict"] = self.verdict
        result["tracking"]["return_gap"] = self.tracking.return_gap
        return result

    def to_markdown(self) -> str:
        t, tol = self.tracking, self.tolerance
        lines = [f"# Paper soak report: {self.verdict}", ""]
        if self.first_session:
            lines.append(f"Paper trading recorded {self.recorded} trading days, {self.first_session} to "
                         f"{self.last_session}.")
        else:
            lines.append("Paper trading has recorded nothing yet: start the engine.")
        lines += [f"Note: {note}" for note in self.notes]
        lines += ["", "| Criterion | Status | Detail |", "|---|---|---|"]
        lines += [f"| {c.name} | {c.status} | {c.detail} |" for c in self.criteria]
        lines += ["", "## Paper against the backtest", ""]
        if t.returns:
            lines += [
                f"Compared over {t.returns} daily returns from the close of {t.base}, the first session at "
                "whose close both had traded.", "",
                "| | Paper | Backtest |", "|---|---|---|",
                f"| Total return | {t.paper_return:+.2%} | {t.backtest_return:+.2%} |",
                f"| Fills | {self.paper_fills} | {self.backtest_fills} |", "",
                f"- Gap in total return: {t.return_gap * 100:+.2f} percentage points "
                f"(tolerance ±{tol.max_return_gap * 100:g}).",
                "- Tracking error: " + (f"{t.tracking_error:.2%} a year" if t.tracking_error is not None
                                        else "not enough days yet") + f" (tolerance {tol.max_tracking_error:.0%}).",
            ]
            if t.correlation is not None:
                lines.append(f"- Correlation of daily returns: {t.correlation:.3f}.")
            if t.worst_day is not None and t.worst_difference is not None:
                lines.append(f"- Largest daily difference: {t.worst_difference:+.2%} on {t.worst_day} (a dividend "
                             "or a different rebalancing day is the usual cause).")
        else:
            lines.append("Not enough days in common yet.")
        s = self.slippage
        lines += ["", "## Fills at the open", ""]
        if s.fills and s.mean_bps is not None and s.worst_bps is not None:
            lines.append(f"{s.fills} opening-auction fills: {s.mean_bps:+.1f} bps from the opening price on "
                         f"average (worst {s.worst_bps:+.1f}); the backtest assumes {s.model_bps:.0f} bps against "
                         "you. Positive means a worse price than the open.")
        else:
            lines.append("No opening-auction fill to compare yet.")
        if self.missing:
            lines += ["", "## Days without a snapshot", "",
                      "The engine was not running when these days' prices became final:",
                      ", ".join(str(d) for d in self.missing) + "."]
        if self.events:
            lines += ["", "## Events to review", "", "| When (UTC) | Event | Detail |", "|---|---|---|"]
            lines += [f"| {e.timestamp:%Y-%m-%d %H:%M} | {REVIEW_EVENTS.get(e.event_type, e.event_type)} | "
                      f"{e.detail or ''} |" for e in self.events]
        if self.reconciliations:
            lines += ["", "## Reconciliations with Alpaca", "", "| When (UTC) | Difference | Explained by |",
                      "|---|---|---|"]
            lines += [f"| {r.timestamp:%Y-%m-%d %H:%M} | {r.detail} | {r.explanation or 'not explained: review'} |"
                      for r in self.reconciliations]
        return "\n".join(lines) + "\n"


def build_report(record: PaperRecord, backtest: BacktestResult | None, activities: Sequence[Activity],
                 raw_bars: Mapping[str, Sequence[Bar]], calendar: TradingCalendar,
                 tolerance: Tolerance = TOLERANCE, notes: Sequence[str] = ()) -> SoakReport:
    paper = by_session(((s.timestamp, s.equity) for s in record.snapshots), calendar)
    simulated = by_session(backtest.equity_curve, calendar) if backtest else {}
    backtest_fills = backtest.fills if backtest else ()
    first, last = (min(paper), max(paper)) if paper else (None, None)
    missing = tuple(d for d in calendar.sessions(first, last) if d not in paper) if first and last else ()
    halted_days = len({session_of(s.timestamp, calendar) for s in record.snapshots if s.halted})

    # Alpaca stamps a fill when it trades; the simulated broker when the bar it traded on is final
    paper_first = min((fill_session(f.timestamp, calendar) for f in record.fills), default=None)
    backtest_first = min((session_of(f.timestamp, calendar) for f in backtest_fills), default=None)
    base = max(paper_first, backtest_first) if paper_first and backtest_first else None
    tracking = compare(paper, simulated, base)
    since = record.started or datetime.min.replace(tzinfo=UTC)
    reconciliations = tuple(explain_reconciliations(record.audit, activities, since))
    events = tuple(sorted((e for e in record.audit if e.event_type in REVIEW_EVENTS), key=lambda e: e.timestamp))

    days = len(paper)
    criteria = [Criterion("Trading days recorded", PASS if days >= tolerance.min_days else NOT_YET,
                          f"{days} of {tolerance.min_days}" + (f"; {len(missing)} missed" if missing else ""))]
    criteria.append(Criterion("Errors, halts and the kill switch", REVIEW if events else PASS,
                              f"{len(events)} to review" if events else "none"))
    unexplained = [r for r in reconciliations if not r.explanation]
    criteria.append(Criterion(
        "Reconciliation differences", REVIEW if unexplained else PASS,
        f"{len(unexplained)} of {len(reconciliations)} not explained by the broker's entries" if unexplained
        else (f"{len(reconciliations)}, all explained by the broker's entries" if reconciliations else "none")))
    if tracking.tracking_error is None:
        criteria.append(Criterion("Tracking error", NOT_YET, "not enough days in common yet"))
        criteria.append(Criterion("Total return gap", NOT_YET, "not enough days in common yet"))
    else:
        criteria.append(Criterion(
            "Tracking error", PASS if tracking.tracking_error < tolerance.max_tracking_error else FAIL,
            f"{tracking.tracking_error:.2%} a year (under {tolerance.max_tracking_error:.0%})"))
        criteria.append(Criterion(
            "Total return gap", PASS if abs(tracking.return_gap) < tolerance.max_return_gap else FAIL,
            f"{tracking.return_gap * 100:+.2f} percentage points (within ±{tolerance.max_return_gap * 100:g})"))
    return SoakReport(tolerance, first, last, days, missing, halted_days, tracking, reconciliations, events,
                      opening_slippage(record.fills, raw_bars), len(record.fills), len(backtest_fills),
                      tuple(criteria), tuple(notes))


# -- the broker's account entries ------------------------------------------------------------

def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def broker_activities(key: str, secret: str, after: datetime, get: Any = None,
                      page_size: int = 100) -> list[Activity]:
    """
    The paper account's non-trade entries since ``after`` (dividends, interest,
    fees), from Alpaca's account activities API; cancelled entries are left out.
    """
    fetch = get or _get_json
    found: list[Activity] = []
    token = None
    while True:
        query = {"category": "non_trade_activity", "after": after.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "direction": "asc", "page_size": str(page_size)}
        if token:
            query["page_token"] = token
        page = fetch(f"{ALPACA_PAPER_API}/v2/account/activities?{urllib.parse.urlencode(query)}",
                     {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        for item in page:
            if item.get("status") == "canceled":
                continue
            when = item.get("created_at") or item.get("date")
            try:
                amount = Decimal(str(item.get("net_amount", "0")))
            except InvalidOperation:
                continue
            if when:
                found.append(Activity(_parse_time(when), str(item.get("activity_type", "")), amount,
                                      str(item.get("symbol") or "")))
        if len(page) < page_size:
            return found
        token = page[-1].get("id")


def _get_json(url: str, headers: Mapping[str, str]) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers=dict(headers))
    # The URL is ALPACA_PAPER_API, an https constant of this module
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310
        body = json.loads(response.read())
    return body if isinstance(body, list) else []


# -- running it on the recorded history ------------------------------------------------------

#: History the backtest loads before the soak starts, as the live engine does
WARM_UP = timedelta(days=550)


def load_record(since: datetime | None = None) -> PaperRecord:
    """What paper trading recorded since ``since`` (default: everything). Needs an app context."""
    from sqlalchemy import select

    from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
    from src.models.user import AuditLog, db

    floor = since or datetime.min.replace(tzinfo=UTC)
    snapshots = tuple(Snapshot(row.timestamp, row.equity, row.halted) for row in db.session.scalars(
        select(EquitySnapshot).where(EquitySnapshot.timestamp >= floor).order_by(EquitySnapshot.timestamp)))
    first_order = db.session.scalars(select(TradingOrder.created_at).where(TradingOrder.created_at >= floor)
                                     .order_by(TradingOrder.created_at).limit(1)).first()
    candidates = [t for t in (first_order, snapshots[0].timestamp if snapshots else None) if t is not None]
    started = since or (min(candidates) if candidates else None)
    opening = {row.client_order_id for row in db.session.scalars(
        select(TradingOrder).where(TradingOrder.time_in_force == "opg"))}
    fills = tuple(PaperFill(row.timestamp, row.symbol, Side(row.side), row.qty, row.price,
                            row.client_order_id in opening) for row in db.session.scalars(
        select(TradingFill).where(TradingFill.timestamp >= floor).order_by(TradingFill.timestamp)))
    rows = db.session.execute(select(AuditLog.timestamp, AuditLog.event_type, AuditLog.event_data).where(
        AuditLog.event_type.like("ENGINE_%"), AuditLog.timestamp >= floor).order_by(AuditLog.timestamp))
    audit = tuple(_audit_entry(*row) for row in rows)
    return PaperRecord(snapshots, fills, audit, started)


def _audit_entry(timestamp: Any, event_type: Any, data: Any) -> AuditEntry:
    detail = data.get("detail") if isinstance(data, dict) else None
    return AuditEntry(timestamp, str(event_type), str(detail or ""))


def run_report(settings: Any, since: datetime | None = None) -> SoakReport:
    """
    The soak report on what paper trading recorded, against a backtest of the
    same days on Alpaca's bars. Needs an app context and the Alpaca keys.
    """
    from .backtest import BacktestConfig, run_backtest
    from .clock import SystemClock
    from .data import AlpacaBarSource
    from .domain import DAILY
    from .lineup import CASH_SLEEVE, LINEUP_SYMBOLS, build_lineup
    from .runtime import risk_limits, sizing_policy
    from .validation import market

    calendar = TradingCalendar(AssetClass.EQUITY)
    record = load_record(since)
    if not record.snapshots or record.started is None:
        return build_report(record, None, (), {}, calendar)
    start = backtest_start(record.started, calendar)
    end = record.snapshots[-1].timestamp + timedelta(minutes=1)
    instruments, bars = market.load(settings, LINEUP_SYMBOLS, start - WARM_UP, end)
    backtest = run_backtest(build_lineup(instruments), bars, BacktestConfig(
        start=start, end=end, starting_cash=record.snapshots[0].equity, risk_limits=risk_limits(settings),
        sizing=sizing_policy(settings), cash_sleeve=instruments[CASH_SLEEVE]))
    key, secret = settings.alpaca_api_key.get_secret_value(), settings.alpaca_secret_key.get_secret_value()
    raw = AlpacaBarSource(SystemClock(), api_key=key, secret_key=secret, raw=True).bars(
        list(instruments.values()), DAILY, record.started - timedelta(days=1), end)
    notes = []
    try:
        activities = broker_activities(key, secret, record.started - timedelta(days=1))
    except OSError as e:  # urllib's errors are OSErrors; the message never holds the keys
        activities = []
        notes.append(f"could not read the account's dividends and interest from Alpaca ({type(e).__name__}), so "
                     "no cash difference is explained")
    return build_report(record, backtest, activities, raw, calendar, notes=notes)
