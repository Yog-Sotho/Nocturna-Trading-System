"""
NOCTURNA Trading System - The month-end payment cycle.

Etula, Rinne, Suominen and Vaittinen, "Dash for Cash: Monthly Market Impact
of Institutional Liquidity Needs" (Review of Financial Studies, 2020).
Institutions that must pay out cash at the month end sell stocks until the
last day whose trades settle in time, which depresses prices; prices recover
once that selling stops, and new money lifts them over the first days of the
next month. With T the last session of the month and s the settlement period
in days, the selling ends at the close of T-(s+1). On US stocks from July 1995
to December 2013 (s = 3) the market returned -17 bps from T-8 to T-4 and +77
bps from T-3 to T+3; when 13 European markets moved from three- to two-day
settlement in 2014, the reversal moved one day closer to the month end.

The strategy holds SPY from the closing auction of T-k to the closing auction
of T+3, so it earns the returns of T-k+1 to T+3, and holds nothing otherwise
(the engine's cash sleeve keeps idle equity in T-bills). The rules were fixed
before any backtest (docs/STRATEGY_SHORTLIST.md, "Fixed rules: month-end"):

* ``settlement``: k = s + 1 for the settlement period in force on the entry
  day: 4 before 5 September 2017 (T+3), 3 from then (T+2), 2 from 28 May 2024
  (T+1),
* ``paper``: k = 4, the paper's US window as published,
* ``turn``: k = 1, the classic turn of the month from T to T+3 (McConnell and
  Xu, "Equity Returns at the Turn of the Month", FAJ 2008).

It trades at the close (``execution`` CLOSE): once a session's bar is final it
targets what to hold at the close of the next session, and the engine trades
in that session's closing auction. The dates depend on the calendar alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from ..calendar import TradingCalendar
from ..domain import NEW_YORK, AssetClass, Instrument
from ..strategy import CLOSE, Strategy, StrategyContext

#: US equities settled in three business days until these dates, then two, then one
#: (SEC Rule 15c6-1: compliance dates of the T+2 and T+1 amendments)
T_PLUS_2_FROM = date(2017, 9, 5)
T_PLUS_1_FROM = date(2024, 5, 28)
#: The position is sold in the closing auction of T+3
EXIT_AFTER = 3
RULES = ("paper", "settlement", "turn")  # by the length of the window, longest first

_CALENDAR = TradingCalendar(AssetClass.EQUITY)


def settlement_days(day: date) -> int:
    """The US equity settlement period on ``day``, in business days."""
    if day >= T_PLUS_1_FROM:
        return 1
    if day >= T_PLUS_2_FROM:
        return 2
    return 3


def entry_offset(rule: str, day: date) -> int:
    """k: the position is bought in the closing auction of T-k."""
    if rule == "settlement":
        return settlement_days(day) + 1
    return 4 if rule == "paper" else 1


def _month_sessions(day: date) -> list[date]:
    first = day.replace(day=1)
    following = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return _CALENDAR.sessions(first, _CALENDAR.previous_session(following))


def held_at_close(day: date, rule: str) -> bool:
    """
    Whether SPY is held at the close of session ``day``: from T-k to T of its
    month, or on the first EXIT_AFTER - 1 sessions of the month (T+1 and T+2
    of the previous month's end).
    """
    sessions = _month_sessions(day)
    index = sessions.index(day)
    to_month_end = len(sessions) - 1 - index  # 0 on T
    return to_month_end <= entry_offset(rule, day) or index < EXIT_AFTER - 1


class MonthEndCycle(Strategy):
    name = "month-end"
    lookback = 1
    execution = CLOSE

    def __init__(self, equity: Instrument, rule: str = "settlement") -> None:
        if rule not in RULES:
            raise ValueError(f"rule must be one of {', '.join(RULES)}")
        self.instruments = (equity,)
        self.rule = rule

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        symbol = self.instruments[0].symbol
        latest = ctx.bars(symbol)[-1].start.astimezone(NEW_YORK).date()
        trading = _CALENDAR.next_session(latest)
        return {symbol: 1.0 if held_at_close(trading, self.rule) else 0.0}
