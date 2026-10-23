"""
NOCTURNA Trading System - A trend filter on the 60/40's equity sleeve.

Faber, "A Quantitative Approach to Tactical Asset Allocation" (Journal of
Wealth Management 2007; SSRN 962461, February 2013 update): "Buy when monthly
price > 10-month SMA. Sell and move to cash when monthly price < 10-month
SMA", on total-return prices at the close of each month's last day, the rest
of the month ignored. Faber reports "broad parameter stability from 3 months
on out to over 12 months".

Here the filter switches only the equity sleeve of the 60/40: at each month
end, SPY is held at its weight while its adjusted close is above the mean of
the last ``months`` month-end closes (that one included), and at zero
otherwise; IEF keeps its weight throughout, and the engine's cash sleeve holds
the rest in T-bills. The rules were fixed before any backtest
(docs/STRATEGY_SHORTLIST.md, "Fixed rules: trend-filter"). Orders go at the
next open like the 60/40's, a session after Faber's close.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain import NEW_YORK, Instrument
from ..strategy import Strategy, StrategyContext
from .trend import closes_month

MONTHS = (8, 10, 12)  # the grid: Faber's 10 and its neighbours
#: Most NYSE sessions in a month (a 31-day month with 23 weekdays and no holiday)
MAX_SESSIONS_PER_MONTH = 23


class TrendFilter(Strategy):
    name = "trend-filter"
    min_history = 1  # decides as soon as ``months`` month ends exist, however few bars that takes

    def __init__(self, equity: Instrument, bond: Instrument, months: int = 10, equity_weight: float = 0.6) -> None:
        if months < 2:
            raise ValueError("months must be at least 2")
        if not 0 < equity_weight <= 1:
            raise ValueError("equity_weight must be in (0, 1]")
        self.instruments = (equity, bond)
        self.months, self.equity_weight = months, equity_weight
        # Enough bars to hold ``months`` month ends, whichever day of a month the newest bar is
        self.lookback = months * MAX_SESSIONS_PER_MONTH + 1

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        equity, bond = (i.symbol for i in self.instruments)
        closes = [b.close for b in ctx.bars(equity) if closes_month(b.start.astimezone(NEW_YORK).date())]
        if len(closes) < self.months:
            return None
        window = closes[-self.months:]
        above = window[-1] > sum(window) / self.months
        return {equity: self.equity_weight if above else 0.0, bond: round(1 - self.equity_weight, 6)}
