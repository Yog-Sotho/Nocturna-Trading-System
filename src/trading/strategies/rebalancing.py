"""
NOCTURNA Trading System - Front-running month-end and threshold rebalancing.

Harvey, Mazzoleni and Melone, "The Unintended Consequences of Rebalancing"
(NBER Working Paper 33554, 2025, revised 2026). Pension funds and other 60/40
investors sell equities and buy bonds after equities outperform, either when
the equity weight drifts past a threshold or at month-end. The paper
simulates those portfolios with S&P 500 and 10-year Treasury futures:

* **Threshold signal:** the drift of the equity weight from 60%, averaged over
  rebalancing thresholds of 0% to 2.5% in steps of 0.1% (eq. 2, appendix B.1);
  a portfolio is reset to 60% when its drift reaches its threshold.
* **Calendar signal:** the drift of a 60/40 portfolio reset on the last
  business day of each month (B.2).
* **Position** (section 4): long equities and short bonds with weight
  w = mean(-Threshold / 1.5%, Calendar'), where Calendar' is
  sign(-Calendar) in the last week of the month (the last five sessions),
  sign(Calendar four sessions earlier) on the first session of a month (the
  reversal), and zero otherwise. Reported: Sharpe 1.11 from 1997 to 2023.

Here SPY stands in for the S&P 500 future and IEF (7-10 year Treasuries) for
the 10-year note, both total-return adjusted. The paper earns the next day's
close-to-close return; daily bars are final only after the close, so the
engine trades at the next open. Two forms:

* ``spread``: long SPY / short IEF with ``scale`` * w on each side (as in the paper),
* ``tilt``: long only, a 60/40 portfolio with SPY moved by ``scale`` * w.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from functools import lru_cache

import numpy as np

from ..calendar import TradingCalendar
from ..domain import NEW_YORK, AssetClass, Instrument
from ..strategy import Strategy, StrategyContext

TARGET = 0.6
THRESHOLDS = np.round(np.arange(0.0, 0.0251, 0.001), 4)  # 0% to 2.5% in 0.1% steps
THRESHOLD_SCALE = 0.015
LAST_WEEK = 5  # sessions
REVERSAL_LAG = 4  # sessions

_CALENDAR = TradingCalendar(AssetClass.EQUITY)


@lru_cache(maxsize=1 << 14)
def _sessions_left_in_month(day: date) -> int:
    """Sessions after ``day`` in the same month."""
    n, current = 0, _CALENDAR.next_session(day)
    while current.month == day.month:
        n += 1
        current = _CALENDAR.next_session(current)
    return n


@lru_cache(maxsize=1 << 14)
def _first_session_of_month(day: date) -> bool:
    return _CALENDAR.previous_session(day).month != day.month


def _drift(weight: np.ndarray | float, equity: float, bond: float) -> np.ndarray | float:
    grown = weight * (1 + equity)
    return grown / (grown + (1 - weight) * (1 + bond))


def rebalancing_signals(days: list[date], equity: np.ndarray, bond: np.ndarray) -> tuple[float, np.ndarray]:
    """
    The Threshold signal after the last day, and the Calendar signal for every
    day, from daily returns of equities and bonds ending on ``days``.
    """
    w_threshold = np.full(len(THRESHOLDS), TARGET)
    w_calendar = TARGET
    calendar = np.empty(len(days))
    threshold = 0.0
    for i, day in enumerate(days):
        drifted = np.asarray(_drift(w_threshold, equity[i], bond[i]))
        deviation = drifted - TARGET
        threshold = float(deviation.mean())
        w_threshold = np.where(np.abs(deviation) >= THRESHOLDS, TARGET, drifted)
        drifted_calendar = float(_drift(w_calendar, equity[i], bond[i]))
        calendar[i] = drifted_calendar - TARGET
        w_calendar = TARGET if _sessions_left_in_month(day) == 0 else drifted_calendar
    return threshold, calendar


def strategy_weight(days: list[date], threshold: float, calendar: np.ndarray) -> float:
    """w on the last day: the equal-weight mean of the scaled Threshold and modified Calendar signals."""
    today = days[-1]
    if _sessions_left_in_month(today) < LAST_WEEK:
        modified = float(np.sign(-calendar[-1]))
    elif _first_session_of_month(today) and len(calendar) > REVERSAL_LAG:
        modified = float(np.sign(calendar[-1 - REVERSAL_LAG]))
    else:
        modified = 0.0
    return 0.5 * (-threshold / THRESHOLD_SCALE + modified)


class RebalancingFlows(Strategy):
    name = "rebalancing"
    lookback = 253  # a year of returns: every threshold portfolio has reset by then

    def __init__(self, equity: Instrument, bond: Instrument, mode: str = "spread", scale: float = 0.5) -> None:
        if mode not in ("spread", "tilt"):
            raise ValueError("mode must be 'spread' or 'tilt'")
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")
        self.instruments = (equity, bond)
        self.mode, self.scale = mode, scale

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        equity, bond = (i.symbol for i in self.instruments)
        closes = {s: {b.start.astimezone(NEW_YORK).date(): b.close for b in ctx.bars(s)} for s in (equity, bond)}
        days = sorted(set(closes[equity]) & set(closes[bond]))
        if len(days) < 30:
            return None
        e = np.array([closes[equity][d] for d in days])
        b = np.array([closes[bond][d] for d in days])
        threshold, calendar = rebalancing_signals(days[1:], e[1:] / e[:-1] - 1, b[1:] / b[:-1] - 1)
        w = strategy_weight(days[1:], threshold, calendar)
        if self.mode == "spread":
            return {equity: self.scale * w, bond: -self.scale * w}
        tilted = min(1.0, max(0.0, TARGET + self.scale * w))
        return {equity: tilted, bond: 1.0 - tilted}
