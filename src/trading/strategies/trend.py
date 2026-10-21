"""
NOCTURNA Trading System - Time-series momentum on a diversified ETF basket.

Moskowitz, Ooi and Pedersen, "Time Series Momentum" (Journal of Financial
Economics, 2012): each month, hold every instrument long or short by the
sign of its past 12-month return, sized to a constant ex-ante volatility,
40% / sigma, where sigma is the annualized exponentially weighted volatility
of daily returns with a 60-day centre of mass; the portfolio is the equal
average across instruments.

``rule="tstat"`` replaces the sign with the t-statistic of the mean daily log
return over the 12 months clipped to [-1, 1], which scales positions by the
strength of the trend; this is the TREND rule of Baltas and Kosowski
("Demystifying Time-Series Momentum Strategies", 2013) as implemented in
QuantConnect's reproduction (the paper itself is not reachable from the build
environment). ``long_only`` holds cash instead of shorts.

Deviations: total returns of adjusted ETF prices rather than excess returns
of futures (no risk-free series is available), annualized with 252 sessions.
Positions change only at month-ends: the targets are computed as of the last
month-end close in the history, so they are a function of the history alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache

import numpy as np

from ..calendar import TradingCalendar
from ..domain import NEW_YORK, AssetClass, Bar, Instrument
from ..strategy import Strategy, StrategyContext

MONTHS_12 = 252
CENTRE_OF_MASS = 60
_CALENDAR = TradingCalendar(AssetClass.EQUITY)


@lru_cache(maxsize=1 << 14)
def _closes_month(day: date) -> bool:
    return _CALENDAR.next_session(day).month != day.month


def _month_end(bars: Sequence[Bar]) -> int | None:
    """Index of the last bar that closed a month (its next session is in another month)."""
    for i in range(len(bars) - 1, -1, -1):
        if _closes_month(bars[i].start.astimezone(NEW_YORK).date()):
            return i
    return None


def ewma_volatility(returns: np.ndarray, centre_of_mass: float = CENTRE_OF_MASS, periods: int = 252) -> float:
    """Annualized volatility from exponentially weighted squared deviations (newest return weighs most)."""
    delta = centre_of_mass / (centre_of_mass + 1)
    weights = (1 - delta) * delta ** np.arange(len(returns) - 1, -1, -1)
    weights /= weights.sum()
    mean = float(weights @ returns)
    return math.sqrt(periods * float(weights @ (returns - mean) ** 2))


def trend_signal(closes: np.ndarray, rule: str) -> float:
    """+1/-1 (``sign``) or the clipped t-statistic (``tstat``) of the last ``MONTHS_12`` returns."""
    window = closes[-MONTHS_12 - 1:]
    if rule == "sign":
        return float(np.sign(window[-1] / window[0] - 1))
    logs = np.diff(np.log(window))
    sd = logs.std()
    if sd == 0:
        return 0.0
    return float(np.clip(logs.mean() / (sd / math.sqrt(len(logs))), -1.0, 1.0))


class TimeSeriesMomentum(Strategy):
    name = "tsmom"
    lookback = MONTHS_12 + 1 + 23  # a year of returns as of a month-end up to a month back

    def __init__(self, instruments: Sequence[Instrument], rule: str = "sign", long_only: bool = False,
                 target_volatility: float = 0.40) -> None:
        if rule not in ("sign", "tstat"):
            raise ValueError("rule must be 'sign' or 'tstat'")
        if not instruments:
            raise ValueError("at least one instrument is required")
        self.instruments = tuple(instruments)
        self.rule, self.long_only, self.target_volatility = rule, long_only, target_volatility

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        weights = {}
        n = len(self.instruments)
        for instrument in self.instruments:
            bars = ctx.bars(instrument.symbol)
            end = _month_end(bars)
            if end is None or end < MONTHS_12:
                weights[instrument.symbol] = 0.0
                continue
            closes = np.fromiter((b.close for b in bars[:end + 1]), dtype=float)
            signal = trend_signal(closes, self.rule)
            if self.long_only:
                signal = max(signal, 0.0)
            returns = np.diff(closes[-MONTHS_12 - 1:]) / closes[-MONTHS_12 - 1:-1]
            volatility = ewma_volatility(returns)
            weights[instrument.symbol] = signal * self.target_volatility / volatility / n if volatility > 0 else 0.0
        return weights
