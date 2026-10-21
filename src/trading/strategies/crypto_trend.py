"""
NOCTURNA Trading System - Crypto trend following with a Donchian ensemble.

Zarattini, Pagani and Barbon, "Catching Crypto Trends: A Tactical Approach
for Bitcoin and Altcoins" (Swiss Finance Institute Research Paper 25-80,
2025), as summarized by CXO Advisory (the paper itself is not reachable from
the build environment):

* nine models with lookbacks of 5, 10, 20, 30, 60, 90, 150, 250 and 360 days;
* a model enters when the daily close rises above the highest close of its
  lookback, and exits on a trailing stop: the higher of the previous stop and
  the midpoint of the highest and lowest closes of the lookback;
* each position is sized to 25% annualized volatility (25% over the 90-day
  standard deviation of daily returns, annualized), leverage capped;
* the ensemble holds the equal-weighted average of the nine models, daily.

Long only, no leverage here (each coin at most ``max_weight`` of the capital
allocated to it), capital split equally between the coins. Our reading where
the summary is silent: a stop set at one close is tested by the next close,
and a model's first stop is the midpoint on its entry day.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..domain import Instrument
from ..strategy import Strategy, StrategyContext

LOOKBACKS = (5, 10, 20, 30, 60, 90, 150, 250, 360)


def in_position(closes: np.ndarray, lookback: int) -> bool:
    """Whether the breakout model with this lookback is long after the last close."""
    if len(closes) <= lookback:
        return False
    windows = sliding_window_view(closes, lookback)  # windows[i] = closes[i : i + lookback]
    highs, lows = windows.max(axis=1).tolist(), windows.min(axis=1).tolist()
    prices = closes.tolist()
    long, stop = False, 0.0
    for t in range(lookback, len(prices)):
        close = prices[t]
        midpoint = (highs[t - lookback + 1] + lows[t - lookback + 1]) / 2  # the lookback ending at t
        if long:
            if close < stop:
                long = False
            else:
                stop = max(stop, midpoint)
        elif close > highs[t - lookback]:  # above the highest of the previous ``lookback`` closes
            long, stop = True, midpoint
    return long


def ensemble_exposure(closes: np.ndarray, lookbacks: Sequence[int] = LOOKBACKS) -> float:
    """Share of the models that are long."""
    return sum(in_position(closes, n) for n in lookbacks) / len(lookbacks)


class DonchianEnsemble(Strategy):
    name = "crypto-trend"

    def __init__(self, instruments: Sequence[Instrument], target_volatility: float = 0.25, vol_window: int = 90,
                 max_weight: float = 1.0, lookbacks: Sequence[int] = LOOKBACKS) -> None:
        if not instruments:
            raise ValueError("at least one instrument is required")
        self.instruments = tuple(instruments)
        self.target_volatility, self.vol_window = target_volatility, vol_window
        self.max_weight, self.lookbacks = max_weight, tuple(lookbacks)
        # Enough history for the longest channel and for its stop to have formed
        self.lookback = 2 * max(self.lookbacks) + 1

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        weights = {}
        for instrument in self.instruments:
            closes = ctx.closes(instrument.symbol)
            if len(closes) <= max(self.vol_window, max(self.lookbacks)):
                weights[instrument.symbol] = 0.0
                continue
            returns = np.diff(closes[-self.vol_window - 1:]) / closes[-self.vol_window - 1:-1]
            volatility = float(returns.std(ddof=1)) * math.sqrt(365)
            size = min(self.max_weight, self.target_volatility / volatility) if volatility > 0 else 0.0
            weights[instrument.symbol] = ensemble_exposure(closes, self.lookbacks) * size / len(self.instruments)
        return weights
