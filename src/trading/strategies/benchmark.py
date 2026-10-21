"""
Benchmark strategies: the yardsticks every candidate strategy must beat after
costs, and reference workloads for the engine's own tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..domain import Instrument
from ..strategy import Strategy, StrategyContext


class BuyAndHold(Strategy):
    """Hold fixed weights; the engine rebalances back to them as prices drift."""

    name = "buy-and-hold"
    lookback = 1

    def __init__(self, weights: Mapping[Instrument, float]) -> None:
        if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
            raise ValueError("weights must be non-negative and sum to at most 1")
        self.instruments: Sequence[Instrument] = tuple(weights)
        self._weights = {i.symbol: float(w) for i, w in weights.items()}

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        return self._weights
