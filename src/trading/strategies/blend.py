"""
NOCTURNA Trading System - A fixed-weight blend of strategies.

Holds several strategies at fixed fractions of its capital and adds up their
targets, so a whole lineup can be validated as one candidate (the promotion
gates apply at portfolio level) and traded as one book. Each component sees
only its own instruments' history, as it would on its own.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..domain import Instrument
from ..strategy import Strategy, StrategyContext, common_execution, consult

_NAME = re.compile(r"[a-z0-9-]{1,16}")


class Blend(Strategy):
    def __init__(self, name: str, components: Sequence[tuple[Strategy, float]]) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"strategy name {name!r} must match {_NAME.pattern}")
        if not components or any(w <= 0 for _, w in components) or sum(w for _, w in components) > 1 + 1e-9:
            raise ValueError("component weights must be positive and sum to at most 1")
        self.name = name
        self._components = list(components)
        instruments: dict[str, Instrument] = {}
        for strategy, _ in self._components:
            for instrument in strategy.instruments:
                known = instruments.setdefault(instrument.symbol, instrument)
                if known != instrument:
                    raise ValueError(f"conflicting definitions for {instrument.symbol}")
        self.instruments = tuple(instruments.values())
        self.execution = common_execution([s for s, _ in self._components])
        self.lookback = max(s.lookback for s, _ in self._components)
        # Called on every new bar; each component waits for its own history, so a component
        # whose data starts later (crypto) does not hold back the others
        self.min_history = 0
        self._last: dict[int, Mapping[str, float]] = {}

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        for i, (strategy, _) in enumerate(self._components):
            weights = consult(strategy, ctx)  # None while the component warms up
            if weights is not None:
                self._last[i] = weights
        combined: dict[str, float] = {}
        for i, (_, allocation) in enumerate(self._components):
            for symbol, weight in self._last.get(i, {}).items():
                combined[symbol] = combined.get(symbol, 0.0) + allocation * weight
        return combined
