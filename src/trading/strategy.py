"""
NOCTURNA Trading System - Strategy interface.

A strategy looks at closed bars and says what it wants to hold, as signed
weights of the capital allocated to it. It never sizes or sends orders: the
engine turns targets into orders, so risk limits, costs and execution are the
same for every strategy and in every mode.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import numpy as np

from .domain import DAILY, Bar, Instrument, Timeframe

_NAME = re.compile(r"[a-z0-9-]{1,16}")


@dataclass(frozen=True)
class StrategyContext:
    """What a strategy may know when it decides: nothing after ``now``."""

    now: datetime
    history: Mapping[str, Sequence[Bar]]  # closed bars per symbol, oldest first
    positions: Mapping[str, Decimal]
    equity: Decimal

    def bars(self, symbol: str) -> Sequence[Bar]:
        return self.history.get(symbol, ())

    def closes(self, symbol: str, n: int | None = None) -> np.ndarray:
        """The last ``n`` closes (all when None) as a float array, oldest first."""
        series = self.bars(symbol)
        if n is not None:
            series = series[-n:] if n else []
        return np.fromiter((b.close for b in series), dtype=float, count=len(series))


class Strategy(ABC):
    """
    Subclasses set ``name`` (1-16 of a-z, 0-9, -), ``instruments``, ``lookback``
    (bars of history needed) and implement ``target_weights``.
    """

    name: str
    instruments: Sequence[Instrument]
    timeframe: Timeframe = DAILY
    lookback: int = 1

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name is not None and not _NAME.fullmatch(name):
            raise ValueError(f"strategy name {name!r} must match {_NAME.pattern}")

    @abstractmethod
    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        """
        Signed target weights for this strategy's instruments, or None to keep the
        previous targets. Instruments left out are targeted at zero.
        """
