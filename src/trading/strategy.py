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
from typing import Any

import numpy as np

from .domain import DAILY, Bar, Instrument, Timeframe

_NAME = re.compile(r"[a-z0-9-]{1,16}")

#: When a strategy's equity orders execute: in the next opening auction (or at once
#: during the session), or in the next closing auction
OPEN = "open"
CLOSE = "close"


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
    (bars of history kept and needed) and implement ``target_weights``.
    ``min_history`` lets a strategy be called with fewer bars than it keeps
    (a blend whose components warm up at different speeds). ``execution``
    CLOSE trades equities in the next session's closing auction: a strategy
    whose dates are known in advance (a calendar) can then trade at the close
    it means, from the bars of the session before.
    """

    name: str
    instruments: Sequence[Instrument]
    timeframe: Timeframe = DAILY
    lookback: int = 1
    min_history: int | None = None
    execution: str = OPEN

    def __init_subclass__(cls, **kwargs: Any) -> None:
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


def common_execution(strategies: Sequence[Strategy]) -> str:
    """
    The execution shared by strategies held together (a blend, a walk-forward switch).

    Raises:
        ValueError: If they trade at different auctions.
    """
    executions = {s.execution for s in strategies}
    if len(executions) > 1:
        raise ValueError("strategies trading at the open and at the close cannot be combined")
    return executions.pop() if executions else OPEN


def needed_history(strategy: Strategy) -> int:
    """Bars per symbol a strategy needs before it is asked for targets."""
    return strategy.lookback if strategy.min_history is None else strategy.min_history


def consult(strategy: Strategy, ctx: StrategyContext) -> Mapping[str, float] | None:
    """
    Ask a strategy held inside another one (a blend, a walk-forward switch) for
    its targets, exactly as the engine would ask it on its own: only once it
    has the history it needs, and showing it only its instruments and lookback.
    Returns None while it is still warming up.
    """
    symbols = {i.symbol for i in strategy.instruments}
    needed = needed_history(strategy)
    history = {s: h for s, h in ctx.history.items() if s in symbols}
    if any(len(history.get(s, ())) < needed for s in symbols):
        return None
    return strategy.target_weights(StrategyContext(
        ctx.now, {s: h[-strategy.lookback:] for s, h in history.items()},
        {s: q for s, q in ctx.positions.items() if s in symbols}, ctx.equity))
