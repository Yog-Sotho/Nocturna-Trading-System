"""
NOCTURNA Trading System - What the live engine runs.

Until a candidate strategy passes the Phase 4 validation gates, the engine
runs only the 60/40 benchmark: it exercises orders, fills, reconciliation and
history in paper trading without claiming an edge. Phase 5 replaces it with
the validated lineup (see docs/ROADMAP.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from .domain import AssetClass, Instrument
from .engine import Allocation
from .strategies.benchmark import BuyAndHold

#: Symbols the lineup trades, and their asset classes.
LINEUP_SYMBOLS: tuple[tuple[str, AssetClass], ...] = (
    ("SPY", AssetClass.EQUITY),
    ("IEF", AssetClass.EQUITY),
)


def build_lineup(instruments: Mapping[str, Instrument]) -> list[Allocation]:
    return [Allocation(BuyAndHold({instruments["SPY"]: 0.6, instruments["IEF"]: 0.4}), Decimal("1"))]
