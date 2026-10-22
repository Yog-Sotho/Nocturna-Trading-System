"""
NOCTURNA Trading System - What the live engine runs.

The engine runs only the 60/40 benchmark: it exercises orders, fills,
reconciliation and history in paper trading without claiming an edge. A
strategy replaces it only after passing the promotion gates (`flask
validate`); none of the Phase 5 candidates has, alone or as the combined
lineup (see docs/ROADMAP.md). The equity the benchmark leaves unused (the
volatility target and the drawdown ladder scale it down) is held in T-bills
(BIL) instead of cash.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from .domain import AssetClass, Instrument
from .engine import Allocation
from .strategies.benchmark import BuyAndHold

#: The T-bill ETF that holds idle equity (SPDR Bloomberg 1-3 Month T-Bill ETF)
CASH_SLEEVE = "BIL"

#: Symbols the lineup trades, the cash sleeve included, and their asset classes.
LINEUP_SYMBOLS: tuple[tuple[str, AssetClass], ...] = (
    ("SPY", AssetClass.EQUITY),
    ("IEF", AssetClass.EQUITY),
    (CASH_SLEEVE, AssetClass.EQUITY),
)


def build_lineup(instruments: Mapping[str, Instrument]) -> list[Allocation]:
    return [Allocation(BuyAndHold({instruments["SPY"]: 0.6, instruments["IEF"]: 0.4}), Decimal("1"))]
