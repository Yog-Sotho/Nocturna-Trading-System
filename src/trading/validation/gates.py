"""
NOCTURNA Trading System - Promotion gates.

The gates from docs/ROADMAP.md that a backtest can decide. A candidate that
passes all of them may paper trade; the paper track record gate is decided
later, by paper trading itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionGates:
    #: Walk-forward out-of-sample Sharpe ratio, annualized, net of costs (portfolio level)
    min_oos_sharpe: float = 1.0
    min_deflated_sharpe: float = 0.95
    max_pbo: float = 0.20
    #: Must still make money with every cost multiplied by this
    stress_multiple: float = 3.0
    #: Median Sharpe of the neighbouring parameter sets over the selected one's
    min_neighbour_ratio: float = 0.5
    #: Deepest out-of-sample drawdown: the 10% kill switch plus 2 points for gaps
    max_drawdown: float = 0.12


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool | None  # None: not applicable or decided later
    value: float | None
    threshold: str
    note: str = ""


def verdict(results: list[GateResult]) -> tuple[bool, str]:
    failed = [r.name for r in results if r.passed is False]
    if failed:
        return False, "rejected: fails " + ", ".join(failed)
    return True, "eligible for paper trading (the paper track record gate is still open)"
