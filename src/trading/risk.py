"""
NOCTURNA Trading System - Pre-trade risk gate.

Every order, from a strategy or an operator, passes here before it reaches a
broker. Phase 2 enforces exposure caps and the kill switch; Phase 3 adds loss
limits, price collars and duplicate detection behind the same interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from .domain import OrderRequest


@dataclass(frozen=True)
class RiskLimits:
    """Caps as fractions of account equity; no leverage by default."""

    max_position_weight: Decimal = Decimal("1")
    max_gross_exposure: Decimal = Decimal("1")
    max_order_notional: Decimal | None = None

    def __post_init__(self) -> None:
        if self.max_position_weight <= 0 or self.max_gross_exposure <= 0:
            raise ValueError("exposure limits must be positive")
        if self.max_order_notional is not None and self.max_order_notional <= 0:
            raise ValueError("max_order_notional must be positive")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""


class RiskGate:
    """
    Checks orders against limits. While halted (the kill switch), only orders
    that reduce an existing position are allowed, so the book can be flattened.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.halted = False
        self.halt_reason = ""

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def resume(self) -> None:
        self.halted = False
        self.halt_reason = ""

    def check(self, request: OrderRequest, price: Decimal | None, positions: Mapping[str, Decimal],
              prices: Mapping[str, Decimal], equity: Decimal) -> RiskDecision:
        """
        Args:
            request: The order to check.
            price: Reference price for the order's symbol (last close).
            positions: Signed quantities, including quantities of open orders.
            prices: Reference prices for every held symbol.
            equity: Account equity at reference prices.
        """
        current = positions.get(request.symbol, Decimal("0"))
        after = current + request.qty * request.side.sign
        reducing = abs(after) < abs(current) and (after == 0 or (after > 0) == (current > 0))
        if self.halted and not reducing:
            return RiskDecision(False, f"trading halted: {self.halt_reason}")
        if price is None or price <= 0:
            return RiskDecision(False, f"no reference price for {request.symbol}")
        if reducing:
            return RiskDecision(True)
        if equity <= 0:
            return RiskDecision(False, "account equity is not positive")

        notional = request.qty * price
        if self.limits.max_order_notional is not None and notional > self.limits.max_order_notional:
            return RiskDecision(False, f"order notional {notional:.2f} exceeds {self.limits.max_order_notional}")
        if abs(after) * price > self.limits.max_position_weight * equity:
            return RiskDecision(False, f"{request.symbol} position would exceed "
                                       f"{self.limits.max_position_weight:.0%} of equity")
        gross = sum((abs(q) * prices[s] for s, q in positions.items() if s != request.symbol and q),
                    Decimal("0")) + abs(after) * price
        if gross > self.limits.max_gross_exposure * equity:
            return RiskDecision(False, f"gross exposure would exceed {self.limits.max_gross_exposure:.0%} of equity")
        return RiskDecision(True)
