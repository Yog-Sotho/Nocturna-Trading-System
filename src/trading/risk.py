"""
NOCTURNA Trading System - Risk engine.

Two layers, both running on the engine's single thread:

* ``RiskGate.check`` runs on every order, from a strategy or an operator,
  before it reaches the broker: kill switch, cool-down, market-data sanity,
  limit-price collar, duplicates, daily order and turnover circuit breakers,
  fat-finger size, and position, asset-class and gross exposure caps. Orders
  that only reduce a position skip the circuit breakers, so the book can
  always be flattened.
* ``RiskGate.on_equity`` runs after every mark-to-market: it keeps the
  equity high-water mark, trips the kill switch when the drawdown from it
  reaches ``max_drawdown``, and starts a cool-down (no new risk for the rest
  of the New York day) once the day's loss reaches ``daily_loss_limit``.

The state that must survive a restart (high-water mark, halt, daily counters)
is exported with ``state_dict`` and restored with ``restore``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from .domain import NEW_YORK, AssetClass, OrderRequest, OrderType, Side

ZERO = Decimal("0")


@dataclass(frozen=True)
class RiskLimits:
    """Limits as fractions of account equity unless stated; no leverage by default."""

    # Exposure
    max_position_weight: Decimal = Decimal("0.6")
    max_gross_exposure: Decimal = Decimal("1")
    max_asset_class_weight: Mapping[AssetClass, Decimal] = field(
        default_factory=lambda: {AssetClass.CRYPTO: Decimal("0.15")})
    # Single orders
    max_order_notional: Decimal | None = None
    #: A single order larger than this fraction of equity is a fat finger
    max_order_weight: Decimal = Decimal("0.6")
    #: How far through the reference price a limit may be (buy above, sell below)
    limit_collar: Decimal = Decimal("0.05")
    #: The same symbol, side and quantity again within this window is refused
    duplicate_window: timedelta = timedelta(minutes=5)
    # Daily circuit breakers (New York calendar day)
    max_orders_per_day: int = 50
    #: Total order notional per day, in multiples of equity
    max_daily_turnover: Decimal = Decimal("2")
    # Market data sanity, for orders that add risk
    max_data_age: Mapping[AssetClass, timedelta] = field(default_factory=lambda: {
        AssetClass.EQUITY: timedelta(days=5), AssetClass.CRYPTO: timedelta(days=2)})
    max_price_jump: Mapping[AssetClass, Decimal] = field(default_factory=lambda: {
        AssetClass.EQUITY: Decimal("0.25"), AssetClass.CRYPTO: Decimal("0.5")})
    # Losses
    #: Drawdown from the equity high-water mark that trips the kill switch
    max_drawdown: Decimal = Decimal("0.10")
    flatten_on_kill: bool = True
    #: Loss since the start of the New York day that starts a cool-down until the next day
    daily_loss_limit: Decimal = Decimal("0.03")

    def __post_init__(self) -> None:
        fractions = [self.max_position_weight, self.max_gross_exposure, self.max_order_weight,
                     self.limit_collar, self.max_drawdown, self.daily_loss_limit,
                     *self.max_asset_class_weight.values(), *self.max_price_jump.values()]
        if any(f <= 0 for f in fractions):
            raise ValueError("risk limits must be positive")
        if self.max_drawdown >= 1 or self.daily_loss_limit >= 1:
            raise ValueError("loss limits must be below 100%")
        if self.max_order_notional is not None and self.max_order_notional <= 0:
            raise ValueError("max_order_notional must be positive")
        if self.max_orders_per_day < 1 or self.max_daily_turnover <= 0:
            raise ValueError("daily circuit breakers must be positive")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class OrderContext:
    """What the gate needs to judge an order; built by the engine."""

    now: datetime
    price: Decimal | None  # reference price (last close)
    exposure: Mapping[str, Decimal]  # signed qty per symbol, including open orders
    prices: Mapping[str, Decimal]  # reference prices for every exposed symbol
    asset_classes: Mapping[str, AssetClass]
    equity: Decimal
    last_bar_end: datetime | None  # newest bar of the order's symbol
    price_jump: Decimal | None  # |last close / previous close - 1|


@dataclass
class RiskState:
    high_water_mark: Decimal | None = None
    last_equity: Decimal | None = None
    day_start_equity: Decimal | None = None
    halted: bool = False
    halt_reason: str = ""
    halt_kind: str = ""  # manual, drawdown
    halted_at: datetime | None = None
    cooling_down: bool = False
    day: date | None = None
    orders_today: int = 0
    turnover_today: Decimal = ZERO


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.state = RiskState()
        self._recent: deque[tuple[datetime, str, Side, Decimal]] = deque()

    # -- state --------------------------------------------------------------

    @property
    def halted(self) -> bool:
        return self.state.halted

    @property
    def halt_reason(self) -> str:
        return self.state.halt_reason

    @property
    def cooling_down(self) -> bool:
        return self.state.cooling_down

    @property
    def drawdown(self) -> Decimal:
        """Current drawdown from the high-water mark, 0 when at a new high."""
        hwm, equity = self.state.high_water_mark, self.state.last_equity
        if not hwm or equity is None or hwm <= 0:
            return ZERO
        return max(ZERO, 1 - equity / hwm)

    def halt(self, reason: str, kind: str = "manual", now: datetime | None = None) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.state.halt_kind = kind
        self.state.halted_at = now

    def resume(self, equity: Decimal | None = None) -> None:
        """
        Lift a halt. After a drawdown kill the high-water mark restarts at the
        current equity; otherwise the switch would trip again at once.
        """
        if self.state.halt_kind == "drawdown" and equity is not None:
            self.state.high_water_mark = equity
            self.state.last_equity = equity
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.halt_kind = ""
        self.state.halted_at = None

    def adjust_for_flow(self, amount: Decimal) -> None:
        """Deposits and withdrawals move equity without being gains or losses."""
        if self.state.high_water_mark is not None:
            self.state.high_water_mark += amount
        if self.state.last_equity is not None:
            self.state.last_equity += amount
        if self.state.day_start_equity is not None:
            self.state.day_start_equity += amount

    def _roll_day(self, now: datetime) -> None:
        today = now.astimezone(NEW_YORK).date()
        if self.state.day != today:
            self.state.day = today
            self.state.orders_today = 0
            self.state.turnover_today = ZERO
            self.state.day_start_equity = self.state.last_equity
            self.state.cooling_down = False

    # -- after every mark-to-market -----------------------------------------

    def on_equity(self, equity: Decimal, now: datetime) -> str | None:
        """
        Update loss tracking with a new mark.

        Returns:
            "kill" when the drawdown limit is reached (the caller halts and, with
            ``flatten_on_kill``, flattens), "cooldown" when the day's loss
            reaches the daily limit, otherwise None.
        """
        self._roll_day(now)
        state = self.state
        state.last_equity = equity
        if state.day_start_equity is None:
            state.day_start_equity = equity
        if state.high_water_mark is None or equity > state.high_water_mark:
            state.high_water_mark = equity
        start = state.day_start_equity
        if start > 0 and equity <= start * (1 - self.limits.daily_loss_limit):
            state.cooling_down = True
        if not state.halted and self.drawdown >= self.limits.max_drawdown:
            self.halt(f"drawdown {self.drawdown:.2%} reached the {self.limits.max_drawdown:.0%} limit",
                      kind="drawdown", now=now)
            return "kill"
        return "cooldown" if state.cooling_down else None

    # -- every order ----------------------------------------------------------

    def check(self, request: OrderRequest, ctx: OrderContext) -> RiskDecision:
        limits = self.limits
        self._roll_day(ctx.now)
        current = ctx.exposure.get(request.symbol, ZERO)
        after = current + request.qty * request.side.sign
        reducing = abs(after) < abs(current) and (after == 0 or (after > 0) == (current > 0))
        asset_class = ctx.asset_classes.get(request.symbol, AssetClass.EQUITY)

        if self.halted and not reducing:
            return RiskDecision(False, f"trading halted: {self.halt_reason}")
        if ctx.price is None or ctx.price <= 0:
            return RiskDecision(False, f"no reference price for {request.symbol}")
        collar = self._collar(request, ctx.price)
        if collar:
            return RiskDecision(False, collar)
        if self._is_duplicate(request, ctx.now):
            return RiskDecision(False, f"duplicate of an order sent in the last {limits.duplicate_window}")
        if reducing:
            return RiskDecision(True)

        if self.cooling_down:
            return RiskDecision(False, f"cool-down: today's loss reached {limits.daily_loss_limit:.0%}")
        if ctx.equity <= 0:
            return RiskDecision(False, "account equity is not positive")
        if ctx.last_bar_end is None or ctx.now - ctx.last_bar_end > limits.max_data_age[asset_class]:
            return RiskDecision(False, f"market data for {request.symbol} is stale")
        jump = limits.max_price_jump[asset_class]
        if ctx.price_jump is not None and ctx.price_jump > jump:
            return RiskDecision(False, f"{request.symbol} moved {ctx.price_jump:.0%} in one bar; "
                                       "suspected bad data")
        if self.state.orders_today >= limits.max_orders_per_day:
            return RiskDecision(False, f"daily order limit of {limits.max_orders_per_day} reached")

        notional = request.qty * ctx.price
        if self.state.turnover_today + notional > limits.max_daily_turnover * ctx.equity:
            return RiskDecision(False, f"daily turnover limit of {limits.max_daily_turnover}x equity reached")
        if limits.max_order_notional is not None and notional > limits.max_order_notional:
            return RiskDecision(False, f"order notional {notional:.2f} exceeds {limits.max_order_notional}")
        if notional > limits.max_order_weight * ctx.equity:
            return RiskDecision(False, f"order of {notional / ctx.equity:.0%} of equity exceeds the "
                                       f"{limits.max_order_weight:.0%} single-order limit")
        if abs(after) * ctx.price > limits.max_position_weight * ctx.equity:
            return RiskDecision(False, f"{request.symbol} position would exceed "
                                       f"{limits.max_position_weight:.0%} of equity")

        values = {s: abs(q) * ctx.prices.get(s, ZERO) for s, q in ctx.exposure.items() if q and s != request.symbol}
        values[request.symbol] = abs(after) * ctx.price
        cap = limits.max_asset_class_weight.get(asset_class)
        if cap is not None:
            in_class = sum((v for s, v in values.items() if ctx.asset_classes.get(s) is asset_class), ZERO)
            if in_class > cap * ctx.equity:
                return RiskDecision(False, f"{asset_class} exposure would exceed {cap:.0%} of equity")
        if sum(values.values(), ZERO) > limits.max_gross_exposure * ctx.equity:
            return RiskDecision(False, f"gross exposure would exceed {limits.max_gross_exposure:.0%} of equity")
        return RiskDecision(True)

    def record(self, request: OrderRequest, price: Decimal, now: datetime) -> None:
        """Count an order that was sent (call after an allowed check)."""
        self._roll_day(now)
        self.state.orders_today += 1
        self.state.turnover_today += request.qty * price
        self._recent.append((now, request.symbol, request.side, request.qty))

    def _collar(self, request: OrderRequest, price: Decimal) -> str:
        if request.order_type not in (OrderType.LIMIT, OrderType.STOP_LIMIT) or request.limit_price is None:
            return ""
        collar = self.limits.limit_collar
        if request.side is Side.BUY and request.limit_price > price * (1 + collar):
            return f"buy limit {request.limit_price} is more than {collar:.0%} above the reference price {price}"
        if request.side is Side.SELL and request.limit_price < price * (1 - collar):
            return f"sell limit {request.limit_price} is more than {collar:.0%} below the reference price {price}"
        return ""

    def _is_duplicate(self, request: OrderRequest, now: datetime) -> bool:
        while self._recent and now - self._recent[0][0] > self.limits.duplicate_window:
            self._recent.popleft()
        return any(s == request.symbol and side is request.side and q == request.qty
                   for _, s, side, q in self._recent)

    # -- persistence ----------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "high_water_mark": str(s.high_water_mark) if s.high_water_mark is not None else None,
            "last_equity": str(s.last_equity) if s.last_equity is not None else None,
            "day_start_equity": str(s.day_start_equity) if s.day_start_equity is not None else None,
            "halted": s.halted, "halt_reason": s.halt_reason, "halt_kind": s.halt_kind,
            "halted_at": s.halted_at.isoformat() if s.halted_at else None,
            "cooling_down": s.cooling_down,
            "day": s.day.isoformat() if s.day else None,
            "orders_today": s.orders_today, "turnover_today": str(s.turnover_today),
        }

    def restore(self, data: Mapping[str, Any]) -> None:
        def dec(key: str) -> Decimal | None:
            return Decimal(data[key]) if data.get(key) is not None else None

        self.state = RiskState(
            high_water_mark=dec("high_water_mark"),
            last_equity=dec("last_equity"),
            day_start_equity=dec("day_start_equity"),
            halted=bool(data.get("halted")),
            halt_reason=data.get("halt_reason") or "",
            halt_kind=data.get("halt_kind") or "",
            halted_at=datetime.fromisoformat(data["halted_at"]) if data.get("halted_at") else None,
            cooling_down=bool(data.get("cooling_down")),
            day=date.fromisoformat(data["day"]) if data.get("day") else None,
            orders_today=int(data.get("orders_today") or 0),
            turnover_today=dec("turnover_today") or ZERO,
        )
