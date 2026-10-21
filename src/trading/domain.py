"""
NOCTURNA Trading System - Trading domain model.

Plain, immutable values shared by backtest, paper and live trading. Money and
quantities are ``Decimal``; bar prices are ``float`` because they feed numeric
indicators and are approximations anyway. Every datetime is timezone-aware UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be a timezone-aware UTC datetime, got {value!r}")


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    OPG = "opg"
    CLS = "cls"


class OrderStatus(StrEnum):
    PENDING_NEW = "pending_new"  # created locally, not yet acknowledged by the broker
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    PENDING_CANCEL = "pending_cancel"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_open(self) -> bool:
        return not self.is_terminal


_TERMINAL = frozenset({OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED})

# Fills can race a cancel request, and a broker may report a fill without a
# separate acknowledgement, so those transitions are allowed.
_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING_NEW: frozenset({
        OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
        OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED,
    }),
    OrderStatus.NEW: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED,
    }),
    OrderStatus.PARTIALLY_FILLED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED, OrderStatus.EXPIRED,
    }),
    OrderStatus.PENDING_CANCEL: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED,
    }),
}


class InvalidTransitionError(ValueError):
    """Raised when an order update would move an order to an impossible state."""


@dataclass(frozen=True)
class Instrument:
    """A tradable asset and the increments the broker accepts for it."""

    symbol: str
    asset_class: AssetClass
    qty_increment: Decimal = Decimal("1")
    min_qty: Decimal = Decimal("1")
    shortable: bool = False

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.qty_increment <= 0 or self.min_qty <= 0:
            raise ValueError("qty_increment and min_qty must be positive")

    def round_qty(self, qty: Decimal) -> Decimal:
        """Round a quantity toward zero to the instrument's increment."""
        steps = (abs(qty) / self.qty_increment).to_integral_value(rounding=ROUND_DOWN)
        rounded = steps * self.qty_increment
        return rounded if qty >= 0 else -rounded


@dataclass(frozen=True)
class Timeframe:
    """Bar length. Daily equity bars follow the New York calendar day."""

    amount: int
    unit: str  # "minute", "hour" or "day"

    def __post_init__(self) -> None:
        if self.amount <= 0 or self.unit not in ("minute", "hour", "day"):
            raise ValueError(f"invalid timeframe {self.amount} {self.unit}")

    def bar_end(self, start: datetime, asset_class: AssetClass) -> datetime:
        """End (exclusive) of the bar that starts at ``start``."""
        if self.unit == "minute":
            return start + timedelta(minutes=self.amount)
        if self.unit == "hour":
            return start + timedelta(hours=self.amount)
        if asset_class is AssetClass.EQUITY:
            # Calendar days in New York, so DST changes do not shift the boundary
            # (aware arithmetic on a ZoneInfo datetime keeps the wall-clock time)
            return (start.astimezone(NEW_YORK) + timedelta(days=self.amount)).astimezone(UTC)
        return start + timedelta(days=self.amount)

    def next_boundary(self, after: datetime, asset_class: AssetClass) -> datetime:
        """The first bar boundary strictly after ``after``."""
        if self.unit == "day":
            if asset_class is AssetClass.EQUITY:
                local = after.astimezone(NEW_YORK)
                midnight = datetime(local.year, local.month, local.day, tzinfo=NEW_YORK)
            else:
                midnight = datetime(after.year, after.month, after.day, tzinfo=UTC)
            return (midnight + timedelta(days=1)).astimezone(UTC)
        step = timedelta(minutes=self.amount) if self.unit == "minute" else timedelta(hours=self.amount)
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return epoch + ((after - epoch) // step + 1) * step

    def __str__(self) -> str:
        return f"{self.amount}{self.unit[0]}"


DAILY = Timeframe(1, "day")


@dataclass(frozen=True)
class Bar:
    """OHLCV over the half-open interval [start, end)."""

    symbol: str
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        _require_utc("start", self.start)
        _require_utc("end", self.end)
        if self.end <= self.start:
            raise ValueError(f"{self.symbol}: bar end must be after start")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError(f"{self.symbol}: prices must be positive and volume non-negative")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"{self.symbol}: high/low do not bracket open/close at {self.start}")


@dataclass(frozen=True)
class OrderRequest:
    """What a strategy or operator asks the broker to do."""

    client_order_id: str
    symbol: str
    side: Side
    qty: Decimal
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.client_order_id or len(self.client_order_id) > 48:
            raise ValueError("client_order_id must be 1-48 characters")
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        needs_limit = self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
        needs_stop = self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
        if needs_limit != (self.limit_price is not None):
            raise ValueError(f"{self.order_type} orders {'need' if needs_limit else 'take no'} limit_price")
        if needs_stop != (self.stop_price is not None):
            raise ValueError(f"{self.order_type} orders {'need' if needs_stop else 'take no'} stop_price")
        for price in (self.limit_price, self.stop_price):
            if price is not None and price <= 0:
                raise ValueError("prices must be positive")


@dataclass(frozen=True)
class Fill:
    """One execution. ``fill_id`` is unique per execution, so replays can be ignored."""

    fill_id: str
    client_order_id: str
    symbol: str
    side: Side
    qty: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_utc("timestamp", self.timestamp)
        if self.qty <= 0 or self.price <= 0 or self.fee < 0:
            raise ValueError("fill qty and price must be positive and fee non-negative")

    @property
    def signed_qty(self) -> Decimal:
        return self.qty * self.side.sign


@dataclass(frozen=True)
class OrderUpdate:
    """A broker report about an order, optionally carrying a fill."""

    client_order_id: str
    status: OrderStatus
    timestamp: datetime
    fill: Fill | None = None
    broker_order_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class Order:
    """An order and its lifecycle. Updates return a new Order."""

    request: OrderRequest
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    broker_order_id: str | None = None
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    reason: str = ""
    fill_ids: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def new(cls, request: OrderRequest, now: datetime) -> Order:
        _require_utc("now", now)
        return cls(request=request, status=OrderStatus.PENDING_NEW, created_at=now, updated_at=now)

    @property
    def client_order_id(self) -> str:
        return self.request.client_order_id

    @property
    def remaining_qty(self) -> Decimal:
        return self.request.qty - self.filled_qty

    def apply(self, update: OrderUpdate) -> Order:
        """
        Apply a broker update.

        Raises:
            InvalidTransitionError: For an impossible transition, an update for another
                order, or a fill that would overfill the order.
        """
        if update.client_order_id != self.client_order_id:
            raise InvalidTransitionError("update is for a different order")
        fill = update.fill
        if fill is not None and fill.fill_id in self.fill_ids:
            return self  # replayed execution report
        if update.status != self.status and update.status not in _ALLOWED_TRANSITIONS.get(self.status, ()):
            raise InvalidTransitionError(f"{self.client_order_id}: {self.status} -> {update.status}")

        filled_qty, avg_price, fill_ids = self.filled_qty, self.avg_fill_price, self.fill_ids
        if fill is not None:
            if fill.symbol != self.request.symbol or fill.side != self.request.side:
                raise InvalidTransitionError(f"{self.client_order_id}: fill does not match the order")
            filled_qty = self.filled_qty + fill.qty
            if filled_qty > self.request.qty:
                raise InvalidTransitionError(f"{self.client_order_id}: fill exceeds order quantity")
            notional = (self.avg_fill_price or Decimal("0")) * self.filled_qty + fill.price * fill.qty
            avg_price = notional / filled_qty
            fill_ids = self.fill_ids | {fill.fill_id}

        return replace(
            self,
            status=update.status,
            updated_at=update.timestamp,
            broker_order_id=update.broker_order_id or self.broker_order_id,
            filled_qty=filled_qty,
            avg_fill_price=avg_price,
            reason=update.reason or self.reason,
            fill_ids=fill_ids,
        )
