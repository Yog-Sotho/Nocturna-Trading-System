"""
NOCTURNA Trading System - Broker interface and the simulated broker.

The engine talks to one Broker interface. ``SimulatedBroker`` executes orders
against bars for backtests and replays; ``AlpacaBroker`` (src/trading/alpaca.py)
sends them to Alpaca's paper API. Order updates flow back through a listener,
in the same shape for both.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_UP, Decimal
from typing import Protocol

from .calendar import TradingCalendar
from .clock import Clock
from .domain import (
    NEW_YORK,
    AssetClass,
    Bar,
    Fill,
    Instrument,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    OrderUpdate,
    Side,
    TimeInForce,
)
from .ledger import Ledger

logger = logging.getLogger(__name__)

UpdateListener = Callable[[OrderUpdate], None]

#: Fill prices are rounded to this increment.
PRICE_SCALE = Decimal("1e-6")
BPS = Decimal("10000")

_EQUITY_CALENDAR = TradingCalendar(AssetClass.EQUITY)

#: Time-in-force values Alpaca accepts for crypto orders.
CRYPTO_TIME_IN_FORCE = frozenset({TimeInForce.GTC, TimeInForce.IOC})


class Broker(Protocol):
    def set_listener(self, listener: UpdateListener) -> None:
        """Register the callback that receives every OrderUpdate."""
        ...

    def submit(self, request: OrderRequest) -> None:
        """Send an order; acceptance, fills and rejections arrive as updates."""
        ...

    def cancel(self, client_order_id: str) -> None:
        ...

    def on_bars(self, bars: Mapping[str, Bar]) -> None:
        """New closed bars. The simulator fills against them; real brokers ignore this."""
        ...

    def positions(self) -> dict[str, Decimal]:
        """Signed quantity per symbol, as the broker sees it."""
        ...

    def cost_basis(self) -> dict[str, tuple[Decimal, Decimal]]:
        """(signed qty, signed cost) per symbol, for starting a ledger from the account."""
        ...

    def cash(self) -> Decimal:
        ...

    def open_orders(self) -> list[Order]:
        ...

    def order(self, client_order_id: str) -> Order | None:
        """The broker's current view of one order, or None if it does not know it."""
        ...


@dataclass(frozen=True)
class CostModel:
    """Proportional fee and adverse slippage, both in basis points of notional."""

    fee_bps: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must not be negative")

    def scaled(self, factor: Decimal) -> CostModel:
        """The same model with costs multiplied, for stress tests."""
        return CostModel(self.fee_bps * factor, self.slippage_bps * factor)


#: Alpaca equities are commission-free; 5 bps covers half-spread and impact on
#: liquid ETFs. Crypto pays Alpaca's 25 bps tier-1 taker fee plus wider spreads.
DEFAULT_COSTS: dict[AssetClass, CostModel] = {
    AssetClass.EQUITY: CostModel(Decimal("0"), Decimal("5")),
    AssetClass.CRYPTO: CostModel(Decimal("25"), Decimal("10")),
}


@dataclass
class _Working:
    order: Order
    sequence: int


@dataclass
class SimulatedBroker:
    """
    Executes orders against closed bars, conservatively.

    An order may only fill on a bar whose trading started after the order was
    placed (for daily equity bars: the session open, not New York midnight), so
    it never trades at a price it could not have had. Crypto has no opening
    auction; there an order up to ``open_allowance`` late still gets the open. Market and OPG orders fill
    at the bar's open, CLS orders at its close, both with adverse slippage.
    Limits fill only when the price trades through them (a touch is not
    enough), at the limit or a better open. Stops trigger on a touch and fill at
    the stop, or at the open if it gapped through, with slippage. DAY, OPG, CLS
    and IOC orders expire after their first eligible bar; GTC orders persist.
    Stop-limit orders are not simulated and are rejected.
    """

    clock: Clock
    instruments: Mapping[str, Instrument]
    starting_cash: Decimal
    costs: Mapping[AssetClass, CostModel] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    #: Optional cap on each fill as a fraction of the bar's volume.
    participation: Decimal | None = None
    #: How late an order may arrive after a bar's trading start and still fill at its open.
    open_allowance: Mapping[AssetClass, timedelta] = field(default_factory=lambda: {
        AssetClass.EQUITY: timedelta(0), AssetClass.CRYPTO: timedelta(minutes=2)})

    def __post_init__(self) -> None:
        self._listener: UpdateListener | None = None
        self._working: dict[str, _Working] = {}
        self._closed: dict[str, Order] = {}
        self._seen_ids: set[str] = set()
        # The simulated account keeps its books with the same Ledger as the engine
        self._book = Ledger(self.starting_cash)
        self._sequence = 0
        self._fill_count = 0

    # -- Broker interface ---------------------------------------------------

    def set_listener(self, listener: UpdateListener) -> None:
        self._listener = listener

    def submit(self, request: OrderRequest) -> None:
        now = self.clock.now()
        order = Order.new(request, now)
        if request.client_order_id in self._seen_ids:
            # Brokers refuse a reused client order id and leave the original order
            # alone, which keeps resubmission after a restart idempotent
            logger.warning("Ignoring duplicate client_order_id %s", request.client_order_id)
            return
        self._seen_ids.add(request.client_order_id)

        reason = self._rejection_reason(request)
        if reason:
            self._emit(OrderUpdate(request.client_order_id, OrderStatus.REJECTED, now, reason=reason))
            return
        self._sequence += 1
        accepted = OrderUpdate(request.client_order_id, OrderStatus.NEW, now, broker_order_id=f"sim-{self._sequence}")
        self._working[request.client_order_id] = _Working(order.apply(accepted), self._sequence)
        self._emit(accepted)

    def cancel(self, client_order_id: str) -> None:
        if client_order_id in self._working:
            self._finish(client_order_id, OrderStatus.CANCELED, self.clock.now(), "canceled")

    def on_bars(self, bars: Mapping[str, Bar]) -> None:
        for client_order_id, working in sorted(self._working.items(), key=lambda kv: kv[1].sequence):
            bar = bars.get(working.order.request.symbol)
            if bar is None or not self._eligible(working.order, bar):
                continue
            self._execute(client_order_id, working, bar)

    def positions(self) -> dict[str, Decimal]:
        return {s: p.qty for s, p in self._book.positions.items()}

    def cost_basis(self) -> dict[str, tuple[Decimal, Decimal]]:
        return {s: (p.qty, p.cost) for s, p in self._book.positions.items()}

    def cash(self) -> Decimal:
        return self._book.cash

    def open_orders(self) -> list[Order]:
        return [w.order for w in sorted(self._working.values(), key=lambda w: w.sequence)]

    def order(self, client_order_id: str) -> Order | None:
        working = self._working.get(client_order_id)
        if working is not None:
            return working.order
        return self._closed.get(client_order_id)

    # -- internals ----------------------------------------------------------

    def trading_start(self, bar: Bar) -> datetime:
        """When trading in ``bar`` begins: the session open for daily equity bars."""
        instrument = self.instruments[bar.symbol]
        if instrument.asset_class is AssetClass.EQUITY and bar.end - bar.start >= timedelta(hours=23):
            return _EQUITY_CALENDAR.session_open(bar.start.astimezone(NEW_YORK).date())
        return bar.start

    def _eligible(self, order: Order, bar: Bar) -> bool:
        allowance = self.open_allowance[self.instruments[bar.symbol].asset_class]
        return order.created_at <= self.trading_start(bar) + allowance

    def _emit(self, update: OrderUpdate) -> None:
        if self._listener is not None:
            self._listener(update)

    def _pending_sells(self, symbol: str) -> Decimal:
        return sum((w.order.remaining_qty for w in self._working.values()
                    if w.order.request.symbol == symbol and w.order.request.side is Side.SELL), Decimal("0"))

    def _rejection_reason(self, request: OrderRequest) -> str:
        instrument = self.instruments.get(request.symbol)
        if instrument is None:
            return f"unknown symbol {request.symbol}"
        if request.order_type is OrderType.STOP_LIMIT:
            return "stop_limit orders are not simulated"
        if instrument.asset_class is AssetClass.CRYPTO and request.time_in_force not in CRYPTO_TIME_IN_FORCE:
            return "crypto orders must be gtc or ioc"
        if request.qty < instrument.min_qty or instrument.round_qty(request.qty) != request.qty:
            return f"qty {request.qty} is not a multiple of {instrument.qty_increment} >= {instrument.min_qty}"
        if request.side is Side.SELL and not instrument.shortable:
            held = self._book.position(request.symbol).qty - self._pending_sells(request.symbol)
            if request.qty > held:
                return f"{request.symbol} cannot be sold short"
        return ""

    def _fill_price(self, request: OrderRequest, bar: Bar, slip: Decimal) -> Decimal | None:
        buy = request.side is Side.BUY
        adverse = (1 + slip) if buy else (1 - slip)
        o, h, low, c = (Decimal(repr(x)) for x in (bar.open, bar.high, bar.low, bar.close))
        if request.order_type is OrderType.MARKET:
            return (c if request.time_in_force is TimeInForce.CLS else o) * adverse
        if request.order_type is OrderType.LIMIT:
            limit = request.limit_price
            if buy:
                return o if o <= limit else (limit if low < limit else None)
            return o if o >= limit else (limit if h > limit else None)
        stop = request.stop_price  # OrderType.STOP
        if buy:
            return (o if o >= stop else stop) * adverse if h >= stop else None
        return (o if o <= stop else stop) * adverse if low <= stop else None

    def _execute(self, client_order_id: str, working: _Working, bar: Bar) -> None:
        order = working.order
        request = order.request
        instrument = self.instruments[request.symbol]
        model = self.costs[instrument.asset_class]
        now = self.clock.now()

        price = self._fill_price(request, bar, model.slippage_bps / BPS)
        qty = order.remaining_qty
        if price is not None and self.participation is not None:
            qty = min(qty, instrument.round_qty(Decimal(repr(bar.volume)) * self.participation))

        if price is not None and qty > 0:
            price = price.quantize(PRICE_SCALE)
            fee, fee_qty = (price * qty * model.fee_bps / BPS).quantize(PRICE_SCALE), Decimal("0")
            if instrument.asset_class is AssetClass.CRYPTO and request.side is Side.BUY:
                # Alpaca charges crypto purchases in the asset received; rounding the
                # fee up to the increment never overstates the holding
                fee_qty = (qty * model.fee_bps / BPS / instrument.qty_increment).to_integral_value(
                    rounding=ROUND_UP) * instrument.qty_increment
                fee = Decimal("0")
                if fee_qty >= qty:
                    self._finish(client_order_id, OrderStatus.REJECTED, now, "order too small to cover the fee")
                    return
            if request.side is Side.BUY and price * qty + fee > self._book.cash:
                status = OrderStatus.CANCELED if order.filled_qty else OrderStatus.REJECTED
                self._finish(client_order_id, status, now, "insufficient buying power")
                return
            self._fill_count += 1
            fill = Fill(f"sim-fill-{self._fill_count}", client_order_id, request.symbol, request.side,
                        qty, price, fee, now, fee_qty=fee_qty)
            self._book.apply_fill(fill)
            status = OrderStatus.FILLED if qty == order.remaining_qty else OrderStatus.PARTIALLY_FILLED
            update = OrderUpdate(client_order_id, status, now, fill=fill)
            working.order = order.apply(update)
            if status is OrderStatus.FILLED:
                self._closed[client_order_id] = self._working.pop(client_order_id).order
            self._emit(update)

        if client_order_id in self._working and request.time_in_force is not TimeInForce.GTC:
            # Single-session orders do not survive their first eligible bar
            status = OrderStatus.CANCELED if request.time_in_force is TimeInForce.IOC else OrderStatus.EXPIRED
            self._finish(client_order_id, status, now, f"unfilled {request.time_in_force} order")

    def _finish(self, client_order_id: str, status: OrderStatus, now: datetime, reason: str) -> None:
        working = self._working.pop(client_order_id, None)
        update = OrderUpdate(client_order_id, status, now, reason=reason)
        if working is not None:
            self._closed[client_order_id] = working.order.apply(update)
        self._emit(update)
