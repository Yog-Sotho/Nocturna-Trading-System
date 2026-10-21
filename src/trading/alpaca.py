"""
NOCTURNA Trading System - Alpaca paper broker (alpaca-py).

Translates between the engine's domain model and Alpaca's Trading API. Order
reports arrive on Alpaca's trade-update websocket and are handed to the
listener from the stream's thread; the engine service queues them so the
engine itself stays single-threaded.

Live trading is disabled: the adapter refuses to start unless the paper-only
guard passes, and it only ever connects to the paper endpoint.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from decimal import ROUND_UP, Decimal
from typing import Any

from src.trading.safety import assert_paper_only

from .broker import BPS, UpdateListener
from .clock import Clock
from .domain import (
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

logger = logging.getLogger(__name__)

#: Alpaca order statuses -> domain statuses. ``done_for_day`` and the
#: replace/hold states keep an order working, so they map to an open status.
_ORDER_STATUS = {
    "pending_new": OrderStatus.PENDING_NEW,
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.NEW,
    "held": OrderStatus.NEW,
    "calculated": OrderStatus.NEW,
    "stopped": OrderStatus.NEW,
    "suspended": OrderStatus.NEW,
    "pending_replace": OrderStatus.NEW,
    "done_for_day": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "pending_cancel": OrderStatus.PENDING_CANCEL,
    "canceled": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}

#: Trade-update events -> domain statuses. Events not listed (pending_replace,
#: restated) change nothing the engine tracks.
_EVENT_STATUS = {
    "pending_new": OrderStatus.PENDING_NEW,
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "partial_fill": OrderStatus.PARTIALLY_FILLED,
    "fill": OrderStatus.FILLED,
    "pending_cancel": OrderStatus.PENDING_CANCEL,
    "canceled": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


def _value(enum_or_str: Any) -> str:
    return getattr(enum_or_str, "value", enum_or_str)


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value)) if value is not None else Decimal("0")


class AlpacaBroker:
    """
    The Broker interface on Alpaca's paper Trading API.

    Crypto fees are estimated from ``crypto_fee_bps`` when a fill is booked
    (Alpaca charges purchases in the asset received and sales in USD); the
    exact amounts are taken from the account at the next reconciliation.
    """

    def __init__(self, *, api_key: str, secret_key: str, instruments: Mapping[str, Instrument], clock: Clock,
                 crypto_fee_bps: Decimal = Decimal("25"), trading_client: Any = None) -> None:
        assert_paper_only()
        if trading_client is None:
            from alpaca.trading.client import TradingClient

            trading_client = TradingClient(api_key, secret_key, paper=True)
        self._client = trading_client
        self._instruments = dict(instruments)
        self._clock = clock
        self._crypto_fee_bps = crypto_fee_bps
        self._listener: UpdateListener | None = None
        self._lock = threading.Lock()
        # Alpaca reports crypto pairs as "BTC/USD" or, in older responses, "BTCUSD"
        self._symbols = {s: s for s in self._instruments}
        self._symbols.update({s.replace("/", ""): s for s in self._instruments})

    # -- Broker interface ---------------------------------------------------

    def set_listener(self, listener: UpdateListener) -> None:
        self._listener = listener

    def submit(self, request: OrderRequest) -> None:
        from alpaca.common.exceptions import APIError

        try:
            order = self._client.submit_order(self._to_alpaca(request))
        except APIError as e:
            if e.status_code == 422 and "client_order_id" in str(e):
                # Already placed (e.g. retried after a restart): report the original
                self._report_existing(request.client_order_id)
                return
            self._emit(OrderUpdate(request.client_order_id, OrderStatus.REJECTED, self._clock.now(),
                                   reason=f"Alpaca rejected the order: {e}"))
            return
        except Exception as e:  # network failure: the order may or may not exist
            logger.error("Order %s submission failed: %s", request.client_order_id, e)
            self._report_existing(request.client_order_id, missing_status=OrderStatus.REJECTED,
                                  missing_reason=f"submission failed: {type(e).__name__}")
            return
        self._emit(self._update_from_order(order))

    def cancel(self, client_order_id: str) -> None:
        try:
            order = self._client.get_order_by_client_id(client_order_id)
            self._client.cancel_order_by_id(order.id)
        except Exception as e:  # already final, or unknown; the stream reports the outcome
            logger.warning("Cancel of %s failed: %s", client_order_id, e)

    def on_bars(self, bars: Mapping[str, Bar]) -> None:
        """Alpaca executes orders itself."""

    def positions(self) -> dict[str, Decimal]:
        held = {}
        for p in self._client.get_all_positions():
            symbol = self._symbols.get(p.symbol, p.symbol)
            qty = abs(_decimal(p.qty))
            held[symbol] = -qty if _value(p.side) == "short" else qty
        return {s: q for s, q in held.items() if q}

    def cost_basis(self) -> dict[str, tuple[Decimal, Decimal]]:
        """(signed qty, signed cost) per symbol, for starting a ledger from the account."""
        basis = {}
        for p in self._client.get_all_positions():
            symbol = self._symbols.get(p.symbol, p.symbol)
            qty = abs(_decimal(p.qty))
            if _value(p.side) == "short":
                qty = -qty
            basis[symbol] = (qty, qty * _decimal(p.avg_entry_price))
        return basis

    def cash(self) -> Decimal:
        return _decimal(self._client.get_account().cash)

    def open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [o for o in (self._to_domain(a) for a in orders) if o is not None]

    def order(self, client_order_id: str) -> Order | None:
        try:
            return self._to_domain(self._client.get_order_by_client_id(client_order_id))
        except Exception:
            return None

    def close_all(self) -> list[str]:
        """Alpaca's own liquidation: cancels every open order and closes every position."""
        responses = self._client.close_all_positions(cancel_orders=True)
        return [str(r.order_id) if r.order_id else f"{r.symbol}: failed ({r.status})" for r in responses]

    # -- trade updates ------------------------------------------------------

    def handle_trade_update(self, update: Any) -> None:
        """Called from the trade-update stream with an alpaca-py TradeUpdate."""
        event = _value(update.event)
        status = _EVENT_STATUS.get(event)
        if status is None:
            return
        order = update.order
        fill = None
        if event in ("fill", "partial_fill") and update.qty and update.price:
            fill = self._fill_from(order, update)
        self._emit(OrderUpdate(order.client_order_id, status, update.timestamp, fill=fill,
                               broker_order_id=str(order.id)))

    def _fill_from(self, order: Any, update: Any) -> Fill:
        symbol = self._symbols.get(order.symbol, order.symbol)
        side = Side(_value(order.side))
        qty, price = _decimal(update.qty), _decimal(update.price)
        fee, fee_qty = Decimal("0"), Decimal("0")
        instrument = self._instruments.get(symbol)
        if instrument is not None and instrument.asset_class is AssetClass.CRYPTO:
            if side is Side.BUY:
                fee_qty = (qty * self._crypto_fee_bps / BPS / instrument.qty_increment).to_integral_value(
                    rounding=ROUND_UP) * instrument.qty_increment
                fee_qty = min(fee_qty, qty / 2)
            else:
                fee = qty * price * self._crypto_fee_bps / BPS
        execution = update.execution_id or f"{order.id}:{_decimal(order.filled_qty)}"
        return Fill(f"alpaca:{execution}", order.client_order_id, symbol, side, qty, price, fee,
                    update.timestamp, fee_qty=fee_qty)

    # -- conversions --------------------------------------------------------

    def _emit(self, update: OrderUpdate) -> None:
        if self._listener is not None:
            self._listener(update)

    def _report_existing(self, client_order_id: str, missing_status: OrderStatus | None = None,
                         missing_reason: str = "") -> None:
        try:
            existing = self._client.get_order_by_client_id(client_order_id)
        except Exception:
            if missing_status is not None:
                self._emit(OrderUpdate(client_order_id, missing_status, self._clock.now(), reason=missing_reason))
            return
        self._emit(self._update_from_order(existing))

    def _update_from_order(self, order: Any) -> OrderUpdate:
        status = _ORDER_STATUS.get(_value(order.status), OrderStatus.NEW)
        return OrderUpdate(order.client_order_id, status, self._clock.now(), broker_order_id=str(order.id),
                           reason=_value(order.status))

    def _to_domain(self, order: Any) -> Order | None:
        symbol = self._symbols.get(order.symbol, order.symbol)
        try:
            request = OrderRequest(
                client_order_id=order.client_order_id, symbol=symbol, side=Side(_value(order.side)),
                qty=_decimal(order.qty), order_type=OrderType(_value(order.order_type or order.type)),
                time_in_force=TimeInForce(_value(order.time_in_force)),
                limit_price=_decimal(order.limit_price) if order.limit_price else None,
                stop_price=_decimal(order.stop_price) if order.stop_price else None,
            )
        except ValueError as e:  # an order type the engine does not model (e.g. trailing stop)
            logger.warning("Ignoring Alpaca order %s: %s", order.client_order_id, e)
            return None
        filled = _decimal(order.filled_qty)
        return Order(
            request=request,
            status=_ORDER_STATUS.get(_value(order.status), OrderStatus.NEW),
            created_at=order.created_at,
            updated_at=order.updated_at,
            broker_order_id=str(order.id),
            filled_qty=filled,
            avg_fill_price=_decimal(order.filled_avg_price) if filled else None,
        )

    def _to_alpaca(self, request: OrderRequest) -> Any:
        from alpaca.trading import enums
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        common = dict(symbol=request.symbol, qty=float(request.qty),
                      side=enums.OrderSide(request.side.value),
                      time_in_force=enums.TimeInForce(request.time_in_force.value),
                      client_order_id=request.client_order_id)
        if request.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if request.order_type is OrderType.LIMIT:
            return LimitOrderRequest(**common, limit_price=float(request.limit_price))
        if request.order_type is OrderType.STOP:
            return StopOrderRequest(**common, stop_price=float(request.stop_price))
        return StopLimitOrderRequest(**common, limit_price=float(request.limit_price),
                                     stop_price=float(request.stop_price))


def load_instruments(client: Any, wanted: Iterable[tuple[str, AssetClass]]) -> dict[str, Instrument]:
    """
    Build instruments from Alpaca's asset records. Equities trade in whole
    shares (fractional orders cannot use the opening auction); crypto uses
    Alpaca's minimum order size and increment.

    Raises:
        ValueError: If an asset is missing or not tradable.
    """
    instruments = {}
    for symbol, asset_class in wanted:
        asset = client.get_asset(symbol)
        if not asset.tradable or _value(asset.status) != "active":
            raise ValueError(f"{symbol} is not tradable on Alpaca")
        if asset_class is AssetClass.CRYPTO:
            increment = _decimal(asset.min_trade_increment) if asset.min_trade_increment else Decimal("1e-9")
            minimum = _decimal(asset.min_order_size) if asset.min_order_size else increment
            instruments[symbol] = Instrument(symbol, asset_class, qty_increment=increment, min_qty=minimum)
        else:
            instruments[symbol] = Instrument(symbol, asset_class, shortable=bool(asset.shortable and
                                                                                 asset.easy_to_borrow))
    return instruments


class TradeUpdateStream:
    """Runs Alpaca's trade-update websocket in a daemon thread."""

    def __init__(self, *, api_key: str, secret_key: str, on_update: Callable[[Any], None],
                 stream: Any = None) -> None:
        if stream is None:
            from alpaca.trading.stream import TradingStream

            stream = TradingStream(api_key, secret_key, paper=True)
        self._stream = stream
        self._on_update = on_update
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        async def handler(update: Any) -> None:
            try:
                self._on_update(update)
            except Exception:
                logger.exception("Trade update handler failed")

        self._stream.subscribe_trade_updates(handler)
        self._thread = threading.Thread(target=self._stream.run, name="alpaca-trade-updates", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        try:
            self._stream.stop()
        except Exception as e:
            logger.warning("Stopping the trade update stream failed: %s", e)
