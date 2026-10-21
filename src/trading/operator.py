"""
NOCTURNA Trading System - Orders from the operator.

An operator's order is described by plain values (so it can cross from the
API to a separate engine process) and turned into an order request on the
engine's own thread, where the instrument (for the default time in force)
and the risk gate are.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from .domain import AssetClass, Order, OrderRequest, OrderType, Side, TimeInForce
from .engine import Engine
from .risk import RiskDecision


class UnknownSymbolError(ValueError):
    """The engine does not trade the order's symbol."""


@dataclass(frozen=True)
class OperatorOrder:
    client_order_id: str
    symbol: str
    side: str
    qty: Decimal
    order_type: str = "market"
    time_in_force: str | None = None  # None: GTC for crypto, DAY for equities
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorOrder:
        def dec(key: str) -> Decimal | None:
            return Decimal(data[key]) if data.get(key) is not None else None

        return cls(client_order_id=data["client_order_id"], symbol=data["symbol"], side=data["side"],
                   qty=Decimal(data["qty"]), order_type=data.get("order_type", "market"),
                   time_in_force=data.get("time_in_force"), limit_price=dec("limit_price"),
                   stop_price=dec("stop_price"), tag=data.get("tag", ""))


def place(engine: Engine, order: OperatorOrder) -> tuple[RiskDecision, Order | None]:
    """
    Send the order through the risk gate (call on the engine thread).

    Raises:
        UnknownSymbolError: If the engine does not trade the symbol.
    """
    instrument = engine.instruments.get(order.symbol)
    if instrument is None:
        raise UnknownSymbolError(f"{order.symbol} is not traded by the engine")
    default = TimeInForce.GTC if instrument.asset_class is AssetClass.CRYPTO else TimeInForce.DAY
    request = OrderRequest(
        client_order_id=order.client_order_id, symbol=order.symbol, side=Side(order.side), qty=order.qty,
        order_type=OrderType(order.order_type),
        time_in_force=TimeInForce(order.time_in_force) if order.time_in_force else default,
        limit_price=order.limit_price, stop_price=order.stop_price, tag=order.tag)
    decision = engine.submit_order(request)
    return decision, engine.orders.get(request.client_order_id)
