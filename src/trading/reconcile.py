"""
NOCTURNA Trading System - Reconciliation with the broker.

The broker is the source of truth for cash, positions and orders. On start-up
and whenever no fills are in flight, the engine is made to agree with it:
unknown open orders are adopted, orders the broker has closed are updated,
and position and cash differences are booked as ledger adjustments. Every
difference is reported, so silent drift is impossible.

An adjustment that raises equity (a deposit, a dividend) moves the risk
gate's high-water mark with it, so it cannot mask a drawdown. One that lowers
equity counts as a loss: a withdrawal can trip the kill switch, which fails
safe, and resuming after a drawdown halt restarts the high-water mark.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .broker import Broker
from .domain import OrderStatus, OrderUpdate
from .engine import Engine


@dataclass(frozen=True)
class Discrepancy:
    item: str  # a symbol, or "cash"
    engine: Decimal
    broker: Decimal

    @property
    def delta(self) -> Decimal:
        return self.broker - self.engine


@dataclass(frozen=True)
class ReconciliationReport:
    timestamp: datetime
    discrepancies: tuple[Discrepancy, ...]
    adopted_orders: tuple[str, ...]
    closed_orders: tuple[str, ...]

    @property
    def in_sync(self) -> bool:
        return not (self.discrepancies or self.adopted_orders or self.closed_orders)


def reconcile(engine: Engine, broker: Broker) -> ReconciliationReport:
    """
    Make the engine agree with the broker. Call only when no order update is
    waiting to be processed, or a fill could be counted twice.
    """
    broker_open = {o.client_order_id: o for o in broker.open_orders()}
    adopted = []
    for client_order_id, order in broker_open.items():
        if client_order_id not in engine.orders:
            engine.sync_order(order)
            adopted.append(client_order_id)

    closed = []
    for order in engine.open_orders():
        client_order_id = order.client_order_id
        if client_order_id not in broker_open:
            final = broker.order(client_order_id)
            if final is None or final.status.is_open:
                # Never reached the broker (e.g. lost in a network failure)
                final = order.apply(OrderUpdate(client_order_id, OrderStatus.CANCELED, engine.clock.now(),
                                                reason="not found at the broker"))
            engine.sync_order(final)
            closed.append(client_order_id)

    discrepancies = []
    ledger = engine.ledger
    equity_before = engine.equity()
    held = broker.positions()
    for symbol in sorted(set(held) | set(ledger.positions)):
        mine, theirs = ledger.position(symbol).qty, held.get(symbol, Decimal("0"))
        if mine != theirs:
            price = engine.prices.get(symbol) or ledger.position(symbol).avg_price
            ledger.adjust_position(symbol, theirs - mine, price)
            discrepancies.append(Discrepancy(symbol, mine, theirs))
    cash = broker.cash()
    if cash != ledger.cash:
        discrepancies.append(Discrepancy("cash", ledger.cash, cash))
        ledger.adjust_cash(cash - ledger.cash)
    gain = engine.equity() - equity_before
    if gain > 0:
        engine.risk.adjust_for_flow(gain)
        engine.emit_risk_state()

    report = ReconciliationReport(engine.clock.now(), tuple(discrepancies), tuple(adopted), tuple(closed))
    if not report.in_sync:
        engine._emit("reconciled", detail=_describe(report))
    return report


def _describe(report: ReconciliationReport) -> str:
    parts = [f"{d.item}: engine {d.engine} -> broker {d.broker}" for d in report.discrepancies]
    if report.adopted_orders:
        parts.append(f"adopted orders {', '.join(report.adopted_orders)}")
    if report.closed_orders:
        parts.append(f"closed orders {', '.join(report.closed_orders)}")
    return "; ".join(parts)
