"""
NOCTURNA Trading System - Recording engine history.

An engine listener that writes orders, fills and notable events to the
database. It runs on the engine's thread with its own app context; a database
failure is logged and never stops trading.
"""

from __future__ import annotations

import logging

from flask import Flask

from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
from src.models.user import db
from src.utils.audit import record_audit_event

from .domain import Fill, Order
from .engine import EngineEvent, EngineSnapshot

logger = logging.getLogger(__name__)

#: Engine events that go to the audit trail, and whether each counts as a success.
_AUDITED = {
    "halt": True,
    "resume": True,
    "reconciled": True,
    "rejected": False,
    "strategy_error": False,
    "broker_error": False,
    "unknown_fill": False,
}


class EngineRecorder:
    def __init__(self, app: Flask) -> None:
        self._app = app

    def __call__(self, event: EngineEvent) -> None:
        with self._app.app_context():
            try:
                if event.kind == "order" and event.order is not None:
                    self._save_order(event.order)
                elif event.kind == "fill" and event.fill is not None:
                    self._save_fill(event.fill)
                db.session.commit()
            except Exception:
                db.session.rollback()
                logger.exception("Recording %s event failed", event.kind)
            if event.kind in _AUDITED:
                details = {"detail": event.detail}
                if event.fill is not None:
                    details.update(symbol=event.fill.symbol, qty=str(event.fill.qty), price=str(event.fill.price))
                record_audit_event(f"ENGINE_{event.kind.upper()}", _AUDITED[event.kind], details)

    def record_snapshot(self, snapshot: EngineSnapshot) -> None:
        with self._app.app_context():
            try:
                db.session.add(EquitySnapshot(
                    timestamp=snapshot.timestamp,
                    equity=snapshot.equity,
                    cash=snapshot.cash,
                    gross_exposure=sum((abs(p.market_value) for p in snapshot.positions), 0),
                    realized_pnl=snapshot.realized_pnl,
                    fees=snapshot.fees,
                    halted=snapshot.halted,
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()
                logger.exception("Recording the equity snapshot failed")

    @staticmethod
    def _save_order(order: Order) -> None:
        request = order.request
        row = TradingOrder.query.filter_by(client_order_id=order.client_order_id).first()
        if row is None:
            row = TradingOrder(client_order_id=order.client_order_id, symbol=request.symbol,
                               side=request.side.value, qty=request.qty, order_type=request.order_type.value,
                               time_in_force=request.time_in_force.value, limit_price=request.limit_price,
                               stop_price=request.stop_price, tag=request.tag or None,
                               created_at=order.created_at)
            db.session.add(row)
        row.broker_order_id = order.broker_order_id
        row.status = order.status.value
        row.filled_qty = order.filled_qty
        row.avg_fill_price = order.avg_fill_price
        row.reason = order.reason or None
        row.updated_at = order.updated_at

    @staticmethod
    def _save_fill(fill: Fill) -> None:
        if TradingFill.query.filter_by(fill_id=fill.fill_id).first() is not None:
            return
        db.session.add(TradingFill(fill_id=fill.fill_id, client_order_id=fill.client_order_id,
                                   symbol=fill.symbol, side=fill.side.value, qty=fill.qty, price=fill.price,
                                   fee=fill.fee, fee_qty=fill.fee_qty, timestamp=fill.timestamp))
