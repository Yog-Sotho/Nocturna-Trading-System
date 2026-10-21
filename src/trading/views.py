"""
NOCTURNA Trading System - What the API shows about the live engine.

Plain JSON-safe dicts built from a service status. The embedded runtime builds
them on request; the separate engine process publishes them to the database,
where the API reads them, so both modes answer with identical documents.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .domain import Order
from .engine import EngineSnapshot
from .service import ServiceStatus


def num(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def order_view(order: Order) -> dict[str, Any]:
    r = order.request
    return {
        'client_order_id': order.client_order_id, 'broker_order_id': order.broker_order_id,
        'symbol': r.symbol, 'side': r.side.value, 'qty': num(r.qty), 'order_type': r.order_type.value,
        'time_in_force': r.time_in_force.value, 'limit_price': num(r.limit_price),
        'stop_price': num(r.stop_price), 'status': order.status.value, 'filled_qty': num(order.filled_qty),
        'avg_fill_price': num(order.avg_fill_price), 'tag': r.tag, 'reason': order.reason,
        'created_at': order.created_at.isoformat(), 'updated_at': order.updated_at.isoformat(),
    }


def status_view(configured: bool, status: ServiceStatus | None) -> dict[str, Any]:
    data: dict[str, Any] = {'configured': configured, 'state': status.state if status else 'stopped'}
    if status is None:
        return data
    data.update(
        started_at=status.started_at.isoformat() if status.started_at else None,
        last_poll=status.last_poll.isoformat() if status.last_poll else None,
        next_poll=status.next_poll.isoformat() if status.next_poll else None,
        last_error=status.last_error or None,
    )
    snapshot = status.snapshot
    if snapshot is not None:
        data.update(
            as_of=snapshot.timestamp.isoformat(), equity=num(snapshot.equity), cash=num(snapshot.cash),
            realized_pnl=num(snapshot.realized_pnl), fees=num(snapshot.fees), halted=snapshot.halted,
            halt_reason=snapshot.halt_reason or None, open_orders=len(snapshot.open_orders),
            positions=len(snapshot.positions),
        )
        risk = snapshot.risk
        data['risk'] = {
            'halt_kind': risk.halt_kind or None, 'drawdown': num(risk.drawdown),
            'high_water_mark': num(risk.high_water_mark), 'cooling_down': risk.cooling_down,
            'orders_today': risk.orders_today, 'turnover_today': num(risk.turnover_today),
            'predicted_volatility': risk.predicted_volatility,
            'volatility_scale': num(risk.volatility_scale), 'drawdown_scale': num(risk.drawdown_scale),
        }
    report = status.last_reconciliation
    if report is not None:
        data['last_reconciliation'] = {
            'at': report.timestamp.isoformat(), 'in_sync': report.in_sync,
            'discrepancies': [{'item': d.item, 'engine': num(d.engine), 'broker': num(d.broker)}
                              for d in report.discrepancies],
        }
    return data


def positions_view(snapshot: EngineSnapshot) -> dict[str, Any]:
    return {'as_of': snapshot.timestamp.isoformat(), 'positions': [
        {'symbol': p.symbol, 'qty': num(p.qty), 'avg_price': num(p.avg_price), 'price': num(p.price),
         'market_value': num(p.market_value), 'unrealized_pnl': num(p.unrealized_pnl)}
        for p in snapshot.positions]}


def engine_views(configured: bool, status: ServiceStatus | None) -> dict[str, Any]:
    """
    The whole published document: ``status`` always; ``positions``, ``targets``
    and ``working`` (open orders) only while the engine has a snapshot.
    """
    snapshot = status.snapshot if status is not None else None
    return {
        'status': status_view(configured, status),
        'positions': positions_view(snapshot) if snapshot is not None else None,
        'targets': {s: num(w) for s, w in snapshot.target_weights.items()} if snapshot is not None else None,
        'working': [order_view(o) for o in snapshot.open_orders] if snapshot is not None else [],
    }
