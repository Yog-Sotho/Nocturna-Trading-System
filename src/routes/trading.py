"""
NOCTURNA Trading System - Trading API.

Reads come from the engine service's published snapshot (live state) and from
the trading history tables; writes become commands executed on the engine
thread. Paper trading only.
"""

import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request

from src.middleware.auth import require_admin, require_auth, require_trading_permissions
from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
from src.trading.domain import AssetClass, Order, OrderRequest, OrderType, Side, TimeInForce
from src.trading.runtime import NOT_CONFIGURED_MESSAGE, TradingNotConfiguredError, TradingRuntime
from src.trading.service import ServiceNotRunningError
from src.utils.audit import record_audit_event
from src.utils.validators import validate_emergency_stop, validate_manual_order

trading_bp = Blueprint('trading', __name__)
logger = logging.getLogger(__name__)

MAX_PAGE = 500


def _runtime() -> TradingRuntime:
    return current_app.extensions['trading']


def format_response(success: bool, data: Any = None, error: str | None = None,
                    message: str | None = None, status_code: int = 200) -> tuple:
    body: dict[str, Any] = {
        'success': success,
        'timestamp': datetime.now(UTC).isoformat(),
        'request_id': getattr(g, 'request_id', None),
    }
    if data is not None:
        body['data'] = data
    if error is not None:
        body['error'] = error
    if message is not None:
        body['message'] = message
    return jsonify(body), status_code


def _num(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _order(order: Order) -> dict:
    r = order.request
    return {
        'client_order_id': order.client_order_id, 'broker_order_id': order.broker_order_id,
        'symbol': r.symbol, 'side': r.side.value, 'qty': _num(r.qty), 'order_type': r.order_type.value,
        'time_in_force': r.time_in_force.value, 'limit_price': _num(r.limit_price),
        'stop_price': _num(r.stop_price), 'status': order.status.value, 'filled_qty': _num(order.filled_qty),
        'avg_fill_price': _num(order.avg_fill_price), 'tag': r.tag, 'reason': order.reason,
        'created_at': order.created_at.isoformat(), 'updated_at': order.updated_at.isoformat(),
    }


def _limit() -> int:
    return max(1, min(request.args.get('limit', 100, type=int) or 100, MAX_PAGE))


def _not_running():
    return format_response(False, error='The trading engine is not running', status_code=409)


# =============================================================================
# READS
# =============================================================================

@trading_bp.route('/health', methods=['GET'])
def trading_health():
    """Public: whether the engine is running (no account data)."""
    status = _runtime().status()
    return format_response(True, data={'engine': status.state if status else 'stopped'})


@trading_bp.route('/status', methods=['GET'])
@require_auth
def get_status():
    runtime = _runtime()
    status = runtime.status()
    data: dict[str, Any] = {'configured': runtime.configured, 'state': status.state if status else 'stopped'}
    if status is not None:
        data.update(
            started_at=status.started_at.isoformat() if status.started_at else None,
            last_poll=status.last_poll.isoformat() if status.last_poll else None,
            next_poll=status.next_poll.isoformat() if status.next_poll else None,
            last_error=status.last_error or None,
        )
        snapshot = status.snapshot
        if snapshot is not None:
            data.update(
                as_of=snapshot.timestamp.isoformat(), equity=_num(snapshot.equity), cash=_num(snapshot.cash),
                realized_pnl=_num(snapshot.realized_pnl), fees=_num(snapshot.fees), halted=snapshot.halted,
                halt_reason=snapshot.halt_reason or None, open_orders=len(snapshot.open_orders),
                positions=len(snapshot.positions),
            )
        report = status.last_reconciliation
        if report is not None:
            data['last_reconciliation'] = {
                'at': report.timestamp.isoformat(), 'in_sync': report.in_sync,
                'discrepancies': [{'item': d.item, 'engine': _num(d.engine), 'broker': _num(d.broker)}
                                  for d in report.discrepancies],
            }
    return format_response(True, data=data)


def _snapshot_or_none():
    status = _runtime().status()
    return status.snapshot if status is not None else None


@trading_bp.route('/positions', methods=['GET'])
@require_auth
def get_positions():
    snapshot = _snapshot_or_none()
    if snapshot is None:
        return _not_running()
    return format_response(True, data={'as_of': snapshot.timestamp.isoformat(), 'positions': [
        {'symbol': p.symbol, 'qty': _num(p.qty), 'avg_price': _num(p.avg_price), 'price': _num(p.price),
         'market_value': _num(p.market_value), 'unrealized_pnl': _num(p.unrealized_pnl)}
        for p in snapshot.positions]})


@trading_bp.route('/targets', methods=['GET'])
@require_auth
def get_targets():
    snapshot = _snapshot_or_none()
    if snapshot is None:
        return _not_running()
    return format_response(True, data={'targets': {s: _num(w) for s, w in snapshot.target_weights.items()}})


@trading_bp.route('/orders', methods=['GET'])
@require_auth
def get_orders():
    """Working orders from the engine, and order history (newest first) from the database."""
    snapshot = _snapshot_or_none()
    history = TradingOrder.query.order_by(TradingOrder.created_at.desc()).limit(_limit()).all()
    return format_response(True, data={
        'working': [_order(o) for o in snapshot.open_orders] if snapshot is not None else [],
        'history': [o.to_dict() for o in history],
    })


@trading_bp.route('/fills', methods=['GET'])
@require_auth
def get_fills():
    fills = TradingFill.query.order_by(TradingFill.timestamp.desc()).limit(_limit()).all()
    return format_response(True, data={'fills': [f.to_dict() for f in fills]})


@trading_bp.route('/equity-curve', methods=['GET'])
@require_auth
def get_equity_curve():
    """Recorded equity snapshots, oldest first (the most recent ``limit``)."""
    rows = EquitySnapshot.query.order_by(EquitySnapshot.timestamp.desc()).limit(_limit()).all()
    return format_response(True, data={'equity_curve': [r.to_dict() for r in reversed(rows)]})


# =============================================================================
# ORDERS
# =============================================================================

@trading_bp.route('/orders', methods=['POST'])
@require_auth
@require_trading_permissions
def submit_order():
    """An operator's order, checked by the same risk gate as strategy orders."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return format_response(False, error='Request body required', status_code=400)
    is_valid, order, errors = validate_manual_order(data)
    if not is_valid:
        return format_response(False, error='Invalid order', data={'validation_errors': errors},
                               status_code=400)

    service = _runtime().service
    engine = service.engine if service is not None and service.running else None
    if engine is None:
        return _not_running()
    instrument = engine.instruments.get(order.symbol)
    if instrument is None:
        return format_response(False, error=f'{order.symbol} is not traded by the engine', status_code=400)
    default_tif = TimeInForce.GTC if instrument.asset_class is AssetClass.CRYPTO else TimeInForce.DAY
    order_request = OrderRequest(
        client_order_id=f"op-{uuid.uuid4().hex[:24]}", symbol=order.symbol, side=Side(order.side),
        qty=order.qty, order_type=OrderType(order.order_type),
        time_in_force=TimeInForce(order.time_in_force) if order.time_in_force else default_tif,
        limit_price=order.limit_price, stop_price=order.stop_price, tag=f"operator:{g.user_id}")
    try:
        decision = service.submit_order(order_request)
    except ServiceNotRunningError:
        return _not_running()
    record_audit_event('ORDER_SUBMITTED' if decision.allowed else 'ORDER_REJECTED', decision.allowed,
                       {'client_order_id': order_request.client_order_id, 'symbol': order.symbol,
                        'side': order.side, 'qty': str(order.qty), 'reason': decision.reason})
    if not decision.allowed:
        return format_response(False, error=f'Rejected by the risk gate: {decision.reason}', status_code=422)
    placed = service.engine.orders.get(order_request.client_order_id)
    return format_response(True, data={'order': _order(placed) if placed else None}, status_code=201)


@trading_bp.route('/orders/<client_order_id>/cancel', methods=['POST'])
@require_auth
@require_trading_permissions
def cancel_order(client_order_id):
    service = _runtime().service
    if service is None or not service.running:
        return _not_running()
    try:
        requested = service.cancel_order(client_order_id)
    except ServiceNotRunningError:
        return _not_running()
    if not requested:
        return format_response(False, error='No working order with that id', status_code=404)
    record_audit_event('ORDER_CANCEL_REQUESTED', True, {'client_order_id': client_order_id})
    return format_response(True, message='Cancel requested', status_code=202)


# =============================================================================
# ENGINE CONTROL (admin)
# =============================================================================

@trading_bp.route('/control/start', methods=['POST'])
@require_auth
@require_admin
def start_engine():
    runtime = _runtime()
    try:
        runtime.start()
    except TradingNotConfiguredError:
        return format_response(False, error=NOT_CONFIGURED_MESSAGE, status_code=503)
    except RuntimeError:
        return format_response(False, error='The engine is already running', status_code=409)
    except Exception:
        logger.exception("Engine start failed")
        return format_response(False, error='Engine start failed', status_code=502)
    record_audit_event('ENGINE_START', True)
    return format_response(True, message='Engine starting', status_code=202)


@trading_bp.route('/control/stop', methods=['POST'])
@require_auth
@require_admin
def stop_engine():
    """Stop the engine thread. Working orders stay at the broker; use the emergency stop to cancel them."""
    _runtime().stop()
    record_audit_event('ENGINE_STOP', True)
    return format_response(True, message='Engine stopped')


@trading_bp.route('/control/resume', methods=['POST'])
@require_auth
@require_admin
def resume_engine():
    """Lift a halt (kill switch) so strategies trade again."""
    service = _runtime().service
    try:
        if service is None:
            raise ServiceNotRunningError
        service.resume()
    except ServiceNotRunningError:
        return _not_running()
    record_audit_event('ENGINE_RESUME', True)
    return format_response(True, message='Trading resumed')


@trading_bp.route('/emergency-stop', methods=['POST'])
@require_auth
@require_admin
@require_trading_permissions
def emergency_stop():
    """
    Kill switch: halt the engine and cancel its orders; with close_all_positions
    (the default), also close every position. Works even when the engine is not
    running, by asking the broker directly.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    is_valid, validated, errors = validate_emergency_stop(data)
    if not is_valid:
        return format_response(False, error='Invalid request data', data={'validation_errors': errors},
                               status_code=400)
    logger.critical("EMERGENCY STOP by user %s: %s (close positions: %s)",
                    getattr(g, 'user_id', '?'), validated.reason, validated.close_all_positions)
    try:
        result = _runtime().emergency_stop(validated.reason, flatten=validated.close_all_positions)
    except TradingNotConfiguredError:
        record_audit_event('EMERGENCY_STOP', False, {'reason': validated.reason}, error='not configured')
        return format_response(False, error=NOT_CONFIGURED_MESSAGE, status_code=503)
    except Exception as e:
        logger.exception("Emergency stop failed")
        record_audit_event('EMERGENCY_STOP', False, {'reason': validated.reason}, error=type(e).__name__)
        return format_response(False, error='Emergency stop failed; check the broker directly',
                               status_code=502)
    record_audit_event('EMERGENCY_STOP', True, {'reason': validated.reason,
                                                'close_all_positions': validated.close_all_positions, **result})
    return format_response(True, message='Emergency stop executed',
                           data={'reason': validated.reason, **result})
