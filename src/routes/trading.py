"""
NOCTURNA Trading System - Trading API.

Reads come from the engine's views (live state) and from the trading history
tables; writes become commands executed on the engine thread, in this process
or in the separate engine process (ENGINE_MODE), through the same runtime
interface. Paper trading only.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request

from src.middleware.auth import require_admin, require_auth, require_trading_permissions
from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
from src.trading.operator import OperatorOrder, UnknownSymbolError
from src.trading.runtime import (
    NOT_CONFIGURED_MESSAGE,
    EngineUnavailableError,
    RemoteRuntime,
    TradingNotConfiguredError,
    TradingRuntime,
)
from src.trading.service import ServiceNotRunningError
from src.utils.audit import record_audit_event
from src.utils.validators import validate_emergency_stop, validate_manual_order

trading_bp = Blueprint('trading', __name__)
logger = logging.getLogger(__name__)

MAX_PAGE = 500


def _runtime() -> TradingRuntime | RemoteRuntime:
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


def _limit() -> int:
    return max(1, min(request.args.get('limit', 100, type=int) or 100, MAX_PAGE))


def _not_running():
    return format_response(False, error='The trading engine is not running', status_code=409)


def _unavailable():
    return format_response(False, error='The engine process is not answering', status_code=503)


# =============================================================================
# READS
# =============================================================================

@trading_bp.route('/health', methods=['GET'])
def trading_health():
    """Public: whether the engine is running (no account data)."""
    return format_response(True, data={'engine': _runtime().views()['status']['state']})


@trading_bp.route('/status', methods=['GET'])
@require_auth
def get_status():
    return format_response(True, data=_runtime().views()['status'])


@trading_bp.route('/positions', methods=['GET'])
@require_auth
def get_positions():
    positions = _runtime().views()['positions']
    if positions is None:
        return _not_running()
    return format_response(True, data=positions)


@trading_bp.route('/targets', methods=['GET'])
@require_auth
def get_targets():
    targets = _runtime().views()['targets']
    if targets is None:
        return _not_running()
    return format_response(True, data={'targets': targets})


@trading_bp.route('/orders', methods=['GET'])
@require_auth
def get_orders():
    """Working orders from the engine, and order history (newest first) from the database."""
    history = TradingOrder.query.order_by(TradingOrder.created_at.desc()).limit(_limit()).all()
    return format_response(True, data={
        'working': _runtime().views()['working'],
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

    operator_order = OperatorOrder(
        client_order_id=f"op-{uuid.uuid4().hex[:24]}", symbol=order.symbol, side=order.side, qty=order.qty,
        order_type=order.order_type, time_in_force=order.time_in_force, limit_price=order.limit_price,
        stop_price=order.stop_price, tag=f"operator:{g.user_id}")
    try:
        decision, placed = _runtime().submit_order(operator_order)
    except ServiceNotRunningError:
        return _not_running()
    except EngineUnavailableError:
        return _unavailable()
    except UnknownSymbolError:
        return format_response(False, error=f'{order.symbol} is not traded by the engine', status_code=400)
    record_audit_event('ORDER_SUBMITTED' if decision.allowed else 'ORDER_REJECTED', decision.allowed,
                       {'client_order_id': operator_order.client_order_id, 'symbol': order.symbol,
                        'side': order.side, 'qty': str(order.qty), 'reason': decision.reason})
    if not decision.allowed:
        return format_response(False, error=f'Rejected by the risk gate: {decision.reason}', status_code=422)
    return format_response(True, data={'order': placed}, status_code=201)


@trading_bp.route('/orders/<client_order_id>/cancel', methods=['POST'])
@require_auth
@require_trading_permissions
def cancel_order(client_order_id):
    try:
        requested = _runtime().cancel_order(client_order_id)
    except ServiceNotRunningError:
        return _not_running()
    except EngineUnavailableError:
        return _unavailable()
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
    try:
        _runtime().start()
    except TradingNotConfiguredError:
        return format_response(False, error=NOT_CONFIGURED_MESSAGE, status_code=503)
    except EngineUnavailableError:
        return _unavailable()
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
    try:
        _runtime().stop()
    except EngineUnavailableError:
        return _unavailable()
    record_audit_event('ENGINE_STOP', True)
    return format_response(True, message='Engine stopped')


@trading_bp.route('/control/resume', methods=['POST'])
@require_auth
@require_admin
def resume_engine():
    """
    Lift a halt so strategies trade again. After a drawdown kill the
    high-water mark restarts at the current equity.
    """
    try:
        _runtime().resume()
    except ServiceNotRunningError:
        return _not_running()
    except EngineUnavailableError:
        return _unavailable()
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
