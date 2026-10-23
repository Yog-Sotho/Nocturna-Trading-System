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
from decimal import Decimal
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request

from src.middleware.auth import require_admin, require_auth, require_trading_permissions
from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
from src.routes import schemas
from src.routes.openapi import documented
from src.trading import profile as profiles
from src.trading.catalog import catalog
from src.trading.operator import OperatorOrder, UnknownSymbolError
from src.trading.profile import RISK_FIELDS, ProfileError
from src.trading.runtime import (
    NOT_CONFIGURED_MESSAGE,
    EngineUnavailableError,
    RemoteRuntime,
    TradingNotConfiguredError,
    TradingRuntime,
)
from src.trading.service import ServiceNotRunningError
from src.utils.audit import record_audit_event
from src.utils.validators import (
    EmergencyStopSchema,
    ManualOrderSchema,
    validate_emergency_stop,
    validate_manual_order,
)

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
@documented("Whether the engine runs", tag="trading", access="public",
            responses={200: ("The engine's state, no account data", schemas.TradingHealth)})
def trading_health():
    """Public: whether the engine is running (no account data)."""
    return format_response(True, data={'engine': _runtime().views()['status']['state']})


@trading_bp.route('/status', methods=['GET'])
@documented("The engine's state, account and risk", tag="trading", access="user",
            responses={200: ("Status", schemas.Status)})
@require_auth
def get_status():
    return format_response(True, data=_runtime().views()['status'])


@trading_bp.route('/positions', methods=['GET'])
@documented("Positions", tag="trading", access="user",
            responses={200: ("Positions", schemas.Positions), 409: "The engine is not running"})
@require_auth
def get_positions():
    positions = _runtime().views()['positions']
    if positions is None:
        return _not_running()
    return format_response(True, data=positions)


@trading_bp.route('/targets', methods=['GET'])
@documented("Target weights", tag="trading", access="user",
            responses={200: ("Targets", schemas.Targets), 409: "The engine is not running"})
@require_auth
def get_targets():
    targets = _runtime().views()['targets']
    if targets is None:
        return _not_running()
    return format_response(True, data={'targets': targets})


@trading_bp.route('/orders', methods=['GET'])
@documented("Working orders and order history", tag="trading", access="user",
            query={"limit": ("integer", "History rows, at most 500 (default 100)")},
            responses={200: ("Orders", schemas.Orders)})
@require_auth
def get_orders():
    """Working orders from the engine, and order history (newest first) from the database."""
    history = TradingOrder.query.order_by(TradingOrder.created_at.desc()).limit(_limit()).all()
    return format_response(True, data={
        'working': _runtime().views()['working'],
        'history': [o.to_dict() for o in history],
    })


@trading_bp.route('/fills', methods=['GET'])
@documented("Fills, newest first", tag="trading", access="user",
            query={"limit": ("integer", "At most 500 (default 100)")},
            responses={200: ("Fills", schemas.Fills)})
@require_auth
def get_fills():
    fills = TradingFill.query.order_by(TradingFill.timestamp.desc()).limit(_limit()).all()
    return format_response(True, data={'fills': [f.to_dict() for f in fills]})


@trading_bp.route('/equity-curve', methods=['GET'])
@documented("Recorded equity", tag="trading", access="user",
            query={"limit": ("integer", "The most recent snapshots, at most 500 (default 100)")},
            responses={200: ("Equity snapshots, oldest first", schemas.EquityCurve)})
@require_auth
def get_equity_curve():
    """Recorded equity snapshots, oldest first (the most recent ``limit``)."""
    rows = EquitySnapshot.query.order_by(EquitySnapshot.timestamp.desc()).limit(_limit()).all()
    return format_response(True, data={'equity_curve': [r.to_dict() for r in reversed(rows)]})


# =============================================================================
# ORDERS
# =============================================================================

@trading_bp.route('/orders', methods=['POST'])
@documented("Place an order", tag="trading", access="trader", body=ManualOrderSchema,
            description="Checked by the same risk gate as the strategies' orders.",
            responses={201: ("Sent; the order as the broker first answered", schemas.PlacedOrder),
                       400: ("Invalid order, or a symbol the engine does not trade", schemas.ValidationErrors,
                             False),
                       409: "The engine is not running", 422: "Rejected by the risk gate",
                       503: "The engine process is not answering"})
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
@documented("Cancel a working order", tag="trading", access="trader", responses={
    202: "Cancel requested", 404: "No working order with that id", 409: "The engine is not running",
    503: "The engine process is not answering"})
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
@documented("Start the engine", tag="control", access="admin", responses={
    202: "Starting", 409: "Already running", 502: "Start-up failed",
    503: "Alpaca keys are not set, or the engine process is not answering"})
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


@trading_bp.route('/control/restart', methods=['POST'])
@documented("Restart the engine with the newest saved profile", tag="control", access="admin",
            description="Stops the engine and starts it again; a profile saved while it ran takes effect.",
            responses={202: "Restarting", 502: "Start-up failed",
                       503: "Alpaca keys are not set, or the engine process is not answering"})
@require_auth
@require_admin
def restart_engine():
    try:
        _runtime().restart()
    except TradingNotConfiguredError:
        return format_response(False, error=NOT_CONFIGURED_MESSAGE, status_code=503)
    except EngineUnavailableError:
        return _unavailable()
    except Exception:
        logger.exception("Engine restart failed")
        return format_response(False, error='Engine restart failed', status_code=502)
    record_audit_event('ENGINE_RESTART', True)
    return format_response(True, message='Engine restarting', status_code=202)


@trading_bp.route('/control/stop', methods=['POST'])
@documented("Stop the engine", tag="control", access="admin",
            description="Working orders stay at the broker; the emergency stop cancels them.",
            responses={200: "Stopped", 503: "The engine process is not answering"})
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
@documented("Lift a halt", tag="control", access="admin",
            description="After a drawdown kill the high-water mark restarts at the current equity.",
            responses={200: "Resumed", 409: "The engine is not running",
                       503: "The engine process is not answering"})
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
@documented("Emergency stop", tag="control", access="admin", body=EmergencyStopSchema,
            body_required=False, rate_limited=False,
            description="Halts the engine and cancels its orders; with close_all_positions (the default) "
                        "also closes every position. Without an engine that answers, asks Alpaca directly.",
            responses={200: ("Done", schemas.EmergencyStopResult), 400: ("Invalid body", schemas.ValidationErrors,
                                                                       False),
                       502: "Failed: check the broker directly", 503: "Alpaca keys are not set"})
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


# =============================================================================
# STRATEGIES AND THE TRADING PROFILE
# =============================================================================

def _setting(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


def _risk_settings(saved: profiles.Profile) -> list[dict[str, Any]]:
    settings = current_app.extensions['settings']
    try:
        effective = profiles.effective_settings(settings, saved)
    except ProfileError:  # the environment changed under a saved profile: show what it now gives
        effective = settings.model_copy(update=dict(saved.risk))
    return [{
        'name': name, 'label': spec.label, 'kind': spec.kind, 'minimum': _setting(spec.minimum),
        'maximum': _setting(spec.maximum), 'off': _setting(spec.off), 'optional': spec.optional,
        'value': _setting(getattr(effective, name)), 'source': 'profile' if name in saved.risk else 'environment',
        'environment': _setting(getattr(settings, name)),
    } for name, spec in RISK_FIELDS.items()]


def _running_profile() -> dict[str, Any] | None:
    views = _runtime().views()
    if views['status']['state'] not in ('starting', 'running'):
        return None
    profile: dict[str, Any] | None = views.get('profile')
    return profile


@trading_bp.route('/strategies', methods=['GET'])
@documented("The strategies the engine can trade", tag="strategies", access="user",
            description="Every registered strategy with its parameters (its validation grid) and what "
                        "validation concluded about it.",
            responses={200: ("The strategies", schemas.Strategies)})
@require_auth
def list_strategies():
    return format_response(True, data={'strategies': catalog()})


@trading_bp.route('/profile', methods=['GET'])
@documented("The trading profile: the strategy mix and the risk limits", tag="strategies", access="user",
            responses={200: ("The saved profile, and the one the engine runs", schemas.ProfileState)})
@require_auth
def get_profile():
    saved = profiles.latest(current_app._get_current_object())
    running = _running_profile()
    return format_response(True, data={
        'saved': saved.to_dict(), 'running': running,
        'pending': running is not None and running['id'] != saved.id, 'risk': _risk_settings(saved)})


@trading_bp.route('/profile', methods=['PUT'])
@documented("Save the trading profile", tag="strategies", access="admin", body=schemas.ProfileBody,
            description="Saves a new version. The engine trades it from its next start; POST /control/restart "
                        "applies it now. Positions in symbols the new mix does not trade are sold. Paper only.",
            responses={201: ("Saved", schemas.ProfileSaved),
                       400: "The engine could not trade it; the error says why"})
@require_auth
@require_admin
def save_profile():
    try:
        parsed = profiles.parse(request.get_json(silent=True))
        profiles.effective_settings(current_app.extensions['settings'], parsed)
    except ProfileError as e:
        return format_response(False, error=str(e), status_code=400)
    saved = profiles.save(current_app._get_current_object(), parsed, created_by=getattr(g, 'username', None),
                          now=datetime.now(UTC))
    record_audit_event('PROFILE_SAVED', True, {
        'profile_id': saved.id, 'mix': [e.to_dict() for e in saved.mix], 'risk': saved.to_dict()['risk']})
    running = _running_profile()
    return format_response(True, data={'profile': saved.to_dict(), 'restart_needed': running is not None},
                           message='Saved: the engine trades it from its next start', status_code=201)
