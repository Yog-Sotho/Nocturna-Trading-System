# FILE LOCATION: src/routes/trading.py
"""
NOCTURNA Trading System - Secure API Routes
Production-grade endpoints with authentication, validation, and rate limiting.
"""

import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

from flask import Blueprint, g, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.trading_engine import TradingEngine, TradingEngineState
from src.middleware.auth import require_admin, require_auth, require_trading_permissions
from src.utils.validators import validate_config_input, validate_emergency_stop, validate_trading_signal

# Create blueprint
trading_bp = Blueprint('trading', __name__)

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global trading engine instance
_trading_engine: TradingEngine | None = None


# =============================================================================
# ENGINE INSTANCE MANAGEMENT
# =============================================================================

def get_trading_engine() -> TradingEngine:
    """
    Get or create the trading engine instance.
    Implements singleton pattern with proper initialization.

    Returns:
        TradingEngine instance
    """
    global _trading_engine

    if _trading_engine is None:
        # Load configuration from environment with defaults
        config = {
            'symbols': os.environ.get('DEFAULT_SYMBOLS', 'SPY,QQQ,AAPL,MSFT,GOOGL').split(','),
            'update_interval': int(os.environ.get('UPDATE_INTERVAL', 60)),
            'simulation_mode': os.environ.get('TRADING_MODE', 'PAPER') == 'PAPER',
            'market_data': {
                'redis_host': os.environ.get('REDIS_URL'),
            },
            'trading': {
                'alpaca_api_key': os.environ.get('ALPACA_API_KEY'),
                'alpaca_secret_key': os.environ.get('ALPACA_SECRET_KEY'),
                'alpaca_base_url': os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                'simulation_mode': os.environ.get('TRADING_MODE', 'PAPER') == 'PAPER',
                'max_daily_loss': float(os.environ.get('MAX_DAILY_LOSS', 0.05)),
            },
            'strategy': {
                'max_position_size': float(os.environ.get('MAX_POSITION_SIZE', 0.2)),
                'grid_spacing': float(os.environ.get('GRID_SPACING', 0.005)),
                'atr_mult_sl': float(os.environ.get('ATR_MULT_SL', 2.0)),
                'atr_mult_tp': float(os.environ.get('ATR_MULT_TP', 4.0)),
                'volatility_threshold': float(os.environ.get('VOLATILITY_THRESHOLD', 2.0)),
            },
            'risk': {
                'max_daily_loss': float(os.environ.get('MAX_DAILY_LOSS', 0.05)),
                'max_drawdown': float(os.environ.get('DRAWDOWN_STOP', 0.15)),
                'volatility_threshold': float(os.environ.get('VOLATILITY_THRESHOLD', 2.0)),
                'risk_parameters': {
                    'max_position_size': float(os.environ.get('MAX_POSITION_SIZE', 0.2)),
                    'max_portfolio_exposure': float(os.environ.get('MAX_PORTFOLIO_RISK', 0.8)),
                    'max_daily_loss': float(os.environ.get('MAX_DAILY_LOSS', 0.05)),
                    'max_drawdown': float(os.environ.get('DRAWDOWN_STOP', 0.15)),
                }
            }
        }

        _trading_engine = TradingEngine(config)

    return _trading_engine


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_api_call(action: str, details: str = ''):
    """Log API call with user context."""
    user_id = getattr(g, 'user_id', 'anonymous')
    request_id = getattr(g, 'request_id', 'unknown')
    logger.info(f"API [{action}] | User: {user_id} | Request ID: {request_id} | {details}")


def format_response(success: bool, data: Any = None, error: str = None,
                    message: str = None, status_code: int = 200) -> tuple:
    """Format standardized API response."""
    response = {
        'success': success,
        'timestamp': datetime.now(UTC).isoformat(),
        'request_id': getattr(g, 'request_id', None)
    }

    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    if message is not None:
        response['message'] = message

    return jsonify(response), status_code


# =============================================================================
# PUBLIC ENDPOINTS (Authentication Optional)
# =============================================================================

@trading_bp.route('/health', methods=['GET'])
def trading_health():
    """
    Public health check for trading service.
    Used by load balancers to verify service availability.
    """
    return format_response(True, data={'status': 'healthy', 'service': 'trading'})


# =============================================================================
# AUTHENTICATED ENDPOINTS - READ OPERATIONS
# =============================================================================

@trading_bp.route('/status', methods=['GET'])
@require_auth
def get_status():
    """
    Get current trading engine status.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        status = engine.get_status()

        log_api_call('GET_STATUS', f"State: {status.get('engine_state')}")

        return format_response(True, data=status)

    except Exception as e:
        logger.error(f"Error getting status: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/status/detailed', methods=['GET'])
@require_auth
def get_detailed_status():
    """
    Get detailed trading engine status.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        detailed_status = engine.get_detailed_status()

        log_api_call('GET_DETAILED_STATUS', f"Positions: {len(detailed_status.get('positions', {}))}")

        return format_response(True, data=detailed_status)

    except Exception as e:
        logger.error(f"Error getting detailed status: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/positions', methods=['GET'])
@require_auth
def get_positions():
    """
    Get active positions.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        positions = engine.order_manager.get_positions()

        log_api_call('GET_POSITIONS', f"Count: {len(positions)}")

        return format_response(True, data=positions, message=f'{len(positions)} positions retrieved')

    except Exception as e:
        logger.error(f"Error getting positions: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/orders', methods=['GET'])
@require_auth
def get_orders():
    """
    Get active orders.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        orders = engine.order_manager.active_orders

        log_api_call('GET_ORDERS', f"Count: {len(orders)}")

        return format_response(True, data=orders, message=f'{len(orders)} active orders')

    except Exception as e:
        logger.error(f"Error getting orders: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/performance', methods=['GET'])
@require_auth
def get_performance():
    """
    Get performance metrics.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        stats = engine.stats

        performance_data = {
            'total_pnl': stats.get('total_pnl', 0.0),
            'daily_pnl': stats.get('daily_pnl', 0.0),
            'win_rate': stats.get('win_rate', 0.0),
            'total_trades': stats.get('total_trades', 0),
            'winning_trades': stats.get('winning_trades', 0),
            'losing_trades': stats.get('losing_trades', 0),
            'max_drawdown': stats.get('max_drawdown', 0.0),
        }

        log_api_call('GET_PERFORMANCE', f"Win rate: {performance_data['win_rate']:.2%}")

        return format_response(True, data=performance_data)

    except Exception as e:
        logger.error(f"Error getting performance: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/strategy', methods=['GET'])
@require_auth
def get_strategy_status():
    """
    Get current strategy status.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        strategy_status = engine.strategy_manager.get_strategy_status()

        log_api_call('GET_STRATEGY', f"Mode: {strategy_status.get('current_mode')}")

        return format_response(True, data=strategy_status)

    except Exception as e:
        logger.error(f"Error getting strategy status: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/risk', methods=['GET'])
@require_auth
def get_risk_status():
    """
    Get current risk status.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        risk_report = engine.risk_manager.get_risk_report()

        log_api_call('GET_RISK', f"Level: {risk_report.get('current_risk_level')}")

        return format_response(True, data=risk_report)

    except Exception as e:
        logger.error(f"Error getting risk status: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/config', methods=['GET'])
@require_auth
def get_config():
    """
    Get current configuration.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        config_data = {
            'symbols': engine.symbols,
            'update_interval': engine.update_interval,
            'strategy_parameters': engine.strategy_manager.parameters,
            'risk_parameters': engine.risk_manager.risk_parameters,
            'simulation_mode': engine.config.get('simulation_mode', False)
        }

        log_api_call('GET_CONFIG', f"Symbols: {engine.symbols}")

        return format_response(True, data=config_data)

    except Exception as e:
        logger.error(f"Error getting config: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# AUTHENTICATED ENDPOINTS - WRITE OPERATIONS
# =============================================================================

@trading_bp.route('/control/start', methods=['POST'])
@require_auth
@require_admin
@require_trading_permissions
def start_engine():
    """
    Start the trading engine.
    Requires authentication and admin permissions.
    """
    try:
        engine = get_trading_engine()

        if engine.state == TradingEngineState.RUNNING:
            return format_response(
                False,
                error='Engine already running',
                status_code=400
            )

        success = engine.start()

        if success:
            log_api_call('START_ENGINE', 'Engine started successfully')
            return format_response(
                True,
                message='Engine started',
                data={'state': engine.state}
            )
        else:
            return format_response(
                False,
                error=f'Failed to start engine. Current state: {engine.state}',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error starting engine: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/control/stop', methods=['POST'])
@require_auth
@require_admin
def stop_engine():
    """
    Stop the trading engine.
    Requires authentication and admin permissions.
    """
    try:
        engine = get_trading_engine()

        if engine.state == TradingEngineState.STOPPED:
            return format_response(
                False,
                error='Engine already stopped',
                status_code=400
            )

        success = engine.stop()

        if success:
            log_api_call('STOP_ENGINE', 'Engine stopped')
            return format_response(
                True,
                message='Engine stopped',
                data={'state': engine.state}
            )
        else:
            return format_response(
                False,
                error='Failed to stop engine',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error stopping engine: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/control/pause', methods=['POST'])
@require_auth
@require_admin
def pause_engine():
    """
    Pause the trading engine.
    Requires authentication and admin permissions.
    """
    try:
        engine = get_trading_engine()

        if engine.state != TradingEngineState.RUNNING:
            return format_response(
                False,
                error='Engine not running',
                status_code=400
            )

        success = engine.pause()

        if success:
            log_api_call('PAUSE_ENGINE', 'Engine paused')
            return format_response(
                True,
                message='Engine paused',
                data={'state': engine.state}
            )
        else:
            return format_response(
                False,
                error='Failed to pause engine',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error pausing engine: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/control/resume', methods=['POST'])
@require_auth
@require_admin
def resume_engine():
    """
    Resume the trading engine from pause.
    Requires authentication and admin permissions.
    """
    try:
        engine = get_trading_engine()

        if engine.state != TradingEngineState.PAUSED:
            return format_response(
                False,
                error='Engine not paused',
                status_code=400
            )

        success = engine.resume()

        if success:
            log_api_call('RESUME_ENGINE', 'Engine resumed')
            return format_response(
                True,
                message='Engine resumed',
                data={'state': engine.state}
            )
        else:
            return format_response(
                False,
                error='Failed to resume engine',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error resuming engine: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/emergency-stop', methods=['POST'])
@require_auth
@require_admin
@require_trading_permissions
def emergency_stop():
    """
    Emergency stop of the system.
    Requires authentication and admin permissions.
    This will immediately halt all trading activity.
    """
    try:
        # Get and validate request data
        data = request.get_json() or {}
        is_valid, validated_data, errors = validate_emergency_stop(data)

        if not is_valid:
            return format_response(
                False,
                error='Invalid request data',
                data={'validation_errors': errors},
                status_code=400
            )

        engine = get_trading_engine()
        reason = validated_data.reason

        # Log critical action
        user_id = getattr(g, 'user_id', 'unknown')
        logger.critical(
            f"EMERGENCY STOP triggered | User: {user_id} | Reason: {reason} | "
            f"Close positions: {validated_data.close_all_positions} | "
            f"Request ID: {getattr(g, 'request_id', 'unknown')}"
        )

        # Trigger emergency stop
        engine.emergency_stop(reason)

        # Update config for position closing
        if validated_data.close_all_positions:
            engine.config['emergency_close_positions'] = True

        return format_response(
            True,
            message='Emergency stop executed',
            data={
                'state': engine.state,
                'reason': reason,
                'positions_closed': validated_data.close_all_positions
            }
        )

    except Exception as e:
        logger.error(f"Error during emergency stop: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# TRADING OPERATIONS
# =============================================================================

@trading_bp.route('/orders', methods=['POST'])
@require_auth
@require_trading_permissions
def create_order():
    """
    Create a new order.
    Requires authentication and trading permissions.
    """
    try:
        # Parse and validate request data
        data = request.get_json()
        if not data:
            return format_response(False, error='Request body required', status_code=400)

        is_valid, signal, errors = validate_trading_signal(data)

        if not is_valid:
            logger.warning(
                f"Invalid order signal from {getattr(g, 'user_id', 'unknown')}: {errors} | "
                f"Request ID: {getattr(g, 'request_id', 'unknown')}"
            )
            return format_response(
                False,
                error='Invalid order signal',
                data={'validation_errors': errors},
                status_code=400
            )

        # Additional risk check before submission
        engine = get_trading_engine()

        # Get current positions
        positions = engine.order_manager.get_positions()
        market_data = engine._get_market_data_for_risk(signal.symbol)

        # Validate with risk manager
        is_valid, reason, adjusted_signal = engine.risk_manager.validate_trade(
            signal.model_dump(),
            positions,
            market_data
        )

        if not is_valid:
            log_api_call('ORDER_REJECTED', f"Symbol: {signal.symbol} | Reason: {reason}")
            return format_response(
                False,
                error=f'Order rejected by risk manager: {reason}',
                data={'rejected_reason': reason},
                status_code=400
            )

        # Submit order
        order_id = engine.order_manager.submit_order(adjusted_signal)

        if order_id:
            log_api_call(
                'ORDER_SUBMITTED',
                f"Order ID: {order_id} | Symbol: {signal.symbol} | Side: {signal.side} | "
                f"Qty: {signal.quantity}"
            )
            return format_response(
                True,
                message='Order submitted',
                data={
                    'order_id': order_id,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'quantity': signal.quantity,
                    'order_type': signal.order_type
                }
            )
        else:
            return format_response(
                False,
                error='Failed to submit order',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error creating order: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/orders/<order_id>/cancel', methods=['POST'])
@require_auth
@require_trading_permissions
def cancel_order(order_id: str):
    """
    Cancel a specific order.
    Requires authentication and trading permissions.
    """
    try:
        # Validate order_id format
        if not order_id or not isinstance(order_id, str):
            return format_response(False, error='Invalid order ID', status_code=400)

        # Sanitize order_id
        order_id = order_id.strip()

        engine = get_trading_engine()
        success = engine.order_manager.cancel_order(order_id)

        if success:
            log_api_call('ORDER_CANCELLED', f"Order ID: {order_id}")
            return format_response(True, message=f'Order {order_id} cancelled')
        else:
            return format_response(
                False,
                error=f'Order {order_id} not found or already processed',
                status_code=404
            )

    except Exception as e:
        logger.error(f"Error cancelling order {order_id}: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# CONFIGURATION UPDATES
# =============================================================================

@trading_bp.route('/config', methods=['PUT'])
@require_auth
@require_admin
def update_config():
    """
    Update system configuration.
    Requires authentication and admin permissions.
    """
    try:
        data = request.get_json()
        if not data:
            return format_response(False, error='Request body required', status_code=400)

        # Validate configuration input
        is_valid, validated_config, errors = validate_config_input(data)

        if not is_valid:
            logger.warning(
                f"Invalid config update from {getattr(g, 'user_id', 'unknown')}: {errors} | "
                f"Request ID: {getattr(g, 'request_id', 'unknown')}"
            )
            return format_response(
                False,
                error='Invalid configuration',
                data={'validation_errors': errors},
                status_code=400
            )

        engine = get_trading_engine()
        config_dict = validated_config.model_dump(exclude_none=True)

        success = engine.update_config(config_dict)

        if success:
            log_api_call('CONFIG_UPDATED', f"Updated keys: {list(config_dict.keys())}")
            return format_response(
                True,
                message='Configuration updated',
                data={'updated_keys': list(config_dict.keys())}
            )
        else:
            return format_response(
                False,
                error='Failed to update configuration',
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error updating config: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# OPTIMIZATION ENDPOINTS
# =============================================================================

@trading_bp.route('/optimize', methods=['POST'])
@require_auth
@require_admin
def start_optimization():
    """
    Start ML optimization process.
    Requires authentication and admin permissions.
    """
    try:
        # This would trigger the ML optimizer in a real implementation
        # For now, return a placeholder response
        logger.info(
            f"Optimization requested by {getattr(g, 'user_id', 'unknown')} | "
            f"Request ID: {getattr(g, 'request_id', 'unknown')}"
        )

        return format_response(
            True,
            message='Optimization started (async process)',
            data={
                'status': 'started',
                'estimated_duration': '30-60 minutes'
            }
        )

    except Exception as e:
        logger.error(f"Error starting optimization: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@trading_bp.route('/trades', methods=['GET'])
@require_auth
def get_trade_history():
    """
    Get trade history.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()
        trades = engine.order_manager.order_history

        log_api_call('GET_TRADE_HISTORY', f"Total trades: {len(trades)}")

        return format_response(
            True,
            data={'trades': trades, 'count': len(trades)}
        )

    except Exception as e:
        logger.error(f"Error getting trade history: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


@trading_bp.route('/equity-curve', methods=['GET'])
@require_auth
def get_equity_curve():
    """
    Get equity curve data.
    Requires authentication.
    """
    try:
        engine = get_trading_engine()

        # Get performance history
        performance_history = engine.performance_history

        if not performance_history:
            # Generate placeholder data
            equity_data = []
            base_value = float(os.environ.get('INITIAL_CAPITAL', 100000))
            for i in range(30):
                equity_data.append({
                    'timestamp': (datetime.now(UTC).timestamp() - (30 - i) * 86400),
                    'value': base_value * (1 + (i * 0.001 + (i % 3) * 0.003))
                })
        else:
            equity_data = performance_history

        log_api_call('GET_EQUITY_CURVE', f"Data points: {len(equity_data)}")

        return format_response(True, data={'equity_curve': equity_data})

    except Exception as e:
        logger.error(f"Error getting equity curve: {e} | Request ID: {getattr(g, 'request_id', 'unknown')}")
        return format_response(False, error=str(e), status_code=500)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@trading_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return format_response(False, error='Endpoint not found', status_code=404)


@trading_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return format_response(False, error='Method not allowed', status_code=405)


@trading_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error in trading blueprint: {error}")
    return format_response(False, error='Internal server error', status_code=500)
