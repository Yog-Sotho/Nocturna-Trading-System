"""
API Routes per NOCTURNA v2.0 Trading Bot
Fornisce endpoint REST per il controllo e monitoraggio del bot.
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Import dei moduli core
from src.core.trading_engine import TradingEngine, TradingEngineState
from src.core.market_data import MarketDataHandler
from src.core.strategy_manager import StrategyManager, TradingMode, MarketState
from src.core.order_manager import OrderExecutionManager
from src.core.risk_manager import RiskManager

# Blueprint per le routes di trading
trading_bp = Blueprint('trading', __name__)

# Configurazione logging
logger = logging.getLogger(__name__)

# Istanza globale del trading engine (in produzione usare dependency injection)
trading_engine = None

def get_trading_engine():
    """Restituisce l'istanza del trading engine."""
    global trading_engine
    if trading_engine is None:
        # Configurazione di default per demo
        config = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'update_interval': 60,
            'simulation_mode': True,  # Modalità simulazione per demo
            'market_data': {
                'redis_host': None,  # Disabilitato per demo
            },
            'trading': {
                'alpaca_api_key': None,  # Disabilitato per demo
                'simulation_mode': True
            },
            'strategy': {
                'max_position_size': 0.2,
                'grid_spacing': 0.005,
                'atr_mult_sl': 2.0
            },
            'risk': {
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'volatility_threshold': 2.0
            }
        }
        trading_engine = TradingEngine(config)
    
    return trading_engine

@trading_bp.route('/status', methods=['GET'])
@cross_origin()
def get_status():
    """Restituisce lo stato corrente del trading engine."""
    try:
        engine = get_trading_engine()
        status = engine.get_status()
        
        return jsonify({
            'success': True,
            'data': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero stato: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/status/detailed', methods=['GET'])
@cross_origin()
def get_detailed_status():
    """Restituisce lo stato dettagliato del sistema."""
    try:
        engine = get_trading_engine()
        detailed_status = engine.get_detailed_status()
        
        return jsonify({
            'success': True,
            'data': detailed_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero stato dettagliato: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/control/start', methods=['POST'])
@cross_origin()
def start_engine():
    """Avvia il trading engine."""
    try:
        engine = get_trading_engine()
        
        if engine.state == TradingEngineState.RUNNING:
            return jsonify({
                'success': False,
                'error': 'Engine già in esecuzione',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        success = engine.start()
        
        return jsonify({
            'success': success,
            'message': 'Engine avviato' if success else 'Errore nell\'avvio',
            'state': engine.state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nell'avvio engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/control/stop', methods=['POST'])
@cross_origin()
def stop_engine():
    """Ferma il trading engine."""
    try:
        engine = get_trading_engine()
        
        if engine.state == TradingEngineState.STOPPED:
            return jsonify({
                'success': False,
                'error': 'Engine già fermato',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        success = engine.stop()
        
        return jsonify({
            'success': success,
            'message': 'Engine fermato' if success else 'Errore nell\'arresto',
            'state': engine.state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nell'arresto engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/control/pause', methods=['POST'])
@cross_origin()
def pause_engine():
    """Mette in pausa il trading engine."""
    try:
        engine = get_trading_engine()
        
        if engine.state != TradingEngineState.RUNNING:
            return jsonify({
                'success': False,
                'error': 'Engine non in esecuzione',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        success = engine.pause()
        
        return jsonify({
            'success': success,
            'message': 'Engine in pausa' if success else 'Errore nella pausa',
            'state': engine.state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nella pausa engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/control/resume', methods=['POST'])
@cross_origin()
def resume_engine():
    """Riprende il trading engine dalla pausa."""
    try:
        engine = get_trading_engine()
        
        if engine.state != TradingEngineState.PAUSED:
            return jsonify({
                'success': False,
                'error': 'Engine non in pausa',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        success = engine.resume()
        
        return jsonify({
            'success': success,
            'message': 'Engine ripreso' if success else 'Errore nella ripresa',
            'state': engine.state,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nella ripresa engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/positions', methods=['GET'])
@cross_origin()
def get_positions():
    """Restituisce le posizioni attive."""
    try:
        engine = get_trading_engine()
        positions = engine.order_manager.get_positions()
        
        # Simula alcune posizioni per demo
        if not positions:
            positions = {
                'AAPL': {
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'side': 'long',
                    'avg_price': 150.25,
                    'market_value': 15025.0,
                    'unrealized_pnl': 182.23,
                    'last_update': datetime.now().isoformat()
                },
                'MSFT': {
                    'symbol': 'MSFT',
                    'quantity': -50,
                    'side': 'short',
                    'avg_price': 280.50,
                    'market_value': -14025.0,
                    'unrealized_pnl': -287.46,
                    'last_update': datetime.now().isoformat()
                },
                'GOOGL': {
                    'symbol': 'GOOGL',
                    'quantity': 25,
                    'side': 'long',
                    'avg_price': 2650.75,
                    'market_value': 66268.75,
                    'unrealized_pnl': -181.50,
                    'last_update': datetime.now().isoformat()
                },
                'TSLA': {
                    'symbol': 'TSLA',
                    'quantity': 75,
                    'side': 'long',
                    'avg_price': 245.80,
                    'market_value': 18435.0,
                    'unrealized_pnl': 409.07,
                    'last_update': datetime.now().isoformat()
                },
                'NVDA': {
                    'symbol': 'NVDA',
                    'quantity': 30,
                    'side': 'long',
                    'avg_price': 485.20,
                    'market_value': 14556.0,
                    'unrealized_pnl': 125.93,
                    'last_update': datetime.now().isoformat()
                }
            }
        
        return jsonify({
            'success': True,
            'data': positions,
            'count': len(positions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero posizioni: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/orders', methods=['GET'])
@cross_origin()
def get_orders():
    """Restituisce gli ordini attivi."""
    try:
        engine = get_trading_engine()
        orders = engine.order_manager.active_orders
        
        # Simula alcuni ordini per demo
        if not orders:
            orders = {
                'order_1': {
                    'id': 'order_1',
                    'symbol': 'AAPL',
                    'side': 'buy',
                    'type': 'limit',
                    'quantity': 50,
                    'price': 152.21,
                    'status': 'PENDING',
                    'submitted_at': datetime.now().isoformat()
                },
                'order_2': {
                    'id': 'order_2',
                    'symbol': 'MSFT',
                    'side': 'sell',
                    'type': 'limit',
                    'quantity': 25,
                    'price': 232.88,
                    'status': 'PENDING',
                    'submitted_at': datetime.now().isoformat()
                },
                'order_3': {
                    'id': 'order_3',
                    'symbol': 'GOOGL',
                    'side': 'buy',
                    'type': 'limit',
                    'quantity': 10,
                    'price': 2670.11,
                    'status': 'PENDING',
                    'submitted_at': datetime.now().isoformat()
                }
            }
        
        return jsonify({
            'success': True,
            'data': orders,
            'count': len(orders),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero ordini: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/performance', methods=['GET'])
@cross_origin()
def get_performance():
    """Restituisce le metriche di performance."""
    try:
        engine = get_trading_engine()
        
        # Simula dati di performance per demo
        performance_data = []
        base_value = 10000
        
        for i in range(8):
            hour = 9 + i
            value = base_value + (i * 50) + (i % 3 * 30)
            pnl = value - base_value
            
            performance_data.append({
                'time': f"{hour:02d}:00",
                'value': value,
                'pnl': pnl
            })
        
        metrics = {
            'total_pnl': 2847.32,
            'daily_pnl': 156.78,
            'win_rate': 68.5,
            'total_trades': engine.stats.get('total_trades', 0),
            'winning_trades': engine.stats.get('winning_trades', 0),
            'losing_trades': engine.stats.get('losing_trades', 0),
            'max_drawdown': engine.stats.get('max_drawdown', 0.0),
            'performance_data': performance_data
        }
        
        return jsonify({
            'success': True,
            'data': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero performance: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/strategy', methods=['GET'])
@cross_origin()
def get_strategy_status():
    """Restituisce lo stato della strategia."""
    try:
        engine = get_trading_engine()
        strategy_status = engine.strategy_manager.get_strategy_status()
        
        return jsonify({
            'success': True,
            'data': strategy_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero stato strategia: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/risk', methods=['GET'])
@cross_origin()
def get_risk_status():
    """Restituisce lo stato del rischio."""
    try:
        engine = get_trading_engine()
        risk_report = engine.risk_manager.get_risk_report()
        
        return jsonify({
            'success': True,
            'data': risk_report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero stato rischio: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/config', methods=['GET'])
@cross_origin()
def get_config():
    """Restituisce la configurazione corrente."""
    try:
        engine = get_trading_engine()
        
        config_data = {
            'symbols': engine.symbols,
            'update_interval': engine.update_interval,
            'strategy_parameters': engine.strategy_manager.parameters,
            'risk_parameters': engine.risk_manager.risk_parameters,
            'simulation_mode': engine.config.get('simulation_mode', False)
        }
        
        return jsonify({
            'success': True,
            'data': config_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel recupero configurazione: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/config', methods=['POST'])
@cross_origin()
def update_config():
    """Aggiorna la configurazione del sistema."""
    try:
        engine = get_trading_engine()
        new_config = request.get_json()
        
        if not new_config:
            return jsonify({
                'success': False,
                'error': 'Configurazione non fornita',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        success = engine.update_config(new_config)
        
        return jsonify({
            'success': success,
            'message': 'Configurazione aggiornata' if success else 'Errore nell\'aggiornamento',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nell'aggiornamento configurazione: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/orders/<order_id>/cancel', methods=['POST'])
@cross_origin()
def cancel_order(order_id):
    """Cancella un ordine specifico."""
    try:
        engine = get_trading_engine()
        success = engine.order_manager.cancel_order(order_id)
        
        return jsonify({
            'success': success,
            'message': f'Ordine {order_id} cancellato' if success else 'Errore nella cancellazione',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nella cancellazione ordine {order_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/emergency-stop', methods=['POST'])
@cross_origin()
def emergency_stop():
    """Arresto di emergenza del sistema."""
    try:
        engine = get_trading_engine()
        data = request.get_json() or {}
        reason = data.get('reason', 'Emergency stop via API')
        
        engine.emergency_stop(reason)
        
        return jsonify({
            'success': True,
            'message': 'Arresto di emergenza eseguito',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nell'arresto di emergenza: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@trading_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check del sistema."""
    try:
        engine = get_trading_engine()
        
        health_data = {
            'status': 'healthy',
            'engine_state': engine.state,
            'uptime': engine.stats.get('uptime', 0),
            'last_update': engine.last_update.isoformat() if engine.last_update else None,
            'components': {
                'market_data': 'healthy',
                'strategy_manager': 'healthy',
                'order_manager': 'healthy',
                'risk_manager': 'healthy'
            }
        }
        
        return jsonify({
            'success': True,
            'data': health_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Errore nel health check: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@trading_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint non trovato',
        'timestamp': datetime.now().isoformat()
    }), 404

@trading_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Metodo non consentito',
        'timestamp': datetime.now().isoformat()
    }), 405

@trading_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Errore interno del server',
        'timestamp': datetime.now().isoformat()
    }), 500

