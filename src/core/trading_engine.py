"""
Trading Engine for NOCTURNA v2.0 Trading System
Production-grade core trading engine with event-driven architecture.
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .market_data import MarketDataHandler
from .strategy_manager import StrategyManager
from .order_manager import OrderExecutionManager
from .risk_manager import RiskManager


class TradingEngineState(Enum):
    """Trading engine state enumeration."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class TradingEngine:
    """
    Production-grade core trading engine.
    Coordinates all system components and manages the trading lifecycle.
    """

    # Maximum analysis timeout in seconds
    MAX_ANALYSIS_TIMEOUT = 60

    # Default update interval in seconds
    DEFAULT_UPDATE_INTERVAL = 60

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # State management
        self.state = TradingEngineState.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        self._state_lock = threading.RLock()

        # Core components
        self.market_data = MarketDataHandler(config.get('market_data', {}))
        self.strategy_manager = StrategyManager(config.get('strategy', {}))
        self.order_manager = OrderExecutionManager(config.get('trading', {}))
        self.risk_manager = RiskManager(config.get('risk', {}))

        # Configuration
        self.symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        self.update_interval = config.get('update_interval', self.DEFAULT_UPDATE_INTERVAL)
        self.max_concurrent_analysis = config.get('max_concurrent_analysis', 5)

        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.running = False
        self._executor_lock = threading.Lock()

        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}

        # Cache and state
        self.symbol_data: Dict[str, Dict] = {}
        self.active_signals: Dict[str, Dict] = {}
        self.performance_history: List[Dict] = []
        self.performance_metrics: Dict[str, Dict] = {}

        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'start_time': None,
            'uptime': 0
        }
        self._stats_lock = threading.RLock()

        # Setup callbacks
        self._setup_callbacks()

        self.logger.info("Trading Engine initialized")

    def _setup_callbacks(self) -> None:
        """Setup internal callbacks between components."""
        # Order filled callback
        self.order_manager.add_order_callback(self._on_order_filled)

        # Position update callback
        self.order_manager.add_position_callback(self._on_position_update)

        # Risk events callback
        self.risk_manager.risk_events.append = lambda event: self._on_risk_event(event)

    def _on_order_filled(self, order: Dict) -> None:
        """Handle order filled event."""
        try:
            with self._stats_lock:
                self.stats['total_trades'] += 1

            # Remove from active signals
            order_id = order.get('id')
            if order_id in self.active_signals:
                del self.active_signals[order_id]

            self.logger.info(f"Order filled: {order_id}")
            self._notify_event('order_filled', {'order': order})

        except Exception as e:
            self.logger.error(f"Error handling order filled: {e}")

    def _on_position_update(self, position: Dict) -> None:
        """Handle position update event."""
        try:
            pnl = position.get('unrealized_pnl', 0)

            with self._stats_lock:
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                elif pnl < 0:
                    self.stats['losing_trades'] += 1

                self.stats['total_pnl'] += pnl

            self._notify_event('position_updated', {'position': position})

        except Exception as e:
            self.logger.error(f"Error handling position update: {e}")

    def _on_risk_event(self, event: Any) -> None:
        """Handle risk event."""
        self.logger.warning(f"Risk event detected: {event}")
        self._notify_event('risk_event', {'event': str(event)})

    # =========================================================================
    # ENGINE LIFECYCLE METHODS
    # =========================================================================

    def start(self) -> bool:
        """
        Start the trading engine.

        Returns:
            True if started successfully
        """
        with self._state_lock:
            if self.state != TradingEngineState.STOPPED:
                self.logger.warning(f"Engine already in state: {self.state}")
                return False

            try:
                self.logger.info("Starting Trading Engine...")
                self.state = TradingEngineState.STARTING

                # Initialize executor
                self.executor = ThreadPoolExecutor(
                    max_workers=self.max_concurrent_analysis,
                    thread_name_prefix="TradingAnalysis"
                )

                # Start components
                self._start_components()

                # Start main loop thread
                self.running = True
                self.main_thread = threading.Thread(
                    target=self._main_loop,
                    daemon=True,
                    name="TradingMainLoop"
                )
                self.main_thread.start()

                # Update state
                self.state = TradingEngineState.RUNNING
                self.start_time = datetime.now(timezone.utc)
                self.stats['start_time'] = self.start_time

                self.logger.info("Trading Engine started successfully")
                self._notify_event('engine_started', {'timestamp': self.start_time})

                return True

            except Exception as e:
                self.logger.error(f"Error starting engine: {e}")
                self.state = TradingEngineState.ERROR
                return False

    def stop(self) -> bool:
        """
        Stop the trading engine gracefully.

        Returns:
            True if stopped successfully
        """
        with self._state_lock:
            try:
                self.logger.info("Stopping Trading Engine...")

                # Signal main loop to stop
                self.running = False

                # Wait for main thread
                if self.main_thread and self.main_thread.is_alive():
                    self.main_thread.join(timeout=30)

                # Shutdown executor
                with self._executor_lock:
                    if self.executor:
                        self.executor.shutdown(wait=True, cancel_futures=True)
                        self.executor = None

                # Stop components
                self._stop_components()

                # Update state
                self.state = TradingEngineState.STOPPED

                self.logger.info("Trading Engine stopped")
                self._notify_event('engine_stopped', {'timestamp': datetime.now(timezone.utc)})

                return True

            except Exception as e:
                self.logger.error(f"Error stopping engine: {e}")
                return False

    def pause(self) -> bool:
        """Pause the trading engine."""
        with self._state_lock:
            if self.state != TradingEngineState.RUNNING:
                return False

            self.state = TradingEngineState.PAUSED
            self.logger.info("Trading Engine paused")
            self._notify_event('engine_paused', {'timestamp': datetime.now(timezone.utc)})

            return True

    def resume(self) -> bool:
        """Resume the trading engine from pause."""
        with self._state_lock:
            if self.state != TradingEngineState.PAUSED:
                return False

            self.state = TradingEngineState.RUNNING
            self.logger.info("Trading Engine resumed")
            self._notify_event('engine_resumed', {'timestamp': datetime.now(timezone.utc)})

            return True

    def emergency_stop(self, reason: str = "Emergency stop triggered") -> None:
        """
        Execute emergency stop of the trading system.
        Immediately halts all trading activity and closes positions if configured.
        """
        try:
            self.logger.critical(f"EMERGENCY STOP: {reason}")

            # Cancel all active orders
            self._cancel_all_orders()

            # Close all positions if configured
            if self.config.get('emergency_close_positions', False):
                self._close_all_positions()

            # Update state
            self.state = TradingEngineState.EMERGENCY_STOP
            self.running = False

            # Shutdown executor
            with self._executor_lock:
                if self.executor:
                    self.executor.shutdown(wait=False, cancel_futures=True)

            self._notify_event('emergency_stop', {
                'reason': reason,
                'timestamp': datetime.now(timezone.utc)
            })

        except Exception as e:
            self.logger.error(f"Error in emergency stop: {e}")

    # =========================================================================
    # COMPONENT MANAGEMENT
    # =========================================================================

    def _start_components(self) -> None:
        """Start all system components."""
        try:
            # Start market data feed
            self.market_data.start_real_time_feed()

            # Start order monitoring
            self.order_manager.start_monitoring()

            # Subscribe to symbols
            for symbol in self.symbols:
                self.market_data.subscribe_to_symbol(symbol, self._on_price_update)

            self.logger.info("All components started")

        except Exception as e:
            self.logger.error(f"Error starting components: {e}")
            raise

    def _stop_components(self) -> None:
        """Stop all system components."""
        try:
            # Stop market data feed
            self.market_data.stop_real_time_feed()

            # Stop order monitoring
            self.order_manager.stop_monitoring()

            self.logger.info("All components stopped")

        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")

    # =========================================================================
    # MAIN TRADING LOOP
    # =========================================================================

    def _main_loop(self) -> None:
        """Main trading loop executed in separate thread."""
        self.logger.info("Main trading loop started")

        while self.running:
            try:
                loop_start = time.time()

                # Check state
                if self.state == TradingEngineState.PAUSED:
                    time.sleep(1)
                    continue

                if self.state == TradingEngineState.EMERGENCY_STOP:
                    break

                # Update market analysis
                self._update_market_analysis()

                # Monitor risk
                self._monitor_risk()

                # Update statistics
                self._update_statistics()

                # Update timestamp
                self.last_update = datetime.now(timezone.utc)

                # Calculate sleep time
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.update_interval - loop_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(10)

        self.logger.info("Main trading loop terminated")

    def _update_market_analysis(self) -> None:
        """Update market analysis for all symbols."""
        try:
            futures = []

            for symbol in self.symbols:
                if self.executor:
                    future = self.executor.submit(self._analyze_symbol, symbol)
                    futures.append((symbol, future))

            # Collect results
            for symbol, future in futures:
                try:
                    result = future.result(timeout=self.MAX_ANALYSIS_TIMEOUT)
                    if result:
                        self._process_analysis_result(symbol, result)

                except FuturesTimeoutError:
                    self.logger.warning(f"Analysis timeout for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error updating market analysis: {e}")

    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol."""
        try:
            # Get historical data
            df = self.market_data.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=500
            )

            if df.empty:
                return None

            # Calculate indicators
            df = self.market_data.calculate_technical_indicators(df)

            # Update strategy
            strategy_result = self.strategy_manager.update_strategy(df, symbol)

            # Cache data
            self.symbol_data[symbol] = {
                'data': df,
                'strategy_result': strategy_result,
                'last_update': datetime.now(timezone.utc)
            }

            return strategy_result

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _process_analysis_result(self, symbol: str, result: Dict) -> None:
        """Process analysis result for a symbol."""
        try:
            signals = result.get('signals', [])

            for signal in signals:
                self._process_trading_signal(signal)

            self._update_symbol_metrics(symbol, result)

        except Exception as e:
            self.logger.error(f"Error processing result for {symbol}: {e}")

    def _process_trading_signal(self, signal: Dict) -> None:
        """Process a trading signal through risk validation."""
        try:
            # Get positions and market data
            positions = self.order_manager.get_positions()
            market_data = self._get_market_data_for_risk(signal['symbol'])

            # Validate with risk manager
            is_valid, reason, adjusted_signal = self.risk_manager.validate_trade(
                signal, positions, market_data
            )

            if not is_valid:
                self.logger.info(f"Signal rejected for {signal['symbol']}: {reason}")
                return

            # Submit order
            order_id = self.order_manager.submit_order(adjusted_signal)

            if order_id:
                self.active_signals[order_id] = adjusted_signal
                self.logger.info(f"Order submitted: {order_id}")
                self._notify_event('signal_executed', {
                    'signal': adjusted_signal,
                    'order_id': order_id
                })

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")

    def _monitor_risk(self) -> None:
        """Monitor portfolio risk."""
        try:
            positions = self.order_manager.get_positions()
            market_data = {
                symbol: self._get_market_data_for_risk(symbol)
                for symbol in self.symbols
            }

            risk_report = self.risk_manager.monitor_portfolio_risk(positions, market_data)
            risk_events = risk_report.get('risk_events', [])

            # Check for critical events
            if 'DRAWDOWN_LIMIT' in risk_events or 'DAILY_LOSS_LIMIT' in risk_events:
                self.emergency_stop("Risk limit exceeded")

            elif risk_report.get('risk_level') == 'CRITICAL':
                self.logger.warning("Critical risk level detected")
                self._handle_critical_risk()

        except Exception as e:
            self.logger.error(f"Error monitoring risk: {e}")

    def _handle_critical_risk(self) -> None:
        """Handle critical risk situations."""
        try:
            # Cancel pending orders
            self._cancel_pending_orders()

            # Reduce exposure
            self._reduce_exposure()

            self._notify_event('critical_risk', {'action': 'risk_reduction'})

        except Exception as e:
            self.logger.error(f"Error handling critical risk: {e}")

    def _get_market_data_for_risk(self, symbol: str) -> Dict:
        """Get market data for risk calculations."""
        try:
            import pandas as pd

            symbol_cache = self.symbol_data.get(symbol, {})
            df = symbol_cache.get('data', pd.DataFrame())

            if df.empty:
                return {}

            latest = df.iloc[-1]

            return {
                'price': latest.get('close', 0),
                'atr': latest.get('atr', 0),
                'volatility': latest.get('atr', 0) / latest.get('close', 1) if latest.get('close', 0) > 0 else 0,
                'avg_atr': df['atr'].tail(20).mean() if 'atr' in df.columns else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting market data for risk: {e}")
            return {}

    def _on_price_update(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Handle price update from market data feed."""
        try:
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {}

            self.symbol_data[symbol]['last_price'] = price
            self.symbol_data[symbol]['last_price_update'] = timestamp

            # Update price history for correlation calculations
            self.risk_manager._update_price_history(symbol, price)

        except Exception as e:
            self.logger.error(f"Error handling price update for {symbol}: {e}")

    # =========================================================================
    # ORDER MANAGEMENT HELPERS
    # =========================================================================

    def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        try:
            active_orders = self.order_manager.active_orders

            for order_id in list(active_orders.keys()):
                self.order_manager.cancel_order(order_id)

            self.logger.info(f"Cancelled {len(active_orders)} orders")

        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")

    def _cancel_pending_orders(self) -> None:
        """Cancel only pending orders."""
        try:
            active_orders = self.order_manager.active_orders
            cancelled = 0

            for order_id, order in active_orders.items():
                if order.status.value in ['PENDING', 'SUBMITTED']:
                    self.order_manager.cancel_order(order_id)
                    cancelled += 1

            self.logger.info(f"Cancelled {cancelled} pending orders")

        except Exception as e:
            self.logger.error(f"Error cancelling pending orders: {e}")

    def _close_all_positions(self) -> None:
        """Close all open positions."""
        try:
            positions = self.order_manager.get_positions()

            for symbol, position in positions.items():
                quantity = abs(position['quantity'])
                side = 'sell' if position['quantity'] > 0 else 'buy'

                close_signal = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'market',
                    'quantity': quantity
                }

                self.order_manager.submit_order(close_signal)

            self.logger.info(f"Closed {len(positions)} positions")

        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")

    def _reduce_exposure(self) -> None:
        """Reduce portfolio exposure by 50%."""
        try:
            positions = self.order_manager.get_positions()

            for symbol, position in positions.items():
                reduce_quantity = abs(position['quantity']) * 0.5
                side = 'sell' if position['quantity'] > 0 else 'buy'

                if reduce_quantity > 0:
                    reduce_signal = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'market',
                        'quantity': reduce_quantity
                    }

                    self.order_manager.submit_order(reduce_signal)

            self.logger.info("Exposure reduced by 50%")

        except Exception as e:
            self.logger.error(f"Error reducing exposure: {e}")

    # =========================================================================
    # STATISTICS AND METRICS
    # =========================================================================

    def _update_statistics(self) -> None:
        """Update system statistics."""
        try:
            with self._stats_lock:
                if self.start_time:
                    self.stats['uptime'] = (
                        datetime.now(timezone.utc) - self.start_time
                    ).total_seconds()

                # Calculate win rate
                total_closed = self.stats['winning_trades'] + self.stats['losing_trades']
                if total_closed > 0:
                    self.stats['win_rate'] = self.stats['winning_trades'] / total_closed
                else:
                    self.stats['win_rate'] = 0.0

                # Update drawdown
                positions = self.order_manager.get_positions()
                current_drawdown = self.risk_manager._calculate_current_drawdown(positions)
                self.stats['max_drawdown'] = max(
                    self.stats['max_drawdown'],
                    current_drawdown
                )

        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")

    def _update_symbol_metrics(self, symbol: str, result: Dict) -> None:
        """Update metrics for a symbol."""
        try:
            if symbol not in self.performance_metrics:
                self.performance_metrics[symbol] = {
                    'signals_generated': 0,
                    'trades_executed': 0,
                    'last_mode': None,
                    'mode_changes': 0
                }

            metrics = self.performance_metrics[symbol]
            metrics['signals_generated'] += len(result.get('signals', []))

            current_mode = result.get('trading_mode')
            if metrics['last_mode'] and metrics['last_mode'] != current_mode:
                metrics['mode_changes'] += 1

            metrics['last_mode'] = current_mode

        except Exception as e:
            self.logger.error(f"Error updating metrics for {symbol}: {e}")

    # =========================================================================
    # EVENT SYSTEM
    # =========================================================================

    def _notify_event(self, event_type: str, data: Dict) -> None:
        """Notify registered event callbacks."""
        callbacks = self.event_callbacks.get(event_type, [])

        for callback in callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Error in event callback for {event_type}: {e}")

    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for an event type."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []

        self.event_callbacks[event_type].append(callback)

    # =========================================================================
    # STATUS AND CONFIGURATION
    # =========================================================================

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            'engine_state': self.state.value,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'uptime': self.stats.get('uptime', 0),
            'symbols': self.symbols,
            'active_orders': len(self.order_manager.active_orders),
            'active_positions': len(self.order_manager.get_positions()),
            'current_mode': self.strategy_manager.current_mode.value,
            'current_market_state': self.strategy_manager.current_market_state.value,
            'risk_level': self.risk_manager.current_risk_level.value,
            'statistics': self.stats.copy(),
            'performance_metrics': self.performance_metrics.copy()
        }

    def get_detailed_status(self) -> Dict:
        """Get detailed engine status."""
        basic_status = self.get_status()

        basic_status.update({
            'positions': self.order_manager.get_positions(),
            'active_orders': {
                oid: orec.to_dict() if hasattr(orec, 'to_dict') else str(orec)
                for oid, orec in self.order_manager.active_orders.items()
            },
            'strategy_status': self.strategy_manager.get_strategy_status(),
            'risk_report': self.risk_manager.get_risk_report(),
            'trading_summary': self.order_manager.get_trading_summary(),
            'symbol_data': {
                symbol: {
                    'last_price': data.get('last_price'),
                    'last_update': data.get('last_price_update'),
                    'strategy_result': data.get('strategy_result', {})
                }
                for symbol, data in self.symbol_data.items()
            }
        })

        return basic_status

    def update_config(self, new_config: Dict) -> bool:
        """Update engine configuration."""
        try:
            self.config.update(new_config)

            # Update symbols if changed
            if 'symbols' in new_config:
                self.symbols = new_config['symbols']

            # Update strategy parameters
            if 'strategy' in new_config:
                self.strategy_manager.parameters.update(new_config['strategy'])

            # Update risk parameters
            if 'risk' in new_config:
                self.risk_manager.risk_parameters.update(new_config['risk'])

            self.logger.info("Configuration updated")
            return True

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
