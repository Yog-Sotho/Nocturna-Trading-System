"""
Trading Engine for NOCTURNA v2.0 Trading System
Production-grade core trading engine with event-driven architecture.
"""

import logging
import threading
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from enum import Enum


from .market_data import MarketDataHandler
from .strategy_manager import StrategyManager
from .order_manager import OrderExecutionManager
from .risk_manager import RiskManager

# Advanced modules — optional, degrade gracefully if unavailable
try:
    from src.advanced.sentiment_analyzer import SentimentAnalyzer
    _SENTIMENT_AVAILABLE = True
except ImportError:
    _SENTIMENT_AVAILABLE = False

try:
    from src.advanced.ml_optimizer import MLOptimizer
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

try:
    from .external_signals import ExternalSignalAggregator
    _EXTERNAL_SIGNALS_AVAILABLE = True
except ImportError:
    _EXTERNAL_SIGNALS_AVAILABLE = False


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

        # Advanced components — optional
        self.sentiment_analyzer = None
        self.ml_optimizer = None
        self._ml_last_optimization: Optional[datetime] = None
        self._ml_optimization_interval = config.get('ml_optimization_interval', 86400)  # 24h default

        if _SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer(config.get('sentiment', {}))
                self.logger.info("Sentiment Analyzer wired into engine")
            except Exception as e:
                self.logger.warning(f"Sentiment Analyzer init failed: {e}")

        if _ML_AVAILABLE:
            try:
                self.ml_optimizer = MLOptimizer(config.get('ml_optimizer', {}))
                self.logger.info("ML Optimizer wired into engine")
            except Exception as e:
                self.logger.warning(f"ML Optimizer init failed: {e}")

        # F16: External signal aggregator
        self.external_signals = None
        if _EXTERNAL_SIGNALS_AVAILABLE:
            try:
                self.external_signals = ExternalSignalAggregator(config.get('external_signals', {}))
                self.logger.info("External Signal Aggregator wired into engine")
            except Exception as e:
                self.logger.warning(f"External Signal Aggregator init failed: {e}")

        # Execution quality tracking
        self._execution_latencies: List[float] = []
        self._max_latency_samples = 1000

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
        self._symbol_data_lock = threading.Lock()
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

        # F11: Circuit breaker for API failures
        self._api_failure_count = 0
        self._api_failure_threshold = 3      # Pause after 3 consecutive failures
        self._api_pause_duration = 300       # 5 minute pause
        self._api_paused_until: Optional[datetime] = None

        # F13: Trade journal — persistent signal log
        self.trade_journal: List[Dict] = []
        self._journal_lock = threading.Lock()

        # F4: Higher timeframe data for multi-TF confirmation
        self._htf_timeframes = config.get('higher_timeframes', ['4h'])
        self._htf_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Setup callbacks
        self._setup_callbacks()

        self.logger.info("Trading Engine initialized")

    def _setup_callbacks(self) -> None:
        """Setup internal callbacks between components."""
        # Order filled callback
        self.order_manager.add_order_callback(self._on_order_filled)

        # Position update callback
        self.order_manager.add_position_callback(self._on_position_update)

        # Risk events callback — use proper observer pattern
        self.risk_manager.add_risk_callback(self._on_risk_event)

    def _on_order_filled(self, order: Dict) -> None:
        """Handle order filled event — track wins/losses on realized P&L only."""
        try:
            realized_pnl = order.get('realized_pnl', None)

            with self._stats_lock:
                self.stats['total_trades'] += 1

                # Only count win/loss when we have realized P&L (trade closed)
                if realized_pnl is not None:
                    if realized_pnl > 0:
                        self.stats['winning_trades'] += 1
                    elif realized_pnl < 0:
                        self.stats['losing_trades'] += 1
                    self.stats['total_pnl'] += realized_pnl

                    # Record trade in risk manager
                    self.risk_manager.record_trade({
                        'symbol': order.get('symbol'),
                        'side': order.get('side'),
                        'pnl': realized_pnl,
                        'quantity': order.get('filled_quantity', 0)
                    })

            # Remove from active signals
            order_id = order.get('id')
            if order_id in self.active_signals:
                del self.active_signals[order_id]

            self.logger.info(f"Order filled: {order_id}")
            self._notify_event('order_filled', {'order': order})

        except Exception as e:
            self.logger.error(f"Error handling order filled: {e}")

    def _on_position_update(self, position: Dict) -> None:
        """Handle position update event — track unrealized P&L without counting wins/losses."""
        try:
            # Update portfolio value in risk manager from position data
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
                    if self.main_thread.is_alive():
                        self.logger.warning("Main loop thread did not stop within 30s timeout")

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
        with self._state_lock:
            try:
                self.logger.critical(f"EMERGENCY STOP: {reason}")

                # Signal main loop to stop first
                self.running = False

                # Cancel all active orders
                self._cancel_all_orders()

                # S4: Stop order monitoring to prevent pending sim fills from completing
                self.order_manager.stop_monitoring()

                # Close all positions if configured
                if self.config.get('emergency_close_positions', False):
                    self._close_all_positions()

                # Update state
                self.state = TradingEngineState.EMERGENCY_STOP

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
        """Main trading loop with bar-boundary alignment (F9)."""
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

                # F11: Check circuit breaker
                if self._api_paused_until:
                    if datetime.now(timezone.utc) < self._api_paused_until:
                        remaining = (self._api_paused_until - datetime.now(timezone.utc)).total_seconds()
                        self.logger.debug(f"API circuit breaker active, {remaining:.0f}s remaining")
                        time.sleep(min(10, remaining))
                        continue
                    else:
                        self._api_paused_until = None
                        self._api_failure_count = 0
                        self.logger.info("API circuit breaker reset")

                # Update market analysis
                self._update_market_analysis()

                # Monitor risk
                self._monitor_risk()

                # Update statistics
                self._update_statistics()

                # Periodic ML parameter optimization
                self._check_ml_optimization()

                # Update timestamp
                self.last_update = datetime.now(timezone.utc)

                # F9: Bar-boundary aligned sleep
                # For 1h strategies, sleep until just after the next hour boundary
                loop_time = time.time() - loop_start
                sleep_time = self._calc_bar_boundary_sleep(loop_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(10)

        self.logger.info("Main trading loop terminated")

    def _calc_bar_boundary_sleep(self, elapsed: float) -> float:
        """
        F9: Calculate sleep time aligned to the next bar boundary.
        For 1h timeframe: wake up at :00:10 of each hour (10s after bar close).
        Falls back to standard interval if bar-boundary alignment is not meaningful.
        """
        try:
            now = datetime.now(timezone.utc)
            # Next hour boundary + 10 seconds (allow bar to finalize)
            next_bar = now.replace(minute=0, second=10, microsecond=0) + timedelta(hours=1)
            sleep_secs = (next_bar - now).total_seconds()

            # Cap: don't sleep more than the configured interval
            max_sleep = max(0, self.update_interval - elapsed)
            return min(sleep_secs, max_sleep)

        except Exception:
            return max(0, self.update_interval - elapsed)

    def _update_market_analysis(self) -> None:
        """Update market analysis for all symbols using as_completed() (F12/V3)."""
        try:
            futures_map = {}

            for symbol in self.symbols:
                if self.executor:
                    future = self.executor.submit(self._analyze_symbol, symbol)
                    futures_map[future] = symbol

            # F12: Process results as they complete, not sequentially
            for future in as_completed(futures_map, timeout=self.MAX_ANALYSIS_TIMEOUT):
                symbol = futures_map[future]
                try:
                    result = future.result(timeout=5)  # Short timeout since already completed
                    if result:
                        self._process_analysis_result(symbol, result)
                        self._api_failure_count = 0  # F11: Reset on success

                except FuturesTimeoutError:
                    self.logger.warning(f"Analysis timeout for {symbol}")
                    self._record_api_failure()
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                    self._record_api_failure()

        except FuturesTimeoutError:
            self.logger.warning("Global analysis timeout — some symbols may have been skipped")
            self._record_api_failure()
        except Exception as e:
            self.logger.error(f"Error updating market analysis: {e}")

    def _record_api_failure(self) -> None:
        """F11: Track API failures and trigger circuit breaker if threshold exceeded."""
        self._api_failure_count += 1
        if self._api_failure_count >= self._api_failure_threshold:
            self._api_paused_until = datetime.now(timezone.utc) + timedelta(seconds=self._api_pause_duration)
            self.logger.warning(
                f"Circuit breaker triggered: {self._api_failure_count} consecutive failures. "
                f"Pausing API calls for {self._api_pause_duration}s"
            )

    def _fetch_higher_tf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """F4: Fetch higher timeframe data for multi-TF confirmation."""
        htf_data = {}
        for tf in self._htf_timeframes:
            try:
                df = self.market_data.get_historical_data(
                    symbol=symbol, timeframe=tf, limit=250
                )
                if not df.empty:
                    df = self.market_data.calculate_technical_indicators(df)
                    htf_data[tf] = df
            except Exception as e:
                self.logger.debug(f"HTF {tf} data unavailable for {symbol}: {e}")
        return htf_data

    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyze a single symbol with full pipeline.
        F3: Thread-safe — passes all mutable context as arguments.
        F4: Fetches higher TF data for multi-timeframe confirmation.
        F16: Merges external signal data into sentiment.
        """
        analysis_start = time.time()
        try:
            # Get historical data
            df = self.market_data.get_historical_data(
                symbol=symbol, timeframe='1h', limit=500
            )

            if df.empty:
                return None

            # Calculate indicators
            df = self.market_data.calculate_technical_indicators(df)

            # Get current positions (snapshot — thread-safe read)
            positions = self.order_manager.get_positions()

            # Get sentiment data (if available)
            sentiment = {}
            if self.sentiment_analyzer:
                try:
                    signal = self.sentiment_analyzer.get_market_sentiment_signal(
                        symbol, lookback_hours=24
                    )
                    sentiment = {symbol: signal}
                except Exception as e:
                    self.logger.debug(f"Sentiment unavailable for {symbol}: {e}")

            # F16: Merge external signal into sentiment data
            if self.external_signals:
                try:
                    ext = self.external_signals.get_composite_signal(symbol)
                    if ext and ext.get('confidence', 0) > 0.2:
                        # Merge external score into sentiment (blend 50/50 with internal)
                        sym_sent = sentiment.get(symbol, {})
                        internal_score = sym_sent.get('sentiment_score', 0.0)
                        internal_conf = sym_sent.get('confidence', 0.0)
                        ext_score = ext['score']
                        ext_conf = ext['confidence']

                        # Weighted blend
                        total_w = internal_conf + ext_conf
                        if total_w > 0:
                            blended_score = (internal_score * internal_conf + ext_score * ext_conf) / total_w
                            blended_conf = min(total_w / 2.0, 1.0)
                        else:
                            blended_score = ext_score
                            blended_conf = ext_conf

                        sentiment[symbol] = {
                            'sentiment_score': blended_score,
                            'confidence': blended_conf,
                            'internal_score': internal_score,
                            'external_score': ext_score,
                            'external_sources': ext.get('source_count', 0),
                        }
                except Exception as e:
                    self.logger.debug(f"External signals unavailable for {symbol}: {e}")

            # F4: Fetch higher timeframe data
            higher_tf = self._fetch_higher_tf_data(symbol)

            # F3: Pass all context as arguments — no shared mutable state
            strategy_result = self.strategy_manager.update_strategy(
                df, symbol,
                positions=positions,
                sentiment=sentiment,
                higher_tf_data=higher_tf
            )

            # Track analysis latency
            latency_ms = (time.time() - analysis_start) * 1000
            self._track_latency(latency_ms, symbol)

            # Cache data — thread-safe write
            with self._symbol_data_lock:
                self.symbol_data[symbol] = {
                    'data': df,
                    'strategy_result': strategy_result,
                    'last_update': datetime.now(timezone.utc),
                    'analysis_latency_ms': latency_ms,
                }

            return strategy_result

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _track_latency(self, latency_ms: float, symbol: str) -> None:
        """Track execution latency for monitoring."""
        self._execution_latencies.append(latency_ms)
        if len(self._execution_latencies) > self._max_latency_samples:
            self._execution_latencies = self._execution_latencies[-self._max_latency_samples:]
        if latency_ms > 5000:  # Log slow analyses
            self.logger.warning(f"Slow analysis for {symbol}: {latency_ms:.0f}ms")

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
        """Process a trading signal through risk validation. F13: Log to trade journal."""
        try:
            # Guard: do not submit orders if engine is not running
            if self.state != TradingEngineState.RUNNING:
                self.logger.info(f"Signal dropped — engine state is {self.state.value}")
                self._journal_log(signal, 'DROPPED', f'engine_state={self.state.value}')
                return

            # Get positions and market data
            positions = self.order_manager.get_positions()
            market_data = self._get_market_data_for_risk(signal['symbol'])

            # Validate with risk manager
            is_valid, reason, adjusted_signal = self.risk_manager.validate_trade(
                signal, positions, market_data
            )

            if not is_valid:
                self.logger.info(f"Signal rejected for {signal['symbol']}: {reason}")
                self._journal_log(signal, 'REJECTED', reason)
                return

            # Submit order
            order_id = self.order_manager.submit_order(adjusted_signal)

            if order_id:
                self.active_signals[order_id] = adjusted_signal
                self.logger.info(f"Order submitted: {order_id}")
                self._journal_log(adjusted_signal, 'EXECUTED', f'order_id={order_id}')
                self._notify_event('signal_executed', {
                    'signal': adjusted_signal,
                    'order_id': order_id
                })
            else:
                self._journal_log(signal, 'SUBMIT_FAILED', 'order_manager returned None')

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            self._journal_log(signal, 'ERROR', str(e))

    def _journal_log(self, signal: Dict, outcome: str, detail: str = '') -> None:
        """F13: Append entry to trade journal for analytics."""
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': signal.get('symbol', ''),
            'side': signal.get('side', ''),
            'mode': signal.get('mode', ''),
            'signal_type': signal.get('signal_type', ''),
            'confidence': signal.get('confidence', 0),
            'outcome': outcome,
            'detail': detail,
        }
        with self._journal_lock:
            self.trade_journal.append(entry)
            # Cap journal size
            if len(self.trade_journal) > 10000:
                self.trade_journal = self.trade_journal[-5000:]

    def _monitor_risk(self) -> None:
        """Monitor portfolio risk and update portfolio value from broker."""
        try:
            positions = self.order_manager.get_positions()

            # Update portfolio value from actual positions (5h fix)
            if self.order_manager.alpaca_client:
                try:
                    account = self.order_manager.alpaca_client.get_account()
                    equity = float(account.equity)
                    self.risk_manager.update_portfolio_value(equity)
                except Exception as e:
                    self.logger.debug(f"Could not update portfolio equity from broker: {e}")

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
            with self._symbol_data_lock:
                symbol_cache = self.symbol_data.get(symbol, {})
                df = symbol_cache.get('data', pd.DataFrame())

            if df.empty:
                return {}

            latest = df.iloc[-1]
            close_price = latest.get('close', 0)
            atr_value = latest.get('atr', 0)

            return {
                'price': close_price,
                'atr': atr_value,
                'volatility': atr_value / close_price if close_price > 0 else 0,
                'avg_atr': df['atr'].tail(20).mean() if 'atr' in df.columns else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting market data for risk: {e}")
            return {}

    def _on_price_update(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Handle price update from market data feed."""
        try:
            with self._symbol_data_lock:
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

    def _check_ml_optimization(self) -> None:
        """
        Periodically trigger ML parameter optimization.
        Runs in background, does not block the main loop.
        Only runs if enough time has elapsed since last optimization.
        """
        if not self.ml_optimizer:
            return

        try:
            now = datetime.now(timezone.utc)

            # Check if enough time has passed
            if self._ml_last_optimization:
                elapsed = (now - self._ml_last_optimization).total_seconds()
                if elapsed < self._ml_optimization_interval:
                    return

            # Need at least some trade history for meaningful optimization
            if self.stats.get('total_trades', 0) < 10:
                return

            self.logger.info("Starting periodic ML parameter optimization")
            self._ml_last_optimization = now

            # Gather recent performance for adaptive tuning
            risk_level_map = {'LOW': 0.1, 'MEDIUM': 0.2, 'HIGH': 0.35, 'CRITICAL': 0.5}
            recent_performance = {
                'win_rate': self.stats.get('win_rate', 0.5),
                'avg_return': self.stats.get('total_pnl', 0) / max(self.stats.get('total_trades', 1), 1),
                'volatility': risk_level_map.get(self.risk_manager.current_risk_level.value, 0.2),
            }

            # Get market data for first symbol as representative
            if self.symbols:
                with self._symbol_data_lock:
                    symbol_cache = self.symbol_data.get(self.symbols[0], {})
                    df = symbol_cache.get('data', pd.DataFrame())

                if not df.empty:
                    adjusted_params = self.ml_optimizer.adaptive_parameter_tuning(
                        df, self.strategy_manager.parameters, recent_performance
                    )

                    # Apply adjusted parameters to strategy manager
                    if adjusted_params:
                        self.strategy_manager.parameters.update(adjusted_params)
                        self.logger.info("ML optimizer updated strategy parameters")
                        self._notify_event('ml_optimization', {
                            'adjusted_params': list(adjusted_params.keys()),
                            'timestamp': now
                        })

        except Exception as e:
            self.logger.error(f"Error in ML optimization check: {e}")

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
        """Get detailed engine status including trade journal stats."""
        basic_status = self.get_status()

        # F13: Trade journal summary
        with self._journal_lock:
            journal_len = len(self.trade_journal)
            recent_outcomes = {}
            for entry in self.trade_journal[-100:]:
                outcome = entry.get('outcome', 'UNKNOWN')
                recent_outcomes[outcome] = recent_outcomes.get(outcome, 0) + 1

        basic_status.update({
            'positions': self.order_manager.get_positions(),
            'active_orders': {
                oid: orec.to_dict() if hasattr(orec, 'to_dict') else str(orec)
                for oid, orec in self.order_manager.active_orders.items()
            },
            'strategy_status': self.strategy_manager.get_strategy_status(),
            'risk_report': self.risk_manager.get_risk_report(),
            'trading_summary': self.order_manager.get_trading_summary(),
            'symbol_data': self._get_symbol_data_snapshot(),
            'trade_journal': {
                'total_entries': journal_len,
                'recent_outcomes': recent_outcomes,
            },
            'circuit_breaker': {
                'failure_count': self._api_failure_count,
                'paused_until': self._api_paused_until.isoformat() if self._api_paused_until else None,
            },
            'execution_quality': self._get_execution_quality(),
            'external_signals': self.external_signals.get_status() if self.external_signals else None,
        })

        return basic_status

    def get_trade_journal(self, last_n: int = 100) -> List[Dict]:
        """F13: Get recent trade journal entries."""
        with self._journal_lock:
            return list(self.trade_journal[-last_n:])

    def _get_execution_quality(self) -> Dict:
        """Get execution quality metrics — latency statistics."""
        if not self._execution_latencies:
            return {'samples': 0}

        import numpy as np
        lats = self._execution_latencies
        return {
            'samples': len(lats),
            'avg_ms': round(float(np.mean(lats)), 1),
            'p50_ms': round(float(np.median(lats)), 1),
            'p95_ms': round(float(np.percentile(lats, 95)), 1),
            'p99_ms': round(float(np.percentile(lats, 99)), 1),
            'max_ms': round(float(np.max(lats)), 1),
            'min_ms': round(float(np.min(lats)), 1),
        }

    def _get_symbol_data_snapshot(self) -> Dict:
        """Get a thread-safe snapshot of symbol data for status reporting."""
        with self._symbol_data_lock:
            return {
                symbol: {
                    'last_price': data.get('last_price'),
                    'last_update': data.get('last_price_update'),
                    'strategy_result': data.get('strategy_result', {})
                }
                for symbol, data in self.symbol_data.items()
            }

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
