"""
Order Execution Manager for NOCTURNA v2.0 Trading System
Production-grade order management with enhanced error handling and validation.
"""

import os
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from enum import Enum
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRecord:
    """Structured order record for tracking."""
    id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    client_order_id: str = ""
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    filled_avg_price: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class OrderExecutionManager:
    """
    Production-grade order execution manager.
    Handles order submission, monitoring, and position tracking.
    """

    # Maximum order history to retain
    MAX_ORDER_HISTORY = 10000
    MAX_DAILY_ORDERS = 500

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Broker client
        self.alpaca_client = None

        # Order management
        self.active_orders: Dict[str, OrderRecord] = {}
        self.order_history: List[OrderRecord] = []
        self.positions: Dict[str, Dict] = {}

        # Trailing stops
        self.trailing_stops: Dict[str, Dict] = {}

        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        self.running = False
        self.monitor_interval = 5  # seconds

        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []

        # Risk limits
        self.max_daily_loss = float(config.get('max_daily_loss', 0.05))
        self.position_limits = config.get('position_limits', {})

        # Daily tracking
        self.daily_order_count = 0
        self.daily_reset_time = datetime.now(timezone.utc).date()

        # Performance tracking
        self.performance_history = deque(maxlen=1000)

        # Thread safety
        self._lock = threading.RLock()

        # Initialize broker client
        self._initialize_broker_client()

    def _initialize_broker_client(self) -> None:
        """Initialize broker API client with proper error handling."""
        try:
            # Load secrets directly from environment — never pass through config dicts
            alpaca_api_key = os.environ.get('ALPACA_API_KEY')
            alpaca_secret_key = os.environ.get('ALPACA_SECRET_KEY')
            alpaca_base_url = os.environ.get(
                'ALPACA_BASE_URL',
                'https://paper-api.alpaca.markets'
            )

            if alpaca_api_key and alpaca_secret_key:
                try:
                    from alpaca_trade_api import REST as AlpacaREST
                    self.alpaca_client = AlpacaREST(
                        key_id=alpaca_api_key,
                        secret_key=alpaca_secret_key,
                        base_url=alpaca_base_url
                    )
                    self.logger.info("Alpaca trading client initialized successfully")

                    # Load existing positions
                    self._load_existing_positions()

                except ImportError:
                    self.logger.error("Alpaca Trade API not installed")
                    self.alpaca_client = None
                except Exception as e:
                    self.logger.error(f"Failed to initialize Alpaca client: {e}")
                    self.alpaca_client = None
            else:
                self.logger.info("Alpaca API keys not configured - running in simulation mode")
                self.alpaca_client = None

        except Exception as e:
            self.logger.error(f"Error initializing broker client: {e}")
            self.alpaca_client = None

    def _load_existing_positions(self) -> None:
        """Load existing positions from broker."""
        if not self.alpaca_client:
            return

        with self._lock:
            try:
                positions = self.alpaca_client.list_positions()
                for pos in positions:
                    self.positions[pos.symbol] = {
                        'symbol': pos.symbol,
                        'quantity': float(pos.qty),
                        'side': 'long' if float(pos.qty) > 0 else 'short',
                        'avg_price': float(pos.avg_entry_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'last_update': datetime.now(timezone.utc)
                    }

                self.logger.info(f"Loaded {len(self.positions)} existing positions")

            except Exception as e:
                self.logger.error(f"Error loading positions: {e}")

    def submit_order(self, signal: Dict) -> Optional[str]:
        """
        Submit a new order based on trading signal.

        Args:
            signal: Trading signal dictionary with order details

        Returns:
            Order ID if successful, None otherwise
        """
        with self._lock:
            try:
                # Validate signal
                if not self._validate_signal(signal):
                    return None

                # Check risk limits
                if not self._check_risk_limits(signal):
                    return None

                # Check for duplicate signal (5j: idempotency)
                if self._is_duplicate_signal(signal):
                    self.logger.info(
                        f"Duplicate signal rejected for {signal['symbol']} {signal['side']}"
                    )
                    return None

                # Check market hours
                if not self._is_market_open():
                    self.logger.info("Market closed, order queued for next session")
                    return None

                # Prepare order data
                order_data = self._prepare_order(signal)
                if not order_data:
                    return None

                # Submit to broker
                order_id = self._submit_to_broker(order_data)

                if order_id:
                    # Register the order
                    self._register_order(order_id, order_data, signal)

                    # Setup trailing stop if applicable
                    if signal.get('trail_trigger'):
                        self._setup_trailing_stop(order_id, signal)

                    self.logger.info(
                        f"Order submitted: {order_id} | "
                        f"Symbol: {signal['symbol']} | "
                        f"Side: {signal['side']} | "
                        f"Qty: {signal['quantity']}"
                    )

                    # Update daily count
                    self._increment_daily_count()

                    return order_id

                return None

            except Exception as e:
                self.logger.error(f"Error submitting order: {e}")
                return None

    def _validate_signal(self, signal: Dict) -> bool:
        """Validate trading signal structure and values."""
        required_fields = ['symbol', 'side', 'type', 'quantity']

        for field_name in required_fields:
            if field_name not in signal:
                self.logger.error(f"Missing required field: {field_name}")
                return False

        # Validate symbol format
        symbol = signal['symbol']
        if not isinstance(symbol, str) or not symbol.isalpha() or len(symbol) > 5:
            self.logger.error(f"Invalid symbol format: {symbol}")
            return False

        # Validate side
        if signal['side'] not in ['buy', 'sell']:
            self.logger.error(f"Invalid side: {signal['side']}")
            return False

        # Validate order type
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if signal['type'] not in valid_types:
            self.logger.error(f"Invalid order type: {signal['type']}")
            return False

        # Validate quantity
        quantity = signal['quantity']
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            self.logger.error(f"Invalid quantity: {quantity}")
            return False

        if quantity > 1000000:
            self.logger.error(f"Quantity too large: {quantity}")
            return False

        # Validate prices for limit orders
        if signal['type'] in ['limit', 'stop_limit']:
            if 'price' not in signal or signal['price'] <= 0:
                self.logger.error("Limit price required for limit orders")
                return False

        if signal['type'] in ['stop', 'stop_limit']:
            if 'stop_price' not in signal or signal['stop_price'] <= 0:
                self.logger.error("Stop price required for stop orders")
                return False

        return True

    def _check_risk_limits(self, signal: Dict) -> bool:
        """Check if order passes risk limits."""
        try:
            # Check daily loss limit
            daily_pnl = sum(
                pos.get('unrealized_pnl', 0) + pos.get('realized_pnl', 0)
                for pos in self.positions.values()
            )

            if daily_pnl < -self.max_daily_loss:
                self.logger.warning("Daily loss limit reached")
                return False

            # Check position limit per symbol
            symbol = signal['symbol']
            current_position = self.positions.get(symbol, {}).get('quantity', 0)
            symbol_limit = self.position_limits.get(symbol, 1.0)

            if signal['side'] == 'buy':
                new_position = current_position + signal['quantity']
            else:
                new_position = current_position - signal['quantity']

            if abs(new_position) > symbol_limit:
                self.logger.warning(f"Position limit exceeded for {symbol}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False

    def _is_market_open(self) -> bool:
        """Check if market is currently open for trading."""
        try:
            if self.alpaca_client:
                clock = self.alpaca_client.get_clock()
                return clock.is_open

            # Fallback: proper US Eastern time market hours (9:30-16:00 ET)
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo("America/New_York")
            now_et = datetime.now(eastern)
            # Weekday (0=Mon, 4=Fri) and between 9:30 and 16:00 ET
            if now_et.weekday() > 4:  # Weekend
                return False
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            return market_open <= now_et < market_close

        except Exception as e:
            self.logger.warning(f"Error checking market hours: {e}, assuming market open")
            return True

    def _prepare_order(self, signal: Dict) -> Optional[Dict]:
        """Prepare order data for broker submission."""
        try:
            order_data = {
                'symbol': signal['symbol'].upper(),
                'qty': str(signal['quantity']),
                'side': signal['side'],
                'type': signal['type'],
                'time_in_force': signal.get('time_in_force', 'day'),
                'client_order_id': str(uuid.uuid4())
            }

            # Add limit price
            if signal['type'] in ['limit', 'stop_limit']:
                order_data['limit_price'] = signal['price']

            # Add stop price
            if signal['type'] in ['stop', 'stop_limit']:
                order_data['stop_price'] = signal['stop_price']

            # Add bracket orders (take profit / stop loss) with validation
            has_sl = signal.get('stop_loss') and signal['stop_loss'] > 0
            has_tp = signal.get('take_profit') and signal['take_profit'] > 0

            if has_sl or has_tp:
                # Validate bracket price logic before submitting
                bracket_valid = True
                if has_sl and has_tp:
                    if signal['side'] == 'buy':
                        # For buy: stop_loss < current price < take_profit
                        if signal['stop_loss'] >= signal.get('take_profit', float('inf')):
                            bracket_valid = False
                    elif signal['side'] == 'sell':
                        # For sell: take_profit < current price < stop_loss
                        if signal['take_profit'] >= signal.get('stop_loss', float('inf')):
                            bracket_valid = False

                if bracket_valid:
                    order_data['order_class'] = 'bracket'
                    if has_sl:
                        order_data['stop_loss'] = {
                            'stop_price': signal['stop_loss']
                        }
                    if has_tp:
                        order_data['take_profit'] = {
                            'limit_price': signal['take_profit']
                        }
                else:
                    self.logger.warning(
                        f"Invalid bracket prices for {signal['symbol']} "
                        f"(SL={signal.get('stop_loss')}, TP={signal.get('take_profit')}), "
                        f"submitting without bracket"
                    )

            return order_data

        except Exception as e:
            self.logger.error(f"Error preparing order: {e}")
            return None

    def _submit_to_broker(self, order_data: Dict) -> Optional[str]:
        """Submit order to broker API."""
        try:
            if self.alpaca_client:
                try:
                    order = self.alpaca_client.submit_order(**order_data)
                    return order.id
                except Exception as e:
                    self.logger.error(f"Broker submission error: {e}")
                    return None

            # Simulation mode — immediately fill market orders
            if self.config.get('simulation_mode', True):
                order_id = f"SIM_{uuid.uuid4().hex[:12]}"
                self.logger.info(f"Simulated order: {order_id}")

                # Schedule simulated fill for market orders
                if order_data.get('type') == 'market':
                    self._schedule_simulated_fill(order_id, order_data)

                return order_id

            self.logger.error("No broker client available")
            return None

        except Exception as e:
            self.logger.error(f"Error submitting to broker: {e}")
            return None

    def _schedule_simulated_fill(self, order_id: str, order_data: Dict) -> None:
        """Simulate immediate fill for market orders in sim mode."""
        def _fill():
            time.sleep(0.5)  # Simulate brief execution delay
            with self._lock:
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    # Use a simulated price (last known or limit price)
                    sim_price = float(order_data.get('limit_price', 100.0))
                    order.filled_avg_price = sim_price
                    order.filled_at = datetime.now(timezone.utc)
                    self._update_position_from_fill(order)
                    del self.active_orders[order_id]
                    self._notify_order_callbacks(order)
                    self.logger.info(f"Simulated fill: {order_id} at {sim_price}")

        fill_thread = threading.Thread(target=_fill, daemon=True, name=f"SimFill-{order_id}")
        fill_thread.start()

    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """
        Check if a signal is a duplicate of a recent active order.
        Prevents double-submission from retries or repeated signals.
        """
        symbol = signal['symbol']
        side = signal['side']
        for order in self.active_orders.values():
            if (order.symbol == symbol and
                    order.side == side and
                    order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED] and
                    (datetime.now(timezone.utc) - order.submitted_at).total_seconds() < 60):
                return True
        return False

    def _register_order(self, order_id: str, order_data: Dict, signal: Dict) -> None:
        """Register order in tracking system."""
        order_record = OrderRecord(
            id=order_id,
            symbol=order_data['symbol'],
            side=order_data['side'],
            order_type=order_data['type'],
            quantity=float(order_data['qty']),
            filled_quantity=0.0,
            status=OrderStatus.SUBMITTED,
            client_order_id=order_data.get('client_order_id', ''),
            submitted_at=datetime.now(timezone.utc),
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            metadata={
                'signal': signal,
                'order_data': order_data
            }
        )

        self.active_orders[order_id] = order_record
        self.order_history.append(order_record)

        # Trim history if needed
        if len(self.order_history) > self.MAX_ORDER_HISTORY:
            self.order_history = self.order_history[-self.MAX_ORDER_HISTORY:]

    def _setup_trailing_stop(self, order_id: str, signal: Dict) -> None:
        """Configure trailing stop for order."""
        if not signal.get('trail_trigger'):
            return

        trailing_config = {
            'order_id': order_id,
            'symbol': signal['symbol'],
            'side': signal['side'],
            'trigger_price': signal['trail_trigger'],
            'offset': signal.get('trail_offset', 0.01),
            'active': False,
            'highest_price': None,
            'lowest_price': None,
            'created_at': datetime.now(timezone.utc)
        }

        self.trailing_stops[order_id] = trailing_config
        self.logger.info(f"Trailing stop configured for order {order_id}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled successfully
        """
        with self._lock:
            try:
                if order_id not in self.active_orders:
                    self.logger.warning(f"Order {order_id} not found")
                    return False

                order = self.active_orders[order_id]

                # Check if order can be cancelled
                if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    self.logger.warning(f"Order {order_id} cannot be cancelled, status: {order.status}")
                    return False

                # Cancel at broker
                if self.alpaca_client:
                    try:
                        self.alpaca_client.cancel_order(order_id)
                    except Exception as e:
                        self.logger.warning(f"Broker cancel error: {e}")

                # Update local status
                order.status = OrderStatus.CANCELLED
                order.cancelled_at = datetime.now(timezone.utc)
                order.last_update = datetime.now(timezone.utc)

                # Remove from active orders
                del self.active_orders[order_id]

                # Remove trailing stop if exists
                if order_id in self.trailing_stops:
                    del self.trailing_stops[order_id]

                self.logger.info(f"Order {order_id} cancelled")

                # Notify callbacks
                self._notify_order_callbacks(order)

                return True

            except Exception as e:
                self.logger.error(f"Error cancelling order {order_id}: {e}")
                return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order."""
        with self._lock:
            # Check active orders
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                return self._order_to_dict(order)

            # Check history
            for order in reversed(self.order_history):
                if order.id == order_id:
                    return self._order_to_dict(order)

            return None

    def _order_to_dict(self, order: OrderRecord) -> Dict:
        """Convert OrderRecord to dictionary."""
        return {
            'id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'type': order.order_type,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'filled_avg_price': order.filled_avg_price,
            'status': order.status.value,
            'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
            'filled_at': order.filled_at.isoformat() if order.filled_at else None,
            'cancelled_at': order.cancelled_at.isoformat() if order.cancelled_at else None,
            'stop_loss': order.stop_loss,
            'take_profit': order.take_profit,
            'error': order.error_message
        }

    def get_positions(self) -> Dict:
        """Get all active positions."""
        with self._lock:
            return self.positions.copy()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol."""
        with self._lock:
            return self.positions.get(symbol)

    def start_monitoring(self) -> None:
        """Start order and position monitoring thread."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="OrderMonitor"
        )
        self.monitor_thread.start()
        self.logger.info("Order monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Order monitoring stopped")

    def _monitoring_worker(self) -> None:
        """Worker thread for continuous monitoring."""
        while self.running:
            try:
                # Update order statuses
                self._update_orders_status()

                # Update positions
                self._update_positions()

                # Process trailing stops
                self._process_trailing_stops()

                # Update daily P&L
                self._update_daily_performance()

                time.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                time.sleep(10)

    def _update_orders_status(self) -> None:
        """Update status of all active orders from broker."""
        if not self.alpaca_client:
            return

        with self._lock:
            for order_id in list(self.active_orders.keys()):
                try:
                    broker_order = self.alpaca_client.get_order(order_id)
                    order = self.active_orders[order_id]

                    # Update status
                    status_map = {
                        'new': OrderStatus.SUBMITTED,
                        'partially_filled': OrderStatus.PARTIALLY_FILLED,
                        'filled': OrderStatus.FILLED,
                        'cancelled': OrderStatus.CANCELLED,
                        'rejected': OrderStatus.REJECTED,
                        'expired': OrderStatus.EXPIRED,
                        'stopped': OrderStatus.CANCELLED
                    }

                    order.status = status_map.get(
                        broker_order.status.lower(),
                        OrderStatus.ERROR
                    )
                    order.filled_quantity = float(broker_order.filled_qty or 0)
                    order.filled_avg_price = float(broker_order.filled_avg_price or 0)
                    order.last_update = datetime.now(timezone.utc)

                    # Handle filled order
                    if order.status == OrderStatus.FILLED:
                        order.filled_at = datetime.now(timezone.utc)
                        self._update_position_from_fill(order)
                        del self.active_orders[order_id]
                        self._notify_order_callbacks(order)

                except Exception as e:
                    self.logger.error(f"Error updating order {order_id}: {e}")

    def _update_positions(self) -> None:
        """Update positions from broker."""
        if not self.alpaca_client:
            return

        with self._lock:
            try:
                broker_positions = self.alpaca_client.list_positions()
                current_symbols = set()

                for pos in broker_positions:
                    symbol = pos.symbol
                    current_symbols.add(symbol)

                    self.positions[symbol] = {
                        'symbol': symbol,
                        'quantity': float(pos.qty),
                        'side': 'long' if float(pos.qty) > 0 else 'short',
                        'avg_price': float(pos.avg_entry_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'last_update': datetime.now(timezone.utc)
                    }

                # Remove closed positions
                for symbol in list(self.positions.keys()):
                    if symbol not in current_symbols:
                        del self.positions[symbol]

            except Exception as e:
                self.logger.error(f"Error updating positions: {e}")

    def _update_position_from_fill(self, order: OrderRecord) -> None:
        """Update position based on filled order."""
        with self._lock:
            symbol = order.symbol
            quantity = order.filled_quantity
            price = order.filled_avg_price

            if symbol not in self.positions:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0,
                    'total_cost': 0
                }

            position = self.positions[symbol]

            if order.side == 'buy':
                new_quantity = position['quantity'] + quantity
                new_total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                position['avg_price'] = new_total_cost / new_quantity if new_quantity != 0 else 0
                position['quantity'] = new_quantity
            else:  # sell
                position['quantity'] -= quantity
                if position['quantity'] <= 0:
                    position['quantity'] = 0
                    position['avg_price'] = 0

            position['last_update'] = datetime.now(timezone.utc)

            # Notify callbacks
            self._notify_position_callbacks(position)

    def _process_trailing_stops(self) -> None:
        """Process active trailing stops."""
        with self._lock:
            for order_id, config in list(self.trailing_stops.items()):
                try:
                    symbol = config['symbol']
                    current_price = self._get_current_price(symbol)

                    if not current_price:
                        continue

                    # Check activation
                    if not config['active']:
                        if (config['side'] == 'buy' and current_price >= config['trigger_price']) or \
                           (config['side'] == 'sell' and current_price <= config['trigger_price']):
                            config['active'] = True
                            config['highest_price'] = current_price
                            config['lowest_price'] = current_price
                            self.logger.info(f"Trailing stop activated for {symbol}")

                        continue

                    # Update price extremes
                    if config['side'] == 'buy':
                        if current_price > config['highest_price']:
                            config['highest_price'] = current_price

                        stop_price = config['highest_price'] * (1 - config['offset'])
                        if current_price <= stop_price:
                            self._execute_trailing_stop(order_id, config, current_price)
                    else:  # sell
                        if current_price < config['lowest_price']:
                            config['lowest_price'] = current_price

                        stop_price = config['lowest_price'] * (1 + config['offset'])
                        if current_price >= stop_price:
                            self._execute_trailing_stop(order_id, config, current_price)

                except Exception as e:
                    self.logger.error(f"Error processing trailing stop {order_id}: {e}")

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.alpaca_client:
                quote = self.alpaca_client.get_latest_quote(symbol)
                return (quote.bid_price + quote.ask_price) / 2

            return None

        except Exception:
            return None

    def _execute_trailing_stop(self, order_id: str, config: Dict, current_price: float) -> None:
        """
        Execute a trailing stop order.
        Submits directly to broker, bypassing market hours and risk limits,
        because trailing stops are protective exits that must execute.
        """
        try:
            symbol = config['symbol']
            position = self.positions.get(symbol, {})
            qty = abs(position.get('quantity', 0))

            if qty > 0:
                close_side = 'sell' if config['side'] == 'buy' else 'buy'
                order_data = {
                    'symbol': symbol.upper(),
                    'qty': str(qty),
                    'side': close_side,
                    'type': 'market',
                    'time_in_force': 'day',
                    'client_order_id': str(uuid.uuid4())
                }

                result_id = self._submit_to_broker(order_data)
                if result_id:
                    self._register_order(result_id, order_data, {
                        'symbol': symbol, 'side': close_side,
                        'type': 'market', 'quantity': qty
                    })
                    self.logger.info(
                        f"Trailing stop executed for {symbol} at {current_price}, order {result_id}"
                    )
                else:
                    self.logger.error(
                        f"Trailing stop FAILED for {symbol} — position may be unprotected"
                    )

            # Remove trailing stop
            if order_id in self.trailing_stops:
                del self.trailing_stops[order_id]

        except Exception as e:
            self.logger.error(f"Error executing trailing stop: {e}")

    def _update_daily_performance(self) -> None:
        """Update daily performance metrics."""
        with self._lock:
            # Reset daily count if new day
            current_date = datetime.now(timezone.utc).date()
            if current_date > self.daily_reset_time:
                self.daily_order_count = 0
                self.daily_reset_time = current_date

            # Calculate total unrealized P&L
            total_unrealized = sum(
                pos.get('unrealized_pnl', 0) for pos in self.positions.values()
            )

            # Record performance
            self.performance_history.append({
                'timestamp': datetime.now(timezone.utc),
                'unrealized_pnl': total_unrealized,
                'positions_count': len(self.positions),
                'active_orders': len(self.active_orders)
            })

    def _increment_daily_count(self) -> None:
        """Increment daily order count."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.daily_reset_time:
            self.daily_order_count = 0
            self.daily_reset_time = current_date

        self.daily_order_count += 1

    def _notify_order_callbacks(self, order: OrderRecord) -> None:
        """Notify registered order callbacks."""
        for callback in self.order_callbacks:
            try:
                callback(self._order_to_dict(order))
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")

    def _notify_position_callbacks(self, position: Dict) -> None:
        """Notify registered position callbacks."""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                self.logger.error(f"Error in position callback: {e}")

    def add_order_callback(self, callback: Callable) -> None:
        """Register a callback for order events."""
        self.order_callbacks.append(callback)

    def add_position_callback(self, callback: Callable) -> None:
        """Register a callback for position events."""
        self.position_callbacks.append(callback)

    def get_trading_summary(self) -> Dict:
        """Get summary of trading activity."""
        with self._lock:
            return {
                'active_orders': len(self.active_orders),
                'active_positions': len(self.positions),
                'daily_orders': self.daily_order_count,
                'trailing_stops': len(self.trailing_stops),
                'total_orders_history': len(self.order_history),
                'last_update': datetime.now(timezone.utc)
            }
