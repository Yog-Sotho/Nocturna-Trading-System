"""
NOCTURNA Trading System - Paper Trading Engine
Production-grade simulated execution engine for risk-free strategy validation.

This module provides a complete paper trading environment that mirrors real
broker execution without risking actual capital. It simulates:
  - Order matching with configurable slippage and latency
  - Position tracking with mark-to-market P&L
  - Portfolio equity curve generation
  - Commission deduction
  - Fill simulation using last-known market prices
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PaperOrderStatus(Enum):
    """Paper order lifecycle states."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PaperOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class PaperOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PaperOrder:
    """Immutable record of a paper order."""
    id: str
    symbol: str
    side: PaperOrderSide
    order_type: PaperOrderType
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    filled_at: datetime | None = None
    reject_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "reject_reason": self.reject_reason,
        }


@dataclass
class PaperPosition:
    """Tracks a single symbol position in the paper portfolio."""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime | None = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry_price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": round(self.avg_entry_price, 4),
            "market_price": round(self.market_price, 4),
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


@dataclass
class PaperTrade:
    """Record of a completed fill (partial or full)."""
    id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    pnl: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": round(self.price, 4),
            "commission": round(self.commission, 4),
            "pnl": round(self.pnl, 2),
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTradingEngine:
    """
    Simulated execution engine for paper trading.

    Thread-safe. Maintains its own cash balance, positions, orders,
    and trade history independently of the real broker.

    Usage:
        engine = PaperTradingEngine({"initial_capital": 100_000})
        order = engine.submit_order("AAPL", "buy", "market", 10)
        engine.update_market_price("AAPL", 185.50)
        engine.process_pending_orders()
        print(engine.get_portfolio_summary())
    """

    MAX_ORDER_HISTORY = 50_000
    MAX_TRADE_HISTORY = 100_000

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PaperTradingEngine")

        # Capital
        self.initial_capital: float = float(config.get("initial_capital", 100_000))
        self.cash: float = self.initial_capital

        # Execution model
        self.commission_rate: float = float(config.get("commission_rate", 0.001))
        self.slippage_rate: float = float(config.get("slippage_rate", 0.0005))
        self.max_position_pct: float = float(config.get("max_position_pct", 0.25))

        # State
        self.positions: dict[str, PaperPosition] = {}
        self.orders: dict[str, PaperOrder] = {}
        self.pending_orders: dict[str, PaperOrder] = {}
        self.trades: list[PaperTrade] = []
        self.equity_curve: list[dict] = []
        self._market_prices: dict[str, float] = {}
        self._order_counter: int = 0

        # Drawdown tracking
        self.peak_equity: float = self.initial_capital
        self.max_drawdown: float = 0.0

        # Thread safety
        self._lock = threading.RLock()

        # Active flag
        self.is_active: bool = False
        self.started_at: datetime | None = None

        self.logger.info(
            f"Paper Trading Engine initialized | Capital: ${self.initial_capital:,.2f} | "
            f"Commission: {self.commission_rate:.2%} | Slippage: {self.slippage_rate:.2%}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> dict:
        """Start the paper trading session."""
        with self._lock:
            if self.is_active:
                return {"success": False, "error": "Paper trading already active"}
            self.is_active = True
            self.started_at = datetime.now(UTC)
            self.logger.info("Paper trading session started")
            return {"success": True, "started_at": self.started_at.isoformat()}

    def stop(self) -> dict:
        """Stop the paper trading session (preserves state)."""
        with self._lock:
            if not self.is_active:
                return {"success": False, "error": "Paper trading not active"}
            self.is_active = False
            # Cancel all pending orders
            cancelled = 0
            for order in list(self.pending_orders.values()):
                order.status = PaperOrderStatus.CANCELLED
                self.orders[order.id] = order
                cancelled += 1
            self.pending_orders.clear()
            self.logger.info(f"Paper trading stopped | Cancelled {cancelled} pending orders")
            return {"success": True, "cancelled_orders": cancelled}

    def reset(self) -> dict:
        """Reset paper trading to initial state. Clears all history."""
        with self._lock:
            self.cash = self.initial_capital
            self.positions.clear()
            self.orders.clear()
            self.pending_orders.clear()
            self.trades.clear()
            self.equity_curve.clear()
            self._market_prices.clear()
            self._order_counter = 0
            self.peak_equity = self.initial_capital
            self.max_drawdown = 0.0
            self.is_active = False
            self.started_at = None
            self.logger.info("Paper trading engine reset to initial state")
            return {"success": True, "capital": self.initial_capital}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def update_market_price(self, symbol: str, price: float) -> None:
        """Update the latest market price for a symbol."""
        if price <= 0:
            return
        with self._lock:
            self._market_prices[symbol] = price
            # Update position mark-to-market
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.market_price = price
                if pos.quantity > 0:
                    pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity
                elif pos.quantity < 0:
                    pos.unrealized_pnl = (pos.avg_entry_price - price) * abs(pos.quantity)

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Bulk update market prices."""
        for symbol, price in prices.items():
            self.update_market_price(symbol, price)

    def get_market_price(self, symbol: str) -> float | None:
        """Get the last known market price for a symbol."""
        return self._market_prices.get(symbol)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict:
        """
        Submit a paper order.

        Args:
            symbol: Ticker symbol (e.g. "AAPL")
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", or "stop_limit"
            quantity: Number of shares (must be > 0)
            limit_price: Required for limit / stop_limit orders
            stop_price: Required for stop / stop_limit orders

        Returns:
            Dict with order details or error
        """
        with self._lock:
            if not self.is_active:
                return {"success": False, "error": "Paper trading is not active"}

            # --- Validate inputs ---
            try:
                side_enum = PaperOrderSide(side.lower())
            except ValueError:
                return {"success": False, "error": f"Invalid side: {side}"}

            try:
                type_enum = PaperOrderType(order_type.lower())
            except ValueError:
                return {"success": False, "error": f"Invalid order type: {order_type}"}

            if quantity <= 0:
                return {"success": False, "error": "Quantity must be > 0"}

            symbol = symbol.upper().strip()
            if not symbol or len(symbol) > 10:
                return {"success": False, "error": f"Invalid symbol: {symbol!r}"}

            if type_enum in (PaperOrderType.LIMIT, PaperOrderType.STOP_LIMIT) and (limit_price is None or limit_price <= 0):
                return {"success": False, "error": "Limit price required and must be > 0"}

            if type_enum in (PaperOrderType.STOP, PaperOrderType.STOP_LIMIT) and (stop_price is None or stop_price <= 0):
                return {"success": False, "error": "Stop price required and must be > 0"}

            # --- Pre-trade risk checks ---
            if side_enum == PaperOrderSide.BUY:
                estimated_cost = quantity * (
                    limit_price or self._market_prices.get(symbol, 0)
                )
                if estimated_cost <= 0:
                    return {
                        "success": False,
                        "error": f"No market price available for {symbol}. "
                                 f"Call update_market_price() first.",
                    }
                # Position concentration check
                total_equity = self._calculate_equity()
                if total_equity > 0 and estimated_cost / total_equity > self.max_position_pct:
                    return {
                        "success": False,
                        "error": f"Order exceeds max position concentration "
                                 f"({self.max_position_pct:.0%} of equity)",
                    }
                if estimated_cost > self.cash:
                    return {"success": False, "error": "Insufficient buying power"}

            # --- Create order ---
            self._order_counter += 1
            order_id = f"paper_{self._order_counter}_{uuid.uuid4().hex[:8]}"

            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                side=side_enum,
                order_type=type_enum,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
            )

            # Market orders attempt immediate fill
            if type_enum == PaperOrderType.MARKET:
                price = self._market_prices.get(symbol)
                if price:
                    self._execute_fill(order, price)
                else:
                    # Queue for fill when price arrives
                    self.pending_orders[order_id] = order
            else:
                self.pending_orders[order_id] = order

            self.orders[order_id] = order
            self._trim_history()

            self.logger.info(
                f"Paper order submitted: {order_id} | {side} {quantity} {symbol} "
                f"@ {order_type} | Status: {order.status.value}"
            )
            return {"success": True, "order": order.to_dict()}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending paper order."""
        with self._lock:
            order = self.pending_orders.pop(order_id, None)
            if order is None:
                return {"success": False, "error": f"Order {order_id} not found or not pending"}
            order.status = PaperOrderStatus.CANCELLED
            self.orders[order_id] = order
            self.logger.info(f"Paper order cancelled: {order_id}")
            return {"success": True, "order": order.to_dict()}

    # ------------------------------------------------------------------
    # Order processing
    # ------------------------------------------------------------------

    def process_pending_orders(self) -> list[dict]:
        """
        Attempt to fill all pending orders against current market prices.
        Call this after updating market prices.

        Returns:
            List of filled order dicts
        """
        filled = []
        with self._lock:
            for order_id in list(self.pending_orders.keys()):
                order = self.pending_orders[order_id]
                price = self._market_prices.get(order.symbol)
                if price is None:
                    continue

                fill_price = self._check_fill(order, price)
                if fill_price is not None:
                    self._execute_fill(order, fill_price)
                    del self.pending_orders[order_id]
                    filled.append(order.to_dict())

            # Update equity curve after processing
            self._record_equity_point()

        return filled

    def _check_fill(self, order: PaperOrder, current_price: float) -> float | None:
        """Determine if an order should fill at the given market price."""
        if order.order_type == PaperOrderType.MARKET:
            return current_price

        elif order.order_type == PaperOrderType.LIMIT:
            if order.side == PaperOrderSide.BUY and current_price <= order.limit_price:
                return min(current_price, order.limit_price)
            if order.side == PaperOrderSide.SELL and current_price >= order.limit_price:
                return max(current_price, order.limit_price)

        elif order.order_type == PaperOrderType.STOP:
            if order.side == PaperOrderSide.BUY and current_price >= order.stop_price:
                return current_price
            if order.side == PaperOrderSide.SELL and current_price <= order.stop_price:
                return current_price

        elif order.order_type == PaperOrderType.STOP_LIMIT:
            # Stop triggered, then limit applies
            if order.side == PaperOrderSide.BUY:
                if current_price >= order.stop_price and current_price <= order.limit_price:
                    return current_price
            else:
                if current_price <= order.stop_price and current_price >= order.limit_price:
                    return current_price

        return None

    def _execute_fill(self, order: PaperOrder, raw_price: float) -> None:
        """Execute a fill with slippage and commission applied."""
        # Apply slippage
        if order.side == PaperOrderSide.BUY:
            fill_price = raw_price * (1 + self.slippage_rate)
        else:
            fill_price = raw_price * (1 - self.slippage_rate)

        fill_price = round(fill_price, 4)
        commission = round(abs(order.quantity * fill_price * self.commission_rate), 4)
        now = datetime.now(UTC)

        # Calculate P&L for closing trades
        pnl = 0.0
        pos = self.positions.get(order.symbol)

        if order.side == PaperOrderSide.BUY:
            cost = order.quantity * fill_price + commission
            if cost > self.cash:
                order.status = PaperOrderStatus.REJECTED
                order.reject_reason = "Insufficient buying power after slippage"
                return
            self.cash -= cost

            # If short, this is a cover (closing)
            if pos and pos.quantity < 0:
                closed_qty = min(order.quantity, abs(pos.quantity))
                pnl = (pos.avg_entry_price - fill_price) * closed_qty
        else:
            proceeds = order.quantity * fill_price - commission
            self.cash += proceeds

            # If long, this is a sell (closing)
            if pos and pos.quantity > 0:
                closed_qty = min(order.quantity, pos.quantity)
                pnl = (fill_price - pos.avg_entry_price) * closed_qty

        # Update position
        self._update_position(order.symbol, order.side, order.quantity, fill_price, now)

        # Update order
        order.status = PaperOrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.filled_at = now

        # Record trade
        trade = PaperTrade(
            id=f"ptrade_{uuid.uuid4().hex[:12]}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            pnl=round(pnl, 2),
            timestamp=now,
        )
        self.trades.append(trade)

        self.logger.info(
            f"Paper fill: {order.symbol} {order.side.value} {order.quantity} "
            f"@ ${fill_price:.4f} | Commission: ${commission:.4f} | PnL: ${pnl:.2f}"
        )

    def _update_position(
        self,
        symbol: str,
        side: PaperOrderSide,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Update position after a fill."""
        if symbol not in self.positions:
            self.positions[symbol] = PaperPosition(symbol=symbol, opened_at=timestamp)

        pos = self.positions[symbol]

        if side == PaperOrderSide.BUY:
            new_qty = pos.quantity + quantity
        else:
            new_qty = pos.quantity - quantity

        if new_qty == 0:
            # Position fully closed
            if pos.quantity > 0:
                pos.realized_pnl += (price - pos.avg_entry_price) * min(quantity, abs(pos.quantity))
            elif pos.quantity < 0:
                pos.realized_pnl += (pos.avg_entry_price - price) * min(quantity, abs(pos.quantity))
            pos.quantity = 0
            pos.avg_entry_price = 0.0
            pos.unrealized_pnl = 0.0
            pos.opened_at = None
        elif (pos.quantity >= 0 and new_qty > 0) or (pos.quantity <= 0 and new_qty < 0):
            # Adding to existing direction — recalculate avg price
            if pos.quantity == 0:
                pos.avg_entry_price = price
                pos.opened_at = timestamp
            else:
                total_cost = abs(pos.quantity) * pos.avg_entry_price + quantity * price
                pos.avg_entry_price = total_cost / abs(new_qty)
            pos.quantity = new_qty
        else:
            # Partial close then flip — simplified: just set to new price
            if pos.quantity > 0:
                closed_qty = min(quantity, pos.quantity)
                pos.realized_pnl += (price - pos.avg_entry_price) * closed_qty
            elif pos.quantity < 0:
                closed_qty = min(quantity, abs(pos.quantity))
                pos.realized_pnl += (pos.avg_entry_price - price) * closed_qty
            pos.quantity = new_qty
            pos.avg_entry_price = price
            pos.opened_at = timestamp

        pos.market_price = price

    # ------------------------------------------------------------------
    # Portfolio queries
    # ------------------------------------------------------------------

    def _calculate_equity(self) -> float:
        """Calculate total portfolio equity (cash + position market values)."""
        equity = self.cash
        for pos in self.positions.values():
            if pos.quantity != 0:
                equity += pos.quantity * pos.market_price
        return equity

    def _record_equity_point(self) -> None:
        """Record a point on the equity curve and update drawdown."""
        equity = self._calculate_equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)

        self.equity_curve.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "drawdown": round(dd, 4),
        })

    def get_portfolio_summary(self) -> dict:
        """Get full portfolio summary."""
        with self._lock:
            equity = self._calculate_equity()
            total_return = (equity - self.initial_capital) / self.initial_capital
            total_realized = sum(p.realized_pnl for p in self.positions.values())
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())

            winning = sum(1 for t in self.trades if t.pnl > 0)
            losing = sum(1 for t in self.trades if t.pnl < 0)
            total_trades = len(self.trades)
            win_rate = winning / total_trades if total_trades > 0 else 0.0

            return {
                "is_active": self.is_active,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "initial_capital": self.initial_capital,
                "cash": round(self.cash, 2),
                "equity": round(equity, 2),
                "total_return": round(total_return, 4),
                "total_return_pct": f"{total_return:.2%}",
                "realized_pnl": round(total_realized, 2),
                "unrealized_pnl": round(total_unrealized, 2),
                "max_drawdown": round(self.max_drawdown, 4),
                "max_drawdown_pct": f"{self.max_drawdown:.2%}",
                "total_trades": total_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(win_rate, 4),
                "open_positions": len([p for p in self.positions.values() if p.quantity != 0]),
                "pending_orders": len(self.pending_orders),
                "total_commissions": round(sum(t.commission for t in self.trades), 2),
            }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        with self._lock:
            return [
                pos.to_dict()
                for pos in self.positions.values()
                if pos.quantity != 0
            ]

    def get_orders(self, status: str | None = None, limit: int = 100) -> list[dict]:
        """Get order history, optionally filtered by status."""
        with self._lock:
            orders = list(self.orders.values())
            if status:
                try:
                    status_enum = PaperOrderStatus(status.upper())
                    orders = [o for o in orders if o.status == status_enum]
                except ValueError:
                    pass
            # Most recent first
            orders.sort(key=lambda o: o.created_at, reverse=True)
            return [o.to_dict() for o in orders[:limit]]

    def get_trades(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        """Get trade history, optionally filtered by symbol."""
        with self._lock:
            trades = self.trades
            if symbol:
                trades = [t for t in trades if t.symbol == symbol.upper()]
            return [t.to_dict() for t in trades[-limit:]]

    def get_equity_curve(self) -> list[dict]:
        """Get the equity curve data."""
        with self._lock:
            return list(self.equity_curve)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Prevent unbounded memory growth."""
        if len(self.orders) > self.MAX_ORDER_HISTORY:
            # Keep only recent orders
            sorted_orders = sorted(
                self.orders.items(),
                key=lambda kv: kv[1].created_at,
                reverse=True,
            )
            self.orders = dict(sorted_orders[: self.MAX_ORDER_HISTORY])

        if len(self.trades) > self.MAX_TRADE_HISTORY:
            self.trades = self.trades[-self.MAX_TRADE_HISTORY:]
