"""
NOCTURNA Trading System - Paper Trading Routes
REST API endpoints for the paper trading engine.
"""

import logging
from datetime import UTC, datetime

from flask import Blueprint, g, jsonify, request

from src.core.paper_trading import PaperTradingEngine
from src.middleware.auth import require_auth, require_trading_permissions

paper_bp = Blueprint("paper_trading", __name__)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level engine instance (lazy-initialized via init_paper_engine)
# ---------------------------------------------------------------------------
_engine: PaperTradingEngine | None = None


def init_paper_engine(config: dict | None = None) -> PaperTradingEngine:
    """Initialize the paper trading engine singleton."""
    global _engine
    if _engine is None:
        _engine = PaperTradingEngine(config or {})
    return _engine


def get_engine() -> PaperTradingEngine:
    """Return the current engine instance or raise."""
    if _engine is None:
        raise RuntimeError("Paper trading engine not initialized. Call init_paper_engine() first.")
    return _engine


def _response(success: bool, data=None, error: str | None = None, status_code: int = 200):
    """Standardized response helper."""
    body: dict = {
        "success": success,
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": getattr(g, "request_id", None),
    }
    if data is not None:
        body["data"] = data
    if error is not None:
        body["error"] = error
    return jsonify(body), status_code


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@paper_bp.route("/start", methods=["POST"])
@require_auth
@require_trading_permissions
def paper_start():
    """Start a paper trading session."""
    try:
        engine = get_engine()
        result = engine.start()
        if result["success"]:
            return _response(True, data=result)
        return _response(False, error=result.get("error", "Unknown error"), status_code=409)
    except Exception as e:
        logger.error(f"Paper start error: {e}")
        return _response(False, error="Failed to start paper trading", status_code=500)


@paper_bp.route("/stop", methods=["POST"])
@require_auth
@require_trading_permissions
def paper_stop():
    """Stop the paper trading session (preserves state)."""
    try:
        engine = get_engine()
        result = engine.stop()
        if result["success"]:
            return _response(True, data=result)
        return _response(False, error=result.get("error", "Unknown error"), status_code=409)
    except Exception as e:
        logger.error(f"Paper stop error: {e}")
        return _response(False, error="Failed to stop paper trading", status_code=500)


@paper_bp.route("/reset", methods=["POST"])
@require_auth
@require_trading_permissions
def paper_reset():
    """Reset paper trading to initial state. Clears all history."""
    try:
        engine = get_engine()
        result = engine.reset()
        return _response(True, data=result)
    except Exception as e:
        logger.error(f"Paper reset error: {e}")
        return _response(False, error="Failed to reset paper trading", status_code=500)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

@paper_bp.route("/orders", methods=["POST"])
@require_auth
@require_trading_permissions
def paper_submit_order():
    """
    Submit a paper trading order.

    JSON body:
        symbol: str (required)
        side: "buy" | "sell" (required)
        order_type: "market" | "limit" | "stop" | "stop_limit" (required)
        quantity: float > 0 (required)
        limit_price: float (required for limit / stop_limit)
        stop_price: float (required for stop / stop_limit)
    """
    try:
        data = request.get_json()
        if not data:
            return _response(False, error="Request body required", status_code=400)

        required = ["symbol", "side", "order_type", "quantity"]
        missing = [f for f in required if f not in data]
        if missing:
            return _response(
                False,
                error=f"Missing required fields: {', '.join(missing)}",
                status_code=400,
            )

        engine = get_engine()
        result = engine.submit_order(
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            order_type=str(data["order_type"]),
            quantity=float(data["quantity"]),
            limit_price=float(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") else None,
        )

        if result["success"]:
            return _response(True, data=result, status_code=201)
        return _response(False, error=result.get("error", "Order rejected"), status_code=400)

    except (ValueError, TypeError) as e:
        return _response(False, error=f"Invalid input: {e}", status_code=400)
    except Exception as e:
        logger.error(f"Paper order error: {e}")
        return _response(False, error="Failed to submit order", status_code=500)


@paper_bp.route("/orders/<order_id>", methods=["DELETE"])
@require_auth
@require_trading_permissions
def paper_cancel_order(order_id: str):
    """Cancel a pending paper order."""
    try:
        engine = get_engine()
        result = engine.cancel_order(order_id)
        if result["success"]:
            return _response(True, data=result)
        return _response(False, error=result.get("error", "Not found"), status_code=404)
    except Exception as e:
        logger.error(f"Paper cancel error: {e}")
        return _response(False, error="Failed to cancel order", status_code=500)


@paper_bp.route("/orders", methods=["GET"])
@require_auth
def paper_get_orders():
    """Get paper order history."""
    try:
        engine = get_engine()
        status = request.args.get("status")
        limit = min(int(request.args.get("limit", 100)), 1000)
        orders = engine.get_orders(status=status, limit=limit)
        return _response(True, data={"orders": orders, "count": len(orders)})
    except Exception as e:
        logger.error(f"Paper orders query error: {e}")
        return _response(False, error="Failed to fetch orders", status_code=500)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@paper_bp.route("/positions", methods=["GET"])
@require_auth
def paper_get_positions():
    """Get all open paper positions."""
    try:
        engine = get_engine()
        positions = engine.get_positions()
        return _response(True, data={"positions": positions, "count": len(positions)})
    except Exception as e:
        logger.error(f"Paper positions error: {e}")
        return _response(False, error="Failed to fetch positions", status_code=500)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@paper_bp.route("/trades", methods=["GET"])
@require_auth
def paper_get_trades():
    """Get paper trade history."""
    try:
        engine = get_engine()
        symbol = request.args.get("symbol")
        limit = min(int(request.args.get("limit", 100)), 1000)
        trades = engine.get_trades(symbol=symbol, limit=limit)
        return _response(True, data={"trades": trades, "count": len(trades)})
    except Exception as e:
        logger.error(f"Paper trades error: {e}")
        return _response(False, error="Failed to fetch trades", status_code=500)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@paper_bp.route("/portfolio", methods=["GET"])
@require_auth
def paper_portfolio():
    """Get full paper trading portfolio summary."""
    try:
        engine = get_engine()
        summary = engine.get_portfolio_summary()
        return _response(True, data=summary)
    except Exception as e:
        logger.error(f"Paper portfolio error: {e}")
        return _response(False, error="Failed to fetch portfolio", status_code=500)


@paper_bp.route("/equity-curve", methods=["GET"])
@require_auth
def paper_equity_curve():
    """Get paper trading equity curve data."""
    try:
        engine = get_engine()
        curve = engine.get_equity_curve()
        return _response(True, data={"equity_curve": curve, "points": len(curve)})
    except Exception as e:
        logger.error(f"Paper equity curve error: {e}")
        return _response(False, error="Failed to fetch equity curve", status_code=500)


# ---------------------------------------------------------------------------
# Market price injection (for external price feeds)
# ---------------------------------------------------------------------------

@paper_bp.route("/prices", methods=["POST"])
@require_auth
@require_trading_permissions
def paper_update_prices():
    """
    Update market prices and process pending orders.

    JSON body:
        prices: {"AAPL": 185.50, "MSFT": 420.10, ...}
    """
    try:
        data = request.get_json()
        if not data or "prices" not in data:
            return _response(False, error="prices dict required", status_code=400)

        engine = get_engine()
        engine.update_market_prices(data["prices"])
        filled = engine.process_pending_orders()

        return _response(True, data={
            "prices_updated": len(data["prices"]),
            "orders_filled": len(filled),
            "filled_orders": filled,
        })
    except Exception as e:
        logger.error(f"Paper price update error: {e}")
        return _response(False, error="Failed to update prices", status_code=500)
