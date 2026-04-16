# FILE LOCATION: src/tests/test_paper_trading.py
"""
Tests for the paper trading engine and API routes.
Covers: order lifecycle, position tracking, P&L, equity curve, API endpoints.
"""

import pytest

from src.core.paper_trading import (
    PaperTradingEngine,
)


@pytest.fixture()
def engine():
    """Fresh paper trading engine for each test."""
    e = PaperTradingEngine({
        "initial_capital": 100_000,
        "commission_rate": 0.001,
        "slippage_rate": 0.0,  # Zero slippage for deterministic tests
        "max_position_pct": 0.25,
    })
    e.start()
    e.update_market_price("AAPL", 150.0)
    e.update_market_price("MSFT", 400.0)
    return e


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start(self):
        e = PaperTradingEngine()
        result = e.start()
        assert result["success"] is True
        assert e.is_active is True

    def test_double_start_fails(self):
        e = PaperTradingEngine()
        e.start()
        result = e.start()
        assert result["success"] is False

    def test_stop(self):
        e = PaperTradingEngine()
        e.start()
        result = e.stop()
        assert result["success"] is True
        assert e.is_active is False

    def test_reset_clears_everything(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.reset()
        assert engine.cash == 100_000
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert engine.is_active is False


# ---------------------------------------------------------------------------
# Order submission
# ---------------------------------------------------------------------------

class TestOrderSubmission:
    def test_market_buy_fills_immediately(self, engine):
        result = engine.submit_order("AAPL", "buy", "market", 10)
        assert result["success"] is True
        order = result["order"]
        assert order["status"] == "FILLED"
        assert order["filled_quantity"] == 10

    def test_market_sell_fills_immediately(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        result = engine.submit_order("AAPL", "sell", "market", 10)
        assert result["success"] is True
        assert result["order"]["status"] == "FILLED"

    def test_limit_order_queued(self, engine):
        result = engine.submit_order("AAPL", "buy", "limit", 10, limit_price=140.0)
        assert result["success"] is True
        assert result["order"]["status"] == "PENDING"

    def test_insufficient_capital_rejected(self, engine):
        # At $150 and 25% concentration limit, max ~166 shares.
        # 10,000 shares exceeds both concentration and buying power.
        result = engine.submit_order("AAPL", "buy", "market", 10_000)
        assert result["success"] is False
        assert "error" in result

    def test_invalid_side_rejected(self, engine):
        result = engine.submit_order("AAPL", "invalid", "market", 10)
        assert result["success"] is False

    def test_invalid_order_type_rejected(self, engine):
        result = engine.submit_order("AAPL", "buy", "bogus", 10)
        assert result["success"] is False

    def test_zero_quantity_rejected(self, engine):
        result = engine.submit_order("AAPL", "buy", "market", 0)
        assert result["success"] is False

    def test_limit_without_price_rejected(self, engine):
        result = engine.submit_order("AAPL", "buy", "limit", 10)
        assert result["success"] is False

    def test_stop_without_price_rejected(self, engine):
        result = engine.submit_order("AAPL", "buy", "stop", 10)
        assert result["success"] is False

    def test_order_when_inactive_rejected(self):
        e = PaperTradingEngine()
        e.update_market_price("AAPL", 150.0)
        result = e.submit_order("AAPL", "buy", "market", 10)
        assert result["success"] is False

    def test_concentration_limit(self, engine):
        """Max position concentration (25%) should block oversized orders."""
        # 25% of 100k = 25k. At $150, max ~166 shares.
        result = engine.submit_order("AAPL", "buy", "market", 200)
        assert result["success"] is False
        assert "concentration" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Order cancellation
# ---------------------------------------------------------------------------

class TestOrderCancellation:
    def test_cancel_pending_order(self, engine):
        result = engine.submit_order("AAPL", "buy", "limit", 10, limit_price=140.0)
        order_id = result["order"]["id"]
        cancel_result = engine.cancel_order(order_id)
        assert cancel_result["success"] is True
        assert cancel_result["order"]["status"] == "CANCELLED"

    def test_cancel_nonexistent_order(self, engine):
        result = engine.cancel_order("fake_order_id")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

class TestPositionTracking:
    def test_buy_creates_position(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        positions = engine.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "AAPL"
        assert positions[0]["quantity"] == 10

    def test_sell_closes_position(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.submit_order("AAPL", "sell", "market", 10)
        positions = engine.get_positions()
        assert len(positions) == 0  # Fully closed

    def test_partial_close(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.submit_order("AAPL", "sell", "market", 5)
        positions = engine.get_positions()
        assert len(positions) == 1
        assert positions[0]["quantity"] == 5


# ---------------------------------------------------------------------------
# P&L calculations
# ---------------------------------------------------------------------------

class TestPnL:
    def test_cash_decreases_on_buy(self, engine):
        initial_cash = engine.cash
        engine.submit_order("AAPL", "buy", "market", 10)
        assert engine.cash < initial_cash

    def test_cash_increases_on_sell(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        cash_after_buy = engine.cash
        engine.submit_order("AAPL", "sell", "market", 10)
        assert engine.cash > cash_after_buy

    def test_commission_applied(self, engine):
        """Commission should reduce effective proceeds."""
        initial = engine.cash
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.submit_order("AAPL", "sell", "market", 10)
        # With 0 slippage, round-trip at same price should lose only commissions
        assert engine.cash < initial  # Lost commission

    def test_realized_pnl_on_profitable_trade(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        # Price goes up
        engine.update_market_price("AAPL", 160.0)
        engine.submit_order("AAPL", "sell", "market", 10)
        trades = engine.get_trades()
        # The sell trade should show positive PnL
        sell_trade = [t for t in trades if t["side"] == "sell"]
        assert len(sell_trade) >= 1
        assert sell_trade[-1]["pnl"] > 0

    def test_unrealized_pnl_updates(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.update_market_price("AAPL", 160.0)
        positions = engine.get_positions()
        assert positions[0]["unrealized_pnl"] > 0


# ---------------------------------------------------------------------------
# Pending order processing
# ---------------------------------------------------------------------------

class TestPendingOrders:
    def test_limit_buy_fills_when_price_drops(self, engine):
        engine.submit_order("AAPL", "buy", "limit", 10, limit_price=145.0)
        assert len(engine.pending_orders) == 1

        engine.update_market_price("AAPL", 144.0)
        filled = engine.process_pending_orders()
        assert len(filled) == 1
        assert filled[0]["status"] == "FILLED"

    def test_limit_buy_does_not_fill_above(self, engine):
        engine.submit_order("AAPL", "buy", "limit", 10, limit_price=145.0)
        engine.update_market_price("AAPL", 146.0)
        filled = engine.process_pending_orders()
        assert len(filled) == 0

    def test_stop_sell_fills_when_price_drops(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.submit_order("AAPL", "sell", "stop", 10, stop_price=145.0)
        engine.update_market_price("AAPL", 144.0)
        filled = engine.process_pending_orders()
        assert len(filled) == 1


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    def test_summary_structure(self, engine):
        summary = engine.get_portfolio_summary()
        assert "equity" in summary
        assert "cash" in summary
        assert "total_return" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary

    def test_equity_equals_cash_when_no_positions(self, engine):
        summary = engine.get_portfolio_summary()
        assert summary["equity"] == summary["cash"]

    def test_equity_curve_records_points(self, engine):
        engine.submit_order("AAPL", "buy", "market", 10)
        engine.process_pending_orders()
        curve = engine.get_equity_curve()
        assert len(curve) >= 1


# ---------------------------------------------------------------------------
# API route integration tests
# ---------------------------------------------------------------------------

class TestPaperTradingRoutes:
    def test_start_endpoint(self, client, auth_headers):
        # First reset to clean state
        client.post("/api/paper/reset", headers=auth_headers)
        resp = client.post("/api/paper/start", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_stop_endpoint(self, client, auth_headers):
        client.post("/api/paper/reset", headers=auth_headers)
        client.post("/api/paper/start", headers=auth_headers)
        resp = client.post("/api/paper/stop", headers=auth_headers)
        assert resp.status_code == 200

    def test_portfolio_endpoint(self, client, auth_headers):
        resp = client.get("/api/paper/portfolio", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data

    def test_submit_order_endpoint(self, client, auth_headers):
        client.post("/api/paper/reset", headers=auth_headers)
        client.post("/api/paper/start", headers=auth_headers)
        # Inject price first
        client.post("/api/paper/prices", headers=auth_headers, json={
            "prices": {"AAPL": 150.0}
        })
        resp = client.post("/api/paper/orders", headers=auth_headers, json={
            "symbol": "AAPL",
            "side": "buy",
            "order_type": "market",
            "quantity": 5,
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True

    def test_positions_endpoint(self, client, auth_headers):
        resp = client.get("/api/paper/positions", headers=auth_headers)
        assert resp.status_code == 200

    def test_trades_endpoint(self, client, auth_headers):
        resp = client.get("/api/paper/trades", headers=auth_headers)
        assert resp.status_code == 200

    def test_orders_endpoint(self, client, auth_headers):
        resp = client.get("/api/paper/orders", headers=auth_headers)
        assert resp.status_code == 200

    def test_unauth_returns_401(self, client):
        resp = client.get("/api/paper/portfolio")
        assert resp.status_code == 401
