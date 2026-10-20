"""
Tests for trading API responses that must report real state (Phase 0).
"""

import pytest

from src.core.trading_engine import TradingEngineState
from src.routes import trading as trading_routes


@pytest.fixture()
def engine():
    """Fresh trading engine singleton per test."""
    trading_routes._trading_engine = None
    eng = trading_routes.get_trading_engine()
    yield eng
    trading_routes._trading_engine = None


class TestNoFabricatedData:
    def test_equity_curve_is_empty_without_history(self, client, auth_headers, engine):
        resp = client.get("/api/trading/equity-curve", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.get_json()["data"]["equity_curve"] == []

    def test_optimize_reports_not_implemented(self, client, admin_headers, engine):
        resp = client.post("/api/trading/optimize", headers=admin_headers)
        assert resp.status_code == 501
        assert resp.get_json()["success"] is False


class TestEmergencyStop:
    @pytest.mark.parametrize("close_all", [True, False])
    def test_close_positions_flag_is_honoured(self, client, admin_headers, engine, monkeypatch, close_all):
        calls = []
        monkeypatch.setattr(engine, "_close_all_positions", lambda: calls.append(True))

        resp = client.post(
            "/api/trading/emergency-stop",
            json={"reason": "test", "close_all_positions": close_all},
            headers=admin_headers,
        )

        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["state"] == TradingEngineState.EMERGENCY_STOP.value
        assert data["close_orders_submitted"] is close_all
        assert bool(calls) is close_all


class TestControlResponses:
    def test_start_reports_state_as_string(self, client, admin_headers, engine, monkeypatch):
        def fake_start():
            engine.state = TradingEngineState.RUNNING
            return True

        monkeypatch.setattr(engine, "start", fake_start)

        resp = client.post("/api/trading/control/start", headers=admin_headers)

        assert resp.status_code == 200
        assert resp.get_json()["data"]["state"] == "RUNNING"
