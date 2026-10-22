"""
Tests for the trading API on the new engine (simulated broker, simulated clock).
"""

import pytest


def _start(client, admin_headers, runtime):
    resp = client.post("/api/v1/trading/control/start", headers=admin_headers)
    assert resp.status_code == 202, resp.get_json()
    assert runtime.wait_running().state == "running"


class TestWithoutAlpacaCredentials:
    def test_status_reports_unconfigured_and_stopped(self, client, auth_headers):
        data = client.get("/api/v1/trading/status", headers=auth_headers).get_json()["data"]
        assert data == {"configured": False, "state": "stopped"}

    def test_start_explains_what_is_missing(self, client, admin_headers):
        resp = client.post("/api/v1/trading/control/start", headers=admin_headers)
        assert resp.status_code == 503
        assert "ALPACA_API_KEY" in resp.get_json()["error"]

    def test_emergency_stop_reports_the_problem(self, client, admin_headers):
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers, json={"reason": "drill"})
        assert resp.status_code == 503

    @pytest.mark.parametrize("path", ["/api/v1/trading/positions", "/api/v1/trading/targets"])
    def test_live_reads_need_a_running_engine(self, client, auth_headers, path):
        assert client.get(path, headers=auth_headers).status_code == 409

    def test_history_reads_work_without_the_engine(self, client, auth_headers):
        for path in ("/api/v1/trading/orders", "/api/v1/trading/fills", "/api/v1/trading/equity-curve"):
            assert client.get(path, headers=auth_headers).status_code == 200

    def test_public_health(self, client):
        assert client.get("/api/v1/trading/health").get_json()["data"] == {"engine": "stopped"}


class TestEngineLifecycle:
    def test_start_status_and_double_start(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        data = client.get("/api/v1/trading/status", headers=auth_headers).get_json()["data"]
        assert data["state"] == "running" and data["configured"] is True
        assert data["equity"] == "100000" and data["open_orders"] == 2  # the 60/40 opening orders
        assert data["last_reconciliation"]["in_sync"] is True
        risk = data["risk"]
        assert risk["drawdown"] == "0" and risk["halt_kind"] is None and risk["cooling_down"] is False
        assert risk["orders_today"] == 2 and risk["drawdown_scale"] == "1"
        assert risk["predicted_volatility"] > 0 and 0 < float(risk["volatility_scale"]) <= 1
        assert client.post("/api/v1/trading/control/start", headers=admin_headers).status_code == 409

    def test_targets_and_working_orders(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        targets = client.get("/api/v1/trading/targets", headers=auth_headers).get_json()["data"]["targets"]
        assert targets == {"SPY": "0.600000", "IEF": "0.400000"}
        orders = client.get("/api/v1/trading/orders", headers=auth_headers).get_json()["data"]
        assert {o["symbol"] for o in orders["working"]} == {"SPY", "IEF"}
        assert {o["symbol"] for o in orders["history"]} == {"SPY", "IEF"}  # recorded in the database

    def test_only_admins_control_the_engine(self, client, auth_headers, sim_runtime):
        for path in ("/api/v1/trading/control/start", "/api/v1/trading/control/stop", "/api/v1/trading/control/resume"):
            assert client.post(path, headers=auth_headers).status_code == 403

    def test_stop(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        assert client.post("/api/v1/trading/control/stop", headers=admin_headers).status_code == 200
        assert client.get("/api/v1/trading/status", headers=auth_headers).get_json()["data"]["state"] == "stopped"


class TestManualOrders:
    def test_order_goes_through_the_risk_gate(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        resp = client.post("/api/v1/trading/orders", headers=auth_headers,
                           json={"symbol": "spy", "side": "buy", "qty": "1"})
        assert resp.status_code == 201, resp.get_json()
        order = resp.get_json()["data"]["order"]
        assert order["symbol"] == "SPY" and order["status"] == "new" and order["time_in_force"] == "day"
        too_big = client.post("/api/v1/trading/orders", headers=auth_headers,
                              json={"symbol": "SPY", "side": "buy", "qty": "100000"})
        assert too_big.status_code == 422 and "risk gate" in too_big.get_json()["error"]

    def test_cancel(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        order = client.post("/api/v1/trading/orders", headers=auth_headers,
                            json={"symbol": "SPY", "side": "buy", "qty": "1", "order_type": "limit",
                                  "limit_price": "1"}).get_json()["data"]["order"]
        cid = order["client_order_id"]
        assert client.post(f"/api/v1/trading/orders/{cid}/cancel", headers=auth_headers).status_code == 202
        assert client.post(f"/api/v1/trading/orders/{cid}/cancel", headers=auth_headers).status_code == 404

    @pytest.mark.parametrize("body,field", [
        ({"symbol": "SPY", "side": "buy", "qty": "0"}, "qty"),
        ({"symbol": "SPY", "side": "hold", "qty": "1"}, "side"),
        ({"symbol": "SPY; DROP", "side": "buy", "qty": "1"}, "symbol"),
        ({"symbol": "SPY", "side": "buy", "qty": "1", "order_type": "limit"}, ""),
    ])
    def test_invalid_orders(self, client, admin_headers, auth_headers, sim_runtime, body, field):
        _start(client, admin_headers, sim_runtime)
        resp = client.post("/api/v1/trading/orders", headers=auth_headers, json=body)
        assert resp.status_code == 400
        assert any(field in e["field"] for e in resp.get_json()["data"]["validation_errors"])

    def test_symbols_outside_the_lineup_are_refused(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        resp = client.post("/api/v1/trading/orders", headers=auth_headers, json={"symbol": "TSLA", "side": "buy",
                                                                             "qty": "1"})
        assert resp.status_code == 400

    def test_orders_need_a_running_engine(self, client, auth_headers):
        resp = client.post("/api/v1/trading/orders", headers=auth_headers, json={"symbol": "SPY", "side": "buy",
                                                                             "qty": "1"})
        assert resp.status_code == 409


class TestEmergencyStop:
    def test_running_engine_halts_and_cancels(self, client, admin_headers, auth_headers, sim_runtime):
        _start(client, admin_headers, sim_runtime)
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers,
                           json={"reason": "drill", "close_all_positions": False})
        assert resp.status_code == 200
        assert resp.get_json()["data"]["engine"] == "halted"
        status = client.get("/api/v1/trading/status", headers=auth_headers).get_json()["data"]
        assert status["halted"] is True and status["halt_reason"] == "drill" and status["open_orders"] == 0
        # A halted engine refuses new exposure until resumed
        blocked = client.post("/api/v1/trading/orders", headers=auth_headers,
                              json={"symbol": "SPY", "side": "buy", "qty": "1"})
        assert blocked.status_code == 422
        assert client.post("/api/v1/trading/control/resume", headers=admin_headers).status_code == 200
        assert client.get("/api/v1/trading/status", headers=auth_headers).get_json()["data"]["halted"] is False

    def test_flatten_without_a_running_engine_goes_straight_to_the_broker(self, client, admin_headers,
                                                                          sim_runtime):
        from decimal import Decimal
        sim_runtime.service = sim_runtime._factory(client.application, sim_runtime._settings)
        broker = sim_runtime.built["broker"]
        broker._book.adjust_position("SPY", Decimal("10"), Decimal("500"))
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers, json={"reason": "drill"})
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["engine"] == "not running" and len(data["close_orders"]) == 1
        assert [o.request.side.value for o in broker.open_orders()] == ["sell"]

    def test_without_flatten_or_engine_it_cancels_every_order_at_the_broker(self, client, admin_headers,
                                                                            sim_runtime):
        from decimal import Decimal

        from src.trading.domain import OrderRequest, Side, TimeInForce
        sim_runtime.service = sim_runtime._factory(client.application, sim_runtime._settings)
        broker = sim_runtime.built["broker"]
        broker.submit(OrderRequest("manual-1", "SPY", Side.BUY, Decimal("1"), time_in_force=TimeInForce.GTC))
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers,
                           json={"reason": "drill", "close_all_positions": False})
        assert resp.status_code == 200
        assert resp.get_json()["data"]["cancelled_orders"] == ["manual-1"] and broker.open_orders() == []

    def test_traders_cannot_trigger_it(self, client, auth_headers):
        assert client.post("/api/v1/trading/emergency-stop", headers=auth_headers, json={}).status_code == 403
