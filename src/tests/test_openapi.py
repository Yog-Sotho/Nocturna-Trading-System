"""
Contract tests: the OpenAPI document (src/routes/openapi.py) describes the API
as it behaves. Every route is declared; the declared access holds; and real
responses, from a full tour of the API, match the declared models, which
forbid unknown fields. A route that answers a status it does not declare
fails here.
"""

import json
import re

import pytest

from src.middleware.auth import create_token
from src.models.user import User
from src.routes.openapi import build_spec, documented_routes

ARGUMENT = re.compile(r"<(?:(\w+)(?:\([^)]*\))?:)?(\w+)>")


def _concrete(rule: str) -> str:
    return ARGUMENT.sub(lambda m: "1" if m.group(1) == "int" else "x", rule)


@pytest.fixture
def spec(app):
    return build_spec(app)


@pytest.fixture
def reader_headers(app, db):
    with app.app_context():
        user = User(username="reader", email="reader@example.com", is_active=True, is_admin=False, roles=["reader"])
        user.set_password("ReaderP@ss1234")
        db.session.add(user)
        db.session.commit()
        token = create_token(str(user.id), {"username": user.username, "roles": user.roles})
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class TestDocument:
    def test_every_route_is_declared(self, app):
        missing = [f"{m} {r.rule}" for r, m, op in documented_routes(app) if op is None]
        assert missing == []

    def test_is_well_formed_openapi_3_1(self, app, spec):
        assert spec["openapi"] == "3.1.0" and spec["info"]["version"] == "v1"
        json.dumps(spec)  # serialisable
        ids = [op["operationId"] for item in spec["paths"].values() for op in item.values()]
        assert len(ids) == len(set(ids))
        for path, item in spec["paths"].items():
            templated = set(re.findall(r"{(\w+)}", path))
            for operation in item.values():
                declared = {p["name"] for p in operation.get("parameters", []) if p["in"] == "path"}
                assert declared == templated, path
                assert operation["responses"] and operation["summary"]

    def test_every_reference_resolves(self, spec):
        schemas = spec["components"]["schemas"]
        refs = set(re.findall(r'"\$ref": "#/components/schemas/([^"]+)"', json.dumps(spec)))
        assert refs and refs <= set(schemas), refs - set(schemas)

    def test_routes_and_paths_match(self, app, spec):
        documented = {(p, m) for p, item in spec["paths"].items() for m in item}
        routes = {(ARGUMENT.sub(lambda x: "{" + x.group(2) + "}", r.rule), m.lower())
                  for r, m, _ in documented_routes(app)}
        assert documented == routes

    def test_served_and_printed(self, app, client):
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200 and response.get_json()["openapi"] == "3.1.0"
        with app.app_context():
            printed = app.test_cli_runner().invoke(args=["openapi"])
        assert printed.exit_code == 0 and json.loads(printed.output)["paths"] == response.get_json()["paths"]

    def test_version_points_at_it(self, client):
        version = client.get("/api/version").get_json()
        assert version["versions"] == ["v1"] and version["openapi"] == "/api/v1/openapi.json"


class TestAccessIsEnforced:
    def test_without_credentials_every_signed_in_route_is_401(self, app, client):
        for rule, method, op in documented_routes(app):
            if op.access in ("user", "trader", "admin"):
                response = client.open(_concrete(rule.rule), method=method)
                assert response.status_code == 401, f"{method} {rule.rule}"

    def test_admin_routes_refuse_a_trader(self, app, client, auth_headers):
        for rule, method, op in documented_routes(app):
            if op.access == "admin":
                response = client.open(_concrete(rule.rule), method=method, headers=auth_headers, json={})
                assert response.status_code == 403, f"{method} {rule.rule}"

    def test_trader_routes_refuse_a_reader(self, app, client, reader_headers):
        for rule, method, op in documented_routes(app):
            if op.access == "trader":
                response = client.open(_concrete(rule.rule), method=method, headers=reader_headers, json={})
                assert response.status_code == 403, f"{method} {rule.rule}"


class Contract:
    """Checks each response against what the document declares for its route, method and status."""

    def __init__(self, app, client):
        self.client = client
        self.operations = {(r.rule, m): op for r, m, op in documented_routes(app)}
        self.rules = {r.rule: r for r, _, _ in documented_routes(app)}
        self.seen = set()

    def _operation(self, method, path):
        for (rule, m), op in self.operations.items():
            if m == method and re.fullmatch(ARGUMENT.sub(r"[^/]+", rule), path.split("?")[0]):
                return rule, op
        raise AssertionError(f"no declared operation for {method} {path}")

    def call(self, method, path, expect, **kwargs):
        response = self.client.open(path, method=method, **kwargs)
        assert response.status_code == expect, (method, path, response.status_code, response.get_data(as_text=True))
        rule, op = self._operation(method, path)
        declared = op.response(response.status_code)
        assert declared is not None, f"{method} {rule} answered {response.status_code}, which it does not declare"
        _, model, required = declared
        self.seen.add((rule, method, response.status_code))
        if op.media_type and response.status_code < 300:
            assert response.content_type == op.media_type
            return response
        body = response.get_json()
        if not op.envelope and response.status_code in op.responses:
            if model is not None:
                model.model_validate(body)
            return response
        assert isinstance(body["success"], bool) and body["success"] == (response.status_code < 400)
        if required:
            assert "data" in body, f"{method} {rule}: data missing"
        if model is not None and "data" in body:
            model.model_validate(body["data"])
        return response


class TestResponsesMatchTheDocument:
    def test_a_tour_of_the_api(self, app, client, admin_user, sim_runtime, monkeypatch):
        api = Contract(app, client)
        headers = {"Content-Type": "application/json"}

        # System
        api.call("GET", "/health", 200)
        api.call("GET", "/ready", 200)
        api.call("GET", "/api/version", 200)
        api.call("GET", "/api/v1/openapi.json", 200)
        api.call("GET", "/metrics", 404)
        from pydantic import SecretStr
        token = "m" * 40
        monkeypatch.setitem(app.extensions, "settings",
                            app.extensions["settings"].model_copy(update={"metrics_token": SecretStr(token)}))
        api.call("GET", "/metrics", 401)
        api.call("GET", "/metrics", 200, headers={"Authorization": f"Bearer {token}"})

        # Auth
        api.call("POST", "/api/v1/auth/login", 401, json={"username": "adminuser", "password": "wrong-password"})
        api.call("POST", "/api/v1/auth/login", 400, json={"username": "a"})
        login = api.call("POST", "/api/v1/auth/login", 200,
                         json={"username": "adminuser", "password": "AdminP@ss123"}).get_json()["data"]
        bearer = {**headers, "Authorization": f"Bearer {login['access_token']}"}
        pair = api.call("POST", "/api/v1/auth/refresh", 200,
                        json={"refresh_token": login["refresh_token"]}).get_json()["data"]
        api.call("POST", "/api/v1/auth/refresh", 400, json={})
        api.call("GET", "/api/v1/auth/me", 200, headers=bearer)

        # Users
        api.call("GET", "/api/v1/users?page=1&per_page=5", 200, headers=bearer)
        new = {"username": "newtrader", "email": "new@example.com", "password": "Sup3r$ecretPass",
               "password_confirm": "Sup3r$ecretPass"}
        created = api.call("POST", "/api/v1/users", 201, headers=bearer, json=new).get_json()["data"]["user"]
        api.call("POST", "/api/v1/users", 409, headers=bearer, json=new)
        api.call("POST", "/api/v1/users", 400, headers=bearer, json={"username": "x"})
        api.call("GET", f"/api/v1/users/{created['id']}", 200, headers=bearer)
        api.call("GET", "/api/v1/users/9999", 404, headers=bearer)
        api.call("PUT", f"/api/v1/users/{created['id']}", 200, headers=bearer, json={"roles": ["trader"]})
        api.call("PUT", f"/api/v1/users/{created['id']}", 400, headers=bearer, json={"trading_mode": "LIVE"})
        api.call("PUT", f"/api/v1/users/{created['id']}/password", 200, headers=bearer,
                 json={"current_password": "Sup3r$ecretPass", "new_password": "An0ther$ecretPass"})

        # API keys
        key = api.call("POST", "/api/v1/api-keys", 201, headers=bearer,
                       json={"name": "bot", "permissions": ["read"], "expires_in_days": 30}).get_json()["data"]
        api.call("POST", "/api/v1/api-keys", 400, headers=bearer, json={"permissions": ["admin"]})
        api.call("GET", "/api/v1/api-keys", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/status", 200, headers={"X-API-Key": key["api_key"]})
        api.call("DELETE", f"/api/v1/api-keys/{key['key']['id']}", 200, headers=bearer)
        api.call("DELETE", f"/api/v1/api-keys/{key['key']['id']}", 404, headers=bearer)

        # Audit log
        api.call("GET", "/api/v1/audit-logs?event_type=LOGIN_SUCCESS", 200, headers=bearer)

        # Trading, engine stopped
        api.call("GET", "/api/v1/trading/health", 200)
        api.call("GET", "/api/v1/trading/status", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/positions", 409, headers=bearer)
        api.call("GET", "/api/v1/trading/targets", 409, headers=bearer)
        api.call("POST", "/api/v1/trading/control/resume", 409, headers=bearer)
        order = {"symbol": "SPY", "side": "buy", "qty": "1"}
        api.call("POST", "/api/v1/trading/orders", 409, headers=bearer, json=order)
        api.call("POST", "/api/v1/trading/orders/nope/cancel", 409, headers=bearer)

        # Running
        api.call("POST", "/api/v1/trading/control/start", 202, headers=bearer)
        sim_runtime.wait_running()
        api.call("POST", "/api/v1/trading/control/start", 409, headers=bearer)
        api.call("GET", "/api/v1/trading/status", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/positions", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/targets", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/orders?limit=5", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/fills", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/equity-curve", 200, headers=bearer)
        placed = api.call("POST", "/api/v1/trading/orders", 201, headers=bearer, json=order).get_json()["data"]
        api.call("POST", "/api/v1/trading/orders", 400, headers=bearer, json={**order, "symbol": "QQQ"})
        api.call("POST", "/api/v1/trading/orders", 400, headers=bearer, json={**order, "qty": "-1"})
        api.call("POST", "/api/v1/trading/orders", 422, headers=bearer, json={**order, "qty": "100000"})
        api.call("POST", f"/api/v1/trading/orders/{placed['order']['client_order_id']}/cancel", 202, headers=bearer)
        api.call("POST", "/api/v1/trading/orders/nope/cancel", 404, headers=bearer)
        api.call("POST", "/api/v1/trading/emergency-stop", 200, headers=bearer,
                 json={"reason": "drill", "close_all_positions": False})
        api.call("POST", "/api/v1/trading/control/resume", 200, headers=bearer)

        # The strategies and the trading profile
        api.call("GET", "/api/v1/trading/strategies", 200, headers=bearer)
        api.call("GET", "/api/v1/trading/profile", 200, headers=bearer)
        api.call("PUT", "/api/v1/trading/profile", 400, headers=bearer, json={"mix": []})
        mix = {"mix": [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "1"}],
               "risk": {"risk_max_drawdown": "0.08"}}
        saved = api.call("PUT", "/api/v1/trading/profile", 201, headers=bearer, json=mix).get_json()["data"]
        assert saved["restart_needed"] is True
        api.call("POST", "/api/v1/trading/control/restart", 202, headers=bearer)
        sim_runtime.wait_running()
        state = api.call("GET", "/api/v1/trading/profile", 200, headers=bearer).get_json()["data"]
        assert state["running"]["id"] == saved["profile"]["id"] and state["pending"] is False
        api.call("POST", "/api/v1/trading/control/stop", 200, headers=bearer)
        # Without a running engine the emergency stop goes to the broker
        api.call("POST", "/api/v1/trading/emergency-stop", 200, headers=bearer,
                 json={"reason": "drill", "close_all_positions": False})
        api.call("POST", "/api/v1/trading/emergency-stop", 400, headers=bearer, json={"close_all_positions": "x"})

        # The strategy lab: a run is queued, its process executes it, the result is read back
        from src.trading import lab

        class InlineLab:
            def launch(self):
                lab.drain(app, app.extensions["settings"])

        monkeypatch.setitem(app.extensions, "lab", InlineLab())
        monkeypatch.setattr(lab, "execute", lambda settings, kind, spec: (
            {"headline": "rejected: fails out-of-sample Sharpe", "passed": False}, "# Validation: tsmom"))
        api.call("POST", "/api/v1/lab/runs", 400, headers=bearer, json={"kind": "optimise"})
        run = api.call("POST", "/api/v1/lab/runs", 202, headers=bearer, json={
            "kind": "validate", "strategy": "tsmom", "start": "2017-01-03", "end": "2026-01-02"}).get_json()["data"]
        api.call("GET", "/api/v1/lab/runs?limit=5", 200, headers=bearer)
        done = api.call("GET", f"/api/v1/lab/runs/{run['run']['id']}", 200, headers=bearer).get_json()["data"]
        assert done["run"]["status"] == "done" and done["run"]["report"] == "# Validation: tsmom"
        api.call("GET", "/api/v1/lab/runs/9999", 404, headers=bearer)
        api.call("GET", "/api/v1/trading/strategies", 200, headers=bearer)

        # Alerts and preferences
        api.call("GET", "/api/v1/settings/alerts", 200, headers=bearer)
        api.call("PUT", "/api/v1/settings/alerts", 200, headers=bearer, json={"categories": {"fills": True}})
        api.call("PUT", "/api/v1/settings/alerts", 400, headers=bearer, json={"categories": {"safety": False}})
        api.call("POST", "/api/v1/settings/alerts/test", 409, headers=bearer)
        api.call("GET", "/api/v1/me/preferences", 200, headers=bearer)
        api.call("PUT", "/api/v1/me/preferences", 200, headers=bearer, json={"theme": "dark", "accent": "teal"})
        api.call("PUT", "/api/v1/me/preferences", 400, headers=bearer, json={"theme": "neon"})

        # Sessions end
        second = {**headers, "Authorization": f"Bearer {pair['access_token']}"}
        api.call("POST", "/api/v1/auth/logout", 200, headers=second, json={"refresh_token": pair["refresh_token"]})
        api.call("POST", "/api/v1/auth/logout-all", 200, headers=bearer)
        api.call("GET", "/api/v1/auth/me", 401, headers=bearer)

        # Every operation of the API was exercised
        exercised = {(rule, method) for rule, method, _ in api.seen}
        assert exercised == set(api.operations), set(api.operations) - exercised
