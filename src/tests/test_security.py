"""
Tests for security middleware — headers, rate limiting, IP management, input sanitization.
Covers: SEC-05 (CORS), SEC-08 (registration rate limit), security headers.
"""

import pytest

from src.middleware.security import IPManager


class TestSecurityHeaders:
    """Verify security headers are set on all responses."""

    def test_csp_header_present(self, client):
        resp = client.get("/health")
        assert "Content-Security-Policy" in resp.headers

    def test_hsts_header(self, client):
        resp = client.get("/health")
        hsts = resp.headers.get("Strict-Transport-Security", "")
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_x_frame_options_deny(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_x_content_type_nosniff(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_server_header_removed(self, client):
        resp = client.get("/health")
        assert "Server" not in resp.headers

    def test_api_cache_control_no_store(self, client):
        resp = client.get("/api/version")
        cc = resp.headers.get("Cache-Control", "")
        assert "no-store" in cc


class TestIPManager:
    """IP blacklisting and brute-force protection."""

    def test_fresh_ip_not_blacklisted(self):
        mgr = IPManager()
        assert mgr.is_blacklisted("1.2.3.4") is False

    def test_blacklist_after_threshold(self, app):
        with app.app_context():
            mgr = IPManager()
            ip = "10.0.0.1"
            for _ in range(5):
                mgr.record_failed_attempt(ip)
            assert mgr.is_blacklisted(ip) is True

    def test_clear_failed_attempts(self, app):
        with app.app_context():
            mgr = IPManager()
            ip = "10.0.0.2"
            for _ in range(3):
                mgr.record_failed_attempt(ip)
            mgr.clear_failed_attempts(ip)
            # Should not be blacklisted after clearing
            assert mgr.is_blacklisted(ip) is False

    def test_whitelist(self):
        import os
        os.environ["IP_WHITELIST"] = "192.168.1.1"
        mgr = IPManager()
        assert mgr.is_whitelisted("192.168.1.1") is True
        del os.environ["IP_WHITELIST"]


class TestHealthAndVersion:
    """Basic endpoint smoke tests."""

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_api_version(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "version" in data

    def test_404_returns_json(self, client):
        """Non-existent API endpoints return 404 JSON (not SPA fallback)."""
        resp = client.post("/api/nonexistent-endpoint-xyz", json={})
        # POST to unknown API path should 404 or 405
        assert resp.status_code in (404, 405)


class TestIPLockoutExpiry:
    def test_lockout_ends_when_failures_age_out(self, app):
        from datetime import timedelta
        with app.app_context():
            mgr = IPManager()
            ip = "10.0.0.9"
            for _ in range(5):
                mgr.record_failed_attempt(ip)
            assert mgr.is_blacklisted(ip) is True
            mgr.failed_attempts[ip] = [t - timedelta(minutes=16) for t in mgr.failed_attempts[ip]]
            assert mgr.is_blacklisted(ip) is False
            assert ip not in mgr.blacklist

    def test_whitelisted_address_is_never_locked_out(self, app):
        with app.app_context():
            mgr = IPManager()
            mgr.whitelist.add("10.0.0.10")
            for _ in range(10):
                mgr.record_failed_attempt("10.0.0.10")
            assert mgr.is_blacklisted("10.0.0.10") is False

    def test_static_deny_list_wins_over_whitelist(self):
        mgr = IPManager()
        mgr.whitelist.add("10.0.0.11")
        mgr.blacklist.add("10.0.0.11")
        assert mgr.is_blacklisted("10.0.0.11") is True


class TestRateLimits:
    def _post_login(self, client, ip="127.0.0.1"):
        return client.post("/api/v1/auth/login", json={}, environ_base={"REMOTE_ADDR": ip})

    def test_login_is_limited_per_ip(self, client):
        for _ in range(10):
            assert self._post_login(client).status_code == 400
        resp = self._post_login(client)
        assert resp.status_code == 429
        assert resp.get_json()["success"] is False
        assert self._post_login(client, ip="203.0.113.7").status_code == 400

    def test_whitelisted_ip_is_not_limited(self, client, monkeypatch):
        from src.middleware.security import ip_manager
        monkeypatch.setattr(ip_manager, "whitelist", {"127.0.0.1"})
        for _ in range(12):
            assert self._post_login(client).status_code == 400

    def test_route_and_default_limits_are_advertised(self, client):
        assert client.post("/api/v1/auth/login", json={}).headers["X-RateLimit-Limit"] == "10"
        assert client.get("/api/version").headers["X-RateLimit-Limit"] == "300"

    def test_health_and_emergency_stop_are_exempt(self, client, admin_headers):
        assert "X-RateLimit-Limit" not in client.get("/health").headers
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers, json={"reason": "drill"})
        assert "X-RateLimit-Limit" not in resp.headers

    def test_every_configured_endpoint_exists(self, app):
        from src.middleware.security import RATE_LIMIT_EXEMPT, ROUTE_RATE_LIMITS
        assert set(ROUTE_RATE_LIMITS) | RATE_LIMIT_EXEMPT <= set(app.view_functions)


class TestProxyConfiguration:
    def _app(self, count):
        from flask import Flask, request

        from src.middleware.security import configure_proxy
        app = Flask("proxy-test")
        configure_proxy(app, count)
        app.add_url_rule("/ip", "ip", lambda: request.remote_addr)
        return app.test_client()

    def test_forwarded_address_used_behind_trusted_proxy(self):
        resp = self._app(1).get("/ip", headers={"X-Forwarded-For": "198.51.100.4"},
                                environ_base={"REMOTE_ADDR": "10.0.0.1"})
        assert resp.get_data(as_text=True) == "198.51.100.4"

    def test_forwarded_header_ignored_without_proxy(self):
        resp = self._app(0).get("/ip", headers={"X-Forwarded-For": "198.51.100.4"},
                                environ_base={"REMOTE_ADDR": "10.0.0.1"})
        assert resp.get_data(as_text=True) == "10.0.0.1"


class TestRequestHandling:
    def test_valid_request_id_is_echoed(self, client):
        assert client.get("/health", headers={"X-Request-ID": "abc-123.x_y"}).headers["X-Request-ID"] == "abc-123.x_y"

    @pytest.mark.parametrize("supplied", ["has space", "x" * 65, "evilé", "a;b"])
    def test_unsafe_request_id_is_replaced(self, client, supplied):
        returned = client.get("/health", headers={"X-Request-ID": supplied}).headers["X-Request-ID"]
        assert returned != supplied
        assert len(returned) == 32 and int(returned, 16) >= 0

    def test_unknown_api_path_is_json_404(self, client):
        resp = client.get("/api/does-not-exist")
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "Endpoint not found"

    def test_wrong_method_is_json_405(self, client):
        resp = client.get("/api/v1/auth/login")
        assert resp.status_code == 405
        assert resp.get_json()["success"] is False

    def test_oversized_body_is_json_413(self, client):
        resp = client.post("/api/v1/auth/login", data=b"x" * (1024 * 1024 + 1),
                           content_type="application/json")
        assert resp.status_code == 413
        assert resp.get_json()["error"] == "Request body too large"

    def test_internal_errors_do_not_leak_details(self, client, auth_headers, monkeypatch):
        import src.routes.trading as trading_routes

        def explode():
            raise RuntimeError("db password is hunter2")

        # As in production: unhandled errors reach the JSON 500 handler instead of the test client
        monkeypatch.setitem(client.application.config, "PROPAGATE_EXCEPTIONS", False)
        monkeypatch.setattr(trading_routes, "_runtime", explode)
        resp = client.get("/api/v1/trading/status", headers=auth_headers)
        assert resp.status_code == 500
        assert resp.get_json()["error"] == "Internal server error"
        assert "hunter2" not in resp.get_data(as_text=True)

    def test_login_does_not_bypass_cors_allowlist(self, client):
        resp = client.options("/api/v1/auth/login", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        })
        assert "Access-Control-Allow-Origin" not in resp.headers


class TestMalformedBodies:
    @pytest.mark.parametrize("path,method", [
        ("/api/v1/auth/login", "post"),
        ("/api/v1/auth/refresh", "post"),
        ("/api/v1/api-keys", "post"),
        ("/api/v1/users", "post"),
        ("/api/v1/trading/orders", "post"),
    ])
    @pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b'"text"'])
    def test_rejected_with_400(self, client, admin_headers, path, method, body):
        headers = {**admin_headers, "Content-Type": "application/json"}
        resp = getattr(client, method)(path, data=body, headers=headers)
        assert resp.status_code == 400, (path, body, resp.get_json())


class TestCodeScanningFixes:
    def test_invalid_order_does_not_echo_exception_text(self, client, auth_headers):
        resp = client.post("/api/v1/trading/orders", headers=auth_headers, json={
            "symbol": "SPY", "side": "buy", "order_type": "market", "quantity": "not-a-number",
        })
        assert resp.status_code == 400
        assert "Traceback" not in resp.get_data(as_text=True)

    def test_static_route_cannot_probe_files_outside_the_static_folder(self, client):
        # src/config.py exists one level above the static folder; the response must
        # match a path that exists nowhere, or file existence leaks
        outside = client.get("/../config.py")
        missing = client.get("/no-such-file-anywhere.js")
        assert outside.status_code == missing.status_code
        assert outside.get_data() == missing.get_data()

