"""
Tests for security middleware — headers, rate limiting, IP management, input sanitization.
Covers: SEC-05 (CORS), SEC-08 (registration rate limit), security headers.
"""

import pytest

from src.middleware.security import InputSanitizer, IPManager


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


class TestInputSanitizer:
    """Input validation and sanitization."""

    def test_sanitize_valid_symbol(self):
        assert InputSanitizer.sanitize_symbol("AAPL") == "AAPL"

    def test_sanitize_lowercase_symbol(self):
        result = InputSanitizer.sanitize_symbol("aapl")
        # Should uppercase or reject
        assert result is None or result == "AAPL"

    def test_reject_invalid_symbol(self):
        assert InputSanitizer.sanitize_symbol("DROP TABLE") is None
        assert InputSanitizer.sanitize_symbol("") is None
        assert InputSanitizer.sanitize_symbol("TOOLONGSYMBOL") is None

    def test_sanitize_numeric(self):
        assert InputSanitizer.sanitize_numeric(42.5) == 42.5
        assert InputSanitizer.sanitize_numeric(-1.0, min_val=0) is None

    def test_sanitize_string_strips_null_bytes(self):
        result = InputSanitizer.sanitize_string("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_string_truncates(self):
        result = InputSanitizer.sanitize_string("x" * 1000, max_length=50)
        assert len(result) == 50

    def test_sanitize_order_signal_valid(self):
        signal = {
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "quantity": 100,
        }
        result = InputSanitizer.sanitize_order_signal(signal)
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["side"] == "buy"

    def test_sanitize_order_signal_invalid_side(self):
        signal = {
            "symbol": "AAPL",
            "side": "hold",  # Invalid
            "type": "market",
            "quantity": 100,
        }
        assert InputSanitizer.sanitize_order_signal(signal) is None

    def test_sanitize_order_signal_missing_symbol(self):
        signal = {"side": "buy", "type": "market", "quantity": 100}
        assert InputSanitizer.sanitize_order_signal(signal) is None


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
        return client.post("/api/auth/login", json={}, environ_base={"REMOTE_ADDR": ip})

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
        assert client.post("/api/auth/login", json={}).headers["X-RateLimit-Limit"] == "10"
        assert client.get("/api/version").headers["X-RateLimit-Limit"] == "300"

    def test_health_and_emergency_stop_are_exempt(self, client, admin_headers):
        assert "X-RateLimit-Limit" not in client.get("/health").headers
        resp = client.post("/api/trading/emergency-stop", headers=admin_headers, json={"reason": "drill"})
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
        resp = client.get("/api/auth/login")
        assert resp.status_code == 405
        assert resp.get_json()["success"] is False

    def test_oversized_body_is_json_413(self, client):
        resp = client.post("/api/auth/login", data=b"x" * (1024 * 1024 + 1),
                           content_type="application/json")
        assert resp.status_code == 413
        assert resp.get_json()["error"] == "Request body too large"

    def test_internal_errors_do_not_leak_details(self, client, auth_headers, monkeypatch):
        import src.routes.trading as trading_routes

        def explode():
            raise RuntimeError("db password is hunter2")

        monkeypatch.setattr(trading_routes, "get_trading_engine", explode)
        resp = client.get("/api/trading/status", headers=auth_headers)
        assert resp.status_code == 500
        assert "hunter2" not in resp.get_data(as_text=True)

    def test_login_does_not_bypass_cors_allowlist(self, client):
        resp = client.options("/api/auth/login", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        })
        assert "Access-Control-Allow-Origin" not in resp.headers


class TestMalformedBodies:
    @pytest.mark.parametrize("path,method", [
        ("/api/auth/login", "post"),
        ("/api/paper/orders", "post"),
        ("/api/paper/prices", "post"),
        ("/api/trading/orders", "post"),
        ("/api/trading/config", "put"),
    ])
    @pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b'"text"'])
    def test_rejected_with_400(self, client, admin_headers, path, method, body):
        headers = {**admin_headers, "Content-Type": "application/json"}
        resp = getattr(client, method)(path, data=body, headers=headers)
        assert resp.status_code == 400, (path, body, resp.get_json())


class TestCodeScanningFixes:
    def test_invalid_paper_order_does_not_echo_exception_text(self, client, auth_headers):
        resp = client.post("/api/paper/orders", headers=auth_headers, json={
            "symbol": "SPY", "side": "buy", "order_type": "market", "quantity": "not-a-number",
        })
        assert resp.status_code == 400
        assert "could not convert" not in resp.get_data(as_text=True)

    def test_static_route_cannot_probe_files_outside_the_static_folder(self, client):
        # src/config.py exists one level above the static folder; the response must
        # match a path that exists nowhere, or file existence leaks
        outside = client.get("/../config.py")
        missing = client.get("/no-such-file-anywhere.js")
        assert outside.status_code == missing.status_code
        assert outside.get_data() == missing.get_data()

    def test_url_stripping_removes_whole_urls(self):
        from src.advanced.sentiment_analyzer import SentimentAnalyzer
        cleaned = SentimentAnalyzer({})._preprocess_text("Read https://example.com/a/b?q=1&x=2 now")
        assert "example" not in cleaned and "q=1" not in cleaned
        assert "read" in cleaned and "now" in cleaned
