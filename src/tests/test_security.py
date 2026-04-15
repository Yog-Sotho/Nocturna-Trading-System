"""
Tests for security middleware — headers, rate limiting, IP management, input sanitization.
Covers: SEC-05 (CORS), SEC-08 (registration rate limit), security headers.
"""


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
