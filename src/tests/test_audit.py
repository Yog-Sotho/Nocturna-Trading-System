"""
Tests for the persisted audit trail.
"""

import pytest

from src.models.user import AuditLog


def _events(app, event_type=None):
    with app.app_context():
        query = AuditLog.query
        if event_type:
            query = query.filter_by(event_type=event_type)
        return query.order_by(AuditLog.id).all()


class TestAuditTrail:
    def test_failed_login_is_recorded_with_request_context(self, client, test_user):
        client.post("/api/v1/auth/login", json={"username": "testuser", "password": "Wrong-Passw0rd!"},
                    headers={"X-Request-ID": "trace-1", "User-Agent": "audit-test"})
        [event] = _events(client.application, "LOGIN_FAILED")
        assert event.success is False
        assert event.user_id == test_user[0].id
        assert event.ip_address == "127.0.0.1"
        assert event.request_id == "trace-1"
        assert event.user_agent == "audit-test"
        assert event.endpoint == "user.login"
        assert event.method == "POST"
        assert "Wrong-Passw0rd!" not in str(event.event_data)

    def test_sensitive_details_are_redacted(self, app, db):
        from src.utils.audit import record_audit_event
        with app.test_request_context("/x", method="POST"):
            record_audit_event("TEST_EVENT", True, {"password": "hunter2", "nested": {"api_key": "ntr_x"}})
        [event] = _events(app, "TEST_EVENT")
        assert event.event_data == {"password": "[REDACTED]", "nested": {"api_key": "[REDACTED]"}}

    def test_emergency_stop_is_recorded(self, client, admin_headers, admin_user, sim_runtime):
        client.post("/api/v1/trading/control/start", headers=admin_headers)
        sim_runtime.wait_running()
        resp = client.post("/api/v1/trading/emergency-stop", headers=admin_headers,
                           json={"reason": "audit drill", "close_all_positions": False})
        assert resp.status_code == 200
        [event] = _events(client.application, "EMERGENCY_STOP")
        assert event.user_id == admin_user[0].id
        assert event.event_data["reason"] == "audit drill"
        assert event.event_data["close_all_positions"] is False

    def test_reads_are_not_recorded(self, client, auth_headers):
        client.get("/api/v1/trading/status", headers=auth_headers)
        assert _events(client.application) == []

    def test_audit_failure_does_not_break_the_request(self, client, test_user, monkeypatch):
        import src.utils.audit as audit

        def broken_session(*args, **kwargs):
            raise RuntimeError("audit database down")

        monkeypatch.setattr(audit, "Session", broken_session)
        resp = client.post("/api/v1/auth/login", json={"username": "testuser", "password": "SecureP@ss123"})
        assert resp.status_code == 200

    @pytest.mark.parametrize("headers_fixture,expected", [("auth_headers", 403), ("admin_headers", 200)])
    def test_audit_log_endpoint_is_admin_only(self, client, request, headers_fixture, expected):
        headers = request.getfixturevalue(headers_fixture)
        assert client.get("/api/v1/audit-logs", headers=headers).status_code == expected
