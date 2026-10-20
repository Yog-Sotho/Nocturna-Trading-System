# FILE LOCATION: src/tests/test_auth.py
"""
Tests for authentication middleware — JWT tokens, API keys, decorators.
Covers: SEC-02 (API key auth), SEC-03 (token blacklist), SEC-08 (rate limiting).
"""

from datetime import UTC

import jwt as pyjwt
import pytest

from src.middleware.auth import (
    APIKeyManager,
    TokenManager,
    generate_api_key,
)
from src.models.user import User


class TestTokenManager:
    """Token creation, verification, and revocation."""

    def test_create_access_token(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            token = tm.create_access_token("user-42", {"roles": ["trader"]})
            assert isinstance(token, str)
            assert len(token) > 20

    def test_verify_valid_token(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            token = tm.create_access_token("user-42")
            payload = tm.verify_token(token, token_type="access")
            assert payload["sub"] == "user-42"
            assert payload["type"] == "access"

    def test_verify_wrong_type_raises(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            token = tm.create_access_token("user-42")
            with app.test_request_context(), pytest.raises(pyjwt.InvalidTokenError, match="type mismatch"):
                tm.verify_token(token, token_type="refresh")

    def test_revoke_token(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            token = tm.create_access_token("user-42")
            with app.test_request_context():
                assert tm.revoke_token(token) is True
                with pytest.raises(pyjwt.InvalidTokenError, match="revoked"):
                    tm.verify_token(token)

    def test_refresh_token_flow(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            refresh = tm.create_refresh_token("user-42")
            with app.test_request_context():
                payload = tm.verify_token(refresh, token_type="refresh")
                assert payload["sub"] == "user-42"
                assert payload["type"] == "refresh"

    def test_expired_token_raises(self, app):
        from datetime import timedelta
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            tm.access_token_expires = timedelta(seconds=-1)
            token = tm.create_access_token("user-42")
            with app.test_request_context(), pytest.raises(pyjwt.ExpiredSignatureError):
                tm.verify_token(token)

    def test_corrupted_token_raises(self, app):
        with app.app_context():
            tm = TokenManager()
            tm.init_app(app)
            with app.test_request_context(), pytest.raises(pyjwt.InvalidTokenError):
                tm.verify_token("not.a.valid.token")


class TestAPIKeyManager:
    """API key registration and validation (SEC-02 fix)."""

    def test_register_and_validate(self):
        mgr = APIKeyManager()
        raw_key = generate_api_key()
        mgr.register_key(raw_key, "user-99", permissions=["read", "trade"])
        result = mgr.validate_key(raw_key)
        assert result is not None
        assert result["user_id"] == "user-99"
        assert "trade" in result["permissions"]

    def test_invalid_key_returns_none(self):
        mgr = APIKeyManager()
        assert mgr.validate_key("bogus_key_12345") is None

    def test_revoke_key(self):
        mgr = APIKeyManager()
        raw_key = generate_api_key()
        mgr.register_key(raw_key, "user-99")
        assert mgr.revoke_key(raw_key) is True
        assert mgr.validate_key(raw_key) is None

    def test_expired_key_returns_none(self, app):
        from datetime import datetime, timedelta
        with app.app_context():
            mgr = APIKeyManager()
            raw_key = generate_api_key()
            mgr.register_key(
                raw_key, "user-99",
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
            result = mgr.validate_key(raw_key)
            assert result is None


class TestAuthEndpoints:
    """Integration tests for auth-protected endpoints."""

    def test_unauth_returns_401(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_auth_me_returns_user(self, client, auth_headers):
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_login_returns_token(self, client, test_user):
        resp = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "SecureP@ss123",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "access_token" in data.get("data", {})

    def test_login_wrong_password(self, client, test_user):
        resp = client.post("/api/auth/login", json={
            "username": "testuser",
            "password": "WrongPassword123!",
        })
        assert resp.status_code == 401

    NEW_USER = {
        "username": "newusertest",
        "email": "newtest@example.com",
        "password": "NewP@ssword123!",
        "password_confirm": "NewP@ssword123!",
    }

    def test_public_registration_is_closed(self, client, db):
        resp = client.post("/api/users", json=self.NEW_USER)
        assert resp.status_code == 401
        with client.application.app_context():
            assert User.query.filter_by(username="newusertest").first() is None

    def test_non_admin_cannot_create_user(self, client, auth_headers):
        resp = client.post("/api/users", json=self.NEW_USER, headers=auth_headers)
        assert resp.status_code == 403

    def test_admin_creates_user_without_issuing_tokens(self, client, admin_headers):
        resp = client.post("/api/users", json=self.NEW_USER, headers=admin_headers)
        assert resp.status_code == 201
        body = resp.get_json()
        assert "access_token" not in body["data"]
        assert body["data"]["user"]["username"] == "newusertest"
        assert body["data"]["user"]["is_admin"] is False

    def test_admin_create_duplicate_email_conflicts(self, client, admin_headers, test_user):
        payload = {**self.NEW_USER, "username": "anotheruser", "email": "test@example.com"}
        resp = client.post("/api/users", json=payload, headers=admin_headers)
        assert resp.status_code == 409


    def test_api_key_in_require_auth(self, client, app, test_user):
        """SEC-02: Verify API key path in require_auth actually works."""
        from src.middleware.auth import api_key_manager, generate_api_key
        with app.app_context():
            user, _ = test_user
            raw_key = generate_api_key()
            api_key_manager.register_key(
                raw_key, str(user.id), permissions=["read", "trade"]
            )
            resp = client.get(
                "/api/auth/me",
                headers={"X-API-Key": raw_key},
            )
            # Should authenticate via API key now (SEC-02 fix)
            assert resp.status_code == 200

    def test_invalid_api_key_returns_401(self, client):
        """SEC-02: Invalid API key must be rejected."""
        resp = client.get(
            "/api/auth/me",
            headers={"X-API-Key": "ntr_invalid_key_12345678901234567890"},
        )
        assert resp.status_code == 401


class TestCreateAdminCommand:
    """`flask create-admin` is the only way to bootstrap the operator account."""

    def test_creates_admin(self, app, db):
        result = app.test_cli_runner().invoke(args=[
            "create-admin", "--username", "Operator", "--email", "Ops@Example.com",
            "--password", "Str0ng!Passw0rd",
        ])
        assert result.exit_code == 0, result.output
        with app.app_context():
            admin = User.query.filter_by(username="operator").first()
            assert admin is not None
            assert admin.is_admin is True
            assert "admin" in admin.roles
            assert admin.email == "ops@example.com"
            assert admin.check_password("Str0ng!Passw0rd")

    def test_rejects_weak_password(self, app, db):
        result = app.test_cli_runner().invoke(args=[
            "create-admin", "--email", "ops@example.com", "--password", "weakpassword",
        ])
        assert result.exit_code != 0
        assert "Invalid admin details" in result.output
        with app.app_context():
            assert User.query.count() == 0

    def test_refuses_duplicate(self, app, db, admin_user):
        result = app.test_cli_runner().invoke(args=[
            "create-admin", "--username", "adminuser", "--email", "other@example.com",
            "--password", "Str0ng!Passw0rd",
        ])
        assert result.exit_code != 0
        assert "already exists" in result.output


class TestLoginLockout:
    """Regression tests for the naive/aware datetime crash and account enumeration."""

    PASSWORD = "SecureP@ss123"

    def _login(self, client, username, password):
        return client.post("/api/auth/login", json={"username": username, "password": password})

    def _clear_ip_lockout(self):
        from src.middleware.security import ip_manager
        ip_manager.blacklist.clear()
        ip_manager.failed_attempts.clear()

    def _fail_five_times(self, client):
        for _ in range(5):
            self._login(client, "testuser", "WrongP@ssword1")
            self._clear_ip_lockout()  # isolate account lockout from IP lockout

    def test_locked_account_returns_401_not_500(self, client, test_user):
        self._fail_five_times(client)
        resp = self._login(client, "testuser", self.PASSWORD)
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "Invalid credentials"

    def test_login_works_again_after_lock_expires(self, app, client, test_user):
        from datetime import UTC, datetime, timedelta

        self._fail_five_times(client)
        with app.app_context():
            user = User.query.filter_by(username="testuser").first()
            assert user.is_locked()
            user.locked_until = datetime.now(UTC) - timedelta(seconds=1)
            _db_commit()

        resp = self._login(client, "testuser", self.PASSWORD)
        assert resp.status_code == 200

    def test_expired_lock_starts_a_fresh_failure_count(self, app, client, test_user):
        from datetime import UTC, datetime, timedelta

        self._fail_five_times(client)
        with app.app_context():
            user = User.query.filter_by(username="testuser").first()
            user.locked_until = datetime.now(UTC) - timedelta(seconds=1)
            _db_commit()

        self._login(client, "testuser", "WrongP@ssword1")
        with app.app_context():
            user = User.query.filter_by(username="testuser").first()
            assert user.failed_login_attempts == 1
            assert not user.is_locked()

    def test_unknown_and_locked_accounts_are_indistinguishable(self, client, test_user):
        unknown = self._login(client, "nosuchuser", self.PASSWORD)
        self._clear_ip_lockout()
        self._fail_five_times(client)
        locked = self._login(client, "testuser", self.PASSWORD)
        assert unknown.status_code == locked.status_code == 401
        assert unknown.get_json()["error"] == locked.get_json()["error"]

    def test_expires_in_matches_token_lifetime(self, client, test_user):
        resp = self._login(client, "testuser", self.PASSWORD)
        assert resp.status_code == 200
        assert resp.get_json()["data"]["expires_in"] == 3600


def _db_commit():
    from src.models.user import db
    db.session.commit()


class TestUTCDateTime:
    def test_round_trip_returns_aware_utc(self, app, db, test_user):
        from datetime import UTC, datetime, timedelta, timezone

        local = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
        with app.app_context():
            user = User.query.filter_by(username="testuser").first()
            user.last_login = local
            db.session.commit()
            db.session.expire_all()
            reloaded = User.query.filter_by(username="testuser").first().last_login
        assert reloaded.tzinfo is not None
        assert reloaded == local
        assert reloaded.utcoffset() == timedelta(0)
        assert reloaded.tzinfo == UTC
