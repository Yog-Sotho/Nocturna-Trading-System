# FILE LOCATION: src/tests/test_auth.py
"""
Tests for authentication middleware — JWT tokens, API keys, decorators.
Covers: SEC-02 (API key auth), SEC-03 (token blacklist), SEC-08 (rate limiting).
"""

from datetime import UTC

import jwt as pyjwt
import pytest

from src.middleware.auth import (
    TokenManager,
    authenticate_api_key,
    issue_api_key,
    parse_api_key,
    roles_for_api_key,
)
from src.models.user import APIKey, User


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


class TestAPIKeyHelpers:
    def test_issued_keys_are_unique_and_parse_into_id_and_secret(self, app):
        with app.app_context():
            (raw_a, id_a, _), (raw_b, id_b, _) = issue_api_key(), issue_api_key()
        assert raw_a != raw_b and id_a != id_b
        key_id, secret = parse_api_key(raw_a)
        assert key_id == id_a
        assert raw_a == f"ntr_{key_id}_{secret}" and len(secret) == 43

    def test_secret_is_stored_only_as_salted_scrypt(self, app):
        from src.middleware.auth import _hash_key_secret, _verify_key_secret
        with app.app_context():
            raw, _, key_hash = issue_api_key()
            _, secret = parse_api_key(raw)
            assert key_hash.startswith("scrypt$16384$8$1$")
            assert secret not in key_hash
            assert _hash_key_secret(secret) != _hash_key_secret(secret)  # per-key salt
            assert _verify_key_secret(secret, key_hash) is True
            assert _verify_key_secret(secret[:-1] + ("A" if secret[-1] != "A" else "B"), key_hash) is False

    @pytest.mark.parametrize("stored", ["", "scrypt$x", "bcrypt$1$2$3$aa$bb", "scrypt$16384$8$1$zz$00"])
    def test_malformed_stored_hash_never_verifies(self, app, stored):
        from src.middleware.auth import _verify_key_secret
        with app.app_context():
            assert _verify_key_secret("s" * 43, stored) is False

    @pytest.mark.parametrize("raw", [
        "", "ntr_", "ntr_not-a-real-key", "xyz_0123456789abcdef_" + "a" * 43,
        "ntr_0123456789ABCDEF_" + "a" * 43, "ntr_0123456789abcdef_" + "a" * 42,
        "ntr_0123456789abcdef_" + "a" * 43 + "\n",
    ])
    def test_malformed_keys_do_not_parse(self, raw):
        assert parse_api_key(raw) is None

    def test_pepper_is_part_of_the_hash(self, app, monkeypatch):
        from src.middleware import auth
        with app.app_context():
            raw, _, key_hash = issue_api_key()
            _, secret = parse_api_key(raw)
            monkeypatch.setattr(auth, "_api_key_pepper", lambda: b"another-pepper")
            assert auth._verify_key_secret(secret, key_hash) is False

    @pytest.mark.parametrize("scopes,owner_roles,owner_is_admin,expected", [
        (["read"], ["trader"], False, ["reader"]),
        (["read", "trade"], ["trader"], False, ["reader", "trader"]),
        (["read", "trade"], [], False, ["reader"]),
        (["read", "trade"], ["admin"], True, ["reader", "trader"]),
        (["read", "admin"], ["admin"], True, ["reader"]),
    ])
    def test_roles_never_exceed_owner_and_never_include_admin(
        self, scopes, owner_roles, owner_is_admin, expected
    ):
        assert sorted(roles_for_api_key(scopes, owner_roles, owner_is_admin)) == expected

    def test_unknown_key_does_not_authenticate(self, app, db):
        with app.app_context():
            assert authenticate_api_key("ntr_not-a-real-key") is None

    def test_unknown_key_id_is_rejected_without_running_scrypt(self, app, db, monkeypatch):
        import hashlib
        calls = []
        real_scrypt = hashlib.scrypt
        monkeypatch.setattr(hashlib, "scrypt", lambda *a, **k: calls.append(1) or real_scrypt(*a, **k))
        with app.app_context():
            assert authenticate_api_key("ntr_0123456789abcdef_" + "a" * 43) is None
        assert calls == []


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


    def test_invalid_api_key_returns_401(self, client):
        """SEC-02: Invalid API key must be rejected."""
        resp = client.get(
            "/api/auth/me",
            headers={"X-API-Key": "ntr_invalid_key_12345678901234567890"},
        )
        assert resp.status_code == 401


class TestAPIKeyEndpoints:
    """DB-backed API keys: created via the API, accepted by auth, scoped, revocable."""

    def _create(self, client, headers, **body):
        return client.post("/api/api-keys", json=body, headers=headers)

    def _key(self, client, headers, permissions):
        resp = self._create(client, headers, name="bot", permissions=permissions)
        assert resp.status_code == 201, resp.get_json()
        return resp.get_json()["data"]

    def test_raw_key_is_returned_once_and_only_its_hash_is_stored(self, client, auth_headers):
        data = self._key(client, auth_headers, ["read"])
        raw = data["api_key"]
        with client.application.app_context():
            record = _get_key(data["key"]["id"])
            key_id, secret = parse_api_key(raw)
            assert record.key_id == key_id
            assert secret not in record.key_hash
        listed = client.get("/api/api-keys", headers=auth_headers).get_json()["data"]["api_keys"]
        assert [k["id"] for k in listed] == [data["key"]["id"]]
        assert listed[0]["key_id"] == key_id
        assert "api_key" not in listed[0] and "key_hash" not in listed[0]
        assert secret not in str(listed)

    def test_read_key_authenticates(self, client, auth_headers):
        raw = self._key(client, auth_headers, ["read"])["api_key"]
        resp = client.get("/api/auth/me", headers={"X-API-Key": raw})
        assert resp.status_code == 200
        assert resp.get_json()["data"]["user"]["username"] == "testuser"

    def test_read_key_cannot_trade(self, client, auth_headers):
        raw = self._key(client, auth_headers, ["read"])["api_key"]
        resp = client.post("/api/trading/orders/probe/cancel", headers={"X-API-Key": raw})
        assert resp.status_code == 403

    def test_trade_key_passes_trading_checks(self, client, auth_headers):
        raw = self._key(client, auth_headers, ["read", "trade"])["api_key"]
        resp = client.post("/api/trading/orders/probe/cancel", headers={"X-API-Key": raw})
        assert resp.status_code not in (401, 403)

    def test_admin_owner_key_never_grants_admin(self, client, admin_headers):
        raw = self._key(client, admin_headers, ["read", "trade"])["api_key"]
        resp = client.get("/api/users", headers={"X-API-Key": raw})
        assert resp.status_code == 403

    @pytest.mark.parametrize("body", [
        {"permissions": ["admin"]},
        {"permissions": ["read", "admin"]},
        {"permissions": []},
        {"permissions": "read"},
        {"permissions": ["read"], "expires_in_days": 0},
        {"permissions": ["read"], "expires_in_days": 366},
        {"permissions": ["read"], "expires_in_days": "30"},
    ])
    def test_invalid_requests_rejected(self, client, auth_headers, body):
        assert self._create(client, auth_headers, **body).status_code == 400

    def test_revoked_key_is_rejected(self, client, auth_headers):
        data = self._key(client, auth_headers, ["read"])
        resp = client.delete(f"/api/api-keys/{data['key']['id']}", headers=auth_headers)
        assert resp.status_code == 200
        assert client.get("/api/auth/me", headers={"X-API-Key": data["api_key"]}).status_code == 401

    def test_cannot_revoke_another_users_key(self, client, auth_headers, admin_headers):
        data = self._key(client, admin_headers, ["read"])
        resp = client.delete(f"/api/api-keys/{data['key']['id']}", headers=auth_headers)
        assert resp.status_code == 404
        assert client.get("/api/auth/me", headers={"X-API-Key": data["api_key"]}).status_code == 200

    def test_expired_key_is_rejected(self, client, auth_headers):
        from datetime import datetime, timedelta
        data = self._key(client, auth_headers, ["read"])
        with client.application.app_context():
            record = _get_key(data["key"]["id"])
            record.expires_at = datetime.now(UTC) - timedelta(minutes=1)
            _db_commit()
        assert client.get("/api/auth/me", headers={"X-API-Key": data["api_key"]}).status_code == 401

    def test_key_of_deactivated_user_is_rejected(self, client, auth_headers, test_user):
        data = self._key(client, auth_headers, ["read"])
        with client.application.app_context():
            user = User.query.filter_by(username="testuser").first()
            user.is_active = False
            _db_commit()
        assert client.get("/api/auth/me", headers={"X-API-Key": data["api_key"]}).status_code == 401

    def test_trading_disabled_user_cannot_trade(self, client, auth_headers):
        with client.application.app_context():
            user = User.query.filter_by(username="testuser").first()
            user.trading_disabled = True
            _db_commit()
        resp = client.post("/api/trading/orders/probe/cancel", headers=auth_headers)
        assert resp.status_code == 403

    def test_user_without_trader_role_cannot_trade(self, client):
        from src.middleware.auth import create_token
        with client.application.app_context():
            user = User(username="viewer", email="viewer@example.com", is_active=True, roles=[])
            user.set_password("ViewerP@ss123")
            _db_add(user)
            token = create_token(str(user.id), {"username": "viewer", "roles": [], "is_admin": False})
        resp = client.post("/api/trading/orders/probe/cancel", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_usage_is_counted(self, client, auth_headers):
        data = self._key(client, auth_headers, ["read"])
        for _ in range(2):
            client.get("/api/auth/me", headers={"X-API-Key": data["api_key"]})
        with client.application.app_context():
            assert _get_key(data["key"]["id"]).use_count == 2


class TestSessionLifecycle:
    """Revocation persists, refresh tokens rotate, password and role changes apply at once."""

    PASSWORD = "SecureP@ss123"

    def _login(self, client):
        resp = client.post("/api/auth/login", json={"username": "testuser", "password": self.PASSWORD})
        assert resp.status_code == 200, resp.get_json()
        return resp.get_json()["data"]

    @staticmethod
    def _bearer(token):
        return {"Authorization": f"Bearer {token}"}

    def _me(self, client, token):
        return client.get("/api/auth/me", headers=self._bearer(token)).status_code

    def _refresh(self, client, refresh_token):
        return client.post("/api/auth/refresh", json={"refresh_token": refresh_token})

    def test_logout_revocation_survives_a_new_token_manager(self, client, test_user, monkeypatch):
        extensions = client.application.extensions
        monkeypatch.setitem(extensions, "token_manager", extensions["token_manager"])
        tokens = self._login(client)
        resp = client.post("/api/auth/logout", headers=self._bearer(tokens["access_token"]))
        assert resp.status_code == 200
        assert self._me(client, tokens["access_token"]) == 401
        with client.application.app_context():
            fresh = TokenManager()
            fresh.init_app(client.application)
            with pytest.raises(pyjwt.InvalidTokenError, match="revoked"):
                fresh.verify_token(tokens["access_token"])

    def test_logout_also_revokes_refresh_token_in_body(self, client, test_user):
        tokens = self._login(client)
        client.post("/api/auth/logout", headers=self._bearer(tokens["access_token"]),
                    json={"refresh_token": tokens["refresh_token"]})
        assert self._refresh(client, tokens["refresh_token"]).status_code == 401

    def test_logout_requires_bearer_token(self, client, auth_headers):
        raw = client.post("/api/api-keys", json={"permissions": ["read"]},
                          headers=auth_headers).get_json()["data"]["api_key"]
        assert client.post("/api/auth/logout", headers={"X-API-Key": raw}).status_code == 400

    def test_refresh_rotates_tokens(self, client, test_user):
        tokens = self._login(client)
        resp = self._refresh(client, tokens["refresh_token"])
        assert resp.status_code == 200
        new = resp.get_json()["data"]
        assert new["refresh_token"] != tokens["refresh_token"]
        assert self._me(client, new["access_token"]) == 200

    def test_refresh_token_reuse_revokes_every_session(self, client, test_user):
        tokens = self._login(client)
        new = self._refresh(client, tokens["refresh_token"]).get_json()["data"]
        assert self._refresh(client, tokens["refresh_token"]).status_code == 401
        assert self._me(client, new["access_token"]) == 401
        assert self._refresh(client, new["refresh_token"]).status_code == 401

    def test_access_token_is_not_a_refresh_token(self, client, test_user):
        tokens = self._login(client)
        assert self._refresh(client, tokens["access_token"]).status_code == 401

    @pytest.mark.parametrize("body", [{}, {"refresh_token": 123}, {"refresh_token": ""}])
    def test_refresh_requires_token(self, client, body):
        assert client.post("/api/auth/refresh", json=body).status_code == 400

    def test_own_password_change_invalidates_old_tokens_and_returns_new(self, client, test_user):
        user, _ = test_user
        tokens = self._login(client)
        resp = client.put(f"/api/users/{user.id}/password", headers=self._bearer(tokens["access_token"]),
                          json={"current_password": self.PASSWORD, "new_password": "N3w!Passw0rd-xyz"})
        assert resp.status_code == 200
        assert self._me(client, tokens["access_token"]) == 401
        assert self._refresh(client, tokens["refresh_token"]).status_code == 401
        assert self._me(client, resp.get_json()["data"]["access_token"]) == 200

    def test_admin_password_reset_invalidates_users_tokens(self, client, test_user, admin_headers):
        user, _ = test_user
        tokens = self._login(client)
        resp = client.put(f"/api/users/{user.id}/password", headers=admin_headers,
                          json={"current_password": "unused", "new_password": "N3w!Passw0rd-xyz"})
        assert resp.status_code == 200
        assert "data" not in resp.get_json()
        assert self._me(client, tokens["access_token"]) == 401

    def test_logout_all(self, client, test_user):
        first, second = self._login(client), self._login(client)
        assert client.post("/api/auth/logout-all", headers=self._bearer(first["access_token"])).status_code == 200
        assert self._me(client, first["access_token"]) == 401
        assert self._me(client, second["access_token"]) == 401
        assert self._me(client, self._login(client)["access_token"]) == 200

    def test_removing_trader_role_applies_to_existing_token(self, client, test_user, admin_headers):
        user, _ = test_user
        tokens = self._login(client)
        trade = self._bearer(tokens["access_token"])
        assert client.post("/api/trading/orders/probe/cancel", headers=trade).status_code not in (401, 403)
        resp = client.put(f"/api/users/{user.id}", headers=admin_headers, json={"roles": ["reader"]})
        assert resp.status_code == 200
        assert client.post("/api/trading/orders/probe/cancel", headers=trade).status_code == 403

    def test_deactivation_applies_to_existing_token(self, client, test_user, admin_headers):
        user, _ = test_user
        tokens = self._login(client)
        client.put(f"/api/users/{user.id}", headers=admin_headers, json={"is_active": False})
        assert self._me(client, tokens["access_token"]) == 401

    def test_revoke_rejects_foreign_tokens(self, app, db):
        foreign = pyjwt.encode({"sub": "1", "jti": "x" * 20, "type": "access", "exp": 9999999999,
                                "iss": "nocturna-trading-system"}, "not-our-key" * 4, algorithm="HS256")
        with app.app_context():
            from src.middleware.auth import token_manager
            from src.models.user import RevokedToken
            assert token_manager.revoke_token(foreign) is False
            assert RevokedToken.query.count() == 0

    def test_expired_revocations_are_purged(self, app, db):
        from datetime import datetime, timedelta

        from src.middleware.auth import token_manager
        from src.models.user import RevokedToken
        with app.app_context():
            _db_add(RevokedToken(jti="old", token_type="access",
                                 expires_at=datetime.now(UTC) - timedelta(minutes=1)))
            token_manager.revoke_token(token_manager.create_access_token("1"))
            assert db.session.get(RevokedToken, "old") is None
            assert RevokedToken.query.count() == 1


class TestUpdateUserValidation:
    @pytest.mark.parametrize("body", [
        {"roles": ["admin"]},
        {"roles": ["trader", "superuser"]},
        {"roles": "trader"},
        {"trading_mode": "LIVE"},
    ])
    def test_rejected(self, client, test_user, admin_headers, body):
        user, _ = test_user
        assert client.put(f"/api/users/{user.id}", headers=admin_headers, json=body).status_code == 400

    def test_rejected_update_changes_nothing(self, client, test_user, admin_headers):
        user, _ = test_user
        resp = client.put(f"/api/users/{user.id}", headers=admin_headers,
                          json={"is_active": False, "trading_disabled": True, "roles": ["admin"]})
        assert resp.status_code == 400
        with client.application.app_context():
            fresh = User.query.filter_by(username="testuser").first()
            assert fresh.is_active is True and fresh.trading_disabled is False

    def test_non_object_body_rejected(self, client, test_user, admin_headers):
        user, _ = test_user
        assert client.put(f"/api/users/{user.id}", headers=admin_headers, json=["roles"]).status_code == 400

    @pytest.mark.parametrize("body", [{"is_active": False}, {"is_admin": False}])
    def test_admin_cannot_lock_themselves_out(self, client, admin_user, admin_headers, body):
        admin, _ = admin_user
        assert client.put(f"/api/users/{admin.id}", headers=admin_headers, json=body).status_code == 400

    def test_roles_in_db_never_grant_admin(self, client, app, db):
        with app.app_context():
            user = User(username="sneaky", email="sneaky@example.com", is_active=True,
                        is_admin=False, roles=["admin", "trader"])
            user.set_password("Sn3aky!Passw0rd")
            _db_add(user)
            assert user.effective_roles() == ["trader"]
        resp = client.post("/api/auth/login", json={"username": "sneaky", "password": "Sn3aky!Passw0rd"})
        token = resp.get_json()["data"]["access_token"]
        assert client.get("/api/users", headers={"Authorization": f"Bearer {token}"}).status_code == 403


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
            assert admin.effective_roles() == ["trader", "admin"]
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


def _db_add(obj):
    from src.models.user import db
    db.session.add(obj)
    db.session.commit()


def _get_key(key_id):
    from src.models.user import db
    return db.session.get(APIKey, key_id)


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
