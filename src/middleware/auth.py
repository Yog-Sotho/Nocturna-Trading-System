# =============================================================================
# NOCTURNA Trading System — Authentication Middleware
# Audit Remediation: F-242 — Unify API-key storage (DB as source of truth)
# =============================================================================
"""
Production-grade authentication middleware for NOCTURNA v2.0.
Handles JWT tokens, API keys, and session management with proper security controls.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

import jwt
from flask import Flask, current_app, g, request

logger = logging.getLogger(__name__)


# =============================================================================
# Token Manager
# =============================================================================


class TokenManager:
    """
    Manages JWT access and refresh tokens with proper lifecycle controls.

    Public method signatures match routes and the test suite:
      create_access_token(user_id, data=None)
      create_refresh_token(user_id)
      verify_token(token, token_type="access")  # raises on failure
      revoke_token(token)
    """

    def __init__(self, app: Flask | None = None) -> None:
        self._jwt_secret: str | None = None
        self._jwt_algorithm = "HS256"
        self.access_token_expires = timedelta(hours=1)
        self.refresh_token_expires = timedelta(days=7)
        self._blacklist: set[str] = set()
        self._blacklist_lock = threading.RLock()
        self._redis = None
        self._blacklist_prefix = "token_blacklist:"
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        """Initialize with Flask app context."""
        self._jwt_secret = app.config.get("JWT_SECRET_KEY") or app.config.get("SECRET_KEY")
        if not self._jwt_secret:
            if app.config.get("ENV") == "production" or os.environ.get("FLASK_ENV") == "production":
                raise ValueError("JWT_SECRET_KEY must be explicitly set in production")
            logger.warning("JWT_SECRET_KEY not set; generating ephemeral key (dev/test only)")
            self._jwt_secret = secrets.token_hex(32)

        if app.config.get("JWT_ACCESS_TOKEN_EXPIRES_MINUTES"):
            self.access_token_expires = timedelta(
                minutes=int(app.config["JWT_ACCESS_TOKEN_EXPIRES_MINUTES"])
            )
        elif app.config.get("JWT_ACCESS_TOKEN_EXPIRES"):
            exp = app.config["JWT_ACCESS_TOKEN_EXPIRES"]
            if isinstance(exp, timedelta):
                self.access_token_expires = exp
        else:
            hours = int(os.environ.get("JWT_EXPIRATION_HOURS", 24))
            self.access_token_expires = timedelta(hours=max(hours, 1))

        app.extensions = getattr(app, "extensions", {})
        app.extensions["token_manager"] = self

    def _encode(self, claims: dict) -> str:
        if not self._jwt_secret:
            raise RuntimeError("JWT secret not configured — call init_app first")
        return jwt.encode(claims, self._jwt_secret, algorithm=self._jwt_algorithm)

    def create_access_token(self, user_id: str | int, data: dict | None = None) -> str:
        """Create a JWT access token. user_id is stored as string in 'sub'."""
        now = datetime.now(UTC)
        jti = secrets.token_urlsafe(32)
        data = data or {}
        claims: dict[str, Any] = {
            "sub": str(user_id),
            "iat": now,
            "exp": now + self.access_token_expires,
            "jti": jti,
            "type": "access",
            "iss": "nocturna-trading-system",
            "username": data.get("username", str(user_id)),
            "roles": data.get("roles", []),
        }
        for key, value in data.items():
            if key not in claims:
                claims[key] = value
        return self._encode(claims)

    def create_refresh_token(self, user_id: str | int, username: str | None = None) -> str:
        """Create a JWT refresh token."""
        now = datetime.now(UTC)
        jti = secrets.token_urlsafe(32)
        claims = {
            "sub": str(user_id),
            "username": username or str(user_id),
            "iat": now,
            "exp": now + self.refresh_token_expires,
            "jti": jti,
            "type": "refresh",
            "iss": "nocturna-trading-system",
        }
        return self._encode(claims)

    def verify_token(self, token: str, token_type: str = "access") -> dict[str, Any]:
        """
        Verify and decode a JWT token.

        Raises jwt.InvalidTokenError / jwt.ExpiredSignatureError on failure
        (matches test suite expectations).
        """
        if not self._jwt_secret:
            raise jwt.InvalidTokenError("JWT secret not configured")

        claims = jwt.decode(
            token,
            self._jwt_secret,
            algorithms=[self._jwt_algorithm],
            options={"verify_signature": True, "require": ["exp", "jti", "type"]},
            issuer="nocturna-trading-system",
        )

        if claims.get("type") != token_type:
            raise jwt.InvalidTokenError(
                f"Token type mismatch: expected {token_type}, got {claims.get('type')}"
            )

        jti = claims.get("jti")
        if jti and self._is_blacklisted(jti):
            raise jwt.InvalidTokenError("Token has been revoked")

        return claims

    def _is_blacklisted(self, jti: str) -> bool:
        with self._blacklist_lock:
            if jti in self._blacklist:
                return True
        if self._redis:
            try:
                return self._redis.exists(f"{self._blacklist_prefix}{jti}") == 1
            except Exception as exc:
                logger.error("Redis blacklist check failed: %s", exc)
        return False

    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding its JTI to the blacklist."""
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            jti = unverified.get("jti")
            if not jti:
                return False
            with self._blacklist_lock:
                self._blacklist.add(jti)
            if self._redis:
                try:
                    exp = unverified.get("exp")
                    ttl = (
                        max(0, int(exp - time.time()))
                        if exp
                        else int(self.refresh_token_expires.total_seconds())
                    )
                    self._redis.setex(f"{self._blacklist_prefix}{jti}", ttl, "1")
                except Exception as exc:
                    logger.error("Failed to persist revoked token: %s", exc)
            return True
        except Exception as exc:
            logger.error("Error revoking token: %s", exc)
            return False

    def revoke_all_user_tokens(self, user_id: int | str) -> int:
        """Best-effort marker for password-change invalidation."""
        logger.info("All tokens for user %s marked for invalidation", user_id)
        return 0


# =============================================================================
# API keys (database-backed)
# =============================================================================

API_KEY_PREFIX = "ntr_"
API_KEY_SCOPES = frozenset({"read", "trade"})

#: Roles an API key can carry. Keys never grant admin.
READER_ROLE = "reader"
TRADER_ROLE = "trader"


def generate_api_key() -> str:
    """Generate a high-entropy API key string."""
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def _api_key_pepper() -> bytes:
    """
    Server-side secret mixed into API key hashes.

    Uses API_KEY_PEPPER when set (required in production). Otherwise it is
    derived from SECRET_KEY, so keys stay valid as long as SECRET_KEY does.
    """
    settings = current_app.extensions.get("settings")
    if settings is not None and settings.api_key_pepper is not None:
        return settings.api_key_pepper.get_secret_value().encode()
    return hmac.new(
        current_app.config["SECRET_KEY"].encode(), b"nocturna-api-key-pepper", hashlib.sha256
    ).hexdigest().encode()


def hash_api_key(raw_key: str) -> str:
    """Hash an API key for storage and lookup (HMAC-SHA256 with the server pepper)."""
    return hmac.new(_api_key_pepper(), raw_key.encode(), hashlib.sha256).hexdigest()


def roles_for_api_key(scopes: list[str], owner_roles: list[str], owner_is_admin: bool) -> list[str]:
    """
    Roles granted by a key: never more than its scopes, never more than its owner.

    "read" grants the reader role; "trade" adds the trader role only if the
    owner can trade. Admin is never granted through an API key.
    """
    roles = [READER_ROLE]
    can_trade = owner_is_admin or TRADER_ROLE in (owner_roles or [])
    if "trade" in scopes and can_trade:
        roles.append(TRADER_ROLE)
    return roles


def authenticate_api_key(raw_key: str) -> tuple[str, str, list[str]] | None:
    """
    Validate a raw API key against the database.

    Returns:
        (user_id, username, roles) for an active, unexpired key whose owner is
        active, otherwise None.
    """
    from src.models.user import APIKey, db

    if not raw_key or not raw_key.startswith(API_KEY_PREFIX):
        return None

    record = APIKey.query.filter_by(key_hash=hash_api_key(raw_key)).first()
    if record is None or not record.is_valid():
        return None
    owner = record.user
    if owner is None or not owner.is_active:
        return None

    record.increment_usage()
    db.session.commit()
    return str(owner.id), owner.username, roles_for_api_key(
        record.permissions or [], owner.roles or [], bool(owner.is_admin)
    )


# =============================================================================
# Module-level singletons and compatibility helpers
# =============================================================================

token_manager = TokenManager()


def create_token(user_id: str, user_data: dict | None = None) -> str:
    """Compatibility helper used by routes and conftest."""
    return token_manager.create_access_token(user_id, user_data or {})


# =============================================================================
# Decorators
# =============================================================================


def _authenticate_request() -> bool:
    """Populate g.user_id/username/roles/auth_method from a bearer token or API key."""
    auth_header = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")

    if auth_header.startswith("Bearer "):
        tm = current_app.extensions.get("token_manager", token_manager)
        try:
            claims = tm.verify_token(auth_header[7:], token_type="access")
        except jwt.InvalidTokenError:
            claims = None
        if claims is not None:
            g.user_id = claims["sub"]
            g.username = claims.get("username", str(claims["sub"]))
            g.roles = claims.get("roles", [])
            g.auth_method = "jwt"
            g.trading_mode = claims.get("trading_mode", "PAPER")
            return True

    if api_key:
        result = authenticate_api_key(api_key)
        if result is not None:
            g.user_id, g.username, g.roles = result
            g.auth_method = "api_key"
            return True

    return False


def require_auth(f: Callable) -> Callable:
    """Require JWT Bearer or X-API-Key authentication."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if not _authenticate_request():
            time.sleep(0.05)
            return {"success": False, "error": "Authentication required"}, 401
        return f(*args, **kwargs)

    return decorated_function


def require_roles(*required_roles: str) -> Callable:
    """Require one of the given roles (admin satisfies every role)."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            user_roles = getattr(g, "roles", []) or []
            if "admin" in user_roles:
                return f(*args, **kwargs)
            if not any(role in user_roles for role in required_roles):
                return {"success": False, "error": "Insufficient permissions"}, 403
            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_admin(f: Callable) -> Callable:
    """Require admin role."""
    return require_roles("admin")(f)


def require_trading_permissions(f: Callable) -> Callable:
    """
    Require the trader role, an account allowed to trade, and a consistent mode.

    The trading_disabled flag is read from the database on every request, so
    an admin disabling an account takes effect immediately.
    """
    from src.models.user import User, db

    @wraps(f)
    @require_roles(TRADER_ROLE)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        user = db.session.get(User, int(g.user_id)) if str(g.user_id).isdigit() else None
        if user is None or not user.is_active or user.trading_disabled:
            return {"success": False, "error": "Trading is disabled for this account"}, 403

        trading_mode = os.environ.get("TRADING_MODE", "PAPER").upper()
        user_trading_mode = getattr(g, "trading_mode", "PAPER")
        if user_trading_mode == "LIVE" and trading_mode != "LIVE":
            return {"success": False, "error": "Live trading not permitted on this server"}, 403
        return f(*args, **kwargs)

    return decorated_function


def optional_auth(f: Callable) -> Callable:
    """Attempt authentication but do not fail if missing."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        _authenticate_request()
        return f(*args, **kwargs)

    return decorated_function
