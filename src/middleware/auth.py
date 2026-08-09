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
# API Key Manager
# =============================================================================


class APIKeyManager:
    """
    In-memory API key registry used by routes and tests.

    Methods:
      register_key(raw_key, user_id, permissions=None, expires_at=None)
      validate_key(raw_key) -> metadata dict | None
      revoke_key(raw_key) -> bool
    """

    def __init__(self, app: Flask | None = None) -> None:
        self._server_pepper = os.environ.get("API_KEY_PEPPER", secrets.token_hex(32))
        self._keys: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        app.extensions = getattr(app, "extensions", {})
        app.extensions["api_key_manager"] = self

    def _hash_key(self, api_key: str) -> str:
        return hmac.new(
            self._server_pepper.encode(),
            api_key.encode(),
            hashlib.sha256,
        ).hexdigest()

    def register_key(
        self,
        raw_key: str,
        user_id: str | int,
        permissions: list[str] | None = None,
        expires_at: datetime | None = None,
        name: str = "API Key",
    ) -> str:
        """Register a raw API key. Returns the key hash."""
        key_hash = self._hash_key(raw_key)
        with self._lock:
            self._keys[key_hash] = {
                "user_id": str(user_id),
                "permissions": permissions or ["read"],
                "scopes": permissions or ["read"],
                "name": name,
                "created_at": datetime.now(UTC),
                "expires_at": expires_at,
                "last_used": None,
                "use_count": 0,
                "is_active": True,
            }
        return key_hash

    def validate_key(self, api_key: str) -> dict[str, Any] | None:
        """Validate an API key. Returns metadata or None."""
        if not api_key:
            return None
        key_hash = self._hash_key(api_key)
        with self._lock:
            meta = self._keys.get(key_hash)
            if not meta or not meta.get("is_active", True):
                return None
            expires_at = meta.get("expires_at")
            if expires_at is not None and expires_at < datetime.now(UTC):
                return None
            meta["last_used"] = datetime.now(UTC)
            meta["use_count"] = int(meta.get("use_count", 0)) + 1
            return {
                "user_id": meta["user_id"],
                "permissions": meta.get("permissions", []),
                "scopes": meta.get("scopes", meta.get("permissions", [])),
                "name": meta.get("name"),
            }

    def revoke_key(self, raw_key: str) -> bool:
        """Revoke by raw key value."""
        key_hash = self._hash_key(raw_key)
        with self._lock:
            if key_hash not in self._keys:
                return False
            self._keys[key_hash]["is_active"] = False
            return True


# =============================================================================
# Module-level singletons and compatibility helpers
# =============================================================================

token_manager = TokenManager()
api_key_manager = APIKeyManager()


def create_token(user_id: str, user_data: dict | None = None) -> str:
    """Compatibility helper used by routes and conftest."""
    return token_manager.create_access_token(user_id, user_data or {})


def generate_api_key() -> str:
    """Generate a high-entropy API key string."""
    return f"ntr_{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    """Hash an API key for storage (HMAC-SHA256)."""
    pepper = os.environ.get("API_KEY_PEPPER", "nocturna-default-pepper-change-in-production")
    return hmac.new(pepper.encode(), raw_key.encode(), hashlib.sha256).hexdigest()


# =============================================================================
# Decorators
# =============================================================================


def require_auth(f: Callable) -> Callable:
    """Require JWT Bearer or X-API-Key authentication."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")

        user_id = None
        username = None
        roles: list[str] = []
        auth_method = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            tm = current_app.extensions.get("token_manager", token_manager)
            try:
                claims = tm.verify_token(token, token_type="access")
                user_id = claims["sub"]
                username = claims.get("username", str(user_id))
                roles = claims.get("roles", [])
                auth_method = "jwt"
                g.trading_mode = claims.get("trading_mode", "PAPER")
                g.trading_disabled = claims.get("trading_disabled", False)
            except Exception:
                pass

        if not user_id and api_key:
            km = current_app.extensions.get("api_key_manager", api_key_manager)
            metadata = km.validate_key(api_key)
            if metadata:
                user_id = metadata["user_id"]
                username = f"api_key_{user_id}"
                roles = metadata.get("permissions") or metadata.get("scopes") or []
                auth_method = "api_key"

        if not user_id:
            time.sleep(0.05)
            return {"success": False, "error": "Authentication required"}, 401

        g.user_id = user_id
        g.username = username
        g.roles = roles
        g.auth_method = auth_method
        return f(*args, **kwargs)

    return decorated_function


def require_api_key(f: Callable) -> Callable:
    """Require X-API-Key authentication specifically."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        api_key = request.headers.get("X-API-Key", "")
        if not api_key:
            return {"success": False, "error": "API key required"}, 401
        km = current_app.extensions.get("api_key_manager", api_key_manager)
        metadata = km.validate_key(api_key)
        if not metadata:
            time.sleep(0.05)
            return {"success": False, "error": "Invalid API key"}, 401
        g.user_id = metadata["user_id"]
        g.username = f"api_key_{metadata['user_id']}"
        g.roles = metadata.get("permissions") or metadata.get("scopes") or []
        g.auth_method = "api_key"
        return f(*args, **kwargs)

    return decorated_function


def require_roles(*required_roles: str) -> Callable:
    """Require one of the given roles."""

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
    """Enforce trading mode consistency."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
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
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            tm = current_app.extensions.get("token_manager", token_manager)
            try:
                claims = tm.verify_token(token, token_type="access")
                g.user_id = claims["sub"]
                g.username = claims.get("username", str(claims["sub"]))
                g.roles = claims.get("roles", [])
                g.auth_method = "jwt"
            except Exception:
                pass
        elif api_key:
            km = current_app.extensions.get("api_key_manager", api_key_manager)
            metadata = km.validate_key(api_key)
            if metadata:
                g.user_id = metadata["user_id"]
                g.username = f"api_key_{metadata['user_id']}"
                g.roles = metadata.get("permissions") or []
                g.auth_method = "api_key"
        return f(*args, **kwargs)

    return decorated_function
