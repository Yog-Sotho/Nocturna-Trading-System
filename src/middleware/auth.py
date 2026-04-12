"""
NOCTURNA Trading System - Authentication Middleware
Production-grade JWT authentication and authorization.
"""

import os
import sys
import logging
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional, Dict, Any, Callable

from flask import request, g, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# =============================================================================
# JWT TOKEN MANAGEMENT
# =============================================================================

class TokenManager:
    """
    Manages JWT token creation, validation, and refresh operations.
    Implements secure token handling with blacklist support.
    """

    def __init__(self, app=None):
        self.app = app
        self.secret_key = None
        self.algorithm = 'HS256'
        self.access_token_expires = timedelta(hours=24)
        self.refresh_token_expires = timedelta(days=30)
        self.token_blacklist = set()
        self.max_token_age = timedelta(hours=24)

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize with Flask app."""
        self.app = app
        self.secret_key = app.config.get('JWT_SECRET_KEY', app.config['SECRET_KEY'])
        self.access_token_expires = app.config.get(
            'JWT_ACCESS_TOKEN_EXPIRES',
            timedelta(hours=24)
        )
        self.refresh_token_expires = app.config.get(
            'JWT_REFRESH_TOKEN_EXPIRES',
            timedelta(days=30)
        )

    def create_access_token(self, identity: str, user_data: Optional[Dict] = None,
                           additional_claims: Optional[Dict] = None) -> str:
        """
        Create a new JWT access token.

        Args:
            identity: Unique user identifier (user_id, username, API key)
            user_data: Additional user information to include in token
            additional_claims: Extra claims to include in token payload

        Returns:
            Encoded JWT token string
        """
        now = datetime.now(timezone.utc)
        payload = {
            'iss': 'nocturna-trading-system',
            'sub': str(identity),
            'iat': now,
            'exp': now + self.access_token_expires,
            'nbf': now,
            'jti': secrets.token_hex(16),
            'type': 'access',
            'user_data': user_data or {},
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def create_refresh_token(self, identity: str) -> str:
        """
        Create a new JWT refresh token.

        Args:
            identity: Unique user identifier

        Returns:
            Encoded JWT refresh token string
        """
        now = datetime.now(timezone.utc)
        payload = {
            'iss': 'nocturna-trading-system',
            'sub': str(identity),
            'iat': now,
            'exp': now + self.refresh_token_expires,
            'nbf': now,
            'jti': secrets.token_hex(16),
            'type': 'refresh',
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str, token_type: str = 'access') -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: Encoded JWT token
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            Decoded token payload

        Raises:
            jwt.InvalidTokenError: If token is invalid
            jwt.ExpiredSignatureError: If token has expired
            jwt.InvalidTokenError: If token type doesn't match
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={'require': ['sub', 'exp', 'iat', 'jti', 'type']}
            )

            # Check token type
            if payload.get('type') != token_type:
                raise jwt.InvalidTokenError(f"Token type mismatch: expected {token_type}")

            # Check if token is blacklisted
            if payload.get('jti') in self.token_blacklist:
                raise jwt.InvalidTokenError("Token has been revoked")

            # Check if token is not yet valid
            now = datetime.now(timezone.utc)
            nbf = payload.get('nbf')
            if nbf and datetime.fromtimestamp(nbf, tz=timezone.utc) > now:
                raise jwt.InvalidTokenError("Token not yet valid")

            return payload

        except jwt.ExpiredSignatureError:
            current_app.logger.warning(f"Expired token: {request.path}")
            raise
        except jwt.InvalidTokenError as e:
            current_app.logger.warning(f"Invalid token: {e} | Path: {request.path}")
            raise

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by adding its JTI to the blacklist.

        Args:
            token: Encoded JWT token to revoke

        Returns:
            True if token was revoked, False if token was invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={'verify_exp': False}
            )
            jti = payload.get('jti')
            if jti:
                self.token_blacklist.add(jti)
                current_app.logger.info(f"Token revoked: {jti}")
                return True
            return False
        except jwt.InvalidTokenError:
            return False

    def get_token_identity(self, token: str) -> Optional[str]:
        """
        Extract identity from token without full verification.
        Used for logging and audit purposes.

        Args:
            token: Encoded JWT token

        Returns:
            Token subject (user identity) or None
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={'verify_exp': False, 'verify_signature': False}
            )
            return payload.get('sub')
        except jwt.InvalidTokenError:
            return None


# =============================================================================
# GLOBAL TOKEN MANAGER INSTANCE
# =============================================================================

token_manager = TokenManager()


def create_token(identity: str, user_data: Optional[Dict] = None,
                additional_claims: Optional[Dict] = None) -> str:
    """Create access token shorthand."""
    return token_manager.create_access_token(identity, user_data, additional_claims)


def verify_token(token: str, token_type: str = 'access') -> Dict[str, Any]:
    """Verify token shorthand."""
    return token_manager.verify_token(token, token_type)


def generate_api_key() -> str:
    """Generate a secure API key for programmatic access."""
    return f"ntr_{secrets.token_hex(24)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return generate_password_hash(api_key, method='pbkdf2:sha256', salt_length=32)


# =============================================================================
# AUTHENTICATION DECORATORS
# =============================================================================

def require_auth(f: Callable) -> Callable:
    """
    Decorator that requires valid JWT authentication.

    Usage:
        @app.route('/protected')
        @require_auth
        def protected_endpoint():
            user_id = g.user_id
            return jsonify({'message': f'Hello {user_id}'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        api_key = request.headers.get('X-API-Key', '')

        token = None
        token_type = 'access'

        # Check Authorization header (Bearer token)
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        # Check API Key header
        elif api_key:
            # API key authentication - create temporary token
            # In production, validate API key against database
            return jsonify({
                'success': False,
                'error': 'API key authentication requires validation against server',
                'request_id': getattr(g, 'request_id', None)
            }), 401
        else:
            current_app.logger.warning(
                f"Missing authentication: {request.path} from {request.remote_addr}"
            )
            return jsonify({
                'success': False,
                'error': 'Authentication required. Provide Bearer token or API key.',
                'request_id': getattr(g, 'request_id', None)
            }), 401

        try:
            # Verify and decode token
            payload = token_manager.verify_token(token, token_type='access')

            # Set user context
            g.user_id = payload.get('sub')
            g.user_data = payload.get('user_data', {})
            g.token_jti = payload.get('jti')
            g.token_type = payload.get('type')

            # Log successful authentication
            current_app.logger.info(
                f"Authenticated request: {request.path} | User: {g.user_id} | "
                f"Request ID: {getattr(g, 'request_id', 'unknown')}"
            )

            return f(*args, **kwargs)

        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'error': 'Token has expired. Please login again.',
                'error_code': 'TOKEN_EXPIRED',
                'request_id': getattr(g, 'request_id', None)
            }), 401

        except jwt.InvalidTokenError as e:
            return jsonify({
                'success': False,
                'error': f'Invalid token: {str(e)}',
                'error_code': 'INVALID_TOKEN',
                'request_id': getattr(g, 'request_id', None)
            }), 401

    return decorated


def require_admin(f: Callable) -> Callable:
    """
    Decorator that requires admin role authentication.

    Must be used after @require_auth decorator.

    Usage:
        @app.route('/admin-only')
        @require_auth
        @require_admin
        def admin_endpoint():
            return jsonify({'message': 'Admin access granted'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check if user_data contains admin role
        user_data = getattr(g, 'user_data', {})
        roles = user_data.get('roles', [])

        if 'admin' not in roles and user_data.get('is_admin') is not True:
            current_app.logger.warning(
                f"Non-admin user attempted admin access: {getattr(g, 'user_id', 'unknown')} | "
                f"Path: {request.path} | Request ID: {getattr(g, 'request_id', 'unknown')}"
            )
            return jsonify({
                'success': False,
                'error': 'Admin access required',
                'error_code': 'FORBIDDEN',
                'request_id': getattr(g, 'request_id', None)
            }), 403

        return f(*args, **kwargs)

    return decorated


def require_trading_permissions(f: Callable) -> Callable:
    """
    Decorator that requires trading permissions.

    Must be used after @require_auth decorator.

    Checks if user has trading enabled and appropriate permissions level.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        user_data = getattr(g, 'user_data', {})

        # Check if trading is enabled for user
        if user_data.get('trading_disabled') is True:
            current_app.logger.warning(
                f"Trading attempt by disabled user: {getattr(g, 'user_id', 'unknown')} | "
                f"Request ID: {getattr(g, 'request_id', 'unknown')}"
            )
            return jsonify({
                'success': False,
                'error': 'Trading is disabled for this account',
                'error_code': 'TRADING_DISABLED',
                'request_id': getattr(g, 'request_id', None)
            }), 403

        # Check trading mode restrictions
        trading_mode = user_data.get('trading_mode', 'LIVE')
        current_mode = os.environ.get('TRADING_MODE', 'PAPER')

        if trading_mode != current_mode:
            if current_mode == 'LIVE' and trading_mode != 'LIVE':
                current_app.logger.warning(
                    f"Live trading attempt by PAPER-only user: {getattr(g, 'user_id', 'unknown')} | "
                    f"Request ID: {getattr(g, 'request_id', 'unknown')}"
                )
                return jsonify({
                    'success': False,
                    'error': 'User not authorized for live trading',
                    'error_code': 'LIVE_TRADING_FORBIDDEN',
                    'request_id': getattr(g, 'request_id', None)
                }), 403

        return f(*args, **kwargs)

    return decorated


def optional_auth(f: Callable) -> Callable:
    """
    Decorator that optionally authenticates the request.

    If a valid token is provided, user context is set.
    If no token or invalid token, request continues without user context.

    Usage:
        @app.route('/optional-auth')
        @optional_auth
        def optional_endpoint():
            if hasattr(g, 'user_id'):
                return jsonify({'message': f'Hello {g.user_id}'})
            else:
                return jsonify({'message': 'Hello anonymous'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            try:
                payload = token_manager.verify_token(token, token_type='access')
                g.user_id = payload.get('sub')
                g.user_data = payload.get('user_data', {})
                g.token_jti = payload.get('jti')
                g.token_type = payload.get('type')
            except (jwt.InvalidTokenError, jwt.ExpiredSignatureError):
                # Silently continue without authentication
                pass

        return f(*args, **kwargs)

    return decorated


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

class APIKeyManager:
    """
    Manages API key authentication for programmatic access.
    Stores hashed keys in database for security.
    """

    def __init__(self):
        self.valid_keys = {}
        self.key_metadata = {}

    def register_key(self, key: str, user_id: str, permissions: list = None,
                    description: str = '', expires_at: datetime = None) -> None:
        """
        Register an API key for a user.

        Args:
            key: The API key (will be hashed before storage)
            user_id: Owner of the API key
            permissions: List of allowed operations
            description: Human-readable description
            expires_at: Optional expiration datetime
        """
        key_hash = hash_api_key(key)
        self.valid_keys[key_hash] = user_id
        self.key_metadata[key_hash] = {
            'user_id': user_id,
            'permissions': permissions or ['read'],
            'description': description,
            'created_at': datetime.now(timezone.utc),
            'expires_at': expires_at,
            'last_used': None,
            'use_count': 0
        }

    def validate_key(self, key: str) -> Optional[Dict]:
        """
        Validate an API key and return its metadata.

        Args:
            key: The API key to validate

        Returns:
            Key metadata dict if valid, None if invalid
        """
        key_hash = hash_api_key(key)

        if key_hash not in self.valid_keys:
            return None

        metadata = self.key_metadata.get(key_hash, {})

        # Check expiration
        if metadata.get('expires_at'):
            if datetime.now(timezone.utc) > metadata['expires_at']:
                current_app.logger.warning(f"Expired API key used")
                return None

        # Update usage statistics
        metadata['last_used'] = datetime.now(timezone.utc)
        metadata['use_count'] = metadata.get('use_count', 0) + 1

        return {
            'user_id': self.valid_keys[key_hash],
            'permissions': metadata.get('permissions', []),
            'created_at': metadata.get('created_at'),
            'last_used': metadata.get('last_used'),
            'use_count': metadata.get('use_count', 0)
        }

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = hash_api_key(key)
        if key_hash in self.valid_keys:
            del self.valid_keys[key_hash]
            if key_hash in self.key_metadata:
                del self.key_metadata[key_hash]
            return True
        return False


api_key_manager = APIKeyManager()


def require_api_key(f: Callable) -> Callable:
    """
    Decorator that requires valid API key authentication.

    Usage:
        @app.route('/api-endpoint')
        @require_api_key
        def api_endpoint():
            return jsonify({'message': 'API access granted'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key', '')

        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key required. Provide X-API-Key header.',
                'request_id': getattr(g, 'request_id', None)
            }), 401

        metadata = api_key_manager.validate_key(api_key)

        if not metadata:
            current_app.logger.warning(
                f"Invalid API key attempt from {request.remote_addr} | "
                f"Path: {request.path} | Request ID: {getattr(g, 'request_id', 'unknown')}"
            )
            return jsonify({
                'success': False,
                'error': 'Invalid API key',
                'request_id': getattr(g, 'request_id', None)
            }), 401

        # Set user context from API key
        g.user_id = metadata['user_id']
        g.user_permissions = metadata['permissions']
        g.auth_method = 'api_key'

        return f(*args, **kwargs)

    return decorated
