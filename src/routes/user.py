# FILE LOCATION: src/routes/user.py
"""
NOCTURNA Trading System - User Routes
Production-grade authentication and user management endpoints.
"""

import logging
from datetime import UTC, datetime, timedelta

import jwt
from flask import Blueprint, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from src.middleware.auth import (
    API_KEY_SCOPES,
    issue_api_key,
    require_admin,
    require_auth,
    token_manager,
)
from src.middleware.security import ip_manager
from src.models.user import ASSIGNABLE_ROLES, APIKey, AuditLog, User, db
from src.utils.audit import record_audit_event
from src.utils.validators import validate_login, validate_registration

user_bp = Blueprint('user', __name__)
logger = logging.getLogger(__name__)

# Hash checked when no real hash applies, so failed logins take the same time
_DUMMY_PASSWORD_HASH = generate_password_hash('nocturna-dummy-password', method='pbkdf2:sha256', salt_length=32)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def log_audit_event(event_type: str, success: bool, details: dict = None,
                    error: str = None):
    """Record an audit event (database and audit log file)."""
    record_audit_event(event_type, success, details, error)


def format_response(success: bool, data=None, error: str = None,
                    message: str = None, status_code: int = 200):
    """Format standardized API response."""
    response = {
        'success': success,
        'timestamp': datetime.now(UTC).isoformat(),
        'request_id': getattr(g, 'request_id', None)
    }

    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    if message is not None:
        response['message'] = message

    return jsonify(response), status_code


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@user_bp.route('/auth/login', methods=['POST'])
def login():
    """
    User login endpoint.
    Validates credentials and returns JWT tokens.
    """
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return format_response(False, error='Request body required', status_code=400)

        # Validate input
        is_valid, validated_data, errors = validate_login(data)
        if not is_valid:
            return format_response(False, error='Invalid input', data={'errors': errors}, status_code=400)

        # Check IP blacklist
        client_ip = request.remote_addr
        if ip_manager.is_blacklisted(client_ip):
            log_audit_event('LOGIN_FAILED', False, {'reason': 'IP blacklisted'})
            return format_response(False, error='Access denied', status_code=403)

        # Find user
        user = User.query.filter(
            (User.username == validated_data.username) |
            (User.email == validated_data.username)
        ).first()

        # Unknown, locked and inactive accounts all get the same response, and a
        # password hash is always checked, so callers cannot enumerate accounts
        if not user:
            check_password_hash(_DUMMY_PASSWORD_HASH, validated_data.password)
            ip_manager.record_failed_attempt(client_ip)
            log_audit_event('LOGIN_FAILED', False, {'reason': 'User not found'})
            return format_response(False, error='Invalid credentials', status_code=401)

        if user.is_locked() or not user.is_active:
            check_password_hash(_DUMMY_PASSWORD_HASH, validated_data.password)
            reason = 'Account locked' if user.is_locked() else 'Account inactive'
            log_audit_event('LOGIN_FAILED', False, {'user_id': user.id, 'reason': reason})
            return format_response(False, error='Invalid credentials', status_code=401)

        # Verify password
        if not user.check_password(validated_data.password):
            user.increment_failed_logins()
            db.session.commit()
            ip_manager.record_failed_attempt(client_ip)
            log_audit_event('LOGIN_FAILED', False, {'user_id': user.id, 'reason': 'Invalid password'})
            return format_response(False, error='Invalid credentials', status_code=401)

        # Reset failed logins
        user.reset_failed_logins()
        user.last_login = datetime.now(UTC)
        db.session.commit()

        # Clear any IP failed attempts
        ip_manager.clear_failed_attempts(client_ip)

        # Generate tokens
        tokens = token_manager.create_token_pair(user)

        log_audit_event('LOGIN_SUCCESS', True, {'user_id': user.id})

        return format_response(True, data={**tokens, 'user': user.to_dict()})

    except Exception as e:
        logger.error(f"Login error: {e}")
        log_audit_event('LOGIN_ERROR', False, {'error': str(e)})
        return format_response(False, error='Login failed', status_code=500)


@user_bp.route('/auth/refresh', methods=['POST'])
def refresh_token():
    """
    Exchange a refresh token for a new access and refresh token pair.

    Refresh tokens are single use: the presented token is revoked. Presenting
    an already-used refresh token is treated as theft and revokes every token
    of the user.
    """
    try:
        data = request.get_json(silent=True)
        raw_token = data.get('refresh_token') if isinstance(data, dict) else None
        if not isinstance(raw_token, str) or not raw_token:
            return format_response(False, error='Refresh token required', status_code=400)

        try:
            claims = token_manager.decode(raw_token, token_type='refresh')
        except jwt.InvalidTokenError:
            log_audit_event('TOKEN_REFRESH_FAILED', False, {'reason': 'invalid token'})
            return format_response(False, error='Invalid refresh token', status_code=401)

        sub = str(claims['sub'])
        user = db.session.get(User, int(sub)) if sub.isdigit() else None
        if user is None or not user.is_active or claims.get('ver', 0) != (user.token_version or 0):
            log_audit_event('TOKEN_REFRESH_FAILED', False, {'reason': 'stale or inactive'})
            return format_response(False, error='Invalid refresh token', status_code=401)

        if token_manager.is_revoked(claims['jti']):
            user.revoke_all_tokens()
            db.session.commit()
            log_audit_event('REFRESH_TOKEN_REUSE', False, {'user_id': user.id})
            return format_response(False, error='Invalid refresh token', status_code=401)

        token_manager.revoke_claims(claims)
        tokens = token_manager.create_token_pair(user)
        return format_response(True, data=tokens)

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        db.session.rollback()
        return format_response(False, error='Token refresh failed', status_code=500)


@user_bp.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """
    Revoke the bearer token of this request, and the refresh token if one is
    sent in the body as {"refresh_token": "..."}.
    """
    try:
        if getattr(g, 'auth_method', None) != 'jwt':
            return format_response(False, error='Logout requires a bearer token', status_code=400)

        token_manager.revoke_token(request.headers['Authorization'][7:], token_type='access')

        data = request.get_json(silent=True)
        raw_refresh = data.get('refresh_token') if isinstance(data, dict) else None
        if isinstance(raw_refresh, str) and raw_refresh:
            try:
                claims = token_manager.decode(raw_refresh, token_type='refresh', verify_exp=False)
            except jwt.InvalidTokenError:
                claims = None
            if claims is not None and str(claims['sub']) == str(g.user_id):
                token_manager.revoke_claims(claims)

        log_audit_event('LOGOUT', True, {'user_id': getattr(g, 'user_id', None)})

        return format_response(True, message='Logged out successfully')

    except Exception as e:
        logger.error(f"Logout error: {e}")
        db.session.rollback()
        return format_response(False, error='Logout failed', status_code=500)


@user_bp.route('/auth/logout-all', methods=['POST'])
@require_auth
def logout_all():
    """Revoke every access and refresh token of the caller (API keys are unaffected)."""
    try:
        user = db.session.get(User, int(g.user_id))
        if user is None:
            return format_response(False, error='User not found', status_code=404)
        user.revoke_all_tokens()
        db.session.commit()

        log_audit_event('LOGOUT_ALL', True, {'user_id': user.id})

        return format_response(True, message='All sessions revoked')

    except Exception as e:
        logger.error(f"Logout-all error: {e}")
        db.session.rollback()
        return format_response(False, error='Logout failed', status_code=500)


@user_bp.route('/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """Get current user information."""
    try:
        user_id = getattr(g, 'user_id', None)
        user = User.query.get(int(user_id))

        if not user:
            return format_response(False, error='User not found', status_code=404)

        return format_response(True, data={'user': user.to_dict()})

    except Exception as e:
        logger.error(f"Get current user error: {e}")
        return format_response(False, error='Failed to get user info', status_code=500)


# =============================================================================
# USER MANAGEMENT (Admin Only)
# =============================================================================

@user_bp.route('/users', methods=['GET'])
@require_auth
@require_admin
def list_users():
    """List all users (admin only)."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)  # Limit max per page

        users = User.query.paginate(page=page, per_page=per_page, error_out=False)

        return format_response(True, data={
            'users': [u.to_dict() for u in users.items],
            'total': users.total,
            'page': page,
            'per_page': per_page,
            'pages': users.pages
        })

    except Exception as e:
        logger.error(f"List users error: {e}")
        return format_response(False, error='Failed to list users', status_code=500)


@user_bp.route('/users/<int:user_id>', methods=['GET'])
@require_auth
@require_admin
def get_user(user_id):
    """Get user by ID (admin only)."""
    try:
        user = User.query.get(user_id)
        if not user:
            return format_response(False, error='User not found', status_code=404)

        return format_response(True, data={'user': user.to_dict(include_sensitive=True)})

    except Exception as e:
        logger.error(f"Get user error: {e}")
        return format_response(False, error='Failed to get user', status_code=500)


@user_bp.route('/users', methods=['POST'])
@require_auth
@require_admin
def create_user():
    """
    Create a user (admin only).
    Single-operator deployment: public self-registration is disabled.
    """
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return format_response(False, error='Request body required', status_code=400)

        # Validate input
        is_valid, validated_data, errors = validate_registration(data)
        if not is_valid:
            return format_response(False, error='Invalid input', data={'errors': errors}, status_code=400)

        # Check if username exists
        if User.query.filter_by(username=validated_data.username).first():
            return format_response(False, error='Username already exists', status_code=409)

        # Check if email exists
        if User.query.filter_by(email=validated_data.email).first():
            return format_response(False, error='Email already registered', status_code=409)

        # Create user
        user = User(
            username=validated_data.username,
            email=validated_data.email,
            is_active=True,
            is_admin=False,
            roles=['trader']
        )
        user.set_password(validated_data.password)

        db.session.add(user)
        db.session.commit()

        log_audit_event('USER_CREATED', True, {
            'user_id': user.id,
            'created_by': getattr(g, 'user_id', None)
        })

        return format_response(True, data={'user': user.to_dict()}, status_code=201)

    except Exception as e:
        logger.error(f"User creation error: {e}")
        db.session.rollback()
        return format_response(False, error='User creation failed', status_code=500)


def _user_update_error(data: dict, is_self: bool) -> str | None:
    """Validate an admin user update before anything is changed."""
    if is_self and 'is_active' in data and not data['is_active']:
        return 'You cannot deactivate your own account'
    if is_self and 'is_admin' in data and not data['is_admin']:
        return 'You cannot remove your own admin rights'
    if 'roles' in data:
        roles = data['roles']
        if not isinstance(roles, list) or not all(isinstance(r, str) and r in ASSIGNABLE_ROLES for r in roles):
            return f'roles must be a list drawn from {sorted(ASSIGNABLE_ROLES)}; use is_admin for admin rights'
    if 'trading_mode' in data and data['trading_mode'] != 'PAPER':
        return 'Only PAPER trading mode is available'
    return None


@user_bp.route('/users/<int:user_id>', methods=['PUT'])
@require_auth
@require_admin
def update_user(user_id):
    """Update user (admin only)."""
    try:
        user = User.query.get(user_id)
        if not user:
            return format_response(False, error='User not found', status_code=404)

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return format_response(False, error='JSON object body required', status_code=400)

        error = _user_update_error(data, is_self=user.id == int(getattr(g, 'user_id', 0)))
        if error:
            return format_response(False, error=error, status_code=400)

        if 'is_active' in data:
            user.is_active = bool(data['is_active'])
        if 'is_admin' in data:
            user.is_admin = bool(data['is_admin'])
        if 'roles' in data:
            user.roles = sorted(set(data['roles']))
        if 'trading_mode' in data:
            user.trading_mode = 'PAPER'
        if 'trading_disabled' in data:
            user.trading_disabled = bool(data['trading_disabled'])

        db.session.commit()

        log_audit_event('USER_UPDATED', True, {
            'user_id': user.id,
            'updated_by': getattr(g, 'user_id', None)
        })

        return format_response(True, data={'user': user.to_dict()})

    except Exception as e:
        logger.error(f"Update user error: {e}")
        db.session.rollback()
        return format_response(False, error='Failed to update user', status_code=500)


@user_bp.route('/users/<int:user_id>/password', methods=['PUT'])
@require_auth
def change_password(user_id):
    """Change user password."""
    try:
        # Users can only change their own password unless admin
        current_user_id = int(getattr(g, 'user_id', 0))
        current_user = User.query.get(current_user_id)

        if current_user_id != user_id and not (current_user and current_user.is_admin):
            return format_response(False, error='Unauthorized', status_code=403)

        user = User.query.get(user_id)
        if not user:
            return format_response(False, error='User not found', status_code=404)

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return format_response(False, error='Request body required', status_code=400)

        current_password = data.get('current_password')
        new_password = data.get('new_password')

        if not current_password or not new_password:
            return format_response(False, error='Current and new password required', status_code=400)

        # Verify current password (unless admin changing another user's password)
        if current_user_id == user_id and not user.check_password(current_password):
            return format_response(False, error='Current password incorrect', status_code=400)

        # Validate new password strength (same rules as registration)
        import re
        password_errors = []
        if len(new_password) < 12:
            password_errors.append('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', new_password):
            password_errors.append('Must contain at least one uppercase letter')
        if not re.search(r'[a-z]', new_password):
            password_errors.append('Must contain at least one lowercase letter')
        if not re.search(r'\d', new_password):
            password_errors.append('Must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password):
            password_errors.append('Must contain at least one special character')

        if password_errors:
            return format_response(
                False,
                error='Password does not meet strength requirements',
                data={'password_errors': password_errors},
                status_code=400
            )

        user.set_password(new_password)
        user.revoke_all_tokens()
        db.session.commit()

        log_audit_event('PASSWORD_CHANGED', True, {'user_id': user.id})

        # Every existing session is now invalid; a user changing their own
        # password gets a fresh pair so the current client keeps working.
        data = token_manager.create_token_pair(user) if current_user_id == user_id else None
        return format_response(True, data=data, message='Password changed successfully')

    except Exception as e:
        logger.error(f"Change password error: {e}")
        db.session.rollback()
        return format_response(False, error='Failed to change password', status_code=500)


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

@user_bp.route('/api-keys', methods=['GET'])
@require_auth
def list_api_keys():
    """List the caller's API keys (metadata only)."""
    try:
        user_id = int(getattr(g, 'user_id', 0))
        keys = APIKey.query.filter_by(user_id=user_id).order_by(APIKey.created_at.desc()).all()
        return format_response(True, data={'api_keys': [k.to_dict() for k in keys]})

    except Exception as e:
        logger.error(f"List API keys error: {e}")
        return format_response(False, error='Failed to list API keys', status_code=500)


@user_bp.route('/api-keys', methods=['POST'])
@require_auth
def create_api_key():
    """
    Create an API key for the caller.

    Body: name (str), description (str), permissions (subset of "read", "trade";
    default ["read"]), expires_in_days (1-365, optional).
    API keys can never carry admin rights.
    """
    try:
        user_id = int(getattr(g, 'user_id', 0))
        user = db.session.get(User, user_id)
        if not user:
            return format_response(False, error='User not found', status_code=404)

        data = request.get_json(silent=True)
        if data is None and not request.get_data(cache=True):
            data = {}  # no body at all: a default read-only key
        if not isinstance(data, dict):
            return format_response(False, error='JSON object body required', status_code=400)
        name = str(data.get('name') or 'API Key')[:100]
        description = str(data.get('description') or '')[:500]

        permissions = data.get('permissions', ['read'])
        if (not isinstance(permissions, list) or not permissions
                or not all(isinstance(p, str) and p in API_KEY_SCOPES for p in permissions)):
            return format_response(
                False, error=f'permissions must be a non-empty subset of {sorted(API_KEY_SCOPES)}',
                status_code=400
            )

        expires_at = None
        expires_in_days = data.get('expires_in_days')
        if expires_in_days is not None:
            if not isinstance(expires_in_days, int) or isinstance(expires_in_days, bool) \
                    or not 1 <= expires_in_days <= 365:
                return format_response(False, error='expires_in_days must be an integer from 1 to 365',
                                       status_code=400)
            expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        raw_key, key_id, key_hash = issue_api_key()
        record = APIKey(
            user_id=user.id,
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            description=description,
            permissions=sorted(set(permissions)),
            is_active=True,
            expires_at=expires_at,
        )
        db.session.add(record)
        db.session.commit()

        log_audit_event('API_KEY_CREATED', True, {'user_id': user.id, 'key_id': record.id,
                                                   'permissions': record.permissions})

        # The raw key is returned exactly once and never stored
        return format_response(True, data={
            'api_key': raw_key,
            'key': record.to_dict(),
            'message': 'Store this key securely. It will not be shown again.'
        }, status_code=201)

    except Exception as e:
        logger.error(f"Create API key error: {e}")
        db.session.rollback()
        return format_response(False, error='Failed to create API key', status_code=500)


@user_bp.route('/api-keys/<int:key_id>', methods=['DELETE'])
@require_auth
def revoke_api_key(key_id):
    """Revoke one of the caller's API keys."""
    try:
        user_id = int(getattr(g, 'user_id', 0))
        record = APIKey.query.filter_by(id=key_id, user_id=user_id).first()
        if record is None or not record.is_active:
            return format_response(False, error='API key not found', status_code=404)

        record.is_active = False
        db.session.commit()

        log_audit_event('API_KEY_REVOKED', True, {'user_id': user_id, 'key_id': key_id})

        return format_response(True, message='API key revoked')

    except Exception as e:
        logger.error(f"Revoke API key error: {e}")
        db.session.rollback()
        return format_response(False, error='Failed to revoke API key', status_code=500)


# =============================================================================
# AUDIT LOG
# =============================================================================

@user_bp.route('/audit-logs', methods=['GET'])
@require_auth
@require_admin
def get_audit_logs():
    """Get audit logs (admin only)."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        per_page = min(per_page, 100)

        event_type = request.args.get('event_type')
        user_id = request.args.get('user_id', type=int)

        query = AuditLog.query

        if event_type:
            query = query.filter(AuditLog.event_type == event_type)

        if user_id:
            query = query.filter(AuditLog.user_id == user_id)

        query = query.order_by(AuditLog.timestamp.desc())

        logs = query.paginate(page=page, per_page=per_page, error_out=False)

        return format_response(True, data={
            'logs': [log.to_dict() for log in logs.items],
            'total': logs.total,
            'page': page,
            'per_page': per_page,
            'pages': logs.pages
        })

    except Exception as e:
        logger.error(f"Get audit logs error: {e}")
        return format_response(False, error='Failed to get audit logs', status_code=500)
