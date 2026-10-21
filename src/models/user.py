# FILE LOCATION: src/models/user.py
"""
NOCTURNA Trading System - Database Models
Production-grade SQLAlchemy models for user management and audit logging.
"""

from datetime import UTC, datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

#: Roles stored in User.roles. Admin rights come only from User.is_admin.
ASSIGNABLE_ROLES = frozenset({"trader", "reader"})
ADMIN_ROLE = "admin"


class UTCDateTime(TypeDecorator):
    """
    Timezone-safe datetime column.

    Values are stored as naive UTC (portable across SQLite and PostgreSQL) and
    always returned as timezone-aware UTC, so comparisons with
    ``datetime.now(UTC)`` never mix naive and aware values. Naive inputs are
    interpreted as UTC.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class User(db.Model):
    """User model for authentication and authorization."""

    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    # User roles and permissions
    roles = Column(JSON, default=list)
    trading_mode = Column(String(10), default='PAPER')  # PAPER or LIVE
    trading_disabled = Column(Boolean, default=False)

    # User settings
    settings = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(UTCDateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(UTCDateTime, default=lambda: datetime.now(UTC),
                        onupdate=lambda: datetime.now(UTC))
    last_login = Column(UTCDateTime)

    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(UTCDateTime)
    # Embedded in every JWT; bumping it invalidates all of the user's tokens
    token_version = Column(Integer, nullable=False, default=0)

    def effective_roles(self) -> list[str]:
        """Roles used for authorization: assignable roles, plus admin if is_admin."""
        roles = sorted(r for r in (self.roles or []) if r in ASSIGNABLE_ROLES)
        if self.is_admin:
            roles.append(ADMIN_ROLE)
        return roles

    def revoke_all_tokens(self) -> None:
        """Invalidate every access and refresh token issued to this user."""
        self.token_version = (self.token_version or 0) + 1

    def set_password(self, password: str) -> None:
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(
            password,
            method='pbkdf2:sha256',
            salt_length=32
        )

    def check_password(self, password: str) -> bool:
        """Verify the user's password."""
        return check_password_hash(self.password_hash, password)

    def is_locked(self) -> bool:
        """Check if the account is locked."""
        return bool(self.locked_until and self.locked_until > datetime.now(UTC))

    def lock_account(self, duration_minutes: int = 15) -> None:
        """Lock the user account for a specified duration."""
        from datetime import timedelta
        self.locked_until = datetime.now(UTC) + timedelta(minutes=duration_minutes)

    def unlock_account(self) -> None:
        """Unlock the user account."""
        self.locked_until = None
        self.failed_login_attempts = 0

    def reset_failed_logins(self) -> None:
        """Reset failed login counter on successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None

    def increment_failed_logins(self) -> None:
        """Increment failed login counter; an expired lock starts a fresh count."""
        if self.locked_until is not None and not self.is_locked():
            self.failed_login_attempts = 0
            self.locked_until = None
        self.failed_login_attempts = (self.failed_login_attempts or 0) + 1
        if self.failed_login_attempts >= 5:
            self.lock_account()

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Convert user to dictionary."""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'roles': self.roles,
            'trading_mode': self.trading_mode,
            'trading_disabled': self.trading_disabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

        if include_sensitive:
            data['settings'] = self.settings

        return data


class APIKey(db.Model):
    """API Key model for programmatic access."""

    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    # Public half of the key, used to find the record; the secret half is only
    # stored as a salted scrypt hash
    key_id = Column(String(16), nullable=False, unique=True, index=True)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)

    # Permissions
    permissions = Column(JSON, default=list)  # read, trade, admin

    # Status
    is_active = Column(Boolean, default=True)

    # Usage limits
    rate_limit = Column(Integer, default=100)  # requests per minute

    # Timestamps
    created_at = Column(UTCDateTime, default=lambda: datetime.now(UTC))
    expires_at = Column(UTCDateTime)
    last_used = Column(UTCDateTime)
    use_count = Column(Integer, default=0)

    # Relationship
    user = relationship('User', backref='api_key_records')

    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        return bool(self.expires_at and self.expires_at < datetime.now(UTC))

    def is_valid(self) -> bool:
        """Check if the API key is valid and active."""
        return bool(self.is_active) and not self.is_expired()

    def to_dict(self) -> dict:
        """Public metadata only — the key itself is never stored or returned."""
        return {
            'id': self.id,
            'key_id': self.key_id,
            'name': self.name,
            'description': self.description,
            'permissions': self.permissions or [],
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'use_count': self.use_count or 0,
        }

    def increment_usage(self) -> None:
        """Increment usage counter and update last used timestamp."""
        self.use_count = (self.use_count or 0) + 1
        self.last_used = datetime.now(UTC)


class RevokedToken(db.Model):
    """JWT IDs revoked before their expiry (logout, refresh rotation)."""

    __tablename__ = 'revoked_tokens'

    jti = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    token_type = Column(String(10), nullable=False)
    expires_at = Column(UTCDateTime, nullable=False, index=True)
    revoked_at = Column(UTCDateTime, default=lambda: datetime.now(UTC), nullable=False)


class AuditLog(db.Model):
    """Audit log model for tracking security-relevant events."""

    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(UTCDateTime, default=lambda: datetime.now(UTC), index=True)

    # Event details
    event_type = Column(String(50), nullable=False, index=True)
    success = Column(Boolean, default=True)
    event_data = Column(JSON, default=dict)

    # User information
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    username = Column(String(50))

    # Request information
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    request_id = Column(String(64), index=True)
    endpoint = Column(String(200))
    method = Column(String(10))

    # Response
    response_status = Column(Integer)
    error_message = Column(Text)

    # Relationship
    user = relationship('User', backref='audit_logs')

    def to_dict(self) -> dict:
        """Convert audit log to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'event_type': self.event_type,
            'success': self.success,
            'user_id': self.user_id,
            'username': self.username,
            'ip_address': self.ip_address,
            'request_id': self.request_id,
            'endpoint': self.endpoint,
            'response_status': self.response_status
        }


def create_admin_user(username: str, email: str, password: str) -> User:
    """
    Create the operator's admin account.

    Single-operator deployment: public registration is disabled, so this is
    the bootstrap path (exposed as the ``flask create-admin`` command).

    Args:
        username: Admin username (already validated and normalised).
        email: Admin email (already validated and normalised).
        password: Plain-text password; stored only as a salted hash.

    Returns:
        The persisted admin user.

    Raises:
        ValueError: If the username or email is already registered.
    """
    if User.query.filter((User.username == username) | (User.email == email)).first():
        raise ValueError(f"A user with username {username!r} or email {email!r} already exists")

    admin = User(
        username=username,
        email=email,
        is_active=True,
        is_admin=True,
        roles=['trader']
    )
    admin.set_password(password)

    db.session.add(admin)
    db.session.commit()

    return admin
