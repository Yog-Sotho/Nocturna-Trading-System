"""
NOCTURNA Trading System - Database Models
Production-grade SQLAlchemy models for user management and audit logging.
"""

import os
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship


db = SQLAlchemy()


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

    # API keys for programmatic access
    api_keys = Column(JSON, default=list)

    # User settings
    settings = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime)

    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)

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
        if self.locked_until and self.locked_until > datetime.now(timezone.utc):
            return True
        return False

    def lock_account(self, duration_minutes: int = 15) -> None:
        """Lock the user account for a specified duration."""
        from datetime import timedelta
        self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)

    def unlock_account(self) -> None:
        """Unlock the user account."""
        self.locked_until = None
        self.failed_login_attempts = 0

    def reset_failed_logins(self) -> None:
        """Reset failed login counter on successful login."""
        self.failed_login_attempts = 0
        self.locked_until = None

    def increment_failed_logins(self) -> None:
        """Increment failed login counter."""
        self.failed_login_attempts += 1
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
    key_hash = Column(String(256), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)

    # Permissions
    permissions = Column(JSON, default=list)  # read, trade, admin

    # Status
    is_active = Column(Boolean, default=True)

    # Usage limits
    rate_limit = Column(Integer, default=100)  # requests per minute

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    use_count = Column(Integer, default=0)

    # Relationship
    user = relationship('User', backref='api_key_records')

    def is_expired(self) -> bool:
        """Check if the API key is expired."""
        if self.expires_at and self.expires_at < datetime.now(timezone.utc):
            return True
        return False

    def is_valid(self) -> bool:
        """Check if the API key is valid and active."""
        return self.is_active and not self.is_expired()

    def increment_usage(self) -> None:
        """Increment usage counter and update last used timestamp."""
        self.use_count += 1
        self.last_used = datetime.now(timezone.utc)


class AuditLog(db.Model):
    """Audit log model for tracking security-relevant events."""

    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

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
    request_id = Column(String(32), index=True)
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


class TradeRecord(db.Model):
    """Trade record model for tracking executed trades."""

    __tablename__ = 'trade_records'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Trade details
    order_id = Column(String(100), index=True)
    symbol = Column(String(10), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy or sell
    order_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    filled_quantity = Column(Float, default=0)
    avg_fill_price = Column(Float, default=0)

    # Execution
    broker_order_id = Column(String(100))
    status = Column(String(20), nullable=False)

    # P&L
    realized_pnl = Column(Float, default=0)
    commission = Column(Float, default=0)

    # Metadata
    metadata = Column(JSON, default=dict)

    # User
    user_id = Column(Integer, ForeignKey('users.id'), index=True)

    # Trading mode used
    trading_mode = Column(String(20))

    def to_dict(self) -> dict:
        """Convert trade record to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'avg_fill_price': self.avg_fill_price,
            'status': self.status,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'trading_mode': self.trading_mode
        }


class SystemConfig(db.Model):
    """System configuration model for storing runtime configuration."""

    __tablename__ = 'system_config'

    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text)
    value_json = Column(JSON)

    # Metadata
    description = Column(Text)
    category = Column(String(50), index=True)

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(Integer, ForeignKey('users.id'))

    def get_value(self):
        """Get the configuration value, parsing JSON if needed."""
        if self.value_json is not None:
            return self.value_json
        return self.value


# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================

def create_default_admin(app=None):
    """Create default admin user if none exists."""
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@nocturna.local')
    admin_password = os.environ.get('ADMIN_PASSWORD', None)

    # Check if admin exists
    admin = User.query.filter_by(username=admin_username).first()
    if admin:
        return admin

    # Create admin user
    admin = User(
        username=admin_username,
        email=admin_email,
        is_active=True,
        is_admin=True,
        roles=['admin', 'trader']
    )

    if admin_password:
        admin.set_password(admin_password)
    else:
        # Generate a random password — print to stderr ONCE, never persist in logs
        import secrets
        import sys as _sys
        temp_password = secrets.token_urlsafe(16)
        admin.set_password(temp_password)
        print(
            f"\n{'='*60}\n"
            f"DEFAULT ADMIN CREATED\n"
            f"Username: {admin_username}\n"
            f"Password: {temp_password}\n"
            f"CHANGE THIS IMMEDIATELY!\n"
            f"{'='*60}\n",
            file=_sys.stderr,
            flush=True
        )

    db.session.add(admin)
    db.session.commit()

    return admin
