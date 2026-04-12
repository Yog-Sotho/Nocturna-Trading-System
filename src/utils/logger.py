"""
NOCTURNA Trading System - Secure Logging Utilities
Production-grade logging with rotation, sanitization, and audit trails.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime, timezone
from typing import Optional
from functools import wraps
import traceback
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class SecureLogFormatter(logging.Formatter):
    """
    Custom formatter that sanitizes log messages.
    Removes sensitive data before logging.
    """

    # Patterns to redact from logs
    REDACT_PATTERNS = [
        (r'(password|secret|key|token|api[_-]?key)["\']?\s*[:=]\s*["\']?[\w\-]+["\']?', r'\1: [REDACTED]'),
        (r'(bearer|auth)["\']?\s*[:=]\s*["\']?[\w\-.\s]+["\']?', r'\1: [REDACTED]'),
        (r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}', '[CARD_NUMBER]'),
        (r'\d{3}-\d{2}-\d{4}', '[SSN]'),
        (r'(?<=api_key=)[^&\s]+', '[API_KEY]'),
        (r'(?<=secret_key=)[^&\s]+', '[SECRET_KEY]'),
    ]

    # Date format for logs
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, include_caller_info: bool = True):
        super().__init__(datefmt=self.DATE_FORMAT)
        self.include_caller_info = include_caller_info

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sanitization."""
        # Sanitize the message
        record.msg = self._sanitize_message(str(record.msg))

        # Add caller info if enabled
        if self.include_caller_info:
            record.caller_info = self._get_caller_info(record)

        return super().format(record)

    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        for pattern, replacement in self.REDACT_PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        return message

    def _get_caller_info(self, record: logging.LogRecord) -> str:
        """Get caller file and line number."""
        if record.filename:
            return f"{os.path.basename(record.filename)}:{record.lineno}"
        return ""


class AuditLogger:
    """
    Audit logger for tracking security-relevant events.
    Writes to a separate audit log file.
    """

    def __init__(self, app=None):
        self.app = app
        self.audit_logger: Optional[logging.Logger] = None

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Initialize audit logger with Flask app."""
        self.app = app

        # Create audit logger
        self.audit_logger = logging.getLogger('nocturna.audit')
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.propagate = False

        # Audit log file path
        audit_log_path = os.environ.get(
            'AUDIT_LOG_FILE',
            os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'audit.log')
        )

        # Create directory if needed
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)

        # File handler with rotation
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                audit_log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)

            # JSON format for audit logs
            json_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"event": "%(message)s", "source": "%(name)s"}'
            )
            file_handler.setFormatter(json_formatter)

            self.audit_logger.addHandler(file_handler)
        except Exception as e:
            if self.app:
                self.app.logger.error(f"Failed to create audit log: {e}")

    def log_event(self, event_type: str, user_id: str, details: dict,
                  success: bool = True, request_id: str = None):
        """
        Log an audit event.

        Args:
            event_type: Type of event (e.g., 'LOGIN', 'TRADE_SUBMIT', 'CONFIG_UPDATE')
            user_id: User identifier
            details: Event-specific details
            success: Whether the action succeeded
            request_id: Request tracking ID
        """
        if not self.audit_logger:
            return

        # Sanitize details
        sanitized_details = self._sanitize_details(details)

        event = {
            'type': event_type,
            'user_id': user_id,
            'success': success,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': sanitized_details
        }

        self.audit_logger.info(json.dumps(event))

    def _sanitize_details(self, details: dict) -> dict:
        """Remove sensitive fields from details."""
        sensitive_fields = [
            'password', 'secret', 'token', 'api_key', 'secret_key',
            'credential', 'authorization', 'ssn', 'card_number'
        ]

        sanitized = {}
        for key, value in details.items():
            key_lower = key.lower()
            if any(field in key_lower for field in sensitive_fields):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            elif isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:50] + '...'
            else:
                sanitized[key] = value

        return sanitized


# Global audit logger instance
audit_logger = AuditLogger()


def setup_secure_logging(app):
    """
    Configure secure logging for the Flask application.

    Args:
        app: Flask application instance
    """
    # Determine log level from environment
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    try:
        log_level = getattr(logging, log_level)
    except AttributeError:
        log_level = logging.INFO

    # Log format
    log_format = os.environ.get(
        'LOG_FORMAT',
        '[%(asctime)s] %(levelname)s [%(name)s:%(caller_info)s] %(message)s'
    )

    # Create secure formatter
    secure_formatter = SecureLogFormatter(include_caller_info=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if os.environ.get('LOG_TO_CONSOLE', 'true').lower() == 'true':
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(secure_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    log_file_path = os.environ.get(
        'LOG_FILE',
        os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nocturna.log')
    )

    # Create log directory
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(secure_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        app.logger.warning(f"Failed to create file handler: {e}")

    # Security logger
    security_logger = logging.getLogger('nocturna.security')
    security_logger.setLevel(logging.WARNING)

    # Audit logger initialization
    global audit_logger
    audit_logger = AuditLogger(app)

    # Log startup info
    app.logger.info("=" * 60)
    app.logger.info("NOCTURNA Trading System - Secure Logging Initialized")
    app.logger.info(f"Log Level: {logging.getLevelName(log_level)}")
    app.logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    app.logger.info("=" * 60)


def log_function_call(logger: logging.Logger = None):
    """
    Decorator to log function calls with arguments and results.
    Sanitizes sensitive arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            func_name = func.__name__
            module_name = func.__module__

            # Sanitize arguments
            safe_args = _sanitize_args(kwargs)

            logger.debug(
                f"Calling {module_name}.{func_name} with args={safe_args}"
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(f"{module_name}.{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(
                    f"{module_name}.{func_name} failed with error: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise

        return wrapper
    return decorator


def _sanitize_args(args: dict) -> dict:
    """Sanitize function arguments for logging."""
    sensitive_keys = [
        'password', 'secret', 'token', 'key', 'api_key', 'secret_key',
        'credential', 'auth'
    ]

    sanitized = {}
    for key, value in args.items():
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_keys):
            sanitized[key] = '[REDACTED]'
        elif isinstance(value, str) and len(value) > 100:
            sanitized[key] = value[:50] + '...[truncated]'
        else:
            sanitized[key] = value

    return sanitized


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return audit_logger
