"""
NOCTURNA Trading System - Security Middleware
Production-grade security headers, rate limiting, and protection mechanisms.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from functools import wraps

from flask import Flask, request, g, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address



# =============================================================================
# RATE LIMITING CONFIGURATION
# =============================================================================

def setup_rate_limiting(app: Flask) -> Limiter:
    """
    Configure rate limiting for the application.

    Args:
        app: Flask application instance

    Returns:
        Configured Limiter instance
    """
    # Determine storage backend
    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        # Use Redis for distributed rate limiting
        storage_uri = redis_url
    else:
        # Fallback to in-memory storage (not recommended for production)
        storage_uri = "memory://"

    # Create limiter with appropriate storage
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri=storage_uri,
        default_limits=[],
        strategy="fixed-window",
        injection_scheme="scheme://netloc",
        headers_enabled=True,
    )

    # =============================================================================
    # DEFAULT RATE LIMITS BY ENDPOINT TYPE
    # =============================================================================

    # Public endpoints - very restrictive
    limiter.limit("100 per minute")(
        lambda: request.endpoint and
        any(request.endpoint.startswith(p) for p in ['health', 'api_version', 'serve'])
    )

    # Authentication endpoints - rate limit to prevent brute force
    limiter.limit("10 per minute")(
        lambda: request.endpoint and
        any(request.endpoint.startswith(p) for p in ['login', 'auth', 'token'])
    )

    # Trading read endpoints - moderate limits
    limiter.limit("120 per minute")(
        lambda: request.endpoint and
        any(request.endpoint.startswith(p) for p in ['status', 'positions', 'orders', 'performance'])
    )

    # Trading write endpoints - stricter limits to prevent abuse
    limiter.limit("30 per minute")(
        lambda: request.endpoint and
        any(request.endpoint in [p] for p in ['submit_order', 'create_order', 'update_config'])
    )

    # System control endpoints - most restrictive
    limiter.limit("20 per minute")(
        lambda: request.endpoint and
        any(request.endpoint in [p] for p in ['start_engine', 'stop_engine', 'emergency_stop'])
    )

    # =============================================================================
    # CONFIGURATION
    # =============================================================================

    # Custom error message
    @limiter.ratelimit_error
    def ratelimit_handler(e):
        """Handle rate limit exceeded errors."""
        current_app.logger.warning(
            f"Rate limit exceeded: {request.path} from {request.remote_addr} | "
            f"Request ID: {getattr(g, 'request_id', 'unknown')}"
        )
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please slow down your requests.',
            'error_code': 'RATE_LIMIT_EXCEEDED',
            'retry_after': int(e.description.split()[-2]) if e.description else 60,
            'request_id': getattr(g, 'request_id', None),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 429

    app.logger.info(f"Rate limiting configured with storage: {storage_uri}")
    return limiter


def get_user_identifier() -> str:
    """
    Get a unique identifier for the current request.
    Combines IP address with authenticated user ID if available.

    Returns:
        Unique identifier string
    """
    ip = get_remote_address()
    user_id = getattr(g, 'user_id', None)

    if user_id:
        return f"{user_id}@{ip}"
    return ip


# =============================================================================
# SECURITY HEADERS
# =============================================================================

def setup_security_headers(app: Flask) -> None:
    """
    Configure security headers for all responses.

    Args:
        app: Flask application instance
    """

    @app.after_request
    def set_security_headers(response):
        """Set security headers on all responses."""

        # =============================================================================
        # CONTENT SECURITY POLICY — no unsafe-inline/unsafe-eval in script-src
        # =============================================================================
        csp_parts = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
            "object-src 'none'",
        ]

        # Add API domain if configured — use proper connect-src directive
        api_domain = os.environ.get('API_DOMAIN')
        if api_domain:
            # Replace the connect-src to include the API domain
            csp_parts = [p for p in csp_parts if not p.startswith("connect-src")]
            ws_domain = api_domain.replace('https', 'wss').replace('http', 'ws')
            csp_parts.append(f"connect-src 'self' {api_domain} {ws_domain}")

        response.headers['Content-Security-Policy'] = '; '.join(csp_parts)

        # =============================================================================
        # HSTS — enforce HTTPS
        # =============================================================================
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains; preload'

        # =============================================================================
        # OTHER SECURITY HEADERS
        # =============================================================================
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['X-Permitted-Cross-Domain-Policies'] = 'none'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()'

        # Cache control - sensitive data should not be cached
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

        # Remove server identification
        response.headers.pop('Server', None)
        response.headers.pop('X-Powered-By', None)

        return response


# =============================================================================
# INPUT VALIDATION AND SANITIZATION
# =============================================================================

class InputSanitizer:
    """
    Sanitizes and validates user input to prevent injection attacks.
    """

    # Allowed characters for various input types
    SAFE_SYMBOL_PATTERN = r'^[A-Z]{1,5}$'  # Stock symbols: 1-5 uppercase letters
    SAFE_NUMERIC_PATTERN = r'^-?\d+\.?\d*$'
    SAFE_EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    @staticmethod
    def sanitize_string(value: str, max_length: int = 255,
                        strip_html: bool = True) -> str:
        """
        Sanitize a string input.

        Args:
            value: Input string to sanitize
            max_length: Maximum allowed length
            strip_html: Whether to strip HTML tags

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)[:max_length]

        # Truncate to max length
        value = value[:max_length]

        # Remove null bytes
        value = value.replace('\x00', '')

        # Strip HTML if requested
        if strip_html:
            import re
            value = re.sub(r'<[^>]*>', '', value)
            value = value.strip()

        return value

    @staticmethod
    def sanitize_symbol(symbol: str) -> Optional[str]:
        """
        Sanitize and validate a stock symbol.

        Args:
            symbol: Symbol to validate

        Returns:
            Sanitized symbol if valid, None if invalid
        """
        import re

        if not symbol or not isinstance(symbol, str):
            return None

        # Uppercase and strip whitespace
        symbol = symbol.upper().strip()

        # Match against pattern
        if not re.match(InputSanitizer.SAFE_SYMBOL_PATTERN, symbol):
            return None

        # Check for dangerous characters
        dangerous = ['<', '>', '"', "'", '\\', '/', '|', '&', ';', '$', '`']
        if any(c in symbol for c in dangerous):
            return None

        return symbol

    @staticmethod
    def sanitize_numeric(value: Any, min_val: float = None,
                         max_val: float = None) -> Optional[float]:
        """
        Sanitize and validate a numeric input.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized numeric value if valid, None if invalid
        """
        try:
            num = float(value)
        except (ValueError, TypeError):
            return None

        if min_val is not None and num < min_val:
            return None
        if max_val is not None and num > max_val:
            return None

        return num

    @staticmethod
    def sanitize_config_dict(data: Dict) -> Dict:
        """
        Sanitize configuration dictionary input.

        Args:
            data: Configuration dictionary to sanitize

        Returns:
            Sanitized configuration dictionary
        """
        if not isinstance(data, dict):
            return {}

        sanitized = {}

        # Define allowed configuration keys and their validation rules
        config_schema = {
            'symbols': {
                'type': list,
                'validator': lambda x: all(InputSanitizer.sanitize_symbol(s) for s in x) if x else True,
                'max_length': 20
            },
            'update_interval': {
                'type': (int, float),
                'min': 1,
                'max': 3600
            },
            'max_position_size': {
                'type': (int, float),
                'min': 0.001,
                'max': 1.0
            },
            'risk_level': {
                'type': str,
                'allowed': ['LOW', 'MEDIUM', 'HIGH', 'AGGRESSIVE']
            },
            'trading_mode': {
                'type': str,
                'allowed': ['PAPER', 'LIVE']
            },
            'grid_spacing': {
                'type': (int, float),
                'min': 0.0001,
                'max': 0.1
            },
            'atr_mult_sl': {
                'type': (int, float),
                'min': 0.5,
                'max': 10.0
            },
            'atr_mult_tp': {
                'type': (int, float),
                'min': 0.5,
                'max': 20.0
            },
            'volatility_threshold': {
                'type': (int, float),
                'min': 0.1,
                'max': 10.0
            },
            'max_daily_loss': {
                'type': (int, float),
                'min': 0.001,
                'max': 0.5
            },
            'max_drawdown': {
                'type': (int, float),
                'min': 0.001,
                'max': 0.5
            }
        }

        for key, value in data.items():
            if key not in config_schema:
                continue

            schema = config_schema[key]

            # Type check
            expected_type = schema['type']
            if not isinstance(value, expected_type):
                continue

            # List validation
            if isinstance(value, list):
                if 'validator' in schema:
                    if not schema['validator'](value):
                        continue
                    sanitized[key] = value[:schema.get('max_length', 20)]
                else:
                    sanitized[key] = value[:schema.get('max_length', 20)]
                continue

            # Numeric range validation
            if isinstance(value, (int, float)):
                num = InputSanitizer.sanitize_numeric(value)
                if num is None:
                    continue
                if 'min' in schema and num < schema['min']:
                    continue
                if 'max' in schema and num > schema['max']:
                    continue
                sanitized[key] = num
                continue

            # String allowed values
            if isinstance(value, str):
                if 'allowed' in schema:
                    if value.upper() in [a.upper() for a in schema['allowed']]:
                        sanitized[key] = value.upper()
                continue

        return sanitized

    @staticmethod
    def sanitize_order_signal(data: Dict) -> Optional[Dict]:
        """
        Sanitize and validate a trading signal/order input.

        Args:
            data: Order/signal dictionary to sanitize

        Returns:
            Sanitized order dictionary if valid, None if invalid
        """
        if not isinstance(data, dict):
            return None

        sanitized = {}

        # Required fields
        symbol = InputSanitizer.sanitize_symbol(data.get('symbol', ''))
        if not symbol:
            return None
        sanitized['symbol'] = symbol

        side = str(data.get('side', '')).lower()
        if side not in ['buy', 'sell']:
            return None
        sanitized['side'] = side

        order_type = str(data.get('type', 'market')).lower()
        if order_type not in ['market', 'limit', 'stop', 'stop_limit']:
            return None
        sanitized['type'] = order_type

        # Quantity validation
        quantity = InputSanitizer.sanitize_numeric(data.get('quantity'), min_val=0.001)
        if quantity is None:
            return None
        sanitized['quantity'] = quantity

        # Optional fields with validation
        if 'price' in data:
            price = InputSanitizer.sanitize_numeric(data.get('price'), min_val=0.01)
            if price:
                sanitized['price'] = price

        if 'stop_price' in data:
            stop_price = InputSanitizer.sanitize_numeric(data.get('stop_price'), min_val=0.01)
            if stop_price:
                sanitized['stop_price'] = stop_price

        if 'stop_loss' in data:
            stop_loss = InputSanitizer.sanitize_numeric(data.get('stop_loss'), min_val=0.01)
            if stop_loss:
                sanitized['stop_loss'] = stop_loss

        if 'take_profit' in data:
            take_profit = InputSanitizer.sanitize_numeric(data.get('take_profit'), min_val=0.01)
            if take_profit:
                sanitized['take_profit'] = take_profit

        return sanitized


# =============================================================================
# IP WHITELISTING / BLACKLISTING
# =============================================================================

class IPManager:
    """
    Manages IP whitelisting and blacklisting for additional security.
    """

    def __init__(self):
        self.whitelist = set()
        self.blacklist = set()
        self.failed_attempts = {}  # Track failed auth attempts
        self.failed_threshold = 5  # Lockout after 5 failed attempts
        self.lockout_duration = timedelta(minutes=15)

        self._load_lists()

    def _load_lists(self):
        """Load IP lists from environment variables."""
        # Load whitelist
        whitelist_str = os.environ.get('IP_WHITELIST', '')
        if whitelist_str:
            self.whitelist.update(
                ip.strip() for ip in whitelist_str.split(',') if ip.strip()
            )

        # Load blacklist
        blacklist_str = os.environ.get('IP_BLACKLIST', '')
        if blacklist_str:
            self.blacklist.update(
                ip.strip() for ip in blacklist_str.split(',') if ip.strip()
            )

    def is_blacklisted(self, ip: str) -> bool:
        """Check if an IP is blacklisted."""
        return ip in self.blacklist

    def is_whitelisted(self, ip: str) -> bool:
        """Check if an IP is whitelisted (bypasses rate limits)."""
        return ip in self.whitelist

    def record_failed_attempt(self, ip: str) -> bool:
        """
        Record a failed authentication attempt.
        Returns True if IP should be blacklisted due to too many failures.
        """
        now = datetime.now(timezone.utc)

        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []

        # Clean old attempts
        self.failed_attempts[ip] = [
            t for t in self.failed_attempts[ip]
            if now - t < self.lockout_duration
        ]

        # Add new attempt
        self.failed_attempts[ip].append(now)

        # Check if threshold exceeded
        if len(self.failed_attempts[ip]) >= self.failed_threshold:
            self.blacklist.add(ip)
            current_app.logger.critical(
                f"IP {ip} blacklisted due to multiple failed authentication attempts"
            )
            return True

        return False

    def clear_failed_attempts(self, ip: str) -> None:
        """Clear failed attempts for an IP."""
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]


ip_manager = IPManager()


def check_ip_blacklist():
    """
    Before request hook to check IP blacklist.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = get_remote_address()

            # Check blacklist
            if ip_manager.is_blacklisted(ip):
                current_app.logger.warning(
                    f"Blacklisted IP attempted access: {ip} | "
                    f"Path: {request.path} | Request ID: {getattr(g, 'request_id', 'unknown')}"
                )
                return jsonify({
                    'success': False,
                    'error': 'Access denied',
                    'request_id': getattr(g, 'request_id', None)
                }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator
