"""
NOCTURNA Trading System - Security Middleware
Production-grade security headers, rate limiting, and protection mechanisms.
"""

import os
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Flask, current_app, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# =============================================================================
# RATE LIMITING CONFIGURATION
# =============================================================================

#: Applies to every route without its own limit, per client IP.
DEFAULT_RATE_LIMIT = "300 per minute"

#: Per-endpoint limits, per client IP. Each replaces the default for that route.
ROUTE_RATE_LIMITS: dict[str, str] = {
    "user.login": "10 per minute",
    "user.refresh_token": "30 per minute",
    "user.change_password": "5 per minute",
    "user.create_user": "5 per minute",
    "user.create_api_key": "10 per hour",
    "trading.create_order": "60 per minute",
    "trading.cancel_order": "60 per minute",
    "trading.update_config": "20 per minute",
    "trading.start_engine": "20 per minute",
    "trading.stop_engine": "20 per minute",
    "trading.pause_engine": "20 per minute",
    "trading.resume_engine": "20 per minute",
    "paper_trading.paper_submit_order": "120 per minute",
    "paper_trading.paper_reset": "10 per minute",
}

#: Never rate limited: load-balancer probes, and the emergency stop, which must
#: always get through.
RATE_LIMIT_EXEMPT = frozenset({"health_check", "trading.emergency_stop"})


def setup_rate_limiting(app: Flask, storage_uri: str | None = None) -> Limiter:
    """
    Attach rate limits to the application's routes.

    Call after every blueprint and route is registered: limits are attached
    to the registered view functions by endpoint name.

    Args:
        app: Flask application instance.
        storage_uri: Limit storage (e.g. the Redis URL); in-process memory if None.

    Returns:
        The configured Limiter, also stored as ``app.extensions["rate_limiter"]``.

    Raises:
        KeyError: If a configured endpoint does not exist (a renamed route).
    """
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        storage_uri=storage_uri or "memory://",
        default_limits=[DEFAULT_RATE_LIMIT],
        strategy="moving-window",
        headers_enabled=True,
        in_memory_fallback_enabled=True,
    )

    @limiter.request_filter
    def _whitelisted() -> bool:
        return ip_manager.is_whitelisted(get_remote_address())

    for endpoint, limit in ROUTE_RATE_LIMITS.items():
        app.view_functions[endpoint] = limiter.limit(limit)(app.view_functions[endpoint])
    for endpoint in RATE_LIMIT_EXEMPT:
        app.view_functions[endpoint] = limiter.exempt(app.view_functions[endpoint])

    # 429 responses are rendered by the global errorhandler in main.py
    app.extensions["rate_limiter"] = limiter
    app.logger.info("Rate limiting configured with %s storage", "redis" if storage_uri else "memory")
    return limiter


def configure_proxy(app: Flask, trusted_proxy_count: int) -> None:
    """
    Trust X-Forwarded-For/-Proto from this many reverse proxies.

    Without it, behind a proxy every client shares the proxy's address, so
    rate limits and login lockouts would apply to all clients at once. Never
    enable it without a proxy: clients could then forge their address.
    """
    if trusted_proxy_count > 0:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=trusted_proxy_count, x_proto=trusted_proxy_count)


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
    def sanitize_symbol(symbol: str) -> str | None:
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
                         max_val: float = None) -> float | None:
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
    def sanitize_config_dict(data: dict) -> dict:
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
                if 'allowed' in schema and value.upper() in [a.upper() for a in schema['allowed']]:
                    sanitized[key] = value.upper()
                continue

        return sanitized

    @staticmethod
    def sanitize_order_signal(data: dict) -> dict | None:
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
    Static allow/deny lists (IP_WHITELIST / IP_BLACKLIST) plus a temporary
    lockout of addresses with repeated failed logins.

    An address is locked out while it has ``failed_threshold`` failures within
    the last ``lockout_duration``; the lockout ends on its own as failures age.
    Whitelisted addresses bypass rate limits and lockout, not the deny list.
    """

    #: Prune every address once this many are tracked, bounding memory.
    MAX_TRACKED = 10_000

    def __init__(self, failed_threshold: int = 5, lockout_duration: timedelta = timedelta(minutes=15)):
        self.whitelist: set[str] = set()
        self.blacklist: set[str] = set()
        self.failed_attempts: dict[str, list[datetime]] = {}
        self.failed_threshold = failed_threshold
        self.lockout_duration = lockout_duration
        self._lock = threading.Lock()
        self._load_lists()

    def _load_lists(self) -> None:
        """Load IP lists from environment variables."""
        for env_name, target in (("IP_WHITELIST", self.whitelist), ("IP_BLACKLIST", self.blacklist)):
            target.update(ip.strip() for ip in os.environ.get(env_name, "").split(",") if ip.strip())

    def _recent_failures(self, ip: str, now: datetime) -> list[datetime]:
        """Failures of ``ip`` inside the window. Caller holds the lock."""
        recent = [t for t in self.failed_attempts.get(ip, []) if now - t < self.lockout_duration]
        if recent:
            self.failed_attempts[ip] = recent
        else:
            self.failed_attempts.pop(ip, None)
        return recent

    def is_blacklisted(self, ip: str) -> bool:
        """True if the address is on the deny list or currently locked out."""
        if ip in self.blacklist:
            return True
        if ip in self.whitelist:
            return False
        with self._lock:
            return len(self._recent_failures(ip, datetime.now(UTC))) >= self.failed_threshold

    def is_whitelisted(self, ip: str) -> bool:
        """Check if an IP is whitelisted (bypasses rate limits and lockout)."""
        return ip in self.whitelist

    def record_failed_attempt(self, ip: str) -> bool:
        """
        Record a failed login from ``ip``.

        Returns:
            True if the address is now locked out.
        """
        now = datetime.now(UTC)
        with self._lock:
            if len(self.failed_attempts) >= self.MAX_TRACKED:
                for tracked in list(self.failed_attempts):
                    self._recent_failures(tracked, now)
            failures = self._recent_failures(ip, now)
            failures.append(now)
            self.failed_attempts[ip] = failures
            count = len(failures)

        if count == self.failed_threshold:
            current_app.logger.warning(
                "IP %s locked out for %s after %d failed logins", ip, self.lockout_duration, count
            )
        return count >= self.failed_threshold

    def clear_failed_attempts(self, ip: str) -> None:
        """Clear failed attempts for an IP."""
        with self._lock:
            self.failed_attempts.pop(ip, None)


ip_manager = IPManager()
