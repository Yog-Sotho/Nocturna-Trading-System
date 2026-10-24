"""
NOCTURNA Trading System - Main Flask Application
Production-grade configuration with security hardening.
"""

import logging
import os
import re
import secrets
import sys
import time as time_module
from datetime import UTC, datetime, timedelta

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, abort, g, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import MethodNotAllowed
from werkzeug.security import safe_join

from src.config import ensure_sqlite_directory, load_settings
from src.db_migrations import upgrade as upgrade_database
from src.middleware.security import configure_proxy, setup_rate_limiting, setup_security_headers
from src.models.user import db
from src.routes import schemas
from src.routes.console import console_bp
from src.routes.openapi import documented
from src.routes.ops import ops_bp
from src.routes.trading import trading_bp
from src.routes.user import user_bp
from src.trading.safety import assert_paper_only
from src.utils.logger import setup_secure_logging

_REQUEST_ID_PATTERN = re.compile(r'[A-Za-z0-9._-]{1,64}')


def _revalidated(response):
    """The dashboard's files keep their names across upgrades: browsers must check for a new version."""
    response.headers['Cache-Control'] = 'no-cache'
    return response


def start_serving(app):
    """
    Start what runs beside the web server: the engine (when ENGINE_AUTOSTART is
    set), the engine monitor and Telegram commands. Call in the process that
    serves requests, after gunicorn's fork: threads started before it do not
    survive, and one Telegram bot must have exactly one poller.
    """
    from src.trading.alerts import start_monitoring
    from src.trading.telegram import start_telegram_commands

    app.extensions['trading'].autostart()
    start_monitoring(app)
    start_telegram_commands(app)


def create_app(config_override=None):
    """
    Application factory for creating Flask app instances.

    Args:
        config_override: Optional dict of config overrides for testing

    Returns:
        Configured Flask application instance
    """
    # Live trading is hard-disabled: refuse to boot with a live configuration
    assert_paper_only()

    # Validate the whole environment up front (fails fast in production)
    settings = load_settings()

    app = Flask(__name__)
    app.extensions['settings'] = settings
    configure_proxy(app, settings.trusted_proxy_count)

    # =============================================================================
    # SECRETS
    # =============================================================================
    if settings.flask_secret_key is None:
        # Development/testing only — load_settings() rejects this in production
        logging.warning(
            "FLASK_SECRET_KEY not set. Using an ephemeral generated key; "
            "tokens will not survive a restart."
        )
        secret_key = secrets.token_hex(32)
    else:
        secret_key = settings.flask_secret_key.get_secret_value()

    app.config['SECRET_KEY'] = secret_key
    app.config['JWT_SECRET_KEY'] = (
        settings.jwt_secret_key.get_secret_value() if settings.jwt_secret_key else secret_key
    )
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=settings.jwt_expiration_hours)

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    database_url = settings.resolved_database_url
    ensure_sqlite_directory(database_url)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Engine options — only apply pool settings for non-SQLite databases
    if database_url.startswith('sqlite'):
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'echo': settings.sqlalchemy_echo}
    else:
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 10,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'max_overflow': 20,
            'echo': settings.sqlalchemy_echo
        }

    # =============================================================================
    # SESSION CONFIGURATION
    # =============================================================================
    app.config['SESSION_COOKIE_SECURE'] = settings.secure_cookies
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = settings.session_cookie_samesite
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=settings.session_lifetime_hours)

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024  # 1 MB; no endpoint takes more

    # =============================================================================
    # PRODUCTION MODE SETTINGS
    # =============================================================================
    app.config['DEBUG'] = settings.flask_debug
    app.config['TESTING'] = False

    # =============================================================================
    # APPLY CONFIG OVERRIDES (for testing)
    # =============================================================================
    if config_override:
        app.config.update(config_override)

    # =============================================================================
    # LOGGING SETUP
    # =============================================================================
    setup_secure_logging(app)

    # =============================================================================
    # CORS CONFIGURATION - STRICT ORIGIN CHECKING
    # =============================================================================
    allowed_list = settings.allowed_origin_list
    if allowed_list:
        # Only allow the configured origins
        cors_config = {
            'resources': {
                r'/*': {
                    'origins': allowed_list,
                    'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                    'allow_headers': ['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
                    'expose_headers': ['X-Request-ID', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
                    'supports_credentials': True,
                    'max_age': 600  # 10 minutes preflight cache
                }
            },
            'automatic_options': True
        }
        CORS(app, **cors_config)
        app.logger.info(f"CORS enabled for allowed origins: {allowed_list}")
    else:
        # Development only — load_settings() requires ALLOWED_ORIGINS in production
        CORS(app, resources={
            r'/*': {
                'origins': ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5000'],
                'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                'allow_headers': ['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
                'supports_credentials': True,
                'max_age': 600
            }
        })
        app.logger.warning("CORS enabled for local development only. Configure ALLOWED_ORIGINS for production.")

    # =============================================================================
    # SECURITY HEADERS
    # =============================================================================
    setup_security_headers(app)

    # =============================================================================
    # INITIALIZE DATABASE
    # =============================================================================
    db.init_app(app)
    if not app.config.get('TESTING'):
        # The schema is owned by the migrations; tests build theirs per test
        upgrade_database(app)
        with app.app_context():
            # gunicorn preloads the app and forks: no pooled connection may be inherited
            db.engine.dispose()

    # =============================================================================
    # INITIALIZE AUTH TOKEN MANAGER
    # =============================================================================
    from src.middleware.auth import token_manager
    token_manager.init_app(app)

    # =============================================================================
    # OPERATOR CLI (admin bootstrap — public registration is disabled)
    # =============================================================================
    from src.cli import register_cli
    register_cli(app)

    # =============================================================================
    # REGISTER BLUEPRINTS
    # =============================================================================
    # Version 1 of the API; /api/version (unversioned) says which versions exist
    app.register_blueprint(user_bp, url_prefix='/api/v1')
    app.register_blueprint(trading_bp, url_prefix='/api/v1/trading')
    app.register_blueprint(console_bp, url_prefix='/api/v1')

    # Live (paper) trading engine: in this process (built and started on demand, never at
    # import), or a separate engine process reached through the database
    from src.trading.alerts import build_alerts
    from src.trading.app_settings import alert_choices
    from src.trading.lab import ProcessLauncher
    from src.trading.runtime import RemoteRuntime, TradingRuntime
    app.extensions['alerts'] = build_alerts(settings, choices=alert_choices(app))
    app.extensions['trading'] = (RemoteRuntime(app, settings) if settings.engine_mode == "external"
                                 else TradingRuntime(app, settings))
    # The strategy lab's runs execute in a process of their own, started on demand
    app.extensions['lab'] = ProcessLauncher()
    app.register_blueprint(ops_bp)

    # =============================================================================
    # REQUEST ID TRACKING
    # =============================================================================
    @app.before_request
    def add_request_id():
        """Add unique request ID for tracing."""
        supplied = request.headers.get('X-Request-ID', '')
        # Accept a caller's ID only if it is safe to log and store
        g.request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else secrets.token_hex(16)
        g.start_time = time_module.time()

    @app.before_request
    def reject_oversized_body():
        """Refuse bodies over MAX_CONTENT_LENGTH before any route reads them."""
        limit = app.config.get('MAX_CONTENT_LENGTH')
        if limit and request.content_length and request.content_length > limit:
            abort(413)

    @app.after_request
    def add_response_headers(response):
        """Add security and tracing headers to all responses."""
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        return response

    # =============================================================================
    # GLOBAL ERROR HANDLERS
    # =============================================================================
    @app.errorhandler(400)
    def bad_request(error):
        app.logger.warning(f"Bad request: {error} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        app.logger.warning(f"Unauthorized access attempt: {request.path} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        app.logger.warning(f"Forbidden access: {request.path} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Forbidden',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'success': False,
            'error': 'Method not allowed',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({
            'success': False,
            'error': 'Request body too large',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        app.logger.warning(f"Rate limit exceeded: {request.path} from {request.remote_addr} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please try again later.',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(UTC).isoformat()
        }), 500

    # =============================================================================
    # HEALTH CHECK ENDPOINT
    # =============================================================================
    @app.route('/health', methods=['GET'])
    @documented("Liveness", tag="system", access="public", envelope=False, rate_limited=False,
                description="Answers while the API process runs; /ready also needs the database.",
                responses={200: ("Alive", schemas.Health)})
    def health_check():
        """Public health check endpoint for load balancers."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(UTC).isoformat()
        })

    # =============================================================================
    # STATIC FILE SERVING
    # =============================================================================
    static_folder = str(settings.static_folder or os.path.join(os.path.dirname(__file__), 'static'))
    app.config['STATIC_FOLDER'] = static_folder

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        """Serve static files or return index.html for SPA."""
        if path == 'api' or path.startswith('api/'):
            # API paths never fall through to the SPA: a known path called with
            # the wrong method is a 405, anything else a 404 (both JSON)
            adapter = app.url_map.bind_to_environ(request.environ)
            api_methods = set(adapter.allowed_methods(request.path)) - {'GET', 'HEAD', 'OPTIONS'}
            if api_methods:
                raise MethodNotAllowed(valid_methods=sorted(api_methods))
            abort(404)
        # safe_join refuses paths that leave the static folder, so the existence
        # check cannot be used to probe for files elsewhere on the server
        static_file = safe_join(static_folder, path) if path else None
        if static_file is not None and os.path.isfile(static_file):
            return _revalidated(send_from_directory(static_folder, path))
        else:
            index_path = os.path.join(static_folder, 'index.html')
            if os.path.exists(index_path):
                return _revalidated(send_from_directory(static_folder, 'index.html'))
            else:
                return jsonify({
                    'success': False,
                    'error': 'Frontend not deployed'
                }), 503

    # =============================================================================
    # API VERSION ENDPOINT
    # =============================================================================
    @app.route('/api/version', methods=['GET'])
    @documented("API versions", tag="system", access="public", envelope=False,
                description="Unversioned on purpose: it says which versions exist.",
                responses={200: ("Versions", schemas.Version)})
    def api_version():
        return jsonify({
            'version': '2.0.0',
            'api_version': 'v1',
            'versions': ['v1'],
            'openapi': '/api/v1/openapi.json',
            'environment': settings.flask_env,
            'timestamp': datetime.now(UTC).isoformat()
        })

    # =============================================================================
    # RATE LIMITING (after every route is registered)
    # =============================================================================
    setup_rate_limiting(app, storage_uri=settings.redis_url)

    app.logger.info("NOCTURNA Trading System initialized successfully")
    app.logger.info(f"Environment: {settings.flask_env}")

    return app


if __name__ == '__main__':
    app = create_app()
    # Only the reloader's serving child may start the engine (see main.py)
    if not app.config['DEBUG'] or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        start_serving(app)

    # =============================================================================
    # PRODUCTION WARNING
    # =============================================================================
    if os.environ.get('FLASK_ENV') == 'production':
        logging.critical(
            "=" * 80 + "\n"
            "WARNING: Running Flask development server in PRODUCTION mode!\n"
            "Use a production WSGI server (gunicorn, uwsgi) instead:\n"
            "gunicorn -c gunicorn.conf.py 'src.main:create_app()'\n"
            "=" * 80
        )

    # Development server with improved settings
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG'],
        threaded=True
    )
