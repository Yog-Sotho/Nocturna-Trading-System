"""
NOCTURNA Trading System - Main Flask Application
Production-grade configuration with security hardening.
"""

import os
import sys
import logging
import secrets
import time as time_module
from datetime import datetime, timedelta, timezone

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, g, jsonify, send_from_directory
from flask_cors import CORS

from src.models.user import db
from src.routes.user import user_bp
from src.routes.trading import trading_bp
from src.middleware.security import setup_security_headers, setup_rate_limiting
from src.utils.logger import setup_secure_logging


def create_app(config_override=None):
    """
    Application factory for creating Flask app instances.

    Args:
        config_override: Optional dict of config overrides for testing

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    # =============================================================================
    # SECURITY: Load configuration from environment variables (not hardcoded)
    # =============================================================================

    # Flask secret key - MUST be set via environment variable
    secret_key = os.environ.get('FLASK_SECRET_KEY', None)
    if secret_key is None:
        if os.environ.get('FLASK_ENV') == 'production':
            raise ValueError(
                "FLASK_SECRET_KEY environment variable MUST be set in production. "
                "Generate one using: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        else:
            # Development mode: generate a warning but allow to continue
            logging.warning(
                "FLASK_SECRET_KEY not set. Using insecure generated key. "
                "Set FLASK_SECRET_KEY environment variable for production."
            )
            secret_key = secrets.token_hex(32)

    app.config['SECRET_KEY'] = secret_key
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secret_key)
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(
        hours=int(os.environ.get('JWT_EXPIRATION_HOURS', 24))
    )

    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    database_url = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'nocturna.db')}"
    )
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20,
        'echo': os.environ.get('SQLALCHEMY_ECHO', 'false').lower() == 'true'
    }

    # =============================================================================
    # SESSION CONFIGURATION
    # =============================================================================
    app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SECURE_COOKIES', 'true').lower() == 'true'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(
        hours=int(os.environ.get('SESSION_LIFETIME_HOURS', 24))
    )

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

    # =============================================================================
    # PRODUCTION MODE SETTINGS
    # =============================================================================
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.config['TESTING'] = False

    # =============================================================================
    # LOGGING SETUP
    # =============================================================================
    setup_secure_logging(app)

    # =============================================================================
    # CORS CONFIGURATION - STRICT ORIGIN CHECKING
    # =============================================================================
    allowed_origins = os.environ.get('ALLOWED_ORIGINS', '')
    if allowed_origins:
        # Production: only allow specific origins
        allowed_list = [origin.strip() for origin in allowed_origins.split(',') if origin.strip()]
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
        # No origins configured - restrict to same origin in production
        if os.environ.get('FLASK_ENV') == 'production':
            raise ValueError(
                "ALLOWED_ORIGINS environment variable MUST be set in production. "
                "Example: ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com"
            )
        else:
            # Development mode: allow local development
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
    # RATE LIMITING SETUP
    # =============================================================================
    setup_rate_limiting(app)

    # =============================================================================
    # SECURITY HEADERS
    # =============================================================================
    setup_security_headers(app)

    # =============================================================================
    # INITIALIZE DATABASE
    # =============================================================================
    db.init_app(app)
    with app.app_context():
        db.create_all()

    # =============================================================================
    # REGISTER BLUEPRINTS
    # =============================================================================
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(trading_bp, url_prefix='/api/trading')

    # =============================================================================
    # REQUEST ID TRACKING
    # =============================================================================
    @app.before_request
    def add_request_id():
        """Add unique request ID for tracing."""
        g.request_id = request.headers.get('X-Request-ID', secrets.token_hex(16))
        g.start_time = time_module.time()

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
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 400

    @app.errorhandler(401)
    def unauthorized(error):
        app.logger.warning(f"Unauthorized access attempt: {request.path} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 401

    @app.errorhandler(403)
    def forbidden(error):
        app.logger.warning(f"Forbidden access: {request.path} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Forbidden',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 403

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 404

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        app.logger.warning(f"Rate limit exceeded: {request.path} from {request.remote_addr} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please try again later.',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error} | Request ID: {g.get('request_id')}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'request_id': g.get('request_id'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

    # =============================================================================
    # HEALTH CHECK ENDPOINT
    # =============================================================================
    @app.route('/health', methods=['GET'])
    def health_check():
        """Public health check endpoint for load balancers."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    @app.route('/health/deep', methods=['GET'])
    def deep_health_check():
        """
        Deep health check — verifies DB, Redis, and trading engine.
        Returns 503 if any critical component is down.
        """
        checks = {}
        overall_healthy = True

        # Database check
        try:
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            checks['database'] = 'healthy'
        except Exception as e:
            checks['database'] = f'unhealthy: {type(e).__name__}'
            overall_healthy = False

        # Redis check
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            try:
                import redis
                r = redis.from_url(redis_url, decode_responses=True)
                r.ping()
                checks['redis'] = 'healthy'
            except Exception as e:
                checks['redis'] = f'unhealthy: {type(e).__name__}'
                overall_healthy = False
        else:
            checks['redis'] = 'not_configured'

        # Trading engine check
        try:
            from src.routes.trading import _trading_engine
            if _trading_engine is not None:
                checks['trading_engine'] = {
                    'state': _trading_engine.state.value,
                    'last_update': (
                        _trading_engine.last_update.isoformat()
                        if _trading_engine.last_update else None
                    )
                }
            else:
                checks['trading_engine'] = 'not_initialized'
        except Exception:
            checks['trading_engine'] = 'unavailable'

        status_code = 200 if overall_healthy else 503
        return jsonify({
            'status': 'healthy' if overall_healthy else 'degraded',
            'checks': checks,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), status_code

    # =============================================================================
    # STATIC FILE SERVING
    # =============================================================================
    static_folder = os.environ.get('STATIC_FOLDER', os.path.join(os.path.dirname(__file__), 'static'))
    app.config['STATIC_FOLDER'] = static_folder

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        """Serve static files or return index.html for SPA."""
        if path != "" and os.path.exists(os.path.join(static_folder, path)):
            return send_from_directory(static_folder, path)
        else:
            index_path = os.path.join(static_folder, 'index.html')
            if os.path.exists(index_path):
                return send_from_directory(static_folder, 'index.html')
            else:
                return jsonify({
                    'success': False,
                    'error': 'Frontend not deployed'
                }), 503

    # =============================================================================
    # API VERSION ENDPOINT
    # =============================================================================
    @app.route('/api/version', methods=['GET'])
    def api_version():
        """Return API version and status."""
        return jsonify({
            'version': '2.0.0',
            'api_version': 'v1',
            'environment': os.environ.get('FLASK_ENV', 'development'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    app.logger.info("NOCTURNA Trading System initialized successfully")
    app.logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")

    return app


# Create the default application instance
app = create_app()


if __name__ == '__main__':
    # =============================================================================
    # PRODUCTION WARNING
    # =============================================================================
    if os.environ.get('FLASK_ENV') == 'production':
        logging.critical(
            "=" * 80 + "\n"
            "WARNING: Running Flask development server in PRODUCTION mode!\n"
            "Use a production WSGI server (gunicorn, uwsgi) instead:\n"
            "gunicorn -w 4 -b 0.0.0.0:5000 'src.main:create_app()' --timeout 120\n"
            "=" * 80
        )

    # Development server with improved settings
    app.run(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', 5000)),
        debug=app.config['DEBUG'],
        threaded=True
    )
