"""
NOCTURNA Trading System - Main Flask Application
Production-grade configuration with security hardening.
"""

import logging
import os
import secrets
import sys
import time as time_module
from datetime import UTC, datetime, timedelta

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS

from src.middleware.security import setup_rate_limiting, setup_security_headers
from src.models.user import db
from src.routes.trading import trading_bp
from src.routes.user import user_bp
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

    # Engine options — only apply pool settings for non-SQLite databases
    if database_url.startswith('sqlite'):
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'echo': os.environ.get('SQLALCHEMY_ECHO', 'false').lower() == 'true'
        }
    else:
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
    # INITIALIZE AUTH TOKEN MANAGER
    # =============================================================================
    from src.middleware.auth import token_manager
    token_manager.init_app(app)

    # =============================================================================
    # REGISTER BLUEPRINTS
    # =============================================================================
    app.register_blueprint(user_bp, url_prefix='/api')
    app.register_blueprint(trading_bp, url_prefix='/api/trading')

    # Paper trading routes
    from src.routes.paper_trading import init_paper_engine, paper_bp
    app.register_blueprint(paper_bp, url_prefix='/api/paper')
    with app.app_context():
        paper_config = {
            'initial_capital': float(os.environ.get('PAPER_INITIAL_CAPITAL', 100000)),
            'commission_rate': float(os.environ.get('PAPER_COMMISSION_RATE', 0.001)),
            'slippage_rate': float(os.environ.get('PAPER_SLIPPAGE_RATE', 0.0005)),
        }
        init_paper_engine(paper_config)

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
    def health_check():
        """Public health check endpoint for load balancers."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(UTC).isoformat()
        })

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
            'timestamp': datetime.now(UTC).isoformat()
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
