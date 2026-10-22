"""
NOCTURNA Trading System - Operations endpoints.

* ``GET /ready``: readiness, unlike ``/health`` (liveness): 503 while the
  database cannot be reached. The engine's state does not count: with the
  engine down the API must stay in service, since its emergency stop then goes
  straight to the broker.
* ``GET /metrics``: Prometheus metrics, only with ``Authorization: Bearer
  <METRICS_TOKEN>``; without METRICS_TOKEN the endpoint does not exist.
"""

import hmac
import logging

import sqlalchemy as sa
from flask import Blueprint, Response, abort, current_app, jsonify, request

from src.models.user import db
from src.trading import metrics

ops_bp = Blueprint('ops', __name__)
logger = logging.getLogger(__name__)


@ops_bp.route('/ready', methods=['GET'])
def ready():
    try:
        db.session.execute(sa.text('SELECT 1'))
    except Exception as e:
        # Probes repeat every few seconds: one line each, not a traceback
        logger.warning('Readiness check: the database is unreachable (%s)', type(e).__name__)
        db.session.rollback()
        return jsonify({'status': 'not ready', 'database': 'unreachable'}), 503
    return jsonify({'status': 'ready', 'database': 'ok'})


@ops_bp.route('/metrics', methods=['GET'])
def prometheus_metrics():
    token = current_app.extensions['settings'].metrics_token
    if token is None:
        abort(404)
    supplied = request.headers.get('Authorization', '')
    expected = f'Bearer {token.get_secret_value()}'
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        abort(401)
    return Response(metrics.render(current_app.extensions['trading'].views()), content_type=metrics.CONTENT_TYPE)
