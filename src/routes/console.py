"""
NOCTURNA Trading System - The dashboard's control panel: the strategy lab,
the alert choices and each user's preferences.

The lab queues backtests and validations that a process of its own executes
(``src/trading/lab.py``); its results are for reading and change nothing
about trading. Alert secrets stay in ``.env``: here the operator only chooses
which configured channels to use and which kinds of alert to send.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from flask import Blueprint, current_app, g, request

from src.middleware.auth import require_admin, require_auth
from src.models.user import User, db
from src.routes import schemas
from src.routes.openapi import documented
from src.routes.trading import format_response
from src.trading import app_settings, lab
from src.trading.alerts import CATEGORIES, CHANNELS, Alert, AlertChoices, AlertDispatcher
from src.utils import preferences
from src.utils.audit import record_audit_event

console_bp = Blueprint('console', __name__)
logger = logging.getLogger(__name__)

MAX_RUNS = 100


def _app():
    return current_app._get_current_object()


# =============================================================================
# THE STRATEGY LAB
# =============================================================================

@console_bp.route('/lab/runs', methods=['GET'])
@documented("The strategy lab's runs, newest first", tag="lab", access="user",
            query={"limit": ("integer", "How many (default 20, at most 100)")},
            responses={200: ("The runs, without their results", schemas.LabRuns)})
@require_auth
def list_lab_runs():
    app = _app()
    lab.kick(app, datetime.now(UTC))
    limit = max(1, min(request.args.get('limit', 20, type=int) or 20, MAX_RUNS))
    return format_response(True, data={'runs': lab.runs(app, limit)})


@console_bp.route('/lab/runs', methods=['POST'])
@documented("Queue a backtest or a validation", tag="lab", access="admin", body=schemas.LabRunBody,
            description="backtest: a mix with its risk limits, as the engine would trade it. validate: the "
                        "promotion gates for one strategy and its parameter grid, as flask validate runs them. "
                        "Runs execute one at a time in a process of their own; nothing here trades.",
            responses={202: ("Queued", schemas.LabRunQueued), 400: "It cannot be run; the error says why",
                       409: "Too many runs are already waiting"})
@require_auth
@require_admin
def create_lab_run():
    app = _app()
    now = datetime.now(UTC)
    try:
        kind, spec = lab.parse_request(current_app.extensions['settings'], request.get_json(silent=True),
                                       now.date())
        run = lab.submit(app, kind, spec, created_by=getattr(g, 'username', None), now=now)
    except lab.LabError as e:
        return format_response(False, error=str(e), status_code=400)
    except lab.LabBusyError as e:
        return format_response(False, error=str(e), status_code=409)
    record_audit_event('LAB_RUN_QUEUED', True, {'run_id': run['id'], 'kind': kind, 'spec': spec})
    try:
        lab.kick(app, now)
    except Exception:
        logger.exception("Starting the lab process failed")
    return format_response(True, data={'run': run}, message='Queued', status_code=202)


@console_bp.route('/lab/runs/<int:run_id>', methods=['GET'])
@documented("A lab run with its result", tag="lab", access="user",
            responses={200: ("The run", schemas.LabRunData), 404: "No such run"})
@require_auth
def get_lab_run(run_id: int):
    run = lab.get(_app(), run_id)
    if run is None:
        return format_response(False, error='No such run', status_code=404)
    return format_response(True, data={'run': run})


# =============================================================================
# ALERTS
# =============================================================================

def _alerts() -> AlertDispatcher:
    return current_app.extensions['alerts']


def _alert_settings(choices: AlertChoices) -> dict[str, Any]:
    configured = set(_alerts().channels)
    return {
        'channels': [{'name': name, 'label': label, 'configured': name in configured,
                      'enabled': name in configured and choices.uses(name)} for name, label in CHANNELS.items()],
        'categories': [{'name': name, 'label': c.label, 'description': c.description, 'locked': c.locked,
                        'enabled': choices.sends(name)} for name, c in CATEGORIES.items()],
    }


def _stored_choices() -> AlertChoices:
    return AlertChoices.from_dict(app_settings.read(_app(), app_settings.ALERTS))


def _parse_choices(data: Any) -> AlertChoices:
    """
    Raises:
        ValueError: For an unknown name, a locked category turned off, or no configured channel left on.
    """
    if not isinstance(data, dict) or set(data) - {'channels', 'categories'}:
        raise ValueError("the body must be an object with channels and categories")
    current = _stored_choices()
    channels, categories = dict(current.channels), dict(current.categories)
    for given, known, out in ((data.get('channels', {}), CHANNELS, channels),
                              (data.get('categories', {}), CATEGORIES, categories)):
        if not isinstance(given, dict) or not all(isinstance(v, bool) for v in given.values()):
            raise ValueError("channels and categories map names to true or false")
        unknown = sorted(set(given) - set(known))
        if unknown:
            raise ValueError(f"unknown: {', '.join(unknown)}")
        out.update(given)
    locked = [name for name, c in CATEGORIES.items() if c.locked and categories.get(name) is False]
    if locked:
        raise ValueError(f"{', '.join(CATEGORIES[n].label for n in locked)} alerts cannot be turned off")
    choices = AlertChoices(channels, {n: on for n, on in categories.items() if not CATEGORIES[n].locked})
    configured = _alerts().channels
    if configured and not any(choices.uses(name) for name in configured):
        raise ValueError("keep at least one configured channel on, or safety alerts reach no one")
    return choices


@console_bp.route('/settings/alerts', methods=['GET'])
@documented("Which alerts are sent, and where", tag="alerts", access="user",
            responses={200: ("The channels and the kinds of alert", schemas.AlertSettings)})
@require_auth
def get_alert_settings():
    return format_response(True, data=_alert_settings(_stored_choices()))


@console_bp.route('/settings/alerts', methods=['PUT'])
@documented("Choose which alerts are sent, and where", tag="alerts", access="admin",
            body=schemas.AlertSettingsBody,
            description="Turns configured channels and optional kinds of alert on or off; what is not "
                        "mentioned keeps its choice. Safety and problem alerts are always sent, to at least one "
                        "channel. Channel addresses and tokens are set in .env only. Every process sees the "
                        "change within seconds.",
            responses={200: ("Saved", schemas.AlertSettings), 400: "Not allowed; the error says why"})
@require_auth
@require_admin
def save_alert_settings():
    try:
        choices = _parse_choices(request.get_json(silent=True))
    except ValueError as e:
        return format_response(False, error=str(e), status_code=400)
    app_settings.write(_app(), app_settings.ALERTS, choices.to_dict(), updated_by=getattr(g, 'username', None),
                       now=datetime.now(UTC))
    record_audit_event('ALERT_SETTINGS_SAVED', True, choices.to_dict())
    return format_response(True, data=_alert_settings(choices), message='Saved')


@console_bp.route('/settings/alerts/test', methods=['POST'])
@documented("Send a test alert", tag="alerts", access="admin",
            description="Sends one alert now to every channel in use and says what each answered.",
            responses={200: ("What each channel answered", schemas.TestAlertResults),
                       409: "No channel is configured in .env"})
@require_auth
@require_admin
def send_test_alert():
    alerts = _alerts()
    if not alerts.enabled:
        return format_response(False, error='No alert channel is configured: set ALERT_TELEGRAM_BOT_TOKEN and '
                                            'ALERT_TELEGRAM_CHAT_ID, ALERT_NTFY_URL or ALERT_WEBHOOK_URL in .env',
                               status_code=409)
    results = alerts.deliver_now(Alert("NOCTURNA test alert", "Alerts reach you.", urgent=True, category="test"))
    record_audit_event('ALERT_TEST', all(error is None for _, error in results),
                       {'results': {name: error or 'sent' for name, error in results}})
    return format_response(True, data={'results': [{'channel': name, 'sent': error is None, 'error': error}
                                                   for name, error in results]})


# =============================================================================
# PREFERENCES
# =============================================================================

def _user() -> User | None:
    user_id = str(getattr(g, 'user_id', '') or '')
    return db.session.get(User, int(user_id)) if user_id.isdigit() else None


@console_bp.route('/me/preferences', methods=['GET'])
@documented("Your dashboard preferences", tag="preferences", access="user",
            responses={200: ("Your preferences", schemas.Preferences), 404: "Signed in with a key of no user"})
@require_auth
def get_preferences():
    user = _user()
    if user is None:
        return format_response(False, error='User not found', status_code=404)
    return format_response(True, data=preferences.of(user))


@console_bp.route('/me/preferences', methods=['PUT'])
@documented("Change your dashboard preferences", tag="preferences", access="user", body=schemas.PreferencesBody,
            description="Only the fields given change. They change how the dashboard looks to you, nothing else.",
            responses={200: ("Saved", schemas.Preferences), 400: "Invalid; the error says why",
                       404: "Signed in with a key of no user"})
@require_auth
def save_preferences():
    user = _user()
    if user is None:
        return format_response(False, error='User not found', status_code=404)
    try:
        chosen = preferences.parse(request.get_json(silent=True), preferences.of(user))
    except preferences.PreferencesError as e:
        return format_response(False, error=str(e), status_code=400)
    preferences.save(user, chosen)
    db.session.commit()
    return format_response(True, data=chosen, message='Saved')
