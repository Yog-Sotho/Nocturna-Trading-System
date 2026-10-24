"""
Tests for the dashboard's control panel API: the strategy lab's runs, the
alert choices and each user's preferences.
"""

from datetime import UTC, datetime

import pytest

from src.models.user import AuditLog, User
from src.trading import app_settings, lab
from src.trading.alerts import AlertChoices, AlertDispatcher

T0 = datetime(2026, 10, 23, 12, tzinfo=UTC)
VALIDATE = {"kind": "validate", "strategy": "tsmom", "start": "2017-01-03", "end": "2026-10-01"}


class FakeLauncher:
    def __init__(self):
        self.launches = 0

    def launch(self):
        self.launches += 1


@pytest.fixture
def launcher(app, monkeypatch):
    fake = FakeLauncher()
    monkeypatch.setitem(app.extensions, "lab", fake)
    return fake


class Sink:
    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.alerts = []

    def send(self, alert):
        if self.error is not None:
            raise self.error
        self.alerts.append(alert)


@pytest.fixture
def channels(app, monkeypatch):
    """Telegram and ntfy configured; the dispatcher reads the stored choices as the real one does."""
    sinks = [Sink("telegram"), Sink("ntfy", error=ConnectionError("no route"))]
    dispatcher = AlertDispatcher(sinks, retry_delays=(), choices=app_settings.alert_choices(app, ttl=0))
    monkeypatch.setitem(app.extensions, "alerts", dispatcher)
    return sinks


# -- the strategy lab ---------------------------------------------------------------------


class TestLabRuns:
    def test_queue_a_run(self, app, client, admin_headers, auth_headers, launcher):
        resp = client.post("/api/v1/lab/runs", headers=admin_headers, json=VALIDATE)
        assert resp.status_code == 202, resp.get_json()
        run = resp.get_json()["data"]["run"]
        assert run["status"] == "queued" and run["title"] == "Validate tsmom (alpaca, 2017-01-03 to 2026-10-01)"
        assert run["created_by"] == "adminuser" and launcher.launches == 1
        listed = client.get("/api/v1/lab/runs", headers=auth_headers).get_json()["data"]["runs"]
        assert [r["id"] for r in listed] == [run["id"]]
        detail = client.get(f"/api/v1/lab/runs/{run['id']}", headers=auth_headers).get_json()["data"]["run"]
        assert detail["result"] is None and detail["report"] is None
        with app.app_context():
            entry = AuditLog.query.filter_by(event_type="LAB_RUN_QUEUED").one()
            assert entry.event_data["run_id"] == run["id"] and entry.event_data["kind"] == "validate"

    def test_a_finished_run_shows_its_result_and_the_strategy_its_verdict(self, app, client, admin_headers,
                                                                          auth_headers, launcher, monkeypatch):
        monkeypatch.setattr(lab, "execute", lambda settings, kind, spec: (
            {"headline": "rejected: fails out-of-sample Sharpe", "passed": False, "curve": []}, "# Validation"))
        run_id = client.post("/api/v1/lab/runs", headers=admin_headers, json=VALIDATE).get_json()["data"]["run"]["id"]
        lab.drain(app, app.extensions["settings"])
        detail = client.get(f"/api/v1/lab/runs/{run_id}", headers=auth_headers).get_json()["data"]["run"]
        assert detail["status"] == "done" and detail["report"] == "# Validation"
        assert detail["headline"] == "rejected: fails out-of-sample Sharpe"
        strategies = client.get("/api/v1/trading/strategies", headers=auth_headers).get_json()["data"]["strategies"]
        verdicts = {s["name"]: s["lab"] for s in strategies}
        assert verdicts["tsmom"]["run_id"] == run_id and verdicts["tsmom"]["passed"] is False
        assert verdicts["benchmark"] is None

    def test_listing_starts_a_process_for_runs_left_waiting(self, app, client, auth_headers, launcher):
        lab.submit(app, "validate", lab.parse_request(app.extensions["settings"], VALIDATE, T0.date())[1],
                   "adminuser", T0)
        client.get("/api/v1/lab/runs?limit=500", headers=auth_headers)
        assert launcher.launches == 1

    def test_only_an_admin_queues_runs(self, client, auth_headers, launcher):
        assert client.post("/api/v1/lab/runs", headers=auth_headers, json=VALIDATE).status_code == 403
        assert client.get("/api/v1/lab/runs").status_code == 401
        assert launcher.launches == 0

    def test_what_cannot_be_run_is_refused(self, client, admin_headers, launcher):
        resp = client.post("/api/v1/lab/runs", headers=admin_headers, json={**VALIDATE, "start": "2025-01-02"})
        assert resp.status_code == 400 and "at least 4 years" in resp.get_json()["error"]
        resp = client.post("/api/v1/lab/runs", headers=admin_headers, data="nope")
        assert resp.status_code == 400 and resp.get_json()["error"] == "the body must be a JSON object"
        assert launcher.launches == 0

    def test_too_many_waiting(self, client, admin_headers, launcher):
        for _ in range(lab.MAX_WAITING):
            assert client.post("/api/v1/lab/runs", headers=admin_headers, json=VALIDATE).status_code == 202
        resp = client.post("/api/v1/lab/runs", headers=admin_headers, json=VALIDATE)
        assert resp.status_code == 409 and "already waiting" in resp.get_json()["error"]

    def test_a_launch_failure_leaves_the_run_queued(self, app, client, admin_headers, monkeypatch):
        class Broken:
            def launch(self):
                raise OSError("no such file")

        monkeypatch.setitem(app.extensions, "lab", Broken())
        resp = client.post("/api/v1/lab/runs", headers=admin_headers, json=VALIDATE)
        assert resp.status_code == 202 and resp.get_json()["data"]["run"]["status"] == "queued"

    def test_an_unknown_run(self, client, auth_headers):
        assert client.get("/api/v1/lab/runs/42", headers=auth_headers).status_code == 404


# -- alerts -------------------------------------------------------------------------------


class TestAlertSettings:
    def test_defaults(self, client, auth_headers):
        data = client.get("/api/v1/settings/alerts", headers=auth_headers).get_json()["data"]
        assert [(c["name"], c["configured"], c["enabled"]) for c in data["channels"]] == [
            ("telegram", False, False), ("ntfy", False, False), ("webhook", False, False)]
        categories = {c["name"]: c for c in data["categories"]}
        assert list(categories) == ["safety", "problems", "warnings", "lifecycle", "fills", "lab"]
        assert categories["safety"]["locked"] and categories["safety"]["enabled"]
        assert not categories["fills"]["locked"] and not categories["fills"]["enabled"]
        assert categories["lab"]["enabled"]

    def test_choosing(self, app, client, admin_headers, auth_headers, channels):
        resp = client.put("/api/v1/settings/alerts", headers=admin_headers,
                          json={"channels": {"ntfy": False}, "categories": {"fills": True, "lifecycle": False}})
        assert resp.status_code == 200, resp.get_json()
        data = client.get("/api/v1/settings/alerts", headers=auth_headers).get_json()["data"]
        assert {c["name"]: c["enabled"] for c in data["channels"]} == {"telegram": True, "ntfy": False,
                                                                        "webhook": False}
        enabled = {c["name"]: c["enabled"] for c in data["categories"]}
        assert enabled["fills"] and not enabled["lifecycle"] and enabled["warnings"]
        # What is not mentioned keeps its choice
        client.put("/api/v1/settings/alerts", headers=admin_headers, json={"categories": {"warnings": False}})
        stored = AlertChoices.from_dict(app_settings.read(app, app_settings.ALERTS))
        assert stored.categories == {"fills": True, "lifecycle": False, "warnings": False}
        assert stored.channels == {"ntfy": False}
        with app.app_context():
            assert AuditLog.query.filter_by(event_type="ALERT_SETTINGS_SAVED").count() == 2

    @pytest.mark.parametrize("body, message", [
        ({"categories": {"safety": False}}, "Safety alerts cannot be turned off"),
        ({"categories": {"problems": False, "safety": False}}, "Safety, Problems alerts cannot be turned off"),
        ({"categories": {"gossip": True}}, "unknown: gossip"),
        ({"channels": {"telegram": "off"}}, "channels and categories map names to true or false"),
        ({"channels": {"telegram": False, "ntfy": False}}, "keep at least one configured channel on"),
        ({"sound": "loud"}, "the body must be an object with channels and categories"),
        ([], "the body must be an object with channels and categories"),
    ])
    def test_refused(self, client, admin_headers, channels, body, message):
        resp = client.put("/api/v1/settings/alerts", headers=admin_headers, json=body)
        assert resp.status_code == 400 and message in resp.get_json()["error"]

    def test_only_an_admin_chooses(self, client, auth_headers):
        assert client.put("/api/v1/settings/alerts", headers=auth_headers, json={}).status_code == 403

    def test_a_test_alert(self, app, client, admin_headers, channels):
        resp = client.post("/api/v1/settings/alerts/test", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.get_json()["data"]["results"] == [
            {"channel": "telegram", "sent": True, "error": None},
            {"channel": "ntfy", "sent": False, "error": "ConnectionError"}]
        assert channels[0].alerts[0].title == "NOCTURNA test alert"
        client.put("/api/v1/settings/alerts", headers=admin_headers, json={"channels": {"ntfy": False}})
        results = client.post("/api/v1/settings/alerts/test", headers=admin_headers).get_json()["data"]["results"]
        assert [r["channel"] for r in results] == ["telegram"]
        with app.app_context():
            assert AuditLog.query.filter_by(event_type="ALERT_TEST").count() == 2

    def test_a_test_alert_needs_a_channel(self, client, admin_headers):
        resp = client.post("/api/v1/settings/alerts/test", headers=admin_headers)
        assert resp.status_code == 409 and "ALERT_TELEGRAM_BOT_TOKEN" in resp.get_json()["error"]

    def test_the_dispatcher_follows_the_choices(self, client, admin_headers, channels, app):
        from src.trading.alerts import Alert

        client.put("/api/v1/settings/alerts", headers=admin_headers, json={"categories": {"fills": True}})
        dispatcher = app.extensions["alerts"]
        dispatcher.notify(Alert("Filled: buy 3 SPY", category="fills"))
        dispatcher.notify(Alert("Engine process started", category="lifecycle"))
        assert dispatcher.flush(5)
        assert [a.title for a in channels[0].alerts] == ["Filled: buy 3 SPY", "Engine process started"]
        client.put("/api/v1/settings/alerts", headers=admin_headers,
                   json={"categories": {"fills": False, "lifecycle": False}})
        dispatcher.notify(Alert("Filled: sell 3 SPY", category="fills"))
        dispatcher.notify(Alert("Trading halted", category="safety"))
        assert dispatcher.flush(5)
        assert [a.title for a in channels[0].alerts][2:] == ["Trading halted"]


# -- preferences --------------------------------------------------------------------------


class TestPreferences:
    def test_defaults(self, client, auth_headers):
        data = client.get("/api/v1/me/preferences", headers=auth_headers).get_json()["data"]
        assert data == {"display_name": "", "theme": "dark", "accent": "violet",
                        "cards": ["account", "equity", "mix", "positions", "risk", "orders", "fills"], "hidden": []}

    def test_saving_some(self, app, client, auth_headers, test_user):
        resp = client.put("/api/v1/me/preferences", headers=auth_headers,
                          json={"display_name": "  Night Owl ", "theme": "light", "cards": ["mix", "equity"],
                                "accent": None})
        assert resp.status_code == 200, resp.get_json()
        data = client.get("/api/v1/me/preferences", headers=auth_headers).get_json()["data"]
        assert data["display_name"] == "Night Owl" and data["theme"] == "light" and data["accent"] == "violet"
        assert data["cards"] == ["mix", "equity", "account", "positions", "risk", "orders", "fills"]
        client.put("/api/v1/me/preferences", headers=auth_headers, json={"accent": "teal", "hidden": ["fills"]})
        data = client.get("/api/v1/me/preferences", headers=auth_headers).get_json()["data"]
        assert (data["theme"], data["accent"], data["hidden"]) == ("light", "teal", ["fills"])
        with app.app_context():
            user = User.query.filter_by(username="testuser").one()
            assert user.settings["dashboard"]["accent"] == "teal"

    def test_preferences_are_each_users_own(self, client, auth_headers, admin_headers):
        client.put("/api/v1/me/preferences", headers=auth_headers, json={"theme": "light"})
        assert client.get("/api/v1/me/preferences", headers=admin_headers).get_json()["data"]["theme"] == "dark"

    @pytest.mark.parametrize("body, message", [
        ({"theme": "neon"}, "theme must be one of: auto, dark, light"),
        ({"accent": "beige"}, "accent must be one of"),
        ({"display_name": "x" * 41}, "at most 40 characters"),
        ({"display_name": "bell\x07"}, "printable text"),
        ({"cards": ["account", "weather"]}, "unknown cards: weather"),
        ({"cards": ["mix", "mix"]}, "cards names a card twice"),
        ({"hidden": "fills"}, "hidden must be a list of card names"),
        ({"font": "comic"}, "unknown fields: font"),
        ([], "the body must be a JSON object"),
    ])
    def test_refused(self, client, auth_headers, body, message):
        resp = client.put("/api/v1/me/preferences", headers=auth_headers, json=body)
        assert resp.status_code == 400 and message in resp.get_json()["error"]

    def test_stored_values_no_longer_valid_fall_back_to_the_defaults(self, app, client, auth_headers, test_user):
        from src.models.user import db

        with app.app_context():
            user = User.query.filter_by(username="testuser").one()
            user.settings = {"dashboard": {"theme": "retired-theme"}}
            db.session.commit()
        assert client.get("/api/v1/me/preferences", headers=auth_headers).get_json()["data"]["theme"] == "dark"

    def test_an_api_key_of_no_user(self, app, client, monkeypatch):
        from src.routes import console

        with app.test_request_context():
            from flask import g

            g.user_id = "api-key"
            assert console._user() is None
