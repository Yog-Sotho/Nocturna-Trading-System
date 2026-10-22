"""
Tests for operations: alerts, the heartbeat, the engine monitors, readiness,
Prometheus metrics and JSON logs.
"""

import json
import logging
import re
import threading
import time
import urllib.error
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import SecretStr

from src.trading.alerts import (
    MIN_PING_GAP,
    Alert,
    AlertDispatcher,
    EngineAlerts,
    EngineMonitor,
    Heartbeat,
    NtfySink,
    ReachabilityMonitor,
    TelegramSink,
    WebhookSink,
    build_alerts,
    event_alert,
    start_monitoring,
)
from src.trading.clock import SimulatedClock
from src.trading.domain import Fill, Side
from src.trading.engine import EngineEvent
from src.trading.metrics import CONTENT_TYPE, render

T0 = datetime(2026, 3, 2, 17, 0, tzinfo=UTC)


class Posts:
    """A fake transport that records every POST; queued exceptions are raised first, one per call."""

    def __init__(self, *failures):
        self.calls = []
        self.failures = list(failures)

    def __call__(self, url, body, headers):
        self.calls.append((url, body, dict(headers)))
        if self.failures:
            raise self.failures.pop(0)


class Collect:
    name = "collect"

    def __init__(self):
        self.alerts = []

    def send(self, alert):
        self.alerts.append(alert)


def _http_error(code):
    return urllib.error.HTTPError("https://api.telegram.org/botSECRET-TOKEN/sendMessage", code, "error", {}, None)


def _dispatcher(*sinks, clock=None):
    return AlertDispatcher(list(sinks), clock=clock, retry_delays=(0.0, 0.0))


def _sent(dispatcher, sink):
    assert dispatcher.flush(5)
    return [a.title for a in sink.alerts]


class TestSinks:
    def test_telegram(self):
        posts = Posts()
        sink = TelegramSink("123:abc", "42", post=posts)
        sink.send(Alert("Kill switch", "drawdown 10%", urgent=True))
        sink.send(Alert("Note", "x" * 5000))
        (url, body, headers), (_, long_body, _) = posts.calls
        assert url == "https://api.telegram.org/bot123:abc/sendMessage"
        assert headers == {"Content-Type": "application/json"}
        assert json.loads(body) == {"chat_id": "42", "text": "Kill switch\ndrawdown 10%", "disable_notification": False}
        long = json.loads(long_body)
        assert len(long["text"]) == 4096 and long["disable_notification"] is True

    def test_ntfy(self):
        posts = Posts()
        NtfySink("https://ntfy.sh/topic", token="tk_x", post=posts).send(Alert("Halt", "é" * 3000, urgent=True))
        NtfySink("https://ntfy.sh/topic", post=posts).send(Alert("Trading resumed"))
        (url, body, headers), (_, plain_body, plain_headers) = posts.calls
        assert url == "https://ntfy.sh/topic"
        assert headers == {"Title": "Halt", "Priority": "5", "Tags": "rotating_light", "Authorization": "Bearer tk_x"}
        assert len(body) <= 4096 and body.decode() == "é" * (len(body) // 2)  # cut on a character boundary
        assert plain_body == b"Trading resumed" and plain_headers["Priority"] == "3"
        assert "Authorization" not in plain_headers

    def test_webhook(self):
        posts = Posts()
        WebhookSink("https://hooks.example/x", post=posts).send(Alert("Broker error", "timeout", urgent=True))
        [(url, body, _)] = posts.calls
        sent = json.loads(body)
        assert url == "https://hooks.example/x"
        assert (sent["title"], sent["message"], sent["urgent"]) == ("Broker error", "timeout", True)
        assert datetime.fromisoformat(sent["timestamp"]).tzinfo is not None

    def test_build_alerts_from_settings(self, app):
        settings = app.extensions["settings"].model_copy(update={
            "alert_telegram_bot_token": SecretStr("123:abc"), "alert_telegram_chat_id": "42",
            "alert_ntfy_url": SecretStr("https://ntfy.sh/t"), "alert_webhook_url": SecretStr("https://h.example")})
        assert [s.name for s in build_alerts(settings)._sinks] == ["telegram", "ntfy", "webhook"]
        assert build_alerts(app.extensions["settings"]).enabled is False


class TestDispatcher:
    def test_sends_to_every_sink(self):
        first, second = Collect(), Collect()
        dispatcher = _dispatcher(first, second)
        dispatcher.notify(Alert("one"))
        dispatcher.notify(Alert("two"))
        assert _sent(dispatcher, first) == ["one", "two"] and _sent(dispatcher, second) == ["one", "two"]

    def test_repeats_within_the_window_are_sent_once(self):
        clock, sink = SimulatedClock(T0), Collect()
        dispatcher = _dispatcher(sink, clock=clock)
        for _ in range(3):
            dispatcher.notify(Alert("Broker error", key="broker_error:timeout"))
        dispatcher.notify(Alert("Trading resumed"))
        dispatcher.notify(Alert("Trading resumed"))  # no key: never deduplicated
        clock.advance_to(T0 + timedelta(minutes=16))
        dispatcher.notify(Alert("Broker error", key="broker_error:timeout"))
        assert _sent(dispatcher, sink) == ["Broker error", "Trading resumed", "Trading resumed", "Broker error"]

    def test_retries_what_is_worth_retrying(self):
        posts = Posts(_http_error(503), urllib.error.URLError("down"))
        dispatcher = _dispatcher(TelegramSink("t", "1", post=posts))
        dispatcher.notify(Alert("a"))
        assert dispatcher.flush(5) and len(posts.calls) == 3  # two failures, then delivered
        rejected = Posts(_http_error(400))
        dispatcher = _dispatcher(TelegramSink("t", "1", post=rejected))
        dispatcher.notify(Alert("b"))
        assert dispatcher.flush(5) and len(rejected.calls) == 1  # a bad request never succeeds

    def test_failures_never_log_the_secret_url(self, caplog):
        posts = Posts(*([urllib.error.URLError("https://api.telegram.org/botSECRET-TOKEN")] * 3))
        dispatcher = _dispatcher(TelegramSink("SECRET-TOKEN", "1", post=posts))
        with caplog.at_level(logging.WARNING, logger="src.trading.alerts"):
            dispatcher.notify(Alert("a"))
            assert dispatcher.flush(5)
        assert "Alert via telegram failed (attempt 3): URLError" in caplog.text
        assert "SECRET-TOKEN" not in caplog.text

    def test_without_sinks_alerts_are_only_logged(self, caplog):
        dispatcher = AlertDispatcher([])
        with caplog.at_level(logging.INFO, logger="src.trading.alerts"):
            dispatcher.notify(Alert("Kill switch", "drawdown", urgent=True))
        assert dispatcher.enabled is False and "Alert: Kill switch. drawdown" in caplog.text


class TestSendTestAlert:
    def _invoke(self, app):
        with app.app_context():
            return app.test_cli_runner().invoke(args=["send-test-alert"])

    def test_reports_each_channel(self, app, monkeypatch):
        sink = Collect()
        failing = TelegramSink("SECRET-TOKEN", "1", post=Posts(_http_error(401)))
        monkeypatch.setitem(app.extensions, "alerts", _dispatcher(sink, failing))
        result = self._invoke(app)
        assert result.exit_code == 1
        assert result.output == "collect: sent\ntelegram: FAILED (HTTP 401)\n"
        assert sink.alerts[0].urgent and "SECRET-TOKEN" not in result.output
        monkeypatch.setitem(app.extensions, "alerts", _dispatcher(Collect()))
        assert self._invoke(app).exit_code == 0

    def test_without_channels(self, app):
        result = self._invoke(app)
        assert result.exit_code != 0 and "No alert channel is configured" in result.output


class TestEventAlerts:
    def _event(self, kind, **fields):
        return EngineEvent(kind, T0, **fields)

    def test_a_kill_switch_is_one_alert(self):
        sink = Collect()
        dispatcher = _dispatcher(sink)
        listener = EngineAlerts(dispatcher)
        listener(self._event("kill_switch", detail="drawdown 10.0% from the high-water mark"))
        listener(self._event("halt", detail="drawdown 10.0% from the high-water mark"))
        listener(self._event("halt", detail="operator: drill"))
        assert _sent(dispatcher, sink) == ["Kill switch: trading halted", "Trading halted"]
        assert all(a.urgent for a in sink.alerts)

    def test_unknown_fill(self):
        fill = Fill("f-1", "manual-1", "SPY", Side.BUY, Decimal("3"), Decimal("500.25"), Decimal("0"), T0)
        alert = event_alert(self._event("unknown_fill", fill=fill))
        assert alert.urgent and alert.message == "buy 3 SPY at 500.25 (client order manual-1)"
        assert alert.key == "unknown_fill:f-1"

    def test_a_live_halt_reaches_the_alerts(self):
        from datetime import date

        from src.tests.trading_fixtures import IEF, SPY, daily_bars, ny
        from src.trading.broker import SimulatedBroker
        from src.trading.data import InMemoryBarSource
        from src.trading.lineup import build_lineup
        from src.trading.service import EngineService

        sink = Collect()
        dispatcher = _dispatcher(sink)
        clock = SimulatedClock(ny(2026, 3, 2, 12))
        history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
                   "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
        service = EngineService(clock=clock, source=InMemoryBarSource(history, clock),
                                broker=SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, Decimal("100000")),
                                allocations=build_lineup({"SPY": SPY, "IEF": IEF}), listeners=[EngineAlerts(dispatcher)])
        service.prepare()
        service.engine.halt("operator: drill")
        assert _sent(dispatcher, sink) == ["Trading halted"] and sink.alerts[0].message == "operator: drill"

    @pytest.mark.parametrize("kind,urgent", [("broker_error", True), ("strategy_error", False),
                                             ("reconciled", False), ("resume", False)])
    def test_problems_alert(self, kind, urgent):
        alert = event_alert(self._event(kind, detail="x"))
        assert alert is not None and alert.urgent is urgent

    @pytest.mark.parametrize("kind", ["order", "fill", "rejected", "risk_state"])
    def test_routine_events_do_not(self, kind):
        assert event_alert(self._event(kind)) is None


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestHeartbeat:
    def _heartbeat(self, url="https://hc-ping.com/uuid", interval=60.0):
        posts, clock = Posts(), FakeClock()
        return Heartbeat(url, interval, post=posts, clock=clock), posts, clock

    def _urls(self, heartbeat, posts):
        assert heartbeat.flush(5)
        return [url for url, _, _ in posts.calls]

    def test_pings_once_per_interval_and_at_once_on_a_change(self):
        heartbeat, posts, clock = self._heartbeat()
        heartbeat.beat(True, "state running")
        clock.t += 30
        heartbeat.beat(True)  # too soon
        clock.t += 30
        heartbeat.beat(True)
        clock.t += 1
        heartbeat.beat(False)  # a change, but healthchecks.io takes 5 pings a minute
        clock.t += MIN_PING_GAP
        heartbeat.beat(False)
        assert self._urls(heartbeat, posts) == ["https://hc-ping.com/uuid", "https://hc-ping.com/uuid",
                                                "https://hc-ping.com/uuid/fail"]
        assert posts.calls[0][1] == b"state running"

    def test_fail_url_keeps_the_slug_and_query(self):
        heartbeat, posts, _ = self._heartbeat("https://hc-ping.com/key/engine?create=1")
        heartbeat.beat(False)
        assert self._urls(heartbeat, posts) == ["https://hc-ping.com/key/engine/fail?create=1"]

    def test_a_failed_ping_is_logged_without_the_url(self, caplog):
        posts = Posts(urllib.error.URLError("https://hc-ping.com/SECRET-UUID"))
        heartbeat = Heartbeat("https://hc-ping.com/SECRET-UUID", post=posts)
        with caplog.at_level(logging.WARNING, logger="src.trading.alerts"):
            heartbeat.beat(True)
            assert heartbeat.flush(5)
        assert "Heartbeat ping failed: URLError" in caplog.text and "SECRET-UUID" not in caplog.text


def _views(state, **status):
    return {"status": {"configured": True, "state": state, **status}, "positions": None, "targets": None,
            "working": []}


class TestEngineMonitor:
    def _monitor(self):
        sink, posts, clock = Collect(), Posts(), FakeClock()
        heartbeat = Heartbeat("https://hc-ping.com/u", post=posts, clock=clock)
        return EngineMonitor(_dispatcher(sink), heartbeat), sink, heartbeat, posts, clock

    def test_alerts_once_per_failure_error_and_cool_down(self):
        monitor, sink, *_ = self._monitor()
        monitor(_views("running", equity="100000"))
        monitor(_views("running", last_error="market data: timeout"))
        monitor(_views("running", last_error="market data: timeout"))
        monitor(_views("running", risk={"cooling_down": True}, equity="96900"))
        monitor(_views("running", risk={"cooling_down": True}))
        monitor(_views("failed", last_error="start-up failed: no account"))
        monitor(_views("failed", last_error="start-up failed: no account"))
        dispatcher = monitor._alerts
        assert _sent(dispatcher, sink) == ["Engine error", "Daily loss limit reached", "Engine failed"]
        assert [a.urgent for a in sink.alerts] == [False, True, True]
        assert sink.alerts[1].message == "No new orders until the next New York trading day (equity 96900)"

    def test_heartbeat_follows_the_engine_state(self):
        monitor, _, heartbeat, posts, clock = self._monitor()
        for state in ("stopped", "starting", "running"):
            monitor(_views(state, equity="100000"))
            clock.t += 61
        monitor(_views("failed"))
        assert heartbeat.flush(5)
        assert [(url, body) for url, body, _ in posts.calls] == [
            ("https://hc-ping.com/u", b"state running, equity 100000"), ("https://hc-ping.com/u/fail", b"state failed")]


class TestReachabilityMonitor:
    def test_alerts_when_a_seen_engine_stops_answering(self):
        sink = Collect()
        dispatcher = _dispatcher(sink)
        monitor = ReachabilityMonitor(dispatcher)
        monitor(_views("unreachable"))  # the API started first: the heartbeat's business
        monitor(_views("running"))
        monitor(_views("unreachable"))
        monitor(_views("unreachable"))
        monitor(_views("stopped"))
        assert _sent(dispatcher, sink) == ["Engine not answering", "Engine answering again"]
        assert sink.alerts[0].urgent


class _Runtime:
    def __init__(self, *states):
        self.states = list(states)

    def views(self):
        return _views(self.states.pop(0) if len(self.states) > 1 else self.states[0])


class TestStartMonitoring:
    def test_nothing_to_report_to_starts_nothing(self, app):
        assert start_monitoring(app) is None

    def test_watches_from_the_web_process(self, app, monkeypatch):
        sink = Collect()
        settings = app.extensions["settings"].model_copy(update={"engine_mode": "external"})
        monkeypatch.setitem(app.extensions, "settings", settings)
        monkeypatch.setitem(app.extensions, "alerts", _dispatcher(sink))
        monkeypatch.setitem(app.extensions, "trading", _Runtime("running", "unreachable"))
        thread = start_monitoring(app, interval=0.01)
        assert thread.daemon and thread.name == "engine-monitor"
        deadline = time.monotonic() + 5
        while not sink.alerts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [a.title for a in sink.alerts] == ["Engine not answering"]


class TestWorkerNotices:
    def test_start_and_stop_are_reported(self, app):
        from src.trading.channel import EngineChannel
        from src.trading.runtime import TradingRuntime
        from src.trading.worker import EngineWorker

        sink = Collect()
        dispatcher = _dispatcher(sink)
        worker = EngineWorker(TradingRuntime(app, app.extensions["settings"]), EngineChannel(app),
                              instance="host:1", alerts=dispatcher)
        stop = threading.Event()
        stop.set()
        worker.run(stop)
        assert _sent(dispatcher, sink) == ["Engine process started", "Engine process stopped"]
        assert sink.alerts[0].message == "instance host:1"


SAMPLE = re.compile(r'^[a-z_]+(\{[a-z_]+="[^"]*"(,[a-z_]+="[^"]*")*\})? (NaN|[+-]Inf|-?[0-9.e+-]+)$')


def _check_exposition(text):
    assert text.endswith("\n")
    declared = set()
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            declared.add(line.split()[2])
        elif not line.startswith("# HELP "):
            assert SAMPLE.match(line), line
            assert line.split("{")[0].split(" ")[0] in declared, line


class TestMetrics:
    def test_running_engine(self, sim_runtime):
        sim_runtime.start()
        sim_runtime.wait_running()
        text = render(sim_runtime.views())
        _check_exposition(text)
        assert 'nocturna_engine_state{state="running"} 1.0' in text
        assert 'nocturna_engine_state{state="stopped"} 0.0' in text
        assert "nocturna_equity_usd 100000.0" in text and "nocturna_engine_halted 0.0" in text
        assert 'nocturna_target_weight_ratio{symbol="SPY"}' in text
        assert "nocturna_drawdown_ratio 0.0" in text

    def test_unreachable_engine(self):
        text = render(_views("unreachable"))
        _check_exposition(text)
        assert 'nocturna_engine_state{state="unreachable"} 1.0' in text and "equity" not in text

    def test_label_values_are_escaped(self):
        views = {**_views("running"), "targets": {'a"b\\c\nd': "0.5"}}
        assert 'nocturna_target_weight_ratio{symbol="a\\"b\\\\c\\nd"} 0.5' in render(views)


class TestOpsRoutes:
    def test_metrics_need_the_token(self, app, client, monkeypatch):
        assert client.get("/metrics").status_code == 404  # METRICS_TOKEN unset
        token = "m" * 40
        settings = app.extensions["settings"].model_copy(update={"metrics_token": SecretStr(token)})
        monkeypatch.setitem(app.extensions, "settings", settings)
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200 and response.headers["Content-Type"] == CONTENT_TYPE
        assert 'nocturna_engine_state{state="stopped"} 1.0' in response.get_data(as_text=True)

    def test_ready(self, client):
        assert client.get("/ready").get_json() == {"status": "ready", "database": "ok"}

    def test_not_ready_without_the_database(self, client, db, monkeypatch):
        def unreachable(*args, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(db.session, "execute", unreachable)
        response = client.get("/ready")
        assert response.status_code == 503 and response.get_json()["database"] == "unreachable"


class TestJsonLogs:
    def test_one_redacted_object_per_line(self, app):
        from src.utils.logger import JsonLogFormatter

        record = logging.LogRecord("src.trading.engine", logging.ERROR, "/app/src/trading/engine.py", 42,
                                   "sending %s", ("api_key=SECRET123",), None)
        with app.test_request_context("/"):
            from flask import g
            g.request_id = "req-1"
            entry = json.loads(JsonLogFormatter().format(record))
        assert entry["level"] == "ERROR" and entry["logger"] == "src.trading.engine"
        assert entry["location"] == "engine.py:42" and entry["request_id"] == "req-1"
        assert "SECRET123" not in entry["message"]
        assert datetime.fromisoformat(entry["timestamp"]).tzinfo is not None

    def test_exceptions_are_included(self):
        from src.utils.logger import JsonLogFormatter

        try:
            raise ValueError("token=abc123")
        except ValueError:
            import sys
            record = logging.LogRecord("x", logging.ERROR, "x.py", 1, "failed", (), sys.exc_info())
        entry = json.loads(JsonLogFormatter().format(record))
        assert "ValueError" in entry["exception"] and "abc123" not in entry["exception"]

    def test_containers_log_json_to_stdout_only(self, app, monkeypatch):
        import logging.handlers

        from src.utils import logger as logger_module

        root, audit = logging.getLogger(), logging.getLogger("nocturna.audit")
        monkeypatch.setattr(root, "handlers", list(root.handlers))
        monkeypatch.setattr(audit, "propagate", audit.propagate)
        monkeypatch.setattr(logger_module, "audit_logger", logger_module.audit_logger)
        level = root.level
        monkeypatch.setenv("LOG_JSON", "true")
        monkeypatch.setenv("LOG_TO_FILE", "false")
        try:
            logger_module.setup_secure_logging(app)
            assert root.handlers and all(not isinstance(h, logging.handlers.RotatingFileHandler)
                                         for h in root.handlers)
            assert all(isinstance(h.formatter, logger_module.JsonLogFormatter) for h in root.handlers)
            assert audit.propagate is True  # audit events join the main log
        finally:
            root.setLevel(level)
