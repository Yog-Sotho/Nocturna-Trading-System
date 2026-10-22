"""
NOCTURNA Trading System - Alerts and the heartbeat.

What needs a person's attention (the kill switch, a halt, the daily loss
limit, broker and strategy errors, a fill the engine did not place, positions
corrected from the broker, a failed engine, an engine process that stopped
answering) is sent to any of:

* Telegram: a bot's ``sendMessage`` (ALERT_TELEGRAM_BOT_TOKEN, ALERT_TELEGRAM_CHAT_ID);
* ntfy: a topic URL such as ``https://ntfy.sh/<topic>`` (ALERT_NTFY_URL, and
  ALERT_NTFY_TOKEN for a protected topic); anyone who knows a public topic's
  name can read it, so pick one that cannot be guessed;
* any webhook that takes a JSON POST (ALERT_WEBHOOK_URL).

Alerts go out from a thread of their own, so the engine never waits on a chat
service, and one repeated within ``DEDUP_WINDOW`` is sent once.

The heartbeat (HEARTBEAT_URL, a healthchecks.io check's ping URL) is the dead
man's switch: it is pinged while the engine runs and pinged at ``/fail`` when
the engine has failed. A stopped engine, a dead process or a dead host sends
nothing, and the check alerts once its grace period runs out.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from src.config import Settings

from .engine import EngineEvent
from .runtime import STALE_AFTER

logger = logging.getLogger(__name__)

#: An alert with the same key as one sent this recently is not sent again
DEDUP_WINDOW = timedelta(minutes=15)
#: Telegram's limit is 4096 characters; ntfy sends longer bodies as attachments
TELEGRAM_MAX_CHARS = 4096
NTFY_MAX_BYTES = 4096
HTTP_TIMEOUT = 10.0
#: healthchecks.io asks for at most 5 pings a minute per check
MIN_PING_GAP = 15.0


@dataclass(frozen=True)
class Alert:
    title: str
    message: str = ""
    urgent: bool = False
    #: Alerts with the same key are sent once per DEDUP_WINDOW; empty never deduplicates
    key: str = ""


Post = Callable[[str, bytes, Mapping[str, str]], None]


def http_post(url: str, body: bytes, headers: Mapping[str, str]) -> None:
    """
    Raises:
        urllib.error.HTTPError: On a non-2xx answer.
        urllib.error.URLError: If the service cannot be reached.
    """
    if urlsplit(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are allowed")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT):  # nosec B310 - scheme checked above
        pass


def _describe_failure(e: Exception) -> str:
    """Never the exception text: it can hold the URL, and the URL holds the secret."""
    return f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__


def _retryable(e: Exception) -> bool:
    return not isinstance(e, urllib.error.HTTPError) or e.code == 429 or e.code >= 500


class Sink(Protocol):
    name: str

    def send(self, alert: Alert) -> None: ...


class TelegramSink:
    name = "telegram"

    def __init__(self, token: str, chat_id: str, post: Post = http_post) -> None:
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._post = post

    def send(self, alert: Alert) -> None:
        text = f"{alert.title}\n{alert.message}" if alert.message else alert.title
        body = {"chat_id": self._chat_id, "text": text[:TELEGRAM_MAX_CHARS],
                "disable_notification": not alert.urgent}
        self._post(self._url, json.dumps(body).encode(), {"Content-Type": "application/json"})


class NtfySink:
    name = "ntfy"

    def __init__(self, url: str, token: str | None = None, post: Post = http_post) -> None:
        self._url = url
        self._token = token
        self._post = post

    def send(self, alert: Alert) -> None:
        headers = {
            # Header values must be Latin-1 for http.client; the message carries anything else
            "Title": alert.title.encode("ascii", "replace").decode(),
            "Priority": "5" if alert.urgent else "3",
            "Tags": "rotating_light" if alert.urgent else "information_source",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        body = (alert.message or alert.title).encode()[:NTFY_MAX_BYTES]
        self._post(self._url, body.decode(errors="ignore").encode(), headers)


class WebhookSink:
    name = "webhook"

    def __init__(self, url: str, post: Post = http_post) -> None:
        self._url = url
        self._post = post

    def send(self, alert: Alert) -> None:
        body = {"title": alert.title, "message": alert.message, "urgent": alert.urgent,
                "timestamp": datetime.now(UTC).isoformat()}
        self._post(self._url, json.dumps(body).encode(), {"Content-Type": "application/json"})


class _Background:
    """Runs jobs in order on one daemon thread, started on first use so it lives in the forked worker."""

    def __init__(self, name: str, maxsize: int = 100) -> None:
        self._name = name
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, job: Callable[[], None]) -> bool:
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            return False
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
        return True

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception:
                logger.exception("%s job failed", self._name)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float) -> bool:
        """Wait until every submitted job has run; False if some are still waiting at ``timeout``."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True


class AlertDispatcher:
    def __init__(self, sinks: list[Sink], clock=None, dedup_window: timedelta = DEDUP_WINDOW,
                 retry_delays: tuple[float, ...] = (2.0, 10.0)) -> None:
        """
        Args:
            retry_delays: Waits before each retry of a send that failed for a reason worth retrying.
        """
        self._sinks = list(sinks)
        self._now = clock.now if clock is not None else (lambda: datetime.now(UTC))
        self._window = dedup_window
        self._retry_delays = retry_delays
        self._lock = threading.Lock()
        self._sent: dict[str, datetime] = {}
        self._background = _Background("alerts")

    @property
    def enabled(self) -> bool:
        return bool(self._sinks)

    def notify(self, alert: Alert) -> None:
        logger.log(logging.WARNING if alert.urgent else logging.INFO, "Alert: %s. %s", alert.title, alert.message)
        if not self._sinks:
            return
        if alert.key:
            now = self._now()
            with self._lock:
                self._sent = {k: at for k, at in self._sent.items() if now - at < self._window}
                if alert.key in self._sent:
                    return
                self._sent[alert.key] = now
        if not self._background.submit(lambda: self._deliver(alert)):
            logger.error("Too many alerts waiting; dropped: %s", alert.title)

    def _deliver(self, alert: Alert) -> None:
        for sink in self._sinks:
            for attempt, delay in enumerate((0.0, *self._retry_delays), start=1):
                time.sleep(delay)
                try:
                    sink.send(alert)
                    break
                except Exception as e:
                    logger.warning("Alert via %s failed (attempt %d): %s", sink.name, attempt, _describe_failure(e))
                    if not _retryable(e):
                        break

    def flush(self, timeout: float = 30.0) -> bool:
        return self._background.flush(timeout)


def build_alerts(settings: Settings) -> AlertDispatcher:
    sinks: list[Sink] = []
    if settings.alert_telegram_bot_token is not None and settings.alert_telegram_chat_id:
        sinks.append(TelegramSink(settings.alert_telegram_bot_token.get_secret_value(),
                                  settings.alert_telegram_chat_id))
    if settings.alert_ntfy_url is not None:
        token = settings.alert_ntfy_token.get_secret_value() if settings.alert_ntfy_token else None
        sinks.append(NtfySink(settings.alert_ntfy_url.get_secret_value(), token))
    if settings.alert_webhook_url is not None:
        sinks.append(WebhookSink(settings.alert_webhook_url.get_secret_value()))
    return AlertDispatcher(sinks)


# -- what is worth an alert ------------------------------------------------------------


class EngineAlerts:
    """An engine listener: the events that need a person become alerts."""

    def __init__(self, alerts: AlertDispatcher) -> None:
        self._alerts = alerts

    def __call__(self, event: EngineEvent) -> None:
        alert = event_alert(event)
        if alert is not None:
            self._alerts.notify(alert)


def event_alert(event: EngineEvent) -> Alert | None:
    detail = event.detail
    match event.kind:
        case "kill_switch":
            # The halt that follows carries the same reason, so it is not sent twice
            return Alert("Kill switch: trading halted", detail, urgent=True, key=f"halt:{detail}")
        case "halt":
            return Alert("Trading halted", detail, urgent=True, key=f"halt:{detail}")
        case "resume":
            return Alert("Trading resumed")
        case "broker_error":
            return Alert("Broker error", detail, urgent=True, key=f"broker_error:{detail}")
        case "strategy_error":
            return Alert("Strategy error", detail, key=f"strategy_error:{detail}")
        case "unknown_fill" if event.fill is not None:
            fill = event.fill
            return Alert("Fill the engine did not place",
                         f"{fill.side.value} {fill.qty} {fill.symbol} at {fill.price} "
                         f"(client order {fill.client_order_id})", urgent=True, key=f"unknown_fill:{fill.fill_id}")
        case "reconciled":
            return Alert("Positions corrected from the broker", detail, key="reconciled")
    return None


class Heartbeat:
    """
    Pings ``url`` while the engine is healthy and ``url``/fail when it is not:
    at once when that changes, otherwise once per ``interval``, and never
    faster than ``MIN_PING_GAP``.
    """

    def __init__(self, url: str, interval: float = 60.0, post: Post = http_post,
                 clock: Callable[[], float] = time.monotonic) -> None:
        parts = urlsplit(url)
        self._ok_url = url
        self._fail_url = urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/fail"))
        self._interval = max(interval, MIN_PING_GAP)
        self._post = post
        self._clock = clock
        self._last: float | None = None
        self._healthy: bool | None = None
        self._background = _Background("heartbeat", maxsize=10)

    def beat(self, healthy: bool, detail: str = "") -> None:
        now = self._clock()
        if self._last is not None:
            gap = now - self._last
            if gap < MIN_PING_GAP or (healthy == self._healthy and gap < self._interval):
                return
        self._last, self._healthy = now, healthy
        url = self._ok_url if healthy else self._fail_url
        self._background.submit(lambda: self._ping(url, detail))

    def _ping(self, url: str, detail: str) -> None:
        try:
            self._post(url, detail.encode(), {"Content-Type": "text/plain; charset=utf-8"})
        except Exception as e:
            logger.warning("Heartbeat ping failed: %s", _describe_failure(e))

    def flush(self, timeout: float = 30.0) -> bool:
        return self._background.flush(timeout)


def build_heartbeat(settings: Settings) -> Heartbeat | None:
    if settings.heartbeat_url is None:
        return None
    return Heartbeat(settings.heartbeat_url.get_secret_value(), settings.heartbeat_interval_seconds)


class EngineMonitor:
    """
    Watches the engine's views where the engine runs: alerts when it fails,
    reports a new error or reaches the daily loss limit, and drives the
    heartbeat. Call it with each views document.
    """

    def __init__(self, alerts: AlertDispatcher, heartbeat: Heartbeat | None = None) -> None:
        self._alerts = alerts
        self._heartbeat = heartbeat
        self._state: str | None = None
        self._error: str | None = None
        self._cooling_down = False

    def __call__(self, views: Mapping[str, Any]) -> None:
        status = views["status"]
        state, error = status["state"], status.get("last_error")
        if state == "failed" and self._state != "failed":
            self._alerts.notify(Alert("Engine failed", error or "", urgent=True))
        if state == "running" and error and error != self._error:
            category = error.partition(":")[0]
            self._alerts.notify(Alert("Engine error", error, key=f"engine_error:{category}"))
        cooling_down = bool((status.get("risk") or {}).get("cooling_down"))
        if cooling_down and not self._cooling_down:
            self._alerts.notify(Alert(
                "Daily loss limit reached",
                f"No new orders until the next New York trading day (equity {status.get('equity')})",
                urgent=True, key="cooling_down"))
        self._state, self._error, self._cooling_down = state, error, cooling_down
        if self._heartbeat is not None and state in ("running", "failed"):
            self._heartbeat.beat(state == "running", _summary(status))


def _summary(status: Mapping[str, Any]) -> str:
    parts = [f"state {status['state']}"]
    for name in ("equity", "halted", "last_error"):
        if status.get(name) is not None:
            parts.append(f"{name} {status[name]}")
    return ", ".join(parts)


class ReachabilityMonitor:
    """
    Watches, from the web API, that a separate engine process still answers:
    the engine process reports everything else itself, but cannot report its
    own death. An engine never seen since this process started is left to the
    heartbeat, so starting the API before the engine does not alert.
    """

    def __init__(self, alerts: AlertDispatcher) -> None:
        self._alerts = alerts
        self._state: str | None = None

    def __call__(self, views: Mapping[str, Any]) -> None:
        state = views["status"]["state"]
        if state == "unreachable" and self._state not in (None, "unreachable"):
            self._alerts.notify(Alert(
                "Engine not answering",
                f"The engine process has not published a heartbeat for {int(STALE_AFTER.total_seconds())} seconds; "
                "trading may have stopped. The emergency stop still reaches the broker.",
                urgent=True, key="engine_unreachable"))
        elif self._state == "unreachable" and state != "unreachable":
            self._alerts.notify(Alert("Engine answering again", f"state {state}"))
        if state != "unreachable" or self._state is not None:
            self._state = state


def start_monitoring(app, interval: float = 5.0) -> threading.Thread | None:
    """
    Watch the engine from the web process: all of it when the engine runs
    here (embedded), only that it answers when it runs as its own process.
    Call in the serving process (after gunicorn's fork), not at app creation.
    """
    settings: Settings = app.extensions["settings"]
    alerts: AlertDispatcher = app.extensions["alerts"]
    runtime = app.extensions["trading"]
    monitor: Callable[[Mapping[str, Any]], None]
    if settings.engine_mode == "external":
        if not alerts.enabled:
            return None
        monitor = ReachabilityMonitor(alerts)
    else:
        heartbeat = build_heartbeat(settings)
        if not alerts.enabled and heartbeat is None:
            return None
        monitor = EngineMonitor(alerts, heartbeat)

    def watch() -> None:
        while True:
            try:
                monitor(runtime.views())
            except Exception:
                logger.exception("Engine monitor failed")
            time.sleep(interval)

    thread = threading.Thread(target=watch, name="engine-monitor", daemon=True)
    thread.start()
    return thread
