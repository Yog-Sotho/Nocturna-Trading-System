"""
NOCTURNA Trading System - The engine as its own process.

``flask engine run`` keeps one ``TradingRuntime`` (the engine service and its
thread) alive in a process of its own, so redeploying or restarting the web
API never restarts trading. Every second the worker:

* runs the commands the API queued (start, stop, resume, halt, orders) in the
  order they were sent, and stores each result;
* publishes the engine's views with a heartbeat, which the API reads.

A stopped worker publishes a final ``stopped`` status, so the API reports the
engine as stopped instead of waiting for the heartbeat to go stale. A worker
refuses to start while another one's heartbeat is fresh: two engines must
never trade one account.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .alerts import Alert, AlertDispatcher
from .channel import EngineChannel
from .operator import OperatorOrder
from .runtime import STALE_AFTER, TradingRuntime

logger = logging.getLogger(__name__)


class AnotherWorkerRunningError(RuntimeError):
    """Another engine worker published a fresh heartbeat."""


class EngineWorker:
    def __init__(self, runtime: TradingRuntime, channel: EngineChannel, interval: float = 1.0,
                 instance: str | None = None, on_step: Callable[[dict[str, Any]], None] | None = None,
                 alerts: AlertDispatcher | None = None) -> None:
        """
        Args:
            on_step: Called with the published views after every step (monitoring, heartbeat).
            alerts: Told when the worker starts and stops; a start without a stop before it was a crash.
        """
        self.runtime = runtime
        self.channel = channel
        self.interval = interval
        self.instance = instance or f"{socket.gethostname()}:{os.getpid()}"
        self._on_step = on_step
        self._alerts = alerts
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "start": lambda p: self.runtime.start(),
            "stop": lambda p: self.runtime.stop(),
            "resume": lambda p: self.runtime.resume(),
            "halt": lambda p: self.runtime.emergency_stop(p["reason"], bool(p["flatten"])),
            "submit_order": self._submit_order,
            "cancel_order": lambda p: {"requested": self.runtime.cancel_order(p["client_order_id"])},
        }

    def _submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision, order = self.runtime.submit_order(OperatorOrder.from_dict(payload))
        return {"allowed": decision.allowed, "reason": decision.reason, "order": order}

    def step(self) -> None:
        for command_id, kind, payload in self.channel.take_pending():
            handler = self._handlers.get(kind)
            try:
                if handler is None:
                    raise ValueError(f"unknown command {kind!r}")
                result = handler(payload)
            except Exception as e:
                logger.warning("Engine command %s (%s) failed: %s", command_id, kind, e)
                self.channel.complete(command_id, error=(type(e).__name__, str(e)))
            else:
                self.channel.complete(command_id, result)
        views = self.runtime.views()
        self.channel.publish(self.instance, views)
        if self._on_step is not None:
            try:
                self._on_step(views)
            except Exception:
                logger.exception("Engine worker step hook failed")

    def ensure_alone(self) -> None:
        """
        Raises:
            AnotherWorkerRunningError: If another worker's heartbeat is fresh and it is not stopped.
        """
        document, at, instance = self.channel.status_with_instance()
        if document is None or at is None or instance == self.instance:
            return  # nothing published yet, or this very worker before a container restart
        state = document.get("status", {}).get("state")
        if datetime.now(UTC) - at <= STALE_AFTER and state != "stopped":
            raise AnotherWorkerRunningError(
                f"another engine worker published a heartbeat {int((datetime.now(UTC) - at).total_seconds())} "
                "seconds ago; stop it first")

    def run(self, stop: threading.Event) -> None:
        """Run until ``stop`` is set; starts the engine first when ENGINE_AUTOSTART is set."""
        self.ensure_alone()
        logger.info("Engine worker %s started", self.instance)
        self.runtime.autostart()
        state = self.runtime.views()["status"]["state"]
        self._notify("Engine process started", f"engine {state}" + (
            ": trading waits until you start it (dashboard, or /start_engine in Telegram)"
            if state == "stopped" else ""))
        try:
            while not stop.is_set():
                try:
                    self.step()
                except Exception:
                    logger.exception("Engine worker step failed")
                stop.wait(self.interval)
        finally:
            self.runtime.stop()
            try:
                self.channel.publish(self.instance, self.runtime.views())
            except Exception:
                logger.exception("Publishing the final engine status failed")
            logger.info("Engine worker %s stopped", self.instance)
            self._notify("Engine process stopped")

    def _notify(self, title: str, detail: str = "") -> None:
        if self._alerts is not None:
            self._alerts.notify(Alert(title, f"instance {self.instance}" + (f"; {detail}" if detail else "")))
