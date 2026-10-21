"""
NOCTURNA Trading System - The database channel between the API and the engine.

When the engine runs as its own process (ENGINE_MODE=external) the two meet
only in the database:

* the engine publishes its views (``views.engine_views``) and a heartbeat to
  the single ``engine_status`` row on every loop;
* the API queues commands in ``engine_commands``; the engine marks each one
  running, executes them in order and stores each result. A command not
  started before it expires is marked expired and never executed (an order
  must not fill minutes late), and an API that stops waiting withdraws its
  command if the engine has not started it, so a request that reported a
  timeout cannot take effect later. Halts never expire and are never withdrawn.

Every method opens its own app context, so it works from any thread.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Flask

from src.models.trading import EngineCommand, EngineStatusRecord
from src.models.user import db

#: How long a command may wait for the engine before it is not executed at all
COMMAND_TTL = {"halt": None, "submit_order": timedelta(seconds=30), "cancel_order": timedelta(seconds=30),
               "resume": timedelta(seconds=60), "start": timedelta(seconds=60), "stop": timedelta(seconds=60)}


#: Extra wait for a command the engine has already started
RUNNING_GRACE = 10.0


class CommandFailedError(RuntimeError):
    """The engine ran the command and it failed; ``kind`` names the engine's exception."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class CommandTimeoutError(TimeoutError):
    """The engine did not answer in time (it may be down, or busy)."""


class EngineChannel:
    def __init__(self, app: Flask, clock=None) -> None:
        self._app = app
        self._now = clock.now if clock is not None else (lambda: datetime.now(UTC))

    # -- engine side ----------------------------------------------------------------

    def publish(self, instance: str, document: dict[str, Any]) -> None:
        with self._app.app_context():
            row = db.session.get(EngineStatusRecord, 1)
            if row is None:
                row = EngineStatusRecord(id=1)
                db.session.add(row)
            row.instance, row.document, row.heartbeat_at = instance, document, self._now()
            db.session.commit()

    def take_pending(self) -> list[tuple[int, str, dict[str, Any]]]:
        """Pending commands oldest first, now marked running; expired ones are marked so and skipped."""
        now = self._now()
        out = []
        with self._app.app_context():
            for row in EngineCommand.query.filter_by(status="pending").order_by(EngineCommand.id).all():
                if row.expires_at is not None and row.expires_at < now:
                    row.status, row.completed_at = "expired", now
                else:
                    row.status = "running"
                    out.append((row.id, row.kind, dict(row.payload)))
            db.session.commit()
        return out

    def complete(self, command_id: int, result: Any = None, error: tuple[str, str] | None = None) -> None:
        with self._app.app_context():
            row = db.session.get(EngineCommand, command_id)
            if row is None:
                return
            row.status = "failed" if error else "done"
            row.result = result
            row.error = f"{error[0]}: {error[1]}" if error else None
            row.completed_at = self._now()
            db.session.commit()

    # -- API side ---------------------------------------------------------------------

    def status(self) -> tuple[dict[str, Any] | None, datetime | None]:
        """The last published document and when it was published."""
        document, at, _ = self.status_with_instance()
        return document, at

    def status_with_instance(self) -> tuple[dict[str, Any] | None, datetime | None, str | None]:
        with self._app.app_context():
            row = db.session.get(EngineStatusRecord, 1)
            if row is None:
                return None, None, None
            return dict(row.document), row.heartbeat_at, row.instance

    def send(self, kind: str, payload: dict[str, Any], requested_by: str | None = None) -> int:
        now = self._now()
        ttl = COMMAND_TTL[kind]
        with self._app.app_context():
            row = EngineCommand(kind=kind, payload=payload, status="pending", requested_by=requested_by,
                                created_at=now, expires_at=now + ttl if ttl is not None else None)
            db.session.add(row)
            db.session.commit()
            return row.id

    def wait(self, command_id: int, timeout: float, interval: float = 0.1) -> Any:
        """
        The command's result once the engine has run it. At ``timeout`` a
        command the engine has not started is withdrawn (halts excepted); one it
        is running is waited for a little longer.

        Raises:
            CommandFailedError: If it failed in the engine.
            CommandTimeoutError: If the engine did not finish it in time.
        """
        deadline = time.monotonic() + timeout
        withdrawn = False
        while True:
            with self._app.app_context():
                row = db.session.get(EngineCommand, command_id)
                if not withdrawn and time.monotonic() >= deadline:
                    withdrawn = True
                    if row.status == "pending" and row.kind != "halt":
                        row.status, row.completed_at = "expired", self._now()
                        db.session.commit()
                    deadline = time.monotonic() + RUNNING_GRACE
                status, result, error = row.status, row.result, row.error
            if status == "done":
                return result
            if status == "failed":
                kind, _, message = (error or "Error: failed").partition(": ")
                raise CommandFailedError(kind, message)
            if status == "expired" or (withdrawn and time.monotonic() >= deadline):
                raise CommandTimeoutError(f"the engine did not run command {command_id} in time"
                                          + ("" if status == "expired" else f" (still {status})"))
            time.sleep(interval)

    def call(self, kind: str, payload: dict[str, Any], timeout: float, requested_by: str | None = None) -> Any:
        return self.wait(self.send(kind, payload, requested_by), timeout)
