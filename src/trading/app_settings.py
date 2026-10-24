"""
NOCTURNA Trading System - Settings changed on the dashboard, shared by every process.

The environment (``.env``) holds what needs a restart and every secret; this
table holds the few choices the operator makes on the dashboard that the
engine process must see too, such as which alerts are sent. Readers cache a
value for a few seconds, so a change reaches every process within ``TTL``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from flask import Flask

from .alerts import AlertChoices

logger = logging.getLogger(__name__)

ALERTS = "alerts"
#: Seconds a value read from the database is reused
TTL = 10.0


def read(app: Flask, key: str) -> Any | None:
    from src.models.trading import AppSetting
    from src.models.user import db

    with app.app_context():
        record = db.session.get(AppSetting, key)
        return None if record is None else record.value


def write(app: Flask, key: str, value: Any, updated_by: str | None, now: datetime) -> None:
    from src.models.trading import AppSetting
    from src.models.user import db

    with app.app_context():
        record = db.session.get(AppSetting, key)
        if record is None:
            db.session.add(AppSetting(key=key, value=value, updated_by=updated_by, updated_at=now))
        else:
            record.value, record.updated_by, record.updated_at = value, updated_by, now
        db.session.commit()


class Cached:
    """``load()``, reused for ``ttl`` seconds; the last good value (or ``fallback``) while it fails."""

    def __init__(self, load: Callable[[], Any], fallback: Any, ttl: float = TTL,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._load = load
        self._value = fallback
        self._ttl = ttl
        self._clock = clock
        self._at: float | None = None
        self._lock = threading.Lock()

    def __call__(self) -> Any:
        with self._lock:
            now = self._clock()
            if self._at is None or now - self._at >= self._ttl:
                try:
                    self._value = self._load()
                except Exception:
                    logger.exception("Reading a dashboard setting failed; using the last value")
                self._at = now
            return self._value


def alert_choices(app: Flask, ttl: float = TTL) -> Callable[[], AlertChoices]:
    """The operator's alert choices as every process reads them."""
    cached = Cached(lambda: AlertChoices.from_dict(read(app, ALERTS)), AlertChoices(), ttl)

    def current() -> AlertChoices:
        choices: AlertChoices = cached()
        return choices

    return current
