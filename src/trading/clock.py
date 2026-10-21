"""
NOCTURNA Trading System - Clocks.

All trading code asks a Clock for the time instead of calling datetime.now(),
so a backtest replays history with exactly the code that runs live.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, timezone-aware UTC."""
        ...


class SystemClock:
    """Wall-clock time."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimulatedClock:
    """Time that only moves forward when the backtest advances it."""

    def __init__(self, start: datetime) -> None:
        self._check(start)
        self._now = start

    @staticmethod
    def _check(value: datetime) -> None:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("SimulatedClock needs timezone-aware UTC datetimes")

    def now(self) -> datetime:
        return self._now

    def advance_to(self, value: datetime) -> None:
        """Move time forward. Raises ValueError if ``value`` is in the past."""
        self._check(value)
        if value < self._now:
            raise ValueError(f"clock cannot go back from {self._now} to {value}")
        self._now = value
