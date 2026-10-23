"""
NOCTURNA Trading System - Trading calendars.

US equity sessions come from the NYSE calendar in ``exchange_calendars``
(holidays and early closes included); crypto trades every day.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import exchange_calendars as xcals

from .domain import AssetClass

#: Earliest session the calendar is built for; covers every backtest we run, with a year to
#: spare before the long-history proxies (``validation/history.py``) start in 1970. From 1970
#: the calendar's sessions match CRSP's trading days exactly; before, it misses holidays.
CALENDAR_START = "1969-01-01"


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start=CALENDAR_START)


class TradingCalendar:
    """Sessions for one asset class."""

    def __init__(self, asset_class: AssetClass) -> None:
        self.asset_class = asset_class

    def sessions(self, start: date, end: date) -> list[date]:
        """Trading days in [start, end], inclusive."""
        if start > end:
            return []
        if self.asset_class is AssetClass.CRYPTO:
            return [start + timedelta(days=i) for i in range((end - start).days + 1)]
        return [ts.date() for ts in _nyse().sessions_in_range(start.isoformat(), end.isoformat())]

    def is_session(self, day: date) -> bool:
        if self.asset_class is AssetClass.CRYPTO:
            return True
        return bool(_nyse().is_session(day.isoformat()))

    def next_session(self, day: date) -> date:
        """The first session strictly after ``day``."""
        if self.asset_class is AssetClass.CRYPTO:
            return day + timedelta(days=1)
        session: date = _nyse().date_to_session((day + timedelta(days=1)).isoformat(), direction="next").date()
        return session

    def previous_session(self, day: date) -> date:
        """The last session strictly before ``day``."""
        if self.asset_class is AssetClass.CRYPTO:
            return day - timedelta(days=1)
        session: date = _nyse().date_to_session((day - timedelta(days=1)).isoformat(), direction="previous").date()
        return session

    def session_close(self, day: date) -> datetime:
        """When the session on ``day`` ends (UTC); early closes included."""
        if self.asset_class is AssetClass.CRYPTO:
            return datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
        close: datetime = _nyse().session_close(day.isoformat()).to_pydatetime()
        return close

    def session_open(self, day: date) -> datetime:
        if self.asset_class is AssetClass.CRYPTO:
            return datetime(day.year, day.month, day.day, tzinfo=UTC)
        opening: datetime = _nyse().session_open(day.isoformat()).to_pydatetime()
        return opening
