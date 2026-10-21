"""
NOCTURNA Trading System - Polling bar feed.

Live trading does not need a websocket for daily strategies: after a bar's
close plus the feed delay, the same historical-bars request a backtest makes
returns it. The feed remembers the newest bar seen per symbol and hands the
engine only bars it has not seen, grouped by the time they became final.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from .data import DEFAULT_FEED_DELAY, BarSource
from .domain import AssetClass, Bar, Instrument, Timeframe


class PollingFeed:
    def __init__(self, source: BarSource, instruments: Sequence[Instrument], timeframe: Timeframe,
                 start: datetime, delays: Mapping[AssetClass, timedelta] | None = None,
                 seen: Mapping[str, datetime] | None = None) -> None:
        """
        Args:
            start: Earliest bar start to deliver.
            seen: Start of the newest bar already held per symbol (e.g. the engine's
                warm-up history); delivery resumes right after it.
        """
        self._source = source
        self._instruments = {i.symbol: i for i in instruments}
        self._timeframe = timeframe
        self._delays = dict(DEFAULT_FEED_DELAY if delays is None else delays)
        seen = seen or {}
        self._next_start: dict[str, datetime] = {
            s: max(start, seen[s] + timedelta(microseconds=1)) if s in seen else start
            for s in self._instruments}

    def next_poll(self, now: datetime) -> datetime:
        """
        When the next bar of any instrument becomes final. Polling then, not on a
        fixed cadence, lets orders reach the broker before the next bar trades.
        """
        return min(
            self._timeframe.next_boundary(now - self._delays[i.asset_class], i.asset_class)
            + self._delays[i.asset_class]
            for i in self._instruments.values()
        )

    def poll(self, now: datetime) -> list[tuple[datetime, dict[str, Bar]]]:
        """
        New closed bars since the previous poll, as (became_final_at, {symbol: bar})
        groups in time order, at most one bar per symbol per group.
        """
        groups: dict[datetime, dict[str, Bar]] = defaultdict(dict)
        earliest = min(self._next_start.values())
        fetched = self._source.bars(list(self._instruments.values()), self._timeframe, earliest, now)
        for symbol, bars in fetched.items():
            delay = self._delays[self._instruments[symbol].asset_class]
            for bar in bars:
                if bar.start < self._next_start[symbol]:
                    continue
                groups[bar.end + delay][symbol] = bar
                self._next_start[symbol] = bar.start + timedelta(microseconds=1)
        return sorted(groups.items())
