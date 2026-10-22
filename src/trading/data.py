"""
NOCTURNA Trading System - Market data.

Every source answers the same question: which *closed* bars start in
[start, end)? A bar is closed once ``bar.end + delay <= clock.now()``, where the
delay is how long the feed needs to finalise it. Backtests read history
through an in-memory source on a simulated clock, live trading polls Alpaca on
the wall clock, and strategies cannot tell the difference or see the future.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Protocol

from .clock import Clock
from .domain import AssetClass, Bar, Instrument, Timeframe

logger = logging.getLogger(__name__)

#: How long after a bar ends its data is final, per asset class. Free Alpaca
#: accounts may only query SIP (all-exchange) equity data that is at least 15
#: minutes old; crypto bars are final almost at once.
DEFAULT_FEED_DELAY: dict[AssetClass, timedelta] = {
    AssetClass.EQUITY: timedelta(minutes=15),
    AssetClass.CRYPTO: timedelta(minutes=1),
}

#: Vendor bars whose high/low miss the open/close by at most this fraction (a
#: rounding artefact of split/dividend adjustment) are repaired, not rejected.
OHLC_REPAIR_TOLERANCE = 1e-4


class DataQualityError(ValueError):
    """Raised when vendor data is inconsistent beyond rounding."""


def is_closed(bar: Bar, now: datetime, delay: timedelta) -> bool:
    return bar.end + delay <= now


class BarSource(Protocol):
    def bars(
        self, instruments: Sequence[Instrument], timeframe: Timeframe, start: datetime, end: datetime
    ) -> dict[str, list[Bar]]:
        """Closed bars starting in [start, end) per symbol, oldest first."""
        ...


def _select(bars: Iterable[Bar], start: datetime, end: datetime, now: datetime, delay: timedelta) -> list[Bar]:
    return sorted((b for b in bars if start <= b.start < end and is_closed(b, now, delay)),
                  key=lambda b: b.start)


class InMemoryBarSource:
    """
    Serves a fixed set of bars, hiding those not yet closed at ``clock.now()``.

    Backtests use it on a SimulatedClock, which makes look-ahead impossible by
    construction: a strategy can only ever receive bars that had closed.
    """

    def __init__(self, bars: Mapping[str, Sequence[Bar]], clock: Clock,
                 delays: Mapping[AssetClass, timedelta] | None = None) -> None:
        self._bars = {symbol: list(series) for symbol, series in bars.items()}
        self._clock = clock
        self._delays = dict(DEFAULT_FEED_DELAY if delays is None else delays)

    def bars(self, instruments: Sequence[Instrument], timeframe: Timeframe, start: datetime,
             end: datetime) -> dict[str, list[Bar]]:
        now = self._clock.now()
        return {
            i.symbol: _select(self._bars.get(i.symbol, ()), start, end, now, self._delays[i.asset_class])
            for i in instruments
        }


def to_bar(symbol: str, start: datetime, timeframe: Timeframe, asset_class: AssetClass,
           o: float, h: float, low: float, c: float, v: float) -> Bar:
    """
    Build a Bar from vendor values, repairing high/low rounding artefacts.

    Raises:
        DataQualityError: If high/low are inconsistent beyond OHLC_REPAIR_TOLERANCE.
    """
    top, bottom = max(o, c), min(o, c)
    if h < top:
        if (top - h) > top * OHLC_REPAIR_TOLERANCE:
            raise DataQualityError(f"{symbol} {start}: high {h} below open/close {top}")
        h = top
    if low > bottom:
        if (low - bottom) > bottom * OHLC_REPAIR_TOLERANCE:
            raise DataQualityError(f"{symbol} {start}: low {low} above open/close {bottom}")
        low = bottom
    return Bar(symbol, start, timeframe.bar_end(start, asset_class), o, h, low, c, v)


class AlpacaBarSource:
    """
    Historical bars from Alpaca's market data API (alpaca-py).

    Equities use the SIP feed (all US exchanges) with split and dividend
    adjustment (``raw=True``: the prices as traded, for comparing fills);
    requests never ask for data newer than the feed delay, which keeps them
    within the free plan. Crypto data needs no API key.
    """

    def __init__(self, clock: Clock, *, api_key: str | None = None, secret_key: str | None = None,
                 delays: Mapping[AssetClass, timedelta] | None = None,
                 stock_client: Any = None, crypto_client: Any = None, raw: bool = False) -> None:
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        self._clock = clock
        self._raw = raw
        self._delays = dict(DEFAULT_FEED_DELAY if delays is None else delays)
        self._stocks = stock_client
        if self._stocks is None and api_key and secret_key:
            self._stocks = StockHistoricalDataClient(api_key, secret_key)
        self._crypto = crypto_client or CryptoHistoricalDataClient(api_key, secret_key)

    @staticmethod
    def _alpaca_timeframe(timeframe: Timeframe) -> Any:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit = {"minute": TimeFrameUnit.Minute, "hour": TimeFrameUnit.Hour, "day": TimeFrameUnit.Day}
        return TimeFrame(timeframe.amount, unit[timeframe.unit])

    def bars(self, instruments: Sequence[Instrument], timeframe: Timeframe, start: datetime,
             end: datetime) -> dict[str, list[Bar]]:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

        now = self._clock.now()
        result: dict[str, list[Bar]] = {i.symbol: [] for i in instruments}
        for asset_class in AssetClass:
            group = [i for i in instruments if i.asset_class is asset_class]
            if not group:
                continue
            delay = self._delays[asset_class]
            # Never request data the feed has not finalised (and, for SIP, may not serve)
            request_end = min(end, now - delay)
            if request_end <= start:
                continue
            symbols = [i.symbol for i in group]
            common = dict(symbol_or_symbols=symbols, timeframe=self._alpaca_timeframe(timeframe),
                          start=start, end=request_end)
            if asset_class is AssetClass.EQUITY:
                if self._stocks is None:
                    raise RuntimeError("equity data needs ALPACA_API_KEY and ALPACA_SECRET_KEY")
                adjustment = Adjustment.RAW if self._raw else Adjustment.ALL
                barset = self._stocks.get_stock_bars(
                    StockBarsRequest(**common, adjustment=adjustment, feed=DataFeed.SIP))
            else:
                barset = self._crypto.get_crypto_bars(CryptoBarsRequest(**common))
            for symbol in symbols:
                raw = barset.data.get(symbol, [])
                converted = [to_bar(symbol, b.timestamp, timeframe, asset_class, float(b.open), float(b.high),
                                    float(b.low), float(b.close), float(b.volume)) for b in raw]
                # Alpaca's end is inclusive and may include the bar still forming
                result[symbol] = _select(converted, start, end, now, delay)
        return result
