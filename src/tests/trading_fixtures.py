"""
Synthetic market data for trading-core tests: deterministic random walks laid
on real calendars (NYSE sessions for equities, every UTC day for crypto).
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np

from src.trading.calendar import TradingCalendar
from src.trading.domain import DAILY, NEW_YORK, AssetClass, Bar, Instrument

SPY = Instrument("SPY", AssetClass.EQUITY, qty_increment=Decimal("1"), min_qty=Decimal("1"), shortable=True)
IEF = Instrument("IEF", AssetClass.EQUITY, qty_increment=Decimal("1"), min_qty=Decimal("1"), shortable=True)
BTC = Instrument("BTC/USD", AssetClass.CRYPTO, qty_increment=Decimal("0.0001"), min_qty=Decimal("0.0001"))


def daily_bars(instrument: Instrument, start: date, end: date, first_price: float = 100.0,
               seed: int = 7, drift: float = 0.0003, vol: float = 0.01, volume: float = 1e6) -> list[Bar]:
    """One bar per session in [start, end]."""
    sessions = TradingCalendar(instrument.asset_class).sessions(start, end)
    rng = np.random.default_rng(seed)
    bars, close = [], first_price
    for day in sessions:
        if instrument.asset_class is AssetClass.EQUITY:
            bar_start = datetime(day.year, day.month, day.day, tzinfo=NEW_YORK).astimezone(UTC)
        else:
            bar_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        open_ = close * (1 + rng.normal(0, vol / 4))
        close = open_ * (1 + rng.normal(drift, vol))
        high = max(open_, close) * (1 + abs(rng.normal(0, vol / 2)))
        low = min(open_, close) * (1 - abs(rng.normal(0, vol / 2)))
        bars.append(Bar(instrument.symbol, bar_start, DAILY.bar_end(bar_start, instrument.asset_class),
                        round(open_, 4), round(high, 4), round(low, 4), round(close, 4), volume))
    return bars


def ny(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """A New York wall-clock time as UTC."""
    return datetime(year, month, day, hour, minute, tzinfo=NEW_YORK).astimezone(UTC)


def bar(instrument: Instrument, start: datetime, o: float, h: float, low: float, c: float,
        volume: float = 1e6) -> Bar:
    return Bar(instrument.symbol, start, DAILY.bar_end(start, instrument.asset_class), o, h, low, c, volume)


ONE_MINUTE = timedelta(minutes=1)
