"""
NOCTURNA Trading System - Where a validation run's bars come from.

``flask validate`` and the dashboard's strategy lab load their data the same
way, so the two always agree:

* ``alpaca``: Alpaca's daily bars, as paper trading sees them (``market.py``);
* ``history``: the long-history proxies for SPY, IEF and BIL from French's
  Data Library and FRED, from 1970 (``history.py``).

With T-bills on (the default), the T-bill ETF BIL is loaded too: idle cash is
held in it and Sharpe ratios are measured in excess of its return.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from src.config import Settings

from ..data import DEFAULT_FEED_DELAY
from ..domain import AssetClass, Bar, Instrument
from ..lineup import CASH_SLEEVE
from . import history, market
from .stats import hurdle_returns

ALPACA, HISTORY = "alpaca", "history"
DATA_SOURCES = (ALPACA, HISTORY)


class DataError(RuntimeError):
    """The bars could not be had; the message says why, for the operator."""


@dataclass(frozen=True)
class MarketData:
    instruments: dict[str, Instrument]
    bars: dict[str, list[Bar]]
    data_label: str
    hurdle_label: str
    #: The files downloaded for the long history (empty for Alpaca)
    sources: list[history.Source] = field(default_factory=list)

    def hurdle(self) -> dict[date, float]:
        """T-bills' return per day (BIL or its proxy)."""
        return dict(hurdle_returns(self.bars[CASH_SLEEVE], DEFAULT_FEED_DELAY[AssetClass.EQUITY]))


def with_tbills(symbols: Sequence[tuple[str, AssetClass]]) -> tuple[tuple[str, AssetClass], ...]:
    known = {s for s, _ in symbols}
    return tuple(symbols) if CASH_SLEEVE in known else (*symbols, (CASH_SLEEVE, AssetClass.EQUITY))


def check_history_symbols(symbols: Sequence[tuple[str, AssetClass]]) -> None:
    """
    Raises:
        DataError: If the long history has no proxy for a symbol.
    """
    unknown = sorted({s for s, _ in symbols} - set(history.SYMBOLS))
    if unknown:
        raise DataError(f"the long history has proxies for {', '.join(history.SYMBOLS)} only, "
                        f"not {', '.join(unknown)}")


def load(settings: Settings, symbols: Sequence[tuple[str, AssetClass]], start: datetime, end: datetime,
         data: str = ALPACA) -> MarketData:
    """
    Bars for ``symbols`` from ``start`` (warm-up included) to ``end``.

    Raises:
        DataError: For an unknown source, a symbol the long history lacks, or long-history data that
            cannot be downloaded or does not cover the period.
        TradingNotConfiguredError: For Alpaca data without the Alpaca keys.
    """
    if data == HISTORY:
        check_history_symbols(symbols)
        try:
            instruments, bars, sources = history.load(start, end)
        except (ValueError, OSError) as e:
            raise DataError(f"long-history data: {e}") from e
        wanted = {s for s, _ in symbols}
        return MarketData({s: i for s, i in instruments.items() if s in wanted},
                          {s: b for s, b in bars.items() if s in wanted},
                          "long-history proxies for SPY, IEF and BIL from French's Data Library and FRED "
                          "(docs/STRATEGY_SHORTLIST.md, Fixed rules: trend-filter)",
                          f"T-bills (French's 1-month rate, as {CASH_SLEEVE})", sources)
    if data != ALPACA:
        raise DataError(f"unknown data source {data!r}; known: {', '.join(DATA_SOURCES)}")
    instruments, bars = market.load(settings, symbols, start, end)
    return MarketData(instruments, bars, "Alpaca daily bars", f"T-bills ({CASH_SLEEVE})")
