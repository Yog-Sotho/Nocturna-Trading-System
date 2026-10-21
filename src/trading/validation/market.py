"""
NOCTURNA Trading System - Market data for validation runs.

Instrument definitions (quantity increments, shortability) come from the
Alpaca paper account and bars from Alpaca market data, exactly as in paper
trading: split- and dividend-adjusted SIP bars for equities, which the free
plan serves from 2016 onwards, and Alpaca's crypto bars.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from src.config import Settings

from ..domain import DAILY, AssetClass, Bar, Instrument
from ..runtime import NOT_CONFIGURED_MESSAGE, TradingNotConfiguredError


def load(settings: Settings, symbols: Sequence[tuple[str, AssetClass]], start: datetime,
         end: datetime) -> tuple[dict[str, Instrument], dict[str, list[Bar]]]:
    """
    Raises:
        TradingNotConfiguredError: Without Alpaca credentials.
    """
    from alpaca.trading.client import TradingClient

    from ..alpaca import load_instruments
    from ..clock import SystemClock
    from ..data import AlpacaBarSource

    if settings.alpaca_api_key is None or settings.alpaca_secret_key is None:
        raise TradingNotConfiguredError(NOT_CONFIGURED_MESSAGE)
    key, secret = settings.alpaca_api_key.get_secret_value(), settings.alpaca_secret_key.get_secret_value()
    instruments = load_instruments(TradingClient(key, secret, paper=True), symbols)
    source = AlpacaBarSource(SystemClock(), api_key=key, secret_key=secret)
    bars = source.bars(list(instruments.values()), DAILY, start, end)
    return instruments, bars
