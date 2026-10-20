"""
NOCTURNA Trading System - Trading safety guard.

Live trading is hard-disabled until the rebuilt risk engine, broker
reconciliation and validation pipeline ship. Only Alpaca paper trading
(or the local simulator) may run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

PAPER_TRADING_MODE = "PAPER"
ALPACA_PAPER_HOST = "paper-api.alpaca.markets"


class LiveTradingDisabledError(RuntimeError):
    """Raised when the configuration would route orders to a live account."""


def assert_paper_only(env: Mapping[str, str] | None = None) -> None:
    """
    Refuse any configuration that could place live orders.

    Args:
        env: Environment mapping to check; defaults to ``os.environ``.

    Raises:
        LiveTradingDisabledError: If TRADING_MODE is not PAPER or
            ALPACA_BASE_URL points anywhere other than the Alpaca paper API.
    """
    env = os.environ if env is None else env

    mode = env.get("TRADING_MODE", PAPER_TRADING_MODE).strip().upper()
    if mode != PAPER_TRADING_MODE:
        raise LiveTradingDisabledError(
            f"TRADING_MODE={mode!r} is not allowed: live trading is disabled in this build. "
            f"Set TRADING_MODE={PAPER_TRADING_MODE}."
        )

    base_url = env.get("ALPACA_BASE_URL", "").strip()
    if base_url and urlparse(base_url).hostname != ALPACA_PAPER_HOST:
        raise LiveTradingDisabledError(
            f"ALPACA_BASE_URL={base_url!r} is not the Alpaca paper endpoint: live trading is "
            f"disabled in this build. Use https://{ALPACA_PAPER_HOST}."
        )
