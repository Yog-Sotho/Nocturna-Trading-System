"""
NOCTURNA Trading System - The app's live trading runtime.

Owns the engine service for one process. The service is built on first start,
not at app start-up, so the API works (and can report the problem) when
Alpaca is unreachable or not configured.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from flask import Flask

from src.config import Settings

from .service import EngineService, ServiceStatus

logger = logging.getLogger(__name__)


#: Shown to operators when the engine cannot start for lack of credentials.
NOT_CONFIGURED_MESSAGE = "Set ALPACA_API_KEY and ALPACA_SECRET_KEY (Alpaca paper account) to trade"


class TradingNotConfiguredError(RuntimeError):
    """Raised when live trading needs Alpaca credentials that are not set."""


def build_service(app: Flask, settings: Settings) -> EngineService:
    """Wire the paper-trading engine to Alpaca."""
    from .alpaca import AlpacaBroker, TradeUpdateStream, load_instruments
    from .clock import SystemClock
    from .data import AlpacaBarSource
    from .lineup import LINEUP_SYMBOLS, build_lineup
    from .persistence import EngineRecorder

    if settings.alpaca_api_key is None or settings.alpaca_secret_key is None:
        raise TradingNotConfiguredError(NOT_CONFIGURED_MESSAGE)
    key = settings.alpaca_api_key.get_secret_value()
    secret = settings.alpaca_secret_key.get_secret_value()

    from alpaca.trading.client import TradingClient

    client = TradingClient(key, secret, paper=True)
    instruments = load_instruments(client, LINEUP_SYMBOLS)
    clock = SystemClock()
    broker = AlpacaBroker(api_key=key, secret_key=secret, instruments=instruments, clock=clock,
                          trading_client=client)
    recorder = EngineRecorder(app)
    return EngineService(
        clock=clock,
        source=AlpacaBarSource(clock, api_key=key, secret_key=secret),
        broker=broker,
        allocations=build_lineup(instruments),
        listeners=[recorder],
        snapshot_recorder=recorder.record_snapshot,
        stream=TradeUpdateStream(api_key=key, secret_key=secret, on_update=broker.handle_trade_update),
    )


class TradingRuntime:
    def __init__(self, app: Flask, settings: Settings,
                 factory: Callable[[Flask, Settings], EngineService] = build_service) -> None:
        self._app = app
        self._settings = settings
        self._factory = factory
        self._lock = threading.Lock()
        self.service: EngineService | None = None

    @property
    def configured(self) -> bool:
        return self._settings.alpaca_api_key is not None and self._settings.alpaca_secret_key is not None

    def start(self) -> None:
        """
        Build the service if needed and start it.

        Raises:
            TradingNotConfiguredError: Without Alpaca credentials.
            RuntimeError: If it is already running.
        """
        with self._lock:
            if self.service is None:
                self.service = self._factory(self._app, self._settings)
            self.service.start()

    def stop(self) -> None:
        with self._lock:
            if self.service is not None:
                self.service.stop()

    def status(self) -> ServiceStatus | None:
        return self.service.status() if self.service is not None else None

    def autostart(self) -> None:
        """Start at process start-up when ENGINE_AUTOSTART is set; failures are logged, not raised."""
        if not self._settings.engine_autostart:
            return
        try:
            self.start()
        except Exception:
            logger.exception("Engine autostart failed")

    def emergency_stop(self, reason: str, flatten: bool) -> dict[str, Any]:
        """
        Halt trading. With a running engine: cancel its orders and optionally
        flatten through the risk gate. Without one: ask the broker directly to
        cancel every order and, with ``flatten``, close every position.
        """
        service = self.service
        if service is not None and service.running:
            sent = service.halt(reason, flatten=flatten)
            return {"engine": "halted", "close_orders": [r.client_order_id for r in sent]}
        if not flatten:
            return {"engine": "not running", "close_orders": []}
        if service is None:
            service = self._factory(self._app, self._settings)
            self.service = service
        closed = service.broker_close_all()
        return {"engine": "not running", "close_orders": closed}
