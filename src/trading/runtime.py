"""
NOCTURNA Trading System - The app's live trading runtime.

The API talks to the engine through a runtime, whichever process the engine
lives in (``ENGINE_MODE``):

* ``TradingRuntime`` (embedded): owns the engine service in this process. The
  service is built on first start, not at app start-up, so the API works (and
  can report the problem) when Alpaca is unreachable or not configured. The
  separate engine process (``flask engine run``) uses one too.
* ``RemoteRuntime`` (external): the engine runs as its own process; reads come
  from the views it publishes and commands go through the database channel.

Both offer ``configured``, ``views``, ``start``, ``stop``, ``restart``,
``resume``, ``submit_order``, ``cancel_order``, ``emergency_stop`` and
``autostart``.

What the engine trades is the trading profile (``profile.py``): the newest
saved one when the engine starts. A newer profile saved while the engine
runs takes effect at the next start, which ``restart`` does in one step.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Flask

from src.config import Settings

from .broker import Broker
from .channel import CommandFailedError, CommandTimeoutError, EngineChannel
from .domain import AssetClass
from .engine import EngineListener
from .operator import OperatorOrder, UnknownSymbolError
from .profile import Profile
from .risk import RiskDecision, RiskLimits
from .service import EngineService, ServiceNotRunningError, ServiceStatus
from .sizing import SizingPolicy
from .views import engine_views, order_view

logger = logging.getLogger(__name__)


#: Shown to operators when the engine cannot start for lack of credentials.
NOT_CONFIGURED_MESSAGE = "Set ALPACA_API_KEY and ALPACA_SECRET_KEY (Alpaca paper account) to trade"


class TradingNotConfiguredError(RuntimeError):
    """Raised when live trading needs Alpaca credentials that are not set."""


class EngineUnavailableError(RuntimeError):
    """The separate engine process did not answer."""


def risk_limits(settings: Settings) -> RiskLimits:
    return RiskLimits(
        max_position_weight=settings.risk_max_position_weight,
        max_order_weight=settings.risk_max_position_weight,
        max_asset_class_weight={AssetClass.CRYPTO: settings.risk_max_crypto_weight},
        max_order_notional=settings.risk_max_order_notional,
        max_orders_per_day=settings.risk_max_orders_per_day,
        max_drawdown=settings.risk_max_drawdown,
        flatten_on_kill=settings.risk_flatten_on_kill,
        daily_loss_limit=settings.risk_daily_loss_limit,
    )


def sizing_policy(settings: Settings) -> SizingPolicy:
    return SizingPolicy(
        target_volatility=settings.sizing_target_volatility or None,
        derisk_start=settings.sizing_derisk_start or None,
        rebalance_band=settings.sizing_rebalance_band,
    )


def _credentials(settings: Settings) -> tuple[str, str]:
    if settings.alpaca_api_key is None or settings.alpaca_secret_key is None:
        raise TradingNotConfiguredError(NOT_CONFIGURED_MESSAGE)
    return settings.alpaca_api_key.get_secret_value(), settings.alpaca_secret_key.get_secret_value()


def build_broker(settings: Settings) -> Broker:
    """The Alpaca paper broker alone, for the emergency path when no engine answers."""
    from alpaca.trading.client import TradingClient

    from .alpaca import AlpacaBroker, load_instruments
    from .clock import SystemClock
    from .lineup import LINEUP_SYMBOLS

    key, secret = _credentials(settings)
    client = TradingClient(key, secret, paper=True)
    return AlpacaBroker(api_key=key, secret_key=secret, instruments=load_instruments(client, LINEUP_SYMBOLS),
                        clock=SystemClock(), trading_client=client)


def held_over(held: list[str], profile: Profile) -> list[tuple[str, AssetClass]]:
    """
    Symbols the account holds that some strategy could trade but the profile's
    mix does not: the engine sells them (``Engine(close_out=...)``). Positions
    in symbols no strategy knows (bought by hand) are left alone.
    """
    from .lineup import CASH_SLEEVE
    from .profile import symbols
    from .validation.candidates import CANDIDATES

    known: dict[str, tuple[str, AssetClass]] = {}
    for spec in CANDIDATES.values():
        for symbol, asset_class in spec.symbols:
            # Alpaca reports crypto pairs as "BTC/USD" or, in older responses, "BTCUSD"
            known[symbol] = known[symbol.replace("/", "")] = (symbol, asset_class)
    traded = {s for s, _ in symbols(profile)} | {CASH_SLEEVE}
    found = {known[s] for s in held if s in known}
    return sorted(p for p in found if p[0] not in traded)


def build_service(app: Flask, settings: Settings, profile: Profile) -> EngineService:
    """Wire the paper-trading engine to Alpaca, trading ``profile``."""
    from .alerts import EngineAlerts
    from .alpaca import AlpacaBroker, TradeUpdateStream, load_instruments
    from .clock import SystemClock
    from .data import AlpacaBarSource
    from .lineup import CASH_SLEEVE
    from .persistence import EngineRecorder
    from .profile import allocations, effective_settings, symbols, warm_up_days

    key, secret = _credentials(settings)

    from alpaca.trading.client import TradingClient

    client = TradingClient(key, secret, paper=True)
    effective = effective_settings(settings, profile)
    positions: Any = client.get_all_positions()
    close_out = held_over([p.symbol for p in positions], profile)
    instruments = load_instruments(client, (*symbols(profile), *close_out))
    clock = SystemClock()
    broker = AlpacaBroker(api_key=key, secret_key=secret, instruments=instruments, clock=clock,
                          trading_client=client)
    recorder = EngineRecorder(app)
    listeners: list[EngineListener] = [recorder]
    if "alerts" in app.extensions:
        listeners.append(EngineAlerts(app.extensions["alerts"]))
    return EngineService(
        clock=clock,
        source=AlpacaBarSource(clock, api_key=key, secret_key=secret),
        broker=broker,
        allocations=allocations(profile, instruments),
        cash_sleeve=instruments[CASH_SLEEVE],
        close_out=[instruments[s] for s, _ in close_out],
        risk_limits=risk_limits(effective),
        sizing=sizing_policy(effective),
        warm_up=timedelta(days=warm_up_days(profile)),
        risk_state=recorder.load_risk_state,
        listeners=listeners,
        snapshot_recorder=recorder.record_snapshot,
        stream=TradeUpdateStream(api_key=key, secret_key=secret, on_update=broker.handle_trade_update),
    )


class TradingRuntime:
    def __init__(self, app: Flask, settings: Settings,
                 factory: Callable[[Flask, Settings, Profile], EngineService] = build_service,
                 profiles: Callable[[Flask], Profile] | None = None,
                 activate: Callable[[Flask, Profile, datetime], Profile] | None = None) -> None:
        """
        Args:
            profiles: The profile to trade at the next start (default: the newest saved).
            activate: Records that the engine started with a profile (default: stamps its row).
        """
        from . import profile as profiles_module

        self._app = app
        self._settings = settings
        self._factory = factory
        self._profiles = profiles or profiles_module.latest
        self._activate = activate or profiles_module.mark_activated
        self._lock = threading.Lock()
        self.service: EngineService | None = None
        #: The profile the current service was built with
        self.profile: Profile | None = None

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
            if self.service is not None and self.service.status().state in ("starting", "running"):
                raise RuntimeError("engine service is already running")
            profile = self._profiles(self._app)
            if self.service is not None and not self.service.running and (
                    self.profile is None or profile.id != self.profile.id):
                self.service = None  # a newer profile: the engine is rebuilt for it
            if self.service is None:
                self.service = self._factory(self._app, self._settings, profile)
                self.profile = profile
            if self.profile is not None:
                # Stamped before the engine thread exists: starting up, it writes to the database too
                self.profile = self._activate(self._app, self.profile, datetime.now(UTC))
            self.service.start()

    def stop(self) -> None:
        with self._lock:
            if self.service is not None:
                self.service.stop()

    def restart(self) -> None:
        """Stop and start again, taking the newest profile."""
        self.stop()
        self.start()

    def status(self) -> ServiceStatus | None:
        return self.service.status() if self.service is not None else None

    def views(self) -> dict[str, Any]:
        views = engine_views(self.configured, self.status())
        views["profile"] = self.profile.to_dict() if self.profile is not None and self.service is not None else None
        return views

    def _running(self) -> EngineService:
        service = self.service
        if service is None or not service.running:
            raise ServiceNotRunningError("the trading engine is not running")
        return service

    def resume(self) -> None:
        self._running().resume()

    def submit_order(self, order: OperatorOrder) -> tuple[RiskDecision, dict[str, Any] | None]:
        """
        Raises:
            ServiceNotRunningError: If the engine is not running.
            UnknownSymbolError: If the engine does not trade the symbol.
        """
        decision, placed = self._running().submit_operator_order(order)
        return decision, order_view(placed) if placed is not None else None

    def cancel_order(self, client_order_id: str) -> bool:
        return self._running().cancel_order(client_order_id)

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
        Halt trading. With a running engine: halt it, cancel its orders and,
        with ``flatten``, have the broker close every position. Without one: ask
        the broker directly to cancel every order and, with ``flatten``, close
        every position.
        """
        service = self.service
        if service is not None and service.running:
            closed = service.halt(reason, flatten=flatten)
            return {"engine": "halted", "close_orders": closed}
        if service is None:
            self.profile = self._profiles(self._app)
            service = self._factory(self._app, self._settings, self.profile)
            self.service = service
        if not flatten:
            return {"engine": "not running", "close_orders": [], "cancelled_orders": service.broker_cancel_all()}
        closed = service.broker_close_all()
        return {"engine": "not running", "close_orders": closed}


#: A published status older than this means the engine process is not running
STALE_AFTER = timedelta(seconds=30)
#: How long a request waits for the engine process to run its command
COMMAND_TIMEOUT = 15.0


class RemoteRuntime:
    """The API's view of an engine that runs as its own process (``flask engine run``)."""

    def __init__(self, app: Flask, settings: Settings, channel: EngineChannel | None = None,
                 broker_factory: Callable[[Settings], Broker] = build_broker,
                 stale_after: timedelta = STALE_AFTER, timeout: float = COMMAND_TIMEOUT) -> None:
        self._settings = settings
        self._channel = channel or EngineChannel(app)
        self._broker_factory = broker_factory
        self._stale_after = stale_after
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return self._settings.alpaca_api_key is not None and self._settings.alpaca_secret_key is not None

    def _published(self) -> dict[str, Any] | None:
        """The engine's document while its heartbeat is fresh, else None."""
        document, at = self._channel.status()
        if document is None or at is None or datetime.now(UTC) - at > self._stale_after:
            return None
        return document

    def views(self) -> dict[str, Any]:
        document = self._published()
        if document is None:
            return {"status": {"configured": self.configured, "state": "unreachable"},
                    "positions": None, "targets": None, "working": [], "profile": None}
        return {"profile": None, **document}

    def _call(self, kind: str, payload: dict[str, Any] | None = None) -> Any:
        if self._published() is None:
            raise EngineUnavailableError("the engine process is not running")
        try:
            return self._channel.call(kind, payload or {}, self._timeout)
        except CommandTimeoutError as e:
            raise EngineUnavailableError("the engine process did not answer") from e
        except CommandFailedError as e:
            raise _ERRORS.get(e.kind, RuntimeError)(str(e)) from e

    def start(self) -> None:
        self._call("start")

    def stop(self) -> None:
        self._call("stop")

    def restart(self) -> None:
        self._call("restart")

    def resume(self) -> None:
        self._call("resume")

    def submit_order(self, order: OperatorOrder) -> tuple[RiskDecision, dict[str, Any] | None]:
        result = self._call("submit_order", order.to_dict())
        return RiskDecision(result["allowed"], result["reason"]), result["order"]

    def cancel_order(self, client_order_id: str) -> bool:
        return bool(self._call("cancel_order", {"client_order_id": client_order_id})["requested"])

    def emergency_stop(self, reason: str, flatten: bool) -> dict[str, Any]:
        """
        Ask the engine process to halt; when it does not answer, go straight to
        the broker. The halt stays queued, so an engine that comes back halts.
        """
        if self._published() is not None:
            command = self._channel.send("halt", {"reason": reason, "flatten": flatten})
            try:
                result: dict[str, Any] = self._channel.wait(command, self._timeout)
                return result
            except CommandTimeoutError:
                logger.error("The engine process did not answer the emergency stop; using the broker directly")
        broker = self._broker_factory(self._settings)
        if flatten:
            return {"engine": "not answering", "close_orders": broker.close_all()}
        return {"engine": "not answering", "close_orders": [], "cancelled_orders": broker.cancel_all()}

    def autostart(self) -> None:
        """The engine process starts itself."""


#: Exceptions the engine process reports by name, raised again in the API
_ERRORS: dict[str, type[Exception]] = {
    "ServiceNotRunningError": ServiceNotRunningError,
    "TradingNotConfiguredError": TradingNotConfiguredError,
    "UnknownSymbolError": UnknownSymbolError,
    "RuntimeError": RuntimeError,
}
