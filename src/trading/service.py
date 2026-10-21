"""
NOCTURNA Trading System - Live engine service.

Runs one Engine for paper trading on its own thread. Nothing else touches the
engine: HTTP handlers and the broker's trade-update stream put commands and
updates on a queue that the engine thread executes in order, and read state
only through the snapshot it publishes. The loop:

* on start, builds the ledger from the broker account, adopts open orders,
  loads warm-up history and acts on the newest bars,
* wakes when the next bar becomes final, polls the bar source and feeds new
  bars to the engine,
* applies broker updates as they arrive,
* records an equity snapshot after each poll that brought new bars,
* reconciles with the broker whenever no order is working and the stream has
  been quiet for a while.

``step()`` performs one pass of the loop, so tests can drive the service on a
simulated clock without threads.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .broker import Broker
from .clock import Clock
from .data import BarSource
from .domain import DAILY, Bar, OrderRequest, OrderUpdate, Timeframe
from .engine import Allocation, Engine, EngineListener, EngineSnapshot
from .feed import PollingFeed
from .ledger import Ledger
from .reconcile import ReconciliationReport, reconcile
from .risk import RiskDecision, RiskGate, RiskLimits

logger = logging.getLogger(__name__)


class ServiceNotRunningError(RuntimeError):
    """Raised when a command needs the engine but the service is not running."""


@dataclass(frozen=True)
class ServiceStatus:
    state: str  # stopped, starting, running, failed
    started_at: datetime | None
    last_poll: datetime | None
    next_poll: datetime | None
    last_error: str
    snapshot: EngineSnapshot | None
    last_reconciliation: ReconciliationReport | None


class EngineService:
    def __init__(self, *, clock: Clock, source: BarSource, broker: Broker, allocations: Sequence[Allocation],
                 risk_limits: RiskLimits | None = None, listeners: Sequence[EngineListener] = (),
                 snapshot_recorder: Callable[[EngineSnapshot], None] | None = None,
                 stream: Any = None, timeframe: Timeframe = DAILY, warm_up: timedelta = timedelta(days=550),
                 poll_latency: timedelta = timedelta(seconds=5), retry_delay: timedelta = timedelta(minutes=1),
                 quiet_period: timedelta = timedelta(minutes=2)) -> None:
        """
        Args:
            stream: Optional object with start()/stop() that delivers broker updates
                (e.g. TradeUpdateStream); started and stopped with the service.
            warm_up: How much history to load before trading.
            quiet_period: How long the broker must have been silent before reconciling,
                so a position change is never seen before the fill that caused it.
        """
        self.clock = clock
        self._source = source
        self._broker = broker
        self._allocations = tuple(allocations)
        self._risk_limits = risk_limits or RiskLimits()
        self._listeners = list(listeners)
        self._snapshot_recorder = snapshot_recorder
        self._stream = stream
        self._timeframe = timeframe
        self._warm_up = warm_up
        self._poll_latency = poll_latency
        self._retry_delay = retry_delay
        self._quiet_period = quiet_period

        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.engine: Engine | None = None
        self._feed: PollingFeed | None = None
        self._state = "stopped"
        self._started_at: datetime | None = None
        self._last_poll: datetime | None = None
        self._next_poll: datetime | None = None
        self._last_update: datetime | None = None
        self._last_error = ""
        self._snapshot: EngineSnapshot | None = None
        self._last_reconciliation: ReconciliationReport | None = None

    # -- lifecycle (any thread) ---------------------------------------------

    def start(self) -> None:
        """Start the engine thread. Raises RuntimeError if it is already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("engine service is already running")
            self._stopping.clear()
            self._state = "starting"
            self._thread = threading.Thread(target=self._run, name="trading-engine", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        self._stopping.set()
        self._queue.put(None)  # wake the loop
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        if self._stream is not None:
            self._stream.stop()
        with self._lock:
            if self._state != "failed":
                self._state = "stopped"

    @property
    def running(self) -> bool:
        return self._state == "running"

    def status(self) -> ServiceStatus:
        with self._lock:
            return ServiceStatus(self._state, self._started_at, self._last_poll, self._next_poll,
                                 self._last_error, self._snapshot, self._last_reconciliation)

    # -- commands (any thread) ----------------------------------------------

    def halt(self, reason: str, flatten: bool = False) -> list[OrderRequest]:
        return self._call(lambda engine: engine.halt(reason, flatten=flatten))

    def resume(self) -> None:
        self._call(lambda engine: engine.resume())

    def submit_order(self, request: OrderRequest) -> RiskDecision:
        return self._call(lambda engine: engine.submit_order(request))

    def _call(self, command: Callable[[Engine], Any], timeout: float = 30.0) -> Any:
        if self.engine is None or self._state != "running":
            raise ServiceNotRunningError("the trading engine is not running")
        if threading.current_thread() is self._thread:
            return self._execute(command)
        future: Future = Future()
        self._queue.put(("call", command, future))
        return future.result(timeout)

    # -- engine thread ------------------------------------------------------

    def _enqueue_update(self, update: OrderUpdate) -> None:
        self._queue.put(("update", update))

    def prepare(self) -> None:
        """Build the engine from the broker account and load history."""
        now = self.clock.now()
        ledger = Ledger.from_account(self._broker.cash(), self._broker.cost_basis())
        self.engine = Engine(clock=self.clock, broker=self._broker, ledger=ledger, allocations=self._allocations,
                             risk=RiskGate(self._risk_limits), listeners=self._listeners)
        # Updates (from this thread or the stream's) are queued and applied in order
        self._broker.set_listener(self._enqueue_update)
        if self._stream is not None:
            self._stream.start()
        self._last_reconciliation = reconcile(self.engine, self._broker)

        instruments = list(self.engine.instruments.values())
        history = self._source.bars(instruments, self._timeframe, now - self._warm_up, now)
        # Hold back the newest bars so the engine acts on them now: a restart must
        # not wait a full bar to trade. Orders already working or filled before the
        # restart are part of the exposure, so nothing is traded twice.
        newest: dict[str, Bar] = {s: bars[-1] for s, bars in history.items() if bars}
        self.engine.warm_up({s: bars[:-1] for s, bars in history.items()})
        self.engine.on_bars(newest)
        self._feed = PollingFeed(self._source, instruments, self._timeframe, now - self._warm_up,
                                 seen=self.engine.latest_bar_starts())
        self._next_poll = self._feed.next_poll(now) + self._poll_latency
        with self._lock:
            self._started_at = now
            self._state = "running"
        self._drain()
        self._publish()

    def _run(self) -> None:
        try:
            self.prepare()
        except Exception as e:
            logger.exception("Engine service failed to start")
            with self._lock:
                self._state = "failed"
                self._last_error = f"start-up failed: {e}"
            return
        while not self._stopping.is_set():
            wait = (self._next_poll - self.clock.now()).total_seconds() if self._next_poll else 30.0
            try:
                item = self._queue.get(timeout=max(0.0, min(wait, 30.0)))
            except queue.Empty:
                item = None
            if item is not None:
                self._handle(item)
            self.step()

    def step(self) -> None:
        """One pass: apply queued items, poll if due, reconcile if idle, publish."""
        if self.engine is None:
            raise ServiceNotRunningError("call prepare() first")
        self._drain()
        now = self.clock.now()
        if self._next_poll is not None and now >= self._next_poll:
            self._poll(now)
        self._drain()
        self._maybe_reconcile()
        self._publish()

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self._handle(item)

    def _handle(self, item: tuple) -> None:
        if item[0] == "update":
            self._last_update = self.clock.now()
            self.engine.on_order_update(item[1])
        elif item[0] == "call":
            _, command, future = item
            try:
                future.set_result(self._execute(command))
            except Exception as e:
                future.set_exception(e)

    def _execute(self, command: Callable[[Engine], Any]) -> Any:
        result = command(self.engine)
        self._drain()  # apply the broker's immediate answers before replying
        self._publish()
        return result

    def _poll(self, now: datetime) -> None:
        try:
            groups = self._feed.poll(now)
            for _, group in groups:
                self.engine.on_bars(group)
            self._drain()
            if groups and self._snapshot_recorder is not None:
                self._snapshot_recorder(self.engine.snapshot())
            self._last_poll = now
            self._next_poll = self._feed.next_poll(now) + self._poll_latency
            self._last_error = ""
        except Exception as e:
            logger.exception("Polling market data failed")
            self._last_error = f"market data: {e}"
            self._next_poll = now + self._retry_delay

    def _maybe_reconcile(self) -> None:
        engine = self.engine
        if any(o.status.is_open for o in engine.orders.values()):
            return
        if self._last_update is not None and self.clock.now() - self._last_update < self._quiet_period:
            return
        last = self._last_reconciliation
        activity = [t for t in (self._last_poll, self._last_update) if t is not None]
        if last is not None and not any(t > last.timestamp for t in activity):
            return  # nothing has happened since the last one
        try:
            self._last_reconciliation = reconcile(engine, self._broker)
        except Exception as e:
            logger.exception("Reconciliation failed")
            self._last_error = f"reconciliation: {e}"

    def _publish(self) -> None:
        snapshot = self.engine.snapshot()
        with self._lock:
            self._snapshot = snapshot
