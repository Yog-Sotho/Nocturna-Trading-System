"""
NOCTURNA Trading System - Trading engine.

The engine is the same object in backtest, paper and live trading. Each time
new closed bars arrive it:

1. records them (bars it has already seen are ignored, so polling can overlap),
2. lets the broker execute working orders against them (the simulator does;
   a real broker reports fills on its own),
3. asks each strategy whose instruments changed for target weights,
4. nets the targets of all strategies into one set of account-level orders,
   sized to equity and cash, sells first,
5. sends every order through the risk gate to the broker.

Broker reports come back through ``on_order_update`` and are booked in the
ledger. Nothing here reads the wall clock: time comes from the injected Clock.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal

from .broker import Broker
from .calendar import TradingCalendar
from .clock import Clock
from .domain import (
    NEW_YORK,
    AssetClass,
    Bar,
    Fill,
    Instrument,
    InvalidTransitionError,
    Order,
    OrderRequest,
    OrderType,
    OrderUpdate,
    Side,
    TimeInForce,
)
from .ledger import Ledger
from .risk import RiskDecision, RiskGate
from .strategy import Strategy, StrategyContext

logger = logging.getLogger(__name__)

ZERO = Decimal("0")
WEIGHT_SCALE = Decimal("1e-6")

#: Alpaca accepts opening-auction (OPG) orders from 19:00 ET the evening before
#: until 09:28 ET; outside that window market orders are sent as DAY orders.
OPG_WINDOW_START = time(19, 0)
OPG_CUTOFF = timedelta(minutes=2)


@dataclass(frozen=True)
class Allocation:
    """A strategy and the fraction of account equity it manages."""

    strategy: Strategy
    weight: Decimal


@dataclass(frozen=True)
class EngineEvent:
    """
    Something worth recording. ``kind`` is one of: order, fill, rejected,
    strategy_error, broker_error, unknown_fill, reconciled, halt, resume.
    """

    kind: str
    timestamp: datetime
    order: Order | None = None
    fill: Fill | None = None
    detail: str = ""


EngineListener = Callable[[EngineEvent], None]


@dataclass(frozen=True)
class PositionView:
    symbol: str
    qty: Decimal
    avg_price: Decimal
    price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True)
class EngineSnapshot:
    timestamp: datetime
    equity: Decimal
    cash: Decimal
    realized_pnl: Decimal
    fees: Decimal
    positions: tuple[PositionView, ...]
    open_orders: tuple[Order, ...]
    target_weights: dict[str, Decimal]
    halted: bool
    halt_reason: str
    last_bar_end: dict[str, datetime]


def _decimal(value: float) -> Decimal:
    return Decimal(repr(value))


def _slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", symbol)[:12]


class Engine:
    def __init__(self, *, clock: Clock, broker: Broker, ledger: Ledger, allocations: Sequence[Allocation],
                 risk: RiskGate | None = None, min_trade_fraction: Decimal = Decimal("0.002"),
                 cash_buffer: Decimal = Decimal("0.005"), listeners: Sequence[EngineListener] = ()) -> None:
        """
        Args:
            min_trade_fraction: Skip adjustments smaller than this fraction of equity
                (avoids churning on rounding), except when closing a position.
            cash_buffer: Fraction of equity kept unspent to absorb slippage and fees.

        Raises:
            ValueError: For invalid allocations or conflicting instrument definitions.
        """
        if not allocations:
            raise ValueError("at least one allocation is required")
        if any(a.weight <= 0 for a in allocations) or sum(a.weight for a in allocations) > 1:
            raise ValueError("allocation weights must be positive and sum to at most 1")
        names = [a.strategy.name for a in allocations]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique")

        self.clock = clock
        self.broker = broker
        self.ledger = ledger
        self.allocations = tuple(allocations)
        self.risk = risk or RiskGate()
        self.min_trade_fraction = min_trade_fraction
        self.cash_buffer = cash_buffer
        self._listeners = list(listeners)

        self.instruments: dict[str, Instrument] = {}
        lookbacks: dict[str, int] = {}
        for allocation in self.allocations:
            for instrument in allocation.strategy.instruments:
                known = self.instruments.setdefault(instrument.symbol, instrument)
                if known != instrument:
                    raise ValueError(f"conflicting definitions for {instrument.symbol}")
                lookbacks[instrument.symbol] = max(lookbacks.get(instrument.symbol, 1),
                                                   allocation.strategy.lookback)
        self._history: dict[str, deque[Bar]] = {s: deque(maxlen=n) for s, n in lookbacks.items()}
        self._targets: dict[str, dict[str, Decimal]] = {}
        self.orders: dict[str, Order] = {}
        self.prices: dict[str, Decimal] = {}
        self._calendar = TradingCalendar(AssetClass.EQUITY)
        broker.set_listener(self.on_order_update)

    # -- events -------------------------------------------------------------

    def add_listener(self, listener: EngineListener) -> None:
        self._listeners.append(listener)

    def _emit(self, kind: str, **fields) -> None:
        event = EngineEvent(kind, self.clock.now(), **fields)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Engine listener failed on %s event", kind)

    # -- market data --------------------------------------------------------

    def _record(self, bar: Bar) -> bool:
        series = self._history.get(bar.symbol)
        if series is None or (series and bar.start <= series[-1].start):
            return False
        series.append(bar)
        self.prices[bar.symbol] = _decimal(bar.close)
        return True

    def warm_up(self, history: Mapping[str, Sequence[Bar]]) -> None:
        """Load history without trading on it (start of a backtest or a restart)."""
        for bars in history.values():
            for bar in sorted(bars, key=lambda b: b.start):
                self._record(bar)

    def latest_bar_starts(self) -> dict[str, datetime]:
        """Start of the newest bar held per symbol (where a data feed should resume)."""
        return {s: h[-1].start for s, h in self._history.items() if h}

    def on_bars(self, bars: Mapping[str, Bar]) -> list[OrderRequest]:
        """
        Handle newly closed bars (at most one per symbol).

        Returns:
            The order requests sent to the broker.
        """
        new = {symbol: bar for symbol, bar in bars.items() if self._record(bar)}
        if not new:
            return []
        self.broker.on_bars(new)
        if self.risk.halted:
            return []

        changed = False
        equity = self.equity()
        for allocation in self.allocations:
            strategy = allocation.strategy
            symbols = [i.symbol for i in strategy.instruments]
            if not any(s in new for s in symbols):
                continue
            if any(len(self._history[s]) < strategy.lookback for s in symbols):
                continue  # still warming up
            ctx = StrategyContext(
                now=self.clock.now(),
                history={s: tuple(self._history[s]) for s in symbols},
                positions={s: self.ledger.position(s).qty for s in symbols},
                equity=equity * allocation.weight,
            )
            try:
                weights = strategy.target_weights(ctx)
                if weights is None:
                    continue
                unknown = set(weights) - set(symbols)
                if unknown:
                    raise ValueError(f"targets for symbols outside the strategy: {sorted(unknown)}")
                self._targets[strategy.name] = {
                    s: _decimal(float(weights.get(s, 0.0))).quantize(WEIGHT_SCALE) for s in symbols}
                changed = True
            except Exception as e:
                logger.exception("Strategy %s failed", strategy.name)
                self._emit("strategy_error", detail=f"{strategy.name}: {e}")
        return self.rebalance() if changed else []

    # -- account state ------------------------------------------------------

    def _marks(self) -> dict[str, Decimal]:
        """Reference prices for every held symbol (entry price until data arrives)."""
        marks = dict(self.prices)
        for symbol, position in self.ledger.positions.items():
            marks.setdefault(symbol, position.avg_price)
        return marks

    def equity(self) -> Decimal:
        return self.ledger.equity(self._marks())

    def exposure(self) -> dict[str, Decimal]:
        """Signed quantity per symbol including the unfilled part of open orders."""
        exposure = {s: p.qty for s, p in self.ledger.positions.items()}
        for order in self.orders.values():
            if order.status.is_open:
                symbol = order.request.symbol
                exposure[symbol] = exposure.get(symbol, ZERO) + order.remaining_qty * order.request.side.sign
        return exposure

    def target_weights(self) -> dict[str, Decimal]:
        """Account-level targets: each strategy's weights times its allocation."""
        combined: dict[str, Decimal] = {}
        for allocation in self.allocations:
            for symbol, weight in self._targets.get(allocation.strategy.name, {}).items():
                combined[symbol] = combined.get(symbol, ZERO) + weight * allocation.weight
        return combined

    # -- orders -------------------------------------------------------------

    def _time_in_force(self, instrument: Instrument, now: datetime) -> TimeInForce:
        if instrument.asset_class is AssetClass.CRYPTO:
            return TimeInForce.GTC
        local = now.astimezone(NEW_YORK)
        today = local.date()
        if self._calendar.is_session(today):
            if self._calendar.session_open(today) <= now < self._calendar.session_close(today):
                return TimeInForce.DAY
            if now < self._calendar.session_open(today):
                next_open = self._calendar.session_open(today)
            else:
                next_open = self._calendar.session_open(self._calendar.next_session(today))
        else:
            next_open = self._calendar.session_open(self._calendar.next_session(today))
        evening_before = datetime.combine(
            self._calendar.previous_session(next_open.astimezone(NEW_YORK).date()), OPG_WINDOW_START, NEW_YORK)
        if evening_before <= now <= next_open - OPG_CUTOFF:
            return TimeInForce.OPG
        return TimeInForce.DAY

    def rebalance(self) -> list[OrderRequest]:
        """Send the orders that move current exposure to the target weights."""
        now = self.clock.now()
        equity = self.equity()
        if equity <= 0:
            return []
        exposure = self.exposure()
        sells: list[tuple[Instrument, Decimal, Decimal]] = []
        buys: list[tuple[Instrument, Decimal, Decimal]] = []
        for symbol, weight in self.target_weights().items():
            instrument = self.instruments[symbol]
            price = self.prices.get(symbol)
            if price is None:
                continue
            if weight < 0 and not instrument.shortable:
                logger.warning("Ignoring short target for %s, which cannot be shorted", symbol)
                weight = ZERO
            current = exposure.get(symbol, ZERO)
            target = instrument.round_qty(weight * equity / price)
            # Positions can sit off the increment grid (crypto fees are taken in the
            # asset), so the order itself is rounded; dust below one increment stays
            delta = instrument.round_qty(target - current)
            closing = target == 0 and current != 0
            if abs(delta) < instrument.min_qty:
                continue
            if not closing and abs(delta) * price < self.min_trade_fraction * equity:
                continue
            (sells if delta < 0 else buys).append((instrument, delta, price))

        # Buys are funded by cash and by proceeds from reducing longs, never by shorting
        long_proceeds = sum((min(-d, max(exposure.get(i.symbol, ZERO), ZERO)) * p for i, d, p in sells), ZERO)
        open_buys = sum((o.remaining_qty * self.prices.get(o.request.symbol, ZERO) for o in self.orders.values()
                         if o.status.is_open and o.request.side is Side.BUY), ZERO)
        available = self.ledger.cash + long_proceeds - open_buys - self.cash_buffer * equity
        needed = sum((d * p for _, d, p in buys), ZERO)
        if needed > 0 and needed > available:
            scale = max(available, ZERO) / needed
            buys = [(i, i.round_qty(d * scale), p) for i, d, p in buys]
            buys = [(i, d, p) for i, d, p in buys if d >= i.min_qty]

        sent = []
        for instrument, delta, _price in sells + buys:
            side = Side.BUY if delta > 0 else Side.SELL
            request = OrderRequest(
                client_order_id=f"nt-{now:%y%m%d%H%M%S}-{_slug(instrument.symbol)}-{side.value[0]}",
                symbol=instrument.symbol, side=side, qty=abs(delta), order_type=OrderType.MARKET,
                time_in_force=self._time_in_force(instrument, now), tag="rebalance")
            if self._send(request).allowed:
                sent.append(request)
        return sent

    def _send(self, request: OrderRequest) -> RiskDecision:
        if request.client_order_id in self.orders:
            logger.warning("Not resending order %s", request.client_order_id)
            return RiskDecision(False, "duplicate client_order_id")
        if request.symbol not in self.instruments:
            return RiskDecision(False, f"unknown symbol {request.symbol}")
        decision = self.risk.check(request, self.prices.get(request.symbol), self.exposure(), self._marks(),
                                   self.equity())
        if not decision.allowed:
            logger.warning("Risk gate rejected %s: %s", request.client_order_id, decision.reason)
            self._emit("rejected", detail=f"{request.client_order_id}: {decision.reason}")
            return decision
        order = Order.new(request, self.clock.now())
        self.orders[request.client_order_id] = order
        self._emit("order", order=order)
        self.broker.submit(request)
        return decision

    def submit_order(self, request: OrderRequest) -> RiskDecision:
        """An operator's order: same risk gate and broker path as strategy orders."""
        return self._send(request)

    def on_order_update(self, update: OrderUpdate) -> None:
        """
        Book a broker report for one of the engine's orders. Fills are booked even
        if the status transition is out of order. Reports about orders the engine
        did not place (e.g. trades made by hand) are not booked here: the next
        reconciliation adopts their effect from the broker, so nothing is counted twice.
        """
        order = self.orders.get(update.client_order_id)
        if order is None:
            logger.warning("Update for unknown order %s (%s)", update.client_order_id, update.status)
            if update.fill is not None:
                self._emit("unknown_fill", fill=update.fill)
            return
        try:
            order = order.apply(update)
            self.orders[update.client_order_id] = order
        except InvalidTransitionError as e:
            logger.error("Inconsistent broker update: %s", e)
            self._emit("broker_error", detail=f"broker update: {e}")
        if update.fill is not None and self.ledger.apply_fill(update.fill):
            self._emit("fill", fill=update.fill, order=order)
        self._emit("order", order=order)

    def sync_order(self, order: Order) -> None:
        """Take the broker's view of an order (reconciliation and restarts)."""
        self.orders[order.client_order_id] = order
        self._emit("order", order=order)

    # -- kill switch --------------------------------------------------------

    def halt(self, reason: str, flatten: bool = False) -> list[OrderRequest]:
        """
        Stop trading: cancel every open order and, with ``flatten``, close every
        position with market orders. Returns the closing orders sent.
        """
        self.risk.halt(reason)
        self._emit("halt", detail=reason)
        for order in list(self.orders.values()):
            if order.status.is_open:
                self.broker.cancel(order.client_order_id)
        if not flatten:
            return []
        now = self.clock.now()
        sent = []
        for symbol, position in self.ledger.positions.items():
            instrument = self.instruments.get(symbol)
            qty = instrument.round_qty(abs(position.qty)) if instrument else abs(position.qty)
            if instrument and qty < instrument.min_qty:
                continue  # dust below the broker's minimum
            side = Side.SELL if position.qty > 0 else Side.BUY
            tif = self._time_in_force(instrument, now) if instrument else TimeInForce.DAY
            request = OrderRequest(f"nt-{now:%y%m%d%H%M%S}-{_slug(symbol)}-x", symbol, side, qty,
                                   OrderType.MARKET, tif, tag="flatten")
            if self._send(request).allowed:
                sent.append(request)
        return sent

    def resume(self) -> None:
        self.risk.resume()
        self._emit("resume")

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> EngineSnapshot:
        marks = self._marks()
        positions = tuple(
            PositionView(s, p.qty, p.avg_price, marks[s], p.market_value(marks[s]), p.unrealized_pnl(marks[s]))
            for s, p in sorted(self.ledger.positions.items()))
        return EngineSnapshot(
            timestamp=self.clock.now(),
            equity=self.ledger.equity(marks),
            cash=self.ledger.cash,
            realized_pnl=self.ledger.realized_pnl,
            fees=self.ledger.fees,
            positions=positions,
            open_orders=tuple(o for o in self.orders.values() if o.status.is_open),
            target_weights=self.target_weights(),
            halted=self.risk.halted,
            halt_reason=self.risk.halt_reason,
            last_bar_end={s: h[-1].end for s, h in self._history.items() if h},
        )
