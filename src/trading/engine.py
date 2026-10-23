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

Between 3 and 4 the sizer caps the targets, scales them to the volatility
target and shrinks them as the drawdown grows; after the mark-to-market in 2
the risk gate may trip the drawdown kill switch or start a cool-down. With a
cash sleeve (a T-bill ETF), the equity the sized targets leave unused is held
in it instead of in cash.

Equity orders go to the next opening auction, or at once during the session;
strategies that ask for it (``execution`` CLOSE) trade in the next closing
auction instead.

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
from decimal import ROUND_DOWN, Decimal
from typing import Any

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
from .risk import OrderContext, RiskDecision, RiskGate
from .sizing import PortfolioSizer, SizingPolicy, SizingResult
from .strategy import CLOSE, Strategy, StrategyContext, common_execution, needed_history

logger = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")
WEIGHT_SCALE = Decimal("1e-6")

#: Alpaca accepts opening-auction (OPG) orders from 19:00 ET the evening before
#: until 09:28 ET; outside that window market orders are sent as DAY orders.
OPG_WINDOW_START = time(19, 0)
OPG_CUTOFF = timedelta(minutes=2)
#: Alpaca rejects closing-auction (CLS) orders from 15:50 ET until 19:00 ET; from
#: 19:00 they go to the next session's close. The cutoff is taken as 10 minutes
#: before the session's close, which is 15:50 on a full day.
CLS_CUTOFF = timedelta(minutes=10)


@dataclass(frozen=True)
class Allocation:
    """A strategy and the fraction of account equity it manages."""

    strategy: Strategy
    weight: Decimal


@dataclass(frozen=True)
class EngineEvent:
    """
    Something worth recording. ``kind`` is one of: order, fill, rejected,
    strategy_error, broker_error, unknown_fill, reconciled, halt, resume,
    kill_switch, risk_state (``data`` holds the risk state to persist).
    """

    kind: str
    timestamp: datetime
    order: Order | None = None
    fill: Fill | None = None
    detail: str = ""
    data: Mapping[str, Any] | None = None


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
class RiskView:
    drawdown: Decimal
    high_water_mark: Decimal | None
    cooling_down: bool
    orders_today: int
    turnover_today: Decimal
    predicted_volatility: float | None
    volatility_scale: Decimal
    drawdown_scale: Decimal
    halt_kind: str


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
    risk: RiskView


def _decimal(value: float) -> Decimal:
    return Decimal(repr(value))


def _slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", symbol)[:12]


class Engine:
    def __init__(self, *, clock: Clock, broker: Broker, ledger: Ledger, allocations: Sequence[Allocation],
                 risk: RiskGate | None = None, sizing: SizingPolicy | None = None,
                 cash_buffer: Decimal = Decimal("0.005"), cash_sleeve: Instrument | None = None,
                 close_out: Sequence[Instrument] = (),
                 listeners: Sequence[EngineListener] = ()) -> None:
        """
        Args:
            risk: Risk gate; default limits when omitted.
            sizing: Sizing policy (volatility target, drawdown ladder, rebalance band);
                the conservative defaults when omitted.
            cash_buffer: Fraction of equity kept unspent to absorb slippage and fees.
            cash_sleeve: A T-bill ETF that holds the equity the targets leave unused
                (None: it stays in cash). No strategy may trade it.
            close_out: Instruments no strategy trades that the account may still hold (left
                by a strategy mix that changed): targeted at zero, so any position is sold.

        Raises:
            ValueError: For invalid allocations or conflicting instrument definitions,
                strategies trading at different auctions, or an invalid cash sleeve.
        """
        if not allocations:
            raise ValueError("at least one allocation is required")
        if any(a.weight <= 0 for a in allocations) or sum(a.weight for a in allocations) > 1:
            raise ValueError("allocation weights must be positive and sum to at most 1")
        names = [a.strategy.name for a in allocations]
        if len(set(names)) != len(names):
            raise ValueError("strategy names must be unique")
        #: Equity orders go to the next closing auction instead of the open
        self.at_close = common_execution([a.strategy for a in allocations]) == CLOSE

        self.clock = clock
        self.broker = broker
        self.ledger = ledger
        self.allocations = tuple(allocations)
        self.risk = risk or RiskGate()
        self.sizer = PortfolioSizer(sizing or SizingPolicy(), self.risk.limits)
        self.cash_buffer = cash_buffer
        self.last_sizing: SizingResult | None = None
        self._listeners = list(listeners)

        self.instruments: dict[str, Instrument] = {}
        lookbacks: dict[str, int] = {}
        for allocation in self.allocations:
            for instrument in allocation.strategy.instruments:
                known = self.instruments.setdefault(instrument.symbol, instrument)
                if known != instrument:
                    raise ValueError(f"conflicting definitions for {instrument.symbol}")
                lookbacks[instrument.symbol] = max(lookbacks.get(instrument.symbol, 1),
                                                   allocation.strategy.lookback, self.sizer.lookback)
        traded = set(self.instruments)
        for instrument in close_out:
            known = self.instruments.setdefault(instrument.symbol, instrument)
            if known != instrument:
                raise ValueError(f"conflicting definitions for {instrument.symbol}")
            lookbacks.setdefault(instrument.symbol, 2)  # a price and the bar before, for the jump check
        #: Symbols held over from an earlier mix, sold at the next rebalance
        self.close_out = tuple(sorted({i.symbol for i in close_out} - traded))
        self.cash_sleeve = cash_sleeve
        if cash_sleeve is not None:
            if cash_sleeve.asset_class is not AssetClass.EQUITY:
                raise ValueError("the cash sleeve must be an equity (a T-bill ETF)")
            if cash_sleeve.symbol in self.instruments:
                raise ValueError(f"the cash sleeve {cash_sleeve.symbol} must not be traded by a strategy "
                                 "or closed out")
            self.instruments[cash_sleeve.symbol] = cash_sleeve
            lookbacks[cash_sleeve.symbol] = 2  # a price and the bar before, for the risk gate's jump check
        self._history: dict[str, deque[Bar]] = {s: deque(maxlen=n) for s, n in lookbacks.items()}
        self._targets: dict[str, dict[str, Decimal]] = {}
        self.orders: dict[str, Order] = {}
        self._open: dict[str, Order] = {}  # the working subset of ``orders``
        self.prices: dict[str, Decimal] = {}
        self._calendar = TradingCalendar(AssetClass.EQUITY)
        broker.set_listener(self.on_order_update)

    # -- events -------------------------------------------------------------

    def add_listener(self, listener: EngineListener) -> None:
        self._listeners.append(listener)

    def _emit(self, kind: str, **fields: Any) -> None:
        event = EngineEvent(kind, self.clock.now(), **fields)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Engine listener failed on %s event", kind)

    def emit_risk_state(self) -> None:
        self._emit("risk_state", data=self.risk.state_dict())

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
        equity = self.equity()
        if self.risk.on_equity(equity, self.clock.now()) == "kill":
            self._emit("kill_switch", detail=self.risk.halt_reason)
            self._guarded(lambda: self.halt(self.risk.halt_reason, flatten=self.risk.limits.flatten_on_kill,
                                            kind="drawdown"))
            return []
        if self.risk.halted:
            if self._liquidation_incomplete():
                self._guarded(self._liquidate)  # the kill switch's liquidation failed: retry
            self.emit_risk_state()
            return []

        for allocation in self.allocations:
            strategy = allocation.strategy
            symbols = [i.symbol for i in strategy.instruments]
            if not any(s in new for s in symbols):
                continue
            if any(len(self._history[s]) < needed_history(strategy) for s in symbols):
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
            except Exception as e:
                logger.exception("Strategy %s failed", strategy.name)
                self._emit("strategy_error", detail=f"{strategy.name}: {e}")
        # Rebalance on every new bar, not only when targets change: volatility and
        # drawdown scaling move with the market. The rebalance band prevents churn.
        sent = self.rebalance() if self._targets else []
        self.emit_risk_state()
        return sent

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
        for order in self._open.values():
            symbol = order.request.symbol
            exposure[symbol] = exposure.get(symbol, ZERO) + order.remaining_qty * order.request.side.sign
        return exposure

    def open_orders(self) -> list[Order]:
        """Orders still working at the broker, oldest first."""
        return list(self._open.values())

    def _store(self, order: Order) -> None:
        self.orders[order.client_order_id] = order
        if order.status.is_open:
            self._open[order.client_order_id] = order
        else:
            self._open.pop(order.client_order_id, None)

    def target_weights(self) -> dict[str, Decimal]:
        """Account-level targets: each strategy's weights times its allocation; zero for close-outs."""
        combined: dict[str, Decimal] = {}
        for allocation in self.allocations:
            for symbol, weight in self._targets.get(allocation.strategy.name, {}).items():
                combined[symbol] = combined.get(symbol, ZERO) + weight * allocation.weight
        for symbol in self.close_out:
            combined[symbol] = ZERO
        return combined

    # -- orders -------------------------------------------------------------

    def _time_in_force(self, instrument: Instrument, now: datetime) -> TimeInForce | None:
        """How a market order is sent now; None when it cannot be sent until later."""
        if instrument.asset_class is AssetClass.CRYPTO:
            return TimeInForce.GTC
        if self.at_close:
            return self._closing_auction(now)
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

    def _closing_auction(self, now: datetime) -> TimeInForce | None:
        """CLS for the next closing auction Alpaca takes orders for now; None from the cutoff until 19:00 ET."""
        today = now.astimezone(NEW_YORK).date()
        session = today if self._calendar.is_session(today) else self._calendar.next_session(today)
        if now >= self._calendar.session_close(session) - CLS_CUTOFF:
            session = self._calendar.next_session(session)
        evening_before = datetime.combine(self._calendar.previous_session(session), OPG_WINDOW_START, NEW_YORK)
        return TimeInForce.CLS if today == session or now >= evening_before else None

    def rebalance(self) -> list[OrderRequest]:
        """Send the orders that move current exposure to the target weights."""
        now = self.clock.now()
        equity = self.equity()
        if equity <= 0:
            return []
        exposure = self.exposure()
        targets = {}
        for symbol, weight in self.target_weights().items():
            if weight < 0 and not self.instruments[symbol].shortable:
                logger.warning("Ignoring short target for %s, which cannot be shorted", symbol)
                weight = ZERO
            targets[symbol] = weight
        self.last_sizing = self.sizer.size(targets, self._history, self.instruments, self.risk.drawdown)
        band = self.sizer.policy.rebalance_band
        sleeve = self.cash_sleeve.symbol if self.cash_sleeve is not None else None

        sells: list[tuple[Instrument, Decimal, Decimal]] = []
        buys: list[tuple[Instrument, Decimal, Decimal]] = []

        def plan(symbol: str, weight: Decimal) -> Decimal:
            """Queue the order that moves ``symbol`` to ``weight``; its signed quantity (0: none)."""
            instrument = self.instruments[symbol]
            price = self.prices.get(symbol)
            if price is None:
                return ZERO
            current = exposure.get(symbol, ZERO)
            target = instrument.round_qty(weight * equity / price)
            # Positions can sit off the increment grid (crypto fees are taken in the
            # asset), so the order itself is rounded; dust below one increment stays
            delta = instrument.round_qty(target - current)
            closing = target == 0 and current != 0
            if abs(delta) < instrument.min_qty:
                return ZERO
            if not closing and abs(delta) * price < band * equity:
                return ZERO  # inside the rebalance band
            if self.risk.cooling_down and abs(current + delta) >= abs(current) and symbol != sleeve:
                return ZERO  # during a cool-down only risk-reducing trades (T-bills add no risk)
            (sells if delta < 0 else buys).append((instrument, delta, price))
            return delta

        held = dict(exposure)
        for symbol, weight in self.last_sizing.weights.items():
            held[symbol] = held.get(symbol, ZERO) + plan(symbol, weight)
        if sleeve is not None:
            # The sleeve holds what the positions leave after this rebalance's trades, not what
            # their targets would: a target moving inside the band must not trade T-bills for cash
            marks = self._marks()
            invested = sum((abs(q) * marks.get(s, ZERO) for s, q in held.items() if s != sleeve), ZERO)
            room = min(ONE, self.risk.limits.max_gross_exposure) - self.cash_buffer - invested / equity
            plan(sleeve, max(room, ZERO).quantize(WEIGHT_SCALE, rounding=ROUND_DOWN))

        # Buys are funded by cash and by proceeds from reducing longs, never by shorting
        long_proceeds = sum((min(-d, max(exposure.get(i.symbol, ZERO), ZERO)) * p for i, d, p in sells), ZERO)
        open_buys = sum((o.remaining_qty * self.prices.get(o.request.symbol, ZERO) for o in self._open.values()
                         if o.request.side is Side.BUY), ZERO)
        available = self.ledger.cash + long_proceeds - open_buys - self.cash_buffer * equity
        # The strategies' positions are funded first; the cash sleeve takes what is left
        sleeve_buys = [(i, d, p) for i, d, p in buys if i.symbol == sleeve]
        buys = [(i, d, p) for i, d, p in buys if i.symbol != sleeve]
        needed = sum((d * p for _, d, p in buys), ZERO)
        if needed > 0 and needed > available:
            scale = max(available, ZERO) / needed
            buys = [(i, i.round_qty(d * scale), p) for i, d, p in buys]
            buys = [(i, d, p) for i, d, p in buys if d >= i.min_qty]
        left = available - sum((d * p for _, d, p in buys), ZERO)
        for instrument, delta, price in sleeve_buys:
            delta = min(delta, instrument.round_qty(max(left, ZERO) / price))
            if delta >= instrument.min_qty:
                buys.append((instrument, delta, price))

        sent = []
        for instrument, delta, _price in sells + buys:
            time_in_force = self._time_in_force(instrument, now)
            if time_in_force is None:
                logger.info("No %s order now: the closing auction takes orders again at 19:00 ET",
                            instrument.symbol)
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            request = OrderRequest(
                client_order_id=f"nt-{now:%y%m%d%H%M%S}-{_slug(instrument.symbol)}-{side.value[0]}",
                symbol=instrument.symbol, side=side, qty=abs(delta), order_type=OrderType.MARKET,
                time_in_force=time_in_force, tag="rebalance")
            if self._send(request).allowed:
                sent.append(request)
        return sent

    def _send(self, request: OrderRequest) -> RiskDecision:
        if request.client_order_id in self.orders:
            logger.warning("Not resending order %s", request.client_order_id)
            return RiskDecision(False, "duplicate client_order_id")
        if request.symbol not in self.instruments:
            return RiskDecision(False, f"unknown symbol {request.symbol}")
        now = self.clock.now()
        price = self.prices.get(request.symbol)
        history: Sequence[Bar] = self._history.get(request.symbol) or ()
        jump = None
        if len(history) >= 2 and history[-2].close > 0:
            jump = abs(_decimal(history[-1].close / history[-2].close) - 1)
        context = OrderContext(
            now=now, price=price, exposure=self.exposure(), prices=self._marks(),
            asset_classes={s: i.asset_class for s, i in self.instruments.items()}, equity=self.equity(),
            last_bar_end=history[-1].end if history else None, price_jump=jump,
            cash_equivalent=self.cash_sleeve is not None and request.symbol == self.cash_sleeve.symbol)
        decision = self.risk.check(request, context)
        if not decision.allowed:
            logger.warning("Risk gate rejected %s: %s", request.client_order_id, decision.reason)
            self._emit("rejected", detail=f"{request.client_order_id}: {decision.reason}")
            return decision
        assert price is not None  # the gate refuses an order without a reference price
        self.risk.record(request, price, now)
        order = Order.new(request, self.clock.now())
        self._store(order)
        self._emit("order", order=order)
        self.broker.submit(request)
        return decision

    def submit_order(self, request: OrderRequest) -> RiskDecision:
        """An operator's order: same risk gate and broker path as strategy orders."""
        decision = self._send(request)
        if decision.allowed:
            self.emit_risk_state()
        return decision

    def cancel_order(self, client_order_id: str) -> bool:
        """Ask the broker to cancel a working order; False if the engine has no such open order."""
        order = self.orders.get(client_order_id)
        if order is None or not order.status.is_open:
            return False
        self.broker.cancel(client_order_id)
        return True

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
            self._store(order)
        except InvalidTransitionError as e:
            logger.error("Inconsistent broker update: %s", e)
            self._emit("broker_error", detail=f"broker update: {e}")
        if update.fill is not None and self.ledger.apply_fill(update.fill):
            self._emit("fill", fill=update.fill, order=order)
        self._emit("order", order=order)

    def sync_order(self, order: Order) -> None:
        """Take the broker's view of an order (reconciliation and restarts)."""
        self._store(order)
        self._emit("order", order=order)

    # -- kill switch --------------------------------------------------------

    def halt(self, reason: str, flatten: bool = False, kind: str = "manual") -> list[str]:
        """
        Stop trading (the kill switch): only position-reducing orders pass the
        risk gate until ``resume``. Without ``flatten`` every open order is
        cancelled. With ``flatten`` the broker's own liquidation cancels every
        order and closes every position: the broker sequences cancels and closes
        correctly, which separate engine orders cannot while cancels are still
        in flight. The liquidation orders are adopted so their fills are booked.

        Returns:
            Client order ids of the liquidation orders (empty without ``flatten``).
        """
        self.risk.halt(reason, kind=kind, now=self.clock.now())
        self._emit("halt", detail=reason)
        self.emit_risk_state()
        if not flatten:
            for order in self.open_orders():
                self.broker.cancel(order.client_order_id)
            return []
        return self._liquidate()

    def _liquidate(self) -> list[str]:
        ids = self.broker.close_all()
        for order in self.broker.open_orders():
            if order.client_order_id not in self.orders:
                self.sync_order(order)
        return ids

    def _liquidation_incomplete(self) -> bool:
        """After a drawdown kill with flatten: positions left and nothing working to close them."""
        if self.risk.state.halt_kind != "drawdown" or not self.risk.limits.flatten_on_kill:
            return False
        if self._open:
            return False
        for symbol, position in self.ledger.positions.items():
            instrument = self.instruments.get(symbol)
            if instrument is None or instrument.round_qty(abs(position.qty)) >= instrument.min_qty:
                return True
        return False

    def _guarded(self, action: Callable[[], object]) -> None:
        """Run an automatic risk action; a broker failure is reported and retried on the next bar."""
        try:
            action()
        except Exception as e:
            logger.exception("Automatic liquidation failed")
            self._emit("broker_error", detail=f"kill switch liquidation failed: {e}")

    def resume(self) -> None:
        self.risk.resume(self.equity())
        self._emit("resume")
        self.emit_risk_state()

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> EngineSnapshot:
        marks = self._marks()
        sizing = self.last_sizing
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
            open_orders=tuple(self._open.values()),
            target_weights=self.target_weights(),
            halted=self.risk.halted,
            halt_reason=self.risk.halt_reason,
            last_bar_end={s: h[-1].end for s, h in self._history.items() if h},
            risk=RiskView(
                drawdown=self.risk.drawdown,
                high_water_mark=self.risk.state.high_water_mark,
                cooling_down=self.risk.cooling_down,
                orders_today=self.risk.state.orders_today,
                turnover_today=self.risk.state.turnover_today,
                predicted_volatility=sizing.volatility if sizing else None,
                volatility_scale=sizing.volatility_scale if sizing else Decimal("1"),
                drawdown_scale=sizing.drawdown_scale if sizing else Decimal("1"),
                halt_kind=self.risk.state.halt_kind,
            ),
        )
