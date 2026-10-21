"""
NOCTURNA Trading System - Backtest runner.

Replays bars through the live engine on a simulated clock: each bar is
delivered at the moment it would have become available (bar end plus feed
delay) and orders fill on later bars in the simulated broker. The validation
pipeline (``src/trading/validation``) builds on this runner.

A drawdown kill halts the engine until an operator resumes it. By default a
backtest stays halted, as an unattended account would; ``resume_after_kill``
simulates an operator who resumes that long after the kill, once the book is
flat, so a long backtest keeps measuring the strategy after its first kill.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from .broker import DEFAULT_COSTS, CostModel, SimulatedBroker
from .clock import SimulatedClock
from .data import DEFAULT_FEED_DELAY
from .domain import AssetClass, Bar, Fill, Order
from .engine import Allocation, Engine, EngineEvent, EngineSnapshot
from .ledger import Ledger
from .risk import RiskGate, RiskLimits
from .sizing import SizingPolicy


@dataclass(frozen=True)
class BacktestConfig:
    start: datetime
    end: datetime
    starting_cash: Decimal = Decimal("100000")
    costs: Mapping[AssetClass, CostModel] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    sizing: SizingPolicy = field(default_factory=SizingPolicy)
    delays: Mapping[AssetClass, timedelta] = field(default_factory=lambda: dict(DEFAULT_FEED_DELAY))
    participation: Decimal | None = None
    #: Simulated operator: resume this long after a drawdown kill (None: stay halted)
    resume_after_kill: timedelta | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.resume_after_kill is not None and self.resume_after_kill < timedelta(0):
            raise ValueError("resume_after_kill must not be negative")


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: tuple[tuple[datetime, Decimal], ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    events: tuple[EngineEvent, ...]
    final: EngineSnapshot
    starting_cash: Decimal

    def metrics(self) -> dict[str, float]:
        """Headline statistics (no risk-free rate). Phase 4 adds the rigorous ones."""
        values = [float(v) for _, v in self.equity_curve]
        kills = float(sum(1 for e in self.events if e.kind == "kill_switch"))
        if len(values) < 2:
            return {"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0,
                    "max_drawdown": 0.0, "fills": float(len(self.fills)), "kill_switch_trips": kills}
        returns = [b / a - 1 for a, b in zip(values[:-1], values[1:], strict=True) if a > 0]
        years = (self.equity_curve[-1][0] - self.equity_curve[0][0]).total_seconds() / (365.25 * 86400)
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
        per_year = len(returns) / years if years > 0 else 0.0
        volatility = math.sqrt(var * per_year)
        peak, max_dd = values[0], 0.0
        for v in values:
            peak = max(peak, v)
            max_dd = min(max_dd, v / peak - 1)
        total = values[-1] / values[0] - 1
        return {
            "total_return": total,
            "cagr": (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 and values[-1] > 0 else 0.0,
            "volatility": volatility,
            "sharpe": mean * per_year / volatility if volatility > 0 else 0.0,
            "max_drawdown": max_dd,
            "fills": float(len(self.fills)),
            "fees": float(self.final.fees),
            "kill_switch_trips": kills,
        }


def run_backtest(allocations: Sequence[Allocation], bars: Mapping[str, Sequence[Bar]],
                 config: BacktestConfig) -> BacktestResult:
    """
    Run strategies over historical bars.

    Bars already final at ``config.start`` only warm up history; trading
    decisions happen for bars that become final in (start, end).
    """
    instruments = {i.symbol: i for a in allocations for i in a.strategy.instruments}
    timeline: dict[datetime, dict[str, Bar]] = defaultdict(dict)
    warm_up: dict[str, list[Bar]] = defaultdict(list)
    for symbol, series in bars.items():
        instrument = instruments.get(symbol)
        if instrument is None:
            continue
        delay = config.delays[instrument.asset_class]
        for bar in series:
            available = bar.end + delay
            if available <= config.start:
                warm_up[symbol].append(bar)  # already final when the backtest starts
            elif available < config.end:
                timeline[available][symbol] = bar

    clock = SimulatedClock(config.start)
    broker = SimulatedBroker(clock, instruments, config.starting_cash, costs=config.costs,
                             participation=config.participation)
    events: list[EngineEvent] = []
    engine = Engine(clock=clock, broker=broker, ledger=Ledger(config.starting_cash), allocations=allocations,
                    risk=RiskGate(config.risk_limits), sizing=config.sizing, listeners=[events.append])
    engine.warm_up(warm_up)

    curve: list[tuple[datetime, Decimal]] = []
    for available, group in sorted(timeline.items()):
        clock.advance_to(max(available, clock.now()))
        if config.resume_after_kill is not None and _operator_would_resume(engine, config.resume_after_kill):
            engine.resume()
        engine.on_bars(group)
        curve.append((clock.now(), engine.equity()))

    return BacktestResult(
        equity_curve=tuple(curve),
        orders=tuple(engine.orders.values()),
        fills=tuple(e.fill for e in events if e.kind == "fill" and e.fill is not None),
        events=tuple(events),
        final=engine.snapshot(),
        starting_cash=config.starting_cash,
    )


def _operator_would_resume(engine: Engine, after: timedelta) -> bool:
    state = engine.risk.state
    return (state.halted and state.halt_kind == "drawdown" and state.halted_at is not None
            and engine.clock.now() - state.halted_at >= after
            and not engine.open_orders() and not engine.ledger.positions)
