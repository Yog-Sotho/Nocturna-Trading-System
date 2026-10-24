"""
NOCTURNA Trading System - Running a validation or a backtest on demand.

``flask validate`` and the dashboard's strategy lab both call these, so a
report from either is the same computation on the same data:

* ``validate_candidate``: the promotion gates for one registered strategy
  family and its whole parameter grid (``pipeline.validate``);
* ``backtest_mix``: one backtest of a trading profile's mix with its risk
  limits, as the engine would trade it, summarised against T-bills.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.config import Settings

from ..backtest import BacktestConfig, BacktestResult, run_backtest
from ..lineup import CASH_SLEEVE
from ..profile import Profile, allocations, effective_settings, symbols, warm_up_days
from ..runtime import risk_limits, sizing_policy
from . import sources
from .candidates import CANDIDATES
from .experiment import WalkForwardConfig
from .gates import PromotionGates
from .pipeline import ValidationConfig, validate
from .report import ValidationReport
from .stats import Summary, daily_returns, summarize

#: A backtest has no operator: a drawdown kill is followed by a simulated resume this much later
RESUME_AFTER_KILL = timedelta(days=30)
#: Points of an equity curve kept for the dashboard
CURVE_POINTS = 600


def thin(curve: Sequence[tuple[datetime, Decimal]], points: int = CURVE_POINTS) -> list[dict[str, str]]:
    """An equity curve with at most about ``points`` marks, evenly spaced, the last one always kept."""
    step = max(1, -(-len(curve) // points))
    kept = list(curve[::step])
    if curve and kept[-1] != curve[-1]:
        kept.append(curve[-1])
    return [{"timestamp": t.isoformat(), "equity": str(v)} for t, v in kept]


def validate_candidate(settings: Settings, name: str, start: datetime, end: datetime, *,
                       data: str = sources.ALPACA, tbills: bool = True,
                       resume_after_kill: timedelta = RESUME_AFTER_KILL,
                       walk_forward: WalkForwardConfig | None = None, gates: PromotionGates | None = None,
                       loaded: Callable[[sources.MarketData], None] | None = None,
                       ) -> tuple[ValidationReport, sources.MarketData]:
    """
    Args:
        loaded: Told the data once it is loaded, before the long computation (the CLI prints it).

    Raises:
        KeyError: For a strategy that is not registered.
        sources.DataError, TradingNotConfiguredError: When the bars cannot be had.
        ValueError: If the period is too short to validate.
    """
    spec = CANDIDATES[name]
    wanted = sources.with_tbills(spec.symbols) if tbills else spec.symbols
    market = sources.load(settings, wanted, start - timedelta(days=spec.warm_up_days), end, data)
    if loaded is not None:
        loaded(market)
    config = ValidationConfig(
        BacktestConfig(start=start, end=end, risk_limits=risk_limits(settings), sizing=sizing_policy(settings),
                       resume_after_kill=resume_after_kill,
                       cash_sleeve=market.instruments[CASH_SLEEVE] if tbills else None),
        walk_forward=walk_forward or WalkForwardConfig(), gates=gates or PromotionGates(),
        hurdle=market.hurdle() if tbills else None, hurdle_label=market.hurdle_label, data_label=market.data_label)
    return validate(spec.build(market.instruments), market.bars, config), market


@dataclass(frozen=True)
class MixBacktest:
    result: BacktestResult
    summary: Summary
    market: sources.MarketData

    def years(self) -> list[dict[str, Any]]:
        """The return of each calendar year."""
        returns = daily_returns(self.result.equity_curve)
        growth: dict[int, float] = {}
        for day, value in zip(returns.dates, returns.values, strict=True):
            growth[day.year] = growth.get(day.year, 1.0) * (1 + float(value))
        return [{"year": year, "return": g - 1} for year, g in sorted(growth.items())]

    @property
    def kill_switch_trips(self) -> int:
        return sum(1 for e in self.result.events if e.kind == "kill_switch")


def backtest_mix(settings: Settings, profile: Profile, start: datetime, end: datetime,
                 data: str = sources.ALPACA) -> MixBacktest:
    """
    Raises:
        sources.DataError, TradingNotConfiguredError: When the bars cannot be had.
        ProfileError: If the profile's limits are inconsistent with the environment's.
    """
    effective = effective_settings(settings, profile)
    market = sources.load(settings, symbols(profile), start - timedelta(days=warm_up_days(profile)), end, data)
    result = run_backtest(allocations(profile, market.instruments), market.bars, BacktestConfig(
        start=start, end=end, risk_limits=risk_limits(effective), sizing=sizing_policy(effective),
        resume_after_kill=RESUME_AFTER_KILL, cash_sleeve=market.instruments[CASH_SLEEVE]))
    return MixBacktest(result, summarize(daily_returns(result.equity_curve).over(market.hurdle())), market)
