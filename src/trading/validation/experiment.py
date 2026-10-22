"""
NOCTURNA Trading System - Validation experiments run through the engine.

Every number here comes from the same engine, simulated broker, costs, risk
gate and sizing as paper trading:

* ``run_trials`` backtests every parameter set of a candidate over the whole
  period. A trial's return on a day depends only on data before it, so any
  slice of a trial's returns is what that parameter set would have earned.
* ``walk_forward`` picks, at the start of each test window, the parameter set
  with the best Sharpe ratio over the preceding training window, using only
  returns already realized, and then *trades* the chosen sets through the
  engine, so switching parameters costs what it would cost live.
* ``cost_stress`` repeats the walk-forward with costs multiplied.
* ``neighbours`` compares the selected parameter set with the sets one grid
  step away: an edge that exists only at one point of the grid is noise.
* ``regimes`` splits out-of-sample returns by calendar year and by market
  state (trend and volatility of a reference symbol).

Candidate strategies must decide from bar history alone (not from the
positions they are shown), so a parameter set can be switched in at any time.

With a ``hurdle`` (T-bills' return per day), every return series carries it,
and Sharpe ratios, the walk-forward's choices and the overfitting statistics
are computed in excess of it.
"""

from __future__ import annotations

import calendar
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import product
from typing import Any

import numpy as np

from ..backtest import BacktestConfig, BacktestResult, run_backtest
from ..domain import Bar, Instrument
from ..engine import Allocation
from ..strategy import Strategy, StrategyContext, common_execution, consult, needed_history
from .stats import Returns, daily_returns, sharpe_ratio, summarize

_NAME = re.compile(r"[a-z0-9-]{1,16}")


@dataclass(frozen=True)
class Candidate:
    """A strategy family to validate: how to build it, and the parameter grid searched."""

    name: str
    factory: Callable[[Mapping[str, Any]], Strategy]
    grid: Mapping[str, Sequence[Any]]
    #: Symbol whose trend and volatility define the market regimes (default: the first instrument)
    regime_symbol: str | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"candidate name {self.name!r} must match {_NAME.pattern}")
        if any(not values for values in self.grid.values()):
            raise ValueError("every grid parameter needs at least one value")

    def parameter_sets(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [dict(zip(keys, values, strict=True)) for values in product(*(self.grid[k] for k in keys))]


@dataclass(frozen=True)
class Trial:
    params: dict[str, Any]
    result: BacktestResult
    returns: Returns


def _run(strategy: Strategy, bars: Mapping[str, Sequence[Bar]], config: BacktestConfig) -> BacktestResult:
    return run_backtest([Allocation(strategy, Decimal("1"))], bars, config)


def _returns(result: BacktestResult, hurdle: Mapping[date, float] | None) -> Returns:
    returns = daily_returns(result.equity_curve)
    return returns if hurdle is None else returns.over(hurdle)


def run_trials(candidate: Candidate, bars: Mapping[str, Sequence[Bar]], config: BacktestConfig,
               hurdle: Mapping[date, float] | None = None) -> list[Trial]:
    trials = []
    for params in candidate.parameter_sets():
        result = _run(candidate.factory(params), bars, config)
        trials.append(Trial(params, result, _returns(result, hurdle)))
    return trials


def returns_matrix(trials: Sequence[Trial]) -> tuple[tuple[date, ...], np.ndarray]:
    """Trials' returns in excess of the hurdle on the dates all of them have (rows: dates, columns: trials)."""
    common = sorted(set.intersection(*(set(t.returns.dates) for t in trials)))
    columns = []
    for t in trials:
        index = {d: i for i, d in enumerate(t.returns.dates)}
        columns.append(t.returns.excess[[index[d] for d in common]])
    return tuple(common), np.column_stack(columns) if columns else np.empty((0, 0))


# -- walk-forward ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardConfig:
    """Windows in calendar months, so a year is a year across leap years."""

    train_months: int = 36
    test_months: int = 12
    #: Anchored: every training window starts at the beginning; otherwise it rolls
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.train_months < 1 or self.test_months < 1:
            raise ValueError("train and test windows must be at least a month")


def add_months(moment: datetime, months: int) -> datetime:
    """The same day and time ``months`` later, clamped to the end of shorter months."""
    total = moment.month - 1 + months
    year, month = moment.year + total // 12, total % 12 + 1
    return moment.replace(year=year, month=month, day=min(moment.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Window:
    train_start: datetime
    test_start: datetime
    test_end: datetime
    choice: int  # index into the candidate's parameter sets
    train_sharpe: float  # annualized, of the chosen set over the training window


def windows(start: datetime, end: datetime, config: WalkForwardConfig) -> list[tuple[datetime, datetime, datetime]]:
    """(train start, test start, test end) for each window; the last test window may be short."""
    out = []
    step = 0
    test_start = add_months(start, config.train_months)
    while test_start < end:
        train_start = start if config.anchored else add_months(test_start, -config.train_months)
        test_end = add_months(start, config.train_months + (step + 1) * config.test_months)
        out.append((train_start, test_start, min(test_end, end)))
        step += 1
        test_start = test_end
    return out


def select(trials: Sequence[Trial], start: datetime, end: datetime) -> tuple[int, float]:
    """The trial with the best Sharpe ratio on days in [start, end), and that Sharpe (annualized)."""
    best, best_sharpe = 0, -math.inf
    for i, trial in enumerate(trials):
        window = trial.returns.between(start.astimezone(UTC).date(), end.astimezone(UTC).date())
        sr = sharpe_ratio(window.excess) * math.sqrt(trial.returns.periods_per_year)
        if sr > best_sharpe:
            best, best_sharpe = i, sr
    return best, best_sharpe


class SwitchingStrategy(Strategy):
    """
    Trades the parameter set chosen for the current walk-forward window. Every
    set is consulted on every bar, as the engine would consult it on its own
    (once its history suffices), so that a set switched in is up to date.
    """

    def __init__(self, name: str, strategies: Sequence[Strategy], schedule: Sequence[tuple[datetime, int]]) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError(f"strategy name {name!r} must match {_NAME.pattern}")
        self.name = name
        self._strategies = list(strategies)
        self._schedule = sorted(schedule)
        instruments: dict[str, Instrument] = {}
        for strategy in self._strategies:
            for instrument in strategy.instruments:
                instruments.setdefault(instrument.symbol, instrument)
        self.instruments = tuple(instruments.values())
        self.execution = common_execution(self._strategies)
        self.lookback = max(s.lookback for s in self._strategies)
        # Each set waits for its own history (a blend may trade before its slowest component)
        self.min_history = min(needed_history(s) for s in self._strategies)
        self._last: dict[int, Mapping[str, float]] = {}

    def active(self, now: datetime) -> int | None:
        current = None
        for start, index in self._schedule:
            if start <= now:
                current = index
        return current

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        for i, strategy in enumerate(self._strategies):
            weights = consult(strategy, ctx)
            if weights is not None:
                self._last[i] = weights
        active = self.active(ctx.now)
        return {} if active is None else self._last.get(active, {})


@dataclass(frozen=True)
class WalkForward:
    windows: tuple[Window, ...]
    result: BacktestResult
    returns: Returns
    strategy: SwitchingStrategy = field(repr=False)


def walk_forward(candidate: Candidate, trials: Sequence[Trial], bars: Mapping[str, Sequence[Bar]],
                 config: BacktestConfig, wf: WalkForwardConfig,
                 hurdle: Mapping[date, float] | None = None) -> WalkForward:
    """
    Raises:
        ValueError: If the period is too short for one training window and a test window.
    """
    spans = windows(config.start, config.end, wf)
    if not spans:
        raise ValueError("the period is shorter than one training window")
    chosen = []
    for train_start, test_start, test_end in spans:
        index, sharpe = select(trials, train_start, test_start)
        chosen.append(Window(train_start, test_start, test_end, index, sharpe))
    params = candidate.parameter_sets()
    strategy = SwitchingStrategy(f"wf-{candidate.name}"[:16].rstrip("-"),
                                 [candidate.factory(p) for p in params],
                                 [(w.test_start, w.choice) for w in chosen])
    result = _run(strategy, bars, replace(config, start=chosen[0].test_start))
    return WalkForward(tuple(chosen), result, _returns(result, hurdle), strategy)


def cost_stress(candidate: Candidate, walk: WalkForward, bars: Mapping[str, Sequence[Bar]],
                config: BacktestConfig, multiples: Sequence[float] = (2.0, 3.0),
                hurdle: Mapping[date, float] | None = None) -> dict[float, Returns]:
    """The walk-forward's out-of-sample returns with every cost multiplied."""
    out = {}
    params = candidate.parameter_sets()
    for multiple in multiples:
        stressed = replace(config, start=walk.windows[0].test_start,
                           costs={k: v.scaled(Decimal(repr(multiple))) for k, v in config.costs.items()})
        strategy = SwitchingStrategy(walk.strategy.name, [candidate.factory(p) for p in params],
                                     [(w.test_start, w.choice) for w in walk.windows])
        out[multiple] = _returns(_run(strategy, bars, stressed), hurdle)
    return out


# -- parameter stability -------------------------------------------------------------------

@dataclass(frozen=True)
class Neighbourhood:
    selected: int
    selected_sharpe: float  # annualized, full period
    neighbour_sharpes: dict[int, float]

    @property
    def ratio(self) -> float | None:
        """Median neighbour Sharpe over the selected Sharpe (None without neighbours)."""
        if not self.neighbour_sharpes or self.selected_sharpe <= 0:
            return None
        return float(np.median(list(self.neighbour_sharpes.values()))) / self.selected_sharpe


def neighbours(candidate: Candidate, sharpes: Sequence[float], selected: int) -> Neighbourhood:
    """Parameter sets that differ from the selected one by one step of one parameter."""
    params = candidate.parameter_sets()
    keys = list(candidate.grid)
    position = {k: list(candidate.grid[k]) for k in keys}

    def steps(p: Mapping[str, Any]) -> list[int]:
        return [position[k].index(p[k]) for k in keys]

    target = steps(params[selected])
    near = {}
    for i, p in enumerate(params):
        diff = [abs(a - b) for a, b in zip(steps(p), target, strict=True)]
        if sum(diff) == 1:
            near[i] = sharpes[i]
    return Neighbourhood(selected, sharpes[selected], near)


# -- regimes --------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeStats:
    label: str
    periods: int
    total_return: float
    sharpe: float  # annualized, in excess of the hurdle


def _stats(label: str, returns: Returns, mask: Sequence[bool]) -> RegimeStats:
    keep = [i for i, m in enumerate(mask) if m]
    values = returns.values[keep]
    sr = sharpe_ratio(returns.excess[keep]) * math.sqrt(returns.periods_per_year) if len(values) > 1 else 0.0
    return RegimeStats(label, len(values), float(np.prod(1 + values) - 1) if len(values) else 0.0, sr)


def regimes(returns: Returns, reference: Sequence[Bar], delay: timedelta,
            trend_days: int = 126, vol_days: int = 21) -> list[RegimeStats]:
    """
    Out-of-sample returns by calendar year, by the reference symbol's trend
    (return over the previous ``trend_days`` bars, known at the time) and by its
    volatility (previous ``vol_days`` bars, above or below the period median).
    """
    years = sorted({d.year for d in returns.dates})
    out = [_stats(str(y), returns, [d.year == y for d in returns.dates]) for y in years]
    closes = np.array([b.close for b in reference], dtype=float)
    known = [(b.end + delay).astimezone(UTC).date() for b in reference]
    trend, vol = {}, {}
    log_returns = np.diff(np.log(closes))
    for i in range(len(reference)):
        if i >= trend_days:
            trend[known[i]] = closes[i] / closes[i - trend_days] - 1
        if i >= vol_days:
            vol[known[i]] = float(log_returns[i - vol_days:i].std(ddof=1))
    # A day's return is earned with the market state known when it began: the previous mark
    previous = [None, *returns.dates[:-1]]
    state_trend = [trend.get(p) if p else None for p in previous]
    state_vol = [vol.get(p) if p else None for p in previous]
    known_vols = [v for v in state_vol if v is not None]
    median = float(np.median(known_vols)) if known_vols else 0.0
    out.append(_stats("rising market", returns, [t is not None and t > 0 for t in state_trend]))
    out.append(_stats("falling market", returns, [t is not None and t <= 0 for t in state_trend]))
    out.append(_stats("calm market", returns, [v is not None and v <= median for v in state_vol]))
    out.append(_stats("volatile market", returns, [v is not None and v > median for v in state_vol]))
    return out


def full_period_sharpes(trials: Sequence[Trial]) -> list[float]:
    """Annualized Sharpe ratio of every trial over the whole period."""
    return [summarize(t.returns).sharpe for t in trials]
