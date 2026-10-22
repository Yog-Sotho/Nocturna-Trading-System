"""
NOCTURNA Trading System - Validating one candidate.

``validate`` runs the whole pipeline: the parameter grid through the engine,
the Deflated Sharpe ratio of the best parameter set, the probability of
backtest overfitting, combinatorial purged cross-validation, the walk-forward
traded through the engine, cost stress, parameter stability and regimes, and
returns the report with the promotion gates decided. With a hurdle (T-bills),
every Sharpe-based gate is measured in excess of it and the cost-stress gate
asks for more than the hurdle earned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ..backtest import BacktestConfig
from ..domain import Bar
from .experiment import (
    Candidate,
    WalkForwardConfig,
    cost_stress,
    full_period_sharpes,
    neighbours,
    regimes,
    returns_matrix,
    run_trials,
    walk_forward,
)
from .gates import GateResult, PromotionGates, verdict
from .overfitting import combinatorial_purged_cv, probability_of_backtest_overfitting
from .report import ValidationReport
from .stats import (
    deflated_sharpe_ratio,
    effective_trials,
    kurtosis,
    probabilistic_sharpe_ratio,
    sharpe_columns,
    sharpe_ratio,
    skewness,
    summarize,
)


@dataclass(frozen=True)
class ValidationConfig:
    backtest: BacktestConfig
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    pbo_blocks: int = 16
    cv_groups: int = 6
    cv_test_groups: int = 2
    #: Training days dropped after each test group in cross-validation
    cv_embargo: int = 5
    stress_multiples: tuple[float, ...] = (2.0, 3.0)
    gates: PromotionGates = field(default_factory=PromotionGates)
    #: The hurdle's return per day (UTC date the day's mark falls on); None: zero
    hurdle: Mapping[date, float] | None = None
    hurdle_label: str = "T-bills"


def validate(candidate: Candidate, bars: Mapping[str, Sequence[Bar]], config: ValidationConfig) -> ValidationReport:
    """
    Raises:
        ValueError: If the period is too short for the walk-forward or the statistics.
    """
    backtest, gates, hurdle = config.backtest, config.gates, config.hurdle
    trials = run_trials(candidate, bars, backtest, hurdle)
    dates, matrix = returns_matrix(trials)
    if len(dates) < 2 * config.pbo_blocks:
        raise ValueError(f"only {len(dates)} common return days; the period is too short to validate")
    params = candidate.parameter_sets()
    sharpes = full_period_sharpes(trials)
    selected = int(np.argmax(sharpes))
    chosen = matrix[:, selected]

    n_effective = effective_trials(matrix)
    deflated = deflated_sharpe_ratio(chosen, sharpe_columns(matrix), n_effective)
    psr = probabilistic_sharpe_ratio(sharpe_ratio(chosen), 0.0, len(chosen), skewness(chosen), kurtosis(chosen))
    overfitting = probability_of_backtest_overfitting(matrix, config.pbo_blocks) if len(trials) > 1 else None
    cv = (combinatorial_purged_cv(matrix, config.cv_groups, config.cv_test_groups, embargo=config.cv_embargo)
          if len(trials) > 1 else None)

    walk = walk_forward(candidate, trials, bars, backtest, config.walk_forward, hurdle)
    oos = summarize(walk.returns)
    stressed = cost_stress(candidate, walk, bars, backtest, config.stress_multiples, hurdle)
    stressed_summary = {m: summarize(r) for m, r in stressed.items()}
    neighbourhood = neighbours(candidate, sharpes, selected)
    reference = candidate.regime_symbol or next(iter(candidate.factory(params[0]).instruments)).symbol
    delay = backtest.delays[next(i for i in candidate.factory(params[0]).instruments
                                 if i.symbol == reference).asset_class]
    regime_stats = regimes(walk.returns, bars.get(reference, ()), delay)

    results = [
        GateResult("out-of-sample Sharpe", oos.sharpe >= gates.min_oos_sharpe, oos.sharpe,
                   f">= {gates.min_oos_sharpe:g}", "walk-forward, traded through the engine, net of costs"),
        GateResult("deflated Sharpe", deflated.dsr >= gates.min_deflated_sharpe, deflated.dsr,
                   f">= {gates.min_deflated_sharpe:g}", f"{n_effective:.1f} effective trials of {len(trials)}"),
        GateResult("backtest overfitting", None if overfitting is None else overfitting.pbo <= gates.max_pbo,
                   None if overfitting is None else overfitting.pbo, f"<= {gates.max_pbo:g}",
                   "single parameter set: nothing was selected" if overfitting is None else
                   f"CSCV, {overfitting.splits} splits"),
    ]
    stress = stressed_summary.get(gates.stress_multiple)
    results.append(GateResult(
        f"profitable at {gates.stress_multiple:g}x costs",
        None if stress is None else stress.total_return > stress.hurdle_return and stress.sharpe > 0,
        None if stress is None else stress.total_return,
        f"> {config.hurdle_label} total return" if hurdle is not None else "> 0 total return",
        "not run" if stress is None else
        f"{config.hurdle_label} returned {stress.hurdle_return:.2%}" if hurdle is not None else ""))
    ratio = neighbourhood.ratio
    results.append(GateResult(
        "parameter stability", None if ratio is None else ratio >= gates.min_neighbour_ratio, ratio,
        f">= {gates.min_neighbour_ratio:g}",
        "no neighbouring parameter sets" if not neighbourhood.neighbour_sharpes
        else "selected Sharpe not positive" if ratio is None else "median neighbour / selected Sharpe"))
    trips = int(walk.result.metrics()["kill_switch_trips"])
    results.append(GateResult(
        "maximum drawdown", oos.max_drawdown >= -gates.max_drawdown, oos.max_drawdown,
        f">= -{gates.max_drawdown:.0%}", f"out of sample; kill switch tripped {trips} time(s)"))
    results.append(GateResult("paper track record", None, None, ">= 60 trading days",
                              "decided by paper trading (Phase 9)"))
    passed, summary = verdict(results)

    return ValidationReport(
        candidate=candidate.name,
        start=backtest.start, end=backtest.end,
        trials=[{"params": p, "sharpe": s} for p, s in zip(params, sharpes, strict=True)],
        selected=params[selected],
        selected_summary=summarize(trials[selected].returns),
        probabilistic_sharpe=psr,
        deflated=deflated,
        overfitting=overfitting,
        cross_validation=None if cv is None else {
            "paths": len(cv.path_sharpes), "splits": cv.splits,
            "sharpes": [s * math.sqrt(trials[selected].returns.periods_per_year) for s in cv.path_sharpes]},
        walk_forward=[{"train_start": w.train_start, "test_start": w.test_start, "test_end": w.test_end,
                       "params": params[w.choice], "train_sharpe": w.train_sharpe} for w in walk.windows],
        out_of_sample=oos,
        yearly_and_regimes=regime_stats,
        cost_stress={1.0: oos, **stressed_summary},
        neighbourhood=neighbourhood,
        kill_switch_trips=trips,
        gates=results,
        passed=passed,
        verdict=summary,
        settings={
            "sharpe_in_excess_of": config.hurdle_label if hurdle is not None else "zero",
            "costs": {k.value: {"fee_bps": str(v.fee_bps), "slippage_bps": str(v.slippage_bps)}
                      for k, v in backtest.costs.items()},
            "resume_after_kill_days": (backtest.resume_after_kill / timedelta(days=1)
                                       if backtest.resume_after_kill is not None else None),
            "walk_forward": {"train_months": config.walk_forward.train_months,
                             "test_months": config.walk_forward.test_months,
                             "anchored": config.walk_forward.anchored},
            "max_drawdown_limit": str(backtest.risk_limits.max_drawdown),
            "target_volatility": (str(backtest.sizing.target_volatility)
                                  if backtest.sizing.target_volatility is not None else None),
        },
    )
