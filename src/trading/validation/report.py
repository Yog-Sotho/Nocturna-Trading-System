"""
NOCTURNA Trading System - Validation report (tear sheet).

``to_markdown`` is for reading; ``to_dict`` is JSON-safe for tools and the
future dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from .experiment import Neighbourhood, RegimeStats
from .gates import GateResult
from .overfitting import OverfittingResult
from .stats import DeflatedSharpe, Summary


@dataclass(frozen=True)
class ValidationReport:
    candidate: str
    start: datetime
    end: datetime
    trials: list[dict[str, Any]]  # params and full-period annualized Sharpe of every trial
    selected: dict[str, Any]
    selected_summary: Summary
    probabilistic_sharpe: float  # P(true Sharpe > 0) for the selected set
    deflated: DeflatedSharpe
    overfitting: OverfittingResult | None
    cross_validation: dict[str, Any] | None
    walk_forward: list[dict[str, Any]]
    out_of_sample: Summary
    yearly_and_regimes: list[RegimeStats]
    cost_stress: dict[float, Summary]
    neighbourhood: Neighbourhood
    kill_switch_trips: int
    gates: list[GateResult]
    passed: bool
    verdict: str
    settings: dict[str, Any]

    # -- JSON ---------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        cv = self.cross_validation
        plain: dict[str, Any] = _plain({
            "candidate": self.candidate, "start": self.start, "end": self.end,
            "verdict": self.verdict, "passed": self.passed,
            "gates": [g.__dict__ for g in self.gates],
            "selected": {"params": self.selected, **self.selected_summary.as_dict()},
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "deflated_sharpe": self.deflated.__dict__,
            "overfitting": None if self.overfitting is None else {
                "pbo": self.overfitting.pbo, "splits": self.overfitting.splits,
                "degradation_slope": self.overfitting.degradation_slope,
                "probability_of_loss": self.overfitting.probability_of_loss},
            "cross_validation": cv,
            "out_of_sample": self.out_of_sample.as_dict(),
            "walk_forward": self.walk_forward,
            "cost_stress": {f"{m:g}x": s.as_dict() for m, s in self.cost_stress.items()},
            "neighbourhood": {"ratio": self.neighbourhood.ratio,
                              "selected_sharpe": self.neighbourhood.selected_sharpe,
                              "neighbours": list(self.neighbourhood.neighbour_sharpes.values())},
            "regimes": [r.__dict__ for r in self.yearly_and_regimes],
            "kill_switch_trips": self.kill_switch_trips,
            "trials": self.trials,
            "settings": self.settings,
        })
        return plain

    # -- Markdown -----------------------------------------------------------------------------

    def to_markdown(self) -> str:
        lines = [
            f"# Validation: {self.candidate}",
            "",
            f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}. **Verdict: {self.verdict}.**",
            "",
            "Every figure comes from the trading engine with the simulated broker, the cost model, "
            "the risk gate and the position sizing used in paper trading.",
            "",
            "## Promotion gates",
            "",
            "| Gate | Result | Value | Threshold | Note |",
            "|---|---|---|---|---|",
        ]
        for g in self.gates:
            result = "pass" if g.passed else "FAIL" if g.passed is False else "n/a"
            lines.append(f"| {g.name} | {result} | {_fmt(g.value)} | {g.threshold} | {g.note} |")

        oos, sel = self.out_of_sample, self.selected_summary
        lines += [
            "",
            "## Out of sample (walk-forward)",
            "",
            _summary_table({"walk-forward": oos, f"selected set, full period {_params(self.selected)}": sel}),
            "",
            f"Kill switch trips out of sample: {self.kill_switch_trips}.",
            "",
            "| Test window | Parameters | Training Sharpe |",
            "|---|---|---|",
        ]
        for w in self.walk_forward:
            lines.append(f"| {w['test_start']:%Y-%m-%d} to {w['test_end']:%Y-%m-%d} | {_params(w['params'])} "
                         f"| {_fmt(w['train_sharpe'])} |")

        d = self.deflated
        lines += [
            "",
            "## Selection bias and overfitting",
            "",
            f"- Trials: {len(self.trials)}, effectively independent: {d.effective_trials:.1f}.",
            f"- Selected set's Sharpe per period: {d.sharpe:.4f}; best expected from unskilled trials: "
            f"{d.threshold:.4f}.",
            f"- Probabilistic Sharpe (true Sharpe > 0): {self.probabilistic_sharpe:.3f}; "
            f"deflated Sharpe: {d.dsr:.3f}.",
        ]
        if self.overfitting is not None:
            o = self.overfitting
            lines += [
                f"- Probability of backtest overfitting: {o.pbo:.3f} over {o.splits} CSCV splits; "
                f"in-sample winner lost out of sample in {o.probability_of_loss:.1%} of splits; "
                f"degradation slope {o.degradation_slope:.2f}.",
            ]
        if self.cross_validation is not None:
            s = np.array(self.cross_validation["sharpes"])
            lines += [f"- Combinatorial purged cross-validation: {len(s)} out-of-sample paths, annualized Sharpe "
                      f"mean {s.mean():.2f}, worst {s.min():.2f}, best {s.max():.2f}."]

        n = self.neighbourhood
        lines += [
            "",
            "## Costs and parameter stability",
            "",
            _summary_table({f"{m:g}x costs": s for m, s in sorted(self.cost_stress.items())}),
            "",
            f"Neighbouring parameter sets: {len(n.neighbour_sharpes)}, median Sharpe "
            f"{_fmt(float(np.median(list(n.neighbour_sharpes.values()))) if n.neighbour_sharpes else None)} "
            f"against {_fmt(n.selected_sharpe)} for the selected set.",
            "",
            "## Years and market regimes (out of sample)",
            "",
            "| Period | Days | Return | Sharpe |",
            "|---|---|---|---|",
        ]
        for r in self.yearly_and_regimes:
            lines.append(f"| {r.label} | {r.periods} | {r.total_return:.2%} | {r.sharpe:.2f} |")
        lines += ["", "## All trials (full period)", "", "| Parameters | Sharpe |", "|---|---|"]
        for t in self.trials:
            lines.append(f"| {_params(t['params'])} | {_fmt(t['sharpe'])} |")
        lines += ["", "## Settings", "", "```", *(f"{k}: {v}" for k, v in self.settings.items()), "```", ""]
        return "\n".join(lines)


def _params(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "(none)"


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.3f}"


def _summary_table(rows: dict[str, Summary]) -> str:
    out = ["| | Total return | CAGR | Volatility | Sharpe | Sortino | Max drawdown |",
           "|---|---|---|---|---|---|---|"]
    for label, s in rows.items():
        out.append(f"| {label} | {s.total_return:.2%} | {s.cagr:.2%} | {s.volatility:.2%} | {s.sharpe:.2f} "
                   f"| {s.sortino:.2f} | {s.max_drawdown:.2%} |")
    return "\n".join(out)


def _plain(value: Any) -> Any:
    """JSON-safe copy: dates as ISO strings, numpy scalars as floats, non-finite floats as None."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
