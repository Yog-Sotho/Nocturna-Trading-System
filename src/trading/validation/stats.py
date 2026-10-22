"""
NOCTURNA Trading System - Performance statistics for validation.

Sharpe ratios here are per period (not annualized) unless a name says
otherwise, as the Probabilistic and Deflated Sharpe ratios require.

* Probabilistic Sharpe Ratio (Bailey and López de Prado, "The Sharpe Ratio
  Efficient Frontier", Journal of Risk, 2012): the probability that the true
  Sharpe ratio exceeds a benchmark, given the sample length, skewness and
  kurtosis of the returns.
* Deflated Sharpe Ratio (Bailey and López de Prado, "The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting and
  Non-Normality", Journal of Portfolio Management, 2014): the PSR against the
  Sharpe ratio expected from the best of N unskilled trials,

      SR0 = sqrt(V[SR_n]) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))),

  with g the Euler-Mascheroni constant. N is the number of *independent*
  trials, estimated from M correlated ones as N = rho + (1 - rho) * M with rho
  the average correlation between trials (same paper, appendix A.3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Returns:
    """Simple returns per period, labelled by the UTC date the period ended."""

    dates: tuple[date, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must have the same length")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def periods_per_year(self) -> float:
        """Observed periods per year (about 252 for equities, 365 for crypto)."""
        if len(self.dates) < 2:
            return 252.0
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        return (len(self.dates) - 1) / years if years > 0 else 252.0

    def between(self, start: date, end: date) -> Returns:
        """The periods ending in [start, end)."""
        keep = [i for i, d in enumerate(self.dates) if start <= d < end]
        return Returns(tuple(self.dates[i] for i in keep), self.values[keep])


def daily_returns(curve: Sequence[tuple[datetime, Decimal | float]]) -> Returns:
    """
    Returns between the last equity marks of consecutive UTC days. Daily bars
    become final just after midnight (UTC for crypto, New York for equities,
    which is early morning UTC), so each day's mark covers the previous day.
    """
    last: dict[date, float] = {}
    for timestamp, equity in curve:
        last[timestamp.astimezone(UTC).date()] = float(equity)
    days = sorted(last)
    values = np.array([last[d] for d in days], dtype=float)
    if len(values) < 2:
        return Returns((), np.array([], dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(values[:-1] > 0, values[1:] / values[:-1] - 1, 0.0)
    return Returns(tuple(days[1:]), returns)


# -- moments and ratios ------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Per-period Sharpe ratio of excess returns (sample standard deviation)."""
    excess = np.asarray(returns, dtype=float) - risk_free
    if len(excess) < 2:
        return 0.0
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd) if sd > 0 else 0.0


def sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """Per-period Sharpe ratio of every column."""
    sd = matrix.std(axis=0, ddof=1)
    mean = matrix.mean(axis=0)
    sharpe: np.ndarray = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    return sharpe


def skewness(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std()
    return float(((r - r.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0


def kurtosis(returns: np.ndarray) -> float:
    """Pearson kurtosis (3 for a normal distribution), as the PSR formula uses."""
    r = np.asarray(returns, dtype=float)
    sd = r.std()
    return float(((r - r.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0


def probabilistic_sharpe_ratio(sharpe: float, benchmark: float, n: int, skew: float = 0.0,
                               kurt: float = 3.0) -> float:
    """
    Probability that the true per-period Sharpe ratio exceeds ``benchmark``,
    for an estimate from ``n`` returns with the given skewness and Pearson kurtosis.
    """
    if n < 2:
        return 0.0
    variance = 1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2
    if variance <= 0:  # only for degenerate samples
        return 1.0 if sharpe > benchmark else 0.0
    return _NORMAL.cdf((sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(variance))


def expected_max_sharpe(n_trials: float, variance: float, mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe estimates with this mean and variance."""
    if n_trials < 2 or variance <= 0:
        return mean
    z = ((1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
         + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e)))
    return mean + math.sqrt(variance) * z


def effective_trials(matrix: np.ndarray) -> float:
    """
    Independent trials implied by M correlated ones: rho + (1 - rho) * M, with
    rho the average pairwise correlation of the trials' returns (clipped to
    [0, 1]; trials without variance are left out of rho).
    """
    m = matrix.shape[1]
    if m < 2:
        return float(m)
    varying = matrix[:, matrix.std(axis=0) > 0]
    if varying.shape[1] < 2:
        return float(m)
    corr = np.corrcoef(varying, rowvar=False)
    k = corr.shape[0]
    rho = (corr.sum() - k) / (k * (k - 1))
    rho = min(1.0, max(0.0, float(rho)))
    return float(rho + (1 - rho) * m)


@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe: float  # per period, of the selected strategy
    threshold: float  # SR0: expected best of the trials without skill
    effective_trials: float
    trial_variance: float
    dsr: float


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes: np.ndarray, n_trials: float) -> DeflatedSharpe:
    """
    DSR of the selected strategy's returns, given every trial's per-period
    Sharpe ratio and the number of independent trials.
    """
    variance = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr = sharpe_ratio(returns)
    threshold = expected_max_sharpe(n_trials, variance)
    dsr = probabilistic_sharpe_ratio(sr, threshold, len(returns), skewness(returns), kurtosis(returns))
    return DeflatedSharpe(sr, threshold, n_trials, variance, dsr)


# -- summary metrics -------------------------------------------------------------------------

def max_drawdown(returns: np.ndarray) -> float:
    """Deepest fall of the compounded curve from its running peak (zero or negative)."""
    if len(returns) == 0:
        return 0.0
    wealth = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return float(min(0.0, (wealth / peak - 1).min()))


@dataclass(frozen=True)
class Summary:
    total_return: float
    cagr: float
    volatility: float  # annualized
    sharpe: float  # annualized
    sortino: float  # annualized
    max_drawdown: float
    periods: int

    def as_dict(self) -> dict[str, float]:
        return {"total_return": self.total_return, "cagr": self.cagr, "volatility": self.volatility,
                "sharpe": self.sharpe, "sortino": self.sortino, "max_drawdown": self.max_drawdown,
                "periods": float(self.periods)}


def summarize(returns: Returns) -> Summary:
    r = returns.values
    if len(r) < 2:
        return Summary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(r))
    per_year = returns.periods_per_year
    growth = float(np.prod(1 + r))
    years = len(r) / per_year
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    return Summary(
        total_return=growth - 1,
        cagr=growth ** (1 / years) - 1 if growth > 0 and years > 0 else -1.0,
        volatility=float(r.std(ddof=1) * math.sqrt(per_year)),
        sharpe=sharpe_ratio(r) * math.sqrt(per_year),
        sortino=float(r.mean() / downside * math.sqrt(per_year)) if downside > 0 else 0.0,
        max_drawdown=max_drawdown(r),
        periods=len(r),
    )
