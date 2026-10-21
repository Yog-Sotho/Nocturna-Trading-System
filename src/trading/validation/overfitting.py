"""
NOCTURNA Trading System - Backtest overfitting.

Both methods take the T x N matrix of per-period returns of N trials (one
column per parameter set) on common dates, and the selection rule "best
Sharpe ratio in sample".

* **Probability of backtest overfitting** by combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado and Zhu, "The Probability
  of Backtest Overfitting", Journal of Computational Finance, 2017): the
  rows are cut into S equal blocks; for each of the C(S, S/2) ways to pick
  half of them as the in-sample set (the rest is out of sample) the best
  in-sample trial is ranked out of sample, w = rank / (N + 1), and
  l = ln(w / (1 - w)). PBO is the share of splits with l <= 0: the in-sample
  winner did no better than the median trial out of sample.
* **Combinatorial purged cross-validation** (López de Prado, "Advances in
  Financial Machine Learning", 2018): the rows are cut into N groups and every
  choice of k groups is a test set. Training rows within ``purge`` rows before
  and ``embargo`` rows after a test group are dropped, so serial correlation
  cannot leak across the boundary. Each group is tested in C(N-1, k-1) splits,
  so the selected trials' test returns assemble into that many complete
  out-of-sample paths, giving a distribution of out-of-sample Sharpe ratios
  rather than a single number.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from .stats import sharpe_columns, sharpe_ratio


@dataclass(frozen=True)
class OverfittingResult:
    pbo: float
    logits: np.ndarray  # one per split
    in_sample: np.ndarray  # the in-sample winner's per-period Sharpe ratio, per split
    out_of_sample: np.ndarray  # the same trial's out-of-sample Sharpe ratio
    degradation_slope: float  # OLS slope of out-of-sample on in-sample Sharpe
    probability_of_loss: float  # share of splits where the winner lost out of sample
    splits: int


def _block_sharpes(sums: np.ndarray, squares: np.ndarray, counts: np.ndarray, blocks) -> np.ndarray:
    """Per-column Sharpe ratio of the rows in ``blocks``, from per-block sums."""
    n = counts[list(blocks)].sum()
    total = sums[list(blocks)].sum(axis=0)
    mean = total / n
    var = (squares[list(blocks)].sum(axis=0) - n * mean ** 2) / (n - 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 1e-15)


def _relative_rank(values: np.ndarray, index: int) -> float:
    """Rank of ``values[index]`` from 1 (worst) to N (best), ties averaged, over N + 1."""
    value = values[index]
    rank = (values < value).sum() + ((values == value).sum() + 1) / 2
    return rank / (len(values) + 1)


def probability_of_backtest_overfitting(matrix: np.ndarray, blocks: int = 16) -> OverfittingResult:
    """
    PBO by CSCV with ``blocks`` (even) equal blocks of rows; the oldest rows
    that do not fill a block are left out.

    Raises:
        ValueError: With fewer than two trials, an odd block count, or too few rows.
    """
    rows, trials = matrix.shape
    if trials < 2:
        raise ValueError("PBO needs at least two trials")
    if blocks < 2 or blocks % 2:
        raise ValueError("the number of blocks must be even and at least 2")
    size = rows // blocks
    if size < 2:
        raise ValueError(f"{rows} rows are too few for {blocks} blocks")
    data = matrix[rows - size * blocks:]
    parts = data.reshape(blocks, size, trials)
    sums, squares = parts.sum(axis=1), (parts ** 2).sum(axis=1)
    counts = np.full(blocks, size)

    logits, in_sample, out_of_sample = [], [], []
    for chosen in combinations(range(blocks), blocks // 2):
        rest = tuple(b for b in range(blocks) if b not in chosen)
        is_sharpe = _block_sharpes(sums, squares, counts, chosen)
        oos_sharpe = _block_sharpes(sums, squares, counts, rest)
        best = int(np.argmax(is_sharpe))
        w = _relative_rank(oos_sharpe, best)
        logits.append(np.log(w / (1 - w)))
        in_sample.append(is_sharpe[best])
        out_of_sample.append(oos_sharpe[best])
    logits_, is_, oos_ = np.array(logits), np.array(in_sample), np.array(out_of_sample)
    slope = float(np.polyfit(is_, oos_, 1)[0]) if np.ptp(is_) > 0 else 0.0
    return OverfittingResult(
        pbo=float((logits_ <= 0).mean()), logits=logits_, in_sample=is_, out_of_sample=oos_,
        degradation_slope=slope, probability_of_loss=float((oos_ < 0).mean()), splits=len(logits_))


@dataclass(frozen=True)
class Split:
    train: np.ndarray  # row indices
    test: np.ndarray
    test_groups: tuple[int, ...]


def purged_splits(rows: int, groups: int, test_groups: int, purge: int = 0, embargo: int = 0) -> list[Split]:
    """
    Every choice of ``test_groups`` of ``groups`` contiguous groups as a test set,
    with ``purge`` training rows dropped before and ``embargo`` after each test group.
    """
    if not 0 < test_groups < groups:
        raise ValueError("test_groups must be between 1 and groups - 1")
    if rows < groups:
        raise ValueError("fewer rows than groups")
    bounds = np.array_split(np.arange(rows), groups)
    splits = []
    for chosen in combinations(range(groups), test_groups):
        test = np.concatenate([bounds[g] for g in chosen])
        excluded = np.zeros(rows, dtype=bool)
        for g in chosen:
            first, last = bounds[g][0], bounds[g][-1]
            excluded[max(0, first - purge):min(rows, last + 1 + embargo)] = True
        splits.append(Split(np.flatnonzero(~excluded), test, chosen))
    return splits


@dataclass(frozen=True)
class CrossValidationResult:
    path_sharpes: np.ndarray  # per-period Sharpe ratio of each out-of-sample path
    paths: np.ndarray  # paths x rows of out-of-sample returns
    selections: np.ndarray  # trial selected in each split
    splits: int


def combinatorial_purged_cv(matrix: np.ndarray, groups: int = 6, test_groups: int = 2, purge: int = 0,
                            embargo: int = 5) -> CrossValidationResult:
    """
    Select the best in-sample trial on every split's training rows and stitch
    the selected trials' test returns into C(groups - 1, test_groups - 1) paths.
    """
    rows = matrix.shape[0]
    splits = purged_splits(rows, groups, test_groups, purge, embargo)
    bounds = np.array_split(np.arange(rows), groups)
    selections = np.array([int(np.argmax(sharpe_columns(matrix[s.train]))) for s in splits])
    n_paths = comb(groups - 1, test_groups - 1)
    paths = np.zeros((n_paths, rows))
    for g, rows_g in enumerate(bounds):
        tested = [i for i, s in enumerate(splits) if g in s.test_groups]
        for path, i in enumerate(tested):
            paths[path, rows_g] = matrix[rows_g, selections[i]]
    sharpes = np.array([sharpe_ratio(p) for p in paths])
    return CrossValidationResult(sharpes, paths, selections, len(splits))
