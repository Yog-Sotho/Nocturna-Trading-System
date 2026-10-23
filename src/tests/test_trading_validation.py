"""
Tests for the validation pipeline: statistics against the published examples,
overfitting estimators against simulations with a known answer, and the
engine-based experiments (walk-forward, cost stress, stability, regimes).
"""

import dataclasses
import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import combinations

import numpy as np
import pytest

from src.tests.trading_fixtures import BIL, IEF, SPY, bar, daily_bars
from src.trading.backtest import BacktestConfig
from src.trading.broker import CostModel
from src.trading.domain import AssetClass
from src.trading.risk import RiskLimits
from src.trading.strategies.benchmark import BuyAndHold
from src.trading.strategy import Strategy, StrategyContext
from src.trading.validation.experiment import (
    Candidate,
    SwitchingStrategy,
    Trial,
    WalkForwardConfig,
    add_months,
    cost_stress,
    neighbours,
    regimes,
    run_trials,
    select,
    walk_forward,
    windows,
)
from src.trading.validation.gates import GateResult, verdict
from src.trading.validation.overfitting import (
    combinatorial_purged_cv,
    probability_of_backtest_overfitting,
    purged_splits,
)
from src.trading.validation.pipeline import ValidationConfig, validate
from src.trading.validation.stats import (
    Returns,
    daily_returns,
    deflated_sharpe_ratio,
    effective_trials,
    expected_max_sharpe,
    hurdle_returns,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_columns,
    sharpe_ratio,
    summarize,
)

D = Decimal


# -- statistics ------------------------------------------------------------------------------

class TestDeflatedSharpe:
    """The numerical example of Bailey and López de Prado (2014), section "A numerical example"."""

    SR = 2.5 / math.sqrt(250)  # annualized 2.5, daily observations
    VARIANCE = 0.5 / 250  # variance of the trials' annualized Sharpe ratios: 1/2

    @pytest.mark.parametrize("trials, skew, kurt, threshold, dsr", [
        (100, -3, 10, 0.1132, 0.9004),
        (46, -3, 10, None, 0.9505),
        (88, 0, 3, None, 0.9505),  # normal returns tolerate more trials
    ])
    def test_published_example(self, trials, skew, kurt, threshold, dsr):
        sr0 = expected_max_sharpe(trials, self.VARIANCE)
        if threshold is not None:
            assert sr0 == pytest.approx(threshold, abs=5e-5)
        assert probabilistic_sharpe_ratio(self.SR, sr0, 1250, skew, kurt) == pytest.approx(dsr, abs=5e-5)

    def test_expected_maximum_matches_simulation(self):
        draws = np.random.default_rng(1).normal(0.0, 1.0, (20000, 100)).max(axis=1)
        assert expected_max_sharpe(100, 1.0) == pytest.approx(draws.mean(), abs=0.05)

    def test_no_selection_no_deflation(self):
        assert expected_max_sharpe(1, 0.5) == 0 and expected_max_sharpe(1.5, 0.5, mean=0.1) == 0.1

    def test_deflation_uses_the_trials(self):
        rng = np.random.default_rng(2)
        returns = rng.normal(0.0005, 0.01, 1500)
        trial_sharpes = rng.normal(0.0, 0.02, 30)
        result = deflated_sharpe_ratio(returns, trial_sharpes, 30)
        assert result.threshold == pytest.approx(expected_max_sharpe(30, float(np.var(trial_sharpes, ddof=1))))
        assert 0 < result.dsr < probabilistic_sharpe_ratio(result.sharpe, 0.0, 1500)


class TestProbabilisticSharpe:
    def test_properties(self):
        assert probabilistic_sharpe_ratio(0.05, 0.05, 1000) == pytest.approx(0.5)
        assert probabilistic_sharpe_ratio(0.05, 0.0, 2000) > probabilistic_sharpe_ratio(0.05, 0.0, 500)
        # Negative skew and fat tails make a positive Sharpe ratio less trustworthy
        assert probabilistic_sharpe_ratio(0.1, 0.0, 500, -2, 9) < probabilistic_sharpe_ratio(0.1, 0.0, 500)
        assert probabilistic_sharpe_ratio(0.1, 0.0, 1) == 0


class TestEffectiveTrials:
    def test_identical_trials_count_once(self):
        column = np.random.default_rng(3).normal(0, 0.01, (500, 1))
        assert effective_trials(np.repeat(column, 10, axis=1)) == pytest.approx(1.0)

    def test_independent_trials_count_fully(self):
        assert effective_trials(np.random.default_rng(4).normal(0, 0.01, (5000, 10))) == pytest.approx(10, abs=0.2)

    def test_correlated_trials_are_in_between(self):
        rng = np.random.default_rng(5)
        common = rng.normal(0, 0.01, (2000, 1))
        n = effective_trials(common + rng.normal(0, 0.01, (2000, 10)))  # correlation 0.5
        assert 5 < n < 6


def _run_hold_bil(bars, config):
    from src.trading.backtest import run_backtest
    from src.trading.engine import Allocation
    return run_backtest([Allocation(BuyAndHold({SPY: 0.0}), D("1"))], bars, dataclasses.replace(
        config, cash_sleeve=BIL, sizing=dataclasses.replace(config.sizing, rebalance_band=D("0"))))


class TestReturnsAndSummary:
    def test_daily_returns_use_the_last_mark_of_each_utc_day(self):
        t = datetime(2026, 3, 2, 0, 15, tzinfo=UTC)
        curve = [(t, D("100")), (t + timedelta(hours=20), D("110")), (t + timedelta(days=1), D("99"))]
        r = daily_returns(curve)
        assert r.dates == (date(2026, 3, 3),) and r.values == pytest.approx([0.99 / 1.1 - 1])

    def test_max_drawdown(self):
        assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
        assert max_drawdown(np.array([0.01, 0.02])) == 0

    def test_summary_annualizes_by_observed_frequency(self):
        dates = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(731))
        r = Returns(dates, np.full(731, 0.0002))
        s = summarize(r)
        assert r.periods_per_year == pytest.approx(365.25, rel=1e-3)
        assert s.cagr == pytest.approx(1.0002 ** 365.25 - 1, rel=1e-3) and s.max_drawdown == 0

    def test_window(self):
        dates = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(10))
        r = Returns(dates, np.arange(10.0)).between(date(2025, 1, 3), date(2025, 1, 6))
        assert r.dates == dates[2:5] and list(r.values) == [2.0, 3.0, 4.0]
        hurdled = Returns(dates, np.arange(10.0)).over({d: 0.5 for d in dates}).between(dates[2], dates[5])
        assert list(hurdled.excess) == [1.5, 2.5, 3.5]


class TestHurdle:
    """Sharpe ratios in excess of T-bills; returns, volatility and drawdown as earned."""

    DATES = tuple(date(2025, 1, 1) + timedelta(days=i) for i in range(6))
    VALUES = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005])

    def test_excess_is_the_returns_without_a_hurdle_and_net_of_it_with_one(self):
        plain = Returns(self.DATES, self.VALUES)
        assert plain.excess is plain.values
        hurdled = plain.over({self.DATES[0]: 0.001, self.DATES[3]: 0.002})  # missing days: zero
        assert list(hurdled.hurdle) == [0.001, 0, 0, 0.002, 0, 0]
        assert hurdled.excess == pytest.approx(self.VALUES - hurdled.hurdle)
        with pytest.raises(ValueError, match="hurdle"):
            Returns(self.DATES, self.VALUES, np.zeros(2))

    def test_summary_measures_sharpe_and_sortino_against_the_hurdle_only(self):
        plain = summarize(Returns(self.DATES, self.VALUES))
        hurdled = summarize(Returns(self.DATES, self.VALUES).over({d: 0.004 for d in self.DATES}))
        per_year = Returns(self.DATES, self.VALUES).periods_per_year
        assert hurdled.sharpe == pytest.approx(sharpe_ratio(self.VALUES - 0.004) * math.sqrt(per_year))
        assert hurdled.sharpe < plain.sharpe and hurdled.sortino < plain.sortino
        assert (hurdled.total_return, hurdled.volatility, hurdled.max_drawdown) == \
            (plain.total_return, plain.volatility, plain.max_drawdown)
        assert hurdled.hurdle_return == pytest.approx(1.004 ** 6 - 1) and plain.hurdle_return == 0

    def test_hurdle_returns_are_dated_when_their_bar_is_known(self):
        bars = daily_bars(BIL, date(2026, 3, 2), date(2026, 3, 6), 91.0, seed=5)
        returns = hurdle_returns(bars, timedelta(minutes=15))
        # Monday's bar ends at midnight New York (05:00 UTC Tuesday) and is known at 05:15 UTC Tuesday
        assert list(returns) == [date(2026, 3, 4), date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 7)]
        assert returns[date(2026, 3, 4)] == pytest.approx(bars[1].close / bars[0].close - 1)

    def test_matches_how_an_equity_curve_is_dated(self):
        """A strategy that holds only the T-bill ETF earns exactly its hurdle: zero excess."""
        bars = {"SPY": daily_bars(SPY, date(2026, 1, 2), date(2026, 3, 31), 100.0, seed=1),
                "BIL": daily_bars(BIL, date(2026, 1, 2), date(2026, 3, 31), 91.0, seed=5, vol=0.001)}
        config = BacktestConfig(start=datetime(2026, 2, 2, tzinfo=UTC), end=datetime(2026, 3, 31, tzinfo=UTC),
                                costs={k: CostModel() for k in AssetClass})
        result = _run_hold_bil(bars, config)
        returns = daily_returns(result.equity_curve).over(hurdle_returns(bars["BIL"], timedelta(minutes=15)))
        held = returns.between(date(2026, 2, 5), date(2026, 3, 31))  # after the first fill
        assert np.abs(held.excess).max() < 1e-4


# -- overfitting ------------------------------------------------------------------------------

class TestProbabilityOfBacktestOverfitting:
    def test_noise_is_overfit_about_half_the_time(self):
        pbos = [probability_of_backtest_overfitting(np.random.default_rng(s).normal(0, 0.01, (800, 20)),
                                                    blocks=8).pbo for s in range(30)]
        assert 0.35 < np.mean(pbos) < 0.75

    def test_a_real_edge_is_not_overfit(self):
        matrix = np.random.default_rng(6).normal(0, 0.01, (2000, 30))
        matrix[:, 7] += 0.002
        result = probability_of_backtest_overfitting(matrix)
        assert result.pbo < 0.05 and result.probability_of_loss == 0 and result.splits == 12870

    def test_matches_a_direct_computation(self):
        matrix = np.random.default_rng(7).normal(0, 0.01, (400, 5))
        result = probability_of_backtest_overfitting(matrix, blocks=4)
        parts = np.split(matrix, 4)
        logits, out_of_sample = [], []
        for chosen in combinations(range(4), 2):
            in_sample = sharpe_columns(np.concatenate([parts[b] for b in chosen]))
            out = sharpe_columns(np.concatenate([parts[b] for b in range(4) if b not in chosen]))
            best = int(np.argmax(in_sample))
            w = ((out < out[best]).sum() + 1) / 6  # rank among 5 trials over N + 1
            logits.append(math.log(w / (1 - w)))
            out_of_sample.append(out[best])
        assert result.logits == pytest.approx(logits)
        assert result.out_of_sample == pytest.approx(out_of_sample)

    @pytest.mark.parametrize("shape, blocks", [((100, 1), 4), ((100, 3), 5), ((10, 3), 8)])
    def test_invalid_inputs(self, shape, blocks):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(np.zeros(shape), blocks)


class TestPurgedCrossValidation:
    def test_splits_purge_and_embargo(self):
        splits = purged_splits(60, groups=6, test_groups=2, purge=2, embargo=3)
        assert len(splits) == 15
        assert all(sum(g in s.test_groups for s in splits) == 5 for g in range(6))
        first = next(s for s in splits if s.test_groups == (1, 3))
        assert set(first.test) == set(range(10, 20)) | set(range(30, 40))
        assert not set(first.train) & set(first.test)
        assert {8, 9, 20, 21, 22, 28, 29, 40, 41, 42}.isdisjoint(first.train)  # purged before, embargoed after
        assert {7, 23, 27, 43}.issubset(first.train)

    def test_paths_select_the_real_edge(self):
        matrix = np.random.default_rng(8).normal(0, 0.01, (1200, 10))
        matrix[:, 3] += 0.002
        result = combinatorial_purged_cv(matrix, groups=6, test_groups=2)
        assert result.paths.shape == (5, 1200) and result.splits == 15
        assert set(result.selections) == {3} and (result.path_sharpes > 0).all()
        assert result.paths[0] == pytest.approx(matrix[:, 3])

    def test_invalid_groups(self):
        with pytest.raises(ValueError):
            purged_splits(10, 4, 4)


# -- experiments through the engine ---------------------------------------------------------------

class _Seasonal(Strategy):
    """Holds SPY only in the months listed: a family with a real parameter."""

    name = "seasonal"
    lookback = 1

    def __init__(self, months, weight=0.6):
        self.instruments = (SPY, IEF)
        self.months, self.weight = set(months), weight

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        return {"SPY": self.weight if ctx.now.month in self.months else 0.0}


def _bars(start=date(2014, 1, 2), end=date(2025, 12, 31)):
    return {"SPY": daily_bars(SPY, start, end, 100.0, seed=1),
            "IEF": daily_bars(IEF, start, end, 80.0, seed=2, drift=0.0001, vol=0.004)}


LOOSE = RiskLimits(max_drawdown=D("0.5"), daily_loss_limit=D("0.5"))
CONFIG = BacktestConfig(start=datetime(2015, 1, 2, tzinfo=UTC), end=datetime(2021, 1, 1, tzinfo=UTC),
                        risk_limits=LOOSE)
WF = WalkForwardConfig(train_months=24, test_months=12)
HOLD = Candidate("hold", lambda p: BuyAndHold({SPY: p["equity"], IEF: round(1 - p["equity"], 6)}),
                 {"equity": [0.4, 0.6]})


class TestCandidate:
    def test_parameter_sets_cover_the_grid_in_order(self):
        c = Candidate("c", lambda p: None, {"a": [1, 2], "b": ["x", "y", "z"]})
        sets = c.parameter_sets()
        assert len(sets) == 6 and sets[0] == {"a": 1, "b": "x"} and sets[-1] == {"a": 2, "b": "z"}

    @pytest.mark.parametrize("name, grid", [("Bad Name", {"a": [1]}), ("ok", {"a": []})])
    def test_validation(self, name, grid):
        with pytest.raises(ValueError):
            Candidate(name, lambda p: None, grid)


class TestWalkForwardSchedule:
    def test_rolling_and_anchored_windows_follow_the_calendar(self):
        start, end = datetime(2015, 1, 1, tzinfo=UTC), datetime(2019, 7, 1, tzinfo=UTC)
        rolling = windows(start, end, WF)
        jan = [datetime(y, 1, 1, tzinfo=UTC) for y in (2015, 2016, 2017, 2018, 2019)]
        assert rolling == [(jan[0], jan[2], jan[3]), (jan[1], jan[3], jan[4]), (jan[2], jan[4], end)]
        anchored = windows(start, end, WalkForwardConfig(24, 12, anchored=True))
        assert all(w[0] == start for w in anchored)

    def test_month_arithmetic_clamps_to_the_month_end(self):
        assert add_months(datetime(2024, 1, 31, tzinfo=UTC), 1) == datetime(2024, 2, 29, tzinfo=UTC)
        assert add_months(datetime(2024, 3, 15, tzinfo=UTC), -15) == datetime(2022, 12, 15, tzinfo=UTC)
        with pytest.raises(ValueError):
            WalkForwardConfig(train_months=0)

    def test_selection_uses_only_returns_realized_before_the_window(self):
        dates = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(40))
        noise = np.tile([0.0002, -0.0002], 20)
        crashing = np.full(40, 0.001) + noise
        crashing[30] = -0.05  # a loss on the first test day itself
        trials = [Trial({"a": 0}, None, Returns(dates, crashing)),
                  Trial({"a": 1}, None, Returns(dates, np.full(40, 0.0008) + noise))]
        test_start = datetime(2020, 1, 31, tzinfo=UTC)
        assert dates[30] == test_start.date()
        # The loss is not known when the window starts; one day later it would flip the choice
        assert select(trials, datetime(2020, 1, 1, tzinfo=UTC), test_start)[0] == 0
        assert select(trials, datetime(2020, 1, 1, tzinfo=UTC), test_start + timedelta(days=1))[0] == 1


class _Counting(Strategy):
    name = "counting"
    lookback = 1

    def __init__(self, weights):
        self.instruments = (SPY,)
        self.weights = weights
        self.calls = 0

    def target_weights(self, ctx):
        self.calls += 1
        return self.weights if self.calls == 1 else None  # only the first call says something


class TestSwitchingStrategy:
    def test_trades_the_active_set_and_keeps_every_set_current(self):
        a, b = _Counting({"SPY": 0.2}), _Counting({"SPY": 0.8})
        t0 = datetime(2026, 1, 5, tzinfo=UTC)
        switching = SwitchingStrategy("wf-test", [a, b], [(t0, 0), (t0 + timedelta(days=10), 1)])
        spy = {"SPY": (bar(SPY, datetime(2026, 1, 2, 5, tzinfo=UTC), 100, 100, 100, 100),)}
        ctx = StrategyContext(t0 - timedelta(days=1), spy, {}, D("1"))
        assert switching.target_weights(ctx) == {}  # before the first window: flat
        assert switching.target_weights(StrategyContext(t0, spy, {}, D("1"))) == {"SPY": 0.2}
        # b answered only on its first call (above), yet its targets are known when it is switched in
        later = StrategyContext(t0 + timedelta(days=10), spy, {}, D("1"))
        assert switching.target_weights(later) == {"SPY": 0.8}
        assert a.calls == b.calls == 3

    def test_name_is_validated(self):
        with pytest.raises(ValueError):
            SwitchingStrategy("Not valid", [_Counting({})], [])

    def test_switching_to_a_set_trading_another_fund_sells_the_first(self):
        """The low-vol candidate's sets hold different funds (USMV or SPLV); a switch must swap them."""
        from src.trading.backtest import run_backtest
        from src.trading.engine import Allocation
        switch_at = datetime(2016, 1, 4, tzinfo=UTC)
        switching = SwitchingStrategy("wf-test", [BuyAndHold({SPY: 0.6}), BuyAndHold({IEF: 0.6})],
                                      [(CONFIG.start, 0), (switch_at, 1)])
        config = dataclasses.replace(CONFIG, end=datetime(2016, 3, 1, tzinfo=UTC))
        result = run_backtest([Allocation(switching, D("1"))], _bars(), config)
        before = [f for f in result.fills if f.timestamp < switch_at]
        after = [f for f in result.fills if f.timestamp >= switch_at]
        assert before and {f.symbol for f in before} == {"SPY"}
        assert {("SPY", "sell"), ("IEF", "buy")} <= {(f.symbol, f.side.value) for f in after}
        assert [p.symbol for p in result.final.positions] == ["IEF"]

    def test_sets_are_consulted_only_once_their_history_suffices(self):
        long_memory = _Counting({"SPY": 0.5})
        long_memory.lookback = 3
        switching = SwitchingStrategy("wf-test", [long_memory], [(datetime(2026, 1, 1, tzinfo=UTC), 0)])
        bars = tuple(bar(SPY, datetime(2026, 1, d, 5, tzinfo=UTC), 100, 100, 100, 100) for d in (5, 6))
        assert switching.target_weights(StrategyContext(datetime(2026, 1, 7, tzinfo=UTC), {"SPY": bars}, {},
                                                        D("1"))) == {}
        assert long_memory.calls == 0


class TestWalkForwardOfABlend:
    def test_a_component_without_data_does_not_hold_the_others_back(self):
        """Regression: the walk-forward once waited for the slowest component's full lookback."""
        from src.tests.trading_fixtures import BTC
        from src.trading.strategies.blend import Blend
        from src.trading.strategies.crypto_trend import DonchianEnsemble
        blend = Candidate("blend", lambda p: Blend("lineup", [(BuyAndHold({SPY: 0.6, IEF: 0.4}), p["core"]),
                                                              (DonchianEnsemble([BTC]), 0.15)]),
                          {"core": [0.8, 0.85]})
        bars = _bars()  # equities only: the crypto component never has enough history
        walk = walk_forward(blend, run_trials(blend, bars, CONFIG), bars, CONFIG, WF)
        first = walk.windows[0].test_start
        assert walk.result.fills and min(f.timestamp for f in walk.result.fills) < first + timedelta(days=7)


class TestWalkForwardThroughTheEngine:
    def test_trades_only_out_of_sample_with_the_selected_sets(self):
        bars = _bars()
        trials = run_trials(HOLD, bars, CONFIG)
        walk = walk_forward(HOLD, trials, bars, CONFIG, WF)
        first = walk.windows[0].test_start
        assert len(walk.windows) == 4 and first == datetime(2017, 1, 2, tzinfo=UTC)
        assert walk.result.fills and min(f.timestamp for f in walk.result.fills) >= first
        assert [w.choice for w in walk.windows] == [select(trials, w.train_start, w.test_start)[0]
                                                    for w in walk.windows]
        assert walk.returns.dates[0] >= first.date()

    def test_switching_parameters_costs_money(self):
        bars = _bars()
        seasonal = Candidate("seasonal", lambda p: _Seasonal(p["months"]),
                             {"months": [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)]})
        config = BacktestConfig(start=CONFIG.start, end=CONFIG.end, risk_limits=LOOSE,
                                costs={AssetClass.EQUITY: CostModel(D("20"), D("20")),
                                       AssetClass.CRYPTO: CostModel()})
        walk = walk_forward(seasonal, run_trials(seasonal, bars, config), bars, config, WF)
        stressed = cost_stress(seasonal, walk, bars, config, multiples=(2.0, 4.0))
        totals = [summarize(r).total_return for r in (walk.returns, stressed[2.0], stressed[4.0])]
        assert totals[0] > totals[1] > totals[2]


class TestNeighbours:
    GRID = Candidate("grid", lambda p: None, {"a": [1, 2, 3], "b": [10, 20, 30]})

    def test_one_step_in_one_parameter(self):
        sharpes = [0.1 * i for i in range(9)]
        centre = neighbours(self.GRID, sharpes, 4)  # a=2, b=20
        assert sorted(centre.neighbour_sharpes) == [1, 3, 5, 7]
        assert centre.ratio == pytest.approx(np.median([0.1, 0.3, 0.5, 0.7]) / 0.4)
        assert sorted(neighbours(self.GRID, sharpes, 0).neighbour_sharpes) == [1, 3]

    def test_ratio_needs_a_positive_selection(self):
        assert neighbours(self.GRID, [-1.0] * 9, 4).ratio is None


class TestRegimes:
    def test_days_are_classified_by_the_state_known_when_they_began(self):
        start = date(2025, 1, 1)
        reference, price = [], 100.0
        for i in range(60):
            day = start + timedelta(days=i)
            price *= 1.01 if i < 30 else 0.99
            reference.append(bar(SPY, datetime(day.year, day.month, day.day, tzinfo=UTC), price, price, price, price))
        known = [(b.end + timedelta(minutes=1)).date() for b in reference]
        returns = Returns(tuple(known), np.full(60, 0.001))
        stats = {r.label: r for r in regimes(returns, reference, timedelta(minutes=1), trend_days=5, vol_days=3)}
        assert stats["2025"].periods == 60
        # The 5-bar trend is known from bar 5 and stays positive until bar 31 (two bars into
        # the decline); the return of day j is classified by bar j - 1: days 6-32 rise
        assert stats["rising market"].periods == 27 and stats["falling market"].periods == 27


class TestVerdict:
    def test_any_failure_rejects(self):
        gates = [GateResult("a", True, 1, ""), GateResult("b", None, None, ""), GateResult("c", False, 0, "")]
        assert verdict(gates) == (False, "rejected: fails c")
        assert verdict(gates[:2])[0] is True


# -- the whole pipeline -----------------------------------------------------------------------

class TestPipeline:
    def test_report_on_a_grid(self):
        config = ValidationConfig(CONFIG, walk_forward=WF, pbo_blocks=8)
        report = validate(HOLD, _bars(), config)
        names = [g.name for g in report.gates]
        assert names == ["out-of-sample Sharpe", "deflated Sharpe", "backtest overfitting", "profitable at 3x costs",
                         "parameter stability", "maximum drawdown", "paper track record"]
        assert report.gates[-1].passed is None
        assert report.overfitting is not None and report.overfitting.splits == 70
        assert report.cross_validation["paths"] == 5
        assert report.passed == all(g.passed is not False for g in report.gates)
        markdown = report.to_markdown()
        for section in ("## Promotion gates", "## Out of sample (walk-forward)", "## Selection bias and overfitting",
                        "## Costs and parameter stability", "## Years and market regimes", "## All trials"):
            assert section in markdown
        data = json.loads(json.dumps(report.to_dict()))
        assert data["candidate"] == "hold" and len(data["trials"]) == 2 and len(data["walk_forward"]) == 4

    def test_a_single_parameter_set_has_nothing_to_overfit(self):
        single = Candidate("single", HOLD.factory, {"equity": [0.6]})
        report = validate(single, _bars(), ValidationConfig(CONFIG, walk_forward=WF, pbo_blocks=8))
        gate = next(g for g in report.gates if g.name == "backtest overfitting")
        assert gate.passed is None and report.overfitting is None and report.cross_validation is None
        assert report.deflated.dsr == pytest.approx(report.probabilistic_sharpe)

    def test_the_regime_symbol_may_be_traded_by_any_parameter_set(self):
        from src.trading.validation.pipeline import _regime_reference
        funds = Candidate("funds", lambda p: BuyAndHold({p["fund"]: 0.6}), {"fund": [SPY, IEF]}, regime_symbol="IEF")
        assert _regime_reference(funds, funds.parameter_sets()) == IEF  # only the second set trades IEF

    def test_a_regime_symbol_no_set_trades_is_refused_before_any_backtest(self, monkeypatch):
        from src.trading.validation import pipeline
        monkeypatch.setattr(pipeline, "run_trials", lambda *a: pytest.fail("backtests ran"))
        funds = Candidate("funds", lambda p: BuyAndHold({SPY: p["equity"]}), {"equity": [0.6]}, regime_symbol="QQQ")
        with pytest.raises(ValueError, match="regime symbol QQQ is not traded by any parameter set of funds"):
            validate(funds, _bars(), ValidationConfig(CONFIG, walk_forward=WF))

    def test_too_short_a_period_is_refused(self):
        short = BacktestConfig(start=datetime(2015, 1, 2, tzinfo=UTC), end=datetime(2015, 1, 20, tzinfo=UTC))
        with pytest.raises(ValueError):
            validate(HOLD, _bars(), ValidationConfig(short, walk_forward=WF))

    def test_against_t_bills_holding_only_t_bills_is_no_edge(self):
        """The sleeve's interest raises returns but no gate: Sharpe and the cost gate are net of it."""
        bars = {**_bars(), "BIL": daily_bars(BIL, date(2014, 1, 2), date(2025, 12, 31), 91.0, seed=5,
                                             drift=0.0002, vol=0.0001)}
        tbills = hurdle_returns(bars["BIL"], timedelta(minutes=15))
        idle = Candidate("idle", lambda p: BuyAndHold({SPY: p["equity"]}), {"equity": [0.001, 0.002]})
        sleeved = dataclasses.replace(CONFIG, cash_sleeve=BIL)
        plain = validate(idle, bars, ValidationConfig(sleeved, walk_forward=WF, pbo_blocks=8))
        report = validate(idle, bars, ValidationConfig(sleeved, walk_forward=WF, pbo_blocks=8, hurdle=tbills,
                                                       hurdle_label="T-bills (BIL)"))
        assert plain.out_of_sample.sharpe > 3  # interest alone looks like a great strategy against zero
        # Against T-bills: no edge, and a little worse for the unspent cash buffer and the costs
        assert report.out_of_sample.sharpe <= 0 and report.out_of_sample.hurdle_return > 0.02
        assert report.out_of_sample.total_return < report.out_of_sample.hurdle_return
        stress = next(g for g in report.gates if g.name == "profitable at 3x costs")
        assert stress.passed is False and stress.threshold == "> T-bills (BIL) total return"
        assert report.settings["sharpe_in_excess_of"] == "T-bills (BIL)"
        assert "in excess of T-bills (BIL)" in report.to_markdown()


# -- operator command -------------------------------------------------------------------------

class TestValidateCommand:
    @pytest.fixture
    def synthetic_market(self, monkeypatch):
        from src.trading.validation import market

        calls = []

        def load(settings, symbols, start, end):
            calls.append((tuple(s for s, _ in symbols), start, end))
            bars = {**_bars(), "BIL": daily_bars(BIL, date(2014, 1, 2), date(2025, 12, 31), 91.0, seed=5,
                                                 vol=0.0001)}
            return {"SPY": SPY, "IEF": IEF, "BIL": BIL}, {s: bars[s] for s, _ in symbols}

        monkeypatch.setattr(market, "load", load)
        return calls

    def test_writes_the_tear_sheet_and_json(self, app, synthetic_market, tmp_path):
        out = tmp_path / "reports" / "benchmark.md"
        result = app.test_cli_runner().invoke(args=[
            "validate", "benchmark", "--start", "2016-01-04", "--end", "2021-01-01", "--output", str(out),
            "--train-months", "24"])
        assert result.exit_code in (0, 1), result.output  # 1: rejected by the gates
        assert "Verdict:" in result.output
        assert out.read_text().startswith("# Validation: benchmark")
        data = json.loads(out.with_suffix(".json").read_text())
        assert len(data["trials"]) == 5 and data["settings"]["resume_after_kill_days"] == 30
        assert (result.exit_code == 0) == data["passed"]
        [(symbols, start, end)] = synthetic_market
        assert symbols == ("SPY", "IEF", "BIL") and start == datetime(2016, 1, 4, tzinfo=UTC) - timedelta(days=400)
        assert data["settings"]["sharpe_in_excess_of"] == "T-bills (BIL)"

    def test_without_t_bills_as_before(self, app, synthetic_market, tmp_path):
        out = tmp_path / "benchmark.md"
        result = app.test_cli_runner().invoke(args=[
            "validate", "benchmark", "--start", "2016-01-04", "--end", "2021-01-01", "--output", str(out),
            "--train-months", "24", "--no-tbills"])
        assert result.exit_code in (0, 1), result.output
        [(symbols, _, _)] = synthetic_market
        assert symbols == ("SPY", "IEF")
        assert json.loads(out.with_suffix(".json").read_text())["settings"]["sharpe_in_excess_of"] == "zero"

    @pytest.fixture
    def long_history(self, app, monkeypatch):
        from src.trading.validation import history, market

        calls = []

        def load(start, end):
            calls.append((start, end))
            bars = {**_bars(), "BIL": daily_bars(BIL, date(2014, 1, 2), date(2025, 12, 31), 91.0, seed=5,
                                                 vol=0.0001)}
            return ({"SPY": SPY, "IEF": IEF, "BIL": BIL}, bars,
                    [history.Source("https://example.test/french.zip", "ab" * 32, date(1926, 7, 1), date(2026, 8, 31))])

        monkeypatch.setattr(history, "load", load)
        monkeypatch.setattr(market, "load", lambda *args: pytest.fail("Alpaca data was loaded"))
        # The proxies need no Alpaca account
        settings = app.extensions["settings"].model_copy(update={"alpaca_api_key": None, "alpaca_secret_key": None})
        monkeypatch.setitem(app.extensions, "settings", settings)
        return calls

    def test_long_history_proxies(self, app, long_history, tmp_path):
        out = tmp_path / "trend-filter.md"
        result = app.test_cli_runner().invoke(args=[
            "validate", "trend-filter", "--data", "history", "--start", "2016-01-04", "--end", "2021-01-01",
            "--output", str(out), "--train-months", "24"])
        assert result.exit_code in (0, 1), result.output
        assert f"Data: https://example.test/french.zip (1926-07-01 to 2026-08-31, sha256 {'ab' * 32})" in result.output
        [(start, end)] = long_history
        assert start == datetime(2016, 1, 4, tzinfo=UTC) - timedelta(days=400)
        assert end == datetime(2021, 1, 1, tzinfo=UTC)
        data = json.loads(out.with_suffix(".json").read_text())
        assert [t["params"] for t in data["trials"]] == [{"months": 8}, {"months": 10}, {"months": 12}]
        assert data["settings"]["data"].startswith("long-history proxies for SPY, IEF and BIL")
        assert data["settings"]["sharpe_in_excess_of"] == "T-bills (French's 1-month rate, as BIL)"
        assert "Data: long-history proxies" in out.read_text()

    def test_long_history_has_no_proxy_for_other_symbols(self, app, long_history):
        result = app.test_cli_runner().invoke(args=["validate", "crypto-trend", "--data", "history"])
        assert result.exit_code != 0
        assert "proxies for SPY, IEF, BIL only, not BTC/USD, ETH/USD, SOL/USD" in result.output
        assert long_history == []  # refused before any download

    def test_long_history_errors_are_reported(self, app, monkeypatch):
        from src.trading.validation import history

        def load(start, end):
            raise ValueError("https://example.test/french.zip ends on 2026-08-31")

        monkeypatch.setattr(history, "load", load)
        result = app.test_cli_runner().invoke(args=["validate", "benchmark", "--data", "history"])
        assert result.exit_code != 0
        assert "long-history data: https://example.test/french.zip ends on 2026-08-31" in result.output

    def test_unknown_candidate(self, app):
        result = app.test_cli_runner().invoke(args=["validate", "nope"])
        assert result.exit_code != 0 and "unknown candidate" in result.output

    def test_needs_alpaca_keys(self, app, monkeypatch):
        settings = app.extensions["settings"].model_copy(update={"alpaca_api_key": None, "alpaca_secret_key": None})
        monkeypatch.setitem(app.extensions, "settings", settings)
        result = app.test_cli_runner().invoke(args=["validate", "benchmark", "--end", "2021-01-01"])
        assert result.exit_code != 0 and "ALPACA_API_KEY" in result.output

    def test_end_must_follow_start(self, app):
        result = app.test_cli_runner().invoke(args=["validate", "benchmark", "--start", "2021-01-01",
                                                    "--end", "2020-01-01"])
        assert result.exit_code != 0 and "--end must be after --start" in result.output
