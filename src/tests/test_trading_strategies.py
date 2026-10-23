"""
Tests for the candidate strategies against hand-computed cases from their
papers' rules: month-end rebalancing flows, the crypto Donchian ensemble,
time-series momentum, the month-end payment cycle and the trend filter.
"""

import math
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from src.tests.trading_fixtures import BIL, BTC, IEF, SPY, bar, daily_bars, ny
from src.trading.calendar import TradingCalendar
from src.trading.domain import NEW_YORK, AssetClass
from src.trading.strategies.crypto_trend import LOOKBACKS, DonchianEnsemble, ensemble_exposure, in_position
from src.trading.strategies.month_end import MonthEndCycle, held_at_close, settlement_days
from src.trading.strategies.rebalancing import (
    THRESHOLDS,
    RebalancingFlows,
    rebalancing_signals,
    strategy_weight,
)
from src.trading.strategies.trend import TimeSeriesMomentum, ewma_volatility, trend_signal
from src.trading.strategies.trend_filter import MONTHS, TrendFilter
from src.trading.strategy import CLOSE, OPEN, StrategyContext, consult

D = Decimal
SESSIONS = TradingCalendar(AssetClass.EQUITY)


def _ctx(history, now=None):
    last = max(b[-1].end for b in history.values() if b)
    return StrategyContext(now or last + timedelta(minutes=15), history, {}, D("100000"))


# -- rebalancing flows ---------------------------------------------------------------------

class TestRebalancingSignals:
    def test_the_papers_example_a_ten_percent_equity_outperformance(self):
        # "if the equity market outperforms the bond market by 10% on a single day within the
        # 60/40 portfolio, both rebalancing signals should increase by approximately 2.26%"
        threshold, calendar = rebalancing_signals([date(2026, 3, 10)], np.array([0.10]), np.array([0.0]))
        assert threshold == pytest.approx(0.0226, abs=5e-5) and calendar[0] == pytest.approx(0.0226, abs=5e-5)

    def test_threshold_portfolios_reset_when_their_drift_reaches_the_threshold(self):
        days = [date(2026, 3, 10), date(2026, 3, 11)]
        threshold, _ = rebalancing_signals(days, np.array([0.02, 0.0]), np.array([0.0, 0.0]))
        # Day 1 drifts every portfolio by about 0.474%; those with a threshold at or below it
        # reset to 60%, so on a flat day 2 only the others still carry the drift
        first = 0.6 * 1.02 / (0.6 * 1.02 + 0.4) - 0.6
        kept = first < THRESHOLDS
        assert threshold == pytest.approx(first * kept.mean())

    def test_the_calendar_portfolio_resets_after_the_last_session_of_the_month(self):
        days = [date(2026, 3, 30), date(2026, 3, 31), date(2026, 4, 1)]
        _, calendar = rebalancing_signals(days, np.array([0.01, 0.01, 0.0]), np.array([0.0, 0.0, 0.0]))
        assert calendar[1] > calendar[0] > 0  # drift accumulates within March
        assert calendar[2] == pytest.approx(0.0)  # reset at the March 31 close; no move on April 1


class TestRebalancingPosition:
    def _days(self, end):
        days = SESSIONS.sessions(end - timedelta(days=20), end)
        return days

    def test_last_week_trades_against_the_calendar_drift(self):
        days = self._days(date(2026, 3, 26))  # four sessions before the month ends
        calendar = np.full(len(days), 0.01)
        assert strategy_weight(days, 0.0, calendar) == pytest.approx(0.5 * -1)

    def test_first_session_reverses_with_the_drift_four_sessions_earlier(self):
        days = self._days(date(2026, 4, 1))
        calendar = np.zeros(len(days))
        calendar[-5] = 0.01
        assert strategy_weight(days, 0.0, calendar) == pytest.approx(0.5 * 1)

    def test_mid_month_only_the_threshold_signal_counts(self):
        days = self._days(date(2026, 3, 13))
        assert strategy_weight(days, 0.015, np.full(len(days), 0.05)) == pytest.approx(-0.5)

    def test_spread_and_tilt(self):
        history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 3, 24), 500.0, seed=1),
                   "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 3, 24), 95.0, seed=2)}
        spread = RebalancingFlows(SPY, IEF, "spread", 0.5).target_weights(_ctx(history))
        assert spread["SPY"] == pytest.approx(-spread["IEF"]) and spread["SPY"] != 0
        tilt = RebalancingFlows(SPY, IEF, "tilt", 1.0).target_weights(_ctx(history))
        assert tilt["SPY"] + tilt["IEF"] == pytest.approx(1.0) and 0 <= tilt["SPY"] <= 1
        assert tilt["SPY"] - 0.6 == pytest.approx(2 * spread["SPY"], abs=1e-9) or tilt["SPY"] in (0.0, 1.0)

    @pytest.mark.parametrize("mode, scale", [("pairs", 0.5), ("spread", 0.0), ("tilt", 1.5)])
    def test_invalid_parameters(self, mode, scale):
        with pytest.raises(ValueError):
            RebalancingFlows(SPY, IEF, mode, scale)


# -- crypto Donchian ensemble --------------------------------------------------------------------

class TestDonchianModel:
    def test_breakout_then_trailing_stop_at_the_channel_midpoint(self):
        # Lookback 3: the close of 11 breaks above 10 (stop: midpoint of 10, 10, 11 = 10.5);
        # 10.8 holds; 10.4 falls through the stop
        assert not in_position(np.array([10.0, 10.0, 10.0]), 3)
        assert in_position(np.array([10.0, 10.0, 10.0, 11.0]), 3)
        assert in_position(np.array([10.0, 10.0, 10.0, 11.0, 10.8]), 3)
        assert not in_position(np.array([10.0, 10.0, 10.0, 11.0, 10.8, 10.4]), 3)

    def test_the_stop_only_rises(self):
        rising = np.array([10.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        assert in_position(rising, 3)
        # Midpoint of (12, 13, 14) = 13 is the stop now: 12.9 exits although it is above 10.5
        assert not in_position(np.append(rising, 12.9), 3)

    def test_a_flat_market_never_breaks_out(self):
        assert ensemble_exposure(np.full(800, 100.0)) == 0

    def test_a_steady_uptrend_engages_every_model(self):
        assert ensemble_exposure(100 * 1.002 ** np.arange(800)) == 1


class TestDonchianEnsemble:
    def _history(self, closes, start=datetime(2023, 1, 1, tzinfo=UTC)):
        return [bar(BTC, start + timedelta(days=i), c, c, c, c) for i, c in enumerate(closes)]

    def test_sized_to_the_volatility_target_and_capped(self):
        rng = np.random.default_rng(0)
        closes = 100 * np.exp(np.cumsum(0.003 + rng.normal(0, 0.03, 800)))
        strategy = DonchianEnsemble([BTC])
        weights = strategy.target_weights(_ctx({"BTC/USD": self._history(closes)}))
        returns = np.diff(closes[-91:]) / closes[-91:-1]
        size = min(1.0, 0.25 / (returns.std(ddof=1) * math.sqrt(365)))
        assert weights["BTC/USD"] == pytest.approx(ensemble_exposure(closes) * size)
        assert strategy.lookback == 2 * max(LOOKBACKS) + 1

    def test_capital_is_split_between_coins_and_short_history_is_flat(self):
        eth = BTC.__class__("ETH/USD", AssetClass.CRYPTO, D("0.001"), D("0.001"))
        up = 100 * 1.002 ** np.arange(800)
        history = {"BTC/USD": self._history(up), "ETH/USD": self._history(up[:300])}
        weights = DonchianEnsemble([BTC, eth]).target_weights(_ctx(history))
        assert weights["ETH/USD"] == 0 and 0 < weights["BTC/USD"] <= 0.5


# -- time-series momentum ------------------------------------------------------------------------

class TestTrendSignal:
    def test_sign_and_clipped_t_statistic(self):
        up = 100 * 1.001 ** np.arange(300)
        assert trend_signal(up, "sign") == 1 and trend_signal(up[::-1], "sign") == -1
        noisy = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.00005, 0.01, 300)))
        t = trend_signal(noisy, "tstat")
        logs = np.diff(np.log(noisy[-253:]))
        expected = logs.mean() / (logs.std() / math.sqrt(252))
        assert t == pytest.approx(max(-1.0, min(1.0, expected)))
        assert trend_signal(up, "tstat") == 1.0

    def test_ewma_volatility_of_a_constant_magnitude_series(self):
        returns = np.tile([0.01, -0.01], 126)
        assert ewma_volatility(returns) == pytest.approx(0.01 * math.sqrt(252), rel=0.01)


class TestTimeSeriesMomentum:
    def _history(self, end):
        return {"SPY": daily_bars(SPY, date(2024, 1, 2), end, 500.0, seed=1, drift=0.001),
                "IEF": daily_bars(IEF, date(2024, 1, 2), end, 95.0, seed=2, drift=-0.001, vol=0.004)}

    def test_positions_change_only_at_month_ends(self):
        strategy = TimeSeriesMomentum([SPY, IEF])
        mid = strategy.target_weights(_ctx(self._history(date(2026, 3, 13))))
        month_end = strategy.target_weights(_ctx(self._history(date(2026, 2, 27))))
        assert mid == month_end
        assert mid["SPY"] > 0 > mid["IEF"]  # rising equities long, falling bonds short

    def test_sized_by_inverse_volatility_over_the_basket(self):
        history = self._history(date(2026, 2, 27))
        weights = TimeSeriesMomentum([SPY, IEF], target_volatility=0.4).target_weights(_ctx(history))
        closes = np.array([b.close for b in history["SPY"]])
        returns = np.diff(closes[-253:]) / closes[-253:-1]
        assert weights["SPY"] == pytest.approx(0.4 / ewma_volatility(returns) / 2)

    def test_long_only_holds_cash_instead_of_shorts(self):
        weights = TimeSeriesMomentum([SPY, IEF], long_only=True).target_weights(_ctx(self._history(date(2026, 2, 27))))
        assert weights["IEF"] == 0 and weights["SPY"] > 0

    def test_too_little_history_is_flat(self):
        history = {"SPY": daily_bars(SPY, date(2025, 6, 2), date(2026, 2, 27), 500.0)}
        assert TimeSeriesMomentum([SPY]).target_weights(_ctx(history)) == {"SPY": 0.0}

    def test_invalid_rule(self):
        with pytest.raises(ValueError):
            TimeSeriesMomentum([SPY], rule="ma")


def test_strategies_run_in_a_backtest():
    from src.trading.backtest import BacktestConfig, run_backtest
    from src.trading.engine import Allocation
    bars = {"SPY": daily_bars(SPY, date(2023, 1, 3), date(2025, 6, 30), 400.0, seed=1),
            "IEF": daily_bars(IEF, date(2023, 1, 3), date(2025, 6, 30), 95.0, seed=2, vol=0.004)}
    config = BacktestConfig(start=ny(2024, 2, 1), end=ny(2025, 6, 1))
    for strategy in (RebalancingFlows(SPY, IEF, "tilt", 0.5), TimeSeriesMomentum([SPY, IEF])):
        result = run_backtest([Allocation(strategy, D("1"))], bars, config)
        assert result.fills and result.final.equity > 0


# -- the month-end payment cycle ------------------------------------------------------------

def _held(rule, days):
    return [d.isoformat()[5:] for d in days if held_at_close(d, rule)]


OCTOBER_2026 = [date(2026, 10, d) for d in (23, 26, 27, 28, 29, 30)] + [date(2026, 11, d) for d in (2, 3, 4, 5)]


class TestMonthEndCycle:
    def test_settlement_shortened_to_two_days_then_one(self):
        assert [settlement_days(d) for d in (date(2017, 9, 1), date(2017, 9, 5), date(2024, 5, 24),
                                             date(2024, 5, 28))] == [3, 2, 2, 1]

    @pytest.mark.parametrize("rule, held", [
        # T = Friday 30 October 2026; T+3 = Wednesday 4 November is sold at the close
        ("paper", ["10-26", "10-27", "10-28", "10-29", "10-30", "11-02", "11-03"]),  # from T-4
        ("settlement", ["10-28", "10-29", "10-30", "11-02", "11-03"]),  # T+1 settlement: from T-2
        ("turn", ["10-29", "10-30", "11-02", "11-03"]),  # from T-1
    ])
    def test_held_from_the_close_of_t_minus_k_to_the_close_of_t_plus_3(self, rule, held):
        assert _held(rule, OCTOBER_2026) == held

    def test_the_settlement_rule_on_the_entry_day_decides(self):
        # August 2017 (T+3): from T-4, 25 August. September 2017 (T+2 since the 5th): from T-3, 26 September
        assert _held("settlement", [date(2017, 8, d) for d in (24, 25, 28)]) == ["08-25", "08-28"]
        assert _held("settlement", [date(2017, 9, d) for d in (25, 26)]) == ["09-26"]
        # May 2024: T+1 from Tuesday the 28th (after Memorial Day), so from T-2, 29 May, to T+2, 4 June
        may = [date(2024, 5, d) for d in (24, 28, 29, 31)] + [date(2024, 6, d) for d in (3, 4, 5)]
        assert _held("settlement", may) == ["05-29", "05-31", "06-03", "06-04"]

    def test_targets_the_close_of_the_session_after_the_newest_bar(self):
        strategy = MonthEndCycle(SPY)
        assert strategy.execution == CLOSE
        for newest, weight in ((date(2026, 10, 27), 1.0), (date(2026, 11, 3), 0.0), (date(2026, 10, 26), 0.0)):
            history = {"SPY": (bar(SPY, ny(newest.year, newest.month, newest.day), 100, 100, 100, 100),)}
            assert strategy.target_weights(_ctx(history)) == {"SPY": weight}

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="rule"):
            MonthEndCycle(SPY, rule="halloween")

    def test_trades_in_the_closing_auctions_of_the_window(self):
        from src.trading.backtest import BacktestConfig, run_backtest
        from src.trading.broker import CostModel
        from src.trading.engine import Allocation
        bars = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 12, 31), 600.0, seed=1),
                "BIL": daily_bars(BIL, date(2025, 1, 2), date(2026, 12, 31), 91.0, seed=5, vol=0.0001)}
        config = BacktestConfig(start=ny(2026, 10, 1), end=ny(2026, 11, 30), cash_sleeve=BIL,
                                costs={k: CostModel() for k in AssetClass})
        result = run_backtest([Allocation(MonthEndCycle(SPY), D("1"))], bars, config)
        closes = {b.start.astimezone(NEW_YORK).date(): b.close for b in bars["SPY"]}
        spy = [f for f in result.fills if f.symbol == "SPY"]
        sessions = [SESSIONS.previous_session(f.timestamp.astimezone(NEW_YORK).date()) for f in spy]
        assert [(d.isoformat(), f.side.value) for d, f in zip(sessions, spy, strict=True)] == [
            ("2026-10-01", "buy"), ("2026-10-05", "sell"),  # the end of September's window: T+1 to T+3
            ("2026-10-28", "buy"),  # T-2 of October (T+1 settlement)
            ("2026-11-04", "sell"),  # T+3
            ("2026-11-25", "buy")]  # T-2 of November (Thanksgiving on the 26th)
        assert all(float(f.price) == pytest.approx(closes[d]) for d, f in zip(sessions, spy, strict=True))
        assert {f.symbol for f in result.fills} == {"SPY", "BIL"}


# -- trend filter ----------------------------------------------------------------------------

def _month_end_history(month_end_closes, first=date(2024, 1, 2), last=None, intra=None):
    """
    SPY bars on every session from ``first``: each month's sessions close at
    that month's value in ``month_end_closes`` (in order), or at ``intra``
    before the month's last session when given; the history stops after ``last``.
    """
    bars, month, value = [], None, None
    values = iter(month_end_closes)
    day = first if SESSIONS.is_session(first) else SESSIONS.next_session(first)
    while True:
        if (day.year, day.month) != month:
            month = (day.year, day.month)
            value = next(values, None)
            if value is None:
                break
        close = value if intra is None or SESSIONS.next_session(day).month != day.month else intra
        bars.append(bar(SPY, ny(day.year, day.month, day.day), close, close, close, close))
        if last is not None and day >= last:
            break
        day = SESSIONS.next_session(day)
    return {"SPY": tuple(bars)}


class TestTrendFilter:
    def test_holds_the_equity_sleeve_while_the_month_end_close_is_above_its_mean(self):
        rising = _month_end_history([100 + i for i in range(10)])  # 109 against a mean of 104.5
        assert TrendFilter(SPY, IEF).target_weights(_ctx(rising)) == {"SPY": 0.6, "IEF": 0.4}

    @pytest.mark.parametrize("closes", [[109 - i for i in range(10)], [100] * 10])
    def test_t_bills_instead_when_below_or_at_the_mean(self, closes):
        # The engine's cash sleeve takes the 60% left over; the bonds keep their 40%
        assert TrendFilter(SPY, IEF).target_weights(_ctx(_month_end_history(closes))) == {"SPY": 0.0, "IEF": 0.4}

    def test_the_mean_is_over_the_last_n_month_end_closes(self):
        closes = [200] * 4 + [100] * 7 + [130]
        history = _ctx(_month_end_history(closes))
        assert TrendFilter(SPY, IEF, months=8).target_weights(history)["SPY"] == 0.6  # mean 103.75
        assert TrendFilter(SPY, IEF, months=12).target_weights(history)["SPY"] == 0.0  # mean 135.83

    def test_prices_inside_a_month_are_ignored(self):
        # Ten rising month ends to October 2024, then November trades at 50 until its last session
        closes = [100 + i for i in range(10)] + [50]
        mid_november = _month_end_history(closes, last=date(2024, 11, 27), intra=50)
        assert TrendFilter(SPY, IEF).target_weights(_ctx(mid_november))["SPY"] == 0.6
        november_end = _month_end_history(closes, last=date(2024, 11, 29))  # the month's last session
        assert TrendFilter(SPY, IEF).target_weights(_ctx(november_end))["SPY"] == 0.0

    def test_no_decision_before_n_month_ends(self):
        assert TrendFilter(SPY, IEF).target_weights(_ctx(_month_end_history([100 + i for i in range(9)]))) is None

    @pytest.mark.parametrize("months", MONTHS)
    def test_its_lookback_holds_n_month_ends_whatever_the_newest_session(self, months):
        strategy = TrendFilter(SPY, IEF, months=months)
        assert strategy.execution == OPEN and strategy.lookback == months * 23 + 1
        full = _month_end_history([100.0] * (months + 14))["SPY"]
        newest = [i for i, b in enumerate(full) if i >= months * 23 + 30][:260]  # every day of a year
        for i in newest:
            history = {"SPY": full[:i + 1], "IEF": full[:i + 1]}
            assert consult(strategy, _ctx(history)) is not None, full[i].start

    @pytest.mark.parametrize("kwargs", [{"months": 1}, {"equity_weight": 0.0}, {"equity_weight": 1.5}])
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            TrendFilter(SPY, IEF, **kwargs)


class TestBlend:
    def test_weights_add_up_and_components_warm_up_separately(self):
        from src.trading.strategies.benchmark import BuyAndHold
        from src.trading.strategies.blend import Blend
        coin = DonchianEnsemble([BTC])  # needs 721 bars
        blend = Blend("lineup", [(BuyAndHold({SPY: 0.6, IEF: 0.4}), 0.8), (coin, 0.2)])
        assert blend.min_history == 0 and blend.lookback == coin.lookback
        history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2025, 3, 3), 500.0),
                   "IEF": daily_bars(IEF, date(2025, 1, 2), date(2025, 3, 3), 95.0), "BTC/USD": ()}
        assert blend.target_weights(_ctx({k: v for k, v in history.items() if v})) == pytest.approx(
            {"SPY": 0.48, "IEF": 0.32})

    @pytest.mark.parametrize("weights", [[], [0.7, 0.7], [0.0]])
    def test_invalid_weights(self, weights):
        from src.trading.strategies.benchmark import BuyAndHold
        from src.trading.strategies.blend import Blend
        with pytest.raises(ValueError):
            Blend("lineup", [(BuyAndHold({SPY: 1.0}), w) for w in weights])

    def test_an_engine_calls_a_blend_before_every_symbol_has_data(self):
        from src.trading.backtest import BacktestConfig, run_backtest
        from src.trading.engine import Allocation
        from src.trading.strategies.benchmark import BuyAndHold
        from src.trading.strategies.blend import Blend
        bars = {"SPY": daily_bars(SPY, date(2024, 1, 2), date(2025, 6, 30), 500.0),
                "IEF": daily_bars(IEF, date(2024, 1, 2), date(2025, 6, 30), 95.0)}  # no BTC data at all
        blend = Blend("lineup", [(BuyAndHold({SPY: 0.6, IEF: 0.4}), 0.8), (DonchianEnsemble([BTC]), 0.2)])
        result = run_backtest([Allocation(blend, D("1"))], bars,
                              BacktestConfig(start=ny(2024, 6, 3), end=ny(2025, 6, 1)))
        assert {f.symbol for f in result.fills} == {"SPY", "IEF"}


class TestCandidateRegistry:
    @pytest.mark.parametrize("name", ["benchmark", "rebalancing", "tsmom", "crypto-trend", "month-end",
                                      "trend-filter", "lineup"])
    def test_every_candidate_builds_from_its_own_symbols(self, name):
        from src.trading.domain import Instrument
        from src.trading.validation.candidates import CANDIDATES
        spec = CANDIDATES[name]
        instruments = {s: Instrument(s, a, D("1") if a is AssetClass.EQUITY else D("0.0001"),
                                     D("1") if a is AssetClass.EQUITY else D("0.0001"), a is AssetClass.EQUITY)
                       for s, a in spec.symbols}
        candidate = spec.build(instruments)
        assert candidate.name == name
        for params in candidate.parameter_sets():
            strategy = candidate.factory(params)
            assert {i.symbol for i in strategy.instruments} <= set(instruments)
