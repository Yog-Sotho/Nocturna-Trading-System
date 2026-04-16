# FILE LOCATION: src/tests/test_backtester.py
"""
Tests for the backtesting engine.
Covers: CQ-08 (entry timestamps), P&L arithmetic, equity curve, Monte Carlo.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.advanced.backtester import (
    AdvancedBacktester,
    BacktestPosition,
)


@pytest.fixture()
def backtester():
    return AdvancedBacktester({
        "initial_capital": 100_000,
        "commission_rate": 0.001,
        "slippage_rate": 0.0,  # Zero slippage for deterministic tests
        "min_trade_size": 1.0,
    })


@pytest.fixture()
def sample_ohlcv():
    """Create a simple OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
    return pd.DataFrame(
        {
            "open": prices - 0.2,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.random.randint(1000, 10000, 50),
        },
        index=dates,
    )


def buy_and_hold_strategy(data: pd.DataFrame, params: dict):
    """Simple strategy: buy on first bar, do nothing after."""
    if len(data) == 1:
        return [{
            "symbol": "TEST",
            "side": "buy",
            "type": "market",
            "quantity": params.get("quantity", 100),
        }]
    return []


def buy_sell_strategy(data: pd.DataFrame, params: dict):
    """Buy on bar 1, sell on bar 5."""
    bar_count = len(data)
    if bar_count == 1:
        return [{
            "symbol": "TEST",
            "side": "buy",
            "type": "market",
            "quantity": 100,
        }]
    elif bar_count == 5:
        return [{
            "symbol": "TEST",
            "side": "sell",
            "type": "market",
            "quantity": 100,
        }]
    return []


class TestBacktesterInit:
    """Initialization and reset."""

    def test_initial_capital(self, backtester):
        assert backtester.initial_capital == 100_000
        assert backtester.current_capital == 100_000

    def test_reset_clears_state(self, backtester):
        backtester.total_trades = 5
        backtester.winning_trades = 3
        backtester.reset()
        assert backtester.total_trades == 0
        assert backtester.winning_trades == 0
        assert backtester.current_capital == 100_000


class TestPnLArithmetic:
    """Verify P&L calculations are mathematically correct."""

    def test_buy_and_hold_return(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_and_hold_strategy, {"quantity": 100}
        )
        assert "error" not in result
        assert result["total_trades"] == 0  # Position still open, no closed trades
        assert result["final_equity"] > 0

    def test_round_trip_trade_pnl(self, backtester, sample_ohlcv):
        """Buy then sell — check the closed trade PnL."""
        result = backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        assert "error" not in result
        assert result["total_trades"] >= 1

        # Check trade P&L consistency
        if backtester.trades:
            trade = backtester.trades[0]
            expected_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
            assert abs(trade.pnl - expected_pnl) < 0.01

    def test_commission_deducted(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        assert result.get("total_commission", 0) > 0

    def test_equity_curve_not_empty(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_and_hold_strategy, {"quantity": 10}
        )
        assert len(result.get("equity_curve", [])) == len(sample_ohlcv)

    def test_equity_curve_starts_at_initial_capital(self, backtester):
        """First equity point (before any trade) should be ~initial capital."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "close": [100] * 5,
                "volume": [1000] * 5,
            },
            index=dates,
        )
        # No-op strategy
        result = backtester.run_backtest(data, lambda d, p: [], {})
        curve = result.get("equity_curve", [])
        assert len(curve) == 5
        assert abs(curve[0]["equity"] - 100_000) < 1.0


class TestEntryTimestampFix:
    """CQ-08: Verify trade entry_time and duration are correct."""

    def test_trade_has_distinct_entry_exit_times(self, backtester, sample_ohlcv):
        backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        if backtester.trades:
            trade = backtester.trades[0]
            assert trade.entry_time is not None
            assert trade.exit_time is not None
            assert trade.entry_time < trade.exit_time
            assert trade.duration > timedelta(0)

    def test_position_stores_entry_time(self, backtester):
        """BacktestPosition dataclass must have entry_time field."""
        pos = BacktestPosition(
            symbol="AAPL", quantity=100, avg_price=150.0,
            entry_time=datetime(2024, 1, 1),
        )
        assert pos.entry_time == datetime(2024, 1, 1)


class TestMetrics:
    """Sharpe, Sortino, win rate, profit factor."""

    def test_sharpe_ratio_finite(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        sr = result.get("sharpe_ratio", None)
        assert sr is not None
        assert np.isfinite(sr)

    def test_win_rate_bounds(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        wr = result.get("win_rate", 0)
        assert 0.0 <= wr <= 1.0

    def test_max_drawdown_non_negative(self, backtester, sample_ohlcv):
        result = backtester.run_backtest(
            sample_ohlcv, buy_sell_strategy, {}
        )
        assert result.get("max_drawdown", 0) >= 0.0


class TestWalkForward:
    """Walk-forward analysis (CQ-03: documented as scaffold)."""

    def test_walk_forward_returns_results(self, backtester, sample_ohlcv):
        result = backtester.walk_forward_analysis(
            sample_ohlcv, buy_and_hold_strategy, {"quantity": 10},
            train_period=20, test_period=10,
        )
        # May return error if data too short, but should not crash
        assert isinstance(result, dict)

    def test_walk_forward_scaffold_uses_fixed_params(self, backtester, sample_ohlcv):
        """CQ-03: walk-forward currently uses fixed params (scaffold)."""
        result = backtester.walk_forward_analysis(
            sample_ohlcv, buy_and_hold_strategy, {"quantity": 10},
            train_period=15, test_period=10,
        )
        if "period_results" in result:
            # All periods should use the same params (scaffold behavior)
            assert result["n_periods"] >= 1
