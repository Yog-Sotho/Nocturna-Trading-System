"""
Tests for the risk management module.
Covers: validate_trade, drawdown tracking, portfolio monitoring, risk levels.
"""

import numpy as np
import pytest
from src.core.risk_manager import RiskLevel, RiskManager


@pytest.fixture()
def risk_config():
    return {
        "initial_capital": 100_000,
        "max_position_size": 0.10,
        "max_portfolio_risk": 0.02,
        "max_drawdown": 0.15,
        "max_daily_loss": 0.03,
        "max_correlation": 0.7,
        "volatility_lookback": 20,
    }


@pytest.fixture()
def risk_manager(risk_config):
    return RiskManager(risk_config)


class TestRiskManagerInit:
    """Initialization and parameter loading."""

    def test_initial_risk_level_is_low(self, risk_manager):
        assert risk_manager.current_risk_level == RiskLevel.LOW

    def test_portfolio_value_set(self, risk_manager):
        assert risk_manager.portfolio_value == 100_000

    def test_drawdown_starts_at_zero(self, risk_manager):
        assert risk_manager.current_drawdown == 0.0
        assert risk_manager.max_drawdown == 0.0


class TestDrawdownTracking:
    """Drawdown via update_portfolio_value (tracks peak)."""

    def test_update_portfolio_tracks_peak(self, risk_manager):
        risk_manager.update_portfolio_value(110_000)
        assert risk_manager.max_portfolio_value == 110_000

    def test_peak_not_overwritten_on_decline(self, risk_manager):
        risk_manager.update_portfolio_value(110_000)
        risk_manager.update_portfolio_value(100_000)
        assert risk_manager.max_portfolio_value == 110_000

    def test_portfolio_value_updated(self, risk_manager):
        risk_manager.update_portfolio_value(95_000)
        assert risk_manager.portfolio_value == 95_000


class TestValidateTrade:
    """Trade signal validation against risk controls."""

    def test_valid_signal_returns_tuple(self, risk_manager):
        signal = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "type": "market",
            "price": 150.0,
        }
        positions = {}
        market_data = {"close": 150.0, "high": 151.0, "low": 149.0, "volume": 1000}
        result = risk_manager.validate_trade(signal, positions, market_data)
        assert isinstance(result, tuple)
        assert len(result) == 3
        valid, reason, adjusted = result
        assert isinstance(valid, bool)
        assert isinstance(reason, str)
        assert isinstance(adjusted, dict)

    def test_missing_fields_rejected(self, risk_manager):
        signal = {"symbol": "AAPL"}
        valid, reason, _ = risk_manager.validate_trade(signal, {}, {})
        assert valid is False

    def test_negative_quantity_rejected(self, risk_manager):
        signal = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": -10,
            "type": "market",
            "price": 150.0,
        }
        valid, reason, _ = risk_manager.validate_trade(signal, {}, {"close": 150.0})
        assert valid is False


class TestRecordTrade:
    """Trade recording and daily stats."""

    def test_record_trade_updates_daily_stats(self, risk_manager):
        trade = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "fill_price": 150.0,
            "pnl": -50.0,
        }
        risk_manager.record_trade(trade)
        assert len(risk_manager.daily_stats["trades"]) == 1

    def test_multiple_trades_accumulate(self, risk_manager):
        for _ in range(3):
            risk_manager.record_trade({
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 10,
                "fill_price": 150.0,
                "pnl": -10.0,
            })
        assert len(risk_manager.daily_stats["trades"]) == 3


class TestPortfolioMonitoring:
    """Portfolio-level risk monitoring."""

    def test_monitor_returns_dict(self, risk_manager):
        positions = {
            "AAPL": {"quantity": 100, "avg_price": 150.0, "current_price": 155.0},
        }
        market_data = {
            "AAPL": {"close": 155.0, "returns": np.random.randn(20) * 0.01}
        }
        result = risk_manager.monitor_portfolio_risk(positions, market_data)
        assert isinstance(result, dict)
        assert "risk_level" in result

    def test_risk_report_structure(self, risk_manager):
        report = risk_manager.get_risk_report()
        assert isinstance(report, dict)
        assert "current_risk_level" in report
        assert "portfolio_metrics" in report
        assert "max_drawdown" in report["portfolio_metrics"]


class TestRiskCallbacks:
    """Observer pattern for risk events."""

    def test_add_callback(self, risk_manager):
        events = []
        risk_manager.add_risk_callback(lambda e: events.append(e))
        assert len(risk_manager._risk_callbacks) >= 1
