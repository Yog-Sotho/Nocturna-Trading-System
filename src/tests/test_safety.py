"""
Tests for the paper-only trading guard (Phase 0 safety freeze).
"""

import pytest

from src.core.order_manager import OrderExecutionManager
from src.core.safety import LiveTradingDisabledError, assert_paper_only
from src.main import create_app


class TestAssertPaperOnly:
    """Configuration checks that block live trading."""

    def test_defaults_to_paper(self):
        assert_paper_only({})

    def test_paper_mode_with_paper_endpoint_allowed(self):
        assert_paper_only({
            "TRADING_MODE": "paper",
            "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        })

    @pytest.mark.parametrize("mode", ["LIVE", "live", "REAL", ""])
    def test_non_paper_mode_rejected(self, mode):
        with pytest.raises(LiveTradingDisabledError, match="live trading is disabled"):
            assert_paper_only({"TRADING_MODE": mode})

    @pytest.mark.parametrize("url", [
        "https://api.alpaca.markets",
        "https://paper-api.alpaca.markets.evil.example",
        "http://localhost:8080",
    ])
    def test_non_paper_endpoint_rejected(self, url):
        with pytest.raises(LiveTradingDisabledError, match="not the Alpaca paper endpoint"):
            assert_paper_only({"TRADING_MODE": "PAPER", "ALPACA_BASE_URL": url})


class TestGuardIsEnforced:
    """The guard runs at app startup and before any broker client is built."""

    def test_create_app_refuses_live_mode(self, monkeypatch):
        monkeypatch.setenv("TRADING_MODE", "LIVE")
        with pytest.raises(LiveTradingDisabledError):
            create_app()

    def test_order_manager_refuses_live_endpoint(self, monkeypatch):
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        with pytest.raises(LiveTradingDisabledError):
            OrderExecutionManager({})
