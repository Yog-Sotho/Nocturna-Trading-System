"""
NOCTURNA Trading System - The strategies an operator can trade, in plain words.

Every registered validation candidate (``validation/candidates.py``) can be
put in the trading profile's mix. This module says what each one does, how
its parameters read on the dashboard, which parameters it starts with, and
what validation concluded about it (docs/ROADMAP.md, "Strategy lineup";
docs/STRATEGY_SHORTLIST.md). Parameters are the candidate's own grid, so a
choice is always one that validation tested.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .domain import Instrument
from .validation.candidates import CANDIDATES
from .validation.experiment import Candidate

DOCS = "https://yog-sotho.github.io/Nocturna-Trading-System/"
#: The verdict of the strategy the engine trades when nothing else is chosen
BENCHMARK = "benchmark"
REJECTED = "rejected"


@dataclass(frozen=True)
class Parameter:
    label: str
    #: How each grid value reads (values without an entry show as they are)
    choices: Mapping[Any, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyInfo:
    title: str
    summary: str
    source: str
    verdict: str
    verdict_detail: str
    defaults: Mapping[str, Any]
    parameters: Mapping[str, Parameter]
    doc: str = DOCS + "ROADMAP/#strategy-lineup"


def _percent(*values: float) -> dict[Any, str]:
    return {v: f"{v:.0%}" for v in values}


STRATEGIES: dict[str, StrategyInfo] = {
    "benchmark": StrategyInfo(
        "60/40 stocks and bonds",
        "Holds SPY and IEF at fixed weights, rebalanced as prices drift.",
        "The yardstick every idea must beat",
        BENCHMARK,
        "Trades on paper today. Held at 60/40 it earned 6.3% a year from 2020 to October 2026 (worst drop "
        "-12%), and 8.1% a year since 1974 on free long-history data.",
        {"equity": 0.6},
        {"equity": Parameter("Stocks (SPY); the rest is bonds (IEF)", _percent(0.4, 0.5, 0.6, 0.7, 0.8))},
    ),
    "rebalancing": StrategyInfo(
        "Front-running pension rebalancing",
        "Trades SPY against IEF ahead of the flows of 60/40 funds that rebalance at month-end or past a "
        "threshold.",
        "Harvey, Mazzoleni and Melone, NBER w33554 (2025)",
        REJECTED,
        "The signal works between closes but fades by the next open, and it churns the portfolio about 43 "
        "times a year.",
        {"mode": "tilt", "scale": 0.5},
        {"mode": Parameter("Form", {"spread": "Long SPY, short IEF", "tilt": "Tilt a 60/40 (long only)"}),
         "scale": Parameter("Signal strength", _percent(0.5, 1.0))},
    ),
    "tsmom": StrategyInfo(
        "Trend following on 12 ETFs",
        "Each month, holds every ETF long (or short) by the sign of its 12-month return, sized to a constant "
        "volatility.",
        "Moskowitz, Ooi and Pedersen, JFE 2012",
        REJECTED,
        "Too weak out of sample.",
        {"rule": "sign", "long_only": True},
        {"rule": Parameter("Trend signal", {"sign": "Sign of the 12-month return",
                                            "tstat": "Strength of the trend (t-statistic)"}),
         "long_only": Parameter("Long only (cash instead of shorts)", {True: "Yes", False: "No"})},
    ),
    "crypto-trend": StrategyInfo(
        "Crypto breakouts",
        "Nine breakout channels (5 to 360 days) on BTC, ETH and SOL, each with a trailing stop, long only.",
        "Zarattini, Pagani and Barbon, SFI 25-80 (2025)",
        REJECTED,
        "Too weak out of sample (2024 to 2026; Alpaca's crypto data starts in 2021).",
        {"target_volatility": 0.25},
        {"target_volatility": Parameter("Volatility per coin", _percent(0.25))},
    ),
    "month-end": StrategyInfo(
        "Month-end payment cycle",
        "Holds SPY from a few sessions before each month's last session to three after it, T-bills otherwise; "
        "trades in the closing auction.",
        "Etula, Rinne, Suominen and Vaittinen, RFS 2020",
        REJECTED,
        "21.4% from 2020 to October 2026, against T-bills' 20.3%.",
        {"rule": "settlement"},
        {"rule": Parameter("Window", {"paper": "From T-4 (the paper's window)",
                                      "settlement": "Follows the settlement cycle",
                                      "turn": "From T-1 (turn of the month)"})},
    ),
    "trend-filter": StrategyInfo(
        "Trend filter on the 60/40",
        "The 60/40 with SPY held only while its month-end close is above its moving average, T-bills "
        "otherwise.",
        "Faber, JWM 2007",
        REJECTED,
        "Tied with the 60/40 since 2017 and a little behind it since 1974, though it hit the kill switch "
        "twice in 52 years instead of five times.",
        {"months": 10},
        {"months": Parameter("Moving average", {8: "8 months", 10: "10 months", 12: "12 months"})},
    ),
    "low-vol": StrategyInfo(
        "Low-volatility funds instead of SPY",
        "The 60/40 with a low-volatility stock fund in place of SPY.",
        "Blitz and van Vliet, JPM 2007; Baker, Bradley and Wurgler, FAJ 2011",
        REJECTED,
        "3.6% a year from 2020 to October 2026, against the 60/40's 6.3%.",
        {"equity": "USMV"},
        {"equity": Parameter("Fund", {"USMV": "USMV (iShares MSCI USA Min Vol)",
                                      "SPLV": "SPLV (Invesco S&P 500 Low Volatility)"})},
    ),
    "lineup": StrategyInfo(
        "60/40 + ETF trend + crypto",
        "A 60/40 core with ETF trend following and crypto breakouts, held at fixed shares.",
        "Phase 5 lineup",
        REJECTED,
        "Didn't beat the plain 60/40 out of sample.",
        {"mix": "60/25/15", "trend_long_only": True},
        {"mix": Parameter("Core / trend / crypto"),
         "trend_long_only": Parameter("Trend part long only", {True: "Yes", False: "No"})},
    ),
}


def placeholder_instruments(name: str) -> dict[str, Instrument]:
    """Instruments for every symbol a candidate trades, enough to build it and read its grid."""
    return {s: Instrument(s, a) for s, a in CANDIDATES[name].symbols}


def candidate(name: str, instruments: Mapping[str, Instrument] | None = None) -> Candidate:
    """
    Raises:
        KeyError: For a name that is not registered.
    """
    return CANDIDATES[name].build(instruments if instruments is not None else placeholder_instruments(name))


def catalog() -> list[dict[str, Any]]:
    """Every registered strategy, in registration order, as the dashboard shows it."""
    entries = []
    for name, spec in CANDIDATES.items():
        info = STRATEGIES[name]
        grid = candidate(name).grid
        execution = candidate(name).factory(dict(info.defaults)).execution
        entries.append({
            "name": name, "title": info.title, "summary": info.summary, "source": info.source,
            "verdict": info.verdict, "verdict_detail": info.verdict_detail, "doc": info.doc,
            "execution": execution, "symbols": [s for s, _ in spec.symbols],
            "asset_classes": sorted({a.value for _, a in spec.symbols}),
            "defaults": dict(info.defaults),
            "parameters": [
                {"name": key, "label": info.parameters[key].label,
                 "choices": [{"value": v, "label": info.parameters[key].choices.get(v, str(v))} for v in values]}
                for key, values in grid.items()],
        })
    return entries
