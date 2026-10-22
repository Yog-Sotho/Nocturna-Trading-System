"""
NOCTURNA Trading System - Candidates the operator can validate.

Each entry names the symbols a candidate trades, how much history it needs
before its first decision, and builds it from the instruments Alpaca reports.
The grids are deliberately small: every extra parameter set is a trial the
Deflated Sharpe ratio and the overfitting test charge for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..domain import AssetClass, Instrument
from ..strategies.benchmark import BuyAndHold
from ..strategies.blend import Blend
from ..strategies.crypto_trend import DonchianEnsemble
from ..strategies.month_end import RULES, MonthEndCycle
from ..strategies.rebalancing import RebalancingFlows
from ..strategies.trend import TimeSeriesMomentum
from ..strategy import Strategy
from .experiment import Candidate

#: A diversified, liquid ETF basket with Alpaca history from 2016: US large and small caps,
#: developed and emerging equities, long and intermediate Treasuries, investment-grade and
#: high-yield credit, gold, commodities, real estate and the US dollar.
TREND_ETFS = ("SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ", "UUP")
CRYPTO = ("BTC/USD", "ETH/USD", "SOL/USD")


@dataclass(frozen=True)
class CandidateSpec:
    description: str
    symbols: tuple[tuple[str, AssetClass], ...]
    build: Callable[[Mapping[str, Instrument]], Candidate]
    #: Calendar days of bars to load before the first trading day
    warm_up_days: int = 400


def _benchmark(instruments: Mapping[str, Instrument]) -> Candidate:
    spy, ief = instruments["SPY"], instruments["IEF"]
    return Candidate(
        "benchmark",
        lambda p: BuyAndHold({spy: p["equity"], ief: round(1 - p["equity"], 6)}),
        {"equity": [0.4, 0.5, 0.6, 0.7, 0.8]},
        regime_symbol="SPY",
    )


def _rebalancing(instruments: Mapping[str, Instrument]) -> Candidate:
    spy, ief = instruments["SPY"], instruments["IEF"]
    return Candidate(
        "rebalancing",
        lambda p: RebalancingFlows(spy, ief, mode=p["mode"], scale=p["scale"]),
        {"mode": ["spread", "tilt"], "scale": [0.5, 1.0]},
        regime_symbol="SPY",
    )


def _tsmom(instruments: Mapping[str, Instrument]) -> Candidate:
    basket = [instruments[s] for s in TREND_ETFS]
    return Candidate(
        "tsmom",
        lambda p: TimeSeriesMomentum(basket, rule=p["rule"], long_only=p["long_only"]),
        {"rule": ["sign", "tstat"], "long_only": [False, True]},
        regime_symbol="SPY",
    )


def _crypto_trend(instruments: Mapping[str, Instrument]) -> Candidate:
    coins = [instruments[s] for s in CRYPTO]
    return Candidate(
        "crypto-trend",
        lambda p: DonchianEnsemble(coins, target_volatility=p["target_volatility"]),
        {"target_volatility": [0.25]},  # the paper's parameters, nothing tuned
        regime_symbol="BTC/USD",
    )


def _month_end(instruments: Mapping[str, Instrument]) -> Candidate:
    spy = instruments["SPY"]
    return Candidate(
        "month-end",
        lambda p: MonthEndCycle(spy, rule=p["rule"]),
        {"rule": list(RULES)},  # fixed before any backtest (docs/STRATEGY_SHORTLIST.md)
        regime_symbol="SPY",
    )


#: Lineup mixes (core 60/40, ETF trend, crypto trend), fixed from first principles before
#: any lineup backtest: the core carries the return, trend diversifies it, crypto is held at
#: the 15% crypto limit
LINEUP_MIXES = {"50/35/15": (0.50, 0.35, 0.15), "60/25/15": (0.60, 0.25, 0.15)}


def _lineup(instruments: Mapping[str, Instrument]) -> Candidate:
    spy, ief = instruments["SPY"], instruments["IEF"]
    basket = [instruments[s] for s in TREND_ETFS]
    coins = [instruments[s] for s in CRYPTO]

    def build(p: Mapping[str, Any]) -> Strategy:
        core, trend, crypto = LINEUP_MIXES[p["mix"]]
        return Blend("lineup", [
            (BuyAndHold({spy: 0.6, ief: 0.4}), core),
            (TimeSeriesMomentum(basket, rule="sign", long_only=p["trend_long_only"]), trend),
            (DonchianEnsemble(coins), crypto),
        ])

    return Candidate("lineup", build, {"mix": list(LINEUP_MIXES), "trend_long_only": [False, True]},
                     regime_symbol="SPY")


CANDIDATES: dict[str, CandidateSpec] = {
    "benchmark": CandidateSpec(
        "SPY/IEF buy-and-hold, equity weight 40-80%",
        (("SPY", AssetClass.EQUITY), ("IEF", AssetClass.EQUITY)),
        _benchmark,
    ),
    "rebalancing": CandidateSpec(
        "front-running 60/40 rebalancing flows in SPY/IEF (Harvey, Mazzoleni and Melone)",
        (("SPY", AssetClass.EQUITY), ("IEF", AssetClass.EQUITY)),
        _rebalancing,
    ),
    "tsmom": CandidateSpec(
        "12-month time-series momentum on 12 ETFs (Moskowitz, Ooi and Pedersen)",
        tuple((s, AssetClass.EQUITY) for s in TREND_ETFS),
        _tsmom,
    ),
    "crypto-trend": CandidateSpec(
        "Donchian breakout ensemble on BTC, ETH and SOL (Zarattini, Pagani and Barbon)",
        tuple((s, AssetClass.CRYPTO) for s in CRYPTO),
        _crypto_trend,
        warm_up_days=760,
    ),
    "month-end": CandidateSpec(
        "SPY around each month's last session, T-bills otherwise (Etula, Rinne, Suominen and Vaittinen)",
        (("SPY", AssetClass.EQUITY),),
        _month_end,
    ),
    "lineup": CandidateSpec(
        "60/40 core + ETF time-series momentum + crypto Donchian ensemble",
        tuple((s, AssetClass.EQUITY) for s in TREND_ETFS) + tuple((s, AssetClass.CRYPTO) for s in CRYPTO),
        _lineup,
    ),
}
