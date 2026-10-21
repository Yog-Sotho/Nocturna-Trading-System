"""
NOCTURNA Trading System - Candidates the operator can validate.

Each entry names the symbols a candidate trades and builds it from the
instruments Alpaca reports. Phase 5 adds the strategy lineup; until then the
60/40 benchmark family exercises the pipeline (choosing its equity weight by
backtest is exactly the kind of selection the pipeline should expose).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..domain import AssetClass, Instrument
from ..strategies.benchmark import BuyAndHold
from .experiment import Candidate


@dataclass(frozen=True)
class CandidateSpec:
    description: str
    symbols: tuple[tuple[str, AssetClass], ...]
    build: Callable[[Mapping[str, Instrument]], Candidate]


def _benchmark(instruments: Mapping[str, Instrument]) -> Candidate:
    spy, ief = instruments["SPY"], instruments["IEF"]
    return Candidate(
        "benchmark",
        lambda p: BuyAndHold({spy: p["equity"], ief: round(1 - p["equity"], 6)}),
        {"equity": [0.4, 0.5, 0.6, 0.7, 0.8]},
        regime_symbol="SPY",
    )


CANDIDATES: dict[str, CandidateSpec] = {
    "benchmark": CandidateSpec(
        "SPY/IEF buy-and-hold, equity weight 40-80%",
        (("SPY", AssetClass.EQUITY), ("IEF", AssetClass.EQUITY)),
        _benchmark,
    ),
}
