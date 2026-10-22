"""
NOCTURNA Trading System - Position sizing.

Turns the strategies' combined target weights into the weights the engine
trades, in three steps:

1. **Caps.** Each weight is clipped to the per-position limit, each asset
   class is scaled into its limit, and the whole book into the gross limit.
2. **Volatility target.** The book is scaled down (never up) so its
   predicted volatility does not exceed the target. Prediction uses an EWMA
   covariance of daily log returns (RiskMetrics, lambda = 0.94), estimated on
   the dates every held instrument traded. Volatility scaling improves the
   Sharpe ratio of risk assets and of balanced portfolios and thins the left
   tail of all assets (Harvey et al., "The Impact of Volatility Targeting",
   JPM 2018).
3. **Drawdown ladder.** Beyond ``derisk_start`` the book shrinks linearly
   with the cushion above the drawdown limit, as in Grossman and Zhou,
   "Optimal Investment Strategies for Controlling Drawdowns" (1993), down to
   ``derisk_floor`` at the limit. The floor is deliberate: shrinking to zero
   (the textbook rule) would park the book in cash just short of the limit,
   where it can never earn its way back and no one is told (CPPI's "cash
   lock"). With the floor, losses that continue reach the limit and trip the
   kill switch, which flattens and halts until the operator resumes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import lru_cache

import numpy as np

from .domain import NEW_YORK, AssetClass, Bar, Instrument
from .risk import RiskLimits

ZERO = Decimal("0")
ONE = Decimal("1")
SCALE = Decimal("1e-6")


@dataclass(frozen=True)
class SizingPolicy:
    #: Annualized portfolio volatility target; None turns volatility targeting off
    target_volatility: Decimal | None = Decimal("0.08")
    ewma_lambda: float = 0.94
    #: Returns needed before the volatility target applies (until then: no scaling)
    min_history: int = 60
    #: Drawdown from the high-water mark where de-risking starts; None turns it off
    derisk_start: Decimal | None = Decimal("0.05")
    #: Fraction of the book kept when the drawdown reaches the limit
    derisk_floor: Decimal = Decimal("0.25")
    #: Adjust a position only when it is this far (fraction of equity) from its target
    rebalance_band: Decimal = Decimal("0.02")

    def __post_init__(self) -> None:
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        if not 0 < self.ewma_lambda < 1:
            raise ValueError("ewma_lambda must be between 0 and 1")
        if self.min_history < 2:
            raise ValueError("min_history must be at least 2")
        if self.rebalance_band < 0:
            raise ValueError("rebalance_band must not be negative")
        if not 0 <= self.derisk_floor <= 1:
            raise ValueError("derisk_floor must be between 0 and 1")

    @classmethod
    def unscaled(cls, rebalance_band: Decimal = Decimal("0.002")) -> SizingPolicy:
        """Caps only: no volatility target, no drawdown ladder."""
        return cls(target_volatility=None, derisk_start=None, rebalance_band=rebalance_band)

    @property
    def lookback(self) -> int:
        """Bars of history the estimator needs."""
        return max(self.min_history + 1, 252)


@dataclass(frozen=True)
class SizingResult:
    weights: dict[str, Decimal]
    capped_weights: dict[str, Decimal]
    volatility: float | None  # predicted annualized volatility of the capped book
    volatility_scale: Decimal
    drawdown_scale: Decimal


def cap_weights(targets: Mapping[str, Decimal], instruments: Mapping[str, Instrument],
                limits: RiskLimits) -> dict[str, Decimal]:
    """Clip to the position limit, then scale asset classes and the book into their limits."""
    capped = {s: max(-limits.max_position_weight, min(limits.max_position_weight, w))
              for s, w in targets.items()}
    for asset_class, cap in limits.max_asset_class_weight.items():
        members = [s for s in capped if instruments[s].asset_class is asset_class]
        total = sum((abs(capped[s]) for s in members), ZERO)
        if total > cap:
            for s in members:
                capped[s] = (capped[s] * cap / total).quantize(SCALE, rounding="ROUND_DOWN")
    gross = sum((abs(w) for w in capped.values()), ZERO)
    if gross > limits.max_gross_exposure:
        capped = {s: (w * limits.max_gross_exposure / gross).quantize(SCALE, rounding="ROUND_DOWN")
                  for s, w in capped.items()}
    return capped


def stop_risk_weight(price: float, stop: float, risk: float = 0.005) -> float:
    """
    Fixed-fractional sizing for a strategy with a protective stop: the weight
    that loses ``risk`` of equity (0.5% by default) if the price moves from
    ``price`` to ``stop``. Long when the stop is below the price, short when it
    is above. A gap through the stop loses more; the caps above still apply.

    Raises:
        ValueError: For non-positive prices, a stop at the price, or a risk outside (0, 1).
    """
    if price <= 0 or stop <= 0 or stop == price:
        raise ValueError("price and stop must be positive and different")
    if not 0 < risk < 1:
        raise ValueError("risk must be between 0 and 1")
    return risk * price / (price - stop)


@lru_cache(maxsize=1 << 16)  # pragma: no mutate - the cache size changes speed, not results
def _day(start: datetime, asset_class: AssetClass) -> date:
    # Daily bars start at midnight: New York for equities, UTC for crypto
    zone = NEW_YORK if asset_class is AssetClass.EQUITY else UTC
    return start.astimezone(zone).date()


def _session_date(bar: Bar, asset_class: AssetClass) -> date:
    return _day(bar.start, asset_class)


def ewma_covariance(history: Mapping[str, Sequence[Bar]], instruments: Mapping[str, Instrument],
                    symbols: Sequence[str], lam: float, min_history: int) -> tuple[np.ndarray, float] | None:
    """
    EWMA covariance of daily log returns, and the number of return periods per
    year, on the dates every symbol has a close. None when there is too little
    common history.
    """
    closes = []
    for symbol in symbols:
        closes.append({_session_date(b, instruments[symbol].asset_class): b.close for b in history.get(symbol, ())})
    common = sorted(set.intersection(*(set(c) for c in closes)))
    if len(common) < min_history + 1:
        return None
    prices = np.array([[c[d] for c in closes] for d in common], dtype=float)
    returns = np.diff(np.log(prices), axis=0)
    seed = min(20, len(returns))
    cov = np.cov(returns[:seed].T, ddof=1).reshape(len(symbols), len(symbols))
    # The recursion cov <- lam * cov + (1 - lam) * r r' over the remaining returns, in closed form
    tail = returns[seed:]
    if len(tail):
        weights = (1 - lam) * lam ** np.arange(len(tail) - 1, -1, -1)
        cov = lam ** len(tail) * cov + (tail * weights[:, None]).T @ tail
    # At least min_history + 1 >= 3 distinct dates, so the span is never zero
    years = (common[-1] - common[0]).days / 365.25
    return cov, len(returns) / years


class PortfolioSizer:
    def __init__(self, policy: SizingPolicy, limits: RiskLimits) -> None:
        self.policy = policy
        self.limits = limits

    @property
    def lookback(self) -> int:
        return self.policy.lookback if self.policy.target_volatility is not None else 1

    def drawdown_scale(self, drawdown: Decimal) -> Decimal:
        start, floor = self.policy.derisk_start, self.policy.derisk_floor
        limit = self.limits.max_drawdown
        if start is None or drawdown <= start:
            return ONE
        if drawdown >= limit:
            return floor
        cushion = (limit - drawdown) / (limit - start)
        return (floor + (ONE - floor) * cushion).quantize(SCALE, rounding="ROUND_DOWN")

    def predicted_volatility(self, weights: Mapping[str, Decimal], history: Mapping[str, Sequence[Bar]],
                             instruments: Mapping[str, Instrument]) -> float | None:
        held = [s for s, w in weights.items() if w]
        if not held:
            return 0.0
        estimate = ewma_covariance(history, instruments, held, self.policy.ewma_lambda, self.policy.min_history)
        if estimate is None:
            return None
        cov, periods = estimate
        w = np.array([float(weights[s]) for s in held])
        volatility = float(np.sqrt(max(w @ cov @ w, 0.0) * periods))
        return volatility if math.isfinite(volatility) else None

    def size(self, targets: Mapping[str, Decimal], history: Mapping[str, Sequence[Bar]],
             instruments: Mapping[str, Instrument], drawdown: Decimal) -> SizingResult:
        capped = cap_weights(targets, instruments, self.limits)
        volatility, vol_scale = None, ONE
        if self.policy.target_volatility is not None:
            volatility = self.predicted_volatility(capped, history, instruments)
            if volatility:
                target = float(self.policy.target_volatility)
                vol_scale = min(ONE, Decimal(repr(target / volatility)).quantize(SCALE, rounding="ROUND_DOWN"))
        dd_scale = self.drawdown_scale(drawdown)
        scale = vol_scale * dd_scale
        weights = {s: (w * scale).quantize(SCALE, rounding="ROUND_DOWN") for s, w in capped.items()}
        return SizingResult(weights, capped, volatility, vol_scale, dd_scale)
