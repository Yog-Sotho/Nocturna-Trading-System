"""
NOCTURNA Trading System - The trading profile: what the engine trades, and within which limits.

The operator sets it on the dashboard:

* the **mix**: registered strategies (``catalog.py``), each with parameters
  from its validation grid and a share of the account. The shares need not
  add up to 100%: the engine's cash sleeve holds the rest in T-bills;
* the **risk limits**: overrides of the environment's ``RISK_*`` and
  ``SIZING_*`` settings, each within the hard bounds below. Leverage is never
  configurable (gross exposure stays at most 100%).

Every save is a new version (``trading_profiles``). The engine takes the
newest at its next start and stamps it ``activated_at`` the first time it
trades it; the soak report measures from that moment. With nothing saved the
engine trades ``DEFAULT_PROFILE``: the 60/40 with the environment's limits.

Each strategy of the mix runs inside a ``Blend`` named after it, so two
strategies of the same class (the 60/40 and low-vol are both buy-and-hold)
keep apart, and the engine's targets are labelled by what the operator chose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from flask import Flask
from sqlalchemy import select

from src.config import Settings

from .catalog import candidate
from .domain import AssetClass, Instrument
from .engine import Allocation
from .lineup import CASH_SLEEVE
from .strategies.blend import Blend
from .strategy import CLOSE, common_execution
from .validation.candidates import CANDIDATES

ZERO, ONE = Decimal(0), Decimal(1)
#: The engine loads at least this much history (EngineService's own default)
MIN_WARM_UP_DAYS = 550
MAX_NOTE = 200


class ProfileError(ValueError):
    """A profile the engine cannot trade; the message says why, for the operator."""


@dataclass(frozen=True)
class RiskField:
    label: str
    kind: str  # fraction, integer, boolean or dollars
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    #: A value outside [minimum, maximum] that is also allowed: 0 turns the control off
    off: Decimal | None = None
    #: None is allowed (no dollar limit)
    optional: bool = False


#: The settings the dashboard may change, by their Settings field names, with hard bounds
RISK_FIELDS: dict[str, RiskField] = {
    "risk_max_drawdown": RiskField("Kill switch: halt at this drop below the peak", "fraction",
                                   Decimal("0.02"), Decimal("0.25")),
    "risk_flatten_on_kill": RiskField("Close every position when the kill switch trips", "boolean"),
    "risk_daily_loss_limit": RiskField("Daily loss limit: no new risk for the rest of the day", "fraction",
                                       Decimal("0.005"), Decimal("0.10")),
    "sizing_derisk_start": RiskField("Start shrinking exposure at this drop (0 turns it off)", "fraction",
                                     Decimal("0.01"), Decimal("0.24"), off=ZERO),
    "sizing_target_volatility": RiskField("Volatility target, a year (0 turns it off)", "fraction",
                                          Decimal("0.02"), Decimal("0.20"), off=ZERO),
    "risk_max_position_weight": RiskField("Largest position", "fraction", Decimal("0.05"), ONE),
    "risk_max_crypto_weight": RiskField("All crypto together", "fraction", Decimal("0.01"), Decimal("0.30")),
    "risk_max_orders_per_day": RiskField("Orders per day", "integer", ONE, Decimal(200)),
    "risk_max_order_notional": RiskField("Largest single order, in dollars (empty: no dollar limit)", "dollars",
                                         Decimal(100), Decimal(10_000_000), optional=True),
    "sizing_rebalance_band": RiskField("Adjust a position only when this far from its target", "fraction",
                                       Decimal("0.005"), Decimal("0.10")),
}


@dataclass(frozen=True)
class MixEntry:
    strategy: str
    params: Mapping[str, Any]
    weight: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "params": dict(self.params), "weight": str(self.weight)}


@dataclass(frozen=True)
class Profile:
    mix: tuple[MixEntry, ...]
    #: Overrides of the environment's settings, by RISK_FIELDS name
    risk: Mapping[str, Any] = field(default_factory=dict)
    id: int | None = None
    note: str = ""
    created_by: str | None = None
    created_at: datetime | None = None
    activated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "mix": [e.to_dict() for e in self.mix],
            "risk": {k: _json(v) for k, v in self.risk.items()}, "note": self.note, "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
        }


DEFAULT_PROFILE = Profile(mix=(MixEntry("benchmark", {"equity": 0.6}, ONE),))


def _json(value: Any) -> Any:
    return str(value) if isinstance(value, Decimal) else value


# -- validation -----------------------------------------------------------------------


def _decimal(value: Any, what: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ProfileError(f"{what} must be a number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as e:
        raise ProfileError(f"{what} must be a number") from e
    if not number.is_finite():
        raise ProfileError(f"{what} must be a number")
    return number


def _grid_value(strategy: str, key: str, value: Any, choices: Sequence[Any]) -> Any:
    """The grid's own value equal to ``value`` (a JSON 1 for a grid's 1.0 is accepted)."""
    for choice in choices:
        if isinstance(choice, bool) or isinstance(value, bool):
            if choice is value:  # True is never the number 1 here
                return choice
        elif (type(choice) is type(value) and choice == value) or (
                isinstance(choice, float) and isinstance(value, int) and float(value) == choice):
            return choice
    readable = ", ".join(str(c) for c in choices)
    raise ProfileError(f"{strategy}: {key} must be one of {readable}")


def _entry(data: Any) -> MixEntry:
    if not isinstance(data, Mapping):
        raise ProfileError("each strategy of the mix must be an object")
    name = data.get("strategy")
    if name not in CANDIDATES:
        raise ProfileError(f"unknown strategy {name!r}; known: {', '.join(CANDIDATES)}")
    grid = candidate(name).grid
    params = data.get("params", {})
    if not isinstance(params, Mapping) or set(params) != set(grid):
        raise ProfileError(f"{name}: give exactly these parameters: {', '.join(grid)}")
    chosen = {key: _grid_value(name, key, params[key], grid[key]) for key in grid}
    weight = _decimal(data.get("weight"), f"{name}: weight")
    if not ZERO < weight <= ONE:
        raise ProfileError(f"{name}: weight must be above 0% and at most 100%")
    return MixEntry(name, chosen, weight.quantize(Decimal("0.0001")))


def _risk(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ProfileError("risk must be an object")
    unknown = sorted(set(data) - set(RISK_FIELDS))
    if unknown:
        raise ProfileError(f"unknown risk setting {unknown[0]}; known: {', '.join(RISK_FIELDS)}")
    risk: dict[str, Any] = {}
    for key, value in data.items():
        spec = RISK_FIELDS[key]
        if spec.kind == "boolean":
            if not isinstance(value, bool):
                raise ProfileError(f"{spec.label}: must be true or false")
            risk[key] = value
            continue
        if value is None:
            if not spec.optional:
                raise ProfileError(f"{spec.label}: a value is required")
            risk[key] = None
            continue
        number = _decimal(value, spec.label)
        if spec.kind == "integer" and number != number.to_integral_value():
            raise ProfileError(f"{spec.label}: must be a whole number")
        inside = spec.minimum is not None and spec.maximum is not None and spec.minimum <= number <= spec.maximum
        if not inside and number != spec.off:
            off = f", or {spec.off} to turn it off" if spec.off is not None else ""
            raise ProfileError(f"{spec.label}: must be between {spec.minimum} and {spec.maximum}{off}")
        risk[key] = int(number) if spec.kind == "integer" else number
    return risk


def parse(data: Any) -> Profile:
    """
    A profile from the dashboard's JSON.

    Raises:
        ProfileError: For anything the engine could not trade.
    """
    if not isinstance(data, Mapping):
        raise ProfileError("the profile must be an object")
    mix_data = data.get("mix")
    if not isinstance(mix_data, list) or not mix_data:
        raise ProfileError("the mix needs at least one strategy")
    mix = tuple(_entry(e) for e in mix_data)
    names = [e.strategy for e in mix]
    if len(set(names)) != len(names):
        raise ProfileError("each strategy can be in the mix once")
    total = sum((e.weight for e in mix), ZERO)
    if total > ONE:
        raise ProfileError(f"the shares add up to {total:.2%}; at most 100% (the rest is held in T-bills)")
    strategies = [candidate(e.strategy).factory(dict(e.params)) for e in mix]
    try:
        common_execution(strategies)
    except ValueError as e:
        closing = [e.strategy for e, s in zip(mix, strategies, strict=True) if s.execution == CLOSE]
        raise ProfileError(f"{', '.join(closing)} trades in the closing auction and cannot be mixed with "
                           "strategies that trade at the open") from e
    risk = _risk(data.get("risk", {}))
    note = data.get("note", "")
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise ProfileError(f"the note must be text of at most {MAX_NOTE} characters")
    return Profile(mix=mix, risk=risk, note=note.strip())


def effective_settings(settings: Settings, profile: Profile) -> Settings:
    """
    The environment's settings with the profile's risk overrides.

    Raises:
        ProfileError: If the result is inconsistent (the shrinking starts at or past the kill switch).
    """
    merged = settings.model_copy(update=dict(profile.risk))
    if merged.sizing_derisk_start and merged.sizing_derisk_start >= merged.risk_max_drawdown:
        raise ProfileError("exposure must start shrinking before the kill switch: lower the drop where it "
                           "starts, or raise the kill switch")
    return merged


# -- building ---------------------------------------------------------------------------


def symbols(profile: Profile) -> tuple[tuple[str, AssetClass], ...]:
    """Every symbol the mix trades, then the cash sleeve, each once."""
    seen: dict[str, AssetClass] = {}
    for entry in profile.mix:
        for symbol, asset_class in CANDIDATES[entry.strategy].symbols:
            seen.setdefault(symbol, asset_class)
    seen.setdefault(CASH_SLEEVE, AssetClass.EQUITY)
    return tuple(seen.items())


def allocations(profile: Profile, instruments: Mapping[str, Instrument]) -> list[Allocation]:
    """The engine's allocations: each strategy in a Blend named after it, at its share."""
    return [Allocation(Blend(e.strategy, [(candidate(e.strategy, instruments).factory(dict(e.params)), 1.0)]),
                       e.weight) for e in profile.mix]


def warm_up_days(profile: Profile) -> int:
    """Calendar days of history the engine loads before trading, enough for the slowest strategy."""
    return max([MIN_WARM_UP_DAYS, *(CANDIDATES[e.strategy].warm_up_days for e in profile.mix)])


# -- storage ------------------------------------------------------------------------------


def _from_record(record: Any) -> Profile:
    mix = tuple(MixEntry(e["strategy"], e["params"], Decimal(e["weight"])) for e in record.mix)
    risk = {k: (Decimal(v) if isinstance(v, str) else v) for k, v in record.risk.items()}
    return Profile(mix=mix, risk=risk, id=record.id, note=record.note or "", created_by=record.created_by,
                   created_at=record.created_at, activated_at=record.activated_at)


def latest(app: Flask) -> Profile:
    """The newest saved profile (what the engine trades from its next start), or the default."""
    from src.models.trading import TradingProfileRecord
    from src.models.user import db

    with app.app_context():
        record = db.session.scalars(select(TradingProfileRecord).order_by(TradingProfileRecord.id.desc())).first()
        return _from_record(record) if record is not None else DEFAULT_PROFILE


def active(app: Flask) -> Profile:
    """The profile the engine started with last (the newest activated), or the default."""
    from src.models.trading import TradingProfileRecord
    from src.models.user import db

    with app.app_context():
        record = db.session.scalars(
            select(TradingProfileRecord).where(TradingProfileRecord.activated_at.is_not(None))
            .order_by(TradingProfileRecord.activated_at.desc(), TradingProfileRecord.id.desc())).first()
        return _from_record(record) if record is not None else DEFAULT_PROFILE


def save(app: Flask, profile: Profile, created_by: str, now: datetime) -> Profile:
    from src.models.trading import TradingProfileRecord
    from src.models.user import db

    with app.app_context():
        record = TradingProfileRecord(mix=[e.to_dict() for e in profile.mix],
                                      risk={k: _json(v) for k, v in profile.risk.items()},
                                      note=profile.note or None, created_by=created_by, created_at=now)
        db.session.add(record)
        db.session.commit()
        return _from_record(record)


def mark_activated(app: Flask, profile: Profile, now: datetime) -> Profile:
    """Stamp a saved profile the first time the engine starts with it; the default has no row."""
    from src.models.trading import TradingProfileRecord
    from src.models.user import db

    if profile.id is None:
        return profile
    with app.app_context():
        record = db.session.get(TradingProfileRecord, profile.id)
        if record is None:
            return profile
        if record.activated_at is None:
            record.activated_at = now
            db.session.commit()
        return _from_record(record)
