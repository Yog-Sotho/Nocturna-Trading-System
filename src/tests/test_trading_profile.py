"""
Tests for the trading profile: the strategy catalog, the rules a mix and its
risk limits must follow, how the engine is built from a profile, its storage,
and the runtime taking a newer profile at its next start.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import SecretStr

from src.tests.trading_fixtures import IEF, SPY, daily_bars, ny
from src.trading import profile as profiles
from src.trading.catalog import STRATEGIES, catalog
from src.trading.domain import AssetClass, Instrument
from src.trading.engine import Engine
from src.trading.ledger import Ledger
from src.trading.profile import DEFAULT_PROFILE, ProfileError, parse
from src.trading.validation.candidates import CANDIDATES

D = Decimal
MIX_6040 = {"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "1"}


def _profile(*entries, risk=None, **extra):
    return {"mix": list(entries) or [MIX_6040], **({"risk": risk} if risk is not None else {}), **extra}


class TestCatalog:
    def test_every_registered_strategy_is_described(self):
        assert list(STRATEGIES) == list(CANDIDATES)
        assert [s["name"] for s in catalog()] == list(CANDIDATES)

    @pytest.mark.parametrize("entry", catalog(), ids=lambda e: e["name"])
    def test_defaults_and_choices_are_the_validation_grid(self, entry):
        grid = profiles.candidate(entry["name"]).grid
        assert [p["name"] for p in entry["parameters"]] == list(grid)
        for parameter in entry["parameters"]:
            assert [c["value"] for c in parameter["choices"]] == list(grid[parameter["name"]])
            assert entry["defaults"][parameter["name"]] in grid[parameter["name"]]
        assert parse(_profile({"strategy": entry["name"], "params": entry["defaults"], "weight": 1}))

    def test_verdicts_and_execution(self):
        by_name = {e["name"]: e for e in catalog()}
        assert by_name["benchmark"]["verdict"] == "benchmark"
        assert {e["verdict"] for n, e in by_name.items() if n != "benchmark"} == {"rejected"}
        assert by_name["month-end"]["execution"] == "close" and by_name["trend-filter"]["execution"] == "open"
        assert by_name["crypto-trend"]["asset_classes"] == ["crypto"]
        assert by_name["low-vol"]["parameters"][0]["choices"][0]["label"].startswith("USMV")


class TestParse:
    def test_the_default_mix(self):
        profile = parse(_profile())
        assert profile.mix == DEFAULT_PROFILE.mix and profile.risk == {} and profile.note == ""

    def test_a_mix_with_shares_below_one_hundred_percent(self):
        profile = parse(_profile({"strategy": "benchmark", "params": {"equity": 0.6}, "weight": 0.7},
                                 {"strategy": "trend-filter", "params": {"months": 10}, "weight": "0.25"},
                                 note="  a mix  "))
        assert [(e.strategy, e.weight) for e in profile.mix] == [("benchmark", D("0.7")), ("trend-filter", D("0.25"))]
        assert profile.note == "a mix"

    @pytest.mark.parametrize("data, message", [
        (None, "must be an object"),
        ({"mix": []}, "at least one strategy"),
        ({"mix": ["x"]}, "must be an object"),
        (_profile({"strategy": "grid", "params": {}, "weight": 1}), "unknown strategy 'grid'"),
        (_profile({"strategy": "benchmark", "params": {}, "weight": 1}), "give exactly these parameters: equity"),
        (_profile({"strategy": "benchmark", "params": {"equity": 0.65}, "weight": 1}), "must be one of 0.4, 0.5"),
        (_profile({"strategy": "tsmom", "params": {"rule": "sign", "long_only": 1}, "weight": 1}),
         "long_only must be one of False, True"),
        (_profile({**MIX_6040, "weight": 0}), "weight must be above 0%"),
        (_profile({**MIX_6040, "weight": "1.5"}), "weight must be above 0%"),
        (_profile({**MIX_6040, "weight": "lots"}), "weight must be a number"),
        (_profile({**MIX_6040, "weight": True}), "weight must be a number"),
        (_profile({**MIX_6040, "weight": "NaN"}), "weight must be a number"),
        (_profile(MIX_6040, {"strategy": "trend-filter", "params": {"months": 10}, "weight": "0.2"}),
         "the shares add up to 120.00%"),
        (_profile(MIX_6040, {**MIX_6040, "weight": "0"}), "weight must be above 0%"),
        (_profile({**MIX_6040, "weight": "0.5"}, {**MIX_6040, "weight": "0.5"}), "once"),
        (_profile({**MIX_6040, "weight": "0.5"}, {"strategy": "month-end", "params": {"rule": "turn"},
                                                  "weight": "0.5"}),
         "month-end trades in the closing auction"),
        (_profile(note=5), "the note must be text"),
        (_profile(note="x" * 201), "at most 200 characters"),
    ])
    def test_what_the_engine_could_not_trade_is_refused(self, data, message):
        with pytest.raises(ProfileError, match=message):
            parse(data)

    def test_a_json_whole_number_matches_a_float_grid_value(self):
        profile = parse(_profile({"strategy": "rebalancing", "params": {"mode": "tilt", "scale": 1}, "weight": 1}))
        assert profile.mix[0].params == {"mode": "tilt", "scale": 1.0}
        assert isinstance(profile.mix[0].params["scale"], float)

    def test_risk_overrides_within_their_bounds(self):
        profile = parse(_profile(risk={"risk_max_drawdown": "0.08", "risk_flatten_on_kill": False,
                                       "sizing_target_volatility": 0, "risk_max_orders_per_day": 20,
                                       "risk_max_order_notional": None, "sizing_derisk_start": "0"}))
        assert profile.risk == {"risk_max_drawdown": D("0.08"), "risk_flatten_on_kill": False,
                                "sizing_target_volatility": D(0), "risk_max_orders_per_day": 20,
                                "risk_max_order_notional": None, "sizing_derisk_start": D(0)}

    @pytest.mark.parametrize("risk, message", [
        ("x", "risk must be an object"),
        ({"risk_max_leverage": 2}, "unknown risk setting risk_max_leverage"),
        ({"risk_max_drawdown": "0.5"}, "between 0.02 and 0.25"),
        ({"risk_max_drawdown": "0.01"}, "between 0.02 and 0.25"),
        ({"sizing_target_volatility": "0.01"}, r"between 0.02 and 0.20, or 0 to turn it off"),
        ({"risk_max_orders_per_day": "2.5"}, "whole number"),
        ({"risk_flatten_on_kill": "yes"}, "true or false"),
        ({"risk_max_drawdown": None}, "a value is required"),
        ({"risk_max_order_notional": 50}, "between 100 and"),
    ])
    def test_risk_outside_its_bounds_is_refused(self, risk, message):
        with pytest.raises(ProfileError, match=message):
            parse(_profile(risk=risk))


class TestEffectiveSettings:
    def test_overrides_replace_the_environment(self, app):
        settings = app.extensions["settings"]
        merged = profiles.effective_settings(settings, parse(_profile(risk={"risk_max_drawdown": "0.08"})))
        assert merged.risk_max_drawdown == D("0.08") and merged.sizing_target_volatility == D("0.08")
        assert settings.risk_max_drawdown == D("0.10")  # the environment's own settings are untouched

    def test_shrinking_must_start_before_the_kill_switch(self, app):
        with pytest.raises(ProfileError, match="start shrinking before the kill switch"):
            profiles.effective_settings(app.extensions["settings"],
                                        parse(_profile(risk={"risk_max_drawdown": "0.04"})))  # derisk from 5%
        assert profiles.effective_settings(
            app.extensions["settings"], parse(_profile(risk={"risk_max_drawdown": "0.04", "sizing_derisk_start": 0})))


class TestBuilding:
    def test_symbols_and_the_cash_sleeve_once(self):
        profile = parse(_profile({**MIX_6040, "weight": "0.5"},
                                 {"strategy": "trend-filter", "params": {"months": 8}, "weight": "0.5"}))
        assert profiles.symbols(profile) == (("SPY", AssetClass.EQUITY), ("IEF", AssetClass.EQUITY),
                                             ("BIL", AssetClass.EQUITY))

    def test_two_buy_and_hold_strategies_trade_side_by_side(self):
        """The 60/40 and low-vol are both BuyAndHold: each runs in a Blend named after it."""
        profile = parse(_profile({**MIX_6040, "weight": "0.5"},
                                 {"strategy": "low-vol", "params": {"equity": "SPLV"}, "weight": "0.3"}))
        instruments = {s: Instrument(s, a) for s, a in profiles.symbols(profile)}
        allocations = profiles.allocations(profile, instruments)
        assert [(a.strategy.name, a.weight) for a in allocations] == [("benchmark", D("0.5")), ("low-vol", D("0.3"))]
        from src.trading.broker import SimulatedBroker
        from src.trading.clock import SimulatedClock
        clock = SimulatedClock(ny(2026, 3, 2))
        engine = Engine(clock=clock, broker=SimulatedBroker(clock, instruments, D("100000")),
                        ledger=Ledger(D("100000")), allocations=allocations)
        assert set(engine.instruments) == {"SPY", "IEF", "SPLV"}  # no cash sleeve given here

    def test_the_engine_warms_up_long_enough_for_the_slowest_strategy(self):
        assert profiles.warm_up_days(DEFAULT_PROFILE) == 550
        crypto = parse(_profile({"strategy": "crypto-trend", "params": {"target_volatility": 0.25}, "weight": "0.1"}))
        assert profiles.warm_up_days(crypto) == 760


class TestStorage:
    def test_nothing_saved_means_the_default(self, app):
        assert profiles.latest(app) == DEFAULT_PROFILE and profiles.active(app) == DEFAULT_PROFILE

    def test_versions_and_activation(self, app):
        now = datetime(2026, 10, 24, 9, tzinfo=UTC)
        first = profiles.save(app, parse(_profile(risk={"risk_max_drawdown": "0.08"})), "ops", now)
        second = profiles.save(app, parse(_profile({**MIX_6040, "weight": "0.5"}, note="half")), "ops",
                               now + timedelta(hours=1))
        assert first.id is not None and second.id == first.id + 1
        assert profiles.latest(app) == second and profiles.active(app) == DEFAULT_PROFILE
        assert first.risk == {"risk_max_drawdown": D("0.08")} and second.note == "half"
        activated = profiles.mark_activated(app, first, now + timedelta(hours=2))
        assert activated.activated_at == now + timedelta(hours=2) and profiles.active(app) == activated
        # A restart with the same profile keeps the moment it was first traded
        again = profiles.mark_activated(app, activated, now + timedelta(days=1))
        assert again.activated_at == now + timedelta(hours=2)
        assert profiles.mark_activated(app, DEFAULT_PROFILE, now) is DEFAULT_PROFILE
        assert second.to_dict()["mix"] == [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "0.5000"}]

    def test_a_vanished_row_is_not_stamped(self, app):
        ghost = profiles.Profile(mix=DEFAULT_PROFILE.mix, id=999)
        assert profiles.mark_activated(app, ghost, datetime(2026, 10, 24, tzinfo=UTC)) is ghost


class TestHeldOver:
    def test_symbols_another_strategy_trades_are_sold_bil_and_hand_trades_are_not(self):
        from src.trading.runtime import held_over
        held = ["SPY", "IEF", "BIL", "BTCUSD", "ETH/USD", "TSLA", "GLD"]
        assert held_over(held, DEFAULT_PROFILE) == [("BTC/USD", AssetClass.CRYPTO), ("ETH/USD", AssetClass.CRYPTO),
                                                    ("GLD", AssetClass.EQUITY)]
        assert held_over(["SPY", "IEF"], DEFAULT_PROFILE) == []


class TestRuntimeTakesTheNewestProfile:
    @pytest.fixture
    def runtime(self, app):
        from src.trading.broker import SimulatedBroker
        from src.trading.clock import SimulatedClock
        from src.trading.data import InMemoryBarSource
        from src.trading.runtime import TradingRuntime
        from src.trading.service import EngineService

        built = []

        def factory(flask_app, settings, profile):
            clock = SimulatedClock(ny(2026, 3, 2, 12))
            instruments = {"SPY": SPY, "IEF": IEF}
            history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
                       "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
            built.append(profile)
            return EngineService(clock=clock, source=InMemoryBarSource(history, clock),
                                 broker=SimulatedBroker(clock, instruments, D("100000")),
                                 allocations=profiles.allocations(profile, instruments))

        settings = app.extensions["settings"].model_copy(
            update={"alpaca_api_key": SecretStr("key"), "alpaca_secret_key": SecretStr("secret")})
        runtime = TradingRuntime(app, settings, factory=factory)
        runtime.built = built
        yield runtime
        runtime.stop()

    def test_rebuilt_only_for_a_newer_profile_and_stamped_when_traded(self, app, runtime):
        runtime.start()
        assert runtime.built == [DEFAULT_PROFILE] and runtime.views()["profile"]["id"] is None
        runtime.stop()
        runtime.start()  # same profile: the service is reused
        assert len(runtime.built) == 1
        saved = profiles.save(app, parse(_profile({**MIX_6040, "weight": "0.5"})), "ops", datetime.now(UTC))
        runtime.restart()
        assert [p.id for p in runtime.built] == [None, saved.id]
        assert runtime.views()["profile"]["id"] == saved.id
        assert runtime.views()["profile"]["activated_at"] is not None
        assert profiles.active(app).id == saved.id
        runtime.stop()
        runtime.start()  # same saved profile again: reused, and the first activation is kept
        assert len(runtime.built) == 2 and profiles.active(app).activated_at == profiles.latest(app).activated_at

    def test_no_profile_in_the_views_before_a_service_exists(self, runtime):
        assert runtime.views()["profile"] is None
