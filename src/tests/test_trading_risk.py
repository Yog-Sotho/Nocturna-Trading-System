"""
Tests for the risk engine: the pre-trade gate, loss limits and the kill
switch, position sizing (caps, volatility target, drawdown ladder), and how
the engine, backtester, service and recorder use them.
"""

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.tests.trading_fixtures import BTC, IEF, SPY, bar, daily_bars, ny
from src.trading.backtest import BacktestConfig, run_backtest
from src.trading.broker import CostModel, SimulatedBroker
from src.trading.calendar import TradingCalendar
from src.trading.clock import SimulatedClock
from src.trading.domain import NEW_YORK, AssetClass, OrderRequest, OrderType, Side
from src.trading.engine import Allocation, Engine, EngineEvent
from src.trading.ledger import Ledger
from src.trading.risk import OrderContext, RiskGate, RiskLimits
from src.trading.sizing import PortfolioSizer, SizingPolicy, cap_weights, ewma_covariance, stop_risk_weight
from src.trading.strategies.benchmark import BuyAndHold
from src.trading.strategy import Strategy, StrategyContext

D = Decimal
NO_COSTS = {AssetClass.EQUITY: CostModel(), AssetClass.CRYPTO: CostModel()}
INSTRUMENTS = {"SPY": SPY, "IEF": IEF, "BTC/USD": BTC}
NOW = ny(2026, 3, 2, 12)
_FRESH = object()


def _req(cid="r", symbol="SPY", side=Side.BUY, qty="10", **kw):
    return OrderRequest(cid, symbol, side, D(qty), **kw)


def _ctx(price="100", exposure=None, equity="10000", now=NOW, last_bar_end=_FRESH, jump=None):
    return OrderContext(
        now=now, price=D(price) if price else None,
        exposure={k: D(v) for k, v in (exposure or {}).items()},
        prices={"SPY": D("100"), "IEF": D("100"), "BTC/USD": D("100")},
        asset_classes={s: i.asset_class for s, i in INSTRUMENTS.items()}, equity=D(equity),
        last_bar_end=now - timedelta(hours=12) if last_bar_end is _FRESH else last_bar_end, price_jump=jump)


def _send(gate, request, ctx):
    """Check an order and, if allowed, count it, as the engine does."""
    decision = gate.check(request, ctx)
    if decision.allowed:
        gate.record(request, ctx.price, ctx.now)
    return decision


# -- the pre-trade gate ------------------------------------------------------------------

class TestExposureLimits:
    def test_position_and_gross_caps(self):
        gate = RiskGate(RiskLimits(max_position_weight=D("0.5"), max_gross_exposure=D("0.8")))
        assert gate.check(_req(qty="50"), _ctx()).allowed
        assert "position would exceed" in gate.check(_req(qty="51"), _ctx()).reason
        assert "gross exposure" in gate.check(_req(qty="40"), _ctx(exposure={"IEF": "50"})).reason

    def test_asset_class_cap(self):
        gate = RiskGate()  # crypto at most 15% of equity
        assert gate.check(_req(symbol="BTC/USD", qty="15"), _ctx()).allowed
        assert "crypto exposure" in gate.check(_req(symbol="BTC/USD", qty="16"), _ctx()).reason

    def test_single_order_limits(self):
        assert "notional" in RiskGate(RiskLimits(max_order_notional=D("500"))).check(_req(qty="6"), _ctx()).reason
        fat_finger = RiskGate(RiskLimits(max_order_weight=D("0.2")))
        assert fat_finger.check(_req(qty="20"), _ctx()).allowed
        assert "single-order" in fat_finger.check(_req(qty="21"), _ctx()).reason

    def test_halt_allows_only_reducing_orders(self):
        gate = RiskGate()
        gate.halt("drill")
        assert "halted" in gate.check(_req(qty="1"), _ctx()).reason
        assert gate.check(_req(side=Side.SELL, qty="5"), _ctx(exposure={"SPY": "10"})).allowed
        assert gate.check(_req(side=Side.SELL, qty="10"), _ctx(exposure={"SPY": "10"})).allowed
        # Selling through zero opens a short: not reducing
        assert not gate.check(_req(side=Side.SELL, qty="15"), _ctx(exposure={"SPY": "10"})).allowed
        assert gate.check(_req(side=Side.BUY, qty="4"), _ctx(exposure={"SPY": "-10"})).allowed


class TestOrderSanity:
    def test_missing_price_or_equity(self):
        assert "reference price" in RiskGate().check(_req(), _ctx(price=None)).reason
        assert "equity" in RiskGate().check(_req(), _ctx(equity="0")).reason

    def test_limit_price_collar(self):
        gate = RiskGate()  # 5% around the reference price of 100

        def limit(side, price, cid="r"):
            return _req(cid, side=side, qty="1", order_type=OrderType.LIMIT, limit_price=D(price))

        assert gate.check(limit(Side.BUY, "105"), _ctx()).allowed
        assert "above the reference price" in gate.check(limit(Side.BUY, "105.01"), _ctx()).reason
        held = _ctx(exposure={"SPY": "10"})
        assert gate.check(limit(Side.SELL, "95"), held).allowed
        # The collar applies even to orders that reduce risk: a fat-fingered price is still wrong
        assert "below the reference price" in gate.check(limit(Side.SELL, "94.99"), held).reason

    def test_duplicates_within_the_window(self):
        gate = RiskGate()
        assert _send(gate, _req("a"), _ctx()).allowed
        assert "duplicate" in _send(gate, _req("b"), _ctx(now=NOW + timedelta(minutes=1))).reason
        assert _send(gate, _req("c", qty="11"), _ctx(now=NOW + timedelta(minutes=1))).allowed
        assert _send(gate, _req("d", side=Side.SELL), _ctx(now=NOW + timedelta(minutes=2),
                                                           exposure={"SPY": "21"})).allowed
        assert _send(gate, _req("e"), _ctx(now=NOW + timedelta(minutes=6))).allowed

    def test_stale_or_jumping_prices_block_new_risk_only(self):
        gate = RiskGate()
        old = NOW - timedelta(days=6)
        assert "stale" in gate.check(_req(), _ctx(last_bar_end=old)).reason
        assert "stale" in gate.check(_req(), _ctx(last_bar_end=None)).reason
        assert gate.check(_req(side=Side.SELL, qty="5"), _ctx(exposure={"SPY": "10"}, last_bar_end=old)).allowed
        assert "bad data" in gate.check(_req(), _ctx(jump=D("0.3"))).reason
        assert gate.check(_req(symbol="BTC/USD", qty="1"), _ctx(jump=D("0.3"))).allowed  # crypto: 50%


class TestCircuitBreakers:
    def test_daily_order_count_resets_the_next_new_york_day(self):
        gate = RiskGate(RiskLimits(max_orders_per_day=2))
        assert _send(gate, _req("a", qty="1"), _ctx()).allowed
        assert _send(gate, _req("b", qty="2"), _ctx()).allowed
        assert "daily order limit" in _send(gate, _req("c", qty="3"), _ctx()).reason
        # Reducing orders are never blocked by circuit breakers
        assert _send(gate, _req("d", side=Side.SELL, qty="1"), _ctx(exposure={"SPY": "3"})).allowed
        # 23:59 is still the same New York day; 00:01 is the next one
        assert not _send(gate, _req("e", qty="3"), _ctx(now=ny(2026, 3, 2, 23, 59))).allowed
        assert _send(gate, _req("f", qty="3"), _ctx(now=ny(2026, 3, 3, 0, 1))).allowed
        assert gate.state.orders_today == 1

    def test_daily_turnover(self):
        gate = RiskGate(RiskLimits(max_daily_turnover=D("1")))  # one equity (10,000) per day
        assert _send(gate, _req("a", qty="60"), _ctx()).allowed
        assert "turnover" in _send(gate, _req("b", qty="41"), _ctx()).reason
        assert _send(gate, _req("c", qty="40"), _ctx()).allowed
        assert gate.state.turnover_today == D("10000")


# -- loss limits and the kill switch -------------------------------------------------------

def _days(n):
    return [ny(2026, 3, 2, 0, 15) + timedelta(days=i) for i in range(n)]


class TestLossLimits:
    def test_drawdown_kill_switch_trips_once(self):
        gate = RiskGate()
        days = _days(7)
        assert gate.on_equity(D("100000"), days[0]) is None
        assert gate.on_equity(D("110000"), days[1]) is None
        # Losses of under 3% a day: no cool-down, drawdown grows
        for day, equity in zip(days[2:5], ("107000", "104000", "101000"), strict=True):
            assert gate.on_equity(D(equity), day) is None
        assert gate.state.high_water_mark == D("110000")
        assert gate.drawdown == 1 - D("101000") / D("110000")
        assert gate.on_equity(D("99000"), days[5]) == "kill"  # exactly 10% below the high
        assert gate.halted and gate.state.halt_kind == "drawdown" and gate.state.halted_at == days[5]
        assert gate.on_equity(D("98000"), days[6]) is None  # already halted

    def test_resume_after_a_drawdown_kill_restarts_the_high_water_mark(self):
        gate = RiskGate()
        gate.on_equity(D("100000"), NOW)
        assert gate.on_equity(D("90000"), NOW + timedelta(days=1)) == "kill"
        gate.resume(D("90000"))
        assert not gate.halted and gate.state.high_water_mark == D("90000") and gate.drawdown == 0

    def test_resume_after_a_manual_halt_keeps_the_high_water_mark(self):
        gate = RiskGate()
        gate.on_equity(D("100000"), NOW)
        gate.on_equity(D("98000"), NOW + timedelta(days=1))
        gate.halt("drill")
        gate.resume(D("98000"))
        assert not gate.halted and gate.state.high_water_mark == D("100000")

    def test_daily_loss_starts_a_cool_down_until_the_next_day(self):
        gate = RiskGate()  # 3% a day
        assert gate.on_equity(D("100000"), ny(2026, 3, 2, 0, 15)) is None
        assert gate.on_equity(D("98500"), ny(2026, 3, 2, 20, 1)) is None
        assert gate.on_equity(D("97000"), ny(2026, 3, 2, 21)) == "cooldown"  # -3% on the day
        late = ny(2026, 3, 2, 22)
        assert gate.on_equity(D("97500"), late) == "cooldown"  # a bounce does not end it
        assert "cool-down" in gate.check(_req(), _ctx(now=late)).reason
        assert gate.check(_req(side=Side.SELL, qty="5"), _ctx(now=late, exposure={"SPY": "10"})).allowed
        # The next day starts from the previous close
        assert gate.on_equity(D("97500"), ny(2026, 3, 3, 0, 15)) is None
        assert not gate.cooling_down and gate.state.day_start_equity == D("97500")

    def test_deposits_move_the_high_water_mark(self):
        gate = RiskGate()
        gate.on_equity(D("100000"), NOW)
        gate.on_equity(D("95000"), NOW + timedelta(days=1))
        gate.adjust_for_flow(D("50000"))
        assert gate.state.high_water_mark == D("150000") and gate.state.last_equity == D("145000")
        assert gate.drawdown == 1 - D("145000") / D("150000")

    def test_state_survives_a_json_round_trip(self):
        gate = RiskGate()
        gate.on_equity(D("100000"), NOW)
        _send(gate, _req("a"), _ctx())
        gate.halt("drill", now=NOW)
        restored = RiskGate()
        restored.restore(json.loads(json.dumps(gate.state_dict())))
        assert restored.state == gate.state

    def test_limits_are_validated(self):
        with pytest.raises(ValueError):
            RiskLimits(max_drawdown=D("1"))
        with pytest.raises(ValueError):
            RiskLimits(max_position_weight=D("0"))
        with pytest.raises(ValueError):
            RiskLimits(max_orders_per_day=0)


# -- sizing ----------------------------------------------------------------------------------

class TestCapWeights:
    def test_clips_positions_then_scales_classes_and_the_book(self):
        limits = RiskLimits(max_position_weight=D("0.5"), max_gross_exposure=D("0.9"),
                            max_asset_class_weight={AssetClass.CRYPTO: D("0.1")})
        capped = cap_weights({"SPY": D("0.7"), "IEF": D("0.3"), "BTC/USD": D("0.2")}, INSTRUMENTS, limits)
        assert capped == {"SPY": D("0.5"), "IEF": D("0.3"), "BTC/USD": D("0.1")}
        scaled = cap_weights({"SPY": D("0.5"), "IEF": D("0.5"), "BTC/USD": D("0.1")}, INSTRUMENTS, limits)
        assert sum(scaled.values()) <= D("0.9")
        assert scaled["SPY"] == scaled["IEF"] and scaled["SPY"] / scaled["BTC/USD"] == 5

    @given(st.dictionaries(st.sampled_from(sorted(INSTRUMENTS)),
                           st.decimals(min_value=-2, max_value=2, places=6), min_size=1))
    def test_caps_always_hold_and_never_flip_or_grow_a_position(self, targets):
        limits = RiskLimits()
        capped = cap_weights(targets, INSTRUMENTS, limits)
        assert all(abs(w) <= limits.max_position_weight for w in capped.values())
        assert abs(capped.get("BTC/USD", D(0))) <= limits.max_asset_class_weight[AssetClass.CRYPTO]
        assert sum(abs(w) for w in capped.values()) <= limits.max_gross_exposure
        for symbol, target in targets.items():
            assert abs(capped[symbol]) <= abs(target)
            assert capped[symbol] == 0 or (capped[symbol] > 0) == (target > 0)


class TestDrawdownLadder:
    @pytest.mark.parametrize("drawdown, scale", [
        ("0", "1"), ("0.05", "1"), ("0.075", "0.625"), ("0.09", "0.4"), ("0.10", "0.25"), ("0.2", "0.25")])
    def test_linear_from_derisk_start_to_the_floor_at_the_kill_limit(self, drawdown, scale):
        assert PortfolioSizer(SizingPolicy(), RiskLimits()).drawdown_scale(D(drawdown)) == D(scale)

    def test_textbook_ladder_without_a_floor(self):
        sizer = PortfolioSizer(SizingPolicy(derisk_floor=D("0")), RiskLimits())
        assert sizer.drawdown_scale(D("0.075")) == D("0.5") and sizer.drawdown_scale(D("0.10")) == 0

    def test_off(self):
        assert PortfolioSizer(SizingPolicy.unscaled(), RiskLimits()).drawdown_scale(D("0.09")) == 1


class TestStopRiskWeight:
    def test_a_stop_hit_loses_the_risk_budget(self):
        weight = stop_risk_weight(100.0, 95.0)  # 0.5% of equity at risk over a 5% stop
        assert weight == pytest.approx(0.1)
        equity = 100000
        units = weight * equity / 100.0
        assert units * (100.0 - 95.0) == pytest.approx(0.005 * equity)

    def test_a_stop_above_the_price_is_a_short(self):
        assert stop_risk_weight(100.0, 104.0, risk=0.01) == pytest.approx(-0.25)

    @pytest.mark.parametrize("price, stop, risk", [(0, 1, 0.005), (100, 100, 0.005), (100, 95, 0), (100, 95, 1)])
    def test_invalid_inputs(self, price, stop, risk):
        with pytest.raises(ValueError):
            stop_risk_weight(price, stop, risk)


def _alternating(instrument, amplitude, n, start=date(2025, 1, 2), phase=1):
    """Daily closes whose log returns alternate exactly between +a and -a."""
    days = TradingCalendar(instrument.asset_class).sessions(start, start + timedelta(days=2 * n))[:n]
    bars, price = [], 100.0
    for i, day in enumerate(days):
        zone = NEW_YORK if instrument.asset_class is AssetClass.EQUITY else UTC
        start_at = datetime(day.year, day.month, day.day, tzinfo=zone).astimezone(UTC)
        if i:
            price *= math.exp(amplitude if (i % 2) == (phase % 2) else -amplitude)
        bars.append(bar(instrument, start_at, price, price, price, price))
    return bars


class TestVolatilityTarget:
    def test_ewma_recovers_a_known_volatility(self):
        history = {"SPY": _alternating(SPY, 0.01, 300)}
        cov, periods = ewma_covariance(history, INSTRUMENTS, ["SPY"], 0.94, 60)
        assert cov[0, 0] == pytest.approx(0.01 ** 2, rel=1e-3)
        assert periods == pytest.approx(252, rel=0.02)

    def test_covariance_offsets_opposite_moves(self):
        history = {"SPY": _alternating(SPY, 0.01, 300), "IEF": _alternating(IEF, 0.01, 300, phase=0)}
        sizer = PortfolioSizer(SizingPolicy(), RiskLimits())
        assert sizer.predicted_volatility({"SPY": D("0.5"), "IEF": D("0.5")}, history, INSTRUMENTS) < 1e-3

    def test_crypto_and_equities_are_compared_on_common_dates(self):
        history = {"SPY": _alternating(SPY, 0.01, 300), "BTC/USD": _alternating(BTC, 0.03, 420)}
        _, crypto_periods = ewma_covariance(history, INSTRUMENTS, ["BTC/USD"], 0.94, 60)
        _, joint_periods = ewma_covariance(history, INSTRUMENTS, ["SPY", "BTC/USD"], 0.94, 60)
        assert crypto_periods == pytest.approx(365.25, rel=0.01)
        assert joint_periods == pytest.approx(252, rel=0.02)

    def test_volatile_books_are_scaled_down_to_the_target(self):
        history = {"SPY": _alternating(SPY, 0.02, 300)}
        result = PortfolioSizer(SizingPolicy(), RiskLimits()).size({"SPY": D("0.6")}, history, INSTRUMENTS, D(0))
        assert result.volatility == pytest.approx(0.6 * 0.02 * math.sqrt(252), rel=0.03)
        assert result.weights["SPY"] * D(repr(result.volatility)) / result.capped_weights["SPY"] == \
            pytest.approx(D("0.08"), rel=D("0.001"))

    def test_calm_books_are_never_levered_up(self):
        history = {"SPY": _alternating(SPY, 0.001, 300)}
        result = PortfolioSizer(SizingPolicy(), RiskLimits()).size({"SPY": D("0.6")}, history, INSTRUMENTS, D(0))
        assert result.volatility_scale == 1 and result.weights == {"SPY": D("0.6")}

    def test_short_history_means_no_scaling(self):
        history = {"SPY": _alternating(SPY, 0.02, 40)}
        result = PortfolioSizer(SizingPolicy(), RiskLimits()).size({"SPY": D("0.6")}, history, INSTRUMENTS, D(0))
        assert result.volatility is None and result.weights == {"SPY": D("0.6")}

    def test_scales_combine(self):
        history = {"SPY": _alternating(SPY, 0.02, 300)}
        sizer = PortfolioSizer(SizingPolicy(), RiskLimits())
        calm = sizer.size({"SPY": D("0.6")}, history, INSTRUMENTS, D(0))
        stressed = sizer.size({"SPY": D("0.6")}, history, INSTRUMENTS, D("0.075"))
        assert stressed.drawdown_scale == D("0.625")
        assert stressed.weights["SPY"] == pytest.approx(calm.weights["SPY"] * D("0.625"), abs=D("1e-6"))

    def test_policy_is_validated(self):
        with pytest.raises(ValueError):
            SizingPolicy(target_volatility=D("0"))
        with pytest.raises(ValueError):
            SizingPolicy(ewma_lambda=1.0)
        with pytest.raises(ValueError):
            SizingPolicy(derisk_floor=D("1.5"))


# -- the engine ------------------------------------------------------------------------------

class _Fixed(Strategy):
    name = "fixed"
    lookback = 1

    def __init__(self, weights, instruments=(SPY, IEF)):
        self.instruments = instruments
        self.weights = weights

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        return self.weights() if callable(self.weights) else self.weights


def _engine(strategy, cash="100000", **kw):
    clock = SimulatedClock(ny(2026, 3, 2))
    broker = SimulatedBroker(clock, {i.symbol: i for i in strategy.instruments}, D(cash), costs=NO_COSTS)
    events = []
    engine = Engine(clock=clock, broker=broker, ledger=Ledger(D(cash)),
                    allocations=[Allocation(strategy, D("1"))], listeners=[events.append], **kw)
    return clock, broker, engine, events


def _feed_day(clock, engine, day, spy=100.0, ief=50.0):
    bars = {"SPY": bar(SPY, ny(2026, 3, day), spy, spy, spy, spy),
            "IEF": bar(IEF, ny(2026, 3, day), ief, ief, ief, ief)}
    clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
    return engine.on_bars(bars)


def _orders(sent):
    return [(r.symbol, r.side, r.qty) for r in sent]


class TestEngineRiskControls:
    def test_drawdown_kill_switch_flattens_and_waits_for_the_operator(self):
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)),
                                                sizing=SizingPolicy.unscaled())
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)  # 500 shares at 100
        assert _feed_day(clock, engine, 4, spy=80.0) == []  # equity 90,000: 10% below the high
        assert engine.risk.halted and engine.risk.state.halt_kind == "drawdown"
        assert [e.kind for e in events].count("kill_switch") == 1
        [close] = [o for o in engine.orders.values() if o.status.is_open]
        assert close.request.side is Side.SELL and close.request.qty == D("500")
        _feed_day(clock, engine, 5, spy=80.0)
        assert engine.ledger.positions == {} and engine.ledger.cash == D("90000")
        assert _feed_day(clock, engine, 6, spy=80.0) == []  # stays halted
        engine.resume()
        assert engine.risk.state.high_water_mark == D("90000")
        assert _orders(_feed_day(clock, engine, 9, spy=80.0)) == [("SPY", Side.BUY, D("562"))]

    def test_a_failed_kill_switch_liquidation_is_retried(self):
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)),
                                                sizing=SizingPolicy.unscaled())
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        close_all = broker.close_all
        broker.close_all = lambda: (_ for _ in ()).throw(ConnectionError("broker down"))
        assert _feed_day(clock, engine, 4, spy=80.0) == []
        assert engine.risk.halted and engine.ledger.position("SPY").qty == D("500")
        assert any(e.kind == "broker_error" and "broker down" in e.detail for e in events)
        broker.close_all = close_all
        _feed_day(clock, engine, 5, spy=80.0)  # retried: the close order is working
        assert [o.request.qty for o in engine.orders.values() if o.status.is_open] == [D("500")]
        _feed_day(clock, engine, 6, spy=80.0)
        assert engine.ledger.positions == {}
        _feed_day(clock, engine, 9, spy=80.0)
        assert not [o for o in engine.orders.values() if o.status.is_open]  # nothing left to retry

    def test_drawdown_ladder_shrinks_the_book(self):
        clock, _, engine, _ = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)),
                                      sizing=SizingPolicy(target_volatility=None))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        # Equity 92,500 is 7.5% below the high, halfway down the ladder: 62.5% of the 50% target
        assert _orders(_feed_day(clock, engine, 4, spy=85.0)) == [("SPY", Side.SELL, D("160"))]
        assert engine.snapshot().risk.drawdown_scale == D("0.625")

    def test_cool_down_blocks_new_risk_for_the_rest_of_the_day(self):
        clock, _, engine, events = _engine(_Fixed({"SPY": 0.4, "IEF": 0.4}), sizing=SizingPolicy.unscaled())
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)  # 400 SPY at 100, 800 IEF at 50
        # SPY -10% is a 4% loss: rebuying SPY waits, trimming IEF does not
        assert _orders(_feed_day(clock, engine, 4, spy=90.0)) == [("IEF", Side.SELL, D("32"))]
        assert engine.risk.cooling_down and engine.snapshot().risk.cooling_down
        assert _orders(_feed_day(clock, engine, 5, spy=90.0)) == [("SPY", Side.BUY, D("26"))]

    def test_halt_with_flatten_uses_the_brokers_liquidation(self):
        clock, broker, engine, _ = _engine(_Fixed({"SPY": 0.3, "IEF": 0.3}))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        engine.submit_order(_req("op-1", qty="5"))  # working when the halt comes
        closed = engine.halt("drill", flatten=True)
        assert len(closed) == 2 and all(c in engine.orders for c in closed)
        assert engine.orders["op-1"].status.value == "canceled"
        _feed_day(clock, engine, 4)
        assert engine.ledger.positions == {} and broker.positions() == {}

    def test_risk_state_is_published_for_persistence(self):
        clock, _, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        _feed_day(clock, engine, 2)
        states = [e.data for e in events if e.kind == "risk_state"]
        assert states and D(states[-1]["high_water_mark"]) == D("100000") and states[-1]["orders_today"] == 1
        restored = RiskGate()
        restored.restore(states[-1])
        assert restored.state == engine.risk.state

    def test_snapshot_reports_the_risk_view(self):
        clock, _, engine, _ = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        _feed_day(clock, engine, 2)
        risk = engine.snapshot().risk
        assert risk.drawdown == 0 and risk.high_water_mark == D("100000") and risk.orders_today == 1
        assert risk.predicted_volatility is None and risk.volatility_scale == 1  # not enough history yet


class TestSimulatedLiquidation:
    def test_dust_below_the_smallest_order_is_left(self):
        clock = SimulatedClock(datetime(2026, 3, 2, tzinfo=UTC))
        broker = SimulatedBroker(clock, {"BTC/USD": BTC, "SPY": SPY}, D("1000"))
        broker._book.adjust_position("BTC/USD", D("0.00004"), D("60000"))
        broker._book.adjust_position("SPY", D("2"), D("100"))
        [close] = broker.close_all()
        assert broker.order(close).request.symbol == "SPY"


class TestReconciliationFlows:
    def test_a_deposit_raises_the_high_water_mark_but_a_shortfall_counts_as_a_loss(self):
        from src.trading.reconcile import reconcile
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        assert engine.risk.state.high_water_mark == D("100000")
        broker._book.adjust_cash(D("10000"))
        reconcile(engine, broker)
        assert engine.risk.state.high_water_mark == D("110000") and engine.risk.drawdown == 0
        assert "risk_state" in [e.kind for e in events[-2:]]  # persisted
        broker._book.adjust_cash(D("-5000"))
        reconcile(engine, broker)
        assert engine.risk.state.high_water_mark == D("110000")
        _feed_day(clock, engine, 4)
        assert engine.risk.drawdown == 1 - D("105000") / D("110000")


# -- backtests ---------------------------------------------------------------------------------

def _crash():
    return {"SPY": daily_bars(SPY, date(2024, 1, 2), date(2026, 6, 30), 500.0, seed=11, drift=-0.003, vol=0.012)}


CRASH = dict(start=datetime(2025, 1, 2, tzinfo=UTC), end=datetime(2026, 6, 1, tzinfo=UTC), costs=NO_COSTS)


class TestBacktestRiskControls:
    def _run(self, sizing):
        result = run_backtest([Allocation(BuyAndHold({SPY: 0.6}), D("1"))], _crash(),
                              BacktestConfig(**CRASH, sizing=sizing))
        kills = [e.timestamp for e in result.events if e.kind == "kill_switch"]
        return result, kills

    def test_the_kill_switch_ends_a_slide(self):
        result, kills = self._run(SizingPolicy.unscaled())
        assert len(kills) == 1 and result.final.halted and not result.final.positions
        # Daily bars: the kill trips at a close and the book is sold at the next open
        assert result.metrics()["max_drawdown"] > -0.11

    def test_a_simulated_operator_resumes_after_a_kill(self):
        result = run_backtest([Allocation(BuyAndHold({SPY: 0.6}), D("1"))], _crash(),
                              BacktestConfig(**CRASH, sizing=SizingPolicy.unscaled(),
                                             resume_after_kill=timedelta(days=30)))
        kills = [e.timestamp for e in result.events if e.kind == "kill_switch"]
        resumes = [e.timestamp for e in result.events if e.kind == "resume"]
        # The relentless crash trips the switch again after each resume
        assert len(kills) >= 2 and len(resumes) >= 1
        assert all(r - k >= timedelta(days=30) for k, r in zip(kills, resumes, strict=False))
        assert result.metrics()["kill_switch_trips"] == len(kills)

    def test_the_ladder_slows_the_slide_and_never_locks_the_book_in_cash(self):
        plain, plain_kills = self._run(SizingPolicy.unscaled())
        laddered, kills = self._run(SizingPolicy())
        # Exposure shrinks with the drawdown, so the limit is reached later and overshot
        # less; continuing losses still trip the kill switch and alert the operator
        assert len(kills) == 1 and kills[0] > plain_kills[0] and laddered.final.halted
        assert plain.metrics()["max_drawdown"] < laddered.metrics()["max_drawdown"] > -0.105


# -- persistence ---------------------------------------------------------------------------------

def _service(**kw):
    from src.trading.data import InMemoryBarSource
    from src.trading.service import EngineService
    history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
               "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
    clock = SimulatedClock(ny(2026, 3, 2, 12))
    broker = SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, D("100000"))
    service = EngineService(clock=clock, source=InMemoryBarSource(history, clock), broker=broker,
                            allocations=[Allocation(BuyAndHold({SPY: 0.6, IEF: 0.4}), D("1"))], **kw)
    return clock, broker, service


class TestRiskStatePersistence:
    def test_a_halt_survives_a_restart(self):
        saved = {"halted": True, "halt_reason": "drawdown 10.00% reached the 10% limit", "halt_kind": "drawdown",
                 "high_water_mark": "120000", "last_equity": "100000"}
        _, broker, service = _service(risk_state=lambda: saved)
        service.prepare()
        snapshot = service.status().snapshot
        assert snapshot.halted and snapshot.risk.halt_kind == "drawdown"
        assert snapshot.open_orders == () and broker.open_orders() == []

    def test_recorder_saves_and_loads_the_state(self, app, db):
        from src.models.trading import RiskStateRecord
        from src.trading.persistence import EngineRecorder
        recorder = EngineRecorder(app)
        assert recorder.load_risk_state() is None
        gate = RiskGate()
        gate.on_equity(D("100000"), NOW)
        recorder(EngineEvent("risk_state", NOW, data=gate.state_dict()))
        gate.halt("drill", now=NOW)
        recorder(EngineEvent("risk_state", NOW, data=gate.state_dict()))
        assert recorder.load_risk_state() == gate.state_dict()
        with app.app_context():
            assert RiskStateRecord.query.count() == 1

    def test_kill_switch_is_audited(self, app, db):
        from src.models.user import AuditLog
        from src.trading.persistence import EngineRecorder
        EngineRecorder(app)(EngineEvent("kill_switch", NOW, detail="drawdown 10.00% reached the 10% limit"))
        with app.app_context():
            assert AuditLog.query.filter_by(event_type="ENGINE_KILL_SWITCH").count() == 1

    def test_service_restores_what_the_recorder_saved(self, app, db):
        from src.trading.persistence import EngineRecorder
        recorder = EngineRecorder(app)
        _, _, first = _service(risk_state=recorder.load_risk_state, listeners=[recorder])
        first.prepare()
        first.engine.halt("drill")
        _, _, second = _service(risk_state=recorder.load_risk_state, listeners=[recorder])
        second.prepare()
        assert second.engine.risk.halted and second.engine.risk.halt_reason == "drill"
        assert second.engine.risk.state.high_water_mark == first.engine.risk.state.high_water_mark
