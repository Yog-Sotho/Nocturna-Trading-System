"""
Tests for the trading core: market data, simulated broker, risk gate, engine,
backtest runner, and parity between the backtest and live (polling) paths.
"""

import dataclasses
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.tests.trading_fixtures import BIL, BTC, IEF, SPY, bar, daily_bars, ny
from src.trading.backtest import BacktestConfig, run_backtest
from src.trading.broker import DEFAULT_COSTS, CostModel, SimulatedBroker
from src.trading.clock import SimulatedClock
from src.trading.data import AlpacaBarSource, DataQualityError, InMemoryBarSource, to_bar
from src.trading.domain import (
    DAILY,
    NEW_YORK,
    AssetClass,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from src.trading.engine import Allocation, Engine
from src.trading.feed import PollingFeed
from src.trading.ledger import Ledger
from src.trading.risk import RiskGate, RiskLimits
from src.trading.strategies.benchmark import BuyAndHold
from src.trading.strategy import CLOSE, Strategy, StrategyContext

D = Decimal
NO_COSTS = {AssetClass.EQUITY: CostModel(), AssetClass.CRYPTO: CostModel()}


# -- market data ---------------------------------------------------------------

class TestInMemoryBarSource:
    def test_bars_appear_only_after_the_feed_delay(self):
        b = bar(SPY, ny(2026, 3, 2), 100, 101, 99, 100.5)
        clock = SimulatedClock(b.end + timedelta(minutes=14))
        source = InMemoryBarSource({"SPY": [b]}, clock)
        assert source.bars([SPY], DAILY, b.start, b.end + timedelta(days=1))["SPY"] == []
        clock.advance_to(b.end + timedelta(minutes=15))
        assert source.bars([SPY], DAILY, b.start, b.end + timedelta(days=1))["SPY"] == [b]

    def test_range_is_half_open(self):
        bars = daily_bars(SPY, date(2026, 3, 2), date(2026, 3, 6))
        source = InMemoryBarSource({"SPY": bars}, SimulatedClock(datetime(2027, 1, 1, tzinfo=UTC)))
        got = source.bars([SPY], DAILY, bars[1].start, bars[3].start)["SPY"]
        assert got == bars[1:3]


class TestVendorBars:
    def test_rounding_artefacts_are_repaired(self):
        b = to_bar("SPY", ny(2026, 3, 2), DAILY, AssetClass.EQUITY, 100.0, 99.999, 99.0, 99.5, 1.0)
        assert b.high == 100.0

    def test_real_inconsistencies_are_rejected(self):
        with pytest.raises(DataQualityError):
            to_bar("SPY", ny(2026, 3, 2), DAILY, AssetClass.EQUITY, 100.0, 98.0, 97.0, 99.5, 1.0)


class _FakeBarClient:
    def __init__(self, bars):
        self.bars, self.requests = bars, []

    def _answer(self, request):
        self.requests.append(request)
        return SimpleNamespace(data={s: self.bars.get(s, []) for s in request.symbol_or_symbols})

    get_stock_bars = get_crypto_bars = _answer


def _vendor(start, price=100.0):
    return SimpleNamespace(timestamp=start, open=price, high=price + 1, low=price - 1, close=price, volume=10.0)


class TestAlpacaBarSource:
    def test_equity_requests_are_sip_adjusted_and_never_too_recent(self):
        from alpaca.data.enums import Adjustment, DataFeed
        day1, day2 = ny(2026, 3, 2), ny(2026, 3, 3)
        now = day2 + timedelta(hours=12)  # day 2's bar is still forming
        stocks = _FakeBarClient({"SPY": [_vendor(day1), _vendor(day2)]})
        source = AlpacaBarSource(SimulatedClock(now), stock_client=stocks, crypto_client=_FakeBarClient({}))
        result = source.bars([SPY], DAILY, day1, now + timedelta(days=5))
        request = stocks.requests[0]
        assert request.feed is DataFeed.SIP and request.adjustment is Adjustment.ALL
        # alpaca-py stores naive UTC and serializes it with the UTC offset
        assert request.to_request_fields()["end"] == (now - timedelta(minutes=15)).isoformat()
        assert [b.start for b in result["SPY"]] == [day1]
        assert result["SPY"][0].end == day2

    def test_raw_equity_bars_are_the_prices_as_traded(self):
        from alpaca.data.enums import Adjustment
        day1 = ny(2026, 3, 2)
        stocks = _FakeBarClient({"SPY": [_vendor(day1)]})
        source = AlpacaBarSource(SimulatedClock(day1 + timedelta(days=2)), stock_client=stocks,
                                 crypto_client=_FakeBarClient({}), raw=True)
        source.bars([SPY], DAILY, day1, day1 + timedelta(days=1))
        assert stocks.requests[0].adjustment is Adjustment.RAW

    def test_crypto_bars_are_utc_days(self):
        start = datetime(2026, 3, 7, tzinfo=UTC)
        crypto = _FakeBarClient({"BTC/USD": [_vendor(start, 60000.0)]})
        source = AlpacaBarSource(SimulatedClock(start + timedelta(days=2)), crypto_client=crypto)
        [b] = source.bars([BTC], DAILY, start, start + timedelta(days=1))["BTC/USD"]
        assert b.end == start + timedelta(days=1)

    def test_equities_need_api_keys(self):
        source = AlpacaBarSource(SimulatedClock(ny(2026, 3, 5)), crypto_client=_FakeBarClient({}))
        with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
            source.bars([SPY], DAILY, ny(2026, 3, 2), ny(2026, 3, 4))


# -- simulated broker -------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.updates = []

    def __call__(self, update):
        self.updates.append(update)

    def statuses(self, cid):
        return [u.status for u in self.updates if u.client_order_id == cid]

    def fills(self):
        return [u.fill for u in self.updates if u.fill is not None]


def _broker(now, cash="1000000", costs=NO_COSTS, **kw):
    clock = SimulatedClock(now)
    broker = SimulatedBroker(clock, {"SPY": SPY, "BTC/USD": BTC, "IEF": IEF}, D(cash), costs=costs, **kw)
    recorder = _Recorder()
    broker.set_listener(recorder)
    return clock, broker, recorder


def _req(cid, symbol="SPY", side=Side.BUY, qty="10", **kw):
    return OrderRequest(cid, symbol, side, D(qty), **kw)


class TestSimulatedBrokerTiming:
    def test_order_before_the_open_fills_at_that_sessions_open(self):
        clock, broker, rec = _broker(ny(2026, 3, 3, 0, 15))
        broker.submit(_req("o1", time_in_force=TimeInForce.OPG))
        b = bar(SPY, ny(2026, 3, 3), 100, 102, 99, 101)
        clock.advance_to(b.end + timedelta(minutes=15))
        broker.on_bars({"SPY": b})
        assert rec.fills()[0].price == D("100")

    def test_order_after_the_open_waits_for_the_next_session(self):
        # Placed at 10:00, after the 09:30 open: filling at that open would be look-ahead
        clock, broker, rec = _broker(ny(2026, 3, 3, 10, 0))
        broker.submit(_req("o1", time_in_force=TimeInForce.GTC))
        same_day = bar(SPY, ny(2026, 3, 3), 100, 102, 99, 101)
        next_day = bar(SPY, ny(2026, 3, 4), 105, 106, 104, 105)
        clock.advance_to(next_day.start)
        broker.on_bars({"SPY": same_day})
        assert rec.fills() == []
        clock.advance_to(next_day.end + timedelta(minutes=15))
        broker.on_bars({"SPY": next_day})
        assert rec.fills()[0].price == D("105")

    def test_crypto_order_just_after_midnight_gets_the_open(self):
        day = datetime(2026, 3, 3, tzinfo=UTC)
        clock, broker, rec = _broker(day + timedelta(minutes=1))
        broker.submit(_req("o1", symbol="BTC/USD", qty="0.5", time_in_force=TimeInForce.GTC))
        b = bar(BTC, day, 60000, 61000, 59000, 60500)
        clock.advance_to(b.end + timedelta(minutes=1))
        broker.on_bars({"BTC/USD": b})
        assert rec.fills()[0].price == D("60000")


class TestSimulatedBrokerPrices:
    START = ny(2026, 3, 3, 0, 15)
    BAR = bar(SPY, ny(2026, 3, 3), 100, 104, 96, 101)

    def _run(self, request, costs=NO_COSTS, b=None):
        clock, broker, rec = _broker(self.START, costs=costs)
        broker.submit(request)
        b = b or self.BAR
        clock.advance_to(b.end + timedelta(minutes=15))
        broker.on_bars({"SPY": b})
        return rec

    def test_market_slippage_is_adverse(self):
        costs = {**NO_COSTS, AssetClass.EQUITY: CostModel(D("0"), D("10"))}
        buy = self._run(_req("b", time_in_force=TimeInForce.OPG), costs).fills()[0]
        sell = self._run(_req("s", side=Side.SELL, time_in_force=TimeInForce.OPG), costs)
        assert buy.price == D("100.1")
        assert sell.updates[-1].status is OrderStatus.REJECTED or sell.fills()[0].price == D("99.9")

    def test_limit_needs_trade_through(self):
        touched = bar(SPY, ny(2026, 3, 3), 100, 101, 97, 100)
        rec = self._run(_req("l", order_type=OrderType.LIMIT, limit_price=D("97")), b=touched)
        assert rec.fills() == [] and rec.statuses("l")[-1] is OrderStatus.EXPIRED
        rec = self._run(_req("l", order_type=OrderType.LIMIT, limit_price=D("98")), b=touched)
        assert rec.fills()[0].price == D("98")

    def test_limit_gets_a_better_open(self):
        rec = self._run(_req("l", order_type=OrderType.LIMIT, limit_price=D("103")))
        assert rec.fills()[0].price == D("100")

    def test_stop_gap_fills_at_the_open(self):
        gap = bar(SPY, ny(2026, 3, 3), 110, 112, 109, 111)
        rec = self._run(_req("s", order_type=OrderType.STOP, stop_price=D("105"), time_in_force=TimeInForce.GTC),
                        b=gap)
        assert rec.fills()[0].price == D("110")

    def test_stop_touch_fills_at_the_stop(self):
        rec = self._run(_req("s", order_type=OrderType.STOP, stop_price=D("103"), time_in_force=TimeInForce.GTC))
        assert rec.fills()[0].price == D("103")

    def test_cls_fills_at_the_close(self):
        rec = self._run(_req("c", time_in_force=TimeInForce.CLS))
        assert rec.fills()[0].price == D("101")

    def test_crypto_fee(self):
        day = datetime(2026, 3, 3, tzinfo=UTC)
        clock, broker, rec = _broker(day, costs=DEFAULT_COSTS)
        broker.submit(_req("c", symbol="BTC/USD", qty="1", time_in_force=TimeInForce.GTC))
        b = bar(BTC, day, 50000, 50000, 50000, 50000)
        clock.advance_to(b.end + timedelta(minutes=1))
        broker.on_bars({"BTC/USD": b})
        fill = rec.fills()[0]
        assert fill.price == D("50050")  # 10 bps slippage
        # Purchases pay the 25 bps fee in the asset received, as on Alpaca
        assert fill.fee == 0 and fill.fee_qty == D("0.0025")
        assert broker.positions() == {"BTC/USD": D("0.9975")}
        assert broker.cash() == D("1000000") - D("50050")

    def test_crypto_sale_pays_the_fee_in_cash(self):
        day = datetime(2026, 3, 3, tzinfo=UTC)
        clock, broker, rec = _broker(day, costs=DEFAULT_COSTS)
        broker._book.adjust_position("BTC/USD", D("1"), D("50000"))
        broker.submit(_req("c", symbol="BTC/USD", side=Side.SELL, qty="1", time_in_force=TimeInForce.GTC))
        b = bar(BTC, day, 50000, 50000, 50000, 50000)
        clock.advance_to(b.end + timedelta(minutes=1))
        broker.on_bars({"BTC/USD": b})
        fill = rec.fills()[0]
        assert fill.price == D("49950") and fill.fee == D("124.875") and fill.fee_qty == 0


class TestSimulatedBrokerLifecycle:
    def test_time_in_force(self):
        clock, broker, rec = _broker(ny(2026, 3, 3, 0, 15))
        no_fill = dict(order_type=OrderType.LIMIT, limit_price=D("1"))
        broker.submit(_req("day", time_in_force=TimeInForce.DAY, **no_fill))
        broker.submit(_req("ioc", time_in_force=TimeInForce.IOC, **no_fill))
        broker.submit(_req("gtc", time_in_force=TimeInForce.GTC, **no_fill))
        b = bar(SPY, ny(2026, 3, 3), 100, 101, 99, 100)
        clock.advance_to(b.end + timedelta(minutes=15))
        broker.on_bars({"SPY": b})
        assert rec.statuses("day")[-1] is OrderStatus.EXPIRED
        assert rec.statuses("ioc")[-1] is OrderStatus.CANCELED
        assert [o.client_order_id for o in broker.open_orders()] == ["gtc"]

    def test_participation_cap_fills_over_several_bars(self):
        clock, broker, rec = _broker(ny(2026, 3, 3, 0, 15), participation=D("0.1"))
        broker.submit(_req("o", qty="150", time_in_force=TimeInForce.GTC))
        for day in (3, 4):
            b = bar(SPY, ny(2026, 3, day), 100, 101, 99, 100, volume=1000)
            clock.advance_to(b.end + timedelta(minutes=15))
            broker.on_bars({"SPY": b})
        assert [f.qty for f in rec.fills()] == [D("100"), D("50")]
        assert rec.statuses("o")[-1] is OrderStatus.FILLED

    @pytest.mark.parametrize("request_,reason", [
        (_req("r", symbol="BTC/USD", side=Side.SELL, qty="1", time_in_force=TimeInForce.GTC), "cannot be sold short"),
        (_req("r", symbol="BTC/USD", qty="1", time_in_force=TimeInForce.DAY), "gtc or ioc"),
        (_req("r", symbol="BTC/USD", qty="0.00005", time_in_force=TimeInForce.GTC), "not a multiple"),
        (_req("r", order_type=OrderType.STOP_LIMIT, limit_price=D("1"), stop_price=D("1")), "not simulated"),
        (_req("r", symbol="XYZ"), "unknown symbol"),
    ])
    def test_rejections(self, request_, reason):
        _, broker, rec = _broker(ny(2026, 3, 3))
        broker.submit(request_)
        assert rec.updates[-1].status is OrderStatus.REJECTED
        assert reason in rec.updates[-1].reason

    def test_duplicate_client_order_id_is_ignored(self):
        _, broker, rec = _broker(ny(2026, 3, 3))
        broker.submit(_req("same"))
        broker.submit(_req("same"))
        assert rec.statuses("same") == [OrderStatus.NEW]

    def test_insufficient_cash(self):
        clock, broker, rec = _broker(ny(2026, 3, 3, 0, 15), cash="500")
        broker.submit(_req("o", qty="10", time_in_force=TimeInForce.OPG))
        b = bar(SPY, ny(2026, 3, 3), 100, 101, 99, 100)
        clock.advance_to(b.end + timedelta(minutes=15))
        broker.on_bars({"SPY": b})
        assert rec.statuses("o")[-1] is OrderStatus.REJECTED
        assert broker.cash() == D("500")


# -- engine -------------------------------------------------------------------------

class _Fixed(Strategy):
    name = "fixed"
    lookback = 1

    def __init__(self, weights, instruments=(SPY, IEF)):
        self.instruments = instruments
        self.weights = weights
        self.seen = []

    def target_weights(self, ctx: StrategyContext) -> Mapping[str, float] | None:
        self.seen.append(ctx)
        return self.weights() if callable(self.weights) else self.weights


def _engine(strategy, now=None, cash="100000", costs=NO_COSTS, **kw):
    clock = SimulatedClock(now or ny(2026, 3, 2))
    broker = SimulatedBroker(clock, {i.symbol: i for i in strategy.instruments}, D(cash), costs=costs)
    events = []
    engine = Engine(clock=clock, broker=broker, ledger=Ledger(D(cash)),
                    allocations=[Allocation(strategy, D("1"))], listeners=[events.append], **kw)
    return clock, broker, engine, events


def _feed_day(clock, engine, day, spy=100.0, ief=50.0):
    bars = {"SPY": bar(SPY, ny(2026, 3, day), spy, spy, spy, spy),
            "IEF": bar(IEF, ny(2026, 3, day), ief, ief, ief, ief)}
    clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
    return engine.on_bars(bars)


class TestEngine:
    def test_sizes_orders_from_weights_with_a_cash_buffer(self):
        clock, _, engine, _ = _engine(_Fixed({"SPY": 0.6, "IEF": 0.4}))
        sent = _feed_day(clock, engine, 2)
        by_symbol = {r.symbol: r for r in sent}
        # $500 cash buffer scales every buy by 0.995, keeping the 60/40 ratio
        assert by_symbol["SPY"].qty == D("597") and by_symbol["IEF"].qty == D("796")
        assert all(r.time_in_force is TimeInForce.OPG for r in sent)

    def test_fills_on_the_next_session_and_does_not_resend(self):
        clock, broker, engine, _ = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        assert len(_feed_day(clock, engine, 2)) == 1
        assert _feed_day(clock, engine, 3) == []  # filled at 03-03 open, now on target
        assert engine.ledger.position("SPY").qty == D("500")
        assert broker.positions() == {"SPY": D("500")}
        assert engine.ledger.cash == broker.cash()

    def test_small_drift_does_not_churn_but_closing_always_trades(self):
        targets = {"SPY": 0.5}
        strategy = _Fixed(lambda: dict(targets), instruments=(SPY,))
        clock, _, engine, _ = _engine(strategy)
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        assert _feed_day(clock, engine, 4, spy=100.1) == []
        targets["SPY"] = 0.0
        [close] = _feed_day(clock, engine, 5)
        assert close.side is Side.SELL and close.qty == D("500")

    def test_sells_go_before_buys(self):
        weights = {"SPY": 1.0, "IEF": 0.0}
        clock, _, engine, _ = _engine(_Fixed(lambda: dict(weights)))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        weights.update(SPY=0.0, IEF=0.9)
        sent = _feed_day(clock, engine, 4)
        assert [r.side for r in sent] == [Side.SELL, Side.BUY]

    def test_short_target_on_unshortable_instrument_is_ignored(self):
        clock, _, engine, _ = _engine(_Fixed({"BTC/USD": -0.5}, instruments=(BTC,)), now=datetime(2026, 3, 2, tzinfo=UTC))
        b = bar(BTC, datetime(2026, 3, 2, tzinfo=UTC), 60000, 60000, 60000, 60000)
        clock.advance_to(b.end + timedelta(minutes=1))
        assert engine.on_bars({"BTC/USD": b}) == []

    def test_a_failing_strategy_does_not_stop_the_engine(self):
        def boom():
            raise RuntimeError("bug")
        clock, _, engine, events = _engine(_Fixed(boom))
        assert _feed_day(clock, engine, 2) == []
        assert any(e.kind == "strategy_error" and "bug" in e.detail for e in events)

    def test_targets_outside_the_universe_are_an_error(self):
        clock, _, engine, events = _engine(_Fixed({"QQQ": 0.5}))
        assert _feed_day(clock, engine, 2) == []
        assert any(e.kind == "strategy_error" for e in events)

    def test_repeated_bars_are_ignored(self):
        strategy = _Fixed({"SPY": 0.5}, instruments=(SPY,))
        clock, _, engine, _ = _engine(strategy)
        b = bar(SPY, ny(2026, 3, 2), 100, 100, 100, 100)
        clock.advance_to(b.end + timedelta(minutes=15))
        engine.on_bars({"SPY": b})
        engine.on_bars({"SPY": b})
        assert len(strategy.seen) == 1

    def test_halt_flattens_and_blocks_new_orders(self):
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        [close] = engine.halt("drill", flatten=True)
        flatten = engine.orders[close].request
        assert flatten.side is Side.SELL and flatten.qty == D("500")
        assert _feed_day(clock, engine, 4) == []
        assert engine.ledger.positions == {}
        assert any(e.kind == "halt" for e in events)
        engine.resume()
        assert len(_feed_day(clock, engine, 5)) == 1

    def test_targets_are_capped_and_rejections_are_reported(self):
        clock, _, engine, events = _engine(_Fixed({"SPY": 0.9}, instruments=(SPY,)),
                                           risk=RiskGate(RiskLimits(max_position_weight=D("0.5"))))
        [order] = _feed_day(clock, engine, 2)
        assert order.qty == D("500")  # sized to the 50% cap, not rejected
        # The working order counts as exposure: 100 more shares would breach the cap
        assert not engine.submit_order(_req("op-1", qty="100")).allowed
        assert any(e.kind == "rejected" for e in events)

    def test_fill_for_an_unknown_order_is_left_to_reconciliation(self):
        from src.trading.domain import Fill, OrderUpdate
        from src.trading.reconcile import reconcile
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        fill = Fill("ext-1", "manual-1", "SPY", Side.BUY, D("3"), D("100"), D("0"), clock.now())
        engine.on_order_update(OrderUpdate("manual-1", OrderStatus.FILLED, clock.now(), fill=fill))
        assert engine.ledger.position("SPY").qty == 0
        assert any(e.kind == "unknown_fill" for e in events)
        # The manual trade happened at the broker; reconciliation adopts it once
        broker._book.adjust_position("SPY", D("3"), D("100"))
        broker._book.adjust_cash(D("-300"))
        reconcile(engine, broker)
        reconcile(engine, broker)
        assert engine.ledger.position("SPY").qty == D("3")
        assert engine.ledger.cash == broker.cash()

    def test_allocations_are_validated(self):
        clock = SimulatedClock(ny(2026, 3, 2))
        broker = SimulatedBroker(clock, {"SPY": SPY}, D("1"))
        with pytest.raises(ValueError):
            Engine(clock=clock, broker=broker, ledger=Ledger(D("1")),
                   allocations=[Allocation(_Fixed({}, (SPY,)), D("0.7")), Allocation(_Fixed({}, (SPY,)), D("0.7"))])


class TestPollSchedule:
    def test_next_poll_is_when_the_next_bar_becomes_final(self):
        source = InMemoryBarSource({}, SimulatedClock(ny(2026, 3, 3)))
        feed = PollingFeed(source, [SPY, BTC], DAILY, ny(2026, 3, 1))
        # 03-03 12:00 ET: next crypto bar is final at 00:01 UTC on 03-04, before
        # the equity bar (00:15 ET = 05:15 UTC)
        assert feed.next_poll(ny(2026, 3, 3, 12)) == datetime(2026, 3, 4, 0, 1, tzinfo=UTC)
        assert PollingFeed(source, [SPY], DAILY, ny(2026, 3, 1)).next_poll(ny(2026, 3, 3, 12)) == \
            ny(2026, 3, 4, 0, 15)

    def test_next_boundary_intraday(self):
        t = datetime(2026, 3, 3, 14, 7, 30, tzinfo=UTC)
        from src.trading.domain import Timeframe
        assert Timeframe(15, "minute").next_boundary(t, AssetClass.EQUITY) == datetime(2026, 3, 3, 14, 15, tzinfo=UTC)


class TestTimeInForce:
    @pytest.mark.parametrize("when,expected", [
        (ny(2026, 3, 3, 0, 15), TimeInForce.OPG),    # overnight before the open
        (ny(2026, 3, 3, 9, 29), TimeInForce.DAY),    # inside Alpaca's OPG cutoff
        (ny(2026, 3, 3, 11, 0), TimeInForce.DAY),    # during the session
        (ny(2026, 3, 3, 17, 0), TimeInForce.DAY),    # after the close, before 19:00
        (ny(2026, 3, 3, 20, 0), TimeInForce.OPG),    # evening before the next session
    ])
    def test_equity_market_orders(self, when, expected):
        _, _, engine, _ = _engine(_Fixed({}, (SPY,)), now=when)
        assert engine._time_in_force(SPY, when) is expected

    def test_saturday_is_inside_the_opg_window(self):
        # Friday 19:00 -> Monday 09:28 is one window
        _, _, engine, _ = _engine(_Fixed({}, (SPY,)), now=ny(2026, 3, 7, 12, 0))
        assert engine._time_in_force(SPY, ny(2026, 3, 7, 12, 0)) is TimeInForce.OPG

    def test_crypto_is_gtc(self):
        _, _, engine, _ = _engine(_Fixed({}, (BTC,)))
        assert engine._time_in_force(BTC, ny(2026, 3, 3)) is TimeInForce.GTC


# -- the cash sleeve and the closing auction -------------------------------------------

class _AtClose(_Fixed):
    name = "at-close"
    execution = CLOSE


def _sleeved(strategy, now=None, cash="100000"):
    clock = SimulatedClock(now or ny(2026, 3, 2))
    instruments = {i.symbol: i for i in (*strategy.instruments, BIL)}
    broker = SimulatedBroker(clock, instruments, D(cash), costs=NO_COSTS)
    events = []
    engine = Engine(clock=clock, broker=broker, ledger=Ledger(D(cash)), allocations=[Allocation(strategy, D("1"))],
                    cash_sleeve=BIL, listeners=[events.append])
    return clock, broker, engine, events


def _feed_sleeved(clock, engine, day, spy=100.0, bil=91.5):
    bars = {"SPY": bar(SPY, ny(2026, 3, day), spy, spy, spy, spy),
            "BIL": bar(BIL, ny(2026, 3, day), bil, bil, bil, bil)}
    clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
    return {(r.symbol, r.side): r for r in engine.on_bars(bars)}


class TestCashSleeve:
    def test_idle_equity_goes_into_the_sleeve_after_the_strategy(self):
        clock, _, engine, _ = _sleeved(_Fixed({"SPY": 0.5}, (SPY,)))
        sent = _feed_sleeved(clock, engine, 2)
        # $99,500 to spend after the 0.5% buffer: $50,000 of SPY, the rest in BIL
        assert sent[("SPY", Side.BUY)].qty == D("500") and sent[("BIL", Side.BUY)].qty == D("540")

    def test_the_sleeve_may_hold_more_than_the_position_cap(self):
        clock, _, engine, _ = _sleeved(_Fixed({"SPY": 0.0}, (SPY,)))
        assert _feed_sleeved(clock, engine, 2)[("BIL", Side.BUY)].qty == D("1087")  # 99.5% of equity

    def test_the_sleeve_is_sold_first_to_fund_a_position(self):
        weights = {"SPY": 0.0}
        clock, _, engine, _ = _sleeved(_Fixed(lambda: dict(weights), (SPY,)))
        _feed_sleeved(clock, engine, 2)
        _feed_sleeved(clock, engine, 3)
        assert engine.ledger.position("BIL").qty == D("1087")
        weights["SPY"] = 0.5
        sent = list(_feed_sleeved(clock, engine, 4).values())
        assert [(r.symbol, r.side) for r in sent] == [("BIL", Side.SELL), ("SPY", Side.BUY)]
        assert sent[0].qty == D("547") and sent[1].qty == D("500")  # BIL down to 49.5%, SPY at 50% of equity

    def test_targets_moving_inside_the_band_do_not_trade_the_sleeve(self):
        """Regression: the sleeve followed the targets, so two drifts inside the band sold T-bills for cash."""
        weights = {"SPY": 0.3, "IEF": 0.2}
        clock, _, engine, _ = _sleeved(_Fixed(lambda: dict(weights), (SPY, IEF)))

        def day(d):
            bars = {s: bar(i, ny(2026, 3, d), p, p, p, p) for s, i, p in (("SPY", SPY, 100.0), ("IEF", IEF, 50.0),
                                                                          ("BIL", BIL, 91.5))}
            clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
            return engine.on_bars(bars)

        day(2)
        day(3)
        weights.update(SPY=0.315, IEF=0.215)  # each 1.5% of equity: inside the 2% band, 3% together
        assert day(4) == []
        weights.update(SPY=0.33)  # SPY now 3% off its holding: it trades, and BIL pays for it
        assert [(r.symbol, r.side) for r in day(5)] == [("BIL", Side.SELL), ("SPY", Side.BUY)]

    def test_short_of_cash_the_strategy_is_scaled_and_the_sleeve_stays_empty(self):
        clock, _, engine, _ = _sleeved(_Fixed({"SPY": 0.5}, (SPY,)), cash="10000")
        engine.cash_buffer = D("0.6")  # $4,000 to spend against the $5,000 the strategy wants
        sent = _feed_sleeved(clock, engine, 2)
        assert sent[("SPY", Side.BUY)].qty == D("40") and ("BIL", Side.BUY) not in sent

    def test_bought_during_a_cool_down_but_not_while_halted(self):
        clock, _, engine, _ = _sleeved(_Fixed({"SPY": 0.5}, (SPY,)))
        engine.risk.state.cooling_down = True
        engine.risk.state.day = ny(2026, 3, 3).astimezone(NEW_YORK).date()
        sent = _feed_sleeved(clock, engine, 2)
        assert list(sent) == [("BIL", Side.BUY)]
        clock2, _, halted, _ = _sleeved(_Fixed({"SPY": 0.5}, (SPY,)))
        halted.halt("drill")
        assert _feed_sleeved(clock2, halted, 2) == {}

    def test_the_kill_switch_sells_it_too(self):
        clock, _, engine, _ = _sleeved(_Fixed({"SPY": 0.5}, (SPY,)))
        _feed_sleeved(clock, engine, 2)
        _feed_sleeved(clock, engine, 3)
        engine.halt("drill", flatten=True)
        _feed_sleeved(clock, engine, 4)
        assert engine.ledger.positions == {}

    def test_a_sleeve_must_be_an_equity_no_strategy_trades(self):
        clock = SimulatedClock(ny(2026, 3, 2))
        broker = SimulatedBroker(clock, {"SPY": SPY, "BTC/USD": BTC}, D("1"))
        for sleeve in (SPY, BTC):
            with pytest.raises(ValueError, match="cash sleeve"):
                Engine(clock=clock, broker=broker, ledger=Ledger(D("1")),
                       allocations=[Allocation(_Fixed({}, (SPY,)), D("1"))], cash_sleeve=sleeve)

    def test_a_backtest_earns_the_sleeve(self):
        history = _history()
        history["BIL"] = daily_bars(BIL, date(2025, 1, 2), date(2026, 6, 30), 91.0, seed=5, drift=0.00015,
                                    vol=0.0001)
        plain = run_backtest([Allocation(BuyAndHold({SPY: 0.3}), D("1"))], history, CONFIG)
        sleeved = run_backtest([Allocation(BuyAndHold({SPY: 0.3}), D("1"))], history,
                               dataclasses.replace(CONFIG, cash_sleeve=BIL))
        held = {p.symbol: p.market_value for p in sleeved.final.positions}
        assert set(held) == {"SPY", "BIL"} and held["BIL"] > sleeved.final.equity * D("0.6")
        assert sleeved.final.equity > plain.final.equity


class TestClosingAuction:
    @pytest.mark.parametrize("when,expected", [
        (ny(2026, 3, 3, 0, 15), TimeInForce.CLS),    # the bar of 03-02 is final: trade at today's close
        (ny(2026, 3, 3, 15, 49), TimeInForce.CLS),
        (ny(2026, 3, 3, 15, 50), None),              # Alpaca rejects CLS from 15:50
        (ny(2026, 3, 3, 18, 59), None),
        (ny(2026, 3, 3, 19, 0), TimeInForce.CLS),    # queued for the next session's close
        (ny(2026, 3, 7, 0, 15), TimeInForce.CLS),    # Saturday: Monday's close
        (ny(2026, 11, 27, 12, 49), TimeInForce.CLS),  # the Friday after Thanksgiving closes at 13:00
        (ny(2026, 11, 27, 12, 50), None),
    ])
    def test_orders_go_to_the_next_closing_auction(self, when, expected):
        _, _, engine, _ = _engine(_AtClose({}, (SPY,)), now=when)
        assert engine._time_in_force(SPY, when) is expected
        assert engine._time_in_force(BTC, when) is TimeInForce.GTC

    def test_a_close_strategy_fills_at_the_close_of_the_next_session(self):
        clock, broker, engine, _ = _engine(_AtClose({"SPY": 0.5}, (SPY,)))
        [order] = _feed_day(clock, engine, 2)
        assert order.time_in_force is TimeInForce.CLS
        bars = {"SPY": bar(SPY, ny(2026, 3, 3), 100, 104, 99, 102), "IEF": bar(IEF, ny(2026, 3, 3), 50, 50, 50, 50)}
        clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
        engine.on_bars(bars)
        [fill] = [o for o in engine.orders.values() if o.filled_qty]
        assert fill.avg_fill_price == D("102")  # the close, not the open

    def test_no_orders_while_the_closing_auction_refuses_them(self):
        clock, _, engine, _ = _engine(_AtClose({"SPY": 0.5}, (SPY,)), now=ny(2026, 3, 3, 16))
        b = bar(SPY, ny(2026, 3, 2), 100, 100, 100, 100)
        assert engine.on_bars({"SPY": b}) == []
        clock.advance_to(ny(2026, 3, 4, 0, 15))  # the next bar: the orders go out
        assert len(engine.on_bars({"SPY": bar(SPY, ny(2026, 3, 3), 100, 100, 100, 100)})) == 1

    def test_open_and_close_strategies_cannot_share_an_engine(self):
        clock = SimulatedClock(ny(2026, 3, 2))
        broker = SimulatedBroker(clock, {"SPY": SPY}, D("1"))
        with pytest.raises(ValueError, match="open and at the close"):
            Engine(clock=clock, broker=broker, ledger=Ledger(D("1")),
                   allocations=[Allocation(_Fixed({}, (SPY,)), D("0.5")), Allocation(_AtClose({}, (SPY,)), D("0.5"))])


# -- backtest and parity -------------------------------------------------------------

def _history(start=date(2025, 1, 2), end=date(2026, 6, 30)):
    return {"SPY": daily_bars(SPY, start, end, 500.0, seed=1),
            "IEF": daily_bars(IEF, start, end, 95.0, seed=2, drift=0.0001, vol=0.004),
            "BTC/USD": daily_bars(BTC, start, end, 60000.0, seed=3, vol=0.03)}


def _sixty_forty():
    return [Allocation(BuyAndHold({SPY: 0.6, IEF: 0.4}), D("0.9")),
            Allocation(type("Btc", (BuyAndHold,), {"name": "btc-hold"})({BTC: 1.0}), D("0.1"))]


CONFIG = BacktestConfig(start=datetime(2025, 3, 3, tzinfo=UTC), end=datetime(2026, 6, 1, tzinfo=UTC))


class TestBacktest:
    def test_no_strategy_ever_sees_an_unfinished_bar(self):
        seen = []

        class Spy(_Fixed):
            def target_weights(self, ctx):
                for series in ctx.history.values():
                    seen.append((ctx.now, series[-1]))
                return {"SPY": 0.5}

        run_backtest([Allocation(Spy({}, (SPY, BTC)), D("1"))], _history(), CONFIG)
        delays = {"SPY": timedelta(minutes=15), "BTC/USD": timedelta(minutes=1)}
        assert seen and all(b.end + delays[b.symbol] <= now for now, b in seen)

    def test_deterministic(self):
        a = run_backtest(_sixty_forty(), _history(), CONFIG)
        b = run_backtest(_sixty_forty(), _history(), CONFIG)
        assert a.equity_curve == b.equity_curve and a.fills == b.fills

    def test_ledger_matches_the_broker_and_marks_to_market(self):
        result = run_backtest(_sixty_forty(), _history(), CONFIG)
        final = result.final
        assert {p.symbol for p in final.positions} == {"SPY", "IEF", "BTC/USD"}
        assert final.equity == final.cash + sum(p.market_value for p in final.positions)
        assert final.cash >= 0
        metrics = result.metrics()
        assert -1 < metrics["max_drawdown"] <= 0 and metrics["fills"] > 3

    def test_higher_costs_lower_returns(self):
        cheap = run_backtest(_sixty_forty(), _history(), CONFIG)
        dear = run_backtest(_sixty_forty(), _history(), BacktestConfig(
            start=CONFIG.start, end=CONFIG.end,
            costs={k: v.scaled(D("3")) for k, v in DEFAULT_COSTS.items()}))
        assert dear.final.equity < cheap.final.equity
        assert dear.final.fees > cheap.final.fees


class TestLivePathParity:
    def test_polling_the_live_feed_reproduces_the_backtest(self):
        """
        The live path (wall-clock polling of a bar source) must trade exactly
        like the backtest: same fills, same prices, same final equity.
        """
        history = _history()
        backtest = run_backtest(_sixty_forty(), history, CONFIG)

        clock = SimulatedClock(CONFIG.start)
        allocations = _sixty_forty()
        instruments = {i.symbol: i for a in allocations for i in a.strategy.instruments}
        broker = SimulatedBroker(clock, instruments, CONFIG.starting_cash)
        engine = Engine(clock=clock, broker=broker, ledger=Ledger(CONFIG.starting_cash), allocations=allocations)
        source = InMemoryBarSource(history, clock)
        engine.warm_up(source.bars(list(instruments.values()), DAILY, datetime(2000, 1, 1, tzinfo=UTC),
                                   CONFIG.start))
        feed = PollingFeed(source, list(instruments.values()), DAILY, datetime(2000, 1, 1, tzinfo=UTC),
                           seen=engine.latest_bar_starts())
        fills = []
        engine.add_listener(lambda e: fills.append(e.fill) if e.kind == "fill" else None)

        now = CONFIG.start
        while True:
            # Wake when the next bar becomes final, plus a few seconds of real-world latency
            now = feed.next_poll(now) + timedelta(seconds=7)
            if now >= CONFIG.end:
                break
            clock.advance_to(now)
            for _, group in feed.poll(now):
                engine.on_bars(group)

        def key(f):
            return f.symbol, f.side, f.qty, f.price

        assert [key(f) for f in fills] == [key(f) for f in backtest.fills]
        assert engine.equity() == backtest.final.equity


class TestReconcile:
    def _desynced(self):
        from src.trading.reconcile import reconcile
        clock, broker, engine, events = _engine(_Fixed({"SPY": 0.5}, instruments=(SPY,)))
        _feed_day(clock, engine, 2)
        _feed_day(clock, engine, 3)
        return clock, broker, engine, events, reconcile

    def test_in_sync_engine_reports_nothing(self):
        _, broker, engine, events, reconcile = self._desynced()
        assert reconcile(engine, broker).in_sync
        assert not any(e.kind == "reconciled" for e in events)

    def test_position_and_cash_drift_is_adopted_and_reported(self):
        _, broker, engine, events, reconcile = self._desynced()
        broker._book.adjust_position("SPY", D("-2"), D("100"))  # e.g. a fee or a manual sale
        broker._book.adjust_cash(D("1000"))                     # a deposit
        report = reconcile(engine, broker)
        assert {d.item: d.delta for d in report.discrepancies} == {"SPY": D("-2"), "cash": D("1000")}
        assert engine.ledger.position("SPY").qty == broker.positions()["SPY"]
        assert engine.ledger.cash == broker.cash()
        assert any(e.kind == "reconciled" for e in events)
        assert reconcile(engine, broker).in_sync

    def test_open_orders_are_adopted_and_lost_ones_closed(self):
        _, broker, engine, _, reconcile = self._desynced()
        broker.submit(OrderRequest("manual-9", "SPY", Side.BUY, D("1"), time_in_force=TimeInForce.GTC))
        from src.trading.domain import Order
        lost = Order.new(OrderRequest("lost-1", "SPY", Side.BUY, D("1")), engine.clock.now())
        engine.sync_order(lost)
        report = reconcile(engine, broker)
        assert report.adopted_orders == ("manual-9",)
        assert report.closed_orders == ("lost-1",)
        assert engine.orders["lost-1"].status is OrderStatus.CANCELED
        assert engine.exposure()["SPY"] == engine.ledger.position("SPY").qty + 1


class TestLedgerFromAccount:
    def test_start_from_broker_positions(self):
        ledger = Ledger.from_account(D("1000"), {"SPY": (D("10"), D("5000")), "IEF": (D("0"), D("0"))})
        assert ledger.starting_cash == D("6000")
        assert ledger.positions == {"SPY": ledger.position("SPY")}
        assert ledger.equity({"SPY": D("510")}) == D("6100")


class TestCloseOut:
    """Symbols held over from an earlier strategy mix are sold, whatever their size."""

    def _engine(self, close_out=(IEF,)):
        clock = SimulatedClock(ny(2026, 3, 2))
        broker = SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, D("100000"), costs=NO_COSTS)
        ledger = Ledger.from_account(D("99000"), {"IEF": (D("20"), D("1000"))})  # 1% of equity: inside the band
        engine = Engine(clock=clock, broker=broker, ledger=ledger, allocations=[Allocation(_Fixed({"SPY": 0.5},
                                                                                                  (SPY,)), D("1"))],
                        close_out=close_out)
        return clock, engine

    def _day(self, clock, engine, day):
        bars = {"SPY": bar(SPY, ny(2026, 3, day), 100.0, 100.0, 100.0, 100.0),
                "IEF": bar(IEF, ny(2026, 3, day), 50.0, 50.0, 50.0, 50.0)}
        clock.advance_to(bars["SPY"].end + timedelta(minutes=15))
        return {(r.symbol, r.side): r for r in engine.on_bars(bars)}

    def test_a_held_over_position_is_sold_in_full(self):
        clock, engine = self._engine()
        assert engine.close_out == ("IEF",) and engine.target_weights()["IEF"] == 0
        sent = self._day(clock, engine, 2)
        assert sent[("IEF", Side.SELL)].qty == D("20")  # all of it, though it is inside the 2% band
        assert ("SPY", Side.BUY) in sent

    def test_without_close_out_a_foreign_position_stays(self):
        clock, engine = self._engine(close_out=())
        assert ("IEF", Side.SELL) not in self._day(clock, engine, 2)

    def test_a_symbol_a_strategy_trades_is_not_closed_out(self):
        clock, engine = self._engine(close_out=(SPY, IEF))
        assert engine.close_out == ("IEF",)
        assert self._day(clock, engine, 2)[("SPY", Side.BUY)].qty > 0

    def test_conflicting_definitions_are_refused(self):
        other = dataclasses.replace(IEF, min_qty=D("5"))
        with pytest.raises(ValueError, match="conflicting definitions for IEF"):
            Engine(clock=SimulatedClock(ny(2026, 3, 2)), broker=SimulatedBroker(SimulatedClock(ny(2026, 3, 2)),
                                                                                  {"IEF": IEF}, D("1")),
                   ledger=Ledger(D("1")), allocations=[Allocation(_Fixed({"IEF": 0.5}, (IEF,)), D("1"))],
                   close_out=[other])

    def test_the_cash_sleeve_cannot_be_closed_out(self):
        clock = SimulatedClock(ny(2026, 3, 2))
        with pytest.raises(ValueError, match="cash sleeve BIL must not be traded by a strategy or closed out"):
            Engine(clock=clock, broker=SimulatedBroker(clock, {"SPY": SPY, "BIL": BIL}, D("1")),
                   ledger=Ledger(D("1")), allocations=[Allocation(_Fixed({"SPY": 0.5}, (SPY,)), D("1"))],
                   cash_sleeve=BIL, close_out=[BIL])
