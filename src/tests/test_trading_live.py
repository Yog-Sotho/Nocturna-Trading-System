"""
Tests for the live trading path: the Alpaca broker adapter (against a fake
client but alpaca-py's real request models), the engine service, and the
recorder that writes engine history to the database.
"""

import json
import threading
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.tests.trading_fixtures import BTC, IEF, SPY, daily_bars, ny
from src.trading.alpaca import AlpacaBroker, TradeUpdateStream, load_instruments
from src.trading.broker import SimulatedBroker
from src.trading.clock import SimulatedClock
from src.trading.data import InMemoryBarSource
from src.trading.domain import (
    AssetClass,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from src.trading.engine import Allocation
from src.trading.service import EngineService, ServiceNotRunningError
from src.trading.strategies.benchmark import BuyAndHold

D = Decimal


# -- fake Alpaca client ------------------------------------------------------------

def _api_error(status, message):
    from alpaca.common.exceptions import APIError
    return APIError(json.dumps({"code": 40010001, "message": message}),
                    SimpleNamespace(response=SimpleNamespace(status_code=status), request=None))


def _alpaca_order(cid, symbol="SPY", side="buy", qty="10", status="accepted", order_type="market",
                  tif="opg", filled_qty="0", filled_avg_price=None, limit_price=None, stop_price=None):
    now = datetime(2026, 3, 3, 5, 15, tzinfo=UTC)
    return SimpleNamespace(id=uuid.uuid4(), client_order_id=cid, symbol=symbol, side=side, qty=qty,
                           status=status, order_type=order_type, type=order_type, time_in_force=tif,
                           filled_qty=filled_qty, filled_avg_price=filled_avg_price, limit_price=limit_price,
                           stop_price=stop_price, created_at=now, updated_at=now)


class FakeTradingClient:
    def __init__(self):
        self.submitted, self.orders, self.cancelled = [], {}, []
        self.fail_with = None
        self.positions = []
        self.account = SimpleNamespace(cash="100000")
        self.assets = {}

    def submit_order(self, request):
        self.submitted.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        order = _alpaca_order(request.client_order_id, request.symbol, request.side.value, str(request.qty),
                              tif=request.time_in_force.value)
        self.orders[request.client_order_id] = order
        return order

    def get_order_by_client_id(self, cid):
        if cid not in self.orders:
            raise _api_error(404, "order not found")
        return self.orders[cid]

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(order_id)

    def get_all_positions(self):
        return self.positions

    def get_account(self):
        return self.account

    def get_orders(self, request):
        return [o for o in self.orders.values() if o.status in ("new", "accepted", "partially_filled")]

    def get_asset(self, symbol):
        return self.assets[symbol]


def _alpaca(monkeypatch=None, instruments=None):
    client = FakeTradingClient()
    broker = AlpacaBroker(api_key="k", secret_key="s", instruments=instruments or {"SPY": SPY, "BTC/USD": BTC},
                          clock=SimulatedClock(ny(2026, 3, 3, 0, 15)), trading_client=client)
    updates = []
    broker.set_listener(updates.append)
    return client, broker, updates


class TestAlpacaBrokerOrders:
    def test_refuses_to_start_outside_paper_mode(self, monkeypatch):
        from src.trading.safety import LiveTradingDisabledError
        monkeypatch.setenv("TRADING_MODE", "LIVE")
        with pytest.raises(LiveTradingDisabledError):
            _alpaca()

    def test_market_opg_order(self):
        client, broker, updates = _alpaca()
        broker.submit(OrderRequest("nt-1", "SPY", Side.BUY, D("10"), time_in_force=TimeInForce.OPG))
        [sent] = client.submitted
        assert type(sent).__name__ == "MarketOrderRequest"
        assert sent.client_order_id == "nt-1" and sent.qty == 10 and sent.time_in_force.value == "opg"
        assert updates[-1].status is OrderStatus.NEW and updates[-1].broker_order_id

    @pytest.mark.parametrize("kw,cls", [
        ({"order_type": OrderType.LIMIT, "limit_price": D("101.5")}, "LimitOrderRequest"),
        ({"order_type": OrderType.STOP, "stop_price": D("95")}, "StopOrderRequest"),
        ({"order_type": OrderType.STOP_LIMIT, "stop_price": D("95"), "limit_price": D("94")}, "StopLimitOrderRequest"),
    ])
    def test_priced_orders(self, kw, cls):
        client, broker, _ = _alpaca()
        broker.submit(OrderRequest("nt-1", "SPY", Side.SELL, D("3"), time_in_force=TimeInForce.GTC, **kw))
        assert type(client.submitted[0]).__name__ == cls

    def test_crypto_quantities_keep_their_decimals(self):
        client, broker, _ = _alpaca()
        broker.submit(OrderRequest("nt-1", "BTC/USD", Side.BUY, D("0.1234"), time_in_force=TimeInForce.GTC))
        assert client.submitted[0].qty == pytest.approx(0.1234)

    def test_duplicate_submission_reports_the_original_order(self):
        client, broker, updates = _alpaca()
        client.orders["nt-1"] = _alpaca_order("nt-1", status="filled", filled_qty="10")
        client.fail_with = _api_error(422, "client_order_id must be unique")
        broker.submit(OrderRequest("nt-1", "SPY", Side.BUY, D("10")))
        assert updates[-1].status is OrderStatus.FILLED

    def test_rejection_carries_alpacas_reason(self):
        client, broker, updates = _alpaca()
        client.fail_with = _api_error(403, "insufficient buying power")
        broker.submit(OrderRequest("nt-1", "SPY", Side.BUY, D("10")))
        assert updates[-1].status is OrderStatus.REJECTED and "insufficient buying power" in updates[-1].reason

    def test_network_failure_checks_whether_the_order_exists(self):
        client, broker, updates = _alpaca()
        client.fail_with = ConnectionError("reset")
        broker.submit(OrderRequest("nt-1", "SPY", Side.BUY, D("10")))
        assert updates[-1].status is OrderStatus.REJECTED and "ConnectionError" in updates[-1].reason
        # ...but if it reached Alpaca after all, its real status is reported
        client.orders["nt-2"] = _alpaca_order("nt-2", status="accepted")
        broker.submit(OrderRequest("nt-2", "SPY", Side.BUY, D("10")))
        assert updates[-1].status is OrderStatus.NEW

    def test_cancel_uses_the_broker_id(self):
        client, broker, _ = _alpaca()
        broker.submit(OrderRequest("nt-1", "SPY", Side.BUY, D("10")))
        broker.cancel("nt-1")
        assert client.cancelled == [client.orders["nt-1"].id]
        broker.cancel("unknown")  # logged, not raised


class TestAlpacaTradeUpdates:
    def _update(self, event, order, qty=None, price=None, execution_id=None):
        return SimpleNamespace(event=event, order=order, qty=qty, price=price,
                               execution_id=execution_id or uuid.uuid4(),
                               timestamp=datetime(2026, 3, 3, 14, 30, tzinfo=UTC))

    def test_fill_becomes_a_domain_fill(self):
        _, broker, updates = _alpaca()
        order = _alpaca_order("nt-1", qty="10")
        execution = uuid.uuid4()
        broker.handle_trade_update(self._update("fill", order, qty=10, price=101.25, execution_id=execution))
        [update] = updates
        assert update.status is OrderStatus.FILLED
        assert update.fill.fill_id == f"alpaca:{execution}"
        assert (update.fill.qty, update.fill.price, update.fill.fee) == (D("10"), D("101.25"), 0)

    def test_crypto_fees_follow_alpacas_rules(self):
        _, broker, updates = _alpaca()
        buy = _alpaca_order("c-1", symbol="BTC/USD", qty="1")
        broker.handle_trade_update(self._update("fill", buy, qty=1, price=50000))
        assert updates[-1].fill.fee_qty == D("0.0025") and updates[-1].fill.fee == 0
        sell = _alpaca_order("c-2", symbol="BTCUSD", side="sell", qty="1")  # legacy symbol form
        broker.handle_trade_update(self._update("fill", sell, qty=1, price=50000))
        assert updates[-1].fill.symbol == "BTC/USD"
        assert updates[-1].fill.fee == D("125") and updates[-1].fill.fee_qty == 0

    @pytest.mark.parametrize("event,status", [
        ("new", OrderStatus.NEW), ("accepted", OrderStatus.NEW), ("canceled", OrderStatus.CANCELED),
        ("expired", OrderStatus.EXPIRED), ("rejected", OrderStatus.REJECTED),
        ("pending_cancel", OrderStatus.PENDING_CANCEL),
    ])
    def test_status_events(self, event, status):
        _, broker, updates = _alpaca()
        broker.handle_trade_update(self._update(event, _alpaca_order("nt-1")))
        assert updates[-1].status is status and updates[-1].fill is None

    def test_irrelevant_events_are_ignored(self):
        _, broker, updates = _alpaca()
        broker.handle_trade_update(self._update("restated", _alpaca_order("nt-1")))
        assert updates == []


class TestAlpacaAccount:
    def test_positions_use_side_for_the_sign_and_normalise_symbols(self):
        client, broker, _ = _alpaca()
        client.positions = [
            SimpleNamespace(symbol="SPY", qty="-5", side="short", avg_entry_price="500"),
            SimpleNamespace(symbol="BTCUSD", qty="0.5", side="long", avg_entry_price="60000"),
        ]
        assert broker.positions() == {"SPY": D("-5"), "BTC/USD": D("0.5")}
        assert broker.cost_basis() == {"SPY": (D("-5"), D("-2500")), "BTC/USD": (D("0.5"), D("30000.0"))}
        assert broker.cash() == D("100000")

    def test_open_orders_skip_types_the_engine_does_not_model(self):
        client, broker, _ = _alpaca()
        client.orders = {"a": _alpaca_order("a", status="partially_filled", filled_qty="4", filled_avg_price="100"),
                         "b": _alpaca_order("b", order_type="trailing_stop")}
        [order] = broker.open_orders()
        assert order.client_order_id == "a" and order.status is OrderStatus.PARTIALLY_FILLED
        assert order.filled_qty == D("4") and order.avg_fill_price == D("100")

    def test_load_instruments(self):
        client = FakeTradingClient()
        client.assets = {
            "BTC/USD": SimpleNamespace(tradable=True, status="active", min_trade_increment="0.0001",
                                       min_order_size="0.0002", shortable=False, easy_to_borrow=False),
            "SPY": SimpleNamespace(tradable=True, status="active", min_trade_increment=None, min_order_size=None,
                                   shortable=True, easy_to_borrow=True),
            "XYZ": SimpleNamespace(tradable=False, status="active", min_trade_increment=None, min_order_size=None,
                                   shortable=False, easy_to_borrow=False),
        }
        got = load_instruments(client, [("BTC/USD", AssetClass.CRYPTO), ("SPY", AssetClass.EQUITY)])
        assert got["BTC/USD"].qty_increment == D("0.0001") and got["BTC/USD"].min_qty == D("0.0002")
        assert got["SPY"].shortable and got["SPY"].qty_increment == D("1")
        with pytest.raises(ValueError, match="not tradable"):
            load_instruments(client, [("XYZ", AssetClass.EQUITY)])


class TestTradeUpdateStream:
    def test_runs_the_stream_in_a_thread_and_survives_handler_errors(self):
        import asyncio
        received = []

        class FakeStream:
            def subscribe_trade_updates(self, handler):
                self.handler = handler

            def run(self):
                asyncio.run(self.handler("first"))
                asyncio.run(self.handler("second"))

            def stop(self):
                pass

        def on_update(update):
            received.append(update)
            if update == "first":
                raise RuntimeError("bad update")

        stream = TradeUpdateStream(api_key="k", secret_key="s", on_update=on_update, stream=FakeStream())
        stream.start()
        stream._thread.join(5)
        stream.stop()
        assert received == ["first", "second"]


# -- engine service -------------------------------------------------------------------

def _service(start=ny(2026, 3, 2, 12), broker=None, **kw):
    history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
               "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
    clock = SimulatedClock(start)
    if broker is None:
        broker = SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, D("100000"))
    else:
        broker.clock = clock
    snapshots = []
    service = EngineService(clock=clock, source=InMemoryBarSource(history, clock), broker=broker,
                            allocations=[Allocation(BuyAndHold({SPY: 0.6, IEF: 0.4}), D("1"))],
                            snapshot_recorder=snapshots.append, **kw)
    return clock, broker, service, snapshots


def _run_until(clock, service, until):
    while clock.now() < until:
        clock.advance_to(min(service.status().next_poll, until))
        service.step()


class TestEngineService:
    def test_acts_on_the_newest_bars_at_start_up(self):
        _, _, service, _ = _service()
        service.prepare()
        status = service.status()
        assert status.state == "running"
        assert {o.request.symbol for o in status.snapshot.open_orders} == {"SPY", "IEF"}
        assert status.next_poll == ny(2026, 3, 3, 0, 15) + timedelta(seconds=5)

    def test_trades_reconciles_and_records_snapshots(self):
        clock, broker, service, snapshots = _service()
        service.prepare()
        _run_until(clock, service, ny(2026, 3, 6, 12))
        snapshot = service.status().snapshot
        assert {p.symbol for p in snapshot.positions} == {"SPY", "IEF"}
        assert not snapshot.open_orders
        assert service.status().last_reconciliation.in_sync
        assert snapshots and snapshots[-1].equity == snapshot.equity
        assert snapshot.cash == broker.cash()

    def test_restart_does_not_trade_twice(self):
        clock, broker, first, _ = _service()
        first.prepare()
        _run_until(clock, first, ny(2026, 3, 4, 12))
        working = [o.client_order_id for o in broker.open_orders()]
        held = broker.positions()
        _, _, restarted, _ = _service(start=clock.now(), broker=broker)
        restarted.prepare()
        # The restart adopts what is already working and submits nothing new
        assert [o.client_order_id for o in broker.open_orders()] == working
        assert sorted(o.client_order_id for o in restarted.status().snapshot.open_orders) == sorted(working)
        assert broker.positions() == held

    def test_commands_run_on_the_engine_thread(self):
        clock, broker, service, _ = _service()
        service.start()
        deadline = time.monotonic() + 10
        while service.status().state != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.status().state == "running"
        caller = threading.current_thread()
        threads = []
        service._call(lambda engine: threads.append(threading.current_thread()))
        assert threads and threads[0] is not caller and threads[0].name == "trading-engine"
        decision = service.submit_order(OrderRequest("manual-1", "SPY", Side.BUY, D("1")))
        assert decision.allowed
        service.halt("drill")
        assert service.status().snapshot.halted
        service.stop()
        assert service.status().state == "stopped"

    def test_commands_need_a_running_service(self):
        _, _, service, _ = _service()
        with pytest.raises(ServiceNotRunningError):
            service.halt("drill")

    def test_start_up_failure_is_reported(self):
        clock, broker, service, _ = _service()
        broker.cash = lambda: (_ for _ in ()).throw(RuntimeError("broker down"))
        service.start()
        service._thread.join(10)
        status = service.status()
        assert status.state == "failed" and "broker down" in status.last_error

    def test_market_data_failure_retries(self):
        clock, _, service, _ = _service()
        service.prepare()
        service._feed.poll = lambda now: (_ for _ in ()).throw(RuntimeError("data down"))
        due = service.status().next_poll
        clock.advance_to(due)
        service.step()
        status = service.status()
        assert "data down" in status.last_error
        assert status.next_poll == due + timedelta(minutes=1)


# -- recorder ---------------------------------------------------------------------------

class TestEngineRecorder:
    def test_orders_fills_events_and_snapshots_are_stored(self, app, db):
        from src.models.trading import EquitySnapshot, TradingFill, TradingOrder
        from src.models.user import AuditLog
        from src.trading.persistence import EngineRecorder
        clock, broker, service, _ = _service()
        recorder = EngineRecorder(app)
        events = []
        service._listeners.extend([recorder, events.append])
        service._snapshot_recorder = recorder.record_snapshot
        service.prepare()
        _run_until(clock, service, ny(2026, 3, 4, 12))
        service.engine.halt("drill")
        with app.app_context():
            orders = TradingOrder.query.all()
            assert len(orders) == len(service.engine.orders)
            assert {o.symbol for o in orders} == {"SPY", "IEF"}
            filled = [o for o in orders if o.status == "filled"]
            assert filled and all(o.filled_qty == o.qty for o in filled)
            fills = TradingFill.query.all()
            assert {f.symbol for f in fills} == {"SPY", "IEF"}
            assert EquitySnapshot.query.count() >= 1
            assert AuditLog.query.filter_by(event_type="ENGINE_HALT").count() == 1
        # Replaying the same events changes nothing
        with app.app_context():
            before = (TradingOrder.query.count(), TradingFill.query.count())
        for order in service.engine.orders.values():
            recorder(SimpleNamespace(kind="order", order=order, fill=None, detail=""))
        domain_fills = [e.fill for e in events if e.kind == "fill"]
        assert len(domain_fills) == before[1]
        for fill in domain_fills:
            recorder(SimpleNamespace(kind="fill", order=None, fill=fill, detail=""))
        with app.app_context():
            assert (TradingOrder.query.count(), TradingFill.query.count()) == before
