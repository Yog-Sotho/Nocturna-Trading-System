"""
Tests for the engine as its own process: the database channel, the engine
worker, the API's remote runtime and the trading routes in external mode.
The worker runs on a thread here; the API side reaches it only through the
database, as it would across processes.
"""

import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from flask import Flask

from src.models.user import db
from src.tests.trading_fixtures import IEF, SPY, daily_bars, ny
from src.trading.broker import SimulatedBroker
from src.trading.channel import CommandTimeoutError, EngineChannel
from src.trading.clock import SimulatedClock
from src.trading.data import InMemoryBarSource
from src.trading.lineup import build_lineup
from src.trading.operator import OperatorOrder, UnknownSymbolError
from src.trading.runtime import EngineUnavailableError, RemoteRuntime, TradingRuntime
from src.trading.service import EngineService, ServiceNotRunningError
from src.trading.worker import AnotherWorkerRunningError, EngineWorker

D = Decimal


def _factory(flask_app, settings, profile=None):
    clock = SimulatedClock(ny(2026, 3, 2, 12))
    history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
               "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
    broker = SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, D("100000"))
    return EngineService(clock=clock, source=InMemoryBarSource(history, clock), broker=broker,
                         allocations=build_lineup({"SPY": SPY, "IEF": IEF}))


class _Broker:
    def __init__(self):
        self.calls = []

    def close_all(self):
        self.calls.append("close_all")
        return ["c-1"]

    def cancel_all(self):
        self.calls.append("cancel_all")
        return ["o-1"]


@pytest.fixture
def settings(app):
    from pydantic import SecretStr
    return app.extensions["settings"].model_copy(update={
        "alpaca_api_key": SecretStr("key"), "alpaca_secret_key": SecretStr("secret"), "engine_mode": "external"})


@pytest.fixture
def shared_app(tmp_path):
    """
    The database the API and the engine process share: a file, not the suite's
    in-memory database. That one is a single connection shared by every thread,
    where one thread's rollback can undo another's commit; two processes on a
    server database each have their own connections.
    """
    shared = Flask("engine-channel-test")
    shared.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{tmp_path / 'channel.db'}"
    db.init_app(shared)
    with shared.app_context():
        db.create_all()
    yield shared
    with shared.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.fixture
def worker(shared_app, settings):
    """An engine worker on its own thread, and the API's remote runtime."""
    local = TradingRuntime(shared_app, settings, factory=_factory)
    engine_worker = EngineWorker(local, EngineChannel(shared_app), interval=0.02, instance="test-worker")
    stop = threading.Event()
    thread = threading.Thread(target=engine_worker.run, args=(stop,), daemon=True)
    thread.start()
    remote = RemoteRuntime(shared_app, settings, broker_factory=lambda s: _Broker(), timeout=10)
    _wait(lambda: remote.views()["status"]["state"] == "stopped")
    yield remote, local
    stop.set()
    thread.join(10)


def _wait(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def _order(cid="op-1", symbol="SPY", qty="1"):
    return OperatorOrder(cid, symbol, "buy", D(qty), tag="operator:1")


class TestRemoteRuntime:
    def test_commands_and_views_go_through_the_database(self, worker):
        remote, local = worker
        assert remote.views()["positions"] is None
        remote.start()
        _wait(lambda: remote.views()["status"]["state"] == "running")
        views = remote.views()
        assert views["status"]["equity"] == "100000" and len(views["working"]) == 2  # the 60/40 opening orders
        decision, order = remote.submit_order(_order())
        assert decision.allowed and order["symbol"] == "SPY" and order["status"] == "new"
        assert remote.cancel_order("op-1") is True and remote.cancel_order("nope") is False
        halted = remote.emergency_stop("drill", flatten=False)
        assert halted["engine"] == "halted"
        _wait(lambda: remote.views()["status"].get("halted") is True)
        remote.resume()
        _wait(lambda: remote.views()["status"].get("halted") is False)
        remote.stop()
        _wait(lambda: remote.views()["status"]["state"] == "stopped")

    def test_engine_errors_come_back_as_the_same_exceptions(self, worker):
        remote, _ = worker
        with pytest.raises(ServiceNotRunningError):
            remote.resume()
        remote.start()
        _wait(lambda: remote.views()["status"]["state"] == "running")
        with pytest.raises(UnknownSymbolError):
            remote.submit_order(_order(symbol="QQQ"))
        with pytest.raises(RuntimeError):
            remote.start()  # already running

    def test_without_an_engine_process(self, app, settings):
        broker = _Broker()
        remote = RemoteRuntime(app, settings, broker_factory=lambda s: broker, timeout=0.2)
        assert remote.views()["status"] == {"configured": True, "state": "unreachable"}
        with pytest.raises(EngineUnavailableError):
            remote.start()
        # The emergency stop still reaches the broker
        assert remote.emergency_stop("drill", flatten=True) == {"engine": "not answering", "close_orders": ["c-1"]}
        assert remote.emergency_stop("drill", flatten=False)["cancelled_orders"] == ["o-1"]
        assert broker.calls == ["close_all", "cancel_all"]

    def test_a_stale_heartbeat_means_unreachable(self, app, settings):
        channel = EngineChannel(app, clock=SimulatedClock(datetime.now(UTC) - timedelta(minutes=5)))
        channel.publish("old", {"status": {"state": "running"}, "positions": None, "targets": None, "working": []})
        assert RemoteRuntime(app, settings).views()["status"]["state"] == "unreachable"


class TestChannel:
    def test_a_command_the_engine_never_started_is_withdrawn(self, app):
        channel = EngineChannel(app)
        command = channel.send("submit_order", _order().to_dict())
        with pytest.raises(CommandTimeoutError):
            channel.wait(command, timeout=0.1)
        assert channel.take_pending() == []  # it can no longer fill after the API gave up

    def test_halts_are_never_withdrawn_or_expired(self, app):
        channel = EngineChannel(app, clock=SimulatedClock(datetime.now(UTC) - timedelta(days=1)))
        command = channel.send("halt", {"reason": "drill", "flatten": True})
        from src.trading import channel as channel_module
        original, channel_module.RUNNING_GRACE = channel_module.RUNNING_GRACE, 0.05
        try:
            with pytest.raises(CommandTimeoutError):
                channel.wait(command, timeout=0.05)
        finally:
            channel_module.RUNNING_GRACE = original
        [(command_id, kind, _)] = EngineChannel(app).take_pending()
        assert (command_id, kind) == (command, "halt")

    def test_expired_commands_are_skipped(self, app):
        old = EngineChannel(app, clock=SimulatedClock(datetime.now(UTC) - timedelta(minutes=5)))
        old.send("submit_order", _order().to_dict())
        assert EngineChannel(app).take_pending() == []


class TestWorker:
    def _worker(self, app, settings, instance="me"):
        return EngineWorker(TradingRuntime(app, settings, factory=_factory), EngineChannel(app), instance=instance)

    def test_refuses_to_run_beside_another_live_worker(self, app, settings):
        EngineChannel(app).publish("other", {"status": {"state": "running"}})
        with pytest.raises(AnotherWorkerRunningError):
            self._worker(app, settings).ensure_alone()
        self._worker(app, settings, instance="other").ensure_alone()  # the same worker after a restart

    def test_a_stopped_or_stale_worker_does_not_block(self, app, settings):
        EngineChannel(app).publish("other", {"status": {"state": "stopped"}})
        self._worker(app, settings).ensure_alone()
        stale = EngineChannel(app, clock=SimulatedClock(datetime.now(UTC) - timedelta(minutes=5)))
        stale.publish("other", {"status": {"state": "running"}})
        self._worker(app, settings).ensure_alone()

    def test_unknown_commands_fail_without_stopping_the_worker(self, app, settings):
        channel = EngineChannel(app)
        command = channel.send("stop", {})
        from src.models.trading import EngineCommand
        from src.models.user import db
        with app.app_context():
            db.session.get(EngineCommand, command).kind = "dance"
            db.session.commit()
        worker = self._worker(app, settings)
        worker.step()
        with app.app_context():
            row = db.session.get(EngineCommand, command)
            assert row.status == "failed" and "unknown command" in row.error


class TestExternalModeRoutes:
    def test_routes_read_and_command_through_the_worker(self, app, client, admin_headers, auth_headers, worker):
        remote, _ = worker
        previous = app.extensions["trading"]
        app.extensions["trading"] = remote
        try:
            assert client.get("/api/v1/trading/health").get_json()["data"]["engine"] == "stopped"
            assert client.post("/api/v1/trading/control/start", headers=admin_headers).status_code == 202
            _wait(lambda: client.get("/api/v1/trading/status", headers=auth_headers)
                  .get_json()["data"]["state"] == "running")
            placed = client.post("/api/v1/trading/orders", headers=auth_headers,
                                 json={"symbol": "SPY", "side": "buy", "qty": "1"})
            assert placed.status_code == 201 and placed.get_json()["data"]["order"]["symbol"] == "SPY"
            assert client.get("/api/v1/trading/positions", headers=auth_headers).status_code == 200
        finally:
            app.extensions["trading"] = previous

    def test_an_unreachable_engine_is_a_503(self, app, client, admin_headers, settings):
        previous = app.extensions["trading"]
        app.extensions["trading"] = RemoteRuntime(app, settings, timeout=0.2)
        try:
            assert client.get("/api/v1/trading/health").get_json()["data"]["engine"] == "unreachable"
            assert client.post("/api/v1/trading/control/start", headers=admin_headers).status_code == 503
        finally:
            app.extensions["trading"] = previous


def test_engine_run_refuses_embedded_mode(app):
    result = app.test_cli_runner().invoke(args=["engine", "run"])
    assert result.exit_code != 0 and "ENGINE_MODE=external" in result.output
