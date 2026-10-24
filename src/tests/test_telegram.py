"""
Tests for Telegram commands: who may command, confirmation, what each command
does through the runtime, polling and the Bot API calls.
"""

import io
import json
import threading

import pytest

from src.trading import telegram as telegram_module
from src.trading.runtime import EngineUnavailableError, RemoteRuntime, TradingNotConfiguredError
from src.trading.service import ServiceNotRunningError
from src.trading.telegram import COMMANDS, TelegramCommands, start_telegram_commands, telegram_call

CHAT = "4242"
NOW = 1_800_000_000.0


class Bot:
    """A fake Bot API: getUpdates answers come from a queue; everything sent is recorded."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def __call__(self, method, params, timeout):
        self.calls.append((method, dict(params)))
        if method == "getUpdates":
            batch = self.batches.pop(0) if self.batches else []
            if isinstance(batch, Exception):
                raise batch
            return batch
        return True

    def replies(self):
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


class Runtime:
    """The runtime interface the routes use, recording what was asked of it."""

    def __init__(self, views=None, error=None):
        self.done = []
        self.error = error
        self._views = views or {"status": {"configured": True, "state": "stopped"}, "positions": None,
                                "targets": None, "working": []}

    def views(self):
        return self._views

    def _act(self, name, result=None):
        if self.error is not None:
            raise self.error
        self.done.append(name)
        return result

    def start(self):
        return self._act("start")

    def stop(self):
        return self._act("stop")

    def resume(self):
        return self._act("resume")

    def emergency_stop(self, reason, flatten):
        return self._act(("emergency_stop", reason, flatten),
                         {"engine": "halted", "close_orders": ["c-1", "c-2"] if flatten else []})


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _message(text, chat_id=CHAT, chat_type="private", date=NOW):
    return {"message_id": 1, "date": date, "text": text, "chat": {"id": int(chat_id), "type": chat_type}}


def _update(update_id, text, **kwargs):
    return {"update_id": update_id, "message": _message(text, **kwargs)}


@pytest.fixture
def make(app):
    def build(runtime=None, bot=None, clock=None):
        runtime = runtime or Runtime()
        bot = bot or Bot()
        commands = TelegramCommands(app, lambda: runtime, CHAT, bot, clock=clock or Clock(), now=lambda: NOW)
        return commands, runtime, bot
    return build


def _confirm_code(reply):
    return reply.rsplit("/confirm ", 1)[1].split()[0]


class TestWhoMayCommand:
    def test_only_the_private_alert_chat(self, make):
        commands, runtime, bot = make(bot=Bot([_update(1, "/status", chat_id="999"),
                                               _update(2, "/status", chat_type="group"),
                                               _update(3, "/status")]))
        commands.poll_once()
        assert len(bot.replies()) == 1 and bot.replies()[0].startswith("Engine: stopped")

    def test_plain_text_is_ignored(self, make):
        commands, _, _ = make()
        assert commands.handle("hello", NOW) is None


class TestChanges:
    def test_need_the_code(self, make, app):
        commands, runtime, _ = make()
        reply = commands.handle("/halt market looks broken", NOW)
        assert "halt trading and cancel every order, keeping the positions" in reply
        assert runtime.done == []
        assert commands.handle("/confirm 00000", NOW).startswith("Wrong code")
        assert commands.handle(f"/confirm {_confirm_code(reply)}", NOW).startswith("Nothing to confirm")
        reply = commands.handle("/halt market looks broken", NOW)
        assert commands.handle(f"/confirm {_confirm_code(reply)}", NOW) == "Emergency stop done (engine: halted)."
        assert runtime.done == [("emergency_stop", "telegram: market looks broken", False)]
        from src.models.user import AuditLog
        with app.app_context():
            row = AuditLog.query.filter_by(event_type="EMERGENCY_STOP").one()
        assert row.success and row.event_data["source"] == "telegram" and not row.event_data["close_all_positions"]

    def test_flatten_and_the_engine_commands(self, make):
        commands, runtime, _ = make()
        for command in ("flatten", "start_engine", "stop_engine", "resume"):
            reply = commands.handle(f"/{command}", NOW)
            commands.handle(f"/confirm {_confirm_code(reply)}", NOW)
        assert runtime.done == [("emergency_stop", "telegram: flatten", True), "start", "stop", "resume"]

    def test_the_code_expires(self, make):
        clock = Clock()
        commands, runtime, _ = make(clock=clock)
        reply = commands.handle("/start_engine", NOW)
        clock.t += telegram_module.CONFIRM_WINDOW + 1
        assert commands.handle(f"/confirm {_confirm_code(reply)}", NOW).startswith("Nothing to confirm")
        assert runtime.done == []

    def test_changes_sent_while_nothing_listened_are_not_run(self, make):
        commands, runtime, _ = make()
        old = NOW - 600
        assert commands.handle("/flatten", old).startswith("Not run: /flatten was sent 10 min ago")
        assert commands.handle("/status", old).startswith("Engine:")  # reads are still answered

    @pytest.mark.parametrize("error,expected", [
        (TradingNotConfiguredError("Set ALPACA_API_KEY"), "Set ALPACA_API_KEY"),
        (EngineUnavailableError("down"), "The engine process is not answering"),
        (ServiceNotRunningError("not running"), "The engine is not running: /start_engine first."),
        (RuntimeError("engine service is already running"), "Not done: engine service is already running"),
    ])
    def test_failures_are_explained_and_audited(self, make, app, error, expected):
        commands, _, _ = make(runtime=Runtime(error=error))
        reply = commands.handle("/start_engine", NOW)
        assert commands.handle(f"/confirm {_confirm_code(reply)}", NOW).startswith(expected)
        from src.models.user import AuditLog
        with app.app_context():
            assert AuditLog.query.filter_by(event_type="ENGINE_START", success=False).count() == 1

    def test_emergency_stop_reaches_the_broker_without_an_engine(self, make, app, settings_with_keys):
        class Broker:
            def close_all(self):
                return ["c-1", "c-2", "c-3"]

        runtime = RemoteRuntime(app, settings_with_keys, broker_factory=lambda s: Broker(), timeout=0.2)
        commands, _, _ = make(runtime=runtime)
        reply = commands.handle("/flatten", NOW)
        assert commands.handle(f"/confirm {_confirm_code(reply)}", NOW) == \
            "Emergency stop done (engine: not answering). 3 closing orders sent."


class TestReads:
    def test_status(self, make):
        views = {"status": {"configured": True, "state": "running", "equity": "101234.5", "cash": "2000",
                            "halted": True, "halt_reason": "drawdown 10.0%", "open_orders": 1, "positions": 2,
                            "last_poll": "2026-03-02T21:05:00+00:00", "last_error": None,
                            "risk": {"drawdown": "0.1002", "cooling_down": True}},
                 "positions": None, "targets": None, "working": []}
        commands, _, _ = make(runtime=Runtime(views))
        assert commands.handle("/status", NOW) == "\n".join([
            "Engine: running", "Equity: $101,234.50  Cash: $2,000.00", "Drawdown: 10.02%",
            "HALTED: drawdown 10.0%", "Daily loss limit reached: no new orders today.",
            "Positions: 2  Working orders: 1", "Last market data: 2026-03-02 21:05 UTC"])

    def test_status_names_the_mix(self, make):
        mix = [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "0.6000"},
               {"strategy": "tsmom", "params": {"rule": "sign", "long_only": True}, "weight": "0.2500"}]
        views = {"status": {"configured": True, "state": "running"}, "profile": {"id": 3, "mix": mix}}
        commands, _, _ = make(runtime=Runtime(views))
        assert commands.handle("/status", NOW) == \
            "Engine: running\nMix: 60% benchmark, 25% tsmom, 15% T-bills (profile #3)"
        whole = {"id": None, "mix": [{"strategy": "benchmark", "params": {"equity": 0.6}, "weight": "1"}]}
        commands, _, _ = make(runtime=Runtime({**views, "profile": whole}))
        assert commands.handle("/status", NOW).endswith("Mix: 100% benchmark (the default profile)")

    def test_status_without_keys(self, make):
        commands, _, _ = make(runtime=Runtime({"status": {"configured": False, "state": "unreachable"}}))
        assert commands.handle("/status", NOW) == "Engine: unreachable\nAlpaca keys are not set."

    def test_positions(self, make):
        views = {"status": {"state": "running"},
                 "positions": {"positions": [{"symbol": "SPY", "qty": "10", "price": "500", "market_value": "5000",
                                              "unrealized_pnl": "-12.5"}]},
                 "working": [{"side": "buy", "qty": "3", "symbol": "IEF", "order_type": "market", "status": "new"}]}
        commands, _, _ = make(runtime=Runtime(views))
        assert commands.handle("/positions", NOW) == (
            "SPY: 10 @ $500.00 = $5,000.00 (P&L $-12.50)\nWorking orders:\nbuy 3 IEF (market, new)")
        commands, _, _ = make()
        assert commands.handle("/positions", NOW) == "No positions to show: the engine is stopped."

    @pytest.mark.parametrize("text", ["/start", "/help", "/nonsense", "/status@NocturnaBot extra"])
    def test_help_and_the_start_button(self, make, text):
        commands, _, _ = make()
        reply = commands.handle(text, NOW)
        assert reply.startswith("Engine:") if text.startswith("/status") else "/start_engine - Start" in reply


class TestPolling:
    def test_offset_confirms_what_was_handled(self, make):
        commands, _, bot = make(bot=Bot([_update(7, "/status"), _update(8, "/help")], []))
        commands.poll_once(timeout=1)
        commands.poll_once(timeout=1)
        polls = [p for m, p in bot.calls if m == "getUpdates"]
        assert "offset" not in polls[0] and polls[1]["offset"] == 9
        assert polls[0]["allowed_updates"] == ["message"] and polls[0]["timeout"] == 1
        assert len(bot.replies()) == 2  # one reply per message

    def test_run_sets_the_menu_and_survives_failures(self, make, monkeypatch):
        monkeypatch.setattr(telegram_module, "BACKOFF", (0.0,))
        stop = threading.Event()

        class Stopping(Bot):
            def __call__(self, method, params, timeout):
                if method == "getUpdates" and not self.batches:
                    stop.set()
                return super().__call__(method, params, timeout)

        bot = Stopping(OSError("network down"), [_update(1, "/status")])
        commands, _, _ = make(bot=bot)
        commands.run(stop)
        menu = next(p for m, p in bot.calls if m == "setMyCommands")
        assert [c["command"] for c in menu["commands"]] == list(COMMANDS)
        assert all(1 <= len(c["command"]) <= 32 and c["command"].islower() for c in menu["commands"])
        assert bot.replies() and bot.replies()[0].startswith("Engine:")


class TestBotApi:
    def test_call(self, monkeypatch):
        seen = {}

        def urlopen(request, timeout):
            seen.update(url=request.full_url, body=json.loads(request.data), timeout=timeout)
            return io.BytesIO(json.dumps({"ok": True, "result": [1]}).encode())

        monkeypatch.setattr(telegram_module.urllib.request, "urlopen", urlopen)
        assert telegram_call("123:abc")("getUpdates", {"timeout": 5}, 15.0) == [1]
        assert seen == {"url": "https://api.telegram.org/bot123:abc/getUpdates", "body": {"timeout": 5},
                        "timeout": 15.0}

    def test_not_ok(self, monkeypatch):
        monkeypatch.setattr(telegram_module.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(
            json.dumps({"ok": False, "description": "Unauthorized"}).encode()))
        with pytest.raises(RuntimeError, match="Telegram sendMessage: Unauthorized"):
            telegram_call("t")("sendMessage", {}, 1.0)

    def test_off_unless_enabled(self, app):
        assert start_telegram_commands(app) is None


@pytest.fixture
def settings_with_keys(app):
    from pydantic import SecretStr
    return app.extensions["settings"].model_copy(update={
        "alpaca_api_key": SecretStr("key"), "alpaca_secret_key": SecretStr("secret"), "engine_mode": "external"})
