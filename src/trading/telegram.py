"""
NOCTURNA Trading System - Telegram commands.

With TELEGRAM_COMMANDS=true the bot that sends the alerts also takes commands,
from the alert chat (ALERT_TELEGRAM_CHAT_ID) only, and only when that chat is
the private chat with the bot:

    /status        engine state, equity, drawdown, halt and cool-down
    /positions     positions and working orders
    /start_engine  start the engine (after a restart, with ENGINE_AUTOSTART=false)
    /stop_engine   stop the engine; positions and working orders stay at Alpaca
    /resume        lift a halt; the high-water mark restarts at the current equity
    /halt          emergency stop that keeps the positions (cancels every order)
    /flatten       emergency stop that also closes every position

Telegram's own /start (the Start button) shows the help. Every command that
changes something waits for ``/confirm <code>`` within ``CONFIRM_WINDOW``: a
tap on the command menu sends at once. Changes sent while nothing was
listening (the server was down) are not run.

The bot runs in the web API's process and acts through the same runtime as
the API's routes, so with the engine in a process of its own, ``/status``
reports an engine that does not answer and ``/halt`` and ``/flatten`` still
reach Alpaca. Every action is recorded in the audit log.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from flask import Flask

from .alerts import TELEGRAM_MAX_CHARS, _describe_failure

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
#: Seconds a getUpdates call waits for a message before returning empty
POLL_TIMEOUT = 50
#: How long a change waits for /confirm
CONFIRM_WINDOW = 60.0
#: A command older than this was sent while nothing was listening
STALE_COMMAND = 120.0
#: Waits after failed polls, growing to the last one
BACKOFF = (5.0, 15.0, 60.0)

COMMANDS = {
    "status": "Engine state, equity, drawdown, halt",
    "positions": "Positions and working orders",
    "start_engine": "Start the engine",
    "stop_engine": "Stop the engine (positions stay)",
    "resume": "Lift a halt",
    "halt": "Emergency stop, keeping positions",
    "flatten": "Emergency stop, closing every position",
    "help": "What the commands do",
}

#: Commands that change something, and how each is described in the confirmation
CHANGES = {
    "start_engine": "start the engine",
    "stop_engine": "stop the engine",
    "resume": "lift the halt and restart the high-water mark at the current equity",
    "halt": "halt trading and cancel every order, keeping the positions",
    "flatten": "halt trading, cancel every order and CLOSE EVERY POSITION",
}

Call = Callable[[str, Mapping[str, Any], float], Any]


def telegram_call(token: str) -> Call:
    """A Bot API caller: method, parameters and timeout in, the ``result`` out."""

    def call(method: str, params: Mapping[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(
            API.format(token=token, method=method), data=json.dumps(params).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        # A fixed https URL, so B310 (file:// and custom schemes) does not apply
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            body = json.loads(response.read())
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method}: {body.get('description', 'not ok')}")
        return body.get("result")

    return call


@dataclass
class _Pending:
    command: str
    argument: str
    code: str
    expires: float


class TelegramCommands:
    def __init__(self, app: Flask, runtime_of: Callable[[], Any], chat_id: str, call: Call,
                 clock: Callable[[], float] = time.monotonic, now: Callable[[], float] = time.time) -> None:
        """
        Args:
            runtime_of: Returns the runtime to act through (read on every command, as the routes do).
            call: Sends one Bot API request (``telegram_call``).
            clock: Monotonic seconds, for confirmation windows.
            now: Unix seconds, to compare with the messages' dates.
        """
        self._app = app
        self._runtime_of = runtime_of
        self._chat_id = str(chat_id)
        self._call = call
        self._clock = clock
        self._now = now
        self._offset: int | None = None
        self._pending: _Pending | None = None

    # -- polling --------------------------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        try:
            self._call("setMyCommands", {"commands": [{"command": c, "description": d}
                                                      for c, d in COMMANDS.items()]}, 10.0)
        except Exception as e:
            logger.warning("Telegram command menu not set: %s", _describe_failure(e))
        failures = 0
        while not stop.is_set():
            try:
                self.poll_once()
                failures = 0
            except Exception as e:
                delay = BACKOFF[min(failures, len(BACKOFF) - 1)]
                failures += 1
                logger.warning("Telegram polling failed (%s); retrying in %.0f s", _describe_failure(e), delay)
                stop.wait(delay)

    def poll_once(self, timeout: int = POLL_TIMEOUT) -> None:
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if self._offset is not None:
            params["offset"] = self._offset
        for update in self._call("getUpdates", params, timeout + 10.0) or []:
            # Confirmed by the next call's offset, so each one is handled once even if handling fails
            self._offset = int(update["update_id"]) + 1
            message = update.get("message")
            if message:
                self._receive(message)

    def _receive(self, message: Mapping[str, Any]) -> None:
        chat = message.get("chat") or {}
        if str(chat.get("id")) != self._chat_id or chat.get("type") != "private":
            logger.warning("Ignored a Telegram message from chat %s (%s)", chat.get("id"), chat.get("type"))
            return
        try:
            reply = self.handle(str(message.get("text") or ""), float(message.get("date") or 0))
        except Exception:
            logger.exception("Telegram command failed")
            reply = "That command failed; see the API's log."
        if reply:
            self._reply(reply)

    def _reply(self, text: str) -> None:
        try:
            self._call("sendMessage", {"chat_id": self._chat_id, "text": text[:TELEGRAM_MAX_CHARS]}, 15.0)
        except Exception as e:
            logger.warning("Telegram reply failed: %s", _describe_failure(e))

    # -- commands -------------------------------------------------------------------

    def handle(self, text: str, sent_at: float) -> str | None:
        """The reply to one message from the operator's chat."""
        if not text.startswith("/"):
            return None
        head, _, argument = text[1:].partition(" ")
        command = head.split("@", 1)[0].lower()
        argument = argument.strip()
        if command == "status":
            return self._status()
        if command == "positions":
            return self._positions()
        if command not in CHANGES and command != "confirm":
            return _help()
        age = self._now() - sent_at
        if age > STALE_COMMAND:
            return (f"Not run: /{command} was sent {int(age // 60)} min ago, while nothing was listening. "
                    "Send it again if you still mean it.")
        if command == "confirm":
            return self._confirm(argument)
        code = f"{secrets.randbelow(10_000):04d}"
        self._pending = _Pending(command, argument, code, self._clock() + CONFIRM_WINDOW)
        return f"This will {CHANGES[command]}.\nSend /confirm {code} within {int(CONFIRM_WINDOW)} s."

    def _confirm(self, code: str) -> str:
        pending, self._pending = self._pending, None
        if pending is None or self._clock() > pending.expires:
            return "Nothing to confirm (or it expired): send the command again."
        if not secrets.compare_digest(code, pending.code):
            return "Wrong code: nothing done. Send the command again."
        return self._execute(pending.command, pending.argument)

    def _execute(self, command: str, argument: str) -> str:
        from src.trading.runtime import EngineUnavailableError, TradingNotConfiguredError
        from src.trading.service import ServiceNotRunningError
        from src.utils.audit import record_audit_event

        runtime = self._runtime_of()
        event = {"start_engine": "ENGINE_START", "stop_engine": "ENGINE_STOP",
                 "resume": "ENGINE_RESUME"}.get(command, "EMERGENCY_STOP")
        details: dict[str, Any] = {"source": "telegram"}
        with self._app.app_context():
            try:
                if command == "start_engine":
                    runtime.start()
                    reply = "Engine starting. /status shows when it runs."
                elif command == "stop_engine":
                    runtime.stop()
                    reply = "Engine stopped. Positions and working orders stay at Alpaca."
                elif command == "resume":
                    runtime.resume()
                    reply = "Halt lifted."
                else:
                    reason = f"telegram: {argument or command}"
                    details.update(reason=reason, close_all_positions=command == "flatten")
                    result = runtime.emergency_stop(reason, flatten=command == "flatten")
                    reply = _emergency_reply(result)
            except TradingNotConfiguredError as e:
                reply = str(e)
            except EngineUnavailableError:
                reply = "The engine process is not answering. /status for details."
            except ServiceNotRunningError:
                reply = "The engine is not running: /start_engine first."
            except RuntimeError as e:
                reply = f"Not done: {e}"
            else:
                record_audit_event(event, True, details)
                return reply
            record_audit_event(event, False, details, error=reply[:200])
            return reply

    def _status(self) -> str:
        views = self._runtime_of().views()
        status = views["status"]
        lines = [f"Engine: {status['state']}"]
        if not status.get("configured"):
            lines.append("Alpaca keys are not set.")
        if status.get("equity") is not None:
            lines.append(f"Equity: {_money(status['equity'])}  Cash: {_money(status.get('cash'))}")
        risk = status.get("risk") or {}
        if risk.get("drawdown") is not None:
            lines.append(f"Drawdown: {float(risk['drawdown']):.2%}")
        if status.get("halted"):
            lines.append(f"HALTED: {status.get('halt_reason') or ''}".rstrip())
        if risk.get("cooling_down"):
            lines.append("Daily loss limit reached: no new orders today.")
        if status.get("open_orders") is not None:
            lines.append(f"Positions: {status.get('positions')}  Working orders: {status['open_orders']}")
        if status.get("last_poll"):
            lines.append(f"Last market data: {_when(status['last_poll'])}")
        if status.get("last_error"):
            lines.append(f"Error: {status['last_error']}")
        profile = views.get("profile")
        if profile:
            lines.append(f"Mix: {_mix(profile)}")
        return "\n".join(lines)

    def _positions(self) -> str:
        views = self._runtime_of().views()
        positions = (views.get("positions") or {}).get("positions")
        if positions is None:
            return f"No positions to show: the engine is {views['status']['state']}."
        lines = [f"{p['symbol']}: {p['qty']} @ {_money(p['price'])} = {_money(p['market_value'])} "
                 f"(P&L {_money(p['unrealized_pnl'])})" for p in positions] or ["No positions."]
        working = views.get("working") or []
        if working:
            lines.append("Working orders:")
            lines += [f"{o['side']} {o['qty']} {o['symbol']} ({o['order_type']}, {o['status']})" for o in working]
        return "\n".join(lines)


def _mix(profile: Mapping[str, Any]) -> str:
    """``60% benchmark, 25% tsmom, 15% T-bills (profile #3)``."""
    shares = [(Decimal(e["weight"]), e["strategy"]) for e in profile["mix"]]
    rest = Decimal(1) - sum((w for w, _ in shares), Decimal(0))
    parts = [f"{w:.0%} {name}" for w, name in shares] + ([f"{rest:.0%} T-bills"] if rest > 0 else [])
    which = f"profile #{profile['id']}" if profile.get("id") is not None else "the default profile"
    return f"{', '.join(parts)} ({which})"


def _help() -> str:
    return "\n".join(f"/{c} - {d}" for c, d in COMMANDS.items()) + \
        "\nChanges ask for /confirm <code> first."


def _emergency_reply(result: Mapping[str, Any]) -> str:
    closing = len(result.get("close_orders") or [])
    cancelled = result.get("cancelled_orders")
    parts = [f"Emergency stop done (engine: {result.get('engine')})."]
    if closing:
        parts.append(f"{closing} closing orders sent.")
    if cancelled is not None:
        parts.append(f"{len(cancelled)} orders cancelled at Alpaca.")
    return " ".join(parts)


def _money(value: Any) -> str:
    return "-" if value is None else f"${float(value):,.2f}"


def _when(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def start_telegram_commands(app: Flask) -> threading.Thread | None:
    """
    Take Telegram commands in this process (TELEGRAM_COMMANDS=true). Call in the
    serving process only (after gunicorn's fork): one bot token must have
    exactly one poller.
    """
    settings = app.extensions["settings"]
    if not settings.telegram_commands:
        return None
    bot = TelegramCommands(app, lambda: app.extensions["trading"], settings.alert_telegram_chat_id,
                           telegram_call(settings.alert_telegram_bot_token.get_secret_value()))
    thread = threading.Thread(target=bot.run, args=(threading.Event(),), name="telegram-commands", daemon=True)
    thread.start()
    return thread
