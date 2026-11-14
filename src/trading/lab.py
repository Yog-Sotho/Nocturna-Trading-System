"""
NOCTURNA Trading System - The strategy lab: backtests and validations on demand.

The dashboard queues a run; ``flask lab run``, a process of its own that the
web API starts (``ProcessLauncher``), executes the queued runs one at a time
and stores each result, so a computation of minutes never holds up the web
server, and never the engine. A run is one of:

* ``backtest``: a strategy mix with its risk limits, as the engine would trade
  it (``runner.backtest_mix``): the equity curve, the return of each year, and
  a summary against T-bills, next to the 60/40 over the same days, data and
  limits (unless the mix is the 60/40);
* ``validate``: the promotion gates for one strategy and its parameter grid,
  as ``flask validate`` runs them (``runner.validate_candidate``): the verdict,
  the tear sheet, and the equity the walk-forward traded out of sample.

Nothing here trades or changes the trading profile: results are for reading.
A run whose process died (a container restart) stops sending its heartbeat
and is marked failed. A queued run can be cancelled before it starts; a
running one is interrupted at its next heartbeat (``drain``). Finished runs
can be deleted. Each result keeps its whole daily equity, which ``get`` leaves
out and ``equity_csv`` turns into a download.
"""

from __future__ import annotations

import _thread
import csv
import io
import logging
import math
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask
from sqlalchemy import delete, func, select, update

from src.config import Settings

from .alerts import Alert, AlertDispatcher
from .catalog import STRATEGIES
from .profile import DEFAULT_PROFILE, Profile, ProfileError, effective_settings, parse, symbols
from .runtime import TradingNotConfiguredError
from .validation import runner, sources
from .validation.candidates import CANDIDATES

logger = logging.getLogger(__name__)

BACKTEST, VALIDATE = "backtest", "validate"
KINDS = (BACKTEST, VALIDATE)
QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)
#: Runs queued or running at once; more are refused until one finishes
MAX_WAITING = 3
#: The long history starts in 1970 (``history.py``)
EARLIEST = date(1970, 1, 1)
MIN_BACKTEST = timedelta(days=30)
#: The walk-forward's default: 36 months to choose the parameters, then at least 12 to test them
MIN_VALIDATE = timedelta(days=4 * 365)
#: Also how soon a running run notices that it was cancelled
HEARTBEAT_EVERY = 5.0
#: A running run without a heartbeat this long lost its process
LAB_STALE_AFTER = timedelta(minutes=5)
MAX_ERROR = 2000
#: The project root, where ``flask --app src.main`` finds the app
ROOT = Path(__file__).resolve().parents[2]


class LabError(ValueError):
    """The request cannot be run; the message says why."""


class LabBusyError(RuntimeError):
    """MAX_WAITING runs are already queued or running."""


class LabStateError(RuntimeError):
    """The run is not in a state that allows this (cancel a finished run, delete a running one)."""


# -- what to run --------------------------------------------------------------------------


def _day(value: Any, what: str) -> date:
    if not isinstance(value, str):
        raise LabError(f"{what} must be a date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise LabError(f"{what} must be a date (YYYY-MM-DD), not {value!r}") from e


def parse_request(settings: Settings, data: Any, today: date) -> tuple[str, dict[str, Any]]:
    """
    The run's kind and its spec (JSON-safe, stored with the run).

    Raises:
        LabError: If the request cannot be run, before anything is queued.
    """
    if not isinstance(data, dict):
        raise LabError("the body must be a JSON object")
    data = {k: v for k, v in data.items() if v is not None}  # null: not given
    kind = data.get("kind")
    if kind not in KINDS:
        raise LabError(f"kind must be one of: {', '.join(KINDS)}")
    allowed = {"kind", "start", "end", "data"} | ({"strategy"} if kind == VALIDATE else {"mix", "risk"})
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise LabError(f"unknown fields for a {kind}: {', '.join(unknown)}")
    start = _day(data.get("start"), "start")
    end = _day(data["end"], "end") if "end" in data else today
    if start < EARLIEST:
        raise LabError(f"start must not be before {EARLIEST}")
    if end > today:
        raise LabError("end must not be after today")
    if end <= start:
        raise LabError("end must be after start")
    source = data.get("data", sources.ALPACA)
    if source not in sources.DATA_SOURCES:
        raise LabError(f"data must be one of: {', '.join(sources.DATA_SOURCES)}")
    spec: dict[str, Any] = {"start": start.isoformat(), "end": end.isoformat(), "data": source}
    if kind == VALIDATE:
        name = data.get("strategy")
        if not isinstance(name, str) or name not in CANDIDATES:
            raise LabError(f"unknown strategy {name!r}; known: {', '.join(sorted(CANDIDATES))}")
        if end - start < MIN_VALIDATE:
            raise LabError("a validation needs at least 4 years: 3 to choose the parameters, 1 to test them")
        wanted = sources.with_tbills(CANDIDATES[name].symbols)
        spec["strategy"] = name
    else:
        try:
            profile = parse({"mix": data.get("mix"), "risk": data.get("risk") or {}})
            effective_settings(settings, profile)
        except ProfileError as e:
            raise LabError(str(e)) from e
        if end - start < MIN_BACKTEST:
            raise LabError("a backtest needs at least 30 days")
        wanted = symbols(profile)
        risk = profile.to_dict()["risk"]
        spec["profile"] = {"mix": [e.to_dict() for e in profile.mix], "risk": risk}
    if source == sources.HISTORY:
        try:
            sources.check_history_symbols(wanted)
        except sources.DataError as e:
            raise LabError(str(e)) from e
    return kind, spec


def describe_mix(mix: Sequence[Mapping[str, Any]]) -> str:
    """``60% benchmark, 40% tsmom``, from mix entries as stored (weights as decimal strings)."""
    return ", ".join(f"{float(e['weight']):.0%} {e['strategy']}" for e in mix)


def title(kind: str, spec: Mapping[str, Any]) -> str:
    what = (f"Validate {spec['strategy']}" if kind == VALIDATE
            else f"Backtest {describe_mix(spec['profile']['mix'])}")
    return f"{what} ({spec['data']}, {spec['start']} to {spec['end']})"


# -- running one --------------------------------------------------------------------------


def _finite(value: Any) -> Any:
    """JSON without NaN or infinity, which PostgreSQL's JSON and browsers refuse."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _midnight(day: str) -> datetime:
    return datetime.combine(date.fromisoformat(day), datetime.min.time(), UTC)


def execute(settings: Settings, kind: str, spec: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """
    The result (JSON-safe, with a one-line ``headline`` and ``notes`` on where the bars fall short
    of the period asked for) and, for a validation, the tear sheet. The long history ends some weeks
    before today: a run on it ends where it does.

    Raises:
        sources.DataError, TradingNotConfiguredError: When the bars cannot be had.
        ValueError: If the period is too short, or the profile no longer fits the environment's limits.
    """
    start, end = _midnight(spec["start"]), _midnight(spec["end"])
    if kind == VALIDATE:
        report, market = runner.validate_candidate(settings, spec["strategy"], start, end, data=spec["data"],
                                                   shorten=True)
        notes = market.notes(start, end)
        if market.late(start):
            notes.append("A validation chooses the parameters on the three years before each year it tests: "
                         "years without bars weaken it. Start the run where the bars start for a fair test.")
        return _finite({
            "headline": report.verdict, "passed": report.passed, "data": market.data_label,
            "hurdle": market.hurdle_label, "curve": runner.thin(report.out_of_sample_curve),
            "daily": runner.daily(report.out_of_sample_curve), "report": report.to_dict(), "notes": notes,
        }), report.to_markdown()
    profile: Profile = parse(spec["profile"])
    backtest = runner.backtest_mix(settings, profile, start, end, spec["data"], shorten=True)
    summary = backtest.summary
    headline = (f"{summary.cagr:.2%} a year, Sharpe {summary.sharpe:.2f}, worst drawdown "
                f"{summary.max_drawdown:.2%} ({backtest.market.hurdle_label} returned "
                f"{summary.hurdle_return:.2%} in all, the mix {summary.total_return:.2%})")
    benchmark = None
    if profile.mix != DEFAULT_PROFILE.mix:
        # The yardstick over the same days, on the same data, within the same limits
        plain = runner.backtest_mix(settings, replace(DEFAULT_PROFILE, risk=profile.risk), start, end,
                                    spec["data"], shorten=True, market=backtest.market)
        benchmark = {"title": STRATEGIES[DEFAULT_PROFILE.mix[0].strategy].title, **_curves(plain),
                     "summary": plain.summary.as_dict()}
    return _finite({
        "headline": headline, "summary": summary.as_dict(), "data": backtest.market.data_label,
        "hurdle": backtest.market.hurdle_label, **_curves(backtest),
        "fills": len(backtest.result.fills), "fees": str(backtest.result.final.fees),
        "starting_cash": str(backtest.result.starting_cash),
        "notes": backtest.market.notes(start, end), "benchmark": benchmark,
    }), None


def _curves(backtest: runner.MixBacktest) -> dict[str, Any]:
    curve = backtest.result.equity_curve
    return {"curve": runner.thin(curve), "daily": runner.daily(curve), "years": backtest.years(),
            "kill_switch_trips": backtest.kill_switch_trips,
            "final_equity": str(curve[-1][1]) if curve else None}


def _failure(e: Exception) -> str:
    if isinstance(e, sources.DataError | TradingNotConfiguredError | ValueError):
        return str(e)[:MAX_ERROR]
    return f"{type(e).__name__}: {e}"[:MAX_ERROR]


# -- storage ------------------------------------------------------------------------------


def _summary(record: Any) -> dict[str, Any]:
    return {
        "id": record.id, "kind": record.kind, "status": record.status, "title": title(record.kind, record.spec),
        "spec": record.spec, "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
        "started_at": record.started_at.isoformat() if record.started_at else None,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
        "headline": (record.result or {}).get("headline"), "error": record.error,
    }


def submit(app: Flask, kind: str, spec: Mapping[str, Any], created_by: str | None, now: datetime) -> dict[str, Any]:
    """
    Raises:
        LabBusyError: If MAX_WAITING runs are already queued or running.
    """
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        waiting = db.session.scalar(select(func.count()).select_from(LabRun)
                                    .where(LabRun.status.in_((QUEUED, RUNNING))))
        if (waiting or 0) >= MAX_WAITING:
            raise LabBusyError(f"{MAX_WAITING} lab runs are already waiting or running; let one finish first")
        record = LabRun(kind=kind, spec=dict(spec), status=QUEUED, created_by=created_by, created_at=now)
        db.session.add(record)
        db.session.commit()
        return _summary(record)


def runs(app: Flask, limit: int = 50) -> list[dict[str, Any]]:
    """The newest runs first, without their results."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        return [_summary(r) for r in db.session.scalars(select(LabRun).order_by(LabRun.id.desc()).limit(limit))]


def get(app: Flask, run_id: int) -> dict[str, Any] | None:
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        record = db.session.get(LabRun, run_id)
        if record is None:
            return None
        return {**_summary(record), "result": _without_daily(record.result), "report": record.report}


def _without_daily(result: Any) -> Any:
    """A result without its daily equity, which only a download needs (``equity_csv``)."""
    if not isinstance(result, dict):
        return result
    trimmed = {k: v for k, v in result.items() if k != "daily"}
    if isinstance(trimmed.get("benchmark"), dict):
        trimmed["benchmark"] = {k: v for k, v in trimmed["benchmark"].items() if k != "daily"}
    return trimmed


def equity_csv(app: Flask, run_id: int) -> tuple[str, str] | None:
    """
    A finished run's daily equity as CSV, with its file name; None for an unknown run.

    Raises:
        LabStateError: If the run has no daily equity: not finished, failed, or made before downloads.
    """
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        record = db.session.get(LabRun, run_id)
        if record is None:
            return None
        result = record.result if isinstance(record.result, dict) else {}
        if record.status != DONE or "daily" not in result:
            raise LabStateError("only a finished run has equity to download" if record.status != DONE else
                                "this run was made before downloads existed: run it again")
        kind, spec = record.kind, record.spec
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    if kind == VALIDATE:
        writer.writerow(["date", f"{spec['strategy']} out of sample"])
        writer.writerows(result["daily"])
        return f"lab-{run_id}-{spec['strategy']}-out-of-sample.csv", out.getvalue()
    benchmark = result.get("benchmark") or {}
    other = dict(benchmark.get("daily") or [])
    writer.writerow(["date", "the mix", *([benchmark["title"]] if other else [])])
    for day, equity in result["daily"]:
        writer.writerow([day, equity, *([other.get(day, "")] if other else [])])
    return f"lab-{run_id}-backtest.csv", out.getvalue()


def cancel(app: Flask, run_id: int, now: datetime) -> dict[str, Any] | None:
    """
    Cancel a queued run (it never starts) or a running one (its process stops it at its next
    heartbeat); None for an unknown run.

    Raises:
        LabStateError: If the run has already finished.
    """
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        changed = db.session.execute(
            update(LabRun).where(LabRun.id == run_id, LabRun.status.in_((QUEUED, RUNNING)))
            .values(status=CANCELLED, finished_at=now, error="cancelled"))
        db.session.commit()
        record = db.session.get(LabRun, run_id)
        if record is None:
            return None
        if getattr(changed, "rowcount", 0) != 1:
            raise LabStateError(f"run #{run_id} has already finished ({record.status})")
        return _summary(record)


def remove(app: Flask, run_id: int) -> bool:
    """
    Delete a finished run; False for an unknown run.

    Raises:
        LabStateError: If the run is queued or running (cancel it first).
    """
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        record = db.session.get(LabRun, run_id)
        if record is None:
            return False
        if record.status not in FINISHED:
            raise LabStateError(f"run #{run_id} is {record.status}: cancel it first")
        db.session.execute(delete(LabRun).where(LabRun.id == run_id, LabRun.status.in_(FINISHED)))
        db.session.commit()
        return True


def latest_verdicts(app: Flask) -> dict[str, dict[str, Any]]:
    """The newest finished validation of each strategy: what the lab last concluded about it."""
    from src.models.trading import LabRun
    from src.models.user import db

    found: dict[str, dict[str, Any]] = {}
    with app.app_context():
        for record in db.session.scalars(select(LabRun).where(LabRun.kind == VALIDATE, LabRun.status == DONE)
                                         .order_by(LabRun.id.desc())):
            name = record.spec["strategy"]
            if name not in found:
                found[name] = {"run_id": record.id, "passed": bool((record.result or {}).get("passed")),
                               "verdict": (record.result or {}).get("headline"), "data": record.spec["data"],
                               "start": record.spec["start"], "end": record.spec["end"],
                               "finished_at": record.finished_at.isoformat() if record.finished_at else None}
    return found


def expire_stale(app: Flask, now: datetime) -> int:
    """Mark failed the running runs whose process stopped sending its heartbeat."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        result = db.session.execute(
            update(LabRun).where(LabRun.status == RUNNING, LabRun.heartbeat_at < now - LAB_STALE_AFTER)
            .values(status=FAILED, finished_at=now,
                    error="the lab process stopped before the run finished (was the container restarted?)"))
        db.session.commit()
        return int(getattr(result, "rowcount", 0) or 0)


def _claim(app: Flask, now: datetime) -> tuple[int, str, dict[str, Any]] | None:
    """The oldest queued run, now running; None when nothing is queued. Two processes never claim one run."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        while True:
            record = db.session.scalars(select(LabRun).where(LabRun.status == QUEUED).order_by(LabRun.id)).first()
            if record is None:
                return None
            run_id, kind, spec = record.id, record.kind, dict(record.spec)
            claimed = db.session.execute(
                update(LabRun).where(LabRun.id == run_id, LabRun.status == QUEUED)
                .values(status=RUNNING, started_at=now, heartbeat_at=now))
            db.session.commit()
            if getattr(claimed, "rowcount", 0) == 1:
                return run_id, kind, spec


def _beat(app: Flask, run_id: int, now: datetime) -> bool:
    """Touch the run's heartbeat; False once it is no longer running (it was cancelled)."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        touched = db.session.execute(update(LabRun).where(LabRun.id == run_id, LabRun.status == RUNNING)
                                     .values(heartbeat_at=now))
        db.session.commit()
        return getattr(touched, "rowcount", 0) == 1


def _finish(app: Flask, run_id: int, now: datetime, result: dict[str, Any] | None = None,
            report: str | None = None, error: str | None = None) -> bool:
    """Store the outcome; False if the run was cancelled meanwhile (it stays cancelled, the result is dropped)."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        stored = db.session.execute(update(LabRun).where(LabRun.id == run_id, LabRun.status == RUNNING).values(
            status=FAILED if error is not None else DONE, finished_at=now, result=result, report=report,
            error=error))
        db.session.commit()
        return getattr(stored, "rowcount", 0) == 1


def has_queued(app: Flask) -> bool:
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        return db.session.scalars(select(LabRun.id).where(LabRun.status == QUEUED)).first() is not None


def running_fresh(app: Flask, now: datetime) -> bool:
    """A run executes and its process is alive: it takes the queued runs after its own."""
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        return db.session.scalars(select(LabRun.id).where(
            LabRun.status == RUNNING, LabRun.heartbeat_at >= now - LAB_STALE_AFTER)).first() is not None


class _Interrupter:
    """
    Stops a run in progress once it is cancelled. The computation is plain Python in this
    process's main thread, so the heartbeat thread interrupts it as Ctrl-C would
    (``interrupt``); only while it computes, so a result being stored is never cut short.
    """

    def __init__(self, interrupt: Callable[[], None] | None) -> None:
        self._interrupt = interrupt
        self._lock = threading.Lock()
        self._computing = False
        self.cancelled = False

    @contextmanager
    def computing(self) -> Iterator[None]:
        with self._lock:
            self._computing = True
        try:
            yield
        finally:
            with self._lock:
                self._computing = False

    def cancel(self) -> None:
        with self._lock:
            if self.cancelled:
                return
            self.cancelled = True
            if self._computing and self._interrupt is not None:
                self._interrupt()


def _heartbeat(app: Flask, run_id: int, now: Callable[[], datetime], every: float,
               on_cancel: Callable[[], None] = lambda: None) -> tuple[threading.Event, threading.Thread]:
    """Touch the run's heartbeat every ``every`` seconds until the event is set; tell ``on_cancel`` if it was cancelled."""
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(every):
            try:
                if not _beat(app, run_id, now()):
                    on_cancel()
                    return
            except Exception:
                logger.exception("Lab heartbeat failed")

    thread = threading.Thread(target=beat, name=f"lab-{run_id}-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _main_thread_interrupt() -> Callable[[], None] | None:
    """Ctrl-C for the main thread, when the runs execute there (``flask lab run``); a thread is not interrupted."""
    return _thread.interrupt_main if threading.current_thread() is threading.main_thread() else None


def drain(app: Flask, settings: Settings, alerts: AlertDispatcher | None = None,
          now: Callable[[], datetime] = lambda: datetime.now(UTC), heartbeat_every: float = HEARTBEAT_EVERY,
          interrupt: Callable[[], Callable[[], None] | None] = _main_thread_interrupt) -> int:
    """
    Run the queued runs, oldest first, until none is left; returns how many ran. A run
    cancelled while it computes is interrupted where ``interrupt`` can (the main thread),
    otherwise left to finish and its result dropped.
    """
    count = 0
    while (claimed := _claim(app, now())) is not None:
        run_id, kind, spec = claimed
        logger.info("Lab run %d started: %s", run_id, title(kind, spec))
        interrupter = _Interrupter(interrupt())
        stop, beating = _heartbeat(app, run_id, now, heartbeat_every, interrupter.cancel)
        try:
            try:
                with interrupter.computing():
                    result, report = execute(settings, kind, spec)
            except KeyboardInterrupt:
                if not interrupter.cancelled:
                    raise
                logger.info("Lab run %d cancelled", run_id)
                count += 1
                continue
        except Exception as e:
            if not isinstance(e, sources.DataError | TradingNotConfiguredError | ValueError):
                logger.exception("Lab run %d failed", run_id)
            error = _failure(e)
            stored = _finish(app, run_id, now(), error=error)
            alert = Alert(f"Lab run #{run_id} failed", f"{title(kind, spec)}: {error}", category="lab")
        else:
            stored = _finish(app, run_id, now(), result=result, report=report)
            alert = Alert(f"Lab run #{run_id} finished", f"{title(kind, spec)}: {result['headline']}",
                          category="lab")
        finally:
            stop.set()
            beating.join()
        count += 1
        if not stored:
            logger.info("Lab run %d cancelled", run_id)
            continue
        logger.info("Lab run %d: %s", run_id, alert.title)
        if alerts is not None:
            alerts.notify(alert)
    return count


# -- the process that runs them -----------------------------------------------------------


class ProcessLauncher:
    """
    Starts ``flask lab run`` unless the process it started last still runs.
    The web API has one worker (gunicorn.conf.py), so one launcher.
    """

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self._command = list(command or (sys.executable, "-m", "flask", "--app", "src.main", "lab", "run"))
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None

    def launch(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            # A fixed command, no shell; its output goes to the container's log
            self._process = subprocess.Popen(  # nosec B603
                self._command, cwd=ROOT, stdin=subprocess.DEVNULL, start_new_session=True)


def kick(app: Flask, now: datetime) -> None:
    """Fail the runs whose process died, and start a process for the queued ones when none is at work."""
    expire_stale(app, now)
    if has_queued(app) and not running_fresh(app, now):
        app.extensions["lab"].launch()
