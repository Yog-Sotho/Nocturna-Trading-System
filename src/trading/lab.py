"""
NOCTURNA Trading System - The strategy lab: backtests and validations on demand.

The dashboard queues a run; ``flask lab run``, a process of its own that the
web API starts (``ProcessLauncher``), executes the queued runs one at a time
and stores each result, so a computation of minutes never holds up the web
server, and never the engine. A run is one of:

* ``backtest``: a strategy mix with its risk limits, as the engine would trade
  it (``runner.backtest_mix``): the equity curve, the return of each year, and
  a summary against T-bills;
* ``validate``: the promotion gates for one strategy and its parameter grid,
  as ``flask validate`` runs them (``runner.validate_candidate``): the verdict,
  the tear sheet, and the equity the walk-forward traded out of sample.

Nothing here trades or changes the trading profile: results are for reading.
A run whose process died (a container restart) stops sending its heartbeat
and is marked failed.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Flask
from sqlalchemy import func, select, update

from src.config import Settings

from .alerts import Alert, AlertDispatcher
from .profile import Profile, ProfileError, effective_settings, parse, symbols
from .runtime import TradingNotConfiguredError
from .validation import runner, sources
from .validation.candidates import CANDIDATES

logger = logging.getLogger(__name__)

BACKTEST, VALIDATE = "backtest", "validate"
KINDS = (BACKTEST, VALIDATE)
QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
#: Runs queued or running at once; more are refused until one finishes
MAX_WAITING = 3
#: The long history starts in 1970 (``history.py``)
EARLIEST = date(1970, 1, 1)
MIN_BACKTEST = timedelta(days=30)
#: The walk-forward's default: 36 months to choose the parameters, then at least 12 to test them
MIN_VALIDATE = timedelta(days=4 * 365)
HEARTBEAT_EVERY = 30.0
#: A running run without a heartbeat this long lost its process
LAB_STALE_AFTER = timedelta(minutes=5)
MAX_ERROR = 2000
#: The project root, where ``flask --app src.main`` finds the app
ROOT = Path(__file__).resolve().parents[2]


class LabError(ValueError):
    """The request cannot be run; the message says why."""


class LabBusyError(RuntimeError):
    """MAX_WAITING runs are already queued or running."""


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
    The result (JSON-safe, with a one-line ``headline``) and, for a validation, the tear sheet.

    Raises:
        sources.DataError, TradingNotConfiguredError: When the bars cannot be had.
        ValueError: If the period is too short, or the profile no longer fits the environment's limits.
    """
    start, end = _midnight(spec["start"]), _midnight(spec["end"])
    if kind == VALIDATE:
        report, market = runner.validate_candidate(settings, spec["strategy"], start, end, data=spec["data"])
        return _finite({
            "headline": report.verdict, "passed": report.passed, "data": market.data_label,
            "hurdle": market.hurdle_label, "curve": runner.thin(report.out_of_sample_curve),
            "report": report.to_dict(),
        }), report.to_markdown()
    profile: Profile = parse(spec["profile"])
    backtest = runner.backtest_mix(settings, profile, start, end, spec["data"])
    summary = backtest.summary
    headline = (f"{summary.cagr:.2%} a year, Sharpe {summary.sharpe:.2f}, worst drawdown "
                f"{summary.max_drawdown:.2%} ({backtest.market.hurdle_label} returned "
                f"{summary.hurdle_return:.2%} in all, the mix {summary.total_return:.2%})")
    return _finite({
        "headline": headline, "summary": summary.as_dict(), "data": backtest.market.data_label,
        "hurdle": backtest.market.hurdle_label, "curve": runner.thin(backtest.result.equity_curve),
        "years": backtest.years(), "kill_switch_trips": backtest.kill_switch_trips,
        "fills": len(backtest.result.fills), "fees": str(backtest.result.final.fees),
        "final_equity": str(backtest.result.equity_curve[-1][1]) if backtest.result.equity_curve else None,
        "starting_cash": str(backtest.result.starting_cash),
    }), None


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
        return {**_summary(record), "result": record.result, "report": record.report}


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


def _beat(app: Flask, run_id: int, now: datetime) -> None:
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        db.session.execute(update(LabRun).where(LabRun.id == run_id, LabRun.status == RUNNING)
                           .values(heartbeat_at=now))
        db.session.commit()


def _finish(app: Flask, run_id: int, now: datetime, result: dict[str, Any] | None = None,
            report: str | None = None, error: str | None = None) -> None:
    from src.models.trading import LabRun
    from src.models.user import db

    with app.app_context():
        record = db.session.get(LabRun, run_id)
        if record is None:
            return
        record.status, record.finished_at = (FAILED if error is not None else DONE), now
        record.result, record.report, record.error = result, report, error
        db.session.commit()


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


def _heartbeat(app: Flask, run_id: int, now: Callable[[], datetime],
               every: float) -> tuple[threading.Event, threading.Thread]:
    """Touch the run's heartbeat every ``every`` seconds until the event is set."""
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(every):
            try:
                _beat(app, run_id, now())
            except Exception:
                logger.exception("Lab heartbeat failed")

    thread = threading.Thread(target=beat, name=f"lab-{run_id}-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def drain(app: Flask, settings: Settings, alerts: AlertDispatcher | None = None,
          now: Callable[[], datetime] = lambda: datetime.now(UTC), heartbeat_every: float = HEARTBEAT_EVERY) -> int:
    """Run the queued runs, oldest first, until none is left; returns how many ran."""
    count = 0
    while (claimed := _claim(app, now())) is not None:
        run_id, kind, spec = claimed
        logger.info("Lab run %d started: %s", run_id, title(kind, spec))
        stop, beating = _heartbeat(app, run_id, now, heartbeat_every)
        try:
            result, report = execute(settings, kind, spec)
        except Exception as e:
            if not isinstance(e, sources.DataError | TradingNotConfiguredError | ValueError):
                logger.exception("Lab run %d failed", run_id)
            error = _failure(e)
            _finish(app, run_id, now(), error=error)
            alert = Alert(f"Lab run #{run_id} failed", f"{title(kind, spec)}: {error}", category="lab")
        else:
            _finish(app, run_id, now(), result=result, report=report)
            alert = Alert(f"Lab run #{run_id} finished", f"{title(kind, spec)}: {result['headline']}",
                          category="lab")
        finally:
            stop.set()
            beating.join()
        logger.info("Lab run %d: %s", run_id, alert.title)
        if alerts is not None:
            alerts.notify(alert)
        count += 1
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
