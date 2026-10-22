"""
NOCTURNA Trading System - Operator CLI commands.

Usage:
    flask --app src.main create-admin --email ops@example.com
    flask --app src.main db current|upgrade|downgrade|check|revision
    flask --app src.main validate benchmark --start 2017-01-01 --output reports/benchmark.md
    flask --app src.main engine run          # the engine as its own process (ENGINE_MODE=external)
"""

import json
import signal
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from alembic import command
from flask import Flask, current_app

from src import db_migrations
from src.models.user import create_admin_user
from src.utils.validators import validate_registration


def register_cli(app: Flask) -> None:
    """Attach operator commands to the Flask CLI."""

    @app.cli.command("create-admin")
    @click.option("--username", envvar="ADMIN_USERNAME", default="admin", show_default=True,
                  help="Admin username (env: ADMIN_USERNAME).")
    @click.option("--email", envvar="ADMIN_EMAIL", required=True,
                  help="Admin email (env: ADMIN_EMAIL).")
    @click.password_option(envvar="ADMIN_PASSWORD",
                           help="Admin password (env: ADMIN_PASSWORD); prompted when omitted.")
    def create_admin(username: str, email: str, password: str) -> None:
        """Create the single operator's admin account."""
        is_valid, validated, errors = validate_registration({
            "username": username,
            "email": email,
            "password": password,
            "password_confirm": password,
        })
        if not is_valid:
            details = "; ".join(f"{e['field'] or 'input'}: {e['message']}" for e in errors)
            raise click.ClickException(f"Invalid admin details: {details}")

        try:
            admin = create_admin_user(validated.username, validated.email, validated.password)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

        click.echo(f"Admin user {admin.username!r} created (id={admin.id}).")

    @app.cli.group("db")
    def db_group() -> None:
        """Database schema migrations (Alembic). The app also upgrades at startup."""

    @db_group.command("current")
    def db_current() -> None:
        """Show the database revision and the latest available one."""
        click.echo(f"database: {db_migrations.current_revision(current_app) or 'not migrated'}")
        click.echo(f"latest:   {db_migrations.head_revision()}")

    @db_group.command("upgrade")
    @click.argument("revision", default="head")
    def db_upgrade(revision: str) -> None:
        """Upgrade the database (default: to the latest revision)."""
        db_migrations.upgrade(current_app, revision)
        click.echo(f"Database at {db_migrations.current_revision(current_app)}.")

    @db_group.command("downgrade")
    @click.argument("revision")
    @click.confirmation_option(prompt="Downgrading can delete data. Back up first. Continue?")
    def db_downgrade(revision: str) -> None:
        """Downgrade the database to REVISION ("base" removes every table)."""
        db_migrations.downgrade(current_app, revision)
        click.echo(f"Database at {db_migrations.current_revision(current_app) or 'base'}.")

    @db_group.command("check")
    def db_check() -> None:
        """Fail if the models have changes that no migration covers."""
        differences = db_migrations.schema_differences(current_app)
        if differences:
            raise click.ClickException(
                "Models differ from the migrated schema; create a revision:\n"
                + "\n".join(f"  {d}" for d in differences)
            )
        click.echo("Models match the migrations.")

    @db_group.command("revision")
    @click.option("-m", "--message", required=True, help="Short description of the change.")
    def db_revision(message: str) -> None:
        """Autogenerate a revision from model changes (review it before committing)."""
        command.revision(db_migrations.alembic_config(), message=message, autogenerate=True)

    @app.cli.command("validate")
    @click.argument("candidate")
    @click.option("--start", type=click.DateTime(["%Y-%m-%d"]), default="2017-01-01", show_default=True,
                  help="First day traded; earlier bars warm up indicators (Alpaca equity data starts in 2016, "
                       "crypto in 2021).")
    @click.option("--end", type=click.DateTime(["%Y-%m-%d"]), default=None, help="Last day (default: today).")
    @click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
                  help="Write the Markdown tear sheet here, and the JSON report next to it.")
    @click.option("--resume-after-kill-days", type=click.IntRange(min=0), default=30, show_default=True,
                  help="Simulated operator: days after a drawdown kill before trading resumes.")
    @click.option("--train-months", type=click.IntRange(min=1), default=36, show_default=True)
    @click.option("--test-months", type=click.IntRange(min=1), default=12, show_default=True)
    @click.option("--min-sharpe", type=float, default=None,
                  help="Out-of-sample Sharpe gate (default 1.0, the portfolio-level gate).")
    def validate_candidate(candidate: str, start: datetime, end: datetime | None, output: Path | None,
                           resume_after_kill_days: int, train_months: int, test_months: int,
                           min_sharpe: float | None) -> None:
        """Validate CANDIDATE against the promotion gates on Alpaca data. Exit code 1 means rejected."""
        from src.trading.backtest import BacktestConfig
        from src.trading.runtime import TradingNotConfiguredError, risk_limits, sizing_policy
        from src.trading.validation import market
        from src.trading.validation.candidates import CANDIDATES
        from src.trading.validation.experiment import WalkForwardConfig
        from src.trading.validation.gates import PromotionGates
        from src.trading.validation.pipeline import ValidationConfig, validate

        spec = CANDIDATES.get(candidate)
        if spec is None:
            raise click.ClickException(f"unknown candidate {candidate!r}; known: {', '.join(sorted(CANDIDATES))}")
        settings = current_app.extensions["settings"]
        start = start.replace(tzinfo=UTC)
        end = (end.replace(tzinfo=UTC) if end else datetime.now(UTC)).replace(hour=0, minute=0, second=0,
                                                                              microsecond=0)
        if end <= start:
            raise click.ClickException("--end must be after --start")
        try:
            instruments, bars = market.load(settings, spec.symbols, start - timedelta(days=spec.warm_up_days), end)
        except TradingNotConfiguredError as e:
            raise click.ClickException(str(e)) from e
        gates = PromotionGates() if min_sharpe is None else PromotionGates(min_oos_sharpe=min_sharpe)
        config = ValidationConfig(
            BacktestConfig(start=start, end=end, risk_limits=risk_limits(settings), sizing=sizing_policy(settings),
                           resume_after_kill=timedelta(days=resume_after_kill_days)),
            walk_forward=WalkForwardConfig(train_months, test_months), gates=gates)
        click.echo(f"Validating {candidate} ({spec.description}) from {start:%Y-%m-%d} to {end:%Y-%m-%d}...")
        try:
            report = validate(spec.build(instruments), bars, config)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        markdown = report.to_markdown()
        if output is None:
            click.echo(markdown)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown, encoding="utf-8")
            output.with_suffix(".json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            click.echo(f"Report written to {output} and {output.with_suffix('.json')}")
        click.echo(f"Verdict: {report.verdict}")
        if not report.passed:
            raise SystemExit(1)

    @app.cli.command("openapi")
    @click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None,
                  help="Write the document to this file instead of printing it.")
    def openapi_command(output: Path | None) -> None:
        """The API's OpenAPI 3.1 document (also served at /api/v1/openapi.json)."""
        from src.routes.openapi import build_spec

        text = json.dumps(build_spec(current_app._get_current_object()), indent=2) + "\n"
        if output is None:
            click.echo(text, nl=False)
        else:
            output.write_text(text)
            click.echo(f"Wrote {output}")

    @app.cli.command("send-test-alert")
    def send_test_alert() -> None:
        """Send an urgent test alert to every configured channel and report each result."""
        from src.trading.alerts import Alert

        alerts = current_app.extensions["alerts"]
        if not alerts.enabled:
            raise click.ClickException(
                "No alert channel is configured: set ALERT_TELEGRAM_BOT_TOKEN and ALERT_TELEGRAM_CHAT_ID, "
                "ALERT_NTFY_URL or ALERT_WEBHOOK_URL")
        results = alerts.deliver_now(Alert("NOCTURNA test alert", "Alerts reach you.", urgent=True))
        for name, error in results:
            click.echo(f"{name}: " + ("sent" if error is None else f"FAILED ({error})"))
        if any(error is not None for _, error in results):
            raise SystemExit(1)

    @app.cli.group("engine")
    def engine_group() -> None:
        """The trading engine as its own process (ENGINE_MODE=external)."""

    @engine_group.command("run")
    def engine_run() -> None:
        """Run the engine until SIGTERM or SIGINT; the API reaches it through the database."""
        from src.trading.alerts import EngineMonitor, build_heartbeat
        from src.trading.channel import EngineChannel
        from src.trading.runtime import TradingRuntime
        from src.trading.worker import AnotherWorkerRunningError, EngineWorker

        settings = current_app.extensions["settings"]
        if settings.engine_mode != "external":
            raise click.ClickException(
                "Set ENGINE_MODE=external for the API and the engine process alike; in embedded mode "
                "the API runs an engine of its own, and two engines must never trade one account.")
        flask_app = current_app._get_current_object()
        alerts = flask_app.extensions["alerts"]
        monitor = EngineMonitor(alerts, build_heartbeat(settings))
        alive_file = settings.engine_alive_file

        def on_step(views) -> None:
            if alive_file is not None:
                alive_file.touch()  # the container health check reads its age
            monitor(views)

        worker = EngineWorker(TradingRuntime(flask_app, settings), EngineChannel(flask_app), on_step=on_step,
                              alerts=alerts)
        stop = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        try:
            worker.run(stop)
        except AnotherWorkerRunningError as e:
            raise click.ClickException(str(e)) from e
        finally:
            alerts.flush(timeout=15)
