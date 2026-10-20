"""
NOCTURNA Trading System - Operator CLI commands.

Usage:
    flask --app src.main create-admin --email ops@example.com
    flask --app src.main db current|upgrade|downgrade|check|revision
"""

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
