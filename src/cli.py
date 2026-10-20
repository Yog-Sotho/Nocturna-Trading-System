"""
NOCTURNA Trading System - Operator CLI commands.

Usage:
    flask --app src.main create-admin --email ops@example.com
"""

import click
from flask import Flask

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
