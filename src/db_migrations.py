"""
NOCTURNA Trading System - Database schema migrations (Alembic).

The schema is owned by the revisions in ``src/migrations/versions``. The app
upgrades the database to the latest revision at startup; the ``flask db``
commands expose the same operations to the operator.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from flask import Flask

from src.config import ConfigurationError
from src.models.user import db

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config() -> Config:
    """Alembic configuration pointing at this package's migration scripts."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    return config


def head_revision() -> str | None:
    """The newest revision in the migration scripts."""
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(app: Flask) -> str | None:
    """The revision the database is at, or None if it was never migrated."""
    with app.app_context(), db.engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def _assert_not_legacy(app: Flask) -> None:
    """Refuse databases created by ``db.create_all()`` before migrations existed."""
    with app.app_context():
        tables = set(sa.inspect(db.engine).get_table_names())
    if "users" in tables and "alembic_version" not in tables:
        raise ConfigurationError(
            "The database was created before schema migrations existed and cannot be "
            "upgraded in place. Back it up, then delete it (SQLite) or drop its tables "
            "(PostgreSQL), restart, and recreate the operator with `flask create-admin`."
        )


def upgrade(app: Flask, revision: str = "head") -> None:
    """Upgrade the database to ``revision`` (default: the latest)."""
    _assert_not_legacy(app)
    with app.app_context():
        command.upgrade(alembic_config(), revision)


def downgrade(app: Flask, revision: str) -> None:
    """Downgrade the database to ``revision`` (``base`` removes every table)."""
    with app.app_context():
        command.downgrade(alembic_config(), revision)


def schema_differences(app: Flask) -> list:
    """Differences between the models and the migrated database (empty when in sync)."""
    from alembic.autogenerate import compare_metadata

    with app.app_context(), db.engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        return compare_metadata(context, db.metadata)
