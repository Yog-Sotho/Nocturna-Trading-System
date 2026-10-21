"""Alembic environment. Runs inside the Flask app context (see src/db_migrations.py)."""

from alembic import context
from flask import current_app
from src.models.user import UTCDateTime, db

target_metadata = db.metadata


def render_item(type_, obj, autogen_context):
    """Render UTCDateTime as its storage type so revisions never import model code."""
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime()"
    return False


def run_migrations_offline() -> None:
    """Emit SQL for the configured database URL without connecting."""
    context.configure(
        url=current_app.config["SQLALCHEMY_DATABASE_URI"],
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations over the app's own engine."""
    with db.engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
            # SQLite cannot ALTER most constraints; batch mode rebuilds the table
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
