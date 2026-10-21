"""
Tests for schema migrations: the revisions build exactly the models' schema,
on SQLite and (when TEST_POSTGRES_URL is set) on PostgreSQL.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from flask import Flask

from src import db_migrations
from src.cli import register_cli
from src.config import ConfigurationError
from src.models.user import db

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
APP_TABLES = {"users", "api_keys", "audit_logs", "revoked_tokens"}


def _tables(app):
    with app.app_context():
        return set(sa.inspect(db.engine).get_table_names())


@pytest.fixture(params=["sqlite", "postgresql"])
def migration_app(request, tmp_path):
    """A bare app on an empty database, so migrations run on a clean schema."""
    if request.param == "postgresql":
        if not POSTGRES_URL:
            pytest.skip("TEST_POSTGRES_URL not set")
        url = POSTGRES_URL
    else:
        url = f"sqlite:///{tmp_path / 'migrations.db'}"

    app = Flask(f"migration-test-{request.param}")
    app.config["SQLALCHEMY_DATABASE_URI"] = url
    db.init_app(app)
    register_cli(app)
    yield app

    with app.app_context():
        db.session.remove()
        if db_migrations.current_revision(app):
            db_migrations.downgrade(app, "base")
        with db.engine.begin() as connection:
            connection.execute(sa.text("DROP TABLE IF EXISTS alembic_version"))
        db.engine.dispose()


class TestRevisions:
    def test_upgrade_builds_the_models_schema(self, migration_app):
        db_migrations.upgrade(migration_app)
        assert db_migrations.current_revision(migration_app) == db_migrations.head_revision()
        assert _tables(migration_app) >= APP_TABLES
        assert db_migrations.schema_differences(migration_app) == []

    def test_upgrade_is_idempotent(self, migration_app):
        db_migrations.upgrade(migration_app)
        db_migrations.upgrade(migration_app)
        assert db_migrations.current_revision(migration_app) == db_migrations.head_revision()

    def test_downgrade_to_base_and_back(self, migration_app):
        db_migrations.upgrade(migration_app)
        db_migrations.downgrade(migration_app, "base")
        assert _tables(migration_app) & APP_TABLES == set()
        db_migrations.upgrade(migration_app)
        assert _tables(migration_app) >= APP_TABLES

    def test_0002_removes_keys_issued_under_the_old_scheme(self, migration_app):
        db_migrations.upgrade(migration_app, "0001")
        with migration_app.app_context(), db.engine.begin() as connection:
            connection.execute(sa.text(
                "INSERT INTO users (username, email, password_hash, token_version) "
                "VALUES ('u', 'u@example.com', 'x', 0)"))
            connection.execute(sa.text(
                "INSERT INTO api_keys (user_id, key_hash, name) "
                "SELECT id, :hash, 'old' FROM users"), {"hash": "a" * 64})
        db_migrations.upgrade(migration_app)
        with migration_app.app_context(), db.engine.connect() as connection:
            assert connection.execute(sa.text("SELECT COUNT(*) FROM api_keys")).scalar() == 0
            assert connection.execute(sa.text("SELECT COUNT(*) FROM users")).scalar() == 1
            columns = {c["name"] for c in sa.inspect(connection).get_columns("api_keys")}
        assert "key_id" in columns
        assert db_migrations.schema_differences(migration_app) == []

    def test_pre_migration_database_is_refused(self, migration_app):
        with migration_app.app_context():
            db.create_all()
        with pytest.raises(ConfigurationError, match="before schema migrations existed"):
            db_migrations.upgrade(migration_app)
        with migration_app.app_context():
            db.drop_all()

    def test_single_head(self):
        from alembic.script import ScriptDirectory
        heads = ScriptDirectory.from_config(db_migrations.alembic_config()).get_heads()
        assert len(heads) == 1


class TestCommands:
    @staticmethod
    def _invoke(app, *args, input=None):
        # The suite's autouse fixtures keep another app's context active; the
        # commands act on current_app, so make it this one
        with app.app_context():
            return app.test_cli_runner().invoke(args=list(args), input=input)

    def test_upgrade_current_and_check(self, migration_app):
        assert "database: not migrated" in self._invoke(migration_app, "db", "current").output

        result = self._invoke(migration_app, "db", "upgrade")
        assert result.exit_code == 0, result.output

        result = self._invoke(migration_app, "db", "check")
        assert result.exit_code == 0, result.output
        assert "Models match the migrations." in result.output

    def test_downgrade_requires_confirmation(self, migration_app):
        self._invoke(migration_app, "db", "upgrade")
        result = self._invoke(migration_app, "db", "downgrade", "base", input="n\n")
        assert result.exit_code != 0
        assert _tables(migration_app) >= APP_TABLES


def test_create_app_migrates_a_fresh_database(tmp_path):
    """Run in a subprocess: create_app() rebinds process-wide engine singletons."""
    db_path = tmp_path / "fresh.db"
    env = {**os.environ, "FLASK_ENV": "development", "TRADING_MODE": "PAPER",
           "DATABASE_URL": f"sqlite:///{db_path}"}
    env.pop("REDIS_URL", None)
    code = (
        "from src.main import create_app\n"
        "from src import db_migrations\n"
        "app = create_app()\n"
        "assert db_migrations.current_revision(app) == db_migrations.head_revision()\n"
    )
    # create_app() must leave no pooled connection behind for a forked worker to inherit
    code = code.replace("app = create_app()\n", "app = create_app()\n"
                        "from src.models.user import db\n"
                        "with app.app_context():\n"
                        "    assert db.engine.pool.checkedin() == 0, db.engine.pool.status()\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert db_path.exists()


def test_gunicorn_runs_one_worker_that_is_never_recycled():
    import runpy
    config = runpy.run_path(str(REPO_ROOT / "gunicorn.conf.py"))
    assert config["workers"] == 1
    assert config["max_requests"] == 0
