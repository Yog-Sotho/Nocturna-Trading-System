"""
Shared pytest fixtures for the Nocturna test suite.
"""

import os

import pytest

# Force test environment BEFORE any app imports
os.environ["FLASK_ENV"] = "testing"
os.environ["FLASK_SECRET_KEY"] = "test-secret-key-not-for-production-32chars!"
os.environ["TRADING_MODE"] = "PAPER"

from src.main import create_app
from src.middleware.auth import create_token
from src.models.user import User
from src.models.user import db as _db


@pytest.fixture(scope="session")
def app():
    """Create a Flask application configured for testing."""
    test_config = {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_ENGINE_OPTIONS": {},
    }
    application = create_app(config_override=test_config)
    yield application


@pytest.fixture(autouse=True)
def db(app):
    """Provide a clean database for each test."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture(autouse=True)
def reset_ip_manager():
    """The IP lockout list is process-global; isolate it between tests."""
    from src.middleware.security import ip_manager
    ip_manager.blacklist.clear()
    ip_manager.failed_attempts.clear()
    yield
    ip_manager.blacklist.clear()
    ip_manager.failed_attempts.clear()


@pytest.fixture(autouse=True)
def reset_rate_limits(app):
    """Rate-limit counters are shared by the session-wide app; start every test at zero."""
    app.extensions["rate_limiter"].reset()
    yield


@pytest.fixture()
def client(app, db):
    """Flask test client."""
    return app.test_client()


@pytest.fixture()
def test_user(app, db):
    """Create a test user and return (user, access_token)."""
    with app.app_context():
        user = User(
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_admin=False,
            roles=["trader"],
        )
        user.set_password("SecureP@ss123")
        db.session.add(user)
        db.session.commit()

        user_data = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "is_admin": user.is_admin,
        }
        token = create_token(str(user.id), user_data)
        return user, token


@pytest.fixture()
def admin_user(app, db):
    """Create an admin user and return (user, access_token)."""
    with app.app_context():
        user = User(
            username="adminuser",
            email="admin@example.com",
            is_active=True,
            is_admin=True,
            roles=["admin", "trader"],
        )
        user.set_password("AdminP@ss123")
        db.session.add(user)
        db.session.commit()

        user_data = {
            "user_id": user.id,
            "username": user.username,
            "roles": user.roles,
            "is_admin": user.is_admin,
        }
        token = create_token(str(user.id), user_data)
        return user, token


@pytest.fixture()
def auth_headers(test_user):
    """Return Authorization headers for a regular test user."""
    _, token = test_user
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


@pytest.fixture()
def admin_headers(admin_user):
    """Return Authorization headers for an admin user."""
    _, token = admin_user
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


@pytest.fixture()
def sim_runtime(app):
    """
    The app's trading runtime, with an engine that trades a simulated broker on a
    simulated clock (Monday 2026-03-02, 12:00 New York) and records to the test DB.
    """
    import time
    from datetime import date
    from decimal import Decimal

    from pydantic import SecretStr

    from src.tests.trading_fixtures import IEF, SPY, daily_bars, ny
    from src.trading.broker import SimulatedBroker
    from src.trading.clock import SimulatedClock
    from src.trading.data import InMemoryBarSource
    from src.trading.lineup import build_lineup
    from src.trading.persistence import EngineRecorder
    from src.trading.runtime import TradingRuntime
    from src.trading.service import EngineService

    built = {}

    def factory(flask_app, settings):
        clock = SimulatedClock(ny(2026, 3, 2, 12))
        history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 6, 30), 500.0, seed=1),
                   "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 6, 30), 95.0, seed=2)}
        broker = SimulatedBroker(clock, {"SPY": SPY, "IEF": IEF}, Decimal("100000"))
        recorder = EngineRecorder(flask_app)
        built.update(clock=clock, broker=broker)
        return EngineService(clock=clock, source=InMemoryBarSource(history, clock), broker=broker,
                             allocations=build_lineup({"SPY": SPY, "IEF": IEF}), listeners=[recorder],
                             snapshot_recorder=recorder.record_snapshot)

    settings = app.extensions["settings"].model_copy(
        update={"alpaca_api_key": SecretStr("key"), "alpaca_secret_key": SecretStr("secret")})
    runtime = TradingRuntime(app, settings, factory=factory)
    runtime.built = built

    def wait_running(timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = runtime.status()
            if status is not None and status.state in ("running", "failed"):
                return status
            time.sleep(0.01)
        raise AssertionError("engine did not start")

    runtime.wait_running = wait_running
    previous = app.extensions["trading"]
    app.extensions["trading"] = runtime
    yield runtime
    runtime.stop()
    app.extensions["trading"] = previous
