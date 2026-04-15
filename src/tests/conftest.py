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
