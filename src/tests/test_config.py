"""
Tests for startup settings validation.
"""

import pytest

from src.config import ConfigurationError, load_settings
from src.main import create_app

MANAGED_VARS = (
    "FLASK_ENV", "FLASK_DEBUG", "FLASK_SECRET_KEY", "JWT_SECRET_KEY", "JWT_EXPIRATION_HOURS",
    "DATABASE_URL", "REDIS_URL", "ALLOWED_ORIGINS", "SECURE_COOKIES",
)
STRONG_A = "a" * 64
STRONG_B = "b" * 64
GOOD_PRODUCTION = {
    "FLASK_ENV": "production",
    "FLASK_SECRET_KEY": STRONG_A,
    "JWT_SECRET_KEY": STRONG_B,
    "ALLOWED_ORIGINS": "https://app.example.com",
    "DATABASE_URL": "postgresql://nocturna:pw@db:5432/nocturna",
}


@pytest.fixture()
def env(monkeypatch):
    """Start every test from an environment without any managed variable."""
    for name in MANAGED_VARS:
        monkeypatch.delenv(name, raising=False)

    def apply(values: dict[str, str]) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    return apply


class TestDevelopmentDefaults:
    def test_no_secrets_needed_outside_production(self, env):
        settings = load_settings()
        assert settings.is_production is False
        assert settings.flask_secret_key is None
        assert settings.jwt_expiration_hours == 1
        assert settings.resolved_database_url.startswith("sqlite:///")

    def test_blank_secret_is_treated_as_unset(self, env):
        env({"FLASK_SECRET_KEY": "   "})
        assert load_settings().flask_secret_key is None

    def test_allowed_origins_are_split_and_trimmed(self, env):
        env({"ALLOWED_ORIGINS": " https://a.example , ,https://b.example "})
        assert load_settings().allowed_origin_list == ["https://a.example", "https://b.example"]


class TestMalformedValues:
    @pytest.mark.parametrize("name,value", [
        ("JWT_EXPIRATION_HOURS", "0"),
        ("JWT_EXPIRATION_HOURS", "not-a-number"),
        ("FLASK_ENV", "staging"),
    ])
    def test_rejected(self, env, name, value):
        env({name: value})
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            load_settings()


class TestProductionValidation:
    def test_valid_production_config_passes(self, env):
        env(GOOD_PRODUCTION)
        assert load_settings().is_production is True

    def test_reports_every_problem_at_once(self, env):
        env({"FLASK_ENV": "production"})
        with pytest.raises(ConfigurationError) as exc:
            load_settings()
        message = str(exc.value)
        for expected in ("FLASK_SECRET_KEY must be set", "JWT_SECRET_KEY must be set",
                         "ALLOWED_ORIGINS", "DATABASE_URL"):
            assert expected in message

    @pytest.mark.parametrize("override,expected", [
        ({"FLASK_SECRET_KEY": "short"}, "FLASK_SECRET_KEY must be at least 32 characters"),
        ({"JWT_SECRET_KEY": STRONG_A}, "JWT_SECRET_KEY must differ from FLASK_SECRET_KEY"),
        ({"FLASK_DEBUG": "true"}, "FLASK_DEBUG must be false"),
        ({"SECURE_COOKIES": "false"}, "SECURE_COOKIES must be true"),
        ({"DATABASE_URL": "sqlite:////tmp/x.db"}, "DATABASE_URL must point at a server database"),
    ])
    def test_single_problem_detected(self, env, override, expected):
        env({**GOOD_PRODUCTION, **override})
        with pytest.raises(ConfigurationError, match=expected):
            load_settings()

    def test_create_app_refuses_unsafe_production_config(self, env):
        env({"FLASK_ENV": "production"})
        with pytest.raises(ConfigurationError):
            create_app()
