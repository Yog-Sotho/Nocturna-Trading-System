"""
Tests for startup settings validation.
"""

import pytest

from src.config import ConfigurationError, ensure_sqlite_directory, load_settings
from src.main import create_app

MANAGED_VARS = (
    "FLASK_ENV", "FLASK_DEBUG", "FLASK_SECRET_KEY", "JWT_SECRET_KEY", "JWT_EXPIRATION_HOURS", "API_KEY_PEPPER",
    "DATABASE_URL", "REDIS_URL", "ALLOWED_ORIGINS", "SECURE_COOKIES", "ALERT_TELEGRAM_BOT_TOKEN",
    "ALERT_TELEGRAM_CHAT_ID", "ALERT_NTFY_URL", "ALERT_NTFY_TOKEN", "ALERT_WEBHOOK_URL", "HEARTBEAT_URL",
    "HEARTBEAT_INTERVAL_SECONDS", "METRICS_TOKEN", "ENGINE_ALIVE_FILE",
)
STRONG_A = "a" * 64
STRONG_B = "b" * 64
GOOD_PRODUCTION = {
    "FLASK_ENV": "production",
    "FLASK_SECRET_KEY": STRONG_A,
    "JWT_SECRET_KEY": STRONG_B,
    "API_KEY_PEPPER": "c" * 64,
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


class TestAlertsAndMonitoring:
    def test_unset_by_default_and_when_blank(self, env):
        env({"ALERT_NTFY_URL": " ", "HEARTBEAT_URL": "", "METRICS_TOKEN": ""})
        settings = load_settings()
        assert settings.alert_ntfy_url is None and settings.heartbeat_url is None and settings.metrics_token is None
        assert settings.heartbeat_interval_seconds == 60

    def test_telegram_needs_token_and_chat(self, env):
        env({"ALERT_TELEGRAM_BOT_TOKEN": "123:abc"})
        with pytest.raises(ConfigurationError, match="ALERT_TELEGRAM_CHAT_ID"):
            load_settings()
        env({"ALERT_TELEGRAM_CHAT_ID": "42"})
        assert load_settings().alert_telegram_chat_id == "42"

    @pytest.mark.parametrize("name", ["ALERT_NTFY_URL", "ALERT_WEBHOOK_URL", "HEARTBEAT_URL"])
    def test_urls_must_be_http(self, env, name):
        env({name: "file:///etc/passwd"})
        with pytest.raises(ConfigurationError, match="must be an http"):
            load_settings()

    def test_heartbeat_respects_the_ping_rate_limit(self, env):
        env({"HEARTBEAT_INTERVAL_SECONDS": "5"})
        with pytest.raises(ConfigurationError):
            load_settings()


class TestDatabaseUrl:
    @pytest.mark.parametrize("url,expected", [
        ("postgresql://u:p@db:5432/n", "postgresql+psycopg://u:p@db:5432/n"),
        ("postgres://u:p@db/n", "postgresql+psycopg://u:p@db/n"),
        ("postgresql+psycopg://u@db/n", "postgresql+psycopg://u@db/n"),
        ("sqlite:////tmp/x.db", "sqlite:////tmp/x.db"),
    ])
    def test_postgres_urls_use_psycopg3(self, env, url, expected):
        env({"DATABASE_URL": url})
        assert load_settings().resolved_database_url == expected


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
        for expected in ("FLASK_SECRET_KEY must be set", "JWT_SECRET_KEY must be set", "API_KEY_PEPPER must be set",
                         "ALLOWED_ORIGINS", "DATABASE_URL"):
            assert expected in message

    @pytest.mark.parametrize("override,expected", [
        ({"FLASK_SECRET_KEY": "short"}, "FLASK_SECRET_KEY must be at least 32 characters"),
        ({"JWT_SECRET_KEY": STRONG_A}, "JWT_SECRET_KEY must differ from FLASK_SECRET_KEY"),
        ({"FLASK_DEBUG": "true"}, "FLASK_DEBUG must be false"),
        ({"SECURE_COOKIES": "false"}, "SECURE_COOKIES must be true"),
        ({"DATABASE_URL": "sqlite:////tmp/x.db"}, "DATABASE_URL must point at a server database"),
        ({"METRICS_TOKEN": "short"}, "METRICS_TOKEN must be at least 32 characters"),
    ])
    def test_single_problem_detected(self, env, override, expected):
        env({**GOOD_PRODUCTION, **override})
        with pytest.raises(ConfigurationError, match=expected):
            load_settings()

    def test_create_app_refuses_unsafe_production_config(self, env):
        env({"FLASK_ENV": "production"})
        with pytest.raises(ConfigurationError):
            create_app()



class TestEnsureSqliteDirectory:
    def test_creates_missing_parent_of_absolute_path(self, tmp_path):
        db_path = tmp_path / "missing" / "nocturna.db"
        ensure_sqlite_directory(f"sqlite:///{db_path}")
        assert db_path.parent.is_dir()

    def test_default_url_points_into_created_directory(self, env, tmp_path, monkeypatch):
        import src.config
        monkeypatch.setattr(src.config, "DEFAULT_SQLITE_PATH", tmp_path / "db" / "nocturna.db")
        ensure_sqlite_directory(load_settings().resolved_database_url)
        assert (tmp_path / "db").is_dir()

    @pytest.mark.parametrize("url", [
        "sqlite:///:memory:", "sqlite:///relative.db", "postgresql://u:p@db/x",
    ])
    def test_leaves_other_urls_alone(self, tmp_path, monkeypatch, url):
        monkeypatch.chdir(tmp_path)
        ensure_sqlite_directory(url)
        assert list(tmp_path.iterdir()) == []
