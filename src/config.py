"""
NOCTURNA Trading System - Application settings.

All configuration comes from environment variables and is validated once at
startup. In production every problem is collected and reported together, so a
misconfigured deployment fails before serving any traffic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SECRET_LENGTH = 32
DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / "database" / "nocturna.db"


class ConfigurationError(RuntimeError):
    """Raised when the environment does not form a valid configuration."""


class Settings(BaseSettings):
    """Typed view of the environment. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    flask_env: Literal["development", "testing", "production"] = "development"
    flask_debug: bool = False

    # Secrets
    flask_secret_key: SecretStr | None = None
    jwt_secret_key: SecretStr | None = None
    jwt_expiration_hours: int = Field(default=1, ge=1, le=24)
    api_key_pepper: SecretStr | None = None

    # Storage
    database_url: str | None = None
    sqlalchemy_echo: bool = False
    redis_url: str | None = None

    # HTTP / browser security
    allowed_origins: str = ""
    secure_cookies: bool = True
    session_cookie_samesite: Literal["Lax", "Strict", "None"] = "Lax"
    session_lifetime_hours: int = Field(default=24, ge=1, le=24 * 30)
    static_folder: Path | None = None

    # Paper trading engine
    paper_initial_capital: float = Field(default=100_000, gt=0)
    paper_commission_rate: float = Field(default=0.001, ge=0, le=0.05)
    paper_slippage_rate: float = Field(default=0.0005, ge=0, le=0.05)

    @field_validator("flask_secret_key", "jwt_secret_key", "api_key_pepper", "database_url", "redis_url",
                     mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat empty env vars (e.g. `FLASK_SECRET_KEY=`) as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def is_production(self) -> bool:
        return self.flask_env == "production"

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{DEFAULT_SQLITE_PATH}"

    def production_problems(self) -> list[str]:
        """Return every reason this configuration is unsafe for production."""
        problems: list[str] = []
        secrets = {
            "FLASK_SECRET_KEY": self.flask_secret_key,
            "JWT_SECRET_KEY": self.jwt_secret_key,
            "API_KEY_PEPPER": self.api_key_pepper,
        }
        for name, secret in secrets.items():
            if secret is None:
                problems.append(f"{name} must be set")
            elif len(secret.get_secret_value()) < MIN_SECRET_LENGTH:
                problems.append(f"{name} must be at least {MIN_SECRET_LENGTH} characters")
        if (
            self.flask_secret_key is not None
            and self.jwt_secret_key is not None
            and self.flask_secret_key.get_secret_value() == self.jwt_secret_key.get_secret_value()
        ):
            problems.append("JWT_SECRET_KEY must differ from FLASK_SECRET_KEY")
        if not self.allowed_origin_list:
            problems.append("ALLOWED_ORIGINS must list the allowed browser origins")
        if self.flask_debug:
            problems.append("FLASK_DEBUG must be false")
        if not self.secure_cookies:
            problems.append("SECURE_COOKIES must be true")
        if self.database_url is None or self.database_url.startswith("sqlite"):
            problems.append("DATABASE_URL must point at a server database (PostgreSQL)")
        return problems


def ensure_sqlite_directory(database_url: str) -> None:
    """
    Create the parent directory of an absolute file-based SQLite URL.

    The default database directory holds only git-ignored files, so a fresh
    checkout does not have it. Relative paths are left alone because
    Flask-SQLAlchemy resolves them against the instance folder.
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return
    path = Path(database_url[len(prefix):])
    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """
    Read and validate settings from the environment.

    Raises:
        ConfigurationError: If a value is malformed, or the configuration is
            unsafe for production while FLASK_ENV=production.
    """
    try:
        settings = Settings()
    except ValueError as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e

    if settings.is_production:
        problems = settings.production_problems()
        if problems:
            raise ConfigurationError(
                "Unsafe production configuration:\n  - " + "\n  - ".join(problems)
                + "\nGenerate secrets with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
    return settings
