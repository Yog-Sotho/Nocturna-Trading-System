"""
NOCTURNA Trading System - The API's response shapes, and request bodies the
routes read without a validator.

These models describe the ``data`` of each response (``src/routes/openapi.py``
publishes them in the OpenAPI document). They forbid unknown fields, so the
contract tests fail when a route returns something the description does not
say. Money and quantities are decimal strings, never floats.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DecimalStr = str
Timestamp = str
EngineState = Literal["stopped", "starting", "running", "failed", "unreachable"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- system ---------------------------------------------------------------------------


class Health(Strict):
    status: Literal["healthy"]
    timestamp: Timestamp


class Ready(Strict):
    status: Literal["ready", "not ready"]
    database: Literal["ok", "unreachable"]


class Version(Strict):
    version: str = Field(description="The application's version")
    api_version: str = Field(description="The current API version")
    versions: list[str] = Field(description="Every API version served, each under /api/<version>")
    openapi: str = Field(description="Path of the current version's OpenAPI document")
    environment: str
    timestamp: Timestamp


# -- auth and users -------------------------------------------------------------------


class TokenPair(Strict):
    access_token: str
    refresh_token: str = Field(description="Single use: a reused refresh token revokes every session")
    token_type: Literal["Bearer"]
    expires_in: int = Field(description="Seconds until the access token expires")


class User(Strict):
    id: int
    username: str
    email: str
    is_active: bool
    is_admin: bool
    roles: list[str]
    trading_mode: Literal["PAPER"]
    trading_disabled: bool
    created_at: Timestamp | None
    last_login: Timestamp | None


class UserDetail(User):
    settings: dict[str, Any] | None


class Login(TokenPair):
    user: User


class UserData(Strict):
    user: User


class UserDetailData(Strict):
    user: UserDetail


class UserPage(Strict):
    users: list[User]
    total: int
    page: int
    per_page: int
    pages: int


class RefreshRequest(Strict):
    refresh_token: str


class LogoutRequest(Strict):
    refresh_token: str | None = Field(default=None, description="Also revoke this refresh token")


class UserUpdate(Strict):
    is_active: bool | None = None
    is_admin: bool | None = None
    roles: list[Literal["reader", "trader"]] | None = Field(default=None, description="Admin rights: is_admin")
    trading_mode: Literal["PAPER"] | None = None
    trading_disabled: bool | None = None


class PasswordChange(Strict):
    current_password: str
    new_password: str = Field(min_length=12, max_length=128)


class ApiKey(Strict):
    id: int
    key_id: str
    name: str
    description: str | None
    permissions: list[Literal["read", "trade"]]
    is_active: bool
    created_at: Timestamp | None
    expires_at: Timestamp | None
    last_used: Timestamp | None
    use_count: int


class ApiKeyCreate(Strict):
    name: str = Field(default="API Key", max_length=100)
    description: str = Field(default="", max_length=500)
    permissions: list[Literal["read", "trade"]] = Field(
        default=["read"], min_length=1, description="An API key never carries admin rights")
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class NewApiKey(Strict):
    api_key: str = Field(description="The key itself: shown once, never stored")
    key: ApiKey
    message: str


class ApiKeyList(Strict):
    api_keys: list[ApiKey]


class AuditEntry(Strict):
    id: int
    timestamp: Timestamp | None
    event_type: str
    success: bool | None
    user_id: int | None
    username: str | None
    ip_address: str | None
    request_id: str | None
    endpoint: str | None
    response_status: int | None


class AuditPage(Strict):
    logs: list[AuditEntry]
    total: int
    page: int
    per_page: int
    pages: int


# -- trading --------------------------------------------------------------------------


class Risk(Strict):
    halt_kind: str | None = Field(description="drawdown, operator or another halt's kind")
    drawdown: DecimalStr = Field(description="From the high-water mark, as a fraction")
    high_water_mark: DecimalStr | None
    cooling_down: bool = Field(description="True after the daily loss limit, until the next trading day")
    orders_today: int
    turnover_today: DecimalStr
    predicted_volatility: float | None
    volatility_scale: DecimalStr | None
    drawdown_scale: DecimalStr | None


class Discrepancy(Strict):
    item: str = Field(description="A symbol, or cash")
    engine: DecimalStr | None
    broker: DecimalStr | None


class Reconciliation(Strict):
    at: Timestamp
    in_sync: bool
    discrepancies: list[Discrepancy]


class Status(Strict):
    """Fields after ``state`` are present once the engine has run in this process (or published)."""

    configured: bool = Field(description="Alpaca keys are set")
    state: EngineState = Field(description="unreachable: the engine process has not published for 30 s")
    started_at: Timestamp | None = None
    last_poll: Timestamp | None = None
    next_poll: Timestamp | None = None
    last_error: str | None = None
    as_of: Timestamp | None = None
    equity: DecimalStr | None = None
    cash: DecimalStr | None = None
    realized_pnl: DecimalStr | None = None
    fees: DecimalStr | None = None
    halted: bool | None = None
    halt_reason: str | None = None
    open_orders: int | None = None
    positions: int | None = None
    risk: Risk | None = None
    last_reconciliation: Reconciliation | None = None


class TradingHealth(Strict):
    engine: EngineState


class Position(Strict):
    symbol: str
    qty: DecimalStr
    avg_price: DecimalStr
    price: DecimalStr
    market_value: DecimalStr
    unrealized_pnl: DecimalStr


class Positions(Strict):
    as_of: Timestamp
    positions: list[Position]


class Targets(Strict):
    targets: dict[str, DecimalStr | None] = Field(description="Target weight of each symbol, as a fraction of equity")


class Order(Strict):
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: Literal["buy", "sell"]
    qty: DecimalStr
    order_type: str
    time_in_force: str
    limit_price: DecimalStr | None
    stop_price: DecimalStr | None
    status: str
    filled_qty: DecimalStr
    avg_fill_price: DecimalStr | None
    tag: str | None
    reason: str | None
    created_at: Timestamp
    updated_at: Timestamp


class Orders(Strict):
    working: list[Order] = Field(description="The engine's working orders")
    history: list[Order] = Field(description="Recorded orders, newest first")


class PlacedOrder(Strict):
    order: Order | None


class Fill(Strict):
    fill_id: str
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: DecimalStr
    price: DecimalStr
    fee: DecimalStr
    fee_qty: DecimalStr
    timestamp: Timestamp


class Fills(Strict):
    fills: list[Fill]


class EquityPoint(Strict):
    timestamp: Timestamp
    equity: DecimalStr
    cash: DecimalStr
    gross_exposure: DecimalStr
    realized_pnl: DecimalStr
    fees: DecimalStr
    halted: bool


class EquityCurve(Strict):
    equity_curve: list[EquityPoint] = Field(description="Oldest first")


class EmergencyStopResult(Strict):
    reason: str | None
    engine: Literal["halted", "not running", "not answering"]
    close_orders: list[str] = Field(description="Client order ids of the orders closing the positions")
    cancelled_orders: list[str] | None = Field(
        default=None, description="Broker order ids cancelled directly, when no engine handled the stop")


class ValidationErrors(Strict):
    validation_errors: list[dict[str, Any]] | None = None
    errors: list[dict[str, Any]] | None = None
