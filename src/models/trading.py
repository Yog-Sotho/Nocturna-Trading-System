"""
NOCTURNA Trading System - Trading history models.

The broker is the source of truth for current state; these tables keep the
history: every order the engine placed with its latest status, every fill,
and periodic equity snapshots for performance reporting. The one exception
is the risk gate's state, which no broker knows: its single row is restored
on restart, so a halt or a drawdown high-water mark survives it.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .user import UTCDateTime, db

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase

    class Model(DeclarativeBase):
        """db.Model as the type checker sees it: it cannot use an attribute of ``db`` as a base."""
else:
    Model = db.Model

#: Exact on PostgreSQL; wide enough for crypto quantities and share prices.
MONEY = Numeric(38, 18)


class TradingOrder(Model):
    __tablename__ = 'trading_orders'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(48), nullable=False, unique=True, index=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    qty: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    order_type: Mapped[str] = mapped_column(String(12), nullable=False)
    time_in_force: Mapped[str] = mapped_column(String(4), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(MONEY)
    stop_price: Mapped[Decimal | None] = mapped_column(MONEY)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    filled_qty: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=0)
    avg_fill_price: Mapped[Decimal | None] = mapped_column(MONEY)
    tag: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            'client_order_id': self.client_order_id,
            'broker_order_id': self.broker_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'qty': str(self.qty),
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'limit_price': str(self.limit_price) if self.limit_price is not None else None,
            'stop_price': str(self.stop_price) if self.stop_price is not None else None,
            'status': self.status,
            'filled_qty': str(self.filled_qty),
            'avg_fill_price': str(self.avg_fill_price) if self.avg_fill_price is not None else None,
            'tag': self.tag,
            'reason': self.reason,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }


class TradingFill(Model):
    __tablename__ = 'trading_fills'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fill_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    client_order_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    qty: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    fee: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=0)
    fee_qty: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=0)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            'fill_id': self.fill_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side,
            'qty': str(self.qty),
            'price': str(self.price),
            'fee': str(self.fee),
            'fee_qty': str(self.fee_qty),
            'timestamp': self.timestamp.isoformat(),
        }


class EquitySnapshot(Model):
    __tablename__ = 'equity_snapshots'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True,
                                                default=lambda: datetime.now(UTC))
    equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    cash: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    gross_exposure: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    fees: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    halted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': str(self.equity),
            'cash': str(self.cash),
            'gross_exposure': str(self.gross_exposure),
            'realized_pnl': str(self.realized_pnl),
            'fees': str(self.fees),
            'halted': self.halted,
        }


class RiskStateRecord(Model):
    """The risk gate's state (``RiskGate.state_dict``); a single row with id 1."""

    __tablename__ = 'risk_state'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class EngineStatusRecord(Model):
    """
    What the separate engine process publishes (``views.engine_views``) and
    when it last did; a single row with id 1. The API treats an old
    heartbeat as an engine that is not running.
    """

    __tablename__ = 'engine_status'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance: Mapped[str] = mapped_column(String(64), nullable=False)
    document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class EngineCommand(Model):
    """A command from the API to the separate engine process, and its result."""

    __tablename__ = 'engine_commands'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # pending, done, failed, expired
    result: Mapped[Any] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text)
    requested_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)  # None: never (emergency halts)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
