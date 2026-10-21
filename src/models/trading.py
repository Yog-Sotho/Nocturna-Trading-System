"""
NOCTURNA Trading System - Trading history models.

The broker is the source of truth for current state; these tables keep the
history: every order the engine placed with its latest status, every fill,
and periodic equity snapshots for performance reporting. The one exception
is the risk gate's state, which no broker knows: its single row is restored
on restart, so a halt or a drawdown high-water mark survives it.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, Integer, Numeric, String, Text

from .user import UTCDateTime, db

#: Exact on PostgreSQL; wide enough for crypto quantities and share prices.
MONEY = Numeric(38, 18)


class TradingOrder(db.Model):
    __tablename__ = 'trading_orders'

    id = Column(Integer, primary_key=True)
    client_order_id = Column(String(48), nullable=False, unique=True, index=True)
    broker_order_id = Column(String(64))
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)
    qty = Column(MONEY, nullable=False)
    order_type = Column(String(12), nullable=False)
    time_in_force = Column(String(4), nullable=False)
    limit_price = Column(MONEY)
    stop_price = Column(MONEY)
    status = Column(String(20), nullable=False, index=True)
    filled_qty = Column(MONEY, nullable=False, default=0)
    avg_fill_price = Column(MONEY)
    tag = Column(String(32))
    reason = Column(Text)
    created_at = Column(UTCDateTime, nullable=False, index=True)
    updated_at = Column(UTCDateTime, nullable=False)

    def to_dict(self) -> dict:
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


class TradingFill(db.Model):
    __tablename__ = 'trading_fills'

    id = Column(Integer, primary_key=True)
    fill_id = Column(String(80), nullable=False, unique=True, index=True)
    client_order_id = Column(String(48), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(4), nullable=False)
    qty = Column(MONEY, nullable=False)
    price = Column(MONEY, nullable=False)
    fee = Column(MONEY, nullable=False, default=0)
    fee_qty = Column(MONEY, nullable=False, default=0)
    timestamp = Column(UTCDateTime, nullable=False, index=True)

    def to_dict(self) -> dict:
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


class EquitySnapshot(db.Model):
    __tablename__ = 'equity_snapshots'

    id = Column(Integer, primary_key=True)
    timestamp = Column(UTCDateTime, nullable=False, index=True, default=lambda: datetime.now(UTC))
    equity = Column(MONEY, nullable=False)
    cash = Column(MONEY, nullable=False)
    gross_exposure = Column(MONEY, nullable=False)
    realized_pnl = Column(MONEY, nullable=False)
    fees = Column(MONEY, nullable=False)
    halted = Column(Boolean, nullable=False, default=False)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'equity': str(self.equity),
            'cash': str(self.cash),
            'gross_exposure': str(self.gross_exposure),
            'realized_pnl': str(self.realized_pnl),
            'fees': str(self.fees),
            'halted': self.halted,
        }


class RiskStateRecord(db.Model):
    """The risk gate's state (``RiskGate.state_dict``); a single row with id 1."""

    __tablename__ = 'risk_state'

    id = Column(Integer, primary_key=True)
    state = Column(JSON, nullable=False)
    updated_at = Column(UTCDateTime, nullable=False)
