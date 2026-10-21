"""trading history: orders, fills, equity snapshots

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('equity_snapshots',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('equity', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('cash', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('gross_exposure', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('realized_pnl', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('fees', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('halted', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_equity_snapshots_timestamp', 'equity_snapshots', ['timestamp'], unique=False)

    op.create_table('trading_fills',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('fill_id', sa.String(length=80), nullable=False),
    sa.Column('client_order_id', sa.String(length=48), nullable=False),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('side', sa.String(length=4), nullable=False),
    sa.Column('qty', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('price', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('fee', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('fee_qty', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trading_fills_client_order_id', 'trading_fills', ['client_order_id'], unique=False)
    op.create_index('ix_trading_fills_fill_id', 'trading_fills', ['fill_id'], unique=True)
    op.create_index('ix_trading_fills_symbol', 'trading_fills', ['symbol'], unique=False)
    op.create_index('ix_trading_fills_timestamp', 'trading_fills', ['timestamp'], unique=False)

    op.create_table('trading_orders',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('client_order_id', sa.String(length=48), nullable=False),
    sa.Column('broker_order_id', sa.String(length=64), nullable=True),
    sa.Column('symbol', sa.String(length=20), nullable=False),
    sa.Column('side', sa.String(length=4), nullable=False),
    sa.Column('qty', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('order_type', sa.String(length=12), nullable=False),
    sa.Column('time_in_force', sa.String(length=4), nullable=False),
    sa.Column('limit_price', sa.Numeric(precision=38, scale=18), nullable=True),
    sa.Column('stop_price', sa.Numeric(precision=38, scale=18), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('filled_qty', sa.Numeric(precision=38, scale=18), nullable=False),
    sa.Column('avg_fill_price', sa.Numeric(precision=38, scale=18), nullable=True),
    sa.Column('tag', sa.String(length=32), nullable=True),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trading_orders_client_order_id', 'trading_orders', ['client_order_id'], unique=True)
    op.create_index('ix_trading_orders_created_at', 'trading_orders', ['created_at'], unique=False)
    op.create_index('ix_trading_orders_status', 'trading_orders', ['status'], unique=False)
    op.create_index('ix_trading_orders_symbol', 'trading_orders', ['symbol'], unique=False)



def downgrade() -> None:
    op.drop_index('ix_trading_orders_symbol', table_name='trading_orders')
    op.drop_index('ix_trading_orders_status', table_name='trading_orders')
    op.drop_index('ix_trading_orders_created_at', table_name='trading_orders')
    op.drop_index('ix_trading_orders_client_order_id', table_name='trading_orders')

    op.drop_table('trading_orders')
    op.drop_index('ix_trading_fills_timestamp', table_name='trading_fills')
    op.drop_index('ix_trading_fills_symbol', table_name='trading_fills')
    op.drop_index('ix_trading_fills_fill_id', table_name='trading_fills')
    op.drop_index('ix_trading_fills_client_order_id', table_name='trading_fills')

    op.drop_table('trading_fills')
    op.drop_index('ix_equity_snapshots_timestamp', table_name='equity_snapshots')

    op.drop_table('equity_snapshots')
