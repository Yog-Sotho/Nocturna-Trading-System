"""trading profiles: the strategy mix and risk limits set on the dashboard

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0006'
down_revision: str | None = '0005'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('trading_profiles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('mix', sa.JSON(), nullable=False),
    sa.Column('risk', sa.JSON(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('created_by', sa.String(length=80), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('activated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trading_profiles_activated_at', 'trading_profiles', ['activated_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_trading_profiles_activated_at', table_name='trading_profiles')
    op.drop_table('trading_profiles')
