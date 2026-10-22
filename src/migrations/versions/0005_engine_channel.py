"""engine channel: status published by the engine process, commands from the API

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('engine_status',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('instance', sa.String(length=64), nullable=False),
    sa.Column('document', sa.JSON(), nullable=False),
    sa.Column('heartbeat_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('engine_commands',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('requested_by', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_engine_commands_status', 'engine_commands', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_engine_commands_status', table_name='engine_commands')
    op.drop_table('engine_commands')
    op.drop_table('engine_status')
