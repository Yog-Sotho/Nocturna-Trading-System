"""baseline schema: users, API keys, audit log, revoked tokens

Revision ID: 0001
Revises:
Create Date: 2026-10-20 22:59:53.195445
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0001'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=50), nullable=False),
    sa.Column('email', sa.String(length=120), nullable=False),
    sa.Column('password_hash', sa.String(length=256), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('is_admin', sa.Boolean(), nullable=True),
    sa.Column('roles', sa.JSON(), nullable=True),
    sa.Column('trading_mode', sa.String(length=10), nullable=True),
    sa.Column('trading_disabled', sa.Boolean(), nullable=True),
    sa.Column('settings', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('last_login', sa.DateTime(), nullable=True),
    sa.Column('failed_login_attempts', sa.Integer(), nullable=True),
    sa.Column('locked_until', sa.DateTime(), nullable=True),
    sa.Column('token_version', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    op.create_table('api_keys',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('key_hash', sa.String(length=64), nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('permissions', sa.JSON(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('rate_limit', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=True),
    sa.Column('last_used', sa.DateTime(), nullable=True),
    sa.Column('use_count', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)

    op.create_table('audit_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('event_type', sa.String(length=50), nullable=False),
    sa.Column('success', sa.Boolean(), nullable=True),
    sa.Column('event_data', sa.JSON(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('username', sa.String(length=50), nullable=True),
    sa.Column('ip_address', sa.String(length=45), nullable=True),
    sa.Column('user_agent', sa.String(length=500), nullable=True),
    sa.Column('request_id', sa.String(length=64), nullable=True),
    sa.Column('endpoint', sa.String(length=200), nullable=True),
    sa.Column('method', sa.String(length=10), nullable=True),
    sa.Column('response_status', sa.Integer(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_logs_event_type', 'audit_logs', ['event_type'], unique=False)
    op.create_index('ix_audit_logs_request_id', 'audit_logs', ['request_id'], unique=False)
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'], unique=False)
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'], unique=False)

    op.create_table('revoked_tokens',
    sa.Column('jti', sa.String(length=64), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('token_type', sa.String(length=10), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('revoked_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('jti')
    )
    op.create_index('ix_revoked_tokens_expires_at', 'revoked_tokens', ['expires_at'], unique=False)
    op.create_index('ix_revoked_tokens_user_id', 'revoked_tokens', ['user_id'], unique=False)



def downgrade() -> None:
    op.drop_index('ix_revoked_tokens_user_id', table_name='revoked_tokens')
    op.drop_index('ix_revoked_tokens_expires_at', table_name='revoked_tokens')

    op.drop_table('revoked_tokens')
    op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_timestamp', table_name='audit_logs')
    op.drop_index('ix_audit_logs_request_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_event_type', table_name='audit_logs')

    op.drop_table('audit_logs')
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')

    op.drop_table('api_keys')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_index('ix_users_email', table_name='users')

    op.drop_table('users')
