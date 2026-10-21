"""API keys: public key id and salted scrypt hash of the secret

Keys become ``ntr_<key_id>_<secret>``. The record is found by key_id and the
secret is verified against a per-key salted scrypt hash. Keys issued under the
previous scheme (one HMAC-SHA256 of the whole key) cannot be converted, because
only their hash was stored, so they are deleted and must be issued again.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0002'
down_revision: str | None = '0001'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(sa.text('DELETE FROM api_keys'))
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    with op.batch_alter_table('api_keys') as batch_op:
        batch_op.add_column(sa.Column('key_id', sa.String(length=16), nullable=False))
        batch_op.alter_column('key_hash', existing_type=sa.String(length=64),
                              type_=sa.String(length=255), existing_nullable=False)
    op.create_index('ix_api_keys_key_id', 'api_keys', ['key_id'], unique=True)


def downgrade() -> None:
    op.execute(sa.text('DELETE FROM api_keys'))
    op.drop_index('ix_api_keys_key_id', table_name='api_keys')
    with op.batch_alter_table('api_keys') as batch_op:
        batch_op.drop_column('key_id')
        batch_op.alter_column('key_hash', existing_type=sa.String(length=255),
                              type_=sa.String(length=64), existing_nullable=False)
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
