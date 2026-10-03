# name: c3d4e5f6a7b8_add_status_client_msg_id_to_messages.py
# description: Schema update for chat resilience Phase 1 — adds status, client_message_id,
#              parent_message_id, CHECK constraint and unique index to messages table.

"""add status, client_message_id, parent_message_id to messages

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-10-02 10:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add status column with default 'complete'
    op.add_column(
        'messages',
        sa.Column('status', sa.String(length=20), server_default=sa.text("'complete'"), nullable=False),
    )
    op.create_index('ix_messages_status', 'messages', ['status'])

    # 2. Add client_message_id and parent_message_id
    op.add_column(
        'messages',
        sa.Column('client_message_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index('ix_messages_client_message_id', 'messages', ['client_message_id'])

    op.add_column(
        'messages',
        sa.Column('parent_message_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        'fk_messages_parent_message_id',
        'messages',
        'messages',
        ['parent_message_id'],
        ['id'],
        ondelete='SET NULL',
    )

    op.add_column(
        'messages',
        sa.Column('seq', sa.BigInteger(), sa.Identity(start=1), nullable=False),
    )
    op.create_index('ix_messages_seq', 'messages', ['seq'])

    op.add_column(
        'messages',
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 3. Add CHECK constraint for status values
    op.create_check_constraint(
        'ck_messages_status',
        'messages',
        "status IN ('streaming', 'complete', 'interrupted', 'failed')",
    )

    # 4. Create Partial Unique Index for idempotent deduplication
    op.create_index(
        'uq_messages_conversation_client_msg',
        'messages',
        ['conversation_id', 'client_message_id'],
        unique=True,
        postgresql_where=sa.text("client_message_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index('uq_messages_conversation_client_msg', table_name='messages')
    op.drop_constraint('ck_messages_status', 'messages', type_='check')
    op.drop_constraint('fk_messages_parent_message_id', 'messages', type_='foreignkey')
    op.drop_column('messages', 'parent_message_id')
    op.drop_index('ix_messages_client_message_id', table_name='messages')
    op.drop_column('messages', 'client_message_id')
    op.drop_index('ix_messages_status', table_name='messages')
    op.drop_column('messages', 'status')
