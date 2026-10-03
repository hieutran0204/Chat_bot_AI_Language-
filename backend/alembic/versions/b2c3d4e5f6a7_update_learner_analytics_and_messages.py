# name: b2c3d4e5f6a7_update_learner_analytics_and_messages.py
# description: Schema update for learner personalization system — renames session_id
#              to conversation_id, adds composite index on user_weakness_log,
#              and adds extraction_status & extraction_error tracking to messages.

"""update learner analytics and messages (rename session_id, composite index, extraction tracking)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f7
Create Date: 2026-09-25 22:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. user_weakness_log updates ───────────────────────────────────────────
    # Drop old session_id index
    op.drop_index('ix_user_weakness_log_session_id', table_name='user_weakness_log')

    # Rename column session_id -> conversation_id
    op.alter_column('user_weakness_log', 'session_id', new_column_name='conversation_id')

    # Create new index for conversation_id
    op.create_index('ix_user_weakness_log_conversation_id', 'user_weakness_log', ['conversation_id'])

    # Create composite index (user_id, created_at) for low-latency profile queries
    op.create_index(
        'ix_user_weakness_log_user_id_created_at',
        'user_weakness_log',
        ['user_id', 'created_at'],
    )

    # ── 2. session_insights updates ────────────────────────────────────────────
    # Drop old unique constraint
    op.drop_constraint('uq_session_insights_session_id', 'session_insights', type_='unique')

    # Rename column session_id -> conversation_id
    op.alter_column('session_insights', 'session_id', new_column_name='conversation_id')

    # Recreate unique constraint on conversation_id
    op.create_unique_constraint(
        'uq_session_insights_conversation_id',
        'session_insights',
        ['conversation_id'],
    )

    # ── 3. messages updates ───────────────────────────────────────────────────
    # Add extraction_status and extraction_error columns
    op.add_column(
        'messages',
        sa.Column('extraction_status', sa.String(length=20), server_default='pending', nullable=False),
    )
    op.add_column(
        'messages',
        sa.Column('extraction_error', sa.Text(), nullable=True),
    )
    op.create_index('ix_messages_extraction_status', 'messages', ['extraction_status'])


def downgrade() -> None:
    # ── 3. Revert messages updates ────────────────────────────────────────────
    op.drop_index('ix_messages_extraction_status', table_name='messages')
    op.drop_column('messages', 'extraction_error')
    op.drop_column('messages', 'extraction_status')

    # ── 2. Revert session_insights updates ────────────────────────────────────
    op.drop_constraint('uq_session_insights_conversation_id', 'session_insights', type_='unique')
    op.alter_column('session_insights', 'conversation_id', new_column_name='session_id')
    op.create_unique_constraint(
        'uq_session_insights_session_id',
        'session_insights',
        ['session_id'],
    )

    # ── 1. Revert user_weakness_log updates ───────────────────────────────────
    op.drop_index('ix_user_weakness_log_user_id_created_at', table_name='user_weakness_log')
    op.drop_index('ix_user_weakness_log_conversation_id', table_name='user_weakness_log')
    op.alter_column('user_weakness_log', 'conversation_id', new_column_name='session_id')
    op.create_index('ix_user_weakness_log_session_id', 'user_weakness_log', ['session_id'])
