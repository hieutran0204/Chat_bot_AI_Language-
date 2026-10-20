# name: e5f6a7b8c9d0_phase_5a_queue_and_batch_extraction.py
# description: Schema updates for Phase 5a (idle scanner, dead letters, batch extraction tracking, session start/end markers)

"""Phase 5a: queue, idle scanner, dead letters, and batch extraction tracking

Revision ID: e5f6a7b8c9d0
Revises: b2c3d4e5f6a7
Create Date: 2026-10-20 15:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. conversations updates ──────────────────────────────────────────────
    op.add_column('conversations', sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('conversations', sa.Column('insight_generated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('conversations', sa.Column('process_attempts', sa.Integer(), server_default='0', nullable=False))
    op.add_column('conversations', sa.Column('last_failed_at', sa.DateTime(timezone=True), nullable=True))

    op.create_index(
        'idx_conv_scanner',
        'conversations',
        ['last_message_at'],
        postgresql_where=sa.text('insight_generated_at IS NULL OR insight_generated_at < last_message_at'),
    )

    # ── 2. user_weakness_log updates ──────────────────────────────────────────
    op.add_column(
        'user_weakness_log',
        sa.Column('message_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        'fk_user_weakness_log_message_id',
        'user_weakness_log',
        'messages',
        ['message_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_index('ix_user_weakness_log_message_id', 'user_weakness_log', ['message_id'])

    # ── 3. session_insights updates ───────────────────────────────────────────
    # Drop legacy 1:1 conversation unique constraint
    op.drop_constraint('uq_session_insights_conversation_id', 'session_insights', type_='unique')

    op.add_column('session_insights', sa.Column('session_start_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('session_insights', sa.Column('session_end_at', sa.DateTime(timezone=True), nullable=True))

    op.create_unique_constraint(
        'uq_session_insight_conv_start',
        'session_insights',
        ['conversation_id', 'session_start_at'],
    )

    # ── 4. extraction_dead_letters table ──────────────────────────────────────
    op.create_table(
        'extraction_dead_letters',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default=sa.text("'{}'"), nullable=False),
        sa.Column('error_type', sa.String(length=50), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
    )
    op.create_index('idx_dead_letters_conv', 'extraction_dead_letters', ['conversation_id'])

    # ── 5. Backfills (Order: last_message_at -> messages -> insight_generated_at)
    op.execute("""
        UPDATE conversations c
        SET last_message_at = COALESCE(
            (SELECT MAX(m.created_at) FROM messages m WHERE m.conversation_id = c.id),
            c.created_at
        )
        WHERE c.last_message_at IS NULL;
    """)

    op.execute("""
        UPDATE messages
        SET extraction_status = 'done'
        WHERE extraction_status = 'pending'
          AND created_at < NOW() - INTERVAL '1 day';
    """)

    op.execute("""
        UPDATE conversations
        SET insight_generated_at = last_message_at
        WHERE insight_generated_at IS NULL
          AND last_message_at IS NOT NULL;
    """)

    op.execute("""
        UPDATE conversations
        SET insight_generated_at = created_at
        WHERE insight_generated_at IS NULL;
    """)


def downgrade() -> None:
    # ── 4. Drop extraction_dead_letters ───────────────────────────────────────
    op.drop_index('idx_dead_letters_conv', table_name='extraction_dead_letters')
    op.drop_table('extraction_dead_letters')

    # ── 3. Revert session_insights ────────────────────────────────────────────
    op.drop_constraint('uq_session_insight_conv_start', 'session_insights', type_='unique')
    op.drop_column('session_insights', 'session_end_at')
    op.drop_column('session_insights', 'session_start_at')
    op.create_unique_constraint(
        'uq_session_insights_conversation_id',
        'session_insights',
        ['conversation_id'],
    )

    # ── 2. Revert user_weakness_log ───────────────────────────────────────────
    op.drop_index('ix_user_weakness_log_message_id', table_name='user_weakness_log')
    op.drop_constraint('fk_user_weakness_log_message_id', 'user_weakness_log', type_='foreignkey')
    op.drop_column('user_weakness_log', 'message_id')

    # ── 1. Revert conversations ───────────────────────────────────────────────
    op.drop_index('idx_conv_scanner', table_name='conversations')
    op.drop_column('conversations', 'last_failed_at')
    op.drop_column('conversations', 'process_attempts')
    op.drop_column('conversations', 'insight_generated_at')
    op.drop_column('conversations', 'last_message_at')
