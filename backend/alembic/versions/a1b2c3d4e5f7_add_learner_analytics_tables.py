"""add learner analytics tables (weakness log, strength log, session insights)

Revision ID: a1b2c3d4e5f7
Revises: c1a2b3d4e5f6
Create Date: 2026-09-25 15:30:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'c1a2b3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── user_weakness_log ──────────────────────────────────────────────────────
    op.create_table(
        'user_weakness_log',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('weakness_type', sa.String(length=60), nullable=False),
        sa.Column('original', sa.Text(), nullable=False),
        sa.Column('suggestion', sa.Text(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('is_repeated', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_user_weakness_log_user_id', 'user_weakness_log', ['user_id'])
    op.create_index('ix_user_weakness_log_session_id', 'user_weakness_log', ['session_id'])
    op.create_index('ix_user_weakness_log_weakness_type', 'user_weakness_log', ['weakness_type'])

    # ── user_strength_log ──────────────────────────────────────────────────────
    op.create_table(
        'user_strength_log',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('strength_type', sa.String(length=60), nullable=False),
        sa.Column('evidence', sa.Text(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False, server_default='0.8'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_user_strength_log_user_id', 'user_strength_log', ['user_id'])
    op.create_index('ix_user_strength_log_strength_type', 'user_strength_log', ['strength_type'])

    # ── session_insights ───────────────────────────────────────────────────────
    op.create_table(
        'session_insights',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('encouragements', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('reminders', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('focus_next', sa.String(length=60), nullable=True),
        sa.Column('session_score', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('streak_days', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('generated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', name='uq_session_insights_session_id'),
    )
    op.create_index('ix_session_insights_user_id', 'session_insights', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_session_insights_user_id', table_name='session_insights')
    op.drop_table('session_insights')

    op.drop_index('ix_user_strength_log_strength_type', table_name='user_strength_log')
    op.drop_index('ix_user_strength_log_user_id', table_name='user_strength_log')
    op.drop_table('user_strength_log')

    op.drop_index('ix_user_weakness_log_weakness_type', table_name='user_weakness_log')
    op.drop_index('ix_user_weakness_log_session_id', table_name='user_weakness_log')
    op.drop_index('ix_user_weakness_log_user_id', table_name='user_weakness_log')
    op.drop_table('user_weakness_log')
