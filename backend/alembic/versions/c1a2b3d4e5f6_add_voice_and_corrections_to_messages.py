"""add voice and corrections to messages

Revision ID: c1a2b3d4e5f6
Revises: bd98b2e236e5
Create Date: 2026-09-12 14:40:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c1a2b3d4e5f6'
down_revision: Union[str, None] = 'bd98b2e236e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('messages', sa.Column('audio_url', sa.Text(), nullable=True))
    op.add_column('messages', sa.Column('audio_duration_ms', sa.Integer(), nullable=True))
    op.add_column('messages', sa.Column('stt_confidence', sa.Float(), nullable=True))
    op.add_column('messages', sa.Column('corrections', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('messages', 'corrections')
    op.drop_column('messages', 'stt_confidence')
    op.drop_column('messages', 'audio_duration_ms')
    op.drop_column('messages', 'audio_url')
