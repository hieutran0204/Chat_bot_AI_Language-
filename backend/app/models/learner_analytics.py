# name: learner_analytics.py
# description: SQLAlchemy ORM models for granular learner weakness/strength tracking
#              and post-session insight cards. Phase 1 & 2 of the personalization system.

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class UserWeaknessLog(Base):
    """
    Persistent log of every correction the AI tutor issued to a learner.

    Unlike the Redis weakness counter (which only stores frequency), this table
    preserves the full context: the original utterance, AI suggestion, explanation,
    and whether the mistake has been seen before (is_repeated).

    This is the PostgreSQL source-of-truth that survives Redis restarts.

    Attributes:
        id: UUID primary key.
        user_id: FK to the owning user.
        conversation_id: FK to the conversation where the error occurred.
        weakness_type: One of the 12 detailed taxonomy types (see WeaknessType).
        original: The learner's original erroneous utterance.
        suggestion: The AI-suggested correct alternative.
        explanation: Short explanation of the rule in simple language.
        is_repeated: True if this weakness_type was seen 3+ times historically.
        created_at: UTC timestamp of when the error was logged.
    """

    __tablename__ = "user_weakness_log"
    __table_args__ = (
        Index("ix_user_weakness_log_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    weakness_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    original: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_repeated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="weakness_logs")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<UserWeaknessLog user={self.user_id} type={self.weakness_type} "
            f"repeated={self.is_repeated}>"
        )


class UserStrengthLog(Base):
    """
    Evidence log of skills the learner is performing well.

    A strength entry is created when:
    - LLM does NOT correct a grammatically complex structure (positive evidence).
    - A weakness_type shows 50%+ reduction in occurrence over 5 recent sessions.

    Attributes:
        id: UUID primary key.
        user_id: FK to the owning user.
        strength_type: Same taxonomy as weakness_type (12 types).
        evidence: The actual sentence the learner used correctly (concrete proof).
        confidence_score: Float 0.0–1.0 from LLM evaluation of correctness certainty.
        created_at: UTC timestamp.
    """

    __tablename__ = "user_strength_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    strength_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="strength_logs")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<UserStrengthLog user={self.user_id} type={self.strength_type} "
            f"score={self.confidence_score}>"
        )


class SessionInsight(Base):
    """
    AI-generated insight card summarising a completed conversation session.

    Contains personalised encouragements (what went well), reminders (what to fix),
    a recommended focus topic, and an overall session performance score.

    Generated by SessionAnalysisService after each conversation closes.

    Attributes:
        id: UUID primary key.
        user_id: FK to the owning user.
        conversation_id: FK to the analysed conversation (unique — one insight per session).
        encouragements: JSON list of positive feedback strings.
        reminders: JSON list of repeated-error warning strings.
        focus_next: weakness_type to prioritise in the next session.
        session_score: 0–100 performance score for this session.
        streak_days: Consecutive days the user has studied.
        generated_at: UTC timestamp of when the insight was produced.
    """

    __tablename__ = "session_insights"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    encouragements: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    reminders: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    focus_next: Mapped[str | None] = mapped_column(String(60), nullable=True)
    session_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    streak_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<SessionInsight conversation={self.conversation_id} score={self.session_score}>"
        )
