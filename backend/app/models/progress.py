# name: progress.py
# description: SQLAlchemy ORM model for daily user learning progress tracking.

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class UserProgress(Base):
    """
    Tracks a user's learning activity on a per-day basis.

    One row per (user_id, date) pair — enforced by a unique constraint.
    The service layer uses INSERT ... ON CONFLICT DO UPDATE (upsert) to
    increment counters safely.

    Attributes:
        id: UUID primary key.
        user_id: Foreign key to the user.
        date: The calendar day this record covers.
        messages_sent: Number of chat messages the user sent that day.
        vocab_learned: Vocabulary items explained (mode=vocabulary).
        study_minutes: Estimated study time in minutes.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "user_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_user_progress_user_date"),
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
    date: Mapped[date] = mapped_column(
        Date, default=lambda: datetime.now(UTC).date(), nullable=False
    )
    messages_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    vocab_learned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    study_minutes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="progress")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<UserProgress user={self.user_id} date={self.date} "
            f"msgs={self.messages_sent}>"
        )
