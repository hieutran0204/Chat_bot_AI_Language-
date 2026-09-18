# name: conversation.py
# description: SQLAlchemy ORM models for conversations and individual messages.

import uuid
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import DateTime, ForeignKey, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ConversationMode(str, Enum):
    """Supported chatbot interaction modes."""

    CONVERSATION = "conversation"    # Free-form English conversation practice
    GRAMMAR = "grammar"              # Grammar explanation with RAG context
    VOCABULARY = "vocabulary"        # Vocabulary lookup and explanation
    DOCUMENT_QA = "document_qa"     # Strict RAG Q&A over uploaded documents


class MessageRole(str, Enum):
    """Valid roles for a chat message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(Base):
    """
    A single chat session between a user and the AI.

    Each conversation has a fixed mode that determines
    which RAG pipeline behavior is used.

    Attributes:
        id: UUID primary key.
        user_id: Owning user.
        mode: ConversationMode enum value.
        title: Auto-generated or user-set session title.
        created_at: UTC session start timestamp.
    """

    __tablename__ = "conversations"

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
    mode: Mapped[str] = mapped_column(
        default=ConversationMode.CONVERSATION, nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="conversations")  # noqa: F821
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} mode={self.mode}>"


class Message(Base):
    """
    A single message within a conversation.

    For document_qa mode, 'sources' contains the retrieved chunk IDs
    and text snippets used to generate the response.

    Attributes:
        id: UUID primary key.
        conversation_id: Parent conversation.
        role: 'user' or 'assistant'.
        content: Full message text.
        sources: JSON list of retrieved chunk references (doc_qa only).
        created_at: UTC message timestamp.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_duration_ms: Mapped[int | None] = mapped_column(nullable=True)
    stt_confidence: Mapped[float | None] = mapped_column(nullable=True)
    corrections: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id} role={self.role} "
            f"conv={self.conversation_id}>"
        )
