# name: conversation.py
# description: SQLAlchemy ORM models for conversations and individual messages.

import uuid
from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Identity, Index, String, Text, text
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


class MessageStatus(str, Enum):
    """Lifecycle status of a chat message."""

    STREAMING = "streaming"          # Currently generating tokens
    COMPLETE = "complete"            # Completed normally and persisted
    INTERRUPTED = "interrupted"      # Client disconnected before stream finished
    FAILED = "failed"                # Provider timeout or error


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
    extraction_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default=text("'pending'"),
        nullable=False,
        index=True,
    )
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default=MessageStatus.COMPLETE,
        server_default=text("'complete'"),
        nullable=False,
        index=True,
    )
    is_fallback: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )
    client_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    parent_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    seq: Mapped[int] = mapped_column(
        BigInteger,
        Identity(start=1),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('streaming', 'complete', 'interrupted', 'failed')",
            name="ck_messages_status",
        ),
        Index(
            "uq_messages_conversation_client_msg",
            "conversation_id",
            "client_message_id",
            unique=True,
            postgresql_where=text("client_message_id IS NOT NULL"),
        ),
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
    parent_message: Mapped["Message | None"] = relationship(
        "Message", remote_side=[id], backref="replies"
    )

    def __repr__(self) -> str:
        return (
            f"<Message id={self.id} role={self.role} "
            f"conv={self.conversation_id}>"
        )
