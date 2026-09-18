# name: document.py
# description: SQLAlchemy ORM models for documents and their vector chunks.

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base


class Document(Base):
    """
    Represents a user-uploaded learning document.

    A document goes through async processing (parsing → chunking → embedding)
    tracked via the 'status' field.

    Attributes:
        id: UUID primary key.
        user_id: Foreign key to the owning user.
        filename: Original file name.
        file_type: Extension (pdf, docx, txt).
        file_path: Server-side storage path.
        status: Processing state — processing | ready | failed.
        chunk_count: Number of chunks generated after processing.
        created_at: UTC upload timestamp.
    """

    __tablename__ = "documents"

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
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="processing", nullable=False
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="documents")  # noqa: F821
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} status={self.status}>"


class DocumentChunk(Base):
    """
    A text chunk extracted from a Document, along with its embedding vector.

    Each chunk is stored with:
    - Raw text content
    - Embedding vector (dimension set by EMBEDDING_DIMENSION env var)
    - Metadata for source traceability (page number, section title, etc.)

    Attributes:
        id: UUID primary key.
        document_id: Foreign key to the parent document.
        content: Raw text of this chunk.
        embedding: pgvector embedding for similarity search.
        chunk_index: Positional index within the document.
        metadata: Arbitrary JSON metadata (page, section, etc.).
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension), nullable=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    # ── Indexes ───────────────────────────────────────────────────────────────
    # IVFFlat index for approximate nearest-neighbor search.
    # 'lists' parameter should be ~ sqrt(num_rows). Start with 100 for dev.
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} doc={self.document_id} "
            f"idx={self.chunk_index}>"
        )
