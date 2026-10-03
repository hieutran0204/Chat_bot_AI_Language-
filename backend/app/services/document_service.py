# name: document_service.py
# description: Business logic for document upload, async ingestion (chunking + embedding),
#              and document lifecycle management.

import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Handles the full document lifecycle:

    1. save_file        — Persist uploaded bytes to disk.
    2. create_record    — Insert Document row in DB with status='processing'.
    3. ingest           — Parse → chunk → embed → store in document_chunks.
    4. list_documents   — Fetch all documents for a user.
    5. delete_document  — Remove document + cascaded chunks from DB and disk.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._ingest_service = IngestService(db)

    # ── File I/O ──────────────────────────────────────────────────────────────

    async def save_file(
        self, file_bytes: bytes, filename: str, user_id: uuid.UUID
    ) -> Path:
        """
        Save uploaded file bytes to disk under a user-scoped directory.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename from the upload.
            user_id: ID of the uploading user.

        Returns:
            Path to the saved file on disk.

        Raises:
            ValueError: If the file extension is not allowed or size exceeds limit.
        """
        if len(file_bytes) > settings.max_file_size_bytes:
            raise ValueError(
                f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB "
                f"(max {settings.max_file_size_mb} MB)"
            )

        ext = Path(filename).suffix.lower().lstrip(".")
        if ext not in settings.allowed_ext_list:
            raise ValueError(f"File type '.{ext}' not allowed. Supported: {settings.allowed_ext_list}")

        # Store under uploads/<user_id>/<unique_name>
        user_dir = Path(settings.upload_dir) / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = user_dir / unique_name
        file_path.write_bytes(file_bytes)

        logger.info("Saved file: %s (%d bytes)", file_path, len(file_bytes))
        return file_path

    # ── DB Operations ─────────────────────────────────────────────────────────

    async def create_record(
        self, user_id: uuid.UUID, filename: str, file_type: str, file_path: Path
    ) -> Document:
        """
        Insert a new Document row with status='processing'.

        Args:
            user_id: Owning user ID.
            filename: Original filename.
            file_type: File extension (pdf, docx, txt).
            file_path: Server-side storage path.

        Returns:
            Created Document ORM instance.
        """
        doc = Document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            status="processing",
        )
        self._db.add(doc)
        await self._db.flush()  # Flush to get the generated ID
        logger.info("Created document record: id=%s filename=%s", doc.id, filename)
        return doc

    async def ingest(self, document: Document) -> None:
        """
        Parse, chunk, embed, and store a document's content.

        Delegates to IngestService (which uses RecursiveChunker and create_embedder).
        Called as an asynchronous background task after returning 202.

        Args:
            document: Document ORM instance with status='processing'.
        """
        await self._ingest_service.ingest_document(document)

    async def list_documents(self, user_id: uuid.UUID) -> list[Document]:
        """
        Fetch all documents belonging to a user.

        Args:
            user_id: User UUID to filter by.

        Returns:
            List of Document ORM instances ordered by creation date desc.
        """
        result = await self._db.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_document(
        self, document_id: uuid.UUID, user_id: uuid.UUID
    ) -> Document | None:
        """
        Fetch a single document, ensuring it belongs to the requesting user.

        Args:
            document_id: UUID of the document.
            user_id: UUID of the requesting user.

        Returns:
            Document ORM instance or None if not found / not owned by user.
        """
        result = await self._db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def delete_document(
        self, document_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        """
        Delete a document and all its chunks from DB and disk.

        Args:
            document_id: UUID of the document to delete.
            user_id: UUID of the requesting user (for ownership check).

        Returns:
            True if deleted, False if document not found.
        """
        doc = await self.get_document(document_id, user_id)
        if not doc:
            return False

        # Remove file from disk
        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted file from disk: %s", file_path)

        await self._db.delete(doc)
        await self._db.commit()
        logger.info("Deleted document %s and all its chunks", document_id)
        return True
