# name: ingest_service.py
# description: Document ingestion service — orchestrates chunking and embedding
#              using core interfaces (IChunker + IEmbedder) wired via pipeline_factory.
#              Separated from document_service.py to keep concerns distinct:
#              DocumentService handles file I/O & DB lifecycle,
#              IngestService handles the chunk/embed pipeline.

import logging
import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.factory.pipeline_factory import create_embedder
from app.adapters.chunkers.recursive_chunker import RecursiveChunker
from app.core.config import settings
from app.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)


class IngestService:
    """
    Handles the chunk-and-embed pipeline for uploaded documents.

    Responsibilities:
    1. Parse the document file into text chunks via IChunker.
    2. Batch-embed all chunks via IEmbedder.
    3. Persist DocumentChunk rows and update Document status.

    Intentionally decoupled from file I/O and DB lifecycle management
    (those remain in DocumentService), so this service can be unit-tested
    with mock adapters and reused in batch ingestion jobs.

    Args:
        db: Async SQLAlchemy session (injected per-request or per-job).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._embedder = create_embedder()  # Singleton — loaded once per process

    async def ingest_document(self, document: Document) -> None:
        """
        Parse, chunk, embed, and store a document's content in document_chunks.

        Called after the document file has been saved and the Document row
        created with status='processing'. Updates status to 'ready' on success
        or 'failed' on error.

        Flow:
        1. Parse file → TextChunks via IChunker.
        2. Batch embed all chunk texts via IEmbedder.
        3. Bulk insert DocumentChunk rows.
        4. Update Document.status + Document.chunk_count.

        Args:
            document: Document ORM instance with status='processing' and a valid file_path.
        """
        try:
            # Step 1: Parse and chunk the document file
            text_chunks = self._chunker.chunk_file(document.file_path)
            logger.info(
                "IngestService: chunked document %s → %d chunks",
                document.id,
                len(text_chunks),
            )

            if not text_chunks:
                raise ValueError(
                    "Document produced no text chunks "
                    "(possibly empty, corrupted, or unsupported format)."
                )

            # Step 2: Batch embed all chunk texts in a single call
            texts = [c.content for c in text_chunks]
            vectors = self._embedder.embed_batch(texts)

            # Step 3: Build and insert DocumentChunk rows
            db_chunks = [
                DocumentChunk(
                    document_id=document.id,
                    content=chunk.content,
                    embedding=vector,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                )
                for chunk, vector in zip(text_chunks, vectors, strict=True)
            ]
            self._db.add_all(db_chunks)

            # Step 4: Mark document as ready
            await self._db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(status="ready", chunk_count=len(text_chunks))
            )
            await self._db.commit()

            logger.info(
                "IngestService: ingestion complete — document=%s, chunks=%d",
                document.id,
                len(text_chunks),
            )

        except Exception as exc:
            logger.exception(
                "IngestService: ingestion failed for document %s: %s",
                document.id,
                exc,
            )
            await self._db.execute(
                update(Document)
                .where(Document.id == document.id)
                .values(status="failed")
            )
            await self._db.commit()
            raise

    async def re_ingest_document(self, document_id: uuid.UUID, file_path: str) -> None:
        """
        Delete existing chunks and re-run ingestion for a document.

        Useful when the embedding model changes or chunks need to be refreshed.

        Args:
            document_id: UUID of the document to re-ingest.
            file_path: Current server-side path of the document file.
        """
        from sqlalchemy import delete
        from app.models.document import DocumentChunk as DC

        # Delete existing chunks
        deleted = await self._db.execute(
            delete(DC).where(DC.document_id == document_id)
        )
        logger.info(
            "IngestService: deleted %d existing chunks for document=%s (re-ingesting)",
            deleted.rowcount,
            document_id,
        )

        # Re-run ingestion with a temporary Document-like object
        from app.models.document import Document as Doc
        from sqlalchemy import select
        result = await self._db.execute(select(Doc).where(Doc.id == document_id))
        doc = result.scalar_one_or_none()

        if doc:
            doc.status = "processing"
            await self._db.commit()
            await self.ingest_document(doc)
