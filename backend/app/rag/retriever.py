# name: retriever.py
# description: pgvector similarity search for relevant document chunks.

import logging
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import DocumentChunk
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """
    A chunk returned from a similarity search, with its score.

    Attributes:
        chunk_id: UUID of the DocumentChunk row.
        document_id: UUID of the parent Document.
        content: Raw text content of the chunk.
        similarity: Cosine similarity score (0–1, higher is better).
        metadata: Source metadata (page, filename, etc.).
    """

    __slots__ = ("chunk_id", "document_id", "content", "similarity", "metadata")

    def __init__(
        self,
        chunk_id: uuid.UUID,
        document_id: uuid.UUID,
        content: str,
        similarity: float,
        metadata: dict,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.similarity = similarity
        self.metadata = metadata

    def to_dict(self) -> dict:
        """Serialize to dict for JSON sources field in Message."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content[:300],  # Truncate for storage
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }


class VectorRetriever:
    """
    Retrieves the most semantically similar document chunks using pgvector.

    Embeds the query, then runs an ANN search against the IVFFlat index
    on document_chunks, optionally filtered by user_id for data isolation.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._embedder = get_embedder()

    async def retrieve(
        self,
        query: str,
        user_id: uuid.UUID,
        top_k: int = settings.retrieval_top_k,
        threshold: float = settings.similarity_threshold,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the top-k chunks most similar to the query.

        Uses cosine similarity (<=> operator in pgvector).
        Filters by user ownership via a JOIN on documents table,
        and optionally restricts to specific document_ids.

        Args:
            query: User's question or search text.
            user_id: ID of the requesting user (for data isolation).
            top_k: Maximum number of chunks to return.
            threshold: Minimum similarity score to include a chunk.
            document_ids: Optional list of document UUIDs to restrict search to.

        Returns:
            List of RetrievedChunk ordered by similarity descending.
        """
        query_vector = self._embedder.embed(query)
        query_vec_str = f"[{','.join(map(str, query_vector))}]"

        # Build raw SQL for pgvector cosine distance with user filtering.
        # We use 1 - cosine_distance = cosine_similarity.
        base_sql = """
            SELECT
                dc.id            AS chunk_id,
                dc.document_id   AS document_id,
                dc.content       AS content,
                dc.metadata      AS metadata,
                1 - (dc.embedding <=> :query_vec::vector) AS similarity
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.user_id = :user_id
              AND d.status = 'ready'
              AND 1 - (dc.embedding <=> :query_vec::vector) >= :threshold
        """
        params: dict = {
            "query_vec": query_vec_str,
            "user_id": str(user_id),
            "threshold": threshold,
        }

        if document_ids:
            doc_id_strs = [str(did) for did in document_ids]
            base_sql += " AND dc.document_id = ANY(:doc_ids)"
            params["doc_ids"] = doc_id_strs

        base_sql += " ORDER BY similarity DESC LIMIT :top_k"
        params["top_k"] = top_k

        result = await self._db.execute(text(base_sql), params)
        rows = result.fetchall()

        chunks = [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                similarity=float(row.similarity),
                metadata=row.metadata or {},
            )
            for row in rows
        ]

        logger.info(
            "Retrieved %d chunks for query=%r (threshold=%.2f, top_k=%d)",
            len(chunks),
            query[:60],
            threshold,
            top_k,
        )
        return chunks
