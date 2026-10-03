# name: pgvector_store.py
# description: IVectorStore adapter using PostgreSQL + pgvector extension.
#              Handles cosine similarity search, chunk upsert, and chunk deletion
#              with strict user-level data isolation.

from __future__ import annotations

import logging
import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.interfaces.vector_store import IVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


class PgVectorStore(IVectorStore):
    """
    Vector store adapter backed by PostgreSQL + pgvector.

    Uses the IVFFlat index on the `document_chunks.embedding` column
    for approximate nearest-neighbor (ANN) search.

    Data isolation: every query is scoped to `user_id` via a JOIN
    on the `documents` table, preventing cross-user chunk leakage.

    Args:
        db: Async SQLAlchemy session (injected per-request).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def similarity_search(
        self,
        query_vector: list[float],
        user_id: uuid.UUID,
        top_k: int = 5,
        threshold: float = 0.5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the top-k chunks most similar to the query vector.

        Uses cosine similarity (1 - cosine_distance) via pgvector's <=> operator.
        Filters by user ownership and optionally by specific document IDs.

        Args:
            query_vector: Pre-computed embedding of the query text.
            user_id: Requesting user ID for data isolation.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0–1) to include a chunk.
            document_ids: Optional list of document UUIDs to restrict search to.

        Returns:
            List of RetrievedChunk ordered by similarity descending.
        """
        query_vec_str = f"[{','.join(map(str, query_vector))}]"

        sql = """
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
            sql += " AND dc.document_id = ANY(:doc_ids)"
            params["doc_ids"] = [str(did) for did in document_ids]

        sql += " ORDER BY similarity DESC LIMIT :top_k"
        params["top_k"] = top_k

        try:
            await self._db.execute(text("SET LOCAL ivfflat.probes = 10"))
        except Exception as exc:
            logger.debug("PgVectorStore: could not SET LOCAL ivfflat.probes: %s", exc)

        result = await self._db.execute(text(sql), params)
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
            "PgVectorStore: retrieved %d chunks (user=%s, threshold=%.2f, top_k=%d)",
            len(chunks),
            user_id,
            threshold,
            top_k,
        )
        return chunks

    async def get_sampled_chunks_for_summary(
        self,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID] | None = None,
        top_k_per_doc: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Sample evenly distributed chunks (e.g. 0%, 50%, 100%) across target documents.

        Args:
            user_id: Requesting user ID for isolation.
            document_ids: Optional document IDs to sample from.
            top_k_per_doc: Maximum chunks to sample per document.

        Returns:
            List of RetrievedChunk ordered by document and chunk_index.
        """
        target_doc_ids: list[uuid.UUID] = []

        if document_ids:
            target_doc_ids = list(document_ids)
        else:
            doc_query = text(
                "SELECT id FROM documents "
                "WHERE user_id = :user_id AND status = 'ready' "
                "ORDER BY created_at DESC LIMIT 3"
            )
            doc_result = await self._db.execute(doc_query, {"user_id": str(user_id)})
            target_doc_ids = [row.id for row in doc_result.fetchall()]

        if not target_doc_ids:
            return []

        sampled_chunks: list[RetrievedChunk] = []
        for did in target_doc_ids:
            chunk_query = text(
                "SELECT id, document_id, content, metadata, chunk_index "
                "FROM document_chunks "
                "WHERE document_id = :doc_id "
                "ORDER BY chunk_index ASC"
            )
            chunk_result = await self._db.execute(chunk_query, {"doc_id": str(did)})
            doc_chunks = chunk_result.fetchall()

            if not doc_chunks:
                continue

            n = len(doc_chunks)
            if n <= top_k_per_doc:
                selected_indices = list(range(n))
            elif top_k_per_doc == 3:
                selected_indices = [0, n // 2, n - 1]
            else:
                # Generalized even sampling
                step = (n - 1) / (top_k_per_doc - 1) if top_k_per_doc > 1 else 0
                selected_indices = sorted(list({int(round(i * step)) for i in range(top_k_per_doc)}))

            for idx in selected_indices:
                row = doc_chunks[idx]
                sampled_chunks.append(
                    RetrievedChunk(
                        chunk_id=row.id,
                        document_id=row.document_id,
                        content=row.content,
                        similarity=1.0,
                        metadata=row.metadata or {},
                    )
                )

        logger.info(
            "PgVectorStore: sampled %d chunks for summary (docs=%d, user=%s)",
            len(sampled_chunks),
            len(target_doc_ids),
            user_id,
        )
        return sampled_chunks

    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunks: list[dict],
    ) -> int:
        """
        Insert a batch of embedded chunks for a document.

        Each chunk dict must have keys:
            - content (str)
            - embedding (list[float])
            - chunk_index (int)
            - metadata (dict)

        Args:
            document_id: UUID of the parent document.
            chunks: List of chunk dicts ready for storage.

        Returns:
            Number of chunks successfully inserted.
        """
        if not chunks:
            return 0

        insert_sql = """
            INSERT INTO document_chunks (id, document_id, content, embedding, chunk_index, metadata)
            VALUES (:id, :document_id, :content, :embedding::vector, :chunk_index, :metadata::jsonb)
            ON CONFLICT (id) DO UPDATE
              SET content = EXCLUDED.content,
                  embedding = EXCLUDED.embedding,
                  chunk_index = EXCLUDED.chunk_index,
                  metadata = EXCLUDED.metadata
        """
        import json

        for chunk in chunks:
            vec_str = f"[{','.join(map(str, chunk['embedding']))}]"
            await self._db.execute(
                text(insert_sql),
                {
                    "id": str(uuid.uuid4()),
                    "document_id": str(document_id),
                    "content": chunk["content"],
                    "embedding": vec_str,
                    "chunk_index": chunk["chunk_index"],
                    "metadata": json.dumps(chunk.get("metadata", {})),
                },
            )

        logger.info(
            "PgVectorStore: upserted %d chunks for document=%s",
            len(chunks),
            document_id,
        )
        return len(chunks)

    async def delete_document_chunks(self, document_id: uuid.UUID) -> int:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: UUID of the document.

        Returns:
            Number of chunks deleted.
        """
        result = await self._db.execute(
            text("DELETE FROM document_chunks WHERE document_id = :doc_id RETURNING id"),
            {"doc_id": str(document_id)},
        )
        count = len(result.fetchall())
        logger.info("PgVectorStore: deleted %d chunks for document=%s", count, document_id)
        return count
