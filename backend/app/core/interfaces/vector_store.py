# name: vector_store.py
# description: Abstract interface (port) for vector similarity search stores.
#              Decouples retrieval logic from any specific vector DB (pgvector, Qdrant, etc.)

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """
    A document chunk returned from a vector similarity search.

    Attributes:
        chunk_id: UUID of the stored chunk.
        document_id: UUID of the parent document.
        content: Raw text content of the chunk.
        similarity: Cosine similarity score (0–1, higher is better).
        metadata: Source metadata dict (page, filename, section, etc.).
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    similarity: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for storage in Message.sources."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content[:300],  # Truncate for storage
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
        }


class IVectorStore(ABC):
    """
    Port interface for vector similarity search over document chunks.

    Implementations must handle embedding lookup, ANN indexing,
    and user-level data isolation.
    """

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: list[float],
        user_id: uuid.UUID,
        top_k: int = 5,
        threshold: float = 0.5,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Find the top-k chunks most similar to the given query vector.

        Args:
            query_vector: Pre-computed embedding of the query.
            user_id: Requesting user's ID — used for strict data isolation.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score to include a result.
            document_ids: Optional filter to restrict search to specific documents.

        Returns:
            List of RetrievedChunk ordered by similarity descending.
        """

    @abstractmethod
    async def upsert_chunks(
        self,
        document_id: uuid.UUID,
        chunks: list[dict],
    ) -> int:
        """
        Store or update a batch of embedded chunks for a document.

        Args:
            document_id: UUID of the parent document.
            chunks: List of chunk dicts with keys: content, embedding, chunk_index, metadata.

        Returns:
            Number of chunks successfully stored.
        """

    @abstractmethod
    async def delete_document_chunks(self, document_id: uuid.UUID) -> int:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: UUID of the document whose chunks should be removed.

        Returns:
            Number of chunks deleted.
        """
