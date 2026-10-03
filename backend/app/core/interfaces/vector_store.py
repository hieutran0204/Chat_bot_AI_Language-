# name: vector_store.py
# description: Abstract interface (port) for vector similarity search stores.
#              Decouples retrieval logic from any specific vector DB (pgvector, Qdrant, etc.)

from __future__ import annotations

import html
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """
    A document chunk returned from a vector similarity search or document sampler.

    Attributes:
        chunk_id: UUID of the stored chunk.
        document_id: UUID of the parent document.
        content: Raw text content of the chunk.
        similarity: Cosine similarity score (0–1, higher is better).
        metadata: Source metadata dict (page, filename, section, etc.).
        preview: Escaped first 120 characters snippet for UI preview.
        index_in_prompt: 1-based index matching [n] in LLM prompt.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    similarity: float
    metadata: dict = field(default_factory=dict)
    preview: str | None = None
    index_in_prompt: int | None = None

    def get_escaped_preview(self, max_length: int = 120) -> str:
        """Return HTML-escaped preview snippet."""
        if self.preview:
            return self.preview
        snippet = (self.content or "")[:max_length]
        return html.escape(snippet)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for SSE sources event (backward-compatible)."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content,
            "similarity": round(self.similarity, 4),
            "metadata": self.metadata,
            "preview": self.get_escaped_preview(),
            "index_in_prompt": self.index_in_prompt,
        }

    def to_storage_dict(self) -> dict:
        """Serialize lightweight reference for database storage in Message.sources."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "chunk_index": self.metadata.get("chunk_index"),
            "similarity": round(self.similarity, 4),
            "preview": self.get_escaped_preview(),
            "index_in_prompt": self.index_in_prompt,
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
    async def get_sampled_chunks_for_summary(
        self,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID] | None = None,
        top_k_per_doc: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Sample representative chunks (e.g., 0%, 50%, 100% chunk_index) for summarization requests.

        Args:
            user_id: Requesting user ID for data isolation.
            document_ids: Optional document IDs to restrict sampling to.
            top_k_per_doc: Number of representative chunks to sample per document.

        Returns:
            List of RetrievedChunk sampled across documents.
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
