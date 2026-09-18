# name: pipeline.py
# description: Abstract interface (port) for RAG pipeline techniques.
#              Every RAG implementation (Naive, Hybrid, GraphRAG, Agentic…)
#              must implement this contract.

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from app.core.interfaces.llm_provider import ChatMessage
from app.core.interfaces.vector_store import RetrievedChunk


class IRagPipeline(ABC):
    """
    Port interface for a RAG pipeline technique.

    Defines the shared contract for all RAG variants so that the
    application service layer (QueryService) is decoupled from
    any specific retrieval or generation strategy.

    Each concrete pipeline receives its dependencies (LLM, vector store,
    embedder, etc.) via constructor injection — never by importing
    concrete adapters directly.
    """

    @abstractmethod
    async def run(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Execute the full RAG pipeline and return the complete response.

        Args:
            user_message: The user's current input text.
            mode: Chatbot mode string (e.g. 'conversation', 'grammar', 'doc_qa').
            user_id: UUID of the requesting user (for data isolation in retrieval).
            user_level: English proficiency level (A1–C2) for prompt personalization.
            conversation_history: Ordered list of previous role/content messages.
            document_ids: Optional filter to restrict retrieval to specific documents.

        Returns:
            Tuple of (full_response_text, list_of_retrieved_chunks).
            retrieved_chunks is empty for non-retrieval modes.
        """

    @abstractmethod
    async def run_stream(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
    ) -> AsyncGenerator[tuple[str, list[RetrievedChunk]], None]:
        """
        Execute the RAG pipeline and stream the response token by token.

        Args:
            user_message: The user's current input text.
            mode: Chatbot mode string.
            user_id: UUID of the requesting user.
            user_level: English proficiency level.
            conversation_history: Ordered list of previous messages.
            document_ids: Optional filter for retrieval.

        Yields:
            Tuple of (token_string, retrieved_chunks).
            retrieved_chunks is non-empty only on the first yield for retrieval modes.
        """
