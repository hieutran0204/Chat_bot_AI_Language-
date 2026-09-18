# name: pipeline.py
# description: Backward-compatibility shim — delegates to the new Hexagonal Architecture
#              (factory + NaiveRagPipeline). Existing code that imports RAGPipeline
#              continues to work without modification.
#
# IMPORTANT: Do NOT add new logic here. Extend pipelines/ instead.

import logging
import uuid
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

# Re-export so existing `from app.rag.pipeline import ...` imports keep working
from app.core.interfaces.llm_provider import ChatMessage  # noqa: F401
from app.core.interfaces.vector_store import RetrievedChunk  # noqa: F401
from app.factory.pipeline_factory import create_pipeline

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Backward-compatibility shim for the original RAGPipeline class.

    Delegates all execution to the new NaiveRagPipeline via pipeline_factory.
    Maintains the same public interface (run / run_stream) so that
    app/api/v1/chat.py requires zero changes.

    For new features, extend app/pipelines/ directly.

    Args:
        db: Async SQLAlchemy session (injected per-request).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._pipeline = create_pipeline(db, pipeline_type="naive")

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
        Stream the RAG response token by token.

        Delegates to NaiveRagPipeline.run_stream().

        Args:
            user_message: The user's current input.
            mode: ConversationMode string value.
            user_id: UUID of the requesting user.
            user_level: English proficiency level (A1–C2).
            conversation_history: Recent message history.
            document_ids: Optional document filter for retrieval.

        Yields:
            Tuple of (token_string, retrieved_chunks).
        """
        async for token, chunks in self._pipeline.run_stream(
            user_message=user_message,
            mode=mode,
            user_id=user_id,
            user_level=user_level,
            conversation_history=conversation_history,
            document_ids=document_ids,
        ):
            yield token, chunks

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
        Execute the RAG pipeline and return the full response (non-streaming).

        Delegates to NaiveRagPipeline.run().

        Args:
            user_message: The user's current input.
            mode: ConversationMode string value.
            user_id: UUID of the requesting user.
            user_level: English proficiency level.
            conversation_history: Recent message history.
            document_ids: Optional document filter.

        Returns:
            Tuple of (full_response_text, retrieved_chunks).
        """
        return await self._pipeline.run(
            user_message=user_message,
            mode=mode,
            user_id=user_id,
            user_level=user_level,
            conversation_history=conversation_history,
            document_ids=document_ids,
        )

