# name: pipeline.py (NaiveRagPipeline)
# description: Naive RAG implementation — single-stage vector retrieval followed by
#              LLM generation. Supports 4 chatbot modes with intent-based retrieval routing.
#              Extends BasePipeline, consumes ILLMProvider + IVectorStore + IEmbedder.

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator

from app.core.interfaces.embedder import IEmbedder
from app.core.interfaces.llm_provider import ChatMessage, ILLMProvider
from app.core.interfaces.vector_store import IVectorStore, RetrievedChunk
from app.models.conversation import ConversationMode
from app.pipelines.base_pipeline import BasePipeline

logger = logging.getLogger(__name__)

# Modes that require vector retrieval
_RETRIEVAL_MODES: frozenset[str] = frozenset(
    {ConversationMode.GRAMMAR, ConversationMode.DOCUMENT_QA}
)


class NaiveRagPipeline(BasePipeline):
    """
    Naive RAG pipeline: embed query → ANN search → prompt build → LLM stream.

    Flow:
    1. Determine if the mode requires retrieval (GRAMMAR, DOCUMENT_QA).
    2. Embed the user query and run cosine similarity search via IVectorStore.
    3. Format retrieved chunks into a numbered context block.
    4. Render the mode-specific system prompt with context injected.
    5. Stream LLM generation token by token via ILLMProvider.

    Modes without retrieval (CONVERSATION, VOCABULARY) bypass steps 2-3.

    Args:
        llm: LLM provider for text generation (Ollama, HuggingFace, etc.).
        vector_store: Vector store for chunk retrieval (pgvector, Qdrant, etc.).
        embedder: Embedding provider for query vectorization.
        top_k: Maximum number of chunks to retrieve per query.
        similarity_threshold: Minimum similarity score to include a chunk.
        history_limit: Maximum conversation turns to include in context.
    """

    def __init__(
        self,
        llm: ILLMProvider,
        vector_store: IVectorStore,
        embedder: IEmbedder,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        history_limit: int = 5,
    ) -> None:
        super().__init__(llm=llm, vector_store=vector_store, history_limit=history_limit)
        self._embedder = embedder
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

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
        Execute the Naive RAG pipeline and stream the LLM response token by token.

        Args:
            user_message: The user's current input text.
            mode: ConversationMode string (conversation|grammar|vocabulary|doc_qa).
            user_id: UUID of the requesting user for retrieval data isolation.
            user_level: English proficiency level (A1–C2) for prompt personalization.
            conversation_history: Ordered list of previous role/content messages.
            document_ids: Optional filter to restrict retrieval to specific documents.

        Yields:
            Tuple of (token_string, retrieved_chunks).
            retrieved_chunks is non-empty only on the first yield for retrieval modes.
        """
        # Step 1: Retrieve relevant chunks (only for retrieval-enabled modes)
        chunks: list[RetrievedChunk] = []

        if mode in _RETRIEVAL_MODES:
            query_vector = self._embedder.embed(user_message)
            chunks = await self._vector_store.similarity_search(
                query_vector=query_vector,
                user_id=user_id,
                top_k=self._top_k,
                threshold=self._similarity_threshold,
                document_ids=document_ids,
            )
            logger.info(
                "NaiveRAG: retrieved %d chunks (mode=%s, user=%s)",
                len(chunks),
                mode,
                user_id,
            )

        # Step 2: Render system prompt with context injected
        system_prompt = self._render_system_prompt(mode, user_level, chunks)

        # Step 3: Assemble full message list
        messages = self._build_message_list(system_prompt, conversation_history, user_message)

        logger.info(
            "NaiveRAG: streaming (mode=%s, chunks=%d, history=%d)",
            mode,
            len(chunks),
            len(conversation_history),
        )

        # Step 4: Stream LLM generation — emit chunks only on first token
        first = True
        async for token in self._llm.stream_chat(messages):
            if first:
                yield token, chunks  # Attach retrieved sources to first token
                first = False
            else:
                yield token, []
