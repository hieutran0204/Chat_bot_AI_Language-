# name: pipeline.py (NaiveRagPipeline)
# description: Naive RAG implementation — single-stage vector retrieval followed by
#              LLM generation. Supports 4 chatbot modes with intent-based retrieval routing.
#              Extends BasePipeline, consumes ILLMProvider + IVectorStore + IEmbedder.

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.interfaces.embedder import IEmbedder
from app.core.interfaces.llm_provider import ChatMessage, ILLMProvider
from app.core.interfaces.vector_store import IVectorStore, RetrievedChunk
from app.models.conversation import ConversationMode
from app.pipelines.base_pipeline import BasePipeline, estimate_tokens
from app.services.profile_loader import ProfileLoader

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
    2. Embed the query and run cosine similarity search via IVectorStore (or sample for summary).
    3. Format retrieved chunks into a numbered context block with security delimiters.
    4. Load learner profile weaknesses and inject into system prompt.
    5. Trim prompt to budget (75% of num_ctx).
    6. Stream LLM generation token by token via ILLMProvider.
    """

    def __init__(
        self,
        llm: ILLMProvider,
        vector_store: IVectorStore,
        embedder: IEmbedder,
        db: AsyncSession | None = None,
        profile_loader: ProfileLoader | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        history_limit: int = 5,
    ) -> None:
        super().__init__(llm=llm, vector_store=vector_store, history_limit=history_limit)
        self._embedder = embedder
        self._db = db
        self._profile_loader = profile_loader or ProfileLoader()
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    async def prepare_context(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
        db: AsyncSession | None = None,
        tracker: Any | None = None,
        search_query: str | None = None,
        is_summary: bool = False,
    ) -> tuple[list[ChatMessage], list[RetrievedChunk]]:
        """
        Execute DB/Vector retrieval and assemble prompt messages in a short-lived DB session.

        Allows closing the database connection BEFORE entering the LLM stream.
        """
        chunks: list[RetrievedChunk] = []
        effective_query = search_query or user_message

        if mode in _RETRIEVAL_MODES:
            if is_summary:
                t_retrieval_start = time.perf_counter()
                top_k_per_doc = getattr(settings, "summarization_top_k_per_doc", 3)
                chunks = await self._vector_store.get_sampled_chunks_for_summary(
                    user_id=user_id,
                    document_ids=document_ids,
                    top_k_per_doc=top_k_per_doc,
                )
                if tracker:
                    tracker.record_step("summary_chunk_sampling", (time.perf_counter() - t_retrieval_start) * 1000.0)
                    tracker.add_metric("chunks_found", len(chunks))
            else:
                t_embed_start = time.perf_counter()
                query_vector = await asyncio.to_thread(self._embedder.embed, effective_query)
                if tracker:
                    tracker.record_step("query_embedding", (time.perf_counter() - t_embed_start) * 1000.0)

                t_retrieval_start = time.perf_counter()
                threshold = self._similarity_threshold if self._similarity_threshold is not None else getattr(settings, "retrieval_similarity_threshold", 0.5)
                top_k = self._top_k if self._top_k is not None else getattr(settings, "retrieval_top_k", 5)
                chunks = await self._vector_store.similarity_search(
                    query_vector=query_vector,
                    user_id=user_id,
                    top_k=top_k,
                    threshold=threshold,
                    document_ids=document_ids,
                )
                if tracker:
                    tracker.record_step("vector_retrieval", (time.perf_counter() - t_retrieval_start) * 1000.0)
                    tracker.add_metric("chunks_found", len(chunks))

        profile_context = ""
        target_db = db or self._db
        if target_db is not None:
            t_profile_start = time.perf_counter()
            try:
                profile_context = await self._profile_loader.get_weakness_context(
                    user_id=user_id,
                    db=target_db,
                )
            except Exception as e:
                logger.warning("NaiveRagPipeline: failed to load learner profile for user=%s: %s", user_id, e)
            if tracker:
                tracker.record_step("profile_load", (time.perf_counter() - t_profile_start) * 1000.0)

        t_prompt_start = time.perf_counter()
        initial_chunk_count = len(chunks)
        initial_history_count = len(conversation_history)

        messages, trimmed_chunks = self.trim_to_budget(
            mode=mode,
            user_level=user_level,
            profile_context=profile_context,
            conversation_history=conversation_history,
            user_message=user_message,
            chunks=chunks,
            num_ctx=getattr(settings, "ollama_num_ctx", 8192),
            is_summary=is_summary,
        )

        if tracker:
            tracker.record_step("prompt_preparation", (time.perf_counter() - t_prompt_start) * 1000.0)
            if profile_context:
                tracker.add_metric("profile_injected", True)
            tracker.add_metric("chunks_trimmed", initial_chunk_count - len(trimmed_chunks))
            tracker.add_metric("history_turns_trimmed", max(0, initial_history_count - len(messages) + 2))
            tracker.add_metric("estimated_tokens_sent", sum(estimate_tokens(m.get("content", "")) for m in messages))

        return messages, trimmed_chunks

    async def stream_from_messages(
        self,
        messages: list[ChatMessage],
        chunks: list[RetrievedChunk],
        tracker: Any | None = None,
    ) -> AsyncGenerator[tuple[str, list[RetrievedChunk]], None]:
        """
        Stream tokens directly from pre-assembled messages without any database dependency.
        """
        first = True
        llm_start = time.perf_counter()
        first_token_time: float | None = None

        async for token in self._llm.stream_chat(messages):
            now = time.perf_counter()
            if first:
                first_token_time = now
                if tracker:
                    tracker.record_step("llm_time_to_first_token", (first_token_time - llm_start) * 1000.0)
                yield token, chunks
                first = False
            else:
                yield token, []

            if tracker:
                tracker.increment_tokens(1)

        if tracker and first_token_time is not None:
            gen_duration = (time.perf_counter() - first_token_time) * 1000.0
            tracker.record_step("llm_token_generation", gen_duration)

    async def run_stream(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
        tracker: Any | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, list[RetrievedChunk]], None]:
        """
        Backward-compatible helper executing retrieval and streaming sequentially.
        """
        db = kwargs.get("db") or self._db
        messages, chunks = await self.prepare_context(
            user_message=user_message,
            mode=mode,
            user_id=user_id,
            user_level=user_level,
            conversation_history=conversation_history,
            document_ids=document_ids,
            db=db,
            tracker=tracker,
        )

        async for token, c in self.stream_from_messages(messages, chunks, tracker=tracker):
            yield token, c
