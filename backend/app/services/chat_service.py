# name: chat_service.py
# description: Business logic for conversation management, message persistence,
#              resilient streaming lifecycle, AnyIO CancelScope, and atomic idempotency.

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
import logging
import uuid
from typing import Any

import anyio
from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from app.core.database import AsyncSessionFactory
from app.core.exceptions import (
    LLMException,
    LLMInternalError,
    LLMTimeout,
)
from app.core.i18n.fallback_messages import FallbackReason, get_fallback_message
from app.core.interfaces.memory import ILearnerProfileStore
from app.core.interfaces.vector_store import RetrievedChunk
from app.core.primitives.messages import AIMessage, HumanMessage
from app.core.profiler import LatencyTracker
from app.core.redis.memory import RedisChatMessageHistory
from app.core.redis.profile import RedisLearnerProfileManager
from app.factory.pipeline_factory import create_pipeline
from app.pipelines.components.intent_classifier import IntentType, classify_intent
from app.pipelines.components.query_rewriter import rewrite_query
from app.models.conversation import (
    Conversation,
    ConversationMode,
    Message,
    MessageRole,
    MessageStatus,
)
from app.models.progress import UserProgress

logger = logging.getLogger(__name__)


async def reap_stale_streaming_messages(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    max_age_seconds: int = 300,
) -> int:
    """
    Find messages stuck in 'streaming' status older than max_age_seconds and mark them 'failed'.

    Runs at application startup and periodic intervals to clean up orphan messages from server crashes.
    """
    factory = session_factory or AsyncSessionFactory
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    async with factory() as session:
        stmt = (
            update(Message)
            .where(
                Message.status == MessageStatus.STREAMING,
                Message.updated_at <= cutoff,
            )
            .values(status=MessageStatus.FAILED)
        )
        result = await session.execute(stmt)
        await session.commit()
        reaped = result.rowcount
        if reaped > 0:
            logger.warning("ReaperTask: recovered %d stale streaming messages to 'failed'.", reaped)
        return reaped


class ChatService:
    """
    Manages conversations and messages for the chat API with resilient streaming.

    Key Architectural Principles:
    - Never hold DB sessions across token streaming.
    - Fail-closed status management: stream fails unless explicitly completed.
    - Idempotency with atomic INSERT ON CONFLICT and safe retry locking.
    - History queries only return clean, completed conversation turns.
    - AnyIO CancelScope(shield=True) guarantees final DB persistence on disconnect.
    """

    def __init__(
        self,
        db: AsyncSession | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        profile_mgr: ILearnerProfileStore | None = None,
    ) -> None:
        self._db = db
        self._session_factory = session_factory or AsyncSessionFactory
        self._profile_mgr: ILearnerProfileStore = profile_mgr or RedisLearnerProfileManager()

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        mode: str = ConversationMode.CONVERSATION,
        title: str | None = None,
    ) -> Conversation:
        """Create a new conversation session."""
        async with self._session_factory() as session:
            conv = Conversation(user_id=user_id, mode=mode, title=title)
            session.add(conv)
            await session.commit()
            await session.refresh(conv)
            logger.info("Created conversation %s mode=%s user=%s", conv.id, mode, user_id)
            return conv

    async def get_conversation(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID, session: AsyncSession | None = None
    ) -> Conversation | None:
        """Fetch a conversation, verifying it belongs to the user."""
        if session is not None:
            result = await session.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            return result.scalar_one_or_none()

        async with self._session_factory() as s:
            result = await s.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_conversations(self, user_id: uuid.UUID) -> list[Conversation]:
        """List all conversations for a user, newest first."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.created_at.desc())
            )
            return list(result.scalars().all())

    async def get_history(
        self,
        conversation_id: uuid.UUID,
        limit: int = 10,
        exclude_user_msg_id: uuid.UUID | None = None,
        session: AsyncSession | None = None,
        force_db: bool = False,
    ) -> list[dict]:
        """
        Retrieve clean recent messages for LLM context.
        Excludes the current in-flight user message and any incomplete turns.
        Excludes fallback turns (is_fallback == True).
        Warms up Redis buffer on cache miss when not forcing direct DB access.

        D12 Strategy:
        - When force_db=False (CONVERSATION/VOCABULARY): reads from Redis sliding-window cache first.
        - When force_db=True (DOCUMENT_QA/GRAMMAR): queries PostgreSQL directly to ensure strict
          relational pairing and fallback exclusion for query rewriter context.
        """
        if not force_db:
            try:
                redis_mem = RedisChatMessageHistory(conversation_id)
                cached_messages = await redis_mem.get_messages(limit=limit)
                if cached_messages:
                    return [
                        {
                            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
                            "content": m.content,
                        }
                        for m in cached_messages
                    ]
            except Exception as exc:
                logger.warning("Redis history cache read failed for conv %s: %s", conversation_id, exc)

        # PostgreSQL fallback / forced clean fetch:
        async def _query_pg(s: AsyncSession) -> list[dict]:
            UserMsg = aliased(Message)
            query = (
                select(UserMsg.content.label("user_content"), Message.content.label("asst_content"))
                .join(UserMsg, Message.parent_message_id == UserMsg.id)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.role == MessageRole.ASSISTANT,
                    Message.status == MessageStatus.COMPLETE,
                    Message.is_fallback == False,
                    UserMsg.status == MessageStatus.COMPLETE,
                )
            )
            if exclude_user_msg_id is not None:
                query = query.where(UserMsg.id != exclude_user_msg_id)

            # limit is total messages; divide by 2 for pairs
            pair_limit = max(1, limit // 2)
            query = query.order_by(Message.seq.desc()).limit(pair_limit)
            pairs = list(reversed((await s.execute(query)).all()))

            history: list[dict] = []
            to_cache: list[HumanMessage | AIMessage] = []
            for user_content, asst_content in pairs:
                history.append({"role": "user", "content": user_content})
                history.append({"role": "assistant", "content": asst_content})
                to_cache.append(HumanMessage(content=user_content))
                to_cache.append(AIMessage(content=asst_content))

            # Warm up Redis buffer with clean history only on normal cache miss
            if to_cache and not force_db:
                try:
                    redis_mem = RedisChatMessageHistory(conversation_id)
                    await redis_mem.add_messages(to_cache)
                except Exception as exc:
                    logger.warning("Failed to warm up Redis buffer for conv %s: %s", conversation_id, exc)

            return history

        if session is not None:
            return await _query_pg(session)
        async with self._session_factory() as s:
            return await _query_pg(s)

    async def get_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
        after_id: uuid.UUID | None = None,
    ) -> list[Message]:
        """
        Retrieve message records for API responses with cursor synchronization.
        Uses Message.seq identity column for strictly ascending, tie-free pagination.
        """
        async with self._session_factory() as session:
            query = select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role != MessageRole.SYSTEM,
            )
            if after_id is not None:
                anchor = await session.get(Message, after_id)
                if anchor is not None:
                    query = query.where(Message.seq > anchor.seq).order_by(Message.seq.asc()).limit(limit)
                    return list((await session.scalars(query)).all())

            # Fetch latest `limit` messages, returned chronologically
            query = query.order_by(Message.seq.desc()).limit(limit)
            results = list((await session.scalars(query)).all())
            return list(reversed(results))

    async def sync_redis_turn(
        self, conversation_id: uuid.UUID, user_content: str, assistant_content: str
    ) -> None:
        """
        Append a completed turn (user + assistant) to the Redis sliding buffer.
        Only called when assistant generation has reached 'complete' status.
        """
        redis_mem = RedisChatMessageHistory(conversation_id)
        await redis_mem.add_messages([
            HumanMessage(content=user_content),
            AIMessage(content=assistant_content),
        ])

    async def update_progress(
        self, session: AsyncSession, user_id: uuid.UUID, mode: str
    ) -> None:
        """Upsert user's daily progress counters."""
        today = datetime.now(UTC).date()
        stmt = pg_insert(UserProgress).values(
            user_id=user_id,
            date=today,
            messages_sent=1,
            vocab_learned=1 if mode == ConversationMode.VOCABULARY else 0,
            study_minutes=1,
            updated_at=datetime.now(UTC),
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_progress_user_date",
            set_={
                "messages_sent": UserProgress.messages_sent + 1,
                "vocab_learned": UserProgress.vocab_learned + (
                    1 if mode == ConversationMode.VOCABULARY else 0
                ),
                "study_minutes": UserProgress.study_minutes + 1,
                "updated_at": datetime.now(UTC),
            },
        )
        await session.execute(stmt)

    async def _finalize(
        self,
        placeholder_id: uuid.UUID,
        parts: list[str],
        final_status: str,
        chunks: list[RetrievedChunk],
        conversation_id: uuid.UUID,
        user_content: str,
        user_id: uuid.UUID,
        conv_mode: str,
        is_fallback: bool = False,
    ) -> None:
        """
        Persist final assistant state inside an isolated AnyIO CancelScope(shield=True).
        Commits status change first, then executes secondary operations (progress, Redis) best-effort.
        Fallback turns are recorded with is_fallback=True and are NEVER synced to Redis.
        """
        full_text = "".join(parts)
        persisted = False
        try:
            async with self._session_factory() as session:
                stmt = (
                    update(Message)
                    .where(Message.id == placeholder_id)
                    .values(
                        content=full_text,
                        status=final_status,
                        is_fallback=is_fallback,
                        updated_at=datetime.now(UTC),
                        sources=[c.to_storage_dict() for c in chunks] if (chunks and not is_fallback) else None,
                    )
                )
                await session.execute(stmt)
                await session.commit()
                persisted = True
                logger.info(
                    "Finalized message %s with status=%s is_fallback=%s (length=%d)",
                    placeholder_id,
                    final_status,
                    is_fallback,
                    len(full_text),
                )
        except Exception as exc:
            logger.exception("CRITICAL: Failed to update message %s status=%s in DB: %s", placeholder_id, final_status, exc)

        # D13: Exclude fallback turns (greeting, processing, failed_doc, not_found) from user learning progress
        if persisted and final_status == MessageStatus.COMPLETE and not is_fallback:
            # Best-effort progress counter
            try:
                async with self._session_factory() as session:
                    await self.update_progress(session, user_id, conv_mode)
                    await session.commit()
            except Exception as exc:
                logger.warning("Failed to update progress for user %s: %s", user_id, exc)

            # Best-effort Redis sliding buffer sync (D5: never sync fallback turns)
            try:
                await self.sync_redis_turn(conversation_id, user_content, full_text)
            except Exception as exc:
                logger.warning("Failed to sync turn to Redis for conv %s: %s", conversation_id, exc)

    async def stream_message(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        user_level: str,
        content: str,
        client_message_id: uuid.UUID | None = None,
        audio_url: str | None = None,
        audio_duration_ms: int | None = None,
        stt_confidence: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
        captured: dict | None = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Execute an asynchronous streaming chat turn yielding SSE events.

        Resilience features:
        - Atomic idempotency on client_message_id.
        - Fail-closed status management: starts as FAILED, only marked COMPLETE after parts confirmed.
        - Step 1 & 2 fully enclosed in error-handling try/finally.
        - AnyIO CancelScope(shield=True) guarantees final DB persistence.
        - Yield 'error' event only outside except/finally blocks.
        """
        user_msg_id: uuid.UUID | None = None
        placeholder_id: uuid.UUID | None = None
        conv_mode = ConversationMode.CONVERSATION

        # ── Step 1: Session 1 — Atomic Idempotency & Placeholder Creation ──────
        try:
            async with self._session_factory() as session:
                conv = await self.get_conversation(conversation_id, user_id, session=session)
                if not conv:
                    yield ("error", {"code": "NOT_FOUND", "retryable": False, "message": "Conversation not found."})
                    return
                conv_mode = conv.mode

                if client_message_id is not None:
                    # Atomic insert user message with partial index on_conflict_do_nothing
                    stmt = (
                        pg_insert(Message)
                        .values(
                            conversation_id=conversation_id,
                            role=MessageRole.USER,
                            content=content,
                            status=MessageStatus.COMPLETE,
                            client_message_id=client_message_id,
                            audio_url=audio_url,
                            audio_duration_ms=audio_duration_ms,
                            stt_confidence=stt_confidence,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[Message.conversation_id, Message.client_message_id],
                            index_where=Message.client_message_id.isnot(None),
                        )
                        .returning(Message.id)
                    )
                    res = await session.execute(stmt)
                    user_msg_id = res.scalar_one_or_none()

                    if user_msg_id is None:
                        # Duplicate request: query existing user message and its reply
                        existing_user = await session.scalar(
                            select(Message).where(
                                Message.conversation_id == conversation_id,
                                Message.client_message_id == client_message_id,
                            )
                        )
                        if existing_user is not None:
                            user_msg_id = existing_user.id
                            existing_reply = await session.scalar(
                                select(Message).where(
                                    Message.parent_message_id == existing_user.id
                                )
                            )
                            if existing_reply is not None:
                                if existing_reply.status == MessageStatus.COMPLETE:
                                    # Replay already-completed answer
                                    yield ("start", {
                                        "user_message_id": str(existing_user.id),
                                        "assistant_message_id": str(existing_reply.id),
                                    })
                                    yield ("token", {"token": existing_reply.content})
                                    yield ("done", {
                                        "message_id": str(existing_reply.id),
                                        "status": MessageStatus.COMPLETE,
                                    })
                                    return

                                # Atomic lock for retry: only claim if failed, interrupted, or stale streaming (>300s)
                                stale_before = datetime.now(UTC) - timedelta(seconds=300)
                                lock_stmt = (
                                    update(Message)
                                    .where(
                                        Message.id == existing_reply.id,
                                        or_(
                                            Message.status.in_([MessageStatus.FAILED, MessageStatus.INTERRUPTED]),
                                            and_(
                                                Message.status == MessageStatus.STREAMING,
                                                Message.updated_at <= stale_before,
                                            ),
                                        ),
                                    )
                                    .values(
                                        status=MessageStatus.STREAMING,
                                        content="",
                                        updated_at=datetime.now(UTC),
                                    )
                                    .returning(Message.id)
                                )
                                lock_res = await session.execute(lock_stmt)
                                if lock_res.scalar_one_or_none() is not None:
                                    placeholder_id = existing_reply.id
                                    await session.commit()
                                else:
                                    yield ("error", {
                                        "code": "IN_PROGRESS",
                                        "retryable": True,
                                        "message": "This message is currently being processed. Please wait.",
                                    })
                                    return

                # New request: insert user message and assistant placeholder
                if user_msg_id is None:
                    new_user = Message(
                        conversation_id=conversation_id,
                        role=MessageRole.USER,
                        content=content,
                        status=MessageStatus.COMPLETE,
                        client_message_id=client_message_id,
                        audio_url=audio_url,
                        audio_duration_ms=audio_duration_ms,
                        stt_confidence=stt_confidence,
                    )
                    session.add(new_user)
                    await session.flush()
                    user_msg_id = new_user.id

                if placeholder_id is None:
                    new_placeholder = Message(
                        conversation_id=conversation_id,
                        role=MessageRole.ASSISTANT,
                        content="",
                        status=MessageStatus.STREAMING,
                        parent_message_id=user_msg_id,
                    )
                    session.add(new_placeholder)
                    await session.commit()
                    placeholder_id = new_placeholder.id

        except Exception as exc:
            logger.exception("Step 1 failed for conv=%s: %s", conversation_id, exc)
            yield ("error", {
                "code": "LLM_UNAVAILABLE",
                "retryable": True,
                "message": "Failed to initialize conversation turn. Please retry.",
            })
            return

        # Emit start event early so client has anchor IDs
        yield ("start", {
            "user_message_id": str(user_msg_id),
            "assistant_message_id": str(placeholder_id),
        })

        # ── Step 2 & 3: Streaming Loop with fail-closed status ───────────────────
        parts: list[str] = []
        chunks: list[RetrievedChunk] = []
        final_status = MessageStatus.FAILED  # Fail-closed default
        is_fallback = False
        fallback_reason: str | None = None
        err_payload: dict[str, Any] | None = None
        tracker = LatencyTracker(conversation_id, mode=conv_mode)

        try:
            # 1. Intent Classification (before any doc checks)
            intent = classify_intent(content)
            if intent == IntentType.GREETING:
                canned = get_fallback_message(FallbackReason.GREETING)
                parts.append(canned)
                yield ("token", {"token": canned})
                final_status = MessageStatus.COMPLETE
                is_fallback = True
                fallback_reason = FallbackReason.GREETING.value

            # 2. Document Status Pre-check for DOCUMENT_QA (if not already fallback)
            if not is_fallback and conv_mode == ConversationMode.DOCUMENT_QA:
                async with self._session_factory() as doc_session:
                    if document_ids:
                        doc_stmt = text(
                            "SELECT status FROM documents WHERE id = ANY(:ids) AND user_id = :uid"
                        )
                        doc_rows = (await doc_session.execute(doc_stmt, {"ids": document_ids, "uid": user_id})).fetchall()
                        statuses = [r.status for r in doc_rows]
                        if any(s == "failed" for s in statuses):
                            canned = get_fallback_message(FallbackReason.FAILED_DOC)
                            parts.append(canned)
                            yield ("token", {"token": canned})
                            final_status = MessageStatus.COMPLETE
                            is_fallback = True
                            fallback_reason = FallbackReason.FAILED_DOC.value
                        elif any(s == "processing" for s in statuses):
                            canned = get_fallback_message(FallbackReason.PROCESSING)
                            parts.append(canned)
                            yield ("token", {"token": canned})
                            final_status = MessageStatus.COMPLETE
                            is_fallback = True
                            fallback_reason = FallbackReason.PROCESSING.value
                    else:
                        doc_stmt = text(
                            "SELECT status, count(*) AS cnt FROM documents "
                            "WHERE user_id = :uid GROUP BY status"
                        )
                        doc_counts = (await doc_session.execute(doc_stmt, {"uid": user_id})).fetchall()
                        status_map = {r.status: r.cnt for r in doc_counts}
                        ready_count = status_map.get("ready", 0)
                        processing_count = status_map.get("processing", 0)
                        failed_count = status_map.get("failed", 0)

                        if ready_count == 0:
                            if processing_count > 0:
                                reason = FallbackReason.PROCESSING
                            elif failed_count > 0:
                                reason = FallbackReason.FAILED_DOC
                            else:
                                reason = FallbackReason.NO_DOCUMENTS

                            canned = get_fallback_message(reason)
                            parts.append(canned)
                            yield ("token", {"token": canned})
                            final_status = MessageStatus.COMPLETE
                            is_fallback = True
                            fallback_reason = reason.value

            if not is_fallback:
                # 3. Query Rewriting and Clean History Fetch (Step A - short separate session)
                is_summary = (intent == IntentType.SUMMARIZATION)
                search_query = content

                # Mode-dependent history strategy (D12):
                # - DOCUMENT_QA & GRAMMAR: force_db=True to guarantee strict relational pairing & fallback exclusion.
                # - CONVERSATION & VOCABULARY: force_db=False to leverage Redis sliding-window cache fast path.
                use_force_db = conv_mode in (ConversationMode.DOCUMENT_QA, ConversationMode.GRAMMAR)
                async with self._session_factory() as hist_session:
                    history = await self.get_history(
                        conversation_id,
                        exclude_user_msg_id=user_msg_id,
                        session=hist_session,
                        force_db=use_force_db,
                    )

                if not is_summary:
                    rewritten_query, triggered, timed_out = await rewrite_query(
                        query=content,
                        history=history,
                    )
                    search_query = rewritten_query
                    tracker.add_metric("rewrite_triggered", triggered)
                    tracker.add_metric("rewrite_timeout", timed_out)

                # 4. Context Preparation (Step C - short separate session)
                async with self._session_factory() as prep_session:
                    pipeline = create_pipeline(prep_session, pipeline_type="naive")
                    prepared_messages, chunks = await pipeline.prepare_context(
                        user_message=content,
                        mode=conv_mode,
                        user_id=user_id,
                        user_level=user_level,
                        conversation_history=history,
                        document_ids=document_ids,
                        db=prep_session,
                        tracker=tracker,
                        search_query=search_query,
                        is_summary=is_summary,
                    )

                # Check zero-chunk fallback for DOCUMENT_QA
                if conv_mode == ConversationMode.DOCUMENT_QA and not chunks:
                    canned = get_fallback_message(FallbackReason.NOT_FOUND)
                    parts.append(canned)
                    yield ("token", {"token": canned})
                    final_status = MessageStatus.COMPLETE
                    is_fallback = True
                    fallback_reason = FallbackReason.NOT_FOUND.value
                else:
                    if chunks:
                        yield ("sources", {"chunks": [c.to_dict() for c in chunks]})
                        tracker.add_metric("similarity_top1", round(chunks[0].similarity, 4) if chunks else None)
                        tracker.add_metric("similarity_mean", round(sum(c.similarity for c in chunks) / len(chunks), 4) if chunks else None)

                    # 5. Pure token streaming with broad 240s total upper bound — NO DB SESSION HELD
                    async with asyncio.timeout(240):
                        async for token, _ in pipeline.stream_from_messages(
                            prepared_messages, chunks, tracker=tracker
                        ):
                            parts.append(token)
                            yield ("token", {"token": token})

                    if not parts:
                        raise LLMInternalError(detail="LLM returned an empty completion.")

                    final_status = MessageStatus.COMPLETE

        except LLMException as exc:
            final_status = MessageStatus.FAILED
            logger.warning("LLM stream exception for conv=%s: %s", conversation_id, exc)
            err_payload = exc.to_client_payload()

        except TimeoutError:
            final_status = MessageStatus.FAILED
            logger.warning("LLM stream hit overall 240s timeout for conv=%s", conversation_id)
            err_payload = LLMTimeout(message="Generation exceeded maximum time limit.").to_client_payload()

        except (asyncio.CancelledError, GeneratorExit):
            final_status = MessageStatus.INTERRUPTED
            logger.info("Client disconnected/cancelled stream for conv=%s", conversation_id)
            raise

        except Exception as exc:
            final_status = MessageStatus.FAILED
            logger.exception("Unexpected error in stream conv=%s: %s", conversation_id, exc)
            err_payload = LLMInternalError().to_client_payload()

        finally:
            tracker.add_metric("is_fallback", is_fallback)
            tracker.add_metric("fallback_reason", fallback_reason)
            with anyio.CancelScope(shield=True):
                await self._finalize(
                    placeholder_id=placeholder_id,
                    parts=parts,
                    final_status=final_status,
                    chunks=chunks,
                    conversation_id=conversation_id,
                    user_content=content,
                    user_id=user_id,
                    conv_mode=conv_mode,
                    is_fallback=is_fallback,
                )
                try:
                    tracker.log_summary()
                except Exception:
                    pass

        # ── Step 4: Emit Terminal Event ──────────────────────────────────────────
        if err_payload:
            yield ("error", err_payload)
            return

        if final_status == MessageStatus.COMPLETE:
            if captured is not None and not is_fallback:
                captured["user_message"] = content
                captured["assistant_text"] = "".join(parts)
                captured["assistant_message_id"] = placeholder_id

            yield ("done", {
                "message_id": str(placeholder_id),
                "status": MessageStatus.COMPLETE,
            })

    async def send_message(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        user_level: str,
        content: str,
        client_message_id: uuid.UUID | None = None,
        audio_url: str | None = None,
        audio_duration_ms: int | None = None,
        stt_confidence: float | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[Message, list[RetrievedChunk]]:
        """Synchronous chat turn with decoupled database sessions."""
        # Session 1: Persist user message, fetch history, prepare context
        async with self._session_factory() as session:
            conv = await self.get_conversation(conversation_id, user_id, session=session)
            if not conv:
                raise ValueError("Conversation not found.")

            tracker = LatencyTracker(conversation_id, mode=conv.mode)
            user_msg = Message(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=content,
                status=MessageStatus.COMPLETE,
                client_message_id=client_message_id,
                audio_url=audio_url,
                audio_duration_ms=audio_duration_ms,
                stt_confidence=stt_confidence,
            )
            session.add(user_msg)
            await session.commit()
            await session.refresh(user_msg)

            user_msg_id = user_msg.id
            conv_mode = conv.mode

        # 1. Intent Classification
        intent = classify_intent(content)
        if intent == IntentType.GREETING:
            canned = get_fallback_message(FallbackReason.GREETING)
            async with self._session_factory() as session:
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=canned,
                    status=MessageStatus.COMPLETE,
                    is_fallback=True,
                    parent_message_id=user_msg_id,
                    sources=None,
                )
                session.add(assistant_msg)
                await session.commit()
                await session.refresh(assistant_msg)
            tracker.add_metric("is_fallback", True)
            tracker.add_metric("fallback_reason", FallbackReason.GREETING.value)
            tracker.log_summary()
            return assistant_msg, []

        # 2. Document Status Pre-check for DOCUMENT_QA
        if conv_mode == ConversationMode.DOCUMENT_QA:
            async with self._session_factory() as doc_session:
                if document_ids:
                    doc_stmt = text("SELECT status FROM documents WHERE id = ANY(:ids) AND user_id = :uid")
                    doc_rows = (await doc_session.execute(doc_stmt, {"ids": document_ids, "uid": user_id})).fetchall()
                    statuses = [r.status for r in doc_rows]
                    if any(s == "failed" for s in statuses):
                        canned = get_fallback_message(FallbackReason.FAILED_DOC)
                        reason = FallbackReason.FAILED_DOC.value
                    elif any(s == "processing" for s in statuses):
                        canned = get_fallback_message(FallbackReason.PROCESSING)
                        reason = FallbackReason.PROCESSING.value
                    else:
                        canned = None
                        reason = None
                else:
                    doc_stmt = text(
                        "SELECT status, count(*) AS cnt FROM documents "
                        "WHERE user_id = :uid GROUP BY status"
                    )
                    doc_counts = (await doc_session.execute(doc_stmt, {"uid": user_id})).fetchall()
                    status_map = {r.status: r.cnt for r in doc_counts}
                    ready_count = status_map.get("ready", 0)
                    processing_count = status_map.get("processing", 0)
                    failed_count = status_map.get("failed", 0)

                    if ready_count == 0:
                        if processing_count > 0:
                            reason_enum = FallbackReason.PROCESSING
                        elif failed_count > 0:
                            reason_enum = FallbackReason.FAILED_DOC
                        else:
                            reason_enum = FallbackReason.NO_DOCUMENTS
                        canned = get_fallback_message(reason_enum)
                        reason = reason_enum.value
                    else:
                        canned = None
                        reason = None

                if canned is not None:
                    async with self._session_factory() as session:
                        assistant_msg = Message(
                            conversation_id=conversation_id,
                            role=MessageRole.ASSISTANT,
                            content=canned,
                            status=MessageStatus.COMPLETE,
                            is_fallback=True,
                            parent_message_id=user_msg_id,
                            sources=None,
                        )
                        session.add(assistant_msg)
                        await session.commit()
                        await session.refresh(assistant_msg)
                    tracker.add_metric("is_fallback", True)
                    tracker.add_metric("fallback_reason", reason)
                    tracker.log_summary()
                    return assistant_msg, []

        # 3. Query Rewriting and History Fetch (Step A)
        is_summary = (intent == IntentType.SUMMARIZATION)
        search_query = content

        # Mode-dependent history strategy (D12):
        use_force_db = conv_mode in (ConversationMode.DOCUMENT_QA, ConversationMode.GRAMMAR)
        async with self._session_factory() as hist_session:
            history = await self.get_history(
                conversation_id,
                exclude_user_msg_id=user_msg_id,
                session=hist_session,
                force_db=use_force_db,
            )

        if not is_summary:
            rewritten_query, triggered, timed_out = await rewrite_query(
                query=content,
                history=history,
            )
            search_query = rewritten_query
            tracker.add_metric("rewrite_triggered", triggered)
            tracker.add_metric("rewrite_timeout", timed_out)

        # 4. Context Preparation (Step C)
        async with self._session_factory() as prep_session:
            pipeline = create_pipeline(prep_session, pipeline_type="naive")
            messages, chunks = await pipeline.prepare_context(
                user_message=content,
                mode=conv_mode,
                user_id=user_id,
                user_level=user_level,
                conversation_history=history,
                document_ids=document_ids,
                db=prep_session,
                tracker=tracker,
                search_query=search_query,
                is_summary=is_summary,
            )

        # Check zero-chunk fallback for DOCUMENT_QA
        if conv_mode == ConversationMode.DOCUMENT_QA and not chunks:
            canned = get_fallback_message(FallbackReason.NOT_FOUND)
            async with self._session_factory() as session:
                assistant_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=canned,
                    status=MessageStatus.COMPLETE,
                    is_fallback=True,
                    parent_message_id=user_msg_id,
                    sources=None,
                )
                session.add(assistant_msg)
                await session.commit()
                await session.refresh(assistant_msg)
            tracker.add_metric("is_fallback", True)
            tracker.add_metric("fallback_reason", FallbackReason.NOT_FOUND.value)
            tracker.log_summary()
            return assistant_msg, []

        if chunks:
            tracker.add_metric("similarity_top1", round(chunks[0].similarity, 4) if chunks else None)
            tracker.add_metric("similarity_mean", round(sum(c.similarity for c in chunks) / len(chunks), 4) if chunks else None)

        # 5. Token generation strictly outside database session
        full_tokens: list[str] = []
        async for token, _ in pipeline.stream_from_messages(messages, chunks, tracker=tracker):
            full_tokens.append(token)
        response_text = "".join(full_tokens)

        # 6. Session 2: Persist assistant response and progress
        async with self._session_factory() as session:
            assistant_msg = Message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=response_text,
                status=MessageStatus.COMPLETE,
                is_fallback=False,
                parent_message_id=user_msg_id,
                sources=[c.to_storage_dict() for c in chunks] if chunks else None,
            )
            session.add(assistant_msg)
            await self.update_progress(session, user_id, conv_mode)
            await session.commit()
            await session.refresh(assistant_msg)

        try:
            await self.sync_redis_turn(conversation_id, content, response_text)
        except Exception as exc:
            logger.warning("Failed to sync Redis turn for conv %s: %s", conversation_id, exc)

        tracker.add_metric("is_fallback", False)
        tracker.log_summary()
        return assistant_msg, chunks
