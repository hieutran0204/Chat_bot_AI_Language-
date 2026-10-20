# name: tasks.py
# description: arq worker tasks: batch extraction, session insight generation, and idle conversation cron.

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import json
import logging
import uuid
import httpx
from pydantic import ValidationError
from arq.jobs import Retry
from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings
from app.core.database import get_short_session
from app.models.conversation import Conversation, Message
from app.models.learner_analytics import ExtractionDeadLetter, SessionInsight, UserWeaknessLog
from app.services.extraction_service import ExtractionService
from app.services.session_analysis_service import SessionAnalysisService, SessionInsightData

logger = logging.getLogger("arq.worker")

TRANSIENT_EXCEPTIONS = (
    asyncio.TimeoutError,
    httpx.TransportError,
    httpx.TimeoutException,
)
PERMANENT_EXCEPTIONS = (
    ValidationError,
    json.JSONDecodeError,
    ValueError,
)


async def process_conversation_job(ctx: dict, conversation_id: str) -> None:
    """
    Background job to batch extract learner errors and generate session insight.

    Lifecycle:
    - Increments process_attempts only on outer attempt (job_try == 1).
    - Resets process_attempts to 0 upon any successful chunk processing.
    - Uses snapshot_ts to avoid race condition with ongoing user messages.
    - Idempotent: deletes existing UserWeaknessLog by message_id before inserting.
    - Safe exception handling: re-raises Retry, logs unhandled errors into last_failed_at.
    """
    job_try: int = ctx.get("job_try", 1)
    redis = ctx.get("redis")
    conv_uuid = uuid.UUID(conversation_id)

    # ── 0. Increment attempts on outer attempt ────────────────────────────────
    if job_try == 1:
        async with get_short_session() as s:
            await s.execute(
                update(Conversation)
                .where(Conversation.id == conv_uuid)
                .values(process_attempts=Conversation.process_attempts + 1)
            )

    # ── 1. Snapshot Conversation (None-safe) ──────────────────────────────────
    async with get_short_session() as s:
        conv = await s.get(Conversation, conv_uuid)

    if conv is None:
        logger.warning("conversation_deleted_before_run", conv_id=str(conv_uuid))
        return

    if conv.last_message_at is None:
        logger.info("conversation_empty_skip", conv_id=str(conv_uuid))
        return

    snapshot_ts: datetime = conv.last_message_at
    session_start: datetime = conv.insight_generated_at or conv.created_at
    user_id: uuid.UUID = conv.user_id
    deadline = datetime.now(timezone.utc) + timedelta(seconds=settings.job_budget_seconds)
    chunk_plus_insight_s = settings.llm_extraction_timeout + settings.llm_insight_timeout

    made_progress = False
    extraction_service = ExtractionService()

    try:
        # ── 2. Chunk loop ─────────────────────────────────────────────────────
        while True:
            remaining = deadline - datetime.now(timezone.utc)
            if remaining < timedelta(seconds=chunk_plus_insight_s):
                logger.info("budget_exhausted_exit_early", conv_id=str(conv_uuid))
                if made_progress:
                    await _reset_process_attempts(conv_uuid)
                return

            async with get_short_session() as s:
                rows = await s.execute(
                    select(Message)
                    .where(Message.conversation_id == conv_uuid)
                    .where(Message.extraction_status == "pending")
                    .where(Message.created_at <= snapshot_ts)
                    .order_by(Message.created_at.asc())
                    .limit(settings.extraction_batch_size)
                )
                chunk = rows.scalars().all()

            if not chunk:
                break

            chunk_ids = [m.id for m in chunk]
            user_messages = [m for m in chunk if m.role == "user"]

            # If chunk contains no user messages (assistant only), mark done & skip LLM
            if not user_messages:
                async with get_short_session() as s:
                    await s.execute(
                        update(Message).where(Message.id.in_(chunk_ids))
                        .values(extraction_status="done")
                    )
                made_progress = True
                continue

            # Fetch context before this chunk
            async with get_short_session() as s:
                ctx_rows = await s.execute(
                    select(Message)
                    .where(Message.conversation_id == conv_uuid)
                    .where(Message.created_at < chunk[0].created_at)
                    .order_by(Message.created_at.desc())
                    .limit(settings.extraction_context_window)
                )
                context = list(reversed(ctx_rows.scalars().all()))

            # Call LLM batch extraction
            try:
                results = await asyncio.wait_for(
                    extraction_service.extract_batch(context=context, user_messages=user_messages),
                    timeout=settings.llm_extraction_timeout,
                )
            except TRANSIENT_EXCEPTIONS as exc:
                if redis:
                    key = f"extraction:errors:transient:{date.today().isoformat()}"
                    await redis.incr(key)
                    await redis.expire(key, 86400 * 30)

                if job_try < settings.arq_max_tries:
                    raise Retry(defer=[5, 30, 120][min(job_try - 1, 2)])

                async with get_short_session() as s:
                    await s.execute(
                        update(Conversation).where(Conversation.id == conv_uuid)
                        .values(last_failed_at=datetime.now(timezone.utc))
                    )
                logger.warning("extraction_transient_exhausted", conv_id=str(conv_uuid), error=str(exc))
                return

            except PERMANENT_EXCEPTIONS as exc:
                if redis:
                    key = f"extraction:errors:permanent:{date.today().isoformat()}"
                    await redis.incr(key)
                    await redis.expire(key, 86400 * 30)

                async with get_short_session() as s:
                    await s.execute(
                        update(Message).where(Message.id.in_(chunk_ids))
                        .values(extraction_status="failed", extraction_error=str(exc))
                    )
                    s.add(ExtractionDeadLetter(
                        conversation_id=conv_uuid,
                        message_ids=chunk_ids,
                        error_type="permanent_extraction",
                        error_message=str(exc),
                    ))
                logger.error("extraction_permanent_error", conv_id=str(conv_uuid), error=str(exc))
                continue

            # Map errors to UserWeaknessLog and Message.corrections
            weakness_log_rows = [
                UserWeaknessLog(
                    user_id=user_id,
                    conversation_id=conv_uuid,
                    message_id=user_messages[item.chunk_index].id,
                    weakness_type=item.type,
                    original=item.original,
                    suggestion=item.suggestion,
                    explanation=item.explanation,
                    is_repeated=False,
                )
                for item in results.errors
                if 0 <= item.chunk_index < len(user_messages)
            ]

            corrections_by_msg_id: dict[uuid.UUID, list] = {m.id: [] for m in chunk}
            for item in results.errors:
                if 0 <= item.chunk_index < len(user_messages):
                    m_id = user_messages[item.chunk_index].id
                    corrections_by_msg_id[m_id].append(item.model_dump(mode="json"))

            async with get_short_session() as s:
                await s.execute(
                    delete(UserWeaknessLog).where(UserWeaknessLog.message_id.in_(chunk_ids))
                )
                if weakness_log_rows:
                    s.add_all(weakness_log_rows)

                for m in chunk:
                    m_corr = corrections_by_msg_id.get(m.id)
                    await s.execute(
                        update(Message).where(Message.id == m.id)
                        .values(
                            corrections=m_corr if m_corr else None,
                            extraction_status="done",
                            extraction_error=None,
                        )
                    )
            made_progress = True

            if results.errors:
                await extraction_service.invalidate_user_cache(user_id)

        # ── 3. Reset process_attempts on progress ─────────────────────────────
        if made_progress:
            await _reset_process_attempts(conv_uuid)

        # ── 4. Check min messages for insight ─────────────────────────────────
        async with get_short_session() as s:
            user_msg_rows = await s.execute(
                select(Message.content)
                .where(Message.conversation_id == conv_uuid)
                .where(Message.role == "user")
                .where(Message.created_at > session_start)
                .where(Message.created_at <= snapshot_ts)
            )
            user_contents = user_msg_rows.scalars().all()

        if len(user_contents) < settings.min_messages_for_insight:
            await _commit_insight_generated(conv_uuid, user_id, snapshot_ts, None, session_start)
            logger.info("insight_skipped_too_short", conv_id=str(conv_uuid))
            return

        async with get_short_session() as s:
            weakness_rows = await s.execute(
                select(UserWeaknessLog.weakness_type)
                .where(UserWeaknessLog.conversation_id == conv_uuid)
                .where(UserWeaknessLog.created_at >= session_start)
                .where(UserWeaknessLog.created_at <= snapshot_ts)
            )
            weakness_types_found = list(set(weakness_rows.scalars().all()))

        # ── 5. Generate Session Insight ───────────────────────────────────────
        insight_service = SessionAnalysisService()
        insight_data: SessionInsightData | None = None
        try:
            insight_data = await asyncio.wait_for(
                insight_service.generate_session_insight(
                    conversation_id=conv_uuid,
                    user_id=user_id,
                    session_start=session_start,
                    session_end=snapshot_ts,
                    user_messages=user_contents,
                    weakness_types_found=weakness_types_found,
                ),
                timeout=settings.llm_insight_timeout,
            )
        except TRANSIENT_EXCEPTIONS as exc:
            async with get_short_session() as s:
                await s.execute(
                    update(Conversation).where(Conversation.id == conv_uuid)
                    .values(last_failed_at=datetime.now(timezone.utc))
                )
            logger.warning("insight_transient_error", conv_id=str(conv_uuid), error=str(exc))
            return

        except PERMANENT_EXCEPTIONS as exc:
            async with get_short_session() as s:
                s.add(ExtractionDeadLetter(
                    conversation_id=conv_uuid,
                    message_ids=[],
                    error_type="permanent_insight",
                    error_message=str(exc),
                ))
            logger.error("insight_permanent_error", conv_id=str(conv_uuid), error=str(exc))
            insight_data = None

        # ── 6. Commit final state ─────────────────────────────────────────────
        await _commit_insight_generated(
            conv_uuid, user_id, snapshot_ts, insight_data, session_start
        )

    except Retry:
        raise
    except Exception as exc:
        async with get_short_session() as s:
            await s.execute(
                update(Conversation).where(Conversation.id == conv_uuid)
                .values(last_failed_at=datetime.now(timezone.utc))
            )
        logger.exception("unhandled_job_exception", conv_id=str(conv_uuid), error=str(exc))
        raise


async def _reset_process_attempts(conversation_id: uuid.UUID) -> None:
    """Reset attempt counter and clear failure marker for a conversation."""
    async with get_short_session() as s:
        await s.execute(
            update(Conversation).where(Conversation.id == conversation_id)
            .values(process_attempts=0, last_failed_at=None)
        )


async def _commit_insight_generated(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    snapshot_ts: datetime,
    insight_data: SessionInsightData | None,
    session_start: datetime | None = None,
) -> None:
    """Persist generated session insight card and advance conversation insight marker."""
    async with get_short_session() as s:
        if insight_data and session_start:
            await s.execute(
                pg_insert(SessionInsight).values(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    session_start_at=session_start,
                    session_end_at=snapshot_ts,
                    encouragements=insight_data.encouragements,
                    reminders=insight_data.reminders,
                    focus_next=insight_data.focus_next,
                    session_score=insight_data.session_score,
                    streak_days=insight_data.streak_days,
                ).on_conflict_do_nothing(
                    index_elements=["conversation_id", "session_start_at"]
                )
            )
        await s.execute(
            update(Conversation).where(Conversation.id == conversation_id)
            .values(
                insight_generated_at=snapshot_ts,
                process_attempts=0,
                last_failed_at=None,
            )
        )


async def scan_idle_conversations(ctx: dict) -> None:
    """
    Cron scanner executed periodically by arq.

    Selects conversations that have been idle past CONVERSATION_IDLE_MINUTES,
    have unextracted messages, and are not currently backing off from errors.
    """
    now = datetime.now(timezone.utc)
    idle_cutoff = now - timedelta(minutes=settings.conversation_idle_minutes)
    backoff_cutoff = now - timedelta(minutes=settings.retry_backoff_minutes)

    async with get_short_session() as s:
        rows = await s.execute(
            select(Conversation.id, Conversation.last_message_at)
            .where(Conversation.last_message_at < idle_cutoff)
            .where(
                or_(
                    Conversation.insight_generated_at.is_(None),
                    Conversation.insight_generated_at < Conversation.last_message_at,
                )
            )
            .where(Conversation.process_attempts < settings.max_process_attempts)
            .where(
                or_(
                    Conversation.last_failed_at.is_(None),
                    Conversation.last_failed_at < backoff_cutoff,
                )
            )
            .order_by(Conversation.last_message_at.asc())
            .limit(20)
        )
        conversations_to_queue = rows.all()

    redis = ctx.get("redis")
    if not redis:
        logger.error("Redis client missing in scan_idle_conversations ctx")
        return

    for conv_id, last_msg_at in conversations_to_queue:
        if last_msg_at is None:
            continue
        job_id = f"proc_conv:{conv_id}:{int(last_msg_at.timestamp())}"
        await redis.enqueue_job(
            "process_conversation_job",
            str(conv_id),
            _job_id=job_id,
        )
        logger.info("enqueued_idle_conversation", conv_id=str(conv_id), job_id=job_id)
