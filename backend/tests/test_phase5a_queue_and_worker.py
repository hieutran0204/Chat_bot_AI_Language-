# name: test_phase5a_queue_and_worker.py
# description: Comprehensive test suite for Phase 5a: arq tasks, idle scanner, semaphore backpressure,
#              get_short_session auto-commit, error taxonomy, and batch extraction idempotency.

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
import json
import uuid
import pytest
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import ValidationError
from arq.jobs import Retry

from app.core.config import settings
from app.core.database import AsyncSessionFactory, get_short_session
from app.core.primitives.messages import Correction
from app.models.conversation import Conversation, ConversationMode, Message, MessageRole, MessageStatus
from app.models.learner_analytics import ExtractionDeadLetter, SessionInsight, UserWeaknessLog
from app.services.extraction_service import BatchCorrectionItem, ExtractionResult, ExtractionService
from app.services.session_analysis_service import SessionInsightData
from app.worker.tasks import (
    PERMANENT_EXCEPTIONS,
    TRANSIENT_EXCEPTIONS,
    process_conversation_job,
    scan_idle_conversations,
)


@pytest.mark.asyncio
async def test_get_short_session_auto_commit_and_rollback():
    """Verify get_short_session automatically commits on success and rolls back on exception."""
    test_conv_id = uuid.uuid4()
    test_user_id = uuid.uuid4()

    # 1. Success block -> auto-commit
    async with get_short_session() as s:
        conv = Conversation(
            id=test_conv_id,
            user_id=test_user_id,
            mode=ConversationMode.CONVERSATION,
            title="AutoCommit Test",
        )
        s.add(conv)

    # Verify persisted in a new independent session
    async with get_short_session() as s:
        fetched = await s.get(Conversation, test_conv_id)
        assert fetched is not None
        assert fetched.title == "AutoCommit Test"

    # 2. Failure block -> rollback
    with pytest.raises(RuntimeError):
        async with get_short_session() as s:
            conv = await s.get(Conversation, test_conv_id)
            conv.title = "Failed Update"
            raise RuntimeError("Simulated failure inside transaction")

    # Verify title was not updated
    async with get_short_session() as s:
        fetched = await s.get(Conversation, test_conv_id)
        assert fetched.title == "AutoCommit Test"

        # Cleanup
        await s.delete(fetched)


@pytest.mark.asyncio
async def test_semaphore_release_once_double_invocation_safe():
    """Verify release_once closure only releases the semaphore once, even if called multiple times."""
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    assert sem._value == 0

    released = False

    async def release_once() -> None:
        nonlocal released
        if not released:
            released = True
            sem.release()

    # Call from generator finally
    await release_once()
    assert sem._value == 1

    # Call again from BackgroundTask
    await release_once()
    assert sem._value == 1  # Still 1, not leaked to 2


@pytest.mark.asyncio
async def test_exception_classification_taxonomy():
    """Verify TRANSIENT and PERMANENT exception classification."""
    # Transient errors
    assert issubclass(asyncio.TimeoutError, TRANSIENT_EXCEPTIONS)
    assert issubclass(httpx.TransportError, TRANSIENT_EXCEPTIONS)
    assert issubclass(httpx.TimeoutException, TRANSIENT_EXCEPTIONS)

    # Permanent errors
    assert issubclass(json.JSONDecodeError, PERMANENT_EXCEPTIONS)
    assert issubclass(ValidationError, PERMANENT_EXCEPTIONS)
    assert issubclass(ValueError, PERMANENT_EXCEPTIONS)

    # Retry must not be caught as transient or permanent
    assert not issubclass(Retry, TRANSIENT_EXCEPTIONS)
    assert not issubclass(Retry, PERMANENT_EXCEPTIONS)


@pytest.mark.asyncio
async def test_process_conversation_job_deleted_conv_graceful_exit():
    """Verify job safely exits when conversation is deleted before job starts."""
    non_existent_id = str(uuid.uuid4())
    ctx = {"job_try": 1, "redis": AsyncMock()}

    # Should execute and return gracefully without raising AttributeError
    await process_conversation_job(ctx, non_existent_id)


@pytest.mark.asyncio
async def test_process_conversation_job_e2e_idempotency_and_insights():
    """
    End-to-End test on database:
    - Prepares Conversation and Message rows.
    - Runs process_conversation_job with mocked LLM extraction.
    - Verifies extraction_status='done', message_id in UserWeaknessLog, and SessionInsight.
    - Runs a second time to verify idempotency (no duplicate weakness logs).
    """
    test_user_id = uuid.uuid4()
    test_conv_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Setup Conversation + 2 user messages + 1 assistant message
    async with get_short_session() as s:
        conv = Conversation(
            id=test_conv_id,
            user_id=test_user_id,
            mode=ConversationMode.CONVERSATION,
            created_at=now - timedelta(minutes=40),
            last_message_at=now - timedelta(minutes=35),
            process_attempts=0,
        )
        s.add(conv)

        m1 = Message(
            conversation_id=test_conv_id,
            role=MessageRole.USER,
            content="I go to school yesterday.",
            created_at=now - timedelta(minutes=39),
            status=MessageStatus.COMPLETE,
            extraction_status="pending",
        )
        m2 = Message(
            conversation_id=test_conv_id,
            role=MessageRole.ASSISTANT,
            content="Nice, what did you do at school?",
            created_at=now - timedelta(minutes=38),
            status=MessageStatus.COMPLETE,
            extraction_status="pending",
        )
        m3 = Message(
            conversation_id=test_conv_id,
            role=MessageRole.USER,
            content="I play football with friend.",
            created_at=now - timedelta(minutes=36),
            status=MessageStatus.COMPLETE,
            extraction_status="pending",
        )
        s.add_all([m1, m2, m3])

    fake_extraction_result = ExtractionResult(
        errors=[
            BatchCorrectionItem(
                chunk_index=0,
                type="tense_past_simple",
                original="go",
                suggestion="went",
                explanation="Dùng thì quá khứ đơn 'went'.",
            ),
            BatchCorrectionItem(
                chunk_index=1,
                type="tense_past_simple",
                original="play",
                suggestion="played",
                explanation="Dùng thì quá khứ đơn 'played'.",
            ),
        ]
    )

    fake_insight = SessionInsightData(
        encouragements=["Great effort expressing your activities!"],
        reminders=["Remember to use past tense verbs when talking about yesterday."],
        focus_next="tense_past_simple",
        session_score=85,
        streak_days=1,
    )

    ctx = {"job_try": 1, "redis": AsyncMock()}

    with patch.object(ExtractionService, "extract_batch", new_callable=AsyncMock) as mock_extract, \
         patch.object(ExtractionService, "invalidate_user_cache", new_callable=AsyncMock) as mock_inval, \
         patch("app.services.session_analysis_service.SessionAnalysisService.generate_session_insight", new_callable=AsyncMock) as mock_insight:

        mock_extract.return_value = fake_extraction_result
        mock_insight.return_value = fake_insight

        # Run 1st execution
        await process_conversation_job(ctx, str(test_conv_id))

        # Verification after Run 1
        async with get_short_session() as s:
            # Check messages status
            msgs = (await s.execute(
                select(Message).where(Message.conversation_id == test_conv_id).order_by(Message.created_at.asc())
            )).scalars().all()

            for msg in msgs:
                assert msg.extraction_status == "done"
                assert msg.extraction_error is None

            # User messages must have corrections populated
            assert msgs[0].corrections is not None
            assert len(msgs[0].corrections) == 1
            assert msgs[0].corrections[0]["type"] == "tense_past_simple"

            # Check UserWeaknessLog
            w_logs = (await s.execute(
                select(UserWeaknessLog).where(UserWeaknessLog.conversation_id == test_conv_id)
            )).scalars().all()
            assert len(w_logs) == 2
            assert all(w.message_id is not None for w in w_logs)

            # Check Conversation state
            updated_conv = await s.get(Conversation, test_conv_id)
            assert updated_conv.process_attempts == 0
            assert updated_conv.last_failed_at is None
            assert updated_conv.insight_generated_at is not None

        # Run 2nd execution (Idempotency check: simulating re-run or replay)
        await process_conversation_job(ctx, str(test_conv_id))

        # Verification after Run 2: ensure no duplicate rows created
        async with get_short_session() as s:
            w_logs_after = (await s.execute(
                select(UserWeaknessLog).where(UserWeaknessLog.conversation_id == test_conv_id)
            )).scalars().all()
            assert len(w_logs_after) == 2  # Idempotent: exactly 2 rows, not 4

            # Cleanup test data
            await s.execute(delete(SessionInsight).where(SessionInsight.conversation_id == test_conv_id))
            await s.execute(delete(UserWeaknessLog).where(UserWeaknessLog.conversation_id == test_conv_id))
            await s.execute(delete(Message).where(Message.conversation_id == test_conv_id))
            await s.execute(delete(Conversation).where(Conversation.id == test_conv_id))


@pytest.mark.asyncio
async def test_scan_idle_conversations_cron():
    """Verify scan_idle_conversations only enqueues eligible conversations with job_id."""
    test_user_id = uuid.uuid4()
    test_conv_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    # Setup an idle conversation (> 30 mins) with unextracted state
    async with get_short_session() as s:
        conv = Conversation(
            id=test_conv_id,
            user_id=test_user_id,
            mode=ConversationMode.CONVERSATION,
            created_at=now - timedelta(minutes=60),
            last_message_at=now - timedelta(minutes=45),
            insight_generated_at=None,
            process_attempts=0,
        )
        s.add(conv)

    mock_redis = AsyncMock()
    ctx = {"redis": mock_redis}

    await scan_idle_conversations(ctx)

    assert mock_redis.enqueue_job.called
    called_args, called_kwargs = mock_redis.enqueue_job.call_args
    assert called_args[0] == "process_conversation_job"
    assert called_args[1] == str(test_conv_id)
    assert called_kwargs["_job_id"].startswith(f"proc_conv:{test_conv_id}:")

    # Cleanup
    async with get_short_session() as s:
        c = await s.get(Conversation, test_conv_id)
        if c:
            await s.delete(c)
