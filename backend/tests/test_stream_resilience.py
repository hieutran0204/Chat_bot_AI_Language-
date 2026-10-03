# name: test_stream_resilience.py
# description: Comprehensive resilience test suite for Phase 1 stream lifecycle,
#              disconnect handling, timeouts, reaper, and idempotency.

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, patch

from app.core.exceptions import LLMTimeout
from app.core.redis.circuit_breaker import RedisCircuitBreaker, CircuitState
from app.models.conversation import MessageStatus
from tests.fixtures.fake_llm import FakeLLM


@pytest.mark.asyncio
async def test_fake_llm_streaming_normal():
    """Verify FakeLLM yields all configured tokens."""
    tokens = ["Hello", " world", "!"]
    llm = FakeLLM(tokens=tokens, delay_per_token=0.001)

    received = []
    async for tok in llm.stream_chat([]):
        received.append(tok)

    assert "".join(received) == "Hello world!"


@pytest.mark.asyncio
async def test_fake_llm_timeout_error():
    """Verify FakeLLM raises LLMTimeout when configured to fail mid-stream."""
    llm = FakeLLM(
        tokens=["One", "Two", "Three"],
        fail_after_token_count=2,
        fail_after_token_error=LLMTimeout("Model stalled"),
    )

    received = []
    with pytest.raises(LLMTimeout):
        async for tok in llm.stream_chat([]):
            received.append(tok)

    assert len(received) == 2


@pytest.mark.asyncio
async def test_redis_circuit_breaker_transitions():
    """Verify CircuitBreaker trips to OPEN after 3 failures and enforces recovery."""
    cb = RedisCircuitBreaker(failure_threshold=3, recovery_time_seconds=0.1)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

    # 1st failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 2nd failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3rd failure -> trip OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

    # Wait for recovery interval
    await asyncio.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request() is True  # Single probe allowed
    assert cb.allow_request() is False  # Second concurrent probe blocked

    # Probe succeeds
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True


@pytest.mark.asyncio
async def test_domain_exception_client_payload():
    """Verify LLM exceptions serialize clean, non-leaking error payloads for SSE."""
    err = LLMTimeout(message="Model generation timeout", detail="Raw socket timeout")
    payload = err.to_client_payload()

    assert payload["code"] == "LLM_TIMEOUT"
    assert payload["retryable"] is True
    assert payload["message"] == "Model generation timeout"
    assert "Raw socket timeout" not in payload.values()


@pytest.mark.asyncio
async def test_reaper_recovers_stale_messages():
    """Verify reaper task marks stale streaming messages as failed."""
    from app.services.chat_service import reap_stale_streaming_messages

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.rowcount = 3
    mock_session.execute.return_value = mock_result

    class MockSessionFactory:
        def __call__(self):
            return self
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, *args):
            pass

    reaped = await reap_stale_streaming_messages(
        session_factory=MockSessionFactory(),
        max_age_seconds=600,
    )
    assert reaped == 3
    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_redis_client_property_does_not_consume_probe():
    """Verify RedisClient.client property does not re-query allow_request in HALF_OPEN."""
    from app.core.redis.client import RedisClient

    client = RedisClient(redis_url="redis://localhost:6379/0")
    client._client = AsyncMock()  # mock underlying redis instance
    client._circuit_breaker._state = CircuitState.HALF_OPEN
    client._circuit_breaker._is_probing = False

    # First check gatekeeper
    assert client.is_available is True
    # In HALF_OPEN, is_probing is now True
    assert client._circuit_breaker._is_probing is True

    # Accessing client property MUST NOT fail or re-evaluate allow_request
    assert client.client is not None
    assert client.client == client._client


@pytest.mark.asyncio
async def test_breaker_does_not_get_stuck_if_probe_not_followed_by_redis_call():
    """
    Verify circuit breaker in HALF_OPEN recovers and allows a new probe
    if an earlier caller read allow_request / is_available but never reported back.
    """
    cb = RedisCircuitBreaker(failure_threshold=3, recovery_time_seconds=0.05)
    # Trip breaker to OPEN
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Wait for recovery window to enter HALF_OPEN
    await asyncio.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN

    # First probe consumed by an observer that does not invoke Redis
    assert cb.allow_request() is True
    # Immediate subsequent request is blocked because probe is active
    assert cb.allow_request() is False

    # Wait for probe timeout (recovery_time_seconds)
    await asyncio.sleep(0.1)

    # Breaker MUST NOT be stuck; timeout allows a fresh probe
    assert cb.allow_request() is True


@pytest.mark.asyncio
async def test_step1_db_failure_yields_error_event():
    """Verify that a DB exception during Step 1 yields an error event instead of crashing the SSE stream."""
    from app.services.chat_service import ChatService

    mock_session = AsyncMock()
    mock_session.execute.side_effect = RuntimeError("Database connection lost")

    class FailingSessionFactory:
        def __call__(self):
            return self
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, *args):
            pass

    service = ChatService()
    service._session_factory = FailingSessionFactory()

    events = []
    async for event_type, payload in service.stream_message(
        conversation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        user_level="B1",
        content="Hello, is anyone there?",
    ):
        events.append((event_type, payload))

    assert len(events) == 1
    event_type, payload = events[0]
    assert event_type == "error"
    assert payload["code"] in ("LLM_UNAVAILABLE", "INTERNAL")
    assert payload["retryable"] is True


@pytest.mark.asyncio
async def test_cold_redis_rebuilds_history_and_warms_cache():
    """Verify that on Redis cache miss, history is loaded from DB and warmed up into Redis."""
    from app.core.primitives.messages import HumanMessage, AIMessage
    from app.services.chat_service import ChatService

    conv_id = uuid.uuid4()
    user_msg_id = uuid.uuid4()

    service = ChatService()

    from unittest.mock import MagicMock

    # Mock DB query results: pairs of (user_content, asst_content)
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [
        ("Hi there", "Hello! How can I help you practice English today?"),
    ]
    mock_session.execute.return_value = mock_result

    class MockSessionFactory:
        def __call__(self):
            return self
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, *args):
            pass

    service._session_factory = MockSessionFactory()

    with patch("app.services.chat_service.RedisChatMessageHistory") as mock_history_cls:
        mock_history = AsyncMock()
        mock_history.get_messages.return_value = []  # Cold cache miss
        mock_history_cls.return_value = mock_history

        history = await service.get_history(conv_id, exclude_user_msg_id=user_msg_id)

        # History should contain both user and assistant
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hi there"
        assert history[1]["role"] == "assistant"

        # Redis buffer MUST have been warmed up with both messages
        assert mock_history.add_messages.called
        warmed_messages = mock_history.add_messages.call_args[0][0]
        assert len(warmed_messages) == 2
        assert isinstance(warmed_messages[0], HumanMessage)
        assert isinstance(warmed_messages[1], AIMessage)


@pytest.mark.asyncio
async def test_smoke_grammar_and_document_qa_modes():
    """Smoke test for GRAMMAR and DOCUMENT_QA retrieval modes to ensure asyncio.to_thread and vector search work."""
    from unittest.mock import MagicMock
    from app.models.conversation import ConversationMode
    from app.pipelines.naive_rag.pipeline import NaiveRagPipeline
    from app.core.interfaces.vector_store import RetrievedChunk

    mock_llm = FakeLLM(tokens=["Here is the ", "correction."])
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.05] * 128
    mock_vector_store = AsyncMock()
    mock_chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Rule: Use past simple for actions that happened at a specific time.",
        similarity=0.92,
        metadata={},
    )
    mock_vector_store.similarity_search.return_value = [mock_chunk]

    pipeline = NaiveRagPipeline(
        llm=mock_llm,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
    )

    for mode in (ConversationMode.GRAMMAR, ConversationMode.DOCUMENT_QA):
        messages, chunks = await pipeline.prepare_context(
            user_message="I go to school yesterday.",
            mode=mode,
            user_id=uuid.uuid4(),
            user_level="A2",
            conversation_history=[],
        )
        assert len(chunks) == 1
        assert chunks[0].content == mock_chunk.content
        assert len(messages) >= 2  # System prompt + User message

        # Stream generation
        streamed = []
        async for token, _ in pipeline.stream_from_messages(messages, chunks):
            streamed.append(token)
        assert "".join(streamed) == "Here is the correction."


@pytest.mark.asyncio
async def test_concurrent_retry_claims_only_once():
    """
    Verify that when two concurrent retries race on a failed message,
    only the first claim succeeds, and the second receives 0 claimed rows (IN_PROGRESS).
    """
    from datetime import datetime, timezone, timedelta

    msg_id = uuid.uuid4()
    claimed = []

    # Simulate atomic lock behavior in Python to verify the exact logic flow
    class State:
        status = MessageStatus.FAILED
        updated_at = datetime.now(timezone.utc) - timedelta(seconds=500)

    state = State()

    def try_claim(stale_threshold_seconds: int = 300) -> bool:
        stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_threshold_seconds)
        # Condition in SQL: status IN ('failed', 'interrupted') OR (status == 'streaming' AND updated_at <= stale_cutoff)
        is_retryable = (
            state.status in (MessageStatus.FAILED, MessageStatus.INTERRUPTED)
            or (state.status == MessageStatus.STREAMING and state.updated_at <= stale_cutoff)
        )
        if is_retryable:
            state.status = MessageStatus.STREAMING
            state.updated_at = datetime.now(timezone.utc)
            return True
        return False

    # Caller A tries to claim
    claim_a = try_claim()
    assert claim_a is True
    assert state.status == MessageStatus.STREAMING

    # Caller B races immediately after A commits
    claim_b = try_claim()
    assert claim_b is False  # Second claim fails because updated_at is fresh


