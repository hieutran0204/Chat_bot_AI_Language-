# name: chat_service.py
# description: Business logic for conversation management and message persistence.

import logging
import uuid
from datetime import UTC, date, datetime

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.primitives.messages import AIMessage, Correction, HumanMessage
from app.core.redis.memory import RedisChatMessageHistory
from app.core.redis.profile import RedisLearnerProfileManager
from app.models.conversation import Conversation, ConversationMode, Message, MessageRole
from app.models.progress import UserProgress
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class ChatService:
    """
    Manages conversations and messages for the chat API.

    Responsibilities:
    - Create and retrieve conversations.
    - Persist messages to PostgreSQL and Redis sliding buffer.
    - Track learner profile weaknesses in Redis long-term cache.
    - Update daily learning progress counters.
    - Build low-latency chat history for LLM context window.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._profile_mgr = RedisLearnerProfileManager()

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        mode: str = ConversationMode.CONVERSATION,
        title: str | None = None,
    ) -> Conversation:
        """
        Create a new conversation session.

        Args:
            user_id: UUID of the owning user.
            mode: ConversationMode value (conversation, grammar, vocabulary, document_qa).
            title: Optional human-readable title (auto-set from first message if None).

        Returns:
            Created Conversation ORM instance.
        """
        conv = Conversation(user_id=user_id, mode=mode, title=title)
        self._db.add(conv)
        await self._db.flush()
        logger.info("Created conversation %s mode=%s user=%s", conv.id, mode, user_id)
        return conv

    async def get_conversation(
        self, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> Conversation | None:
        """
        Fetch a conversation, verifying it belongs to the user.

        Args:
            conversation_id: UUID of the conversation.
            user_id: UUID of the requesting user.

        Returns:
            Conversation ORM instance or None.
        """
        result = await self._db.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_conversations(self, user_id: uuid.UUID) -> list[Conversation]:
        """
        List all conversations for a user, newest first.

        Args:
            user_id: UUID of the requesting user.

        Returns:
            List of Conversation ORM instances.
        """
        result = await self._db.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_history(
        self, conversation_id: uuid.UUID, limit: int = 10
    ) -> list[dict]:
        """
        Retrieve recent messages from a conversation with Redis-first cache strategy.

        1. Queries Redis sliding window memory buffer for instant retrieval.
        2. Falls back to PostgreSQL if Redis is empty or offline, then warms up Redis.

        Args:
            conversation_id: UUID of the conversation.
            limit: Maximum number of messages to return.

        Returns:
            List of role/content dicts.
        """
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

        # Cache miss or Redis offline: fallback to PostgreSQL
        result = await self._db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role != MessageRole.SYSTEM,
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        db_messages = list(reversed(result.scalars().all()))

        # Warm up Redis buffer asynchronously
        to_cache = []
        for msg in db_messages:
            if msg.role == MessageRole.USER:
                to_cache.append(HumanMessage(content=msg.content, audio_url=msg.audio_url))
            elif msg.role == MessageRole.ASSISTANT:
                to_cache.append(AIMessage(content=msg.content, audio_url=msg.audio_url))
        if to_cache:
            await redis_mem.add_messages(to_cache)

        return [{"role": msg.role, "content": msg.content} for msg in db_messages]

    async def get_messages(
        self, conversation_id: uuid.UUID, limit: int = 100
    ) -> list[Message]:
        """
        Retrieve full message records from a conversation for API responses.

        Args:
            conversation_id: UUID of the conversation.
            limit: Maximum number of messages to return.

        Returns:
            List of Message ORM instances ordered chronologically (oldest first).
        """
        result = await self._db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role != MessageRole.SYSTEM,
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def save_user_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        audio_url: str | None = None,
        audio_duration_ms: int | None = None,
        stt_confidence: float | None = None,
    ) -> Message:
        """
        Persist a user message to PostgreSQL and append to Redis chat buffer.

        Args:
            conversation_id: UUID of the conversation.
            content: User's message text.
            audio_url: Optional audio recording path.
            audio_duration_ms: Optional audio duration in milliseconds.
            stt_confidence: Optional STT transcription confidence score.

        Returns:
            Created Message ORM instance.
        """
        msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content,
            audio_url=audio_url,
            audio_duration_ms=audio_duration_ms,
            stt_confidence=stt_confidence,
        )
        self._db.add(msg)
        await self._db.flush()

        # Cache in Redis sliding buffer
        redis_mem = RedisChatMessageHistory(conversation_id)
        await redis_mem.add_message(
            HumanMessage(
                content=content,
                audio_url=audio_url,
                audio_duration_ms=audio_duration_ms,
                stt_confidence=stt_confidence,
            )
        )
        return msg

    async def save_assistant_message(
        self,
        conversation_id: uuid.UUID,
        content: str,
        corrections: list[Correction] | None = None,
        audio_url: str | None = None,
        sources: list[RetrievedChunk] | None = None,
        user_id: uuid.UUID | None = None,
    ) -> Message:
        """
        Persist an assistant message with voice metadata and corrections.

        Also updates the learner's long-term profile in Redis if corrections exist.

        Args:
            conversation_id: UUID of the conversation.
            content: Full assistant response text.
            corrections: Optional list of identified linguistic slips.
            audio_url: Optional audio synthesis URL.
            sources: Retrieved chunks used to generate the response.
            user_id: Optional user ID to record long-term weaknesses.

        Returns:
            Created Message ORM instance.
        """
        sources_json = [c.to_dict() for c in sources] if sources else None
        corrections_json = [c.model_dump(mode="json") for c in corrections] if corrections else None

        msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=content,
            audio_url=audio_url,
            corrections=corrections_json,
            sources=sources_json,
        )
        self._db.add(msg)
        await self._db.flush()

        # Cache in Redis sliding buffer
        redis_mem = RedisChatMessageHistory(conversation_id)
        await redis_mem.add_message(
            AIMessage(
                content=content,
                audio_url=audio_url,
                corrections=corrections,
            )
        )

        # Track long-term learner weaknesses in Redis
        if user_id and corrections:
            for corr in corrections:
                await self._profile_mgr.record_weakness(
                    user_id=user_id,
                    weakness_key=corr.type,
                )

        return msg

    async def update_progress(
        self,
        user_id: uuid.UUID,
        mode: str,
    ) -> None:
        """
        Upsert the user's daily learning progress counters.

        Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE to safely increment
        counters without race conditions.

        Args:
            user_id: UUID of the user.
            mode: Conversation mode — 'vocabulary' increments vocab_learned.
        """
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
        await self._db.execute(stmt)
