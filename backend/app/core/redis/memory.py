# name: memory.py
# description: Redis-backed sliding-window chat message history for low-latency session memory.

from __future__ import annotations

import json
import logging
import uuid
from typing import Sequence

from app.core.config import settings
from app.core.primitives.messages import BaseMessage
from app.core.redis.client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class RedisChatMessageHistory:
    """
    Manages short-term conversation sliding window in Redis.

    Messages are serialized to JSON and stored in a Redis list (RPUSH/LRANGE).
    Keys have an automatic TTL (default: 24 hours) to preserve RAM.

    Key Pattern:
        langai:conversation:{conversation_id}:messages
    """

    def __init__(
        self,
        conversation_id: uuid.UUID | str,
        redis_client: RedisClient | None = None,
        ttl_seconds: int = 86400,
    ) -> None:
        self.conversation_id = str(conversation_id)
        self.redis = redis_client or get_redis_client()
        self.ttl_seconds = ttl_seconds
        self.key = f"langai:conversation:{self.conversation_id}:messages"

    async def add_message(self, message: BaseMessage) -> None:
        """
        Append a message to the conversation history.

        Args:
            message: BaseMessage subclass instance.
        """
        if not self.redis.is_available or self.redis.client is None:
            logger.debug("Redis unavailable; skipping message buffer append for conv %s", self.conversation_id)
            return

        try:
            payload = json.dumps(message.to_dict(), ensure_ascii=False)
            pipe = self.redis.client.pipeline()
            pipe.rpush(self.key, payload)
            pipe.expire(self.key, self.ttl_seconds)
            await pipe.execute()
        except Exception as exc:
            logger.warning("Failed to append message to Redis for conv %s: %s", self.conversation_id, exc)

    async def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Append multiple messages in a single atomic pipeline operation.

        Args:
            messages: Sequence of BaseMessage instances.
        """
        if not messages or not self.redis.is_available or self.redis.client is None:
            return

        try:
            payloads = [json.dumps(m.to_dict(), ensure_ascii=False) for m in messages]
            pipe = self.redis.client.pipeline()
            pipe.rpush(self.key, *payloads)
            pipe.expire(self.key, self.ttl_seconds)
            await pipe.execute()
        except Exception as exc:
            logger.warning("Failed to bulk append messages to Redis for conv %s: %s", self.conversation_id, exc)

    async def get_messages(self, limit: int = 20) -> list[BaseMessage]:
        """
        Retrieve the most recent messages from the sliding window.

        Args:
            limit: Maximum number of recent messages to return.

        Returns:
            Chronologically ordered list of BaseMessage instances.
        """
        if not self.redis.is_available or self.redis.client is None:
            return []

        try:
            # Fetch the last `limit` elements: -limit to -1
            raw_items = await self.redis.client.lrange(self.key, -limit, -1)
            messages: list[BaseMessage] = []
            for item in raw_items:
                data = json.loads(item)
                messages.append(BaseMessage.from_dict(data))
            return messages
        except Exception as exc:
            logger.warning("Failed to read messages from Redis for conv %s: %s", self.conversation_id, exc)
            return []

    async def clear(self) -> None:
        """Purge the chat buffer for this conversation session."""
        if not self.redis.is_available or self.redis.client is None:
            return
        try:
            await self.redis.client.delete(self.key)
        except Exception as exc:
            logger.warning("Failed to clear Redis history for conv %s: %s", self.conversation_id, exc)
