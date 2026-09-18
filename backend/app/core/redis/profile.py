# name: profile.py
# description: Redis-backed learner profile manager for persistent long-term memory (CEFR level & weakness tracking).

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any
from pydantic import BaseModel, Field

from app.core.redis.client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class LearnerProfile(BaseModel):
    """
    Cached representation of a learner's persistent profile.

    Attributes:
        user_id: String UUID of the learner.
        level: Assessed English proficiency level (A1, A2, B1, B2, C1, C2).
        weaknesses: Frequency map of persistent language slips (e.g. {"past_tense": 4}).
        last_session: ISO timestamp of last interaction.
    """

    user_id: str
    level: str = "A2"
    weaknesses: dict[str, int] = Field(default_factory=dict)
    last_session: str | None = None

    def top_weaknesses(self, top_n: int = 5) -> list[tuple[str, int]]:
        """Return the top N most frequent weaknesses sorted descending."""
        sorted_items = sorted(self.weaknesses.items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:top_n]


class RedisLearnerProfileManager:
    """
    Manages long-term learner memory cached in Redis.

    Unlike conversation chat buffers, learner profiles have NO TTL (or very long TTL)
    because they represent accumulated learning history.
    PostgreSQL remains the source of truth; Redis serves as a warm cache
    injected into prompt templates at the start of each chat session.

    Key Pattern:
        langai:profile:{user_id}:level       -> string (e.g. "B1")
        langai:profile:{user_id}:weaknesses  -> hash {"past_tense": "5", "preposition": "3"}
        langai:profile:{user_id}:last_session -> string ISO timestamp
    """

    def __init__(self, redis_client: RedisClient | None = None) -> None:
        self.redis = redis_client or get_redis_client()

    def _keys(self, user_id: uuid.UUID | str) -> tuple[str, str, str]:
        uid = str(user_id)
        return (
            f"langai:profile:{uid}:level",
            f"langai:profile:{uid}:weaknesses",
            f"langai:profile:{uid}:last_session",
        )

    async def get_profile(
        self, user_id: uuid.UUID | str, fallback_level: str = "A2"
    ) -> LearnerProfile:
        """
        Fetch full learner profile from Redis.

        Args:
            user_id: User identifier.
            fallback_level: Default proficiency level if not yet cached.

        Returns:
            LearnerProfile instance.
        """
        uid = str(user_id)
        if not self.redis.is_available or self.redis.client is None:
            return LearnerProfile(user_id=uid, level=fallback_level)

        level_key, weak_key, session_key = self._keys(uid)
        try:
            pipe = self.redis.client.pipeline()
            pipe.get(level_key)
            pipe.hgetall(weak_key)
            pipe.get(session_key)
            level, raw_weaknesses, last_session = await pipe.execute()

            weaknesses = {k: int(v) for k, v in (raw_weaknesses or {}).items()}
            return LearnerProfile(
                user_id=uid,
                level=level or fallback_level,
                weaknesses=weaknesses,
                last_session=last_session,
            )
        except Exception as exc:
            logger.warning("Failed to get learner profile from Redis for %s: %s", uid, exc)
            return LearnerProfile(user_id=uid, level=fallback_level)

    async def set_level(self, user_id: uuid.UUID | str, level: str) -> None:
        """Cache user's CEFR proficiency level."""
        if not self.redis.is_available or self.redis.client is None:
            return
        level_key, _, _ = self._keys(user_id)
        try:
            await self.redis.client.set(level_key, level)
        except Exception as exc:
            logger.warning("Failed to cache level in Redis for %s: %s", user_id, exc)

    async def record_weakness(
        self,
        user_id: uuid.UUID | str,
        weakness_key: str,
        delta: int = 1,
        max_items: int = 15,
    ) -> None:
        """
        Increment the frequency of a detected learner weakness.

        Applies pruning if weaknesses count exceeds max_items to prevent
        unbounded context growth.

        Args:
            user_id: Learner UUID.
            weakness_key: Identifier of error (e.g. "past_tense", "third_person_s").
            delta: Count increment.
            max_items: Maximum distinct weaknesses tracked in the hot cache.
        """
        if not self.redis.is_available or self.redis.client is None:
            return
        _, weak_key, _ = self._keys(user_id)
        try:
            new_val = await self.redis.client.hincrby(weak_key, weakness_key, delta)

            # Check if pruning is necessary
            count = await self.redis.client.hlen(weak_key)
            if count > max_items:
                all_weak = await self.redis.client.hgetall(weak_key)
                # Sort ascending to remove least frequent
                sorted_items = sorted(all_weak.items(), key=lambda x: int(x[1]))
                to_remove = [k for k, _ in sorted_items[: count - max_items]]
                if to_remove:
                    await self.redis.client.hdel(weak_key, *to_remove)
                    logger.debug("Pruned %d stale weaknesses for user %s", len(to_remove), user_id)
        except Exception as exc:
            logger.warning("Failed to record weakness in Redis for %s: %s", user_id, exc)

    async def update_last_session(self, user_id: uuid.UUID | str) -> None:
        """Update last active session timestamp."""
        if not self.redis.is_available or self.redis.client is None:
            return
        _, _, session_key = self._keys(user_id)
        try:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await self.redis.client.set(session_key, now_iso)
        except Exception as exc:
            logger.warning("Failed to update last session in Redis for %s: %s", user_id, exc)
