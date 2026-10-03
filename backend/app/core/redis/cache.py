# name: cache.py
# description: Redis general-purpose caching manager for dictionary lookups, grammar fixes, and LLM responses.

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.core.redis.client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    General key-value cache manager with TTL support.

    Used to memoize dictionary API responses, grammar checks,
    and repeated translation/explanation requests.
    """

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        default_ttl: int = 3600,
    ) -> None:
        self.redis = redis_client or get_redis_client()
        self.default_ttl = default_ttl

    async def get(self, key: str) -> str | None:
        """Fetch a string value by key."""
        if not self.redis.is_available or self.redis.client is None:
            return None
        try:
            return await self.redis.client.get(key)
        except Exception as exc:
            logger.warning("Cache GET failed for key '%s': %s", key, exc)
            return None

    async def get_json(self, key: str) -> Any | None:
        """Fetch and deserialize a JSON value."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning("JSON decode failed for cache key '%s': %s", key, exc)
            return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Store a string value with TTL."""
        if not self.redis.is_available or self.redis.client is None:
            return
        ttl_val = ttl if ttl is not None else self.default_ttl
        try:
            await self.redis.client.set(key, value, ex=ttl_val)
        except Exception as exc:
            logger.warning("Cache SET failed for key '%s': %s", key, exc)

    async def set_json(self, key: str, data: Any, ttl: int | None = None) -> None:
        """Serialize and store an object as JSON with TTL."""
        try:
            payload = json.dumps(data, ensure_ascii=False)
            await self.set(key, payload, ttl=ttl)
        except Exception as exc:
            logger.warning("JSON encode failed for cache key '%s': %s", key, exc)

    async def delete(self, key: str) -> None:
        """Remove a cached key."""
        if not self.redis.is_available or self.redis.client is None:
            return
        try:
            await self.redis.client.delete(key)
        except Exception as exc:
            logger.warning("Cache DELETE failed for key '%s': %s", key, exc)

    # ── Convenience Key Generators ──────────────────────────────────────────

    @staticmethod
    def dict_key(word: str) -> str:
        """Generate cache key for a dictionary word lookup."""
        return f"langai:dict:{word.strip().lower()}"

    @staticmethod
    def grammar_key(sentence: str) -> str:
        """Generate cache key for a grammar analysis check."""
        h = hashlib.sha256(sentence.strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"langai:grammar:{h}"
