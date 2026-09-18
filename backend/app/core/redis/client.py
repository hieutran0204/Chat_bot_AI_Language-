# name: client.py
# description: Async Redis client with connection pooling, health checks, and lifecycle management.

from __future__ import annotations

import logging
from typing import Any
import redis.asyncio as aioredis
from redis.asyncio.client import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Manages the async Redis connection pool.

    Provides high-performance non-blocking I/O for chat buffers,
    learner profile caching, and dictionary lookup memoization.
    Supports graceful fallback if Redis is temporarily unreachable.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._url = redis_url or settings.redis_url
        self._pool: aioredis.ConnectionPool | None = None
        self._client: Redis | None = None
        self._is_connected: bool = False

    async def connect(self) -> None:
        """Initialize the connection pool and verify connectivity via PING."""
        try:
            logger.info("Connecting to Redis at: %s", self._url)
            self._pool = aioredis.ConnectionPool.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            await self._client.ping()
            self._is_connected = True
            logger.info("Redis connected successfully.")
        except Exception as exc:
            self._is_connected = False
            logger.warning(
                "Could not connect to Redis (%s). Running with memory fallback.", exc
            )

    async def disconnect(self) -> None:
        """Close client connection and release pool resources safely."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.disconnect()
        self._is_connected = False
        logger.info("Redis disconnected.")

    @property
    def is_available(self) -> bool:
        """Return True if Redis client is connected and available."""
        return self._is_connected and self._client is not None

    @property
    def client(self) -> Redis | None:
        """Direct access to the underlying aioredis.Redis instance."""
        return self._client if self._is_connected else None

    async def ping(self) -> bool:
        """Perform a live health-check ping against Redis."""
        if not self._client:
            return False
        try:
            return bool(await self._client.ping())
        except Exception as exc:
            logger.warning("Redis ping failed: %s", exc)
            self._is_connected = False
            return False


# Global process-wide singleton
_redis_instance: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """Get or instantiate the global RedisClient singleton."""
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisClient()
    return _redis_instance
