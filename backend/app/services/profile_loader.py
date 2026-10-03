# name: profile_loader.py
# description: Loads learner weakness profile from Redis (hot cache) or PostgreSQL (fallback)
#              and formats it as a system prompt injection string for personalized AI tutoring.
#              Max 300 tokens injected to keep latency impact minimal.

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis.profile import RedisLearnerProfileManager
from app.services.weakness_log_service import WeaknessLogService

logger = logging.getLogger(__name__)

# Maximum weaknesses to inject into system prompt
_MAX_WEAKNESSES = 3


class ProfileLoader:
    """
    Builds a personalized context string for LLM system prompt injection.

    Load order (fastest to slowest):
      1. Redis hot cache via RedisLearnerProfileManager.get_profile()
      2. PostgreSQL fallback via WeaknessLogService.get_top_weaknesses()
      3. Empty string — new user or cold start (graceful degradation, no crash)

    Expected latency impact:
      - Cache hit:  +2–5ms
      - Cache miss: +15–30ms (PostgreSQL query)
      - No data:    +0ms (returns "" immediately)
    """

    def __init__(self, redis_profile_mgr: RedisLearnerProfileManager | None = None) -> None:
        self._redis = redis_profile_mgr or RedisLearnerProfileManager()

    async def get_weakness_context(
        self,
        user_id: uuid.UUID,
        db: AsyncSession,
    ) -> str:
        """
        Build the {profile_section} string to inject into the system prompt.

        Args:
            user_id: UUID of the learner.
            db: Active AsyncSession for PostgreSQL fallback query.

        Returns:
            Formatted profile context string, or "" if no data available.
        """
        top: list[tuple[str, int]] = []

        # ── Path 1: Redis hot cache ────────────────────────────────────────────
        try:
            profile = await self._redis.get_profile(user_id)
            top = profile.top_weaknesses(_MAX_WEAKNESSES)
        except Exception:
            logger.warning("ProfileLoader: Redis unavailable, falling back to PostgreSQL.")

        # ── Path 2: PostgreSQL fallback (cold start or Redis miss) ────────────
        if not top:
            try:
                svc = WeaknessLogService(db)
                top = await svc.get_top_weaknesses(user_id, top_n=_MAX_WEAKNESSES)
            except Exception:
                logger.warning("ProfileLoader: PostgreSQL query failed, returning empty profile.")

        # ── Path 3: No data yet (new user) ────────────────────────────────────
        if not top:
            return ""

        lines = "\n".join(
            f"  {i + 1}. {wtype} — {count} occurrences"
            for i, (wtype, count) in enumerate(top)
        )
        profile_text = (
            "--- LEARNER PROFILE ---\n"
            f"This learner's top persistent weaknesses:\n{lines}\n"
            "If the learner makes any of these errors during this conversation, "
            "gently correct them with a brief friendly reminder and a concrete example.\n"
            "--- END PROFILE ---"
        )
        logger.debug("ProfileLoader: injecting %d weaknesses for user=%s", len(top), user_id)
        return profile_text
