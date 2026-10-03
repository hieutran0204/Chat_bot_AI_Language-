# name: weakness_log_service.py
# description: Service for persisting learner corrections to PostgreSQL (user_weakness_log)
#              and strength evidence (user_strength_log), with is_repeated detection.
#              This is the PostgreSQL source-of-truth that backs the Redis hot cache.

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.primitives.messages import Correction
from app.models.learner_analytics import UserStrengthLog, UserWeaknessLog

logger = logging.getLogger(__name__)

# Number of past occurrences required to flag is_repeated = True
_REPEATED_THRESHOLD: int = 3
# Look-back window for repeated detection
_REPEATED_WINDOW_DAYS: int = 30


class WeaknessLogService:
    """
    Manages persistent learner weakness and strength records in PostgreSQL.

    Responsibilities:
    1. log_corrections()  — Persist AI-issued corrections from a chat turn.
                            Detects is_repeated by querying recent history.
    2. log_strength()     — Record a skill the learner demonstrated correctly.
    3. get_weakness_counts() — Return frequency map of weakness_type → count
                               for a user over a configurable number of days.
    4. get_top_weaknesses() — Return sorted list of most frequent weakness types.
    5. get_recent_strengths() — Return recently confirmed strength types.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Core Write Operations ─────────────────────────────────────────────────

    async def log_corrections(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        corrections: list[Correction],
    ) -> list[UserWeaknessLog]:
        """
        Persist a list of AI-issued corrections for a single chat turn.

        Checks the past 30-day history for each weakness_type to determine
        whether to flag is_repeated = True.

        Args:
            user_id: UUID of the learner.
            conversation_id: UUID of the active conversation.
            corrections: List of Correction instances from the AI response.

        Returns:
            List of created UserWeaknessLog ORM instances.
        """
        if not corrections:
            return []

        # Pre-fetch recent occurrence counts for all unique types in this batch
        unique_types = list({c.type for c in corrections})
        counts = await self._count_recent_occurrences(user_id, unique_types)

        logs: list[UserWeaknessLog] = []
        for correction in corrections:
            is_repeated = counts.get(correction.type, 0) >= _REPEATED_THRESHOLD
            entry = UserWeaknessLog(
                user_id=user_id,
                conversation_id=conversation_id,
                weakness_type=correction.type,
                original=correction.original,
                suggestion=correction.suggestion,
                explanation=correction.explanation,
                is_repeated=is_repeated,
            )
            self._db.add(entry)
            logs.append(entry)

            if is_repeated:
                logger.info(
                    "Repeated weakness detected: user=%s type=%s (seen %d times in last %d days)",
                    user_id,
                    correction.type,
                    counts[correction.type],
                    _REPEATED_WINDOW_DAYS,
                )

        await self._db.flush()
        logger.info(
            "Logged %d corrections to user_weakness_log: user=%s conversation=%s",
            len(logs),
            user_id,
            conversation_id,
        )
        return logs

    async def log_strength(
        self,
        user_id: uuid.UUID,
        strength_type: str,
        evidence: str,
        confidence_score: float = 0.8,
    ) -> UserStrengthLog:
        """
        Record a skill the learner demonstrated correctly.

        Called when the AI observes a complex correct structure without correcting it,
        or when a previously-weak skill shows sustained improvement.

        Args:
            user_id: UUID of the learner.
            strength_type: Weakness taxonomy type now used correctly.
            evidence: The actual sentence/phrase demonstrating the skill.
            confidence_score: Confidence in the strength assessment (0.0–1.0).

        Returns:
            Created UserStrengthLog ORM instance.
        """
        entry = UserStrengthLog(
            user_id=user_id,
            strength_type=strength_type,
            evidence=evidence,
            confidence_score=confidence_score,
        )
        self._db.add(entry)
        await self._db.flush()
        logger.debug(
            "Logged strength: user=%s type=%s score=%.2f",
            user_id,
            strength_type,
            confidence_score,
        )
        return entry

    # ── Read / Analytics Operations ───────────────────────────────────────────

    async def get_weakness_counts(
        self,
        user_id: uuid.UUID,
        days: int = 30,
    ) -> dict[str, int]:
        """
        Aggregate weakness occurrence counts over the past N days.

        Args:
            user_id: UUID of the learner.
            days: Look-back window in calendar days (default: 30).

        Returns:
            Dict mapping weakness_type -> occurrence count, sorted descending.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        result = await self._db.execute(
            select(
                UserWeaknessLog.weakness_type,
                func.count(UserWeaknessLog.id).label("cnt"),
            )
            .where(
                UserWeaknessLog.user_id == user_id,
                UserWeaknessLog.created_at >= since,
            )
            .group_by(UserWeaknessLog.weakness_type)
            .order_by(func.count(UserWeaknessLog.id).desc())
        )
        return {row.weakness_type: row.cnt for row in result.all()}

    async def get_top_weaknesses(
        self,
        user_id: uuid.UUID,
        top_n: int = 5,
        days: int = 30,
    ) -> list[tuple[str, int]]:
        """
        Return the top N most frequent weakness types for a learner.

        Args:
            user_id: UUID of the learner.
            top_n: Maximum number of weaknesses to return.
            days: Look-back window in calendar days.

        Returns:
            List of (weakness_type, count) tuples sorted by count descending.
        """
        counts = await self.get_weakness_counts(user_id, days=days)
        return list(counts.items())[:top_n]

    async def get_repeated_weaknesses(
        self,
        user_id: uuid.UUID,
        days: int = 30,
    ) -> list[str]:
        """
        Return weakness types that have been flagged as repeated (is_repeated=True).

        Args:
            user_id: UUID of the learner.
            days: Look-back window in calendar days.

        Returns:
            List of distinct weakness_type strings that recurred.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        result = await self._db.execute(
            select(UserWeaknessLog.weakness_type)
            .where(
                UserWeaknessLog.user_id == user_id,
                UserWeaknessLog.is_repeated == True,  # noqa: E712
                UserWeaknessLog.created_at >= since,
            )
            .distinct()
        )
        return [row.weakness_type for row in result.all()]

    async def get_recent_strengths(
        self,
        user_id: uuid.UUID,
        top_n: int = 5,
        days: int = 14,
    ) -> list[tuple[str, float]]:
        """
        Return recently confirmed strength types, averaged by confidence score.

        Args:
            user_id: UUID of the learner.
            top_n: Maximum number of strengths to return.
            days: Look-back window in calendar days.

        Returns:
            List of (strength_type, avg_confidence) tuples sorted by confidence desc.
        """
        since = datetime.now(UTC) - timedelta(days=days)
        result = await self._db.execute(
            select(
                UserStrengthLog.strength_type,
                func.avg(UserStrengthLog.confidence_score).label("avg_score"),
            )
            .where(
                UserStrengthLog.user_id == user_id,
                UserStrengthLog.created_at >= since,
            )
            .group_by(UserStrengthLog.strength_type)
            .order_by(func.avg(UserStrengthLog.confidence_score).desc())
            .limit(top_n)
        )
        return [(row.strength_type, float(row.avg_score)) for row in result.all()]

    # ── Private Helpers ───────────────────────────────────────────────────────

    async def _count_recent_occurrences(
        self,
        user_id: uuid.UUID,
        weakness_types: list[str],
    ) -> dict[str, int]:
        """
        Batch-count past occurrences of specific weakness types within the look-back window.

        Args:
            user_id: UUID of the learner.
            weakness_types: List of weakness type strings to check.

        Returns:
            Dict mapping weakness_type -> past occurrence count.
        """
        if not weakness_types:
            return {}
        since = datetime.now(UTC) - timedelta(days=_REPEATED_WINDOW_DAYS)
        result = await self._db.execute(
            select(
                UserWeaknessLog.weakness_type,
                func.count(UserWeaknessLog.id).label("cnt"),
            )
            .where(
                UserWeaknessLog.user_id == user_id,
                UserWeaknessLog.weakness_type.in_(weakness_types),
                UserWeaknessLog.created_at >= since,
            )
            .group_by(UserWeaknessLog.weakness_type)
        )
        return {row.weakness_type: row.cnt for row in result.all()}
