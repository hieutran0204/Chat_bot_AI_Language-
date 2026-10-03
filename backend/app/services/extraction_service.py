# name: extraction_service.py
# description: Background service for async correction extraction after chat stream completion.
#              Calls LLM to classify learner errors, persists to PostgreSQL via WeaknessLogService,
#              and invalidates Redis cache. Runs independently of the request DB session lifecycle.

from __future__ import annotations

import json
import logging
import uuid

from app.core.database import AsyncSessionFactory
from app.core.primitives.messages import ALL_WEAKNESS_TYPES, Correction
from app.core.redis.profile import RedisLearnerProfileManager
from app.factory.pipeline_factory import create_llm
from app.models.conversation import Message
from app.services.weakness_log_service import WeaknessLogService

logger = logging.getLogger(__name__)

# ── Failure Counter ───────────────────────────────────────────────────────────
# In-memory counter for extraction failures. Exposed via /health endpoint.
# NOTE: This in-memory counter is designed for single-worker deployment.
# When deploying with multi-worker (uvicorn --workers > 1), migrate this to
# Redis INCR (e.g. langai:metrics:extraction_failures) or query
# PostgreSQL: SELECT COUNT(*) FROM messages WHERE extraction_status = 'failed'.
_extraction_failures: int = 0


def _inc_failure() -> None:
    """Increment the module-level extraction failure counter."""
    global _extraction_failures
    _extraction_failures += 1


def get_extraction_failure_count() -> int:
    """
    Return the total number of extraction failures since process startup.

    Used by the /health endpoint to detect regression without requiring log tail.
    A non-zero value means corrections are failing to be logged persistently.

    Returns:
        Integer count of extraction failures.
    """
    return _extraction_failures


# ── Extraction Prompt ─────────────────────────────────────────────────────────
_EXTRACTION_PROMPT_TEMPLATE = """You are an English error analysis assistant.

Analyze this conversation turn from an English learning session.
Identify ALL language errors made by the LEARNER only (not the AI tutor).

LEARNER said: "{user_message}"
AI TUTOR responded: "{assistant_text}"

For each error, output a JSON array with this EXACT schema:
[
  {{
    "original": "<exact erroneous phrase or word from the learner's message>",
    "suggestion": "<corrected, native-sounding alternative>",
    "type": "<EXACTLY one of: {taxonomy_list}>",
    "explanation": "<brief explanation in Vietnamese or simple English, max 1 sentence>"
  }}
]

Rules:
- Use ONLY the type codes listed above. Never invent new types.
- If the learner made no errors, output an empty array: []
- Output ONLY valid JSON. No prose, no markdown, no code fences.
"""

_TAXONOMY_LIST = ", ".join(ALL_WEAKNESS_TYPES)


class ExtractionService:
    """
    Background extraction service that classifies learner errors from a completed chat turn.

    Design constraints:
    - Must create its OWN AsyncSession using AsyncSessionFactory (never reuse request session).
    - Must be called as a FastAPI BackgroundTask — runs after response is sent.
    - All exceptions must be caught internally to prevent silent failures.
    - Failures are counted and exposed via get_extraction_failure_count().
    """

    def __init__(self) -> None:
        self._redis = RedisLearnerProfileManager()

    async def extract_and_log(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_message: str,
        assistant_text: str,
        assistant_message_id: uuid.UUID | None = None,
    ) -> None:
        """
        Background task entry point — extract corrections, persist to PostgreSQL, update cache.

        Creates an independent AsyncSession from AsyncSessionFactory (not the request session).
        Called by FastAPI BackgroundTasks after stream_chat() event_generator() completes.

        Args:
            user_id: UUID of the learner.
            conversation_id: UUID of the conversation session.
            user_message: The learner's original message text (needed for Correction.original).
            assistant_text: The full AI response text (context for classification).
            assistant_message_id: Optional UUID of the assistant Message ORM instance to update.
        """
        # ── Idempotency Guard ─────────────────────────────────────────────────
        if assistant_message_id is not None:
            async with AsyncSessionFactory() as check_session:
                msg = await check_session.get(Message, assistant_message_id)
                if msg and msg.extraction_status == "done":
                    logger.info(
                        "ExtractionService: message %s already processed (status=done). Skipping.",
                        assistant_message_id,
                    )
                    return

        try:
            corrections = await self._call_llm_extraction(user_message, assistant_text)

            # ── Own session — independent of request lifecycle ────────────────
            async with AsyncSessionFactory() as session:
                if corrections:
                    svc = WeaknessLogService(session)
                    await svc.log_corrections(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        corrections=corrections,
                    )

                # ── Dual-sync: update assistant message record with corrections & status ──
                if assistant_message_id is not None:
                    msg = await session.get(Message, assistant_message_id)
                    if msg:
                        corrections_json = [c.model_dump(mode="json") for c in corrections] if corrections else []
                        msg.corrections = corrections_json
                        msg.extraction_status = "done"
                        msg.extraction_error = None

                await session.commit()

            # ── Invalidate Redis cache (pure invalidation: DELETE keys) ────────
            # Next read will re-populate from PostgreSQL with fresh aggregate
            if corrections:
                await self._invalidate_redis_cache(user_id)

            logger.info(
                "ExtractionService: processed extraction for user=%s conv=%s msg=%s (corrections=%d)",
                user_id, conversation_id, assistant_message_id, len(corrections),
            )

        except Exception as exc:
            _inc_failure()
            logger.exception(
                "ExtractionService: extraction FAILED for user=%s session=%s "
                "(failure_count=%d)",
                user_id, conversation_id, get_extraction_failure_count(),
            )
            # Update message status to failed if assistant_message_id provided
            if assistant_message_id is not None:
                try:
                    async with AsyncSessionFactory() as error_session:
                        msg = await error_session.get(Message, assistant_message_id)
                        if msg:
                            msg.extraction_status = "failed"
                            msg.extraction_error = str(exc)
                            await error_session.commit()
                except Exception:
                    logger.exception(
                        "ExtractionService: failed to update extraction_status to 'failed' on message %s",
                        assistant_message_id,
                    )

    async def _call_llm_extraction(
        self,
        user_message: str,
        assistant_text: str,
    ) -> list[Correction]:
        """
        Call the LLM with a structured extraction prompt to classify corrections.

        Uses a non-streaming .chat() call since we need the full JSON output.
        Temperature is set to 0 for deterministic, consistent classification.

        Args:
            user_message: Learner's original message.
            assistant_text: AI tutor's response text.

        Returns:
            List of validated Correction instances. Empty list if none found or parse fails.
        """
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(
            user_message=user_message,
            assistant_text=assistant_text,
            taxonomy_list=_TAXONOMY_LIST,
        )

        llm = create_llm()
        raw_output = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # Deterministic classification
            max_tokens=512,    # Corrections are short; cap to control latency
        )
        return self._parse_corrections(raw_output)

    def _parse_corrections(self, raw_output: str) -> list[Correction]:
        """
        Parse and validate LLM JSON output into Correction model instances.

        Skips any entries with invalid or unrecognized weakness_type values
        to avoid poisoning the analytics store with bad data.

        Args:
            raw_output: Raw LLM response string (expected to be a JSON array).

        Returns:
            List of valid Correction instances. Empty list on parse failure.
        """
        # Strip markdown code fences if LLM ignores the "no markdown" instruction
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "ExtractionService: failed to parse LLM output as JSON. "
                "Raw output (first 200 chars): %s", cleaned[:200]
            )
            return []

        if not isinstance(data, list):
            logger.warning("ExtractionService: expected JSON array, got %s", type(data).__name__)
            return []

        corrections: list[Correction] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                corr = Correction(**item)
                corrections.append(corr)
            except Exception as exc:
                logger.debug(
                    "ExtractionService: skipping invalid correction entry %s — %s", item, exc
                )

        logger.debug(
            "ExtractionService: parsed %d/%d corrections from LLM output",
            len(corrections), len(data),
        )
        return corrections

    async def _invalidate_redis_cache(
        self,
        user_id: uuid.UUID,
    ) -> None:
        """
        Invalidate the Redis weakness cache so next read refreshes from PostgreSQL.

        Implements pure read-through cache invalidation: write PostgreSQL → evict Redis.
        Next call to ProfileLoader will rebuild fresh aggregated counts from PostgreSQL.

        Args:
            user_id: UUID of the learner whose cache entry to evict.
        """
        try:
            await self._redis.invalidate_profile(user_id)
            logger.debug("ExtractionService: invalidated Redis profile cache for user=%s", user_id)
        except Exception:
            logger.warning(
                "ExtractionService: Redis cache eviction failed for user=%s.", user_id
            )
