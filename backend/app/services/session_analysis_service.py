# name: session_analysis_service.py
# description: Service and interface for generating SessionInsight cards post-conversation.

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pydantic import BaseModel, Field

from app.factory.pipeline_factory import create_llm

logger = logging.getLogger(__name__)


class SessionInsightData(BaseModel):
    """Structured data payload for a generated session insight."""

    encouragements: list[str] = Field(default_factory=list)
    reminders: list[str] = Field(default_factory=list)
    focus_next: str | None = None
    session_score: int = Field(default=80, ge=0, le=100)
    streak_days: int = Field(default=1, ge=0)


class ISessionInsightService(ABC):
    """Interface for generating post-session learning analytics insights."""

    @abstractmethod
    async def generate_session_insight(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        session_start: datetime,
        session_end: datetime,
        user_messages: list[str],
        weakness_types_found: list[str],
    ) -> SessionInsightData:
        """
        Generate personalized learning insight for a completed session.

        Args:
            conversation_id: UUID of the conversation session.
            user_id: UUID of the learner.
            session_start: Start timestamp of the session window.
            session_end: End timestamp of the session window.
            user_messages: List of user message contents in this session.
            weakness_types_found: List of error categories detected in this session.

        Returns:
            SessionInsightData with encouragements, reminders, and score.
        """
        pass


_INSIGHT_PROMPT_TEMPLATE = """You are an encouraging and supportive English tutor analyzing a completed study session.

Session Information:
- Total Learner Messages: {msg_count}
- Errors Identified: {errors_summary}

Learner's sample sentences:
{samples}

Generate a JSON object summarizing the session with this EXACT structure:
{{
  "encouragements": ["<1-2 short positive observations on effort, fluency, or grammar in Vietnamese or English>"],
  "reminders": ["<1-2 friendly reminders about specific errors to pay attention to next time>"],
  "focus_next": "<name of top error to focus on, or null>",
  "session_score": <integer from 60 to 100 based on participation and accuracy>,
  "streak_days": 1
}}

Rules:
- Output ONLY valid JSON. No markdown fences, no explanatory text.
"""


class SessionAnalysisService(ISessionInsightService):
    """Default implementation of session insight generator using LLM."""

    async def generate_session_insight(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        session_start: datetime,
        session_end: datetime,
        user_messages: list[str],
        weakness_types_found: list[str],
    ) -> SessionInsightData:
        """Generate session insight via LLM."""
        if not user_messages:
            return SessionInsightData(
                encouragements=["Good effort practicing English today!"],
                reminders=[],
                focus_next=None,
                session_score=80,
                streak_days=1,
            )

        samples = "\n".join(f"- {m}" for m in user_messages[:5])
        errors_summary = ", ".join(weakness_types_found) if weakness_types_found else "None"

        prompt = _INSIGHT_PROMPT_TEMPLATE.format(
            msg_count=len(user_messages),
            errors_summary=errors_summary,
            samples=samples,
        )

        llm = create_llm()
        raw = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1]).strip()

        data = json.loads(cleaned)
        return SessionInsightData.model_validate(data)
