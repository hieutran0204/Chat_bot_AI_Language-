# name: memory.py
# description: Abstract interfaces (ports) for short-term chat memory buffer and long-term learner profile storage.

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Sequence

from app.core.primitives.messages import BaseMessage


class IMemoryStore(ABC):
    """
    Port interface for conversation chat memory buffers.

    Decouples short-term session storage (Redis list, in-memory, Postgres)
    from application services and pipeline orchestration.
    """

    @abstractmethod
    async def add_message(self, message: BaseMessage) -> None:
        """
        Append a single message to the conversation memory buffer.

        Args:
            message: BaseMessage subclass instance (HumanMessage, AIMessage, etc.).
        """
        pass

    @abstractmethod
    async def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Append multiple messages atomically to the conversation memory buffer.

        Args:
            messages: Sequence of BaseMessage instances.
        """
        pass

    @abstractmethod
    async def get_messages(self, limit: int = 20) -> list[BaseMessage]:
        """
        Retrieve the most recent messages from the memory sliding window.

        Args:
            limit: Maximum number of recent messages to return.

        Returns:
            Chronologically ordered list of BaseMessage instances.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Purge all stored messages for this conversation session."""
        pass


class ILearnerProfileStore(ABC):
    """
    Port interface for learner profile and long-term weakness tracking.

    Decouples learner state caching from application services.
    """

    @abstractmethod
    async def get_profile(self, user_id: uuid.UUID | str) -> Any:
        """
        Fetch the current cached profile for a learner.

        Args:
            user_id: Learner UUID or string identifier.

        Returns:
            LearnerProfile or dictionary of profile attributes.
        """
        pass

    @abstractmethod
    async def set_level(self, user_id: uuid.UUID | str, level: str) -> None:
        """
        Update the learner's assessed CEFR proficiency level (A1-C2).

        Args:
            user_id: Learner UUID or string identifier.
            level: CEFR level string.
        """
        pass

    @abstractmethod
    async def record_weakness(
        self, user_id: uuid.UUID | str, weakness_key: str, increment: int = 1
    ) -> None:
        """
        Increment the occurrence count for a specific linguistic weakness.

        Args:
            user_id: Learner UUID or string identifier.
            weakness_key: Category or key of the weakness (e.g. 'past_tense').
            increment: Count to add.
        """
        pass

    @abstractmethod
    async def update_last_session(self, user_id: uuid.UUID | str) -> None:
        """
        Record the current timestamp as the learner's last active session.

        Args:
            user_id: Learner UUID or string identifier.
        """
        pass

    @abstractmethod
    async def invalidate_profile(self, user_id: uuid.UUID | str) -> None:
        """
        Evict the cached profile for a learner.

        Args:
            user_id: Learner UUID or string identifier.
        """
        pass
