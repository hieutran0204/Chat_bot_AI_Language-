# name: messages.py
# description: LangChain-standard message abstractions with voice-first and learner correction fields.

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Supported roles in a chat conversation."""

    SYSTEM = "system"
    HUMAN = "user"
    AI = "assistant"
    TOOL = "tool"


class Correction(BaseModel):
    """
    Represents a targeted correction identified by the AI tutor.

    Attributes:
        original: The learner's original utterance or fragment with errors.
        suggestion: Natural native-like alternative or corrected form.
        type: Category of the error (grammar, pronunciation, vocab).
        explanation: Brief explanation in simple Vietnamese or English.
    """

    original: str
    suggestion: str
    type: Literal["grammar", "pronunciation", "vocab"]
    explanation: str | None = None


class BaseMessage(BaseModel):
    """
    Base message abstraction for all conversational turns.

    Voice is a first-class citizen: audio metadata is natively supported
    alongside textual content.

    Attributes:
        content: Raw textual content of the message.
        role: Functional role of the sender (system, user, assistant, tool).
        audio_url: URL/path to raw audio recording (voice from user or TTS speech).
        audio_duration_ms: Duration of audio in milliseconds.
        additional_kwargs: Extensible dictionary for provider-specific metadata.
    """

    content: str
    role: MessageRole
    audio_url: str | None = None
    audio_duration_ms: int | None = None
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to serializable dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseMessage:
        """
        Reconstruct a specialized message subclass from dictionary representation.

        Args:
            data: Serialized message dictionary.

        Returns:
            Instantiated concrete message instance (HumanMessage, AIMessage, etc.).
        """
        role = data.get("role")
        if role == MessageRole.HUMAN or role == "user":
            return HumanMessage(**data)
        if role == MessageRole.AI or role == "assistant":
            return AIMessage(**data)
        if role == MessageRole.SYSTEM:
            return SystemMessage(**data)
        if role == MessageRole.TOOL:
            return ToolMessage(**data)
        return cls(**data)


class SystemMessage(BaseMessage):
    """Directive message setting AI persona, background knowledge, or constraints."""

    role: MessageRole = MessageRole.SYSTEM


class HumanMessage(BaseMessage):
    """
    Message originating from the human learner.

    Attributes:
        stt_confidence: Confidence score from Speech-To-Text model (0.0 to 1.0).
                        Allows tutor to politely ask for clarification if confidence is low.
    """

    role: MessageRole = MessageRole.HUMAN
    stt_confidence: float | None = None


class AIMessage(BaseMessage):
    """
    Message originating from the AI Speaking Tutor.

    Attributes:
        corrections: Decoupled list of learner language slips/errors.
                     Allows conversational flow to remain friendly and natural,
                     while delivering structured feedback to the UI.
        tool_calls: Structured tool call requests if LLM requested external tool execution.
    """

    role: MessageRole = MessageRole.AI
    corrections: list[Correction] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ToolMessage(BaseMessage):
    """
    Response returned from executing an external tool.

    Attributes:
        tool_call_id: Unique identifier matching the AI tool call request.
        name: Name of the executed tool.
    """

    role: MessageRole = MessageRole.TOOL
    tool_call_id: str
    name: str


def messages_to_ollama(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """
    Convert LangChain-standard messages to the format expected by Ollama / Chat APIs.

    Args:
        messages: List of BaseMessage instances.

    Returns:
        List of dicts with 'role' and 'content' keys.
    """
    return [
        {
            "role": msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role),
            "content": msg.content,
        }
        for msg in messages
    ]
