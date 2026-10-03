# name: chat.py (schemas)
# description: Pydantic request/response schemas for chat and conversation endpoints.

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.conversation import ConversationMode


from app.core.primitives.messages import Correction


class CreateConversationRequest(BaseModel):
    """Request body to start a new conversation."""

    mode: ConversationMode = ConversationMode.CONVERSATION
    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    """Conversation summary returned to the client."""

    id: uuid.UUID
    mode: str
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    """Request body for sending a chat message."""

    conversation_id: uuid.UUID
    message: str = Field(min_length=1, max_length=2000)
    client_message_id: uuid.UUID | None = None
    audio_url: str | None = None
    audio_duration_ms: int | None = None
    stt_confidence: float | None = None
    # Optional: restrict retrieval to specific documents (for doc_qa mode)
    document_ids: list[uuid.UUID] | None = None


class SourceChunk(BaseModel):
    """A retrieved source chunk included in the response."""

    chunk_id: str
    document_id: str
    content: str
    similarity: float
    metadata: dict
    preview: str | None = None
    index_in_prompt: int | None = None


class ChatResponse(BaseModel):
    """Full chat response for non-streaming endpoint."""

    conversation_id: uuid.UUID
    message_id: uuid.UUID
    response: str
    mode: str
    status: str = "complete"
    audio_url: str | None = None
    corrections: list[Correction] | None = None
    sources: list[SourceChunk] = []


class MessageResponse(BaseModel):
    """Single message record."""

    id: uuid.UUID
    role: str
    content: str
    status: str = "complete"
    client_message_id: uuid.UUID | None = None
    parent_message_id: uuid.UUID | None = None
    audio_url: str | None = None
    audio_duration_ms: int | None = None
    stt_confidence: float | None = None
    corrections: list[Correction] | None = None
    sources: list[SourceChunk] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
