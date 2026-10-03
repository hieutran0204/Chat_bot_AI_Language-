# name: chat.py (API router)
# description: Chat endpoints — create conversations, send messages with SSE streaming,
#              and retrieve conversation history.

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import AsyncSessionFactory, get_db
from app.core.exceptions import LLMException
from app.models.conversation import ConversationMode
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    SourceChunk,
)
from app.services.chat_service import ChatService
from app.services.extraction_service import ExtractionService

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

# Module-level singleton — stateless, safe to reuse across requests
_extraction_service = ExtractionService()


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new chat conversation session.

    Args:
        body: Mode and optional title.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        ConversationResponse with id and mode.
    """
    service = ChatService(db)
    conv = await service.create_conversation(
        user_id=current_user.id,
        mode=body.mode,
        title=body.title,
    )
    await db.commit()
    return conv


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all conversations for the current user.

    Args:
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        List of ConversationResponse objects, newest first.
    """
    service = ChatService(db)
    return await service.list_conversations(current_user.id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    after_id: uuid.UUID | None = None,
    current_user: User = Depends(get_current_user),
):
    """
    Get messages in a conversation with optional cursor synchronization (after_id).

    Args:
        conversation_id: UUID of the conversation.
        after_id: Optional UUID anchor for client incremental synchronization.
        current_user: Authenticated user from JWT.

    Returns:
        Ordered list of MessageResponse objects.

    Raises:
        HTTPException 404: If conversation not found.
    """
    service = ChatService()
    conv = await service.get_conversation(conversation_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    messages = await service.get_messages(conversation_id, limit=100, after_id=after_id)
    return messages


async def _validate_document_access(document_ids: list[uuid.UUID] | None, user_id: uuid.UUID) -> None:
    """
    Validate that all specified document_ids exist, belong to user, and are in 'ready' status.

    Raises:
        HTTPException 404: If any document_id is not found, not owned by user, or not ready.
    """
    if not document_ids:
        return
    unique_ids = list(set(document_ids))
    async with AsyncSessionFactory() as session:
        stmt = text(
            "SELECT id FROM documents "
            "WHERE id = ANY(:ids) AND user_id = :uid AND status = 'ready'"
        )
        result = await session.execute(stmt, {"ids": unique_ids, "uid": user_id})
        ready_ids = [row[0] for row in result.fetchall()]
        if len(ready_ids) != len(unique_ids):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more documents not found or not accessible.",
            )


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Send a message and receive a streaming SSE response with decoupled DB sessions.

    Uses Server-Sent Events (text/event-stream) for real-time token delivery.
    Does NOT hold database connections during token generation.
    """
    await _validate_document_access(body.document_ids, current_user.id)
    chat_service = ChatService()
    captured: dict = {}

    async def event_generator() -> AsyncGenerator[str, None]:
        async for event_type, payload in chat_service.stream_message(
            conversation_id=body.conversation_id,
            user_id=current_user.id,
            user_level=current_user.level,
            content=body.message,
            client_message_id=body.client_message_id,
            audio_url=body.audio_url,
            audio_duration_ms=body.audio_duration_ms,
            stt_confidence=body.stt_confidence,
            document_ids=body.document_ids,
            captured=captured,
        ):
            yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        # Generator exhausted normally — schedule extraction only if completed
        if captured.get("user_message") and captured.get("assistant_text"):
            background_tasks.add_task(
                _extraction_service.extract_and_log,
                user_id=current_user.id,
                conversation_id=body.conversation_id,
                user_message=captured["user_message"],
                assistant_text=captured["assistant_text"],
                assistant_message_id=captured.get("assistant_message_id"),
            )
            logger.debug(
                "Scheduled extraction background task for user=%s session=%s msg=%s",
                current_user.id, body.conversation_id, captured.get("assistant_message_id"),
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # Disable Nginx proxy buffering
        },
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Send a message and receive the full response (non-streaming).
    """
    await _validate_document_access(body.document_ids, current_user.id)
    chat_service = ChatService()
    try:
        assistant_msg, chunks = await chat_service.send_message(
            conversation_id=body.conversation_id,
            user_id=current_user.id,
            user_level=current_user.level,
            content=body.message,
            client_message_id=body.client_message_id,
            audio_url=body.audio_url,
            audio_duration_ms=body.audio_duration_ms,
            stt_confidence=body.stt_confidence,
            document_ids=body.document_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except LLMException as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.to_client_payload())

    # Schedule background extraction
    if body.message and assistant_msg.content:
        background_tasks.add_task(
            _extraction_service.extract_and_log,
            user_id=current_user.id,
            conversation_id=body.conversation_id,
            user_message=body.message,
            assistant_text=assistant_msg.content,
            assistant_message_id=assistant_msg.id,
        )

    conv = await chat_service.get_conversation(body.conversation_id, current_user.id)
    mode = conv.mode if conv else ConversationMode.CONVERSATION

    return ChatResponse(
        conversation_id=body.conversation_id,
        message_id=assistant_msg.id,
        response=assistant_msg.content,
        mode=mode,
        status=assistant_msg.status,
        audio_url=assistant_msg.audio_url,
        corrections=assistant_msg.corrections,
        sources=[SourceChunk(**c.to_dict()) for c in chunks],
    )
