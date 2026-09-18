# name: chat.py (API router)
# description: Chat endpoints — create conversations, send messages with SSE streaming,
#              and retrieve conversation history.

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.conversation import ConversationMode
from app.models.user import User
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    SourceChunk,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all messages in a conversation (chat history).

    Args:
        conversation_id: UUID of the conversation.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        Ordered list of MessageResponse objects.

    Raises:
        HTTPException 404: If conversation not found.
    """
    service = ChatService(db)
    conv = await service.get_conversation(conversation_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    messages = await service.get_messages(conversation_id, limit=100)
    return messages


@router.post("/stream")
async def stream_chat(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send a message and receive a streaming SSE response.

    Uses Server-Sent Events (text/event-stream) for real-time token delivery.

    SSE Event types:
    - 'sources': JSON list of retrieved chunks (sent once at start, doc_qa/grammar only).
    - 'token': Individual text token from the LLM.
    - 'done': Signals the end of the stream with the full message_id.
    - 'error': Sent if the pipeline fails mid-stream.

    Args:
        body: ChatRequest with conversation_id, message, and optional document_ids.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        StreamingResponse with text/event-stream content type.

    Raises:
        HTTPException 404: If the conversation is not found.
    """
    chat_service = ChatService(db)
    conv = await chat_service.get_conversation(body.conversation_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    # Persist user message
    await chat_service.save_user_message(
        conversation_id=body.conversation_id,
        content=body.message,
        audio_url=body.audio_url,
        audio_duration_ms=body.audio_duration_ms,
        stt_confidence=body.stt_confidence,
    )
    history = await chat_service.get_history(body.conversation_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        """
        Inner generator that runs the RAG pipeline and yields SSE events.

        Flow:
        1. Start RAG pipeline stream.
        2. On first yield with chunks → send 'sources' event.
        3. Yield each token as a 'token' event.
        4. After stream completes → persist assistant message, send 'done'.
        5. On error → send 'error' event.
        """
        pipeline = RAGPipeline(db)
        full_response: list[str] = []
        final_chunks: list[RetrievedChunk] = []

        try:
            async for token, chunks in pipeline.run_stream(
                user_message=body.message,
                mode=conv.mode,
                user_id=current_user.id,
                user_level=current_user.level,
                conversation_history=history,
                document_ids=body.document_ids,
            ):
                # First chunk batch — send sources event
                if chunks:
                    final_chunks = chunks
                    sources_payload = [c.to_dict() for c in chunks]
                    yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

                # Yield each token
                full_response.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            # Persist assistant response
            assistant_msg = await chat_service.save_assistant_message(
                conversation_id=body.conversation_id,
                content="".join(full_response),
                sources=final_chunks,
                user_id=current_user.id,
            )
            await chat_service.update_progress(current_user.id, conv.mode)
            await db.commit()

            yield f"event: done\ndata: {json.dumps({'message_id': str(assistant_msg.id)})}\n\n"

        except Exception as exc:
            logger.exception("Stream error for user=%s: %s", current_user.id, exc)
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send a message and receive the full response (non-streaming).

    Useful for testing, simple clients, or Flutter fallback.

    Args:
        body: ChatRequest with conversation_id, message, and optional document_ids.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        ChatResponse with full response text and sources.

    Raises:
        HTTPException 404: If the conversation is not found.
    """
    chat_service = ChatService(db)
    conv = await chat_service.get_conversation(body.conversation_id, current_user.id)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")

    await chat_service.save_user_message(
        conversation_id=body.conversation_id,
        content=body.message,
        audio_url=body.audio_url,
        audio_duration_ms=body.audio_duration_ms,
        stt_confidence=body.stt_confidence,
    )
    history = await chat_service.get_history(body.conversation_id)

    pipeline = RAGPipeline(db)
    response_text, chunks = await pipeline.run(
        user_message=body.message,
        mode=conv.mode,
        user_id=current_user.id,
        user_level=current_user.level,
        conversation_history=history,
        document_ids=body.document_ids,
    )

    assistant_msg = await chat_service.save_assistant_message(
        conversation_id=body.conversation_id,
        content=response_text,
        sources=chunks,
        user_id=current_user.id,
    )
    await chat_service.update_progress(current_user.id, conv.mode)
    await db.commit()

    return ChatResponse(
        conversation_id=body.conversation_id,
        message_id=assistant_msg.id,
        response=response_text,
        mode=conv.mode,
        audio_url=assistant_msg.audio_url,
        corrections=assistant_msg.corrections,
        sources=[SourceChunk(**c.to_dict()) for c in chunks],
    )
