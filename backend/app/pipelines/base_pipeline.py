# name: base_pipeline.py
# description: Abstract base class for RAG pipelines — implements IRagPipeline and provides
#              shared helper methods (context formatting, message assembly, non-stream run()).
#              Concrete pipelines extend this and override run_stream().

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator

from app.core.interfaces.llm_provider import ChatMessage, ILLMProvider
from app.core.interfaces.pipeline import IRagPipeline
from app.core.interfaces.vector_store import IVectorStore, RetrievedChunk
from app.models.conversation import ConversationMode

logger = logging.getLogger(__name__)

# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {
    ConversationMode.CONVERSATION: """You are an enthusiastic English conversation partner.
The learner is a {level}-level Vietnamese speaker practicing spoken English.

Rules:
- Respond naturally and conversationally. Keep replies to 2-4 sentences.
- Gently correct significant grammar mistakes by echoing the correct form.
- Always end with a follow-up question to keep the conversation going.
- Never lecture — keep the tone warm and encouraging.
""",

    ConversationMode.GRAMMAR: """You are a patient English grammar expert for Vietnamese learners.
{context_section}

Rules:
- Explain grammar rules clearly using simple language.
- Always provide 2-3 concrete examples.
- When helpful, briefly note how the rule differs from Vietnamese grammar.
- End with a short practice exercise for the learner.
""",

    ConversationMode.VOCABULARY: """You are a friendly English vocabulary coach for Vietnamese learners.

Rules:
- Provide the word's definition, pronunciation tip, and part of speech.
- Give 3 example sentences showing it used in different contexts.
- Mention common collocations or phrases.
- Note any common mistakes Vietnamese speakers make with this word.
""",

    ConversationMode.DOCUMENT_QA: """You are a helpful study assistant for an English learner.
Answer ONLY using the information in the provided context below.

Context from study materials:
{context_section}

Rules:
- Base your answer strictly on the context. Do NOT use outside knowledge.
- If the answer is not in the context, say: "I couldn't find this in your materials."
- Cite the source (e.g., "According to page 5...") when possible.
- Keep answers concise and easy to understand.
""",
}


class BasePipeline(IRagPipeline):
    """
    Abstract base for all RAG pipeline implementations.

    Provides:
    - Shared system prompt templates for the 4 chatbot modes.
    - _build_context_section(): format retrieved chunks into a numbered block.
    - _build_message_list(): assemble [system] + [history] + [user] message list.
    - run(): non-streaming implementation that collects run_stream() output.

    Subclasses must inject ILLMProvider and IVectorStore, and implement run_stream().

    Args:
        llm: LLM provider for text generation.
        vector_store: Vector store for chunk retrieval.
        history_limit: Max conversation turns to include in context (default: 5).
    """

    def __init__(
        self,
        llm: ILLMProvider,
        vector_store: IVectorStore,
        history_limit: int = 5,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._history_limit = history_limit

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _build_context_section(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a numbered context block for the system prompt.

        Args:
            chunks: Retrieved chunks from the vector store.

        Returns:
            Formatted multi-line context string, or a fallback message if empty.
        """
        if not chunks:
            return "(No relevant context found in your documents.)"

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("filename", "Unknown")
            page = chunk.metadata.get("page", "")
            page_ref = f", page {page}" if page else ""
            parts.append(f"[{i}] Source: {source}{page_ref}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    def _build_message_list(
        self,
        system_prompt: str,
        conversation_history: list[ChatMessage],
        user_message: str,
    ) -> list[ChatMessage]:
        """
        Assemble the full message list to send to the LLM.

        Structure: [system] + [last N history turns] + [current user message]

        Args:
            system_prompt: Rendered system prompt string.
            conversation_history: Previous messages (role/content dicts).
            user_message: The current user input.

        Returns:
            Complete ordered message list ready for the LLM provider.
        """
        messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
        # Limit history to avoid context window overflow (N turns = 2N messages)
        tail = conversation_history[-(self._history_limit * 2):]
        messages.extend(tail)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _render_system_prompt(
        self,
        mode: str,
        user_level: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        Render the system prompt template for a given mode.

        Args:
            mode: ConversationMode string value.
            user_level: English proficiency level (A1–C2).
            chunks: Retrieved chunks (used in context_section placeholder).

        Returns:
            Fully rendered system prompt string.
        """
        template = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS[ConversationMode.CONVERSATION])
        context_section = self._build_context_section(chunks)
        return template.format(level=user_level, context_section=context_section)

    # ── Non-streaming run() ───────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Execute the pipeline and return the full response (non-streaming).

        Collects all tokens from run_stream() into a single string.
        Useful for background tasks, testing, and non-streaming clients.

        Args:
            user_message: The user's current input.
            mode: ConversationMode string value.
            user_id: UUID of the requesting user.
            user_level: English proficiency level.
            conversation_history: Recent message history.
            document_ids: Optional document filter for retrieval.

        Returns:
            Tuple of (full_response_text, retrieved_chunks).
        """
        full_text: list[str] = []
        final_chunks: list[RetrievedChunk] = []

        async for token, chunks in self.run_stream(
            user_message, mode, user_id, user_level, conversation_history, document_ids
        ):
            full_text.append(token)
            if chunks:
                final_chunks = chunks

        return "".join(full_text), final_chunks

    async def run_stream(
        self,
        user_message: str,
        mode: str,
        user_id: uuid.UUID,
        user_level: str,
        conversation_history: list[ChatMessage],
        document_ids: list[uuid.UUID] | None = None,
    ) -> AsyncGenerator[tuple[str, list[RetrievedChunk]], None]:
        """Must be implemented by subclasses."""
        raise NotImplementedError
