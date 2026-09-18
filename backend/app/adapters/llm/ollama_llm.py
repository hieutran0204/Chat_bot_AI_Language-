# name: ollama_llm.py
# description: ILLMProvider adapter for a locally running Ollama instance.
#              Supports both streaming (SSE-compatible) and non-streaming chat.

import logging
from collections.abc import AsyncGenerator

import httpx

from app.core.interfaces.llm_provider import (
    ChatMessage,
    ILLMProvider,
    MessageInput,
    normalize_messages,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when LLM generation fails."""


class OllamaLLM(ILLMProvider):
    """
    Async LLM adapter for a locally running Ollama instance.

    Communicates with the Ollama HTTP API (/api/chat) for both
    streaming and non-streaming generation. Supports both LangChain BaseMessage
    objects and standard chat dicts.

    Args:
        base_url: Ollama server base URL (default: http://localhost:11434).
        model: Chat model name (e.g. 'llama3.1:8b', 'mistral:7b', 'gemma2:9b').
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat request to Ollama and return the full response.

        Args:
            messages: List of BaseMessage instances or role/content message dicts.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated assistant message string.

        Raises:
            LLMError: If Ollama returns an error or is unreachable.
        """
        normalized = normalize_messages(messages)
        payload = {
            "model": self.model,
            "messages": normalized,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
            except httpx.HTTPStatusError as exc:
                raise LLMError(
                    f"Ollama HTTP error {exc.response.status_code}: {exc}"
                ) from exc
            except Exception as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc

    async def stream_chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response from Ollama token by token.

        Args:
            messages: List of BaseMessage instances or role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.

        Raises:
            LLMError: If Ollama is unreachable or returns an error.
        """
        import json

        normalized = normalize_messages(messages)
        payload = {
            "model": self.model,
            "messages": normalized,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
            except httpx.HTTPStatusError as exc:
                raise LLMError(f"Ollama stream error: {exc}") from exc
            except Exception as exc:
                raise LLMError(f"Ollama stream failed: {exc}") from exc
