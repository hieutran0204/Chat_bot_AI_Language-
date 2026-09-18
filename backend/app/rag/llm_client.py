# name: llm_client.py
# description: LLM client supporting Ollama (primary) and HuggingFace Inference API (fallback).
#              Provides both streaming and non-streaming generation with chat history support.

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Type alias for a chat message dict
ChatMessage = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class LLMError(Exception):
    """Raised when LLM generation fails on all configured providers."""


# ── Ollama Client ─────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Async LLM client for a locally running Ollama instance.

    Supports streaming chat completion via Server-Sent Events.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434).
        model: Chat model name (e.g. 'llama3.1:8b', 'mistral:7b').
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat request to Ollama and return the full response.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated assistant message string.

        Raises:
            LLMError: If Ollama returns an error or is unreachable.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat", json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
            except httpx.HTTPStatusError as exc:
                raise LLMError(f"Ollama HTTP error {exc.response.status_code}: {exc}") from exc
            except Exception as exc:
                raise LLMError(f"Ollama request failed: {exc}") from exc

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response from Ollama token by token.

        Suitable for Server-Sent Events in FastAPI endpoints.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.

        Raises:
            LLMError: If Ollama is unreachable or returns an error.
        """
        import json

        payload = {
            "model": self.model,
            "messages": messages,
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


# ── HuggingFace Inference API Client ─────────────────────────────────────────

class HuggingFaceClient:
    """
    LLM client using HuggingFace Inference API as a fallback.

    Suitable when Ollama is unavailable or for lightweight cloud fallback.
    Requires HUGGINGFACE_API_KEY and a text-generation model endpoint.

    Args:
        api_key: HuggingFace API token.
        model: HuggingFace model ID (must support text-generation).
    """

    BASE_URL = "https://api-inference.huggingface.co/models"

    def __init__(self, api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        self.api_key = api_key
        self.model = model

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat-formatted request to HuggingFace Inference API.

        Converts the messages list to a single prompt string using
        the '<s>[INST] ... [/INST]' Mistral format.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Generated text string.

        Raises:
            LLMError: If the API returns an error.
        """
        # Build prompt from messages (Mistral instruct format)
        prompt = self._build_prompt(messages)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.BASE_URL}/{self.model}", headers=headers, json=payload
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data[0].get("generated_text", "")
                raise LLMError(f"Unexpected HuggingFace response format: {data}")
            except httpx.HTTPStatusError as exc:
                raise LLMError(f"HuggingFace API error {exc.response.status_code}: {exc}") from exc
            except Exception as exc:
                raise LLMError(f"HuggingFace request failed: {exc}") from exc

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """
        Convert a list of chat messages to a Mistral instruct prompt string.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            Formatted prompt string.
        """
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                prompt_parts.append(f"{content} [/INST] ")
            elif role == "assistant":
                prompt_parts.append(f"{content}</s><s>[INST] ")
        return "".join(prompt_parts)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Non-streaming fallback: yields the full response as a single chunk.

        HuggingFace free Inference API does not support true streaming.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Yields:
            Full response text as a single yield.
        """
        result = await self.chat(messages, temperature, max_tokens)
        yield result


# ── LLM Provider Factory ──────────────────────────────────────────────────────

def get_llm_client() -> OllamaClient | HuggingFaceClient:
    """
    Return the configured LLM client.

    Provider is determined by LLM_PROVIDER setting:
    - 'ollama': Local Ollama instance (default).
    - 'huggingface': HuggingFace Inference API (fallback).

    Returns:
        LLM client instance.

    Raises:
        ValueError: If an unknown provider is configured.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        logger.info("LLM: Ollama (%s)", settings.ollama_chat_model)
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
        )

    if provider == "huggingface":
        logger.info("LLM: HuggingFace Inference API")
        return HuggingFaceClient(api_key=settings.huggingface_api_key)

    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'ollama' or 'huggingface'.")
