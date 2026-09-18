# name: huggingface_llm.py
# description: ILLMProvider adapter for the HuggingFace Inference API.
#              Used as fallback when Ollama is unavailable.
#              Note: Free tier does not support true streaming — stream_chat() yields full response.

import logging
from collections.abc import AsyncGenerator

import httpx

from app.core.interfaces.llm_provider import ChatMessage, ILLMProvider

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when HuggingFace LLM generation fails."""


class HuggingFaceLLM(ILLMProvider):
    """
    LLM adapter using the HuggingFace Inference API.

    Suitable as a cloud fallback when Ollama is down or unavailable.
    Requires a valid HUGGINGFACE_API_KEY and a text-generation model endpoint.

    Prompt format: Mistral instruct (<s>[INST]…[/INST]) by default.
    Override _build_prompt() for other model families.

    Args:
        api_key: HuggingFace API bearer token.
        model: HuggingFace model ID supporting text-generation inference.
    """

    BASE_URL = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    ) -> None:
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

        Converts the messages list to a Mistral instruct prompt string internally.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Generated text string.

        Raises:
            LLMError: If the API returns an error.
        """
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
                raise LLMError(
                    f"HuggingFace API error {exc.response.status_code}: {exc}"
                ) from exc
            except Exception as exc:
                raise LLMError(f"HuggingFace request failed: {exc}") from exc

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Non-streaming fallback: yields the full response as a single chunk.

        HuggingFace free Inference API does not support true streaming.
        The full response is fetched then yielded in one shot.

        Args:
            messages: List of role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Yields:
            Full response text as a single yield.
        """
        logger.debug("HuggingFace stream_chat: no true streaming, returning full response")
        result = await self.chat(messages, temperature, max_tokens)
        yield result

    def _build_prompt(self, messages: list[ChatMessage]) -> str:
        """
        Convert chat messages to a Mistral instruct prompt string.

        Args:
            messages: List of {role, content} dicts.

        Returns:
            Formatted prompt string compatible with Mistral instruct models.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(f"{content} [/INST] ")
            elif role == "assistant":
                parts.append(f"{content}</s><s>[INST] ")
        return "".join(parts)
