# name: huggingface_llm.py
# description: ILLMProvider adapter for the HuggingFace Inference API.
#              Used as fallback when Ollama is unavailable.
#              Note: Free tier does not support true streaming — stream_chat() yields full response.

import asyncio
import logging
from collections.abc import AsyncGenerator

import httpx

from app.core.exceptions import (
    LLMException,
    LLMInternalError,
    LLMTimeout,
    LLMUnavailable,
)
from app.core.interfaces.llm_provider import (
    ChatMessage,
    ILLMProvider,
    MessageInput,
    normalize_messages,
)

logger = logging.getLogger(__name__)

# Maximum number of retries when the HuggingFace API returns 503 (model cold-start).
_MAX_503_RETRIES = 3
# Backoff delays in seconds for each retry attempt (index 0 = first retry).
_RETRY_BACKOFF = [15.0, 30.0, 60.0]

LLMError = LLMException


class HuggingFaceLLM(ILLMProvider):
    """
    LLM adapter using the HuggingFace Inference API.

    Suitable as a cloud fallback when Ollama is down or unavailable.
    Requires a valid HUGGINGFACE_API_KEY and a text-generation model endpoint.

    Prompt format: Mistral instruct (<s>[INST]…[/INST]) by default.
    Override _build_prompt() for other model families.

    503 handling: HuggingFace Inference API returns 503 while a model is warming
    up (cold-start). This adapter retries up to _MAX_503_RETRIES times with
    exponential backoff before raising LLMError.

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
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a chat-formatted request to HuggingFace Inference API.

        Accepts both LangChain BaseMessage objects and plain role/content dicts
        (normalised via normalize_messages for interface consistency with OllamaLLM).

        Converts the messages list to a Mistral instruct prompt string internally.
        Retries automatically on 503 (model cold-start) with backoff.

        Args:
            messages: List of BaseMessage instances or role/content message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Generated text string.

        Raises:
            LLMError: If the API returns a non-503 error or max retries exceeded.
        """
        normalized = normalize_messages(messages)
        prompt = self._build_prompt(normalized)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(_MAX_503_RETRIES + 1):
                try:
                    response = await client.post(
                        f"{self.BASE_URL}/{self.model}", headers=headers, json=payload
                    )

                    # HuggingFace returns 503 while the model is warming up (cold-start).
                    # Retry with backoff instead of treating it as a hard failure.
                    if response.status_code == 503 and attempt < _MAX_503_RETRIES:
                        wait = _RETRY_BACKOFF[attempt]
                        logger.warning(
                            "HuggingFace model cold-start (503), retrying in %.0fs "
                            "(attempt %d/%d)",
                            wait,
                            attempt + 1,
                            _MAX_503_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, list):
                        return data[0].get("generated_text", "")
                    raise LLMInternalError(f"Unexpected HuggingFace response format: {data}")

                except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                    raise LLMUnavailable(
                        message="Could not connect to HuggingFace Inference API.",
                        detail=str(exc),
                    ) from exc
                except (httpx.ReadTimeout, asyncio.TimeoutError) as exc:
                    raise LLMTimeout(
                        message="HuggingFace API request timed out.",
                        detail=str(exc),
                    ) from exc
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        raise LLMInternalError(
                            message=f"Model '{self.model}' not found on HuggingFace.",
                            detail=str(exc),
                        ) from exc
                    if exc.response.status_code == 429:
                        from app.core.exceptions import LLMRateLimited
                        raise LLMRateLimited(
                            message="HuggingFace rate limit exceeded.",
                            detail=str(exc),
                        ) from exc
                    raise LLMUnavailable(
                        message=f"HuggingFace returned HTTP {exc.response.status_code}.",
                        detail=str(exc),
                    ) from exc
                except (LLMTimeout, LLMUnavailable, LLMInternalError):
                    raise
                except Exception as exc:
                    raise LLMInternalError(f"HuggingFace request failed: {exc}") from exc

        # Exhausted all retries on 503.
        raise LLMUnavailable(
            message=f"HuggingFace model '{self.model}' unavailable after cold-start retries.",
            detail=f"Exhausted {_MAX_503_RETRIES} retries on 503.",
        )

    async def stream_chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Non-streaming fallback: yields the full response as a single chunk.

        HuggingFace free Inference API does not support true streaming.
        The full response is fetched then yielded in one shot.

        Args:
            messages: List of BaseMessage instances or role/content message dicts.
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

        Handles conversations with or without a leading system message:
        - With system:    <s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user} [/INST]
        - Without system: <s>[INST] {user} [/INST]

        Multi-turn assistant turns re-open an [INST] block for subsequent user messages.

        Args:
            messages: List of {role, content} dicts (already normalised).

        Returns:
            Formatted prompt string compatible with Mistral instruct models.
        """
        parts: list[str] = []
        has_system = any(m["role"] == "system" for m in messages)
        first_user = True  # Tracks whether <s>[INST] has been prepended yet

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System message is folded into the first [INST] block per Mistral spec.
                parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")

            elif role == "user":
                if not has_system and first_user:
                    # No system message: we must open the <s>[INST] block ourselves.
                    parts.append(f"<s>[INST] {content} [/INST] ")
                else:
                    parts.append(f"{content} [/INST] ")
                first_user = False

            elif role == "assistant":
                # Close the current turn and open the next [INST] block.
                parts.append(f"{content}</s><s>[INST] ")

        return "".join(parts)
