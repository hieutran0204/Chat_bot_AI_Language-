# name: ollama_llm.py
# description: ILLMProvider adapter for a locally running Ollama instance.
#              Supports both streaming (SSE-compatible) and non-streaming chat.

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings
from app.core.exceptions import (
    LLMException,
    LLMInternalError,
    LLMTimeout,
    LLMUnavailable,
)
from app.core.interfaces.llm_provider import (
    ILLMProvider,
    MessageInput,
    normalize_messages,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
LLMError = LLMException


class OllamaLLM(ILLMProvider):
    """
    Async LLM adapter for a locally running Ollama instance.

    Communicates with the Ollama HTTP API (/api/chat) for both
    streaming and non-streaming generation.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        ttft_timeout: float = 90.0,
        inter_chunk_timeout: float = 25.0,
        num_ctx: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ttft_timeout = ttft_timeout
        self.inter_chunk_timeout = inter_chunk_timeout
        self.num_ctx = num_ctx or getattr(settings, "ollama_num_ctx", 8192)

    def _get_client_timeout(self) -> httpx.Timeout:
        """Create a client timeout config with generous read allowance for model cold-start TTFT."""
        return httpx.Timeout(
            connect=10.0,
            read=self.ttft_timeout,
            write=10.0,
            pool=10.0,
        )

    async def chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat request to Ollama and return the full response."""
        normalized = normalize_messages(messages)
        payload = {
            "model": self.model,
            "messages": normalized,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        timeout = self._get_client_timeout()
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                raise LLMUnavailable(
                    message="Could not connect to Ollama server.",
                    detail=str(exc),
                ) from exc
            except (httpx.ReadTimeout, asyncio.TimeoutError) as exc:
                raise LLMTimeout(
                    message="Ollama generation timed out.",
                    detail=str(exc),
                ) from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise LLMInternalError(
                        message=f"Ollama model '{self.model}' not found on server.",
                        detail=str(exc),
                    ) from exc
                raise LLMUnavailable(
                    message=f"Ollama returned HTTP error {exc.response.status_code}.",
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                raise LLMInternalError(detail=str(exc)) from exc

    async def stream_chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat tokens with TTFT timeout (90s) and per-token inter-chunk timeout (25s).
        """
        normalized = normalize_messages(messages)
        payload = {
            "model": self.model,
            "messages": normalized,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        timeout = self._get_client_timeout()
        first_token_received = False

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()

                    line_iterator = response.aiter_lines().__aiter__()

                    saw_done = False
                    while True:
                        try:
                            # Apply inter-chunk timeout after the first token arrives
                            if first_token_received:
                                line = await asyncio.wait_for(
                                    line_iterator.__anext__(),
                                    timeout=self.inter_chunk_timeout,
                                )
                            else:
                                line = await line_iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError as exc:
                            raise LLMTimeout(
                                message="Ollama stalled between tokens.",
                                detail=f"Exceeded {self.inter_chunk_timeout}s inter-chunk timeout.",
                            ) from exc

                        if line.strip():
                            chunk = json.loads(line)
                            if "error" in chunk:
                                raise LLMUnavailable(
                                    message="Ollama reported an error mid-stream.",
                                    detail=str(chunk["error"]),
                                )

                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                first_token_received = True
                                yield token
                            if chunk.get("done"):
                                saw_done = True
                                prompt_eval_count = chunk.get("prompt_eval_count")
                                if prompt_eval_count:
                                    logger.debug("Ollama prompt_eval_count: %s / %s", prompt_eval_count, self.num_ctx)
                                    if prompt_eval_count >= self.num_ctx * 0.90:
                                        logger.warning(
                                            "Prompt eval count %s approached or exceeded 90%% of num_ctx %s",
                                            prompt_eval_count,
                                            self.num_ctx,
                                        )
                                break

                    if not saw_done:
                        raise LLMUnavailable(
                            message="Ollama stream ended unexpectedly.",
                            detail="Stream terminated without final 'done' flag.",
                        )

            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                raise LLMUnavailable(
                    message="Ollama server connection failed.",
                    detail=str(exc),
                ) from exc
            except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                raise LLMUnavailable(
                    message="Ollama server dropped stream connection unexpectedly.",
                    detail=str(exc),
                ) from exc
            except httpx.PoolTimeout as exc:
                raise LLMUnavailable(
                    message="Ollama client connection pool is exhausted.",
                    detail=str(exc),
                ) from exc
            except httpx.ReadTimeout as exc:
                raise LLMTimeout(
                    message="Ollama exceeded response timeout before sending tokens.",
                    detail=str(exc),
                ) from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise LLMInternalError(
                        message=f"Model '{self.model}' not found in Ollama.",
                        detail=str(exc),
                    ) from exc
                raise LLMUnavailable(
                    message=f"Ollama stream error {exc.response.status_code}.",
                    detail=str(exc),
                ) from exc
            except (LLMTimeout, LLMUnavailable, LLMInternalError):
                raise
            except (asyncio.CancelledError, GeneratorExit):
                # Starlette client disconnection: let it unwind cleanly to exit context manager
                raise
            except Exception as exc:
                raise LLMInternalError(
                    message="Unexpected failure in Ollama stream.",
                    detail=str(exc),
                ) from exc
