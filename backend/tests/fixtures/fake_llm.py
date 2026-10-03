# name: fake_llm.py
# description: Configurable FakeLLM fixture for testing streaming resilience, timeouts, and cancellations.

import asyncio
from collections.abc import AsyncGenerator

from app.core.exceptions import LLMTimeout
from app.core.interfaces.llm_provider import ILLMProvider, MessageInput


class FakeLLM(ILLMProvider):
    """
    Configurable fake LLM for resilient stream testing without running actual model servers.

    Can simulate:
    - Normal token streaming with token delay
    - Hanging before first token (TTFT timeout)
    - Hanging mid-stream after N tokens
    - Raising LLMTimeout or LLMUnavailable on demand
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        delay_per_token: float = 0.01,
        hang_before_first_token_seconds: float = 0.0,
        hang_after_token_count: int | None = None,
        hang_after_token_seconds: float = 0.0,
        fail_before_stream_error: Exception | None = None,
        fail_after_token_count: int | None = None,
        fail_after_token_error: Exception | None = None,
    ) -> None:
        self.tokens = tokens or ["Hello", " ", "there!", " How", " can", " I", " assist", " you?"]
        self.delay_per_token = delay_per_token
        self.hang_before_first_token_seconds = hang_before_first_token_seconds
        self.hang_after_token_count = hang_after_token_count
        self.hang_after_token_seconds = hang_after_token_seconds
        self.fail_before_stream_error = fail_before_stream_error
        self.fail_after_token_count = fail_after_token_count
        self.fail_after_token_error = fail_after_token_error

    async def chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Return all tokens joined as a full response."""
        if self.fail_before_stream_error:
            raise self.fail_before_stream_error
        if self.hang_before_first_token_seconds > 0:
            await asyncio.sleep(self.hang_before_first_token_seconds)
        return "".join(self.tokens)

    async def stream_chat(
        self,
        messages: MessageInput,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream configured tokens according to simulated conditions."""
        if self.fail_before_stream_error:
            raise self.fail_before_stream_error

        if self.hang_before_first_token_seconds > 0:
            await asyncio.sleep(self.hang_before_first_token_seconds)

        yielded_count = 0
        for token in self.tokens:
            if self.fail_after_token_count is not None and yielded_count >= self.fail_after_token_count:
                if self.fail_after_token_error:
                    raise self.fail_after_token_error
                raise LLMTimeout("Simulated stream crash after token count")

            if self.hang_after_token_count is not None and yielded_count >= self.hang_after_token_count:
                if self.hang_after_token_seconds > 0:
                    await asyncio.sleep(self.hang_after_token_seconds)

            if self.delay_per_token > 0:
                await asyncio.sleep(self.delay_per_token)

            yield token
            yielded_count += 1
