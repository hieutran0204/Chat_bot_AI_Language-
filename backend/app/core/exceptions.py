# name: exceptions.py
# description: Domain-specific exceptions for LLM operations, chat streaming, and resilient error contracts.

from typing import Any


class LLMException(Exception):
    """
    Base exception for all LLM provider and streaming failures.

    Provides a clean mapping to client-facing SSE error payloads without
    leaking internal stack traces or raw library exceptions.
    """

    def __init__(
        self,
        message: str = "An error occurred during LLM generation.",
        code: str = "INTERNAL",
        retryable: bool = False,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable
        self.detail = detail or message

    def to_client_payload(self) -> dict[str, Any]:
        """Generate client-safe SSE error payload."""
        return {
            "code": self.code,
            "retryable": self.retryable,
            "message": self.message,
        }


class LLMTimeout(LLMException):
    """Raised when the LLM times out during connection, TTFT, or token stream."""

    def __init__(
        self,
        message: str = "The AI model took too long to generate tokens.",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="LLM_TIMEOUT",
            retryable=True,
            detail=detail,
        )


class LLMUnavailable(LLMException):
    """Raised when the LLM service is offline, cold-starting, or dropping TCP connections."""

    def __init__(
        self,
        message: str = "The AI service is temporarily unavailable. Please retry shortly.",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="LLM_UNAVAILABLE",
            retryable=True,
            detail=detail,
        )


class LLMRateLimited(LLMException):
    """Raised when a user exceeds the allowed message frequency."""

    def __init__(
        self,
        message: str = "You are sending messages too quickly. Please slow down.",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="RATE_LIMITED",
            retryable=False,
            detail=detail,
        )


class LLMInternalError(LLMException):
    """Raised for unexpected internal pipeline or provider errors."""

    def __init__(
        self,
        message: str = "An unexpected error occurred while processing your request.",
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="INTERNAL",
            retryable=False,
            detail=detail,
        )
