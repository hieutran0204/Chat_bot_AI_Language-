# name: circuit_breaker.py
# description: Lightweight in-memory circuit breaker for resilient Redis operations.

import asyncio
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"          # Healthy, traffic flows normally
    OPEN = "open"              # Tripped, traffic fails fast without hitting Redis
    HALF_OPEN = "half_open"    # Testing recovery, exactly one probe allowed


class RedisCircuitBreaker:
    """
    In-memory circuit breaker protecting Redis dependencies from blocking request threads.

    Transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold (default: 3).
    - OPEN -> HALF_OPEN: When recovery_time_seconds (default: 30s) has elapsed.
    - HALF_OPEN -> CLOSED: When a single probe request succeeds.
    - HALF_OPEN -> OPEN: When the single probe request fails.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_time_seconds: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_time_seconds = recovery_time_seconds
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._probe_lock = asyncio.Lock()
        self._is_probing = False
        self._probe_started = 0.0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, updating if recovery time elapsed."""
        now = time.monotonic()
        if self._state == CircuitState.OPEN:
            if (now - self._last_state_change) >= self.recovery_time_seconds:
                self._state = CircuitState.HALF_OPEN
                self._is_probing = False
                self._probe_started = 0.0
                logger.info("RedisCircuitBreaker: tripped OPEN -> HALF_OPEN (probing recovery).")
        return self._state

    def allow_request(self) -> bool:
        """Determine whether an operation is allowed to hit Redis."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.OPEN:
            return False
        # HALF_OPEN: allow probe if no probe running or if previous probe timed out
        now = time.monotonic()
        if not self._is_probing or (now - self._probe_started) >= self.recovery_time_seconds:
            self._is_probing = True
            self._probe_started = now
            return True
        return False

    def record_success(self) -> None:
        """Record successful Redis operation, closing the circuit."""
        if self._state != CircuitState.CLOSED:
            logger.info("RedisCircuitBreaker: probe succeeded, circuit CLOSED (Redis restored).")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._is_probing = False

    def record_failure(self, error: Exception | None = None) -> None:
        """Record Redis failure, incrementing threshold or tripping circuit OPEN."""
        self._failure_count += 1
        now = time.monotonic()
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._last_state_change = now
            self._is_probing = False
            logger.warning(
                "RedisCircuitBreaker: circuit tripped OPEN (failures=%d, error=%s). Will retry in %.1fs",
                self._failure_count, error, self.recovery_time_seconds,
            )


# Process-wide shared singleton circuit breaker
_global_circuit_breaker: RedisCircuitBreaker | None = None


def get_redis_circuit_breaker() -> RedisCircuitBreaker:
    """Retrieve the shared Redis circuit breaker singleton."""
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = RedisCircuitBreaker()
    return _global_circuit_breaker
