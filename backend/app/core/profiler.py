# name: profiler.py
# description: Execution timing and latency profiling utilities for chatbot requests,
#              measuring embedding, vector retrieval, LLM prefill, token generation, and DB I/O.

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Generator

# ── Dedicated Profiling Logger Setup ──────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "chat_latency.log"

profiling_logger = logging.getLogger("chat_profiler")
profiling_logger.setLevel(logging.INFO)

# Avoid duplicate handlers on hot reload
if not profiling_logger.handlers:
    _file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    _file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler.setFormatter(_file_formatter)
    profiling_logger.addHandler(_file_handler)
    profiling_logger.propagate = False


class LatencyTracker:
    """
    Tracks and records execution time for every discrete phase of a chat turn.

    Phases recorded:
    1. DB/Redis User Msg & History Fetch
    2. Query Embedding Vectorization
    3. Vector Store Similarity Search (pgvector)
    4. Prompt & Context Assembly
    5. LLM Time to First Token (TTFT / Prefill)
    6. LLM Token Generation Streaming (and tok/s calculation)
    7. DB/Redis Assistant Msg Save & Memory Cache

    Outputs formatted execution summaries to `logs/chat_latency.log` and app logs.
    """

    def __init__(self, conversation_id: str | Any, mode: str = "conversation") -> None:
        """
        Initialize the tracker for a conversation request.

        Args:
            conversation_id: Identifier of the active conversation.
            mode: Chatbot mode (conversation, grammar, vocabulary, doc_qa).
        """
        self.conversation_id = str(conversation_id)
        self.mode = mode
        self.start_total_time = time.perf_counter()
        self.step_timings: dict[str, float] = {}
        self.extra_info: dict[str, Any] = {}
        self.token_count: int = 0

    @contextmanager
    def track(self, step_name: str) -> Generator[None, None, None]:
        """
        Measure execution duration of a code block synchronously.

        Args:
            step_name: Descriptive label for the execution phase.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.step_timings[step_name] = self.step_timings.get(step_name, 0.0) + elapsed_ms

    def record_step(self, step_name: str, duration_ms: float) -> None:
        """
        Record a manually measured step duration in milliseconds.

        Args:
            step_name: Descriptive label for the execution phase.
            duration_ms: Duration in milliseconds.
        """
        self.step_timings[step_name] = duration_ms

    def add_metric(self, key: str, value: Any) -> None:
        """
        Store arbitrary contextual metadata (e.g. chunk count, token count).

        Args:
            key: Metadata key name.
            value: Metadata value.
        """
        self.extra_info[key] = value

    def increment_tokens(self, count: int = 1) -> None:
        """Increment generated token counter."""
        self.token_count += count

    def log_summary(self) -> str:
        """
        Calculate totals, identify the primary bottleneck, and write to log file.

        Returns:
            Formatted multi-line summary string.
        """
        total_ms = (time.perf_counter() - self.start_total_time) * 1000.0
        lines = [
            "",
            "=" * 78,
            f"⏱️  [CHAT PROFILING] Conv: {self.conversation_id} | Mode: {self.mode}",
            "-" * 78,
        ]

        max_step_name = ""
        max_step_time = -1.0

        for step, ms in self.step_timings.items():
            pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
            extra = ""
            if step == "llm_generation" and self.token_count > 0:
                gen_sec = ms / 1000.0
                tok_per_sec = (self.token_count / gen_sec) if gen_sec > 0 else 0.0
                extra = f"  ({self.token_count} tokens, ~{tok_per_sec:.1f} tok/s)"
            elif step == "retrieval" and "chunks_found" in self.extra_info:
                extra = f"  ({self.extra_info['chunks_found']} chunks retrieved)"

            lines.append(f"  • {step:<26}: {ms:8.2f} ms ({pct:5.1f}%){extra}")

            if ms > max_step_time:
                max_step_time = ms
                max_step_name = step

        lines.append("-" * 78)
        lines.append(f"  🔥 TOTAL LATENCY           : {total_ms:8.2f} ms (~{total_ms / 1000.0:.2f} s)")
        if max_step_name:
            bottleneck_pct = (max_step_time / total_ms * 100.0) if total_ms > 0 else 0.0
            lines.append(f"  ⭐ PRIMARY BOTTLENECK      : {max_step_name} ({bottleneck_pct:.1f}% of total duration)")
        lines.append("=" * 78)

        summary_text = "\n".join(lines)

        # Write to dedicated chat_latency.log file
        profiling_logger.info(summary_text)

        # Also log a concise one-liner to standard console logger
        std_logger = logging.getLogger("app.services.chat_service")
        std_logger.info(
            "⏱️ [Profiling] Conv %s finished in %.2fs (Bottleneck: %s @ %.2fms)",
            self.conversation_id,
            total_ms / 1000.0,
            max_step_name,
            max_step_time,
        )

        # Log structured Phase 2 metrics
        rag_metrics = self.to_rag_metrics()
        profiling_logger.info("RAG_METRICS: %s", json.dumps(rag_metrics))

        return summary_text

    def to_rag_metrics(self) -> dict:
        """
        Return structured dictionary of the 14 standard Phase 2 privacy-safe metrics.
        No user message content or personally identifiable information is included.
        """
        return {
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "retrieval_count": self.extra_info.get("chunks_found", 0),
            "similarity_top1": self.extra_info.get("similarity_top1"),
            "similarity_mean": self.extra_info.get("similarity_mean"),
            "chunks_trimmed": self.extra_info.get("chunks_trimmed", 0),
            "history_turns_trimmed": self.extra_info.get("history_turns_trimmed", 0),
            "estimated_tokens_sent": self.extra_info.get("estimated_tokens_sent", 0),
            "prompt_eval_count_ollama": self.extra_info.get("prompt_eval_count_ollama"),
            "rewrite_triggered": self.extra_info.get("rewrite_triggered", False),
            "rewrite_timeout": self.extra_info.get("rewrite_timeout", False),
            "fallback_reason": self.extra_info.get("fallback_reason"),
            "is_fallback": self.extra_info.get("is_fallback", False),
            "ttft_ms": round(self.step_timings.get("llm_time_to_first_token", 0.0), 2),
        }
