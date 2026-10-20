# name: config.py
# description: Centralized application settings loaded from environment variables using Pydantic Settings.

from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from .env file.

    All settings can be overridden by environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Language AI Chatbot"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ── Security ─────────────────────────────────────────────────────────────
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "language_ai"
    postgres_password: str
    postgres_db: str = "language_ai_db"

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # ── HuggingFace ──────────────────────────────────────────────────────────
    huggingface_api_key: str = ""
    huggingface_embed_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding_dimension: int = 768

    # ── LLM & Embedding Providers ────────────────────────────────────────────
    # "huggingface" | "ollama"
    embedding_provider: str = "huggingface"
    # "ollama" | "huggingface"
    llm_provider: str = "ollama"

    # ── RAG Configuration ────────────────────────────────────────────────────
    chunk_size: int = 100
    chunk_overlap: int = 20
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.5
    similarity_threshold: float = 0.5  # Backward compatibility alias
    max_context_tokens: int = 3000
    conversation_history_limit: int = 5
    ollama_num_ctx: int = 8192
    summarization_top_k_per_doc: int = 3
    summarization_content_word_threshold: int = 3
    query_rewriting_enabled: bool = True
    hard_fallback_enabled: bool = True
    query_rewriter_timeout_seconds: float = 2.5
    max_message_length: int = 2000

    # ── File Upload ──────────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    upload_dir: str = "./uploads"
    allowed_extensions: str = "pdf,docx,txt"

    # ── Redis & Arq Queue ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Arq & Worker Settings ────────────────────────────────────────────────
    ollama_num_parallel: int = 4
    arq_max_jobs: int | None = None
    arq_max_tries: int = 3
    arq_job_timeout: int = 330          # Must be >= job_budget + extraction + insight + 30
    job_budget_seconds: int = 180       # 3 minutes chunk loop budget
    llm_extraction_timeout: int = 60    # Per chunk extraction call
    llm_insight_timeout: int = 60       # Session insight call
    extraction_batch_size: int = 20     # Max messages per chunk
    extraction_context_window: int = 5  # Preceding messages for conversational context
    conversation_idle_minutes: int = 30 # Idle threshold to trigger extraction
    retry_backoff_minutes: int = 60     # Flat retry backoff
    max_process_attempts: int = 5       # Max dead letter attempts
    min_messages_for_insight: int = 5   # Min user messages to produce a session insight

    @property
    def effective_arq_max_jobs(self) -> int:
        """Return configured arq_max_jobs or dynamically compute from ollama_num_parallel."""
        if self.arq_max_jobs is not None:
            return self.arq_max_jobs
        return max(1, self.ollama_num_parallel // 4)

    @property
    def n_chat(self) -> int:
        """Available concurrent slots dedicated to interactive chat backpressure."""
        slots = self.ollama_num_parallel - self.effective_arq_max_jobs
        assert slots >= 1, "n_chat must be >= 1. Increase ollama_num_parallel."
        return slots

    @model_validator(mode="after")
    def validate_arq_timeout_math(self) -> "Settings":
        """Validate that ARQ_JOB_TIMEOUT allows for budget + one extra chunk + insight + margin."""
        min_required = (
            self.job_budget_seconds
            + self.llm_extraction_timeout
            + self.llm_insight_timeout
            + 30
        )
        assert self.arq_job_timeout >= min_required, (
            f"arq_job_timeout ({self.arq_job_timeout}s) must be >= {min_required}s "
            f"(budget + extraction + insight + 30s margin)"
        )
        return self

    @property
    def allowed_ext_list(self) -> list[str]:
        """Return list of allowed file extensions."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Return max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Return cached Settings instance.

    Uses lru_cache so the .env file is only parsed once per process.
    """
    return Settings()


settings = get_settings()
