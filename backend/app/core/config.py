# name: config.py
# description: Centralized application settings loaded from environment variables using Pydantic Settings.

from functools import lru_cache
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
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.5
    max_context_tokens: int = 3000
    conversation_history_limit: int = 5

    # ── File Upload ──────────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    upload_dir: str = "./uploads"
    allowed_extensions: str = "pdf,docx,txt"

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

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
