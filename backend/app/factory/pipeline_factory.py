# name: pipeline_factory.py
# description: Dependency injection factory — reads settings and wires the correct
#              adapters into a fully configured IRagPipeline instance.
#              Central place for all provider selection logic (LLM, embedder, vector store).

from __future__ import annotations

import logging
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.interfaces.embedder import IEmbedder
from app.core.interfaces.llm_provider import ILLMProvider
from app.core.interfaces.pipeline import IRagPipeline
from app.core.interfaces.vector_store import IVectorStore

logger = logging.getLogger(__name__)


# ── Embedder Factory ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def create_embedder() -> IEmbedder:
    """
    Create and cache the configured embedder adapter (process-wide singleton).

    Provider selected by `EMBEDDING_PROVIDER` setting:
    - 'huggingface': Local sentence-transformers model (recommended).
    - 'ollama': Ollama nomic-embed-text (fallback).

    Returns:
        Cached IEmbedder instance.

    Raises:
        ValueError: If an unsupported provider is configured.
    """
    provider = settings.embedding_provider.lower()

    if provider == "huggingface":
        from app.adapters.embedders.huggingface_embedder import HuggingFaceEmbedder
        logger.info("Embedder: HuggingFace (%s)", settings.huggingface_embed_model)
        return HuggingFaceEmbedder(model_name=settings.huggingface_embed_model)

    if provider == "ollama":
        from app.adapters.embedders.ollama_embedder import OllamaEmbedder
        logger.info(
            "Embedder: Ollama (%s @ %s)",
            settings.ollama_embed_model,
            settings.ollama_base_url,
        )
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    raise ValueError(
        f"Unknown embedding provider: {provider!r}. Use 'huggingface' or 'ollama'."
    )


# ── LLM Factory ───────────────────────────────────────────────────────────────

def create_llm() -> ILLMProvider:
    """
    Create the configured LLM provider adapter.

    A new instance is created per call (stateless HTTP clients).
    Provider selected by `LLM_PROVIDER` setting:
    - 'ollama': Local Ollama instance (default).
    - 'huggingface': HuggingFace Inference API (cloud fallback).

    Returns:
        ILLMProvider instance.

    Raises:
        ValueError: If an unsupported provider is configured.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        from app.adapters.llm.ollama_llm import OllamaLLM
        logger.info("LLM: Ollama (%s)", settings.ollama_chat_model)
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
        )

    if provider == "huggingface":
        from app.adapters.llm.huggingface_llm import HuggingFaceLLM
        logger.info("LLM: HuggingFace Inference API")
        return HuggingFaceLLM(api_key=settings.huggingface_api_key)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Use 'ollama' or 'huggingface'."
    )


# ── Vector Store Factory ──────────────────────────────────────────────────────

def create_vector_store(db: AsyncSession) -> IVectorStore:
    """
    Create the configured vector store adapter for a given DB session.

    Currently only pgvector is supported. Future adapters (Qdrant, Weaviate)
    can be added here without touching pipeline or service code.

    Args:
        db: Async SQLAlchemy session (injected per-request).

    Returns:
        IVectorStore instance scoped to the provided DB session.
    """
    from app.adapters.vector_stores.pgvector_store import PgVectorStore
    return PgVectorStore(db)


# ── Pipeline Factory ──────────────────────────────────────────────────────────

def create_pipeline(db: AsyncSession, pipeline_type: str = "naive") -> IRagPipeline:
    """
    Wire and return a fully configured RAG pipeline.

    This is the single entry point for obtaining a pipeline instance.
    All dependency resolution (embedder, LLM, vector store) happens here.

    Args:
        db: Async SQLAlchemy session (injected per-request via FastAPI Depends).
        pipeline_type: Pipeline technique to use. Currently supported:
            - 'naive' (default): Single-stage vector retrieval + LLM generation.

    Returns:
        Fully wired IRagPipeline instance.

    Raises:
        ValueError: If an unsupported pipeline_type is requested.
    """
    embedder = create_embedder()
    llm = create_llm()
    vector_store = create_vector_store(db)

    if pipeline_type == "naive":
        from app.pipelines.naive_rag.pipeline import NaiveRagPipeline
        logger.info("PipelineFactory: wiring NaiveRagPipeline")
        return NaiveRagPipeline(
            llm=llm,
            vector_store=vector_store,
            embedder=embedder,
            top_k=settings.retrieval_top_k,
            similarity_threshold=settings.similarity_threshold,
            history_limit=settings.conversation_history_limit,
        )

    raise ValueError(
        f"Unknown pipeline type: {pipeline_type!r}. Supported: ['naive']"
    )
