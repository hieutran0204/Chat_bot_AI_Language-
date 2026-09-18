# name: embedder.py
# description: Embedding layer supporting HuggingFace sentence-transformers (primary)
#              and Ollama nomic-embed-text (fallback), selected via EMBEDDING_PROVIDER env.

import logging
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """Raised when an embedding request fails on all providers."""


# ── HuggingFace Embedder ──────────────────────────────────────────────────────

class HuggingFaceEmbedder:
    """
    Generates embeddings using sentence-transformers via HuggingFace.

    The model is loaded locally on first call and cached in memory.
    No API key required for local inference — HUGGINGFACE_API_KEY is
    only used if you switch to the Inference API endpoint.

    Args:
        model_name: HuggingFace model ID (e.g. 'BAAI/bge-large-en-v1.5').
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading HuggingFace embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded (dim=%d)", self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Float list of embedding values.
        """
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in one batched call for efficiency.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]


# ── Ollama Embedder ───────────────────────────────────────────────────────────

class OllamaEmbedder:
    """
    Generates embeddings using Ollama's local embedding model.

    Requires Ollama to be running at OLLAMA_BASE_URL with
    the OLLAMA_EMBED_MODEL pulled (e.g. 'ollama pull nomic-embed-text').

    Args:
        base_url: Ollama server URL.
        model: Embedding model name.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string via Ollama.

        Args:
            text: Input text to embed.

        Returns:
            Float list of embedding values.

        Raises:
            EmbedderError: If Ollama is unreachable or returns an error.
        """
        import ollama

        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            raise EmbedderError(f"Ollama embedding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts by calling Ollama sequentially.

        Ollama does not support native batch embedding, so this
        iterates over texts. For large ingestion jobs, prefer HuggingFace.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(text) for text in texts]


# ── Provider Factory ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbedder | OllamaEmbedder:
    """
    Return the configured embedder instance (cached singleton).

    Provider is selected by the EMBEDDING_PROVIDER setting:
    - 'huggingface': Uses local sentence-transformers model.
    - 'ollama': Uses Ollama's nomic-embed-text model.

    Returns:
        Embedder instance ready to use.
    """
    provider = settings.embedding_provider.lower()

    if provider == "huggingface":
        logger.info("Embedder: HuggingFace (%s)", settings.huggingface_embed_model)
        return HuggingFaceEmbedder(model_name=settings.huggingface_embed_model)

    if provider == "ollama":
        logger.info(
            "Embedder: Ollama (%s @ %s)",
            settings.ollama_embed_model,
            settings.ollama_base_url,
        )
        return OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'huggingface' or 'ollama'.")
