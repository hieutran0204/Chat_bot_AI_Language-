# name: ollama_embedder.py
# description: IEmbedder adapter using Ollama's local embedding model (e.g. nomic-embed-text).
#              Used as fallback when HuggingFace sentence-transformers is not preferred.

import logging

from app.core.interfaces.embedder import IEmbedder

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """Raised when an Ollama embedding request fails."""


class OllamaEmbedder(IEmbedder):
    """
    Generates embeddings using a locally running Ollama instance.

    Requires Ollama to be running at `base_url` with the target model
    already pulled (e.g. `ollama pull nomic-embed-text`).

    Note: Ollama does not support native batch embedding — embed_batch()
    calls embed() sequentially. For large ingestion jobs, prefer HuggingFaceEmbedder.

    Args:
        base_url: Ollama server base URL (default: http://localhost:11434).
        model: Embedding model name (default: 'nomic-embed-text').
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ) -> None:
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
            logger.debug("Ollama embed: model=%s text_len=%d", self.model, len(text))
            response = ollama.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            raise EmbedderError(f"Ollama embedding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts by calling Ollama sequentially.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors in the same order as input.
        """
        logger.debug("Ollama batch embed: %d texts", len(texts))
        return [self.embed(text) for text in texts]
