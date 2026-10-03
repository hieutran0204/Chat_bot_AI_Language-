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
        timeout: Request timeout in seconds (default: 30.0).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string via Ollama.

        Uses an explicit ollama.Client pointed at self.base_url so that
        custom host configurations (remote server, non-default port) are
        honoured instead of falling back to the library's default localhost client.

        Args:
            text: Input text to embed.

        Returns:
            Float list of embedding values.

        Raises:
            EmbedderError: If Ollama is unreachable or returns an error.
        """
        import ollama

        try:
            logger.debug("Ollama embed: model=%s base_url=%s text_len=%d", self.model, self.base_url, len(text))
            # Use an explicit Client so self.base_url is always respected.
            # The module-level ollama.embeddings() uses a default Client that reads
            # OLLAMA_HOST from the environment and would silently ignore self.base_url.
            client = ollama.Client(host=self.base_url, timeout=self.timeout)
            response = client.embeddings(model=self.model, prompt=text)
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
