# name: huggingface_embedder.py
# description: IEmbedder adapter using HuggingFace sentence-transformers (local inference).
#              Model is lazy-loaded on first call and cached in memory for the process lifetime.

import logging

from app.core.interfaces.embedder import IEmbedder

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder(IEmbedder):
    """
    Generates dense embeddings using sentence-transformers via HuggingFace.

    The model is downloaded once and loaded locally — no API key required
    for local inference. Suitable as the primary embedding provider.

    Args:
        model_name: HuggingFace model ID.
                    Default: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
                    Alternative: 'BAAI/bge-large-en-v1.5' (English-only, higher quality)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading HuggingFace embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model ready (dim=%d)", self._dimension)
        return self._model

    @property
    def dimension(self) -> int | None:
        """Return embedding dimension if model has been loaded, else None."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string using the local sentence-transformer model.

        Args:
            text: Input text to embed.

        Returns:
            Normalized float list representing the embedding vector.
        """
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in one efficient batched call.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
        """
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]
