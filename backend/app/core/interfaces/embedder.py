# name: embedder.py
# description: Abstract interface (port) for text embedding providers.
#              Any concrete embedder (HuggingFace, Ollama, OpenAI, etc.)
#              must implement this contract.

from abc import ABC, abstractmethod


class IEmbedder(ABC):
    """
    Port interface for embedding text into dense vectors.

    Implementations must be deterministic for the same model —
    never mix embedders between ingestion and retrieval phases.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a dense vector.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single batched call.

        Prefer this over calling embed() in a loop for large ingestion jobs.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input text, in the same order.
        """

    @property
    def dimension(self) -> int | None:
        """
        Return the embedding dimension, if known at init time.

        Implementations may override this to return a concrete int.
        Returns None by default (dimension unknown until first call).
        """
        return None
