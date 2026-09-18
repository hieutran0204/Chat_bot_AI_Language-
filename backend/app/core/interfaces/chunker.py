# name: chunker.py
# description: Abstract interface (port) for document parsing and text chunking strategies.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextChunk:
    """
    A single text chunk extracted from a document.

    Attributes:
        content: The raw chunk text.
        chunk_index: Zero-based position of this chunk within the document.
        metadata: Source traceability info (page, section, filename, etc.).
    """

    content: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class IChunker(ABC):
    """
    Port interface for document parsing and chunking strategies.

    Implementations handle specific formats (PDF, DOCX, TXT, Markdown…)
    and apply various splitting strategies (recursive character, sentence, semantic, etc.).
    """

    @abstractmethod
    def chunk_file(self, file_path: str | Path) -> list[TextChunk]:
        """
        Parse a document file and split it into overlapping text chunks.

        Args:
            file_path: Absolute or relative path to the document file.

        Returns:
            Ordered list of TextChunk objects ready for embedding.

        Raises:
            ValueError: If the file extension is unsupported.
            FileNotFoundError: If the file does not exist.
        """

    @abstractmethod
    def chunk_text(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        """
        Split a raw text string into chunks (no file I/O).

        Useful for chunking text already in memory (e.g. web scrape, API response).

        Args:
            text: The raw input text.
            metadata: Optional metadata to attach to every resulting chunk.

        Returns:
            Ordered list of TextChunk objects.
        """
