# name: chunker.py
# description: Document parsing and text chunking for PDF, DOCX, and TXT files.
#              Uses RecursiveCharacterTextSplitter to respect sentence boundaries.

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    A single text chunk extracted from a document.

    Attributes:
        content: The raw chunk text.
        chunk_index: Position of this chunk within the document.
        metadata: Source traceability info (page, section, filename, etc.).
    """

    content: str
    chunk_index: int
    metadata: dict


class DocumentChunker:
    """
    Parses documents and splits them into overlapping text chunks.

    Supports PDF (via PyMuPDF), DOCX (via python-docx), and plain TXT.
    Uses LangChain's RecursiveCharacterTextSplitter to avoid cutting mid-sentence.

    Args:
        chunk_size: Max characters per chunk.
        chunk_overlap: Character overlap between adjacent chunks.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_splitter(self):
        """Build a LangChain text splitter that prioritises paragraph/sentence breaks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_pdf(self, file_path: Path) -> list[tuple[str, dict]]:
        """
        Extract text from a PDF file page by page.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            List of (text, metadata) tuples, one per page.
        """
        import fitz  # PyMuPDF

        pages: list[tuple[str, dict]] = []
        with fitz.open(str(file_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((text, {"page": page_num, "filename": file_path.name}))

        logger.info("Parsed %d pages from PDF: %s", len(pages), file_path.name)
        return pages

    def _parse_docx(self, file_path: Path) -> list[tuple[str, dict]]:
        """
        Extract text from a DOCX file, grouped by paragraphs.

        Args:
            file_path: Absolute path to the DOCX file.

        Returns:
            List of (text, metadata) tuples.
        """
        from docx import Document

        doc = Document(str(file_path))
        full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        logger.info("Parsed DOCX: %s", file_path.name)
        return [(full_text, {"filename": file_path.name})]

    def _parse_txt(self, file_path: Path) -> list[tuple[str, dict]]:
        """
        Read plain text file content.

        Args:
            file_path: Absolute path to the TXT file.

        Returns:
            Single-element list with full text and metadata.
        """
        text = file_path.read_text(encoding="utf-8", errors="replace")
        logger.info("Parsed TXT: %s", file_path.name)
        return [(text, {"filename": file_path.name})]

    # ── Main Chunking Logic ───────────────────────────────────────────────────

    def chunk_file(self, file_path: str | Path) -> list[TextChunk]:
        """
        Parse a document and split it into overlapping text chunks.

        Dispatches to the appropriate parser based on file extension,
        then applies RecursiveCharacterTextSplitter.

        Args:
            file_path: Path to the document file.

        Returns:
            Ordered list of TextChunk objects ready for embedding.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        ext = path.suffix.lower().lstrip(".")
        parsers = {"pdf": self._parse_pdf, "docx": self._parse_docx, "txt": self._parse_txt}

        if ext not in parsers:
            raise ValueError(f"Unsupported file type: .{ext}")

        raw_sections = parsers[ext](path)
        splitter = self._get_splitter()
        chunks: list[TextChunk] = []

        for text, meta in raw_sections:
            splits = splitter.split_text(text)
            for split_text in splits:
                clean = split_text.strip()
                if len(clean) < 20:  # Skip noise / header fragments
                    continue
                chunks.append(
                    TextChunk(
                        content=clean,
                        chunk_index=len(chunks),
                        metadata=meta,
                    )
                )

        logger.info(
            "Chunked %s → %d chunks (size=%d overlap=%d)",
            path.name,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
