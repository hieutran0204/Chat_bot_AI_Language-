# name: __init__.py
# description: Core interfaces package — abstract "ports" for the RAG system.
#              All adapters must implement these contracts.

from app.core.interfaces.chunker import IChunker
from app.core.interfaces.embedder import IEmbedder
from app.core.interfaces.llm_provider import ILLMProvider
from app.core.interfaces.pipeline import IRagPipeline
from app.core.interfaces.vector_store import IVectorStore

__all__ = [
    "IEmbedder",
    "ILLMProvider",
    "IVectorStore",
    "IChunker",
    "IRagPipeline",
]
