# name: document.py (schemas)
# description: Pydantic request/response schemas for document management and chunk ingestion.

import uuid

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """
    Document metadata returned to the API client.

    Attributes:
        id: Unique UUID of the document record.
        filename: Original file name uploaded by user.
        file_type: Extension or type of document (pdf, docx, txt).
        status: Ingestion status ('processing', 'ready', 'failed').
        chunk_count: Number of vectorized chunks stored for this document.
    """

    id: uuid.UUID
    filename: str
    file_type: str
    status: str
    chunk_count: int

    model_config = {"from_attributes": True}


class DocumentChunkResponse(BaseModel):
    """
    Metadata and content for an individual document chunk.

    Attributes:
        id: Unique UUID of the chunk record.
        chunk_index: Index of chunk within parent document.
        content: Text content of the chunk.
        metadata: Page, heading, or filename metadata dict.
    """

    id: uuid.UUID
    chunk_index: int
    content: str
    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}
