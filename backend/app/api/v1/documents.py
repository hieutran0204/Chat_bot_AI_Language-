# name: documents.py (API router)
# description: Document management endpoints — upload, list, delete.
#              Ingestion runs as a background task after returning 202.

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentResponse)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a document for RAG ingestion.

    Returns 202 immediately after saving the file. Ingestion
    (chunking + embedding) runs asynchronously in the background.
    Poll GET /documents to check when status changes to 'ready'.

    Supports: PDF, DOCX, TXT (max 20 MB).

    Args:
        file: Uploaded file via multipart/form-data.
        background_tasks: FastAPI background task runner.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        DocumentResponse with id and status='processing'.

    Raises:
        HTTPException 400: If file type or size is invalid.
    """
    service = DocumentService(db)

    try:
        file_bytes = await file.read()
        file_path = await service.save_file(file_bytes, file.filename, current_user.id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    ext = file_path.suffix.lstrip(".").lower()
    document = await service.create_record(
        user_id=current_user.id,
        filename=file.filename,
        file_type=ext,
        file_path=file_path,
    )
    await db.commit()

    # Run ingestion in the background (does not block response)
    background_tasks.add_task(service.ingest, document)

    logger.info(
        "Upload accepted: doc=%s user=%s filename=%s",
        document.id, current_user.id, file.filename,
    )
    return document


@router.get("/", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List all documents uploaded by the current user.

    Args:
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        List of DocumentResponse objects ordered by upload date desc.
    """
    service = DocumentService(db)
    docs = await service.list_documents(current_user.id)
    return docs


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a document and all its vector chunks.

    Args:
        document_id: UUID of the document to delete.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Raises:
        HTTPException 404: If document not found or not owned by user.
    """
    service = DocumentService(db)
    deleted = await service.delete_document(document_id, current_user.id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get a single document's status and metadata.

    Use this to poll ingestion status (processing → ready | failed).

    Args:
        document_id: UUID of the document.
        db: Injected async database session.
        current_user: Authenticated user from JWT.

    Returns:
        DocumentResponse with current status.

    Raises:
        HTTPException 404: If document not found or not owned by user.
    """
    service = DocumentService(db)
    doc = await service.get_document(document_id, current_user.id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc
