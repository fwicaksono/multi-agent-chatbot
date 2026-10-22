from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_storage, settings_dep
from app.core.config import Settings
from app.core.errors import NotFoundError
from app.db.session import get_db
from app.ingestion.storage import FileStorage
from app.repositories import documents as doc_repo
from app.schemas.api import DocumentOut, UploadResult
from app.services import agents as agent_service
from app.services import documents as doc_service

router = APIRouter(prefix="/agents/{agent_id}/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    agent_id: uuid.UUID, session: AsyncSession = Depends(get_db, scope="function")
) -> list[DocumentOut]:
    await agent_service.get_or_404(session, agent_id)
    return [DocumentOut.model_validate(d) for d in await doc_repo.list_documents(session, agent_id)]


@router.post("", response_model=UploadResult, status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    agent_id: uuid.UUID,
    files: list[UploadFile] = File(..., description="One or more PDF/DOCX files"),
    session: AsyncSession = Depends(get_db, scope="function"),
    storage: FileStorage = Depends(get_storage),
    settings: Settings = Depends(settings_dep),
) -> UploadResult:
    """Queue documents for ingestion. Returns immediately; poll the list for status."""
    await agent_service.get_or_404(session, agent_id)
    return await doc_service.upload(session, storage, settings, agent_id, files)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    agent_id: uuid.UUID,
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> DocumentOut:
    return DocumentOut.model_validate(await doc_service.get_or_404(session, agent_id, document_id))


@router.get("/{document_id}/file")
async def download_document(
    agent_id: uuid.UUID,
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> FileResponse:
    """Serve the original file inline (PDF citations deep-link with #page=N)."""
    doc = await doc_service.get_or_404(session, agent_id, document_id)
    path = Path(doc.storage_path)
    if not path.exists():
        raise NotFoundError("The original file is no longer available in storage.")
    return FileResponse(
        path, media_type=doc.content_type, filename=doc.filename, content_disposition_type="inline"
    )


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess_document(
    agent_id: uuid.UUID,
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> DocumentOut:
    return DocumentOut.model_validate(await doc_service.reprocess(session, agent_id, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    agent_id: uuid.UUID,
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db, scope="function"),
    storage: FileStorage = Depends(get_storage),
) -> Response:
    await doc_service.delete(session, storage, agent_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
