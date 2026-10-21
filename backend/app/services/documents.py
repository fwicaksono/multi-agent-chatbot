"""Document upload and lifecycle use cases."""

from __future__ import annotations

import hashlib
import uuid

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationFailed
from app.core.logging import get_logger
from app.db.models import Document, DocumentStatus
from app.ingestion.errors import IngestionError
from app.ingestion.storage import FileStorage
from app.ingestion.validation import validate_upload
from app.repositories import documents as doc_repo
from app.schemas.api import DocumentOut, UploadRejection, UploadResult

logger = get_logger(__name__)


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    # Read one byte past the limit so oversize files are detected without
    # buffering arbitrarily large uploads.
    return await upload.read(max_bytes + 1)


async def upload(
    session: AsyncSession,
    storage: FileStorage,
    settings: Settings,
    agent_id: uuid.UUID,
    files: list[UploadFile],
) -> UploadResult:
    if not files:
        raise ValidationFailed("No files were uploaded.")
    if len(files) > settings.max_files_per_upload:
        raise ValidationFailed(
            f"Too many files ({len(files)}); upload at most "
            f"{settings.max_files_per_upload} at once."
        )

    accepted: list[DocumentOut] = []
    rejected: list[UploadRejection] = []
    for upload_file in files:
        filename = (upload_file.filename or "unnamed").split("/")[-1].split("\\")[-1][:255]
        data = await _read_limited(upload_file, settings.max_upload_bytes)
        try:
            kind = validate_upload(filename, data, settings.max_upload_bytes)
        except IngestionError as exc:
            rejected.append(UploadRejection(filename=filename, code=exc.code, message=exc.message))
            continue

        sha256 = hashlib.sha256(data).hexdigest()
        if dup := await doc_repo.find_duplicate(session, agent_id, sha256):
            rejected.append(
                UploadRejection(
                    filename=filename,
                    code="duplicate_document",
                    message=f"'{filename}' is identical to '{dup.filename}', which is already "
                    "in this agent's knowledge base.",
                )
            )
            continue

        document_id = uuid.uuid4()
        path = storage.save(agent_id, document_id, kind.file_type, data)
        doc = await doc_repo.create_document(
            session,
            id=document_id,
            agent_id=agent_id,
            filename=filename,
            file_type=kind.file_type,
            content_type=kind.content_type,
            size_bytes=len(data),
            sha256=sha256,
            storage_path=str(path),
        )
        accepted.append(DocumentOut.model_validate(doc))
        logger.info(
            "document.queued",
            extra={"agent_id": str(agent_id), "document_id": str(doc.id), "size": len(data)},
        )
    return UploadResult(accepted=accepted, rejected=rejected)


async def get_or_404(
    session: AsyncSession, agent_id: uuid.UUID, document_id: uuid.UUID
) -> Document:
    doc = await doc_repo.get_document(session, agent_id, document_id)
    if doc is None:
        raise NotFoundError("Document not found.", details={"document_id": str(document_id)})
    return doc


async def delete(
    session: AsyncSession, storage: FileStorage, agent_id: uuid.UUID, document_id: uuid.UUID
) -> None:
    doc = await get_or_404(session, agent_id, document_id)
    path = doc.storage_path
    await doc_repo.delete_document(session, doc)
    await session.commit()
    storage.delete_file(path)
    logger.info(
        "document.deleted", extra={"agent_id": str(agent_id), "document_id": str(document_id)}
    )


async def reprocess(session: AsyncSession, agent_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    doc = await get_or_404(session, agent_id, document_id)
    if doc.status == DocumentStatus.PROCESSING:
        raise ValidationFailed("This document is already being processed.")
    return await doc_repo.requeue(session, doc)
