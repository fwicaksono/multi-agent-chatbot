"""Document persistence, including the database-backed ingestion queue.

The `documents` table doubles as the job queue: a document with
status=processing and no live lock is a pending job. Workers claim jobs with
`SELECT ... FOR UPDATE SKIP LOCKED`, so several workers can run concurrently
and a job whose worker crashed is retried once its lock goes stale.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStage, DocumentStatus


async def list_documents(session: AsyncSession, agent_id: uuid.UUID) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.agent_id == agent_id).order_by(Document.created_at.desc())
    )
    return list(rows)


async def get_document(
    session: AsyncSession, agent_id: uuid.UUID, document_id: uuid.UUID
) -> Document | None:
    return await session.scalar(
        select(Document).where(Document.id == document_id, Document.agent_id == agent_id)
    )


async def find_duplicate(
    session: AsyncSession, agent_id: uuid.UUID, sha256: str
) -> Document | None:
    return await session.scalar(
        select(Document).where(
            Document.agent_id == agent_id,
            Document.sha256 == sha256,
            Document.status != DocumentStatus.FAILED,
        )
    )


async def create_document(session: AsyncSession, **fields: Any) -> Document:
    doc = Document(**fields)
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


async def delete_document(session: AsyncSession, doc: Document) -> None:
    # Chunks (embeddings + full-text index) are removed by ON DELETE CASCADE.
    await session.delete(doc)
    await session.flush()


async def requeue(session: AsyncSession, doc: Document) -> Document:
    await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    doc.status = DocumentStatus.PROCESSING
    doc.stage = DocumentStage.QUEUED
    doc.error_code = None
    doc.error_message = None
    doc.attempts = 0
    doc.locked_at = None
    doc.chunk_count = 0
    await session.flush()
    await session.refresh(doc)
    return doc


# --- Worker queue operations -------------------------------------------------


async def claim_next(session: AsyncSession, stale_after_seconds: int) -> Document | None:
    stale_before = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
    candidate = (
        select(Document.id)
        .where(
            Document.status == DocumentStatus.PROCESSING,
            or_(Document.locked_at.is_(None), Document.locked_at < stale_before),
        )
        .order_by(Document.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    return await session.scalar(
        update(Document)
        .where(Document.id == candidate)
        .values(locked_at=func.now(), attempts=Document.attempts + 1)
        .returning(Document)
    )


async def set_stage(session: AsyncSession, document_id: uuid.UUID, stage: DocumentStage) -> None:
    # Also refreshes the lock, acting as a heartbeat for long jobs.
    await session.execute(
        update(Document).where(Document.id == document_id).values(stage=stage, locked_at=func.now())
    )


async def mark_ready(session: AsyncSession, document_id: uuid.UUID, **fields: Any) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status=DocumentStatus.READY,
            stage=DocumentStage.DONE,
            error_code=None,
            error_message=None,
            locked_at=None,
            processed_at=func.now(),
            **fields,
        )
    )


async def mark_failed(
    session: AsyncSession, document_id: uuid.UUID, code: str, message: str
) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(
            status=DocumentStatus.FAILED,
            stage=DocumentStage.FAILED,
            error_code=code,
            error_message=message,
            locked_at=None,
            processed_at=func.now(),
        )
    )


async def release(
    session: AsyncSession, document_id: uuid.UUID, retry_in_seconds: int, stale_after_seconds: int
) -> None:
    """Put a job back in the queue after a transient failure.

    The lock timestamp is back-dated so the job becomes claimable again in
    roughly `retry_in_seconds`.
    """
    locked_at = datetime.now(UTC) - timedelta(
        seconds=max(stale_after_seconds - retry_in_seconds, 0)
    )
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(stage=DocumentStage.QUEUED, locked_at=locked_at)
    )


async def exists(session: AsyncSession, document_id: uuid.UUID) -> bool:
    return (await session.scalar(select(Document.id).where(Document.id == document_id))) is not None
