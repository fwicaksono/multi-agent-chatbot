"""Agent persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent, Document, DocumentStatus


@dataclass(slots=True)
class DocumentCounts:
    total: int = 0
    ready: int = 0
    processing: int = 0
    failed: int = 0


async def list_agents(session: AsyncSession) -> list[tuple[Agent, DocumentCounts]]:
    counts = (
        select(
            Document.agent_id,
            func.count().label("total"),
            func.sum(case((Document.status == DocumentStatus.READY, 1), else_=0)).label("ready"),
            func.sum(case((Document.status == DocumentStatus.PROCESSING, 1), else_=0)).label(
                "processing"
            ),
            func.sum(case((Document.status == DocumentStatus.FAILED, 1), else_=0)).label("failed"),
        )
        .group_by(Document.agent_id)
        .subquery()
    )
    rows = await session.execute(
        select(Agent, counts)
        .outerjoin(counts, counts.c.agent_id == Agent.id)
        .order_by(Agent.created_at.desc())
    )
    return [
        (
            row.Agent,
            DocumentCounts(
                total=row.total or 0,
                ready=row.ready or 0,
                processing=row.processing or 0,
                failed=row.failed or 0,
            ),
        )
        for row in rows
    ]


async def document_counts(session: AsyncSession, agent_id: uuid.UUID) -> DocumentCounts:
    rows = await session.execute(
        select(Document.status, func.count())
        .where(Document.agent_id == agent_id)
        .group_by(Document.status)
    )
    counts = DocumentCounts()
    for status, n in rows:
        setattr(counts, status, n)
        counts.total += n
    return counts


async def get_agent(session: AsyncSession, agent_id: uuid.UUID) -> Agent | None:
    return await session.get(Agent, agent_id)


async def create_agent(session: AsyncSession, **fields: Any) -> Agent:
    agent = Agent(**fields)
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    return agent


async def update_agent(session: AsyncSession, agent: Agent, **fields: Any) -> Agent:
    for key, value in fields.items():
        setattr(agent, key, value)
    await session.flush()
    await session.refresh(agent)
    return agent


async def delete_agent(session: AsyncSession, agent: Agent) -> None:
    # ON DELETE CASCADE removes documents, chunks, conversations and usage.
    await session.delete(agent)
    await session.flush()
