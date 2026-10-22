from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_retriever, get_storage
from app.db.session import get_db
from app.ingestion.storage import FileStorage
from app.rag.citations import build_sources
from app.repositories import agents as agent_repo
from app.retrieval.retriever import HybridRetriever
from app.schemas.api import (
    AgentCreate,
    AgentOut,
    AgentUpdate,
    DocumentCountsOut,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services import agents as agent_service

router = APIRouter(prefix="/agents", tags=["agents"])


def _counts(c: agent_repo.DocumentCounts) -> DocumentCountsOut:
    return DocumentCountsOut(total=c.total, ready=c.ready, processing=c.processing, failed=c.failed)


async def _out(session: AsyncSession, agent) -> AgentOut:
    out = AgentOut.model_validate(agent)
    out.document_counts = _counts(await agent_repo.document_counts(session, agent.id))
    return out


@router.get("", response_model=list[AgentOut])
async def list_agents(session: AsyncSession = Depends(get_db, scope="function")) -> list[AgentOut]:
    result = []
    for agent, counts in await agent_repo.list_agents(session):
        out = AgentOut.model_validate(agent)
        out.document_counts = _counts(counts)
        result.append(out)
    return result


@router.post("", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
async def create_agent(
    data: AgentCreate, session: AsyncSession = Depends(get_db, scope="function")
) -> AgentOut:
    agent = await agent_service.create(session, data)
    return await _out(session, agent)


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(
    agent_id: uuid.UUID, session: AsyncSession = Depends(get_db, scope="function")
) -> AgentOut:
    return await _out(session, await agent_service.get_or_404(session, agent_id))


@router.patch("/{agent_id}", response_model=AgentOut)
async def update_agent(
    agent_id: uuid.UUID,
    data: AgentUpdate,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> AgentOut:
    agent = await agent_service.update(session, agent_id, data)
    return await _out(session, agent)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: uuid.UUID,
    session: AsyncSession = Depends(get_db, scope="function"),
    storage: FileStorage = Depends(get_storage),
) -> Response:
    await agent_service.delete(session, storage, agent_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{agent_id}/retrieve", response_model=RetrieveResponse)
async def retrieve(
    agent_id: uuid.UUID,
    body: RetrieveRequest,
    session: AsyncSession = Depends(get_db, scope="function"),
    retriever: HybridRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Run retrieval only (no LLM). Used by the evaluation script and for debugging."""
    await agent_service.get_or_404(session, agent_id)
    result = await retriever.retrieve(
        session, agent_id, body.query, top_k=body.top_k, mode=body.mode
    )
    return RetrieveResponse(query=body.query, mode=body.mode, sources=build_sources(result.hits))
