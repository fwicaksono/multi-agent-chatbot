from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.repositories import usage as usage_repo
from app.schemas.api import AgentUsageOut, AgentUsageSummary
from app.services import agents as agent_service

router = APIRouter(tags=["usage"])


@router.get("/usage", response_model=list[AgentUsageSummary])
async def usage_summary(session: AsyncSession = Depends(get_db)) -> list[AgentUsageSummary]:
    return [AgentUsageSummary(**row) for row in await usage_repo.all_agents_usage(session)]


@router.get("/agents/{agent_id}/usage", response_model=AgentUsageOut)
async def agent_usage(
    agent_id: uuid.UUID,
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_db),
) -> AgentUsageOut:
    await agent_service.get_or_404(session, agent_id)
    data = await usage_repo.agent_usage(session, agent_id, days)
    return AgentUsageOut(agent_id=agent_id, **data)
