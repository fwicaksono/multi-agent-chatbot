"""Token usage and cost records."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Agent, UsageRecord
from app.llm.pricing import estimate_cost


async def record_usage(
    session: AsyncSession,
    *,
    agent_id: uuid.UUID,
    kind: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
    reported_cost: float | None = None,
    conversation_id: uuid.UUID | None = None,
) -> UsageRecord:
    if reported_cost is not None:
        cost, source = reported_cost, "provider"
    else:
        estimated = estimate_cost(model, prompt_tokens, completion_tokens)
        cost, source = (estimated, "estimate") if estimated is not None else (0.0, "unknown")
    record = UsageRecord(
        agent_id=agent_id,
        conversation_id=conversation_id,
        kind=kind,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        cost_source=source,
    )
    session.add(record)
    await session.flush()
    return record


def _totals():
    return (
        func.count(UsageRecord.id).label("requests"),
        func.coalesce(func.sum(UsageRecord.prompt_tokens), 0).label("prompt_tokens"),
        func.coalesce(func.sum(UsageRecord.completion_tokens), 0).label("completion_tokens"),
        func.coalesce(func.sum(UsageRecord.cost_usd), 0).label("cost_usd"),
    )


async def agent_usage(session: AsyncSession, agent_id: uuid.UUID, days: int = 30) -> dict:
    where = UsageRecord.agent_id == agent_id
    total = (await session.execute(select(*_totals()).where(where))).one()

    by_model = await session.execute(
        select(UsageRecord.provider, UsageRecord.model, UsageRecord.kind, *_totals())
        .where(where)
        .group_by(UsageRecord.provider, UsageRecord.model, UsageRecord.kind)
        .order_by(func.sum(UsageRecord.cost_usd).desc())
    )

    since = datetime.now(UTC) - timedelta(days=days)
    day = cast(UsageRecord.created_at, Date)
    daily = await session.execute(
        select(day.label("day"), *_totals())
        .where(where, UsageRecord.created_at >= since)
        .group_by(day)
        .order_by(day)
    )
    unknown = await session.scalar(
        select(func.count()).where(where, UsageRecord.cost_source == "unknown")
    )
    return {
        "totals": dict(total._mapping),
        "by_model": [dict(r._mapping) for r in by_model],
        "daily": [dict(r._mapping) for r in daily],
        "unpriced_requests": unknown or 0,
    }


async def all_agents_usage(session: AsyncSession) -> list[dict]:
    rows = await session.execute(
        select(Agent.id.label("agent_id"), Agent.name.label("agent_name"), *_totals())
        .select_from(Agent)
        .outerjoin(UsageRecord, UsageRecord.agent_id == Agent.id)
        .group_by(Agent.id, Agent.name)
        .order_by(func.coalesce(func.sum(UsageRecord.cost_usd), 0).desc())
    )
    return [dict(r._mapping) for r in rows]
