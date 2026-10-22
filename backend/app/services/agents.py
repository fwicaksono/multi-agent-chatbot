"""Agent management use cases."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError, ValidationFailed
from app.core.logging import get_logger
from app.db.models import Agent
from app.ingestion.storage import FileStorage
from app.llm.registry import available_providers
from app.repositories import agents as agent_repo
from app.schemas.api import AgentCreate, AgentUpdate

logger = get_logger(__name__)


def _validate_providers(provider: str | None, fallback_provider: str | None) -> None:
    names = available_providers()
    for field, value in (("provider", provider), ("fallback_provider", fallback_provider)):
        if value and value not in names:
            raise ValidationFailed(
                f"Unknown {field.replace('_', ' ')} '{value}'. Choose one of: {', '.join(names)}.",
                details={"field": field, "available": list(names)},
            )


async def get_or_404(session: AsyncSession, agent_id: uuid.UUID) -> Agent:
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise NotFoundError("Agent not found.", details={"agent_id": str(agent_id)})
    return agent


async def create(session: AsyncSession, data: AgentCreate) -> Agent:
    _validate_providers(data.provider, data.fallback_provider)
    if bool(data.fallback_provider) != bool(data.fallback_model):
        raise ValidationFailed("Set both a fallback provider and a fallback model, or neither.")
    agent = await agent_repo.create_agent(session, **data.model_dump())
    logger.info(
        "agent.created",
        extra={"agent_id": str(agent.id), "provider": agent.provider, "model": agent.model},
    )
    return agent


async def update(session: AsyncSession, agent_id: uuid.UUID, data: AgentUpdate) -> Agent:
    agent = await get_or_404(session, agent_id)
    changes = data.model_dump(exclude_unset=True)
    # Required columns: an explicit null must be a 422, not a database error.
    for key in ("name", "provider", "model"):
        if key in changes:
            if changes[key] is None or not str(changes[key]).strip():
                raise ValidationFailed(f"'{key}' cannot be empty.", details={"field": key})
            changes[key] = changes[key].strip()
    if "temperature" in changes and changes["temperature"] is None:
        raise ValidationFailed("'temperature' cannot be null.", details={"field": "temperature"})
    for key in ("description", "instructions"):
        if key in changes and changes[key] is None:
            changes[key] = ""
    for key in ("fallback_provider", "fallback_model"):
        if key in changes and isinstance(changes[key], str) and not changes[key].strip():
            changes[key] = None
    _validate_providers(changes.get("provider"), changes.get("fallback_provider"))

    fb_provider = changes.get("fallback_provider", agent.fallback_provider)
    fb_model = changes.get("fallback_model", agent.fallback_model)
    if bool(fb_provider) != bool(fb_model):
        raise ValidationFailed("Set both a fallback provider and a fallback model, or neither.")

    before = (agent.provider, agent.model)
    agent = await agent_repo.update_agent(session, agent, **changes)
    if before != (agent.provider, agent.model):
        logger.info(
            "agent.provider_changed",
            extra={
                "agent_id": str(agent.id),
                "from": f"{before[0]}:{before[1]}",
                "to": f"{agent.provider}:{agent.model}",
            },
        )
    return agent


async def delete(session: AsyncSession, storage: FileStorage, agent_id: uuid.UUID) -> None:
    agent = await get_or_404(session, agent_id)
    await agent_repo.delete_agent(session, agent)
    await session.commit()  # DB first: if this fails, files are left intact
    storage.delete_agent(agent_id)
    logger.info("agent.deleted", extra={"agent_id": str(agent_id)})
