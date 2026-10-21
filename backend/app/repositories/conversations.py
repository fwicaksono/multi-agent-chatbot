"""Conversation and message persistence."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Message


async def list_conversations(
    session: AsyncSession, agent_id: uuid.UUID
) -> list[tuple[Conversation, int]]:
    counts = (
        select(Message.conversation_id, func.count().label("n"))
        .group_by(Message.conversation_id)
        .subquery()
    )
    rows = await session.execute(
        select(Conversation, counts.c.n)
        .outerjoin(counts, counts.c.conversation_id == Conversation.id)
        .where(Conversation.agent_id == agent_id)
        .order_by(Conversation.updated_at.desc())
    )
    return [(row.Conversation, row.n or 0) for row in rows]


async def get_conversation(
    session: AsyncSession, conversation_id: uuid.UUID, with_messages: bool = False
) -> Conversation | None:
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    if with_messages:
        stmt = stmt.options(selectinload(Conversation.messages))
    return await session.scalar(stmt)


async def create_conversation(
    session: AsyncSession, agent_id: uuid.UUID, title: str | None = None
) -> Conversation:
    conv = Conversation(agent_id=agent_id, title=title or "New conversation")
    session.add(conv)
    await session.flush()
    await session.refresh(conv)
    return conv


async def delete_conversation(session: AsyncSession, conv: Conversation) -> None:
    await session.delete(conv)
    await session.flush()


async def recent_messages(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int
) -> list[Message]:
    """The last `limit` successful messages, oldest first."""
    rows = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.status == "complete")
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return list(reversed(list(rows)))


async def add_message(session: AsyncSession, conversation_id: uuid.UUID, **fields: Any) -> Message:
    msg = Message(conversation_id=conversation_id, **fields)
    session.add(msg)
    await session.flush()
    await session.refresh(msg)
    return msg


async def touch(session: AsyncSession, conv: Conversation, title: str | None = None) -> None:
    if title:
        conv.title = title
    conv.updated_at = func.now()
    await session.flush()
