"""Conversation and message persistence."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
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


@dataclass(slots=True)
class ConversationMemory:
    """What the model gets to remember of a conversation.

    `recent` are the last messages, sent verbatim. `unsummarized` are older
    messages not yet folded into `summary` (normally empty: the summary is
    updated after each turn, but if that has not finished yet they are sent
    verbatim so nothing is ever silently forgotten).
    """

    summary: str | None
    summary_until: datetime | None
    unsummarized: list[Message] = field(default_factory=list)
    recent: list[Message] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.summary or self.unsummarized or self.recent)


async def load_memory(
    session: AsyncSession,
    conversation: Conversation,
    window: int,
    max_unsummarized: int | None = None,
) -> ConversationMemory:
    """Load the summary and the messages after it.

    With `max_unsummarized`, only the newest `window + max_unsummarized`
    messages are loaded (prompt path: bounded even if summaries keep failing).
    Without it, every message after the summary is loaded, oldest first
    (background summariser, which needs the oldest ones).
    """
    stmt = select(Message).where(
        Message.conversation_id == conversation.id, Message.status == "complete"
    )
    if conversation.summary_until is not None:
        stmt = stmt.where(Message.created_at > conversation.summary_until)
    if max_unsummarized is None:
        messages = list(await session.scalars(stmt.order_by(Message.created_at)))
    else:
        newest = stmt.order_by(Message.created_at.desc()).limit(window + max_unsummarized)
        messages = list(reversed(list(await session.scalars(newest))))
    split = max(len(messages) - window, 0)
    return ConversationMemory(
        summary=conversation.summary,
        summary_until=conversation.summary_until,
        unsummarized=messages[:split],
        recent=messages[split:],
    )


async def save_summary(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    summary: str,
    until: datetime,
    expected_until: datetime | None,
) -> bool:
    """Store a new summary unless another one was saved meanwhile (optimistic)."""
    current = (
        Conversation.summary_until.is_(None)
        if expected_until is None
        else Conversation.summary_until == expected_until
    )
    result = await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id, current)
        .values(summary=summary, summary_until=until)
    )
    return result.rowcount == 1
