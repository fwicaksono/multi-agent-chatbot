from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_chat_service
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.session import get_db
from app.repositories import conversations as conv_repo
from app.schemas.api import (
    ChatRequestIn,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationUpdate,
    MessageOut,
)
from app.services import agents as agent_service
from app.services.chat import ChatEvent, ChatService

logger = get_logger(__name__)
router = APIRouter(tags=["conversations"])


@router.get("/agents/{agent_id}/conversations", response_model=list[ConversationOut])
async def list_conversations(
    agent_id: uuid.UUID, session: AsyncSession = Depends(get_db, scope="function")
) -> list[ConversationOut]:
    await agent_service.get_or_404(session, agent_id)
    out = []
    for conv, count in await conv_repo.list_conversations(session, agent_id):
        item = ConversationOut.model_validate(conv)
        item.message_count = count
        out.append(item)
    return out


@router.post(
    "/agents/{agent_id}/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    agent_id: uuid.UUID,
    body: ConversationCreate | None = None,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> ConversationOut:
    await agent_service.get_or_404(session, agent_id)
    conv = await conv_repo.create_conversation(session, agent_id, body.title if body else None)
    return ConversationOut.model_validate(conv)


async def _get_or_404(session: AsyncSession, conversation_id: uuid.UUID, with_messages=False):
    conv = await conv_repo.get_conversation(session, conversation_id, with_messages=with_messages)
    if conv is None:
        raise NotFoundError("Conversation not found.")
    return conv


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, session: AsyncSession = Depends(get_db, scope="function")
) -> ConversationDetail:
    conv = await _get_or_404(session, conversation_id, with_messages=True)
    messages = [MessageOut.model_validate(m) for m in conv.messages]
    return ConversationDetail(
        id=conv.id,
        agent_id=conv.agent_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=len(messages),
        messages=messages,
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
async def rename_conversation(
    conversation_id: uuid.UUID,
    body: ConversationUpdate,
    session: AsyncSession = Depends(get_db, scope="function"),
) -> ConversationOut:
    conv = await _get_or_404(session, conversation_id)
    conv.title = body.title.strip()
    await session.flush()
    await session.refresh(conv)
    return ConversationOut.model_validate(conv)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, session: AsyncSession = Depends(get_db, scope="function")
) -> Response:
    await conv_repo.delete_conversation(session, await _get_or_404(session, conversation_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _sse(event: ChatEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data, default=str)}\n\n"


@router.post("/conversations/{conversation_id}/messages", response_model=None)
async def send_message(
    conversation_id: uuid.UUID,
    body: ChatRequestIn,
    stream: bool = Query(True, description="Stream the answer as Server-Sent Events"),
    chat: ChatService = Depends(get_chat_service),
) -> StreamingResponse | JSONResponse:
    """Ask the conversation's agent a question.

    With `stream=true` (default) the response is `text/event-stream` with the
    events documented in `app/services/chat.py`. With `stream=false` the
    final `done` (or `error`) payload is returned as JSON.
    """
    # Validation and persisting the user message happen before streaming
    # starts, so a missing conversation is a normal 404, not a broken stream.
    ctx = await chat.start_turn(conversation_id, body.content)

    if not stream:
        result: dict = {}
        sources: list = []
        async for event in chat.run_turn(ctx):
            if event.event == "sources":
                sources = event.data["sources"]
            if event.event in ("done", "error"):
                result = {"event": event.event, **event.data, "sources": sources}
        status_code = 200 if result.get("event") == "done" else 502
        return JSONResponse(result, status_code=status_code)

    async def body_iter() -> AsyncIterator[str]:
        turn = chat.run_turn(ctx)
        try:
            async for event in turn:
                yield _sse(event)
        except Exception:  # last-resort guard: always terminate the stream cleanly
            logger.exception("chat.stream_crashed", extra={"conversation_id": str(conversation_id)})
            yield _sse(
                ChatEvent(
                    "error",
                    {
                        "code": "internal_error",
                        "message": "Something went wrong generating "
                        "the answer. Check the server logs and try again.",
                    },
                )
            )
        finally:
            # On client disconnect, close the turn now (not at garbage collection)
            # so the partial answer is saved and upstream connections are released.
            await turn.aclose()

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
