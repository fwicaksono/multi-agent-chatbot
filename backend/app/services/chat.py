"""Chat orchestration: rewrite -> retrieve -> generate -> cite -> persist.

The service emits a sequence of events that the API turns into Server-Sent
Events (or collects into one JSON response for non-streaming clients):

    meta      conversation/message ids and the configured provider/model
    rewrite   the standalone query used for retrieval (follow-up questions)
    sources   every passage handed to the model (what was consulted)
    fallback  the primary provider failed; switching to the next one
    token     a piece of the answer
    done      final answer, citations, provider/model used, tokens and cost
    error     a user-facing error (provider failure, embedding failure, ...)

Database sessions are short-lived and never held open while waiting on the LLM.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError
from app.core.logging import get_logger
from app.db.models import Agent
from app.db.session import session_scope
from app.ingestion.chunker import count_tokens
from app.llm.base import ChatMessage, ChatRequest, StreamDelta, StreamEnd, TokenUsage
from app.llm.errors import ProviderError
from app.llm.router import FallbackEvent, LLMRouter, Target, build_chain
from app.rag.citations import build_citations, build_sources, strip_citation_markers
from app.rag.prompts import build_answer_messages, build_rewrite_messages
from app.repositories import agents as agent_repo
from app.repositories import chunks as chunk_repo
from app.repositories import conversations as conv_repo
from app.repositories import usage as usage_repo
from app.retrieval.retriever import HybridRetriever

logger = get_logger(__name__)

NO_DOCUMENTS_ANSWER = (
    "I don't know yet: my knowledge base has no ready documents. Upload PDF or DOCX "
    "files in the **Knowledge** tab and ask again once they show as *ready*."
)


@dataclass(slots=True)
class ChatEvent:
    event: str
    data: dict[str, Any]


@dataclass(slots=True)
class TurnContext:
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    question: str
    agent_id: uuid.UUID
    agent_name: str
    instructions: str
    temperature: float
    chain: list[Target]
    history: list[ChatMessage] = field(default_factory=list)


@dataclass(slots=True)
class TurnAccounting:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost


# Strong references to fire-and-forget tasks (asyncio keeps only weak ones).
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 60 else title[:57].rstrip() + "..."


class ChatService:
    def __init__(self, settings: Settings, router: LLMRouter, retriever: HybridRetriever) -> None:
        self.settings = settings
        self.router = router
        self.retriever = retriever

    def chain_for(self, agent: Agent) -> list[Target]:
        fallback = (
            Target(agent.fallback_provider, agent.fallback_model)
            if agent.fallback_provider and agent.fallback_model
            else None
        )
        return build_chain(
            Target(agent.provider, agent.model), fallback, self.settings.fallback_chain
        )

    async def start_turn(self, conversation_id: uuid.UUID, question: str) -> TurnContext:
        """Validate the request and persist the user's message (before streaming)."""
        async with session_scope() as session:
            conv = await conv_repo.get_conversation(session, conversation_id)
            if conv is None:
                raise NotFoundError("Conversation not found.")
            agent = await agent_repo.get_agent(session, conv.agent_id)
            if agent is None:  # pragma: no cover - cascade makes this unreachable
                raise NotFoundError("Agent not found.")

            history = await conv_repo.recent_messages(
                session, conv.id, self.settings.chat_history_messages
            )
            user_msg = await conv_repo.add_message(session, conv.id, role="user", content=question)
            await conv_repo.touch(
                session, conv, title=_title_from(question) if not history else None
            )
            return TurnContext(
                conversation_id=conv.id,
                user_message_id=user_msg.id,
                question=question,
                agent_id=agent.id,
                agent_name=agent.name,
                instructions=agent.instructions,
                temperature=agent.temperature,
                chain=self.chain_for(agent),
                history=[
                    ChatMessage(
                        m.role,
                        strip_citation_markers(m.content) if m.role == "assistant" else m.content,
                    )
                    for m in history
                ],
            )

    async def run_turn(self, ctx: TurnContext) -> AsyncIterator[ChatEvent]:
        started = time.perf_counter()
        primary = ctx.chain[0]
        accounting = TurnAccounting()
        fallbacks: list[dict[str, Any]] = []
        yield ChatEvent(
            "meta",
            {
                "conversation_id": str(ctx.conversation_id),
                "user_message_id": str(ctx.user_message_id),
                "provider": primary.provider,
                "model": primary.model,
            },
        )

        async with session_scope() as session:
            ready_chunks = await chunk_repo.count_ready_chunks(session, ctx.agent_id)

        # 1. Rewrite follow-ups into a standalone query for retrieval (skipped
        #    when there is nothing to retrieve from).
        standalone = ctx.question
        if ready_chunks and ctx.history and self.settings.query_rewrite_enabled:
            standalone = await self._rewrite(ctx, accounting)
            if standalone != ctx.question:
                yield ChatEvent("rewrite", {"query": standalone})

        # 2. Retrieve from this agent's knowledge base only.
        try:
            async with session_scope() as session:
                result = (
                    await self.retriever.retrieve(session, ctx.agent_id, standalone)
                    if ready_chunks
                    else None
                )
                if result and result.embedding:
                    rec = await usage_repo.record_usage(
                        session,
                        agent_id=ctx.agent_id,
                        conversation_id=ctx.conversation_id,
                        kind="embedding_query",
                        provider=result.embedding.provider,
                        model=result.embedding.model,
                        prompt_tokens=result.embedding.prompt_tokens,
                    )
                    accounting.add(0, 0, float(rec.cost_usd))
        except AppError as exc:
            logger.warning("chat.retrieval_failed", extra={"error_code": exc.code})
            async for event in self._fail(ctx, exc, primary, standalone, started, fallbacks):
                yield event
            return

        hits = result.hits if result else []
        sources = build_sources(hits)
        yield ChatEvent("sources", {"sources": sources})

        if not ready_chunks:
            yield ChatEvent("token", {"text": NO_DOCUMENTS_ANSWER})
            async for event in self._finish(
                ctx,
                NO_DOCUMENTS_ANSWER,
                sources,
                None,
                None,
                TokenUsage(),
                standalone,
                accounting,
                fallbacks,
                started,
            ):
                yield event
            return

        # 3. Generate a grounded answer, streaming tokens as they arrive.
        messages = build_answer_messages(
            agent_name=ctx.agent_name,
            agent_instructions=ctx.instructions,
            history=ctx.history,
            question=ctx.question,
            hits=hits,
            standalone_question=standalone,
        )
        request = ChatRequest(
            model=primary.model,
            messages=messages,
            temperature=ctx.temperature,
            max_tokens=self.settings.chat_max_output_tokens,
        )
        answer: list[str] = []
        end: StreamEnd | None = None
        try:
            async for event in self.router.stream(request, ctx.chain):
                if isinstance(event, StreamDelta):
                    answer.append(event.text)
                    yield ChatEvent("token", {"text": event.text})
                elif isinstance(event, FallbackEvent):
                    fallbacks.append(event.to_dict())
                    yield ChatEvent("fallback", event.to_dict())
                elif isinstance(event, StreamEnd):
                    end = event
        except (asyncio.CancelledError, GeneratorExit):
            # The client went away (tab closed or "Stop" pressed). Save the partial
            # answer and its estimated cost from a separate task: this coroutine is
            # being cancelled and cannot reliably await anything any more.
            _spawn(
                self._save_cancelled(
                    ctx,
                    "".join(answer),
                    sources,
                    messages,
                    primary,
                    standalone,
                    accounting,
                    fallbacks,
                    started,
                )
            )
            raise
        except ProviderError as exc:
            logger.error(
                "chat.generation_failed",
                extra={
                    "agent_id": str(ctx.agent_id),
                    "error_code": exc.code,
                    "provider": exc.provider,
                    "cause": repr(exc.__cause__),
                },
            )
            async for event in self._fail(
                ctx, exc, primary, standalone, started, fallbacks, partial="".join(answer)
            ):
                yield event
            return

        if end and end.finish_reason == "length":
            # Hit CHAT_MAX_OUTPUT_TOKENS: say so instead of ending mid-sentence silently.
            note = (
                "\n\n_(The answer was cut off at the maximum length. Ask me to continue, "
                "or ask a narrower question.)_"
            )
            answer.append(note)
            yield ChatEvent("token", {"text": note})
        text = "".join(answer).strip()
        used = end or StreamEnd(provider=primary.provider, model=primary.model)
        async for event in self._finish(
            ctx,
            text,
            sources,
            used.provider,
            used.model,
            used.usage,
            standalone,
            accounting,
            fallbacks,
            started,
        ):
            yield event

    # ------------------------------------------------------------------
    async def _rewrite(self, ctx: TurnContext, accounting: TurnAccounting) -> str:
        request = ChatRequest(
            model=ctx.chain[0].model,
            messages=build_rewrite_messages(ctx.history, ctx.question),
            temperature=0.0,
            max_tokens=200,
        )
        try:
            response, _ = await self.router.complete(request, ctx.chain)
        except ProviderError as exc:
            # Not fatal: fall back to retrieving with the raw question.
            logger.warning("chat.rewrite_failed", extra={"error_code": exc.code})
            return ctx.question
        async with session_scope() as session:
            rec = await usage_repo.record_usage(
                session,
                agent_id=ctx.agent_id,
                conversation_id=ctx.conversation_id,
                kind="query_rewrite",
                provider=response.provider,
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                reported_cost=response.usage.cost_usd,
            )
        accounting.add(
            response.usage.prompt_tokens, response.usage.completion_tokens, float(rec.cost_usd)
        )
        rewritten = response.content.strip().strip('"').splitlines()
        return rewritten[0].strip() if rewritten and rewritten[0].strip() else ctx.question

    async def _finish(
        self,
        ctx: TurnContext,
        text: str,
        sources: list[dict[str, Any]],
        provider: str | None,
        model: str | None,
        usage: TokenUsage,
        standalone: str,
        accounting: TurnAccounting,
        fallbacks: list[dict[str, Any]],
        started: float,
    ) -> AsyncIterator[ChatEvent]:
        citations = build_citations(text, sources)
        latency_ms = int((time.perf_counter() - started) * 1000)
        async with session_scope() as session:
            if provider and model:
                rec = await usage_repo.record_usage(
                    session,
                    agent_id=ctx.agent_id,
                    conversation_id=ctx.conversation_id,
                    kind="chat",
                    provider=provider,
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    reported_cost=usage.cost_usd,
                )
                accounting.add(usage.prompt_tokens, usage.completion_tokens, float(rec.cost_usd))
            msg = await conv_repo.add_message(
                session,
                ctx.conversation_id,
                role="assistant",
                content=text,
                citations=citations,
                sources=sources,
                provider=provider,
                model=model,
                rewritten_query=standalone if standalone != ctx.question else None,
                fallback_events=fallbacks,
                prompt_tokens=accounting.prompt_tokens,
                completion_tokens=accounting.completion_tokens,
                cost_usd=accounting.cost_usd,
                latency_ms=latency_ms,
            )
        logger.info(
            "chat.turn_completed",
            extra={
                "agent_id": str(ctx.agent_id),
                "provider": provider,
                "model": model,
                "sources": len(sources),
                "citations": len(citations),
                "fallbacks": len(fallbacks),
                "latency_ms": latency_ms,
                "cost_usd": round(accounting.cost_usd, 6),
            },
        )
        yield ChatEvent(
            "done",
            {
                "message_id": str(msg.id),
                "content": text,
                "citations": citations,
                "provider": provider,
                "model": model,
                "rewritten_query": msg.rewritten_query,
                "fallback_events": fallbacks,
                "usage": {
                    "prompt_tokens": accounting.prompt_tokens,
                    "completion_tokens": accounting.completion_tokens,
                    "cost_usd": round(accounting.cost_usd, 6),
                },
                "latency_ms": latency_ms,
            },
        )

    async def _save_cancelled(
        self,
        ctx: TurnContext,
        partial: str,
        sources: list[dict[str, Any]],
        messages: list[ChatMessage],
        primary: Target,
        standalone: str,
        accounting: TurnAccounting,
        fallbacks: list[dict[str, Any]],
        started: float,
    ) -> None:
        # The provider never sent its usage numbers, so estimate them locally.
        target = (
            Target(fallbacks[-1]["next_provider"], fallbacks[-1]["next_model"])
            if fallbacks
            else primary
        )
        prompt_tokens = sum(count_tokens(m.content) for m in messages)
        completion_tokens = count_tokens(partial)
        try:
            async with session_scope() as session:
                rec = await usage_repo.record_usage(
                    session,
                    agent_id=ctx.agent_id,
                    conversation_id=ctx.conversation_id,
                    kind="chat",
                    provider=target.provider,
                    model=target.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                accounting.add(prompt_tokens, completion_tokens, float(rec.cost_usd))
                await conv_repo.add_message(
                    session,
                    ctx.conversation_id,
                    role="assistant",
                    content=partial.strip(),
                    status="cancelled",
                    citations=build_citations(partial, sources),
                    sources=sources,
                    provider=target.provider,
                    model=target.model,
                    rewritten_query=standalone if standalone != ctx.question else None,
                    fallback_events=fallbacks,
                    prompt_tokens=accounting.prompt_tokens,
                    completion_tokens=accounting.completion_tokens,
                    cost_usd=accounting.cost_usd,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            logger.info(
                "chat.turn_cancelled",
                extra={"agent_id": str(ctx.agent_id), "partial_chars": len(partial)},
            )
        except Exception:  # e.g. the conversation was deleted meanwhile
            logger.exception("chat.cancel_save_failed")

    async def _fail(
        self,
        ctx: TurnContext,
        exc: AppError,
        primary: Target,
        standalone: str,
        started: float,
        fallbacks: list[dict[str, Any]],
        partial: str = "",
    ) -> AsyncIterator[ChatEvent]:
        async with session_scope() as session:
            msg = await conv_repo.add_message(
                session,
                ctx.conversation_id,
                role="assistant",
                content=partial,
                status="error",
                provider=primary.provider,
                model=primary.model,
                rewritten_query=standalone if standalone != ctx.question else None,
                fallback_events=fallbacks,
                error=exc.to_dict(),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        yield ChatEvent("error", {"message_id": str(msg.id), **exc.to_dict()})
