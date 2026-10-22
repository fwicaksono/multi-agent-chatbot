"""Conversation memory: recent messages verbatim, older ones in a rolling summary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Conversation, UsageRecord
from app.db.session import session_scope
from app.ingestion.pipeline import IngestionPipeline
from app.llm import registry
from app.llm.base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ModelInfo,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    TokenUsage,
)
from app.llm.router import LLMRouter
from app.repositories import agents as agent_repo
from app.repositories import conversations as conv_repo
from app.repositories import documents as doc_repo
from app.retrieval.retriever import HybridRetriever
from app.services import chat as chat_module
from app.services.chat import ChatService
from app.workers.ingestion import IngestionWorker
from tests.conftest import SAMPLE_DATA, FakeEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("db")]

SUMMARY = "- User asked about annual leave: 18 days (employee_handbook.pdf)."


class RecordingProvider(LLMProvider):
    name = "recording"
    display_name = "Recording"
    required_env = ("NONE",)
    suggested_models = ("r",)
    requests: list[ChatRequest] = []

    def is_configured(self) -> bool:
        return True

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        RecordingProvider.requests.append(request)
        is_summary = "maintain the memory" in request.messages[0].content
        text = SUMMARY if is_summary else request.messages[-1].content.split("\n")[-2]
        return ChatResponse(text, self.name, request.model, TokenUsage(30, 10))

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        RecordingProvider.requests.append(request)
        yield StreamDelta("Answer [1].")
        yield StreamEnd(self.name, request.model, TokenUsage(20, 3))

    async def list_models(self) -> list[ModelInfo]:
        return []


@pytest.fixture
async def setup(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    registry.register_provider(RecordingProvider)
    RecordingProvider.requests = []
    settings = get_settings()
    monkeypatch.setattr(settings, "chat_history_messages", 2)
    embedder = FakeEmbedder(settings)
    service = ChatService(
        settings, LLMRouter(registry.ProviderFactory(settings)), HybridRetriever(settings, embedder)
    )
    async with session_scope() as s:
        agent = await agent_repo.create_agent(
            s, name="A", description="", instructions="", provider="recording", model="r"
        )
        path = SAMPLE_DATA / "hr" / "employee_handbook.pdf"
        doc = await doc_repo.create_document(
            s,
            agent_id=agent.id,
            filename=path.name,
            file_type="pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256="x",
            storage_path=str(path),
        )
        conv = await conv_repo.create_conversation(s, agent.id)
    worker = IngestionWorker(settings, IngestionPipeline.from_settings(settings, embedder))
    await worker._store(doc.id, agent.id, await worker.pipeline.run(path, "pdf", path.name))
    return service, settings, conv.id


async def turn(service: ChatService, conv_id, question: str) -> None:
    ctx = await service.start_turn(conv_id, question)
    _ = [e async for e in service.run_turn(ctx)]
    await asyncio.gather(*chat_module._background_tasks)


def last_answer_request() -> ChatRequest:
    return [r for r in RecordingProvider.requests if "Grounding rules" in r.messages[0].content][-1]


async def test_old_messages_are_summarised_and_given_to_the_model(setup):
    service, _, conv_id = setup
    await turn(service, conv_id, "How much annual leave?")
    await turn(service, conv_id, "And sick leave?")  # turn 1 now leaves the 2-message window
    async with session_scope() as s:
        conv = await s.get(Conversation, conv_id)
        assert conv.summary == SUMMARY and conv.summary_until is not None
        usage = list(
            await s.scalars(select(UsageRecord).where(UsageRecord.kind == "history_summary"))
        )
    assert usage, "summary calls are tracked as usage"

    await turn(service, conv_id, "What did I ask first?")
    request = last_answer_request()
    assert SUMMARY in request.messages[0].content  # summary is part of the memory
    history = request.messages[1:-1]
    assert [m.content for m in history] == ["And sick leave?", "Answer."]  # only the window
    assert request.messages[-1].content.startswith("What did I ask first?")


async def test_messages_are_kept_verbatim_until_summarised(setup, monkeypatch):
    service, _, conv_id = setup

    async def no_summary(self, ctx):  # e.g. summary still running or the call failed
        return None

    monkeypatch.setattr(ChatService, "_update_summary", no_summary)
    await turn(service, conv_id, "How much annual leave?")
    await turn(service, conv_id, "And sick leave?")
    await turn(service, conv_id, "And maternity leave?")
    history = last_answer_request().messages[1:-1]
    assert [m.content for m in history] == [
        "How much annual leave?",
        "Answer.",
        "And sick leave?",
        "Answer.",
    ], "nothing is forgotten while no summary exists"


async def test_disabled_summary_keeps_only_the_window(setup, monkeypatch):
    service, settings, conv_id = setup
    monkeypatch.setattr(settings, "chat_summary_enabled", False)
    for q in ("How much annual leave?", "And sick leave?", "And maternity leave?"):
        await turn(service, conv_id, q)
    request = last_answer_request()
    assert "Earlier in this conversation" not in request.messages[0].content
    assert [m.content for m in request.messages[1:-1]] == ["And sick leave?", "Answer."]


async def test_concurrent_summary_updates_do_not_overwrite_each_other(setup):
    _, _, conv_id = setup
    async with session_scope() as s:
        conv = await s.get(Conversation, conv_id)
        first = await conv_repo.save_summary(
            s, conv_id, "A", until=conv.created_at, expected_until=None
        )
        stale = await conv_repo.save_summary(
            s, conv_id, "B", until=conv.created_at, expected_until=None
        )
    assert first is True and stale is False


async def _seed_messages(conv_id, n: int) -> None:
    async with session_scope() as s:
        for i in range(n):
            await conv_repo.add_message(s, conv_id, role="user", content=f"question {i}")
            text = " ".join(f"clause{j} policy{i} detail{j * i}" for j in range(40))
            await conv_repo.add_message(s, conv_id, role="assistant", content=f"answer {i}: {text}")


async def test_verbatim_history_stays_bounded_when_summaries_keep_failing(setup, monkeypatch):
    service, _, conv_id = setup

    async def broken_summary(self, ctx):
        return None

    monkeypatch.setattr(ChatService, "_update_summary", broken_summary)
    await _seed_messages(conv_id, 100)  # 200 old messages, never summarised
    await turn(service, conv_id, "And now?")
    history = last_answer_request().messages[1:-1]
    assert len(history) <= 2 * 2, f"{len(history)} messages sent verbatim"


async def test_large_backlog_is_summarised_in_bounded_batches(setup):
    service, _, conv_id = setup
    await _seed_messages(conv_id, 150)  # e.g. memory switched on for a long conversation
    await turn(service, conv_id, "And now?")
    summary_requests = [
        r for r in RecordingProvider.requests if "maintain the memory" in r.messages[0].content
    ]
    assert summary_requests, "the backlog is summarised"
    from app.ingestion.chunker import count_tokens

    biggest = max(count_tokens(r.messages[1].content) for r in summary_requests)
    assert biggest < 12_000, f"summary prompt of {biggest} tokens"
    async with session_scope() as s:
        conv = await s.get(Conversation, conv_id)
        memory = await conv_repo.load_memory(s, conv, 2)
    assert memory.unsummarized == [], "the backlog is fully caught up"


async def test_rewrite_uses_only_the_latest_turns(setup, monkeypatch):
    service, settings, conv_id = setup
    monkeypatch.setattr(settings, "chat_history_messages", 50)
    await _seed_messages(conv_id, 20)  # 40 messages, all inside the window
    await turn(service, conv_id, "And for contractors?")
    rewrite = next(
        r for r in RecordingProvider.requests if "standalone search query" in r.messages[0].content
    )
    transcript = rewrite.messages[1].content
    turns = transcript.count("USER:") + transcript.count("ASSISTANT:")
    assert turns <= 6, f"{turns} messages in the rewrite prompt"
    # ...while the answer itself still sees the whole window
    assert len(last_answer_request().messages[1:-1]) == 40
