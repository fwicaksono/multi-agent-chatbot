"""Worker and chat robustness: long jobs, idempotent indexing, disconnects, truncation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Chunk, Message, UsageRecord
from app.db.session import session_scope
from app.ingestion.extractors.base import ExtractedDocument, TextBlock
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


async def _agent(provider: str = "openai", model: str = "m"):
    async with session_scope() as s:
        return await agent_repo.create_agent(
            s, name="A", description="", instructions="", provider=provider, model=model
        )


async def _queued_doc(agent_id, path):
    async with session_scope() as s:
        return await doc_repo.create_document(
            s,
            agent_id=agent_id,
            filename=path.name,
            file_type="pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256=path.name,
            storage_path=str(path),
        )


class SlowExtractor:
    """Takes longer than the stale-lock timeout, like OCR of a big scan."""

    def extract(self, path):
        time.sleep(2.5)
        return ExtractedDocument(blocks=[TextBlock("Annual leave is 18 days.", page_number=1)])


async def test_long_job_is_not_reclaimed_and_indexed_once(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "worker_stale_lock_seconds", 1)
    pipeline = IngestionPipeline.from_settings(settings, FakeEmbedder(settings))
    pipeline.extractors["pdf"] = SlowExtractor()
    worker = IngestionWorker(settings, pipeline)

    agent = await _agent()
    doc = await _queued_doc(agent.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf")

    job = asyncio.create_task(worker.process_next())
    await asyncio.sleep(1.8)  # well past the 1 s stale timeout, still extracting
    async with session_scope() as s:
        stolen = await doc_repo.claim_next(s, stale_after_seconds=1)
    assert stolen is None, "heartbeat should keep the running job locked"
    assert await job is True

    async with session_scope() as s:
        doc = await doc_repo.get_document(s, agent.id, doc.id)
        assert doc.status == "ready"
        n = await s.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
    assert n == 1


async def test_indexing_is_idempotent():
    settings = get_settings()
    worker = IngestionWorker(
        settings, IngestionPipeline.from_settings(settings, FakeEmbedder(settings))
    )
    agent = await _agent()
    path = SAMPLE_DATA / "hr" / "employee_handbook.pdf"
    doc = await _queued_doc(agent.id, path)
    result = await worker.pipeline.run(path, "pdf", path.name)
    assert await worker._store(doc.id, agent.id, result)
    assert await worker._store(doc.id, agent.id, result)  # e.g. a job that ran twice
    async with session_scope() as s:
        n = await s.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
        )
    assert n == len(result.chunks)


class ScriptedStream(LLMProvider):
    name = "scripted-stream"
    display_name = "Scripted"
    required_env = ("NONE",)
    suggested_models = ("s",)
    finish_reason = "stop"

    def is_configured(self) -> bool:
        return True

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse("q", self.name, request.model, TokenUsage(1, 1))

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        for word in ["Annual ", "leave ", "is ", "18 ", "days ", "[1]."]:
            await asyncio.sleep(0.05)
            yield StreamDelta(word)
        yield StreamEnd(self.name, request.model, TokenUsage(50, 6), self.finish_reason)

    async def list_models(self) -> list[ModelInfo]:
        return []


@pytest.fixture
async def chat_service(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    registry.register_provider(ScriptedStream)
    settings = get_settings()
    embedder = FakeEmbedder(settings)
    service = ChatService(
        settings, LLMRouter(registry.ProviderFactory(settings)), HybridRetriever(settings, embedder)
    )
    agent = await _agent("scripted-stream", "s")
    path = SAMPLE_DATA / "hr" / "employee_handbook.pdf"
    doc = await _queued_doc(agent.id, path)
    worker = IngestionWorker(settings, IngestionPipeline.from_settings(settings, embedder))
    await worker._store(doc.id, agent.id, await worker.pipeline.run(path, "pdf", path.name))
    async with session_scope() as s:
        conv = await conv_repo.create_conversation(s, agent.id)
    return service, conv.id


async def test_disconnect_mid_answer_saves_partial_answer_and_usage(chat_service):
    service, conv_id = chat_service
    ctx = await service.start_turn(conv_id, "How much annual leave?")
    turn = service.run_turn(ctx)
    async for event in turn:
        if event.event == "token":
            break  # the browser goes away after the first token
    await turn.aclose()
    await asyncio.gather(*chat_module._background_tasks)

    async with session_scope() as s:
        msgs = list(await s.scalars(select(Message).where(Message.conversation_id == conv_id)))
        usage = list(await s.scalars(select(UsageRecord).where(UsageRecord.kind == "chat")))
    assistant = [m for m in msgs if m.role == "assistant"]
    assert len(assistant) == 1 and assistant[0].status == "cancelled"
    assert assistant[0].content == "Annual"
    assert assistant[0].sources, "sources are kept so [n] markers stay clickable"
    assert usage and usage[0].prompt_tokens > 0  # estimated, so cost tracking stays honest


async def test_truncated_answer_is_flagged(chat_service, monkeypatch):
    service, conv_id = chat_service
    monkeypatch.setattr(ScriptedStream, "finish_reason", "length")
    ctx = await service.start_turn(conv_id, "How much annual leave?")
    events = [e async for e in service.run_turn(ctx)]
    done = events[-1]
    assert done.event == "done"
    assert "cut off at the maximum length" in done.data["content"]
