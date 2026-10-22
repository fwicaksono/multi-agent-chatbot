"""End-to-end API flow with a fake embedder and a scripted LLM provider.

Covers user workflows 1-6: create agent -> upload -> process -> chat with
citations -> follow-up -> switch provider and keep chatting.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.api import deps
from app.core.config import get_settings
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
from app.retrieval.retriever import HybridRetriever
from app.services.chat import ChatService
from app.workers.ingestion import IngestionWorker
from tests.conftest import SAMPLE_DATA, FakeEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("db")]


class CitingProvider(LLMProvider):
    """Answers with a citation to passage [1] and records what it was sent."""

    name = "citing"
    display_name = "Citing test provider"
    required_env = ("NONE",)
    suggested_models = ("cite-1",)
    requests: list[ChatRequest] = []

    def is_configured(self) -> bool:
        return True

    async def _complete(self, request: ChatRequest) -> ChatResponse:
        CitingProvider.requests.append(request)
        return ChatResponse(
            "How many days of annual leave are there?", self.name, request.model, TokenUsage(20, 8)
        )

    async def _stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        CitingProvider.requests.append(request)
        for piece in ["Employees get ", "18 days ", "of annual leave [1]."]:
            yield StreamDelta(piece)
        yield StreamEnd(self.name, request.model, TokenUsage(100, 12))

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("cite-1")]


@pytest.fixture
async def client(db, monkeypatch) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    registry.register_provider(CitingProvider)
    CitingProvider.requests = []

    settings = get_settings()
    embedder = FakeEmbedder(settings)
    factory = registry.ProviderFactory(settings)
    retriever = HybridRetriever(settings, embedder)
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[deps.get_retriever] = lambda: retriever
    app.dependency_overrides[deps.get_chat_service] = lambda: ChatService(
        settings, LLMRouter(factory), retriever
    )
    app.dependency_overrides[deps.get_storage] = lambda: deps.FileStorage(settings.upload_dir)
    app.state.worker = IngestionWorker(
        settings, IngestionPipeline.from_settings(settings, embedder)
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app  # type: ignore[attr-defined]
        yield c


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_full_user_workflow(client):
    # 1. Create an agent.
    r = await client.post(
        "/api/agents",
        json={
            "name": "Agent HR",
            "instructions": "Be helpful.",
            "provider": "citing",
            "model": "cite-1",
        },
    )
    assert r.status_code == 201, r.text
    agent = r.json()

    # Unknown providers are rejected with a helpful message.
    bad = await client.post("/api/agents", json={"name": "x", "provider": "nope", "model": "m"})
    assert bad.status_code == 422 and "Choose one of" in bad.json()["error"]["message"]

    # 2. Upload several files at once; the empty one is rejected up front.
    handbook = (SAMPLE_DATA / "hr" / "employee_handbook.pdf").read_bytes()
    files = [
        ("files", ("employee_handbook.pdf", handbook, "application/pdf")),
        (
            "files",
            (
                "corrupted.pdf",
                (SAMPLE_DATA / "invalid" / "corrupted.pdf").read_bytes(),
                "application/pdf",
            ),
        ),
        ("files", ("empty.pdf", b"", "application/pdf")),
    ]
    r = await client.post(f"/api/agents/{agent['id']}/documents", files=files)
    assert r.status_code == 202
    body = r.json()
    assert [d["filename"] for d in body["accepted"]] == ["employee_handbook.pdf", "corrupted.pdf"]
    assert body["rejected"][0]["code"] == "empty_file"

    # Duplicate uploads are detected.
    dup = await client.post(
        f"/api/agents/{agent['id']}/documents",
        files=[("files", ("copy.pdf", handbook, "application/pdf"))],
    )
    assert dup.json()["rejected"][0]["code"] == "duplicate_document"

    # 3. Status: processing until the worker runs, then ready / failed.
    docs = (await client.get(f"/api/agents/{agent['id']}/documents")).json()
    assert {d["status"] for d in docs} == {"processing"}
    worker = client.app.state.worker
    while await worker.process_next():
        pass
    docs = {
        d["filename"]: d for d in (await client.get(f"/api/agents/{agent['id']}/documents")).json()
    }
    assert docs["employee_handbook.pdf"]["status"] == "ready"
    assert docs["employee_handbook.pdf"]["chunk_count"] > 0
    assert docs["corrupted.pdf"]["status"] == "failed"
    assert "corrupted" in docs["corrupted.pdf"]["error_message"]

    # 4 + 5. Chat: streamed answer with citations to the source document and page.
    conv = (await client.post(f"/api/agents/{agent['id']}/conversations")).json()
    r = await client.post(
        f"/api/conversations/{conv['id']}/messages", json={"content": "How much annual leave?"}
    )
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "meta" and names[1] == "sources" and names[-1] == "done"
    done = events[-1][1]
    assert done["content"] == "Employees get 18 days of annual leave [1]."
    assert done["citations"][0]["document_name"] == "employee_handbook.pdf"
    assert done["citations"][0]["page_number"] is not None
    assert done["usage"]["prompt_tokens"] >= 100

    # Follow-up question: history is sent and the query is rewritten for retrieval.
    r = await client.post(
        f"/api/conversations/{conv['id']}/messages?stream=false",
        json={"content": "And what about sick leave?"},
    )
    assert r.status_code == 200
    assert r.json()["rewritten_query"] == "How many days of annual leave are there?"
    answer_request = CitingProvider.requests[-1]
    assert [m.role for m in answer_request.messages][:3] == ["system", "user", "assistant"]

    # 6. Switch provider/model; history is preserved and the next answer uses it.
    r = await client.patch(f"/api/agents/{agent['id']}", json={"model": "cite-2"})
    assert r.json()["model"] == "cite-2"
    r = await client.post(
        f"/api/conversations/{conv['id']}/messages?stream=false", json={"content": "Thanks"}
    )
    assert r.json()["model"] == "cite-2"
    detail = (await client.get(f"/api/conversations/{conv['id']}")).json()
    assert len(detail["messages"]) == 6

    # Usage is tracked per agent.
    usage = (await client.get(f"/api/agents/{agent['id']}/usage")).json()
    kinds = {row["kind"] for row in usage["by_model"]}
    assert {"chat", "query_rewrite", "embedding_ingest", "embedding_query"} <= kinds

    # Deleting the agent removes everything.
    assert (await client.delete(f"/api/agents/{agent['id']}")).status_code == 204
    assert (await client.get(f"/api/agents/{agent['id']}")).status_code == 404
    assert (await client.get(f"/api/conversations/{conv['id']}")).status_code == 404


async def test_agent_without_documents_says_it_does_not_know(client):
    agent = (
        await client.post("/api/agents", json={"name": "Empty", "provider": "citing", "model": "c"})
    ).json()
    conv = (await client.post(f"/api/agents/{agent['id']}/conversations")).json()
    r = await client.post(
        f"/api/conversations/{conv['id']}/messages?stream=false", json={"content": "Anything?"}
    )
    assert r.json()["content"].startswith("I don't know")
    # A follow-up must not pay for a query rewrite either: nothing to retrieve.
    r = await client.post(
        f"/api/conversations/{conv['id']}/messages?stream=false", json={"content": "And now?"}
    )
    assert r.json()["content"].startswith("I don't know")
    assert CitingProvider.requests == []  # no LLM call needed


async def test_errors_use_a_consistent_envelope(client):
    r = await client.get("/api/agents/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"
    r = await client.post("/api/agents", json={"name": ""})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


async def test_patch_with_null_values_is_validated_not_a_500(client):
    agent = (
        await client.post("/api/agents", json={"name": "A", "provider": "citing", "model": "c"})
    ).json()
    url = f"/api/agents/{agent['id']}"
    for body in ({"temperature": None}, {"name": None}, {"model": None}, {"provider": None}):
        r = await client.patch(url, json=body)
        assert r.status_code == 422, (body, r.text)
        assert r.json()["error"]["code"] == "validation_error"
    r = await client.patch(url, json={"description": None, "instructions": None, "model": " c2 "})
    assert r.status_code == 200
    assert (r.json()["description"], r.json()["instructions"], r.json()["model"]) == ("", "", "c2")
