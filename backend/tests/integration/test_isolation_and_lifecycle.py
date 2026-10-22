"""Knowledge-base isolation and deletion guarantees, against a real database."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.models import Chunk, Conversation, Document, UsageRecord
from app.db.session import session_scope
from app.ingestion.pipeline import IngestionPipeline
from app.repositories import agents as agent_repo
from app.repositories import chunks as chunk_repo
from app.repositories import conversations as conv_repo
from app.repositories import documents as doc_repo
from app.repositories import usage as usage_repo
from app.retrieval.retriever import HybridRetriever
from tests.conftest import SAMPLE_DATA, FakeEmbedder

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("db")]


async def _agent(name: str):
    async with session_scope() as s:
        return await agent_repo.create_agent(
            s, name=name, description="", instructions="", provider="openai", model="gpt-4.1-mini"
        )


async def _ingest(agent_id, path, file_type):
    settings = get_settings()
    embedder = FakeEmbedder(settings)
    result = await IngestionPipeline.from_settings(settings, embedder).run(
        path, file_type, path.name
    )
    async with session_scope() as s:
        doc = await doc_repo.create_document(
            s,
            agent_id=agent_id,
            filename=path.name,
            file_type=file_type,
            content_type="application/octet-stream",
            size_bytes=path.stat().st_size,
            sha256=path.name,
            storage_path=str(path),
        )
        await chunk_repo.insert_chunks(s, agent_id, doc.id, result.chunks)
        await doc_repo.mark_ready(s, doc.id, chunk_count=len(result.chunks))
    return doc, embedder


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
async def test_retrieval_never_crosses_agents(mode):
    hr = await _agent("HR")
    legal = await _agent("Legal")
    await _ingest(hr.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    _, embedder = await _ingest(
        legal.id, SAMPLE_DATA / "legal" / "service_agreement_template.docx", "docx"
    )
    retriever = HybridRetriever(get_settings(), embedder)

    async with session_scope() as s:
        # A legal question asked to the HR agent must only see HR chunks...
        hr_hits = await retriever.retrieve(s, hr.id, "termination notice period fees", mode=mode)
        legal_hits = await retriever.retrieve(s, legal.id, "annual leave days", mode=mode)

    assert {h.hit.document_name for h in hr_hits.hits} <= {"employee_handbook.pdf"}
    assert {h.hit.document_name for h in legal_hits.hits} <= {"service_agreement_template.docx"}


async def test_keyword_search_finds_exact_terms():
    agent = await _agent("Legal")
    await _ingest(agent.id, SAMPLE_DATA / "legal" / "service_agreement_template.docx", "docx")
    async with session_scope() as s:
        hits = await chunk_repo.keyword_search(
            s, agent.id, "BANI arbitration Jakarta", 5, ["english", "indonesian"]
        )
    assert hits and "Governing Law" in (hits[0].section or "")


async def test_deleted_document_is_never_retrieved_again():
    agent = await _agent("HR")
    doc, embedder = await _ingest(agent.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    retriever = HybridRetriever(get_settings(), embedder)

    async with session_scope() as s:
        assert (await retriever.retrieve(s, agent.id, "annual leave")).hits
        await doc_repo.delete_document(s, await doc_repo.get_document(s, agent.id, doc.id))

    async with session_scope() as s:
        assert (await retriever.retrieve(s, agent.id, "annual leave")).hits == []
        assert await s.scalar(select(func.count()).select_from(Chunk)) == 0


async def test_processing_documents_are_not_searchable():
    agent = await _agent("HR")
    doc, embedder = await _ingest(agent.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    async with session_scope() as s:
        await doc_repo.requeue(s, await doc_repo.get_document(s, agent.id, doc.id))
    async with session_scope() as s:
        assert await chunk_repo.count_ready_chunks(s, agent.id) == 0


async def test_deleting_agent_removes_all_its_data():
    hr = await _agent("HR")
    other = await _agent("Other")
    await _ingest(hr.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    await _ingest(other.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    async with session_scope() as s:
        conv = await conv_repo.create_conversation(s, hr.id)
        await conv_repo.add_message(s, conv.id, role="user", content="hi")
        await usage_repo.record_usage(
            s, agent_id=hr.id, kind="chat", provider="openai", model="gpt-4o-mini", prompt_tokens=10
        )

    async with session_scope() as s:
        await agent_repo.delete_agent(s, await agent_repo.get_agent(s, hr.id))

    async with session_scope() as s:
        for model in (Document, Chunk, Conversation, UsageRecord):
            remaining = await s.scalar(
                select(func.count()).select_from(model).where(model.agent_id == hr.id)
            )
            assert remaining == 0, model.__name__
        # The other agent is untouched.
        assert await chunk_repo.count_ready_chunks(s, other.id) > 0


async def test_worker_queue_claims_each_job_once():
    agent = await _agent("HR")
    async with session_scope() as s:
        for i in range(3):
            await doc_repo.create_document(
                s,
                agent_id=agent.id,
                filename=f"{i}.pdf",
                file_type="pdf",
                content_type="application/pdf",
                size_bytes=1,
                sha256=str(i),
                storage_path="/nope",
            )
    claimed = []
    for _ in range(4):
        async with session_scope() as s:
            doc = await doc_repo.claim_next(s, stale_after_seconds=600)
            claimed.append(doc.filename if doc else None)
    assert sorted(c for c in claimed if c) == ["0.pdf", "1.pdf", "2.pdf"]
    assert claimed[-1] is None


async def test_indonesian_chunks_are_stemmed_and_mixed_languages_work():
    """Indonesian documents get Indonesian stemming; English ones keep English rules."""
    agent = await _agent("Mixed")
    await _ingest(agent.id, SAMPLE_DATA / "hr_id" / "peraturan_perusahaan.pdf", "pdf")
    await _ingest(agent.id, SAMPLE_DATA / "hr" / "employee_handbook.pdf", "pdf")
    langs = ["english", "indonesian"]

    async with session_scope() as s:
        languages = {
            row.filename: row.language
            for row in await s.execute(
                select(Document.filename, Chunk.meta["language"].astext.label("language"))
                .join(Chunk, Chunk.document_id == Document.id)
                .where(Chunk.agent_id == agent.id)
            )
        }
        # "mengundurkan" (question) and "pengunduran" (document) share the stem "undur".
        resign = await chunk_repo.keyword_search(s, agent.id, "mengundurkan diri", 5, langs)
        # "tahunan" in the question matches "tahunan"/"tahun" in the document.
        leave_id = await chunk_repo.keyword_search(s, agent.id, "jatah cuti tahunan", 5, langs)
        leave_en = await chunk_repo.keyword_search(s, agent.id, "annual leave carry over", 5, langs)
        english_only = await chunk_repo.keyword_search(s, agent.id, "mengundurkan", 5, ["english"])

    assert languages == {
        "peraturan_perusahaan.pdf": "indonesian",
        "employee_handbook.pdf": "english",
    }
    assert resign and resign[0].document_name == "peraturan_perusahaan.pdf"
    assert resign[0].page_number == 1
    assert leave_id and leave_id[0].document_name == "peraturan_perusahaan.pdf"
    assert leave_en and leave_en[0].document_name == "employee_handbook.pdf"
    assert english_only == []  # English rules cannot match "pengunduran" from "mengundurkan"
