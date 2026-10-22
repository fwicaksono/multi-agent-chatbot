"""Retrieval fusion, citation parsing, prompt construction and pricing."""

from __future__ import annotations

import uuid

import pytest

from app.llm import pricing
from app.llm.base import ChatMessage
from app.rag.citations import build_citations, build_sources, cited_indexes
from app.rag.prompts import IDK_ANSWER, build_answer_messages, build_rewrite_messages
from app.repositories.chunks import ChunkHit, to_or_tsquery_text
from app.retrieval.fusion import reciprocal_rank_fusion


def hit(
    name: str, score: float = 0.5, page: int | None = 1, section: str | None = None
) -> ChunkHit:
    return ChunkHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, name),
        document_id=uuid.uuid5(uuid.NAMESPACE_URL, "doc"),
        document_name=f"{name}.pdf",
        file_type="pdf",
        content=f"content of {name}",
        page_number=page,
        section=section,
        chunk_index=0,
        score=score,
    )


# --------------------------------------------------------------------------- fusion


def test_rrf_rewards_agreement_between_retrievers():
    a, b, c, d = hit("a"), hit("b"), hit("c"), hit("d")
    fused = reciprocal_rank_fusion([a, b, c], [c, d], k=60)
    assert fused[0].hit.chunk_id == c.chunk_id  # found by both
    assert fused[0].matched_by == ["vector", "keyword"]
    assert {f.hit.chunk_id for f in fused} == {a.chunk_id, b.chunk_id, c.chunk_id, d.chunk_id}


def test_rrf_handles_single_source():
    fused = reciprocal_rank_fusion([], [hit("x"), hit("y")])
    assert [f.hit.document_name for f in fused] == ["x.pdf", "y.pdf"]
    assert fused[0].vector_rank is None and fused[0].keyword_rank == 1


def test_or_tsquery_builder_sanitises_input():
    assert to_or_tsquery_text("What's the notice-period (in days)?!") == (
        "what | the | notice | period | in | days"
    )
    assert to_or_tsquery_text("?? ! &&") == ""


# --------------------------------------------------------------------------- citations


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Leave is 18 days [2].", [2]),
        ("A [1][3] and B [1, 2].", [1, 3, 2]),
        ("Range [2-4].", [2, 3, 4]),
        ("Out of range [9] ignored, [0] too.", []),
        ("No citations here.", []),
    ],
)
def test_cited_indexes(answer, expected):
    assert cited_indexes(answer, max_index=5) == expected


def test_citations_are_the_cited_subset_of_sources():
    from app.retrieval.fusion import FusedHit

    hits = [
        FusedHit(hit("a", page=3), 0.03),
        FusedHit(hit("b", page=None, section="4. Term"), 0.02),
    ]
    sources = build_sources(hits)
    assert sources[0]["location"] == "page 3"
    assert sources[1]["location"] == 'section "4. Term"'
    citations = build_citations("Per the policy [2].", sources)
    assert [c["document_name"] for c in citations] == ["b.pdf"]


# --------------------------------------------------------------------------- prompts


def test_answer_prompt_contains_rules_context_and_history():
    from app.retrieval.fusion import FusedHit

    history = [ChatMessage("user", "Tell me about leave"), ChatMessage("assistant", "Sure [1].")]
    msgs = build_answer_messages(
        agent_name="Agent HR",
        agent_instructions="Be friendly.",
        history=history,
        question="What about sick leave?",
        hits=[FusedHit(hit("handbook", page=2), 0.1)],
        standalone_question="How many sick leave days do employees get?",
    )
    system = msgs[0].content
    assert msgs[0].role == "system"
    assert "Be friendly." in system and IDK_ANSWER in system
    assert "[1] Source: handbook.pdf (page 2)" in system
    assert msgs[1:3] == history
    assert "How many sick leave days" in msgs[-1].content


def test_empty_context_is_explicit():
    msgs = build_answer_messages(
        agent_name="A", agent_instructions="", history=[], question="q", hits=[]
    )
    assert "no relevant passages" in msgs[0].content


def test_rewrite_prompt_includes_transcript():
    msgs = build_rewrite_messages(
        [ChatMessage("user", "How long is annual leave?"), ChatMessage("assistant", "18 days")],
        "What about the contract then?",
    )
    assert "USER: How long is annual leave?" in msgs[1].content
    assert "What about the contract then?" in msgs[1].content


# --------------------------------------------------------------------------- pricing


def test_cost_estimation_and_snapshot_names():
    assert pricing.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert pricing.estimate_cost("openai/gpt-4o-mini-2024-07-18", 1_000_000) == pytest.approx(0.15)
    assert pricing.estimate_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert pricing.estimate_cost("totally-unknown-model", 100, 100) is None


# --------------------------------------------------------------------------- pgvector version


@pytest.mark.parametrize("version,expected", [("0.8.0", True), ("0.10.1", True), ("0.7.4", False)])
async def test_iterative_scan_only_on_pgvector_0_8_plus(monkeypatch, version, expected):
    from app.repositories import chunks

    class FakeSession:
        async def scalar(self, _stmt):
            return version

    monkeypatch.setattr(chunks, "_iterative_scan_supported", None)
    assert await chunks._supports_iterative_scan(FakeSession()) is expected
