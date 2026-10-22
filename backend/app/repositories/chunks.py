"""Chunk persistence and the two retrieval queries (vector + keyword).

Every query here takes an `agent_id` and filters on it -- there is no way to
search across agents. Only chunks of documents whose status is `ready` are
searchable, so a document that is being re-processed or was just deleted can
never leak into answers.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import cast, func, insert, literal, select, text
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus
from app.ingestion.language import query_terms
from app.ingestion.pipeline import PreparedChunk

_INSERT_BATCH = 200


@dataclass(slots=True)
class ChunkHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    file_type: str
    content: str
    page_number: int | None
    section: str | None
    chunk_index: int
    score: float  # cosine similarity (vector) or ts_rank (keyword)


async def insert_chunks(
    session: AsyncSession,
    agent_id: uuid.UUID,
    document_id: uuid.UUID,
    chunks: list[PreparedChunk],
) -> None:
    for start in range(0, len(chunks), _INSERT_BATCH):
        rows = [
            {
                "id": uuid.uuid4(),
                "agent_id": agent_id,
                "document_id": document_id,
                "chunk_index": p.chunk.index,
                "content": p.chunk.text,
                "page_number": p.chunk.page_number,
                "section": p.chunk.section,
                "token_count": p.chunk.token_count,
                "embedding": p.embedding,
                # Each chunk is indexed with the rules of its own language.
                "tsv": func.to_tsvector(_regconfig(p.language), p.indexed_text),
                "metadata": {"ocr": p.chunk.ocr, "language": p.language},
            }
            for p in chunks[start : start + _INSERT_BATCH]
        ]
        await session.execute(insert(Chunk).values(rows))


def _base_query(agent_id: uuid.UUID):
    return (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.filename,
            Document.file_type,
            Chunk.content,
            Chunk.page_number,
            Chunk.section,
            Chunk.chunk_index,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.agent_id == agent_id, Document.status == DocumentStatus.READY)
    )


def _hit(row, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=row.id,
        document_id=row.document_id,
        document_name=row.filename,
        file_type=row.file_type,
        content=row.content,
        page_number=row.page_number,
        section=row.section,
        chunk_index=row.chunk_index,
        score=float(score),
    )


_iterative_scan_supported: bool | None = None


async def _supports_iterative_scan(session: AsyncSession) -> bool:
    global _iterative_scan_supported
    if _iterative_scan_supported is None:
        version = await session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        parts = tuple(int(p) for p in re.findall(r"\d+", version or "0")[:2])
        _iterative_scan_supported = parts >= (0, 8)
    return _iterative_scan_supported


async def vector_search(
    session: AsyncSession, agent_id: uuid.UUID, embedding: list[float], limit: int
) -> list[ChunkHit]:
    # With a WHERE filter, a plain HNSW scan returns the global top-ef_search
    # neighbours and *then* filters, which can starve small agents. pgvector
    # >= 0.8 "iterative scans" keep walking the graph until enough rows pass.
    # Older pgvector (e.g. some managed databases) rejects the setting, so it
    # is only applied when supported.
    if await _supports_iterative_scan(session):
        await session.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
    distance = Chunk.embedding.cosine_distance(embedding)
    rows = await session.execute(
        _base_query(agent_id)
        .add_columns(distance.label("distance"))
        .order_by(distance)
        .limit(limit)
    )
    hits = [_hit(r, 1.0 - r.distance) for r in rows]
    return sorted(hits, key=lambda h: h.score, reverse=True)


def _regconfig(language: str):
    return cast(literal(language), REGCONFIG)


def to_or_tsquery_text(query: str, languages: list[str]) -> str:
    """'Berapa hari cuti tahunan?' -> 'hari | cuti | tahunan'.

    OR semantics (instead of websearch_to_tsquery's AND) suit natural-language
    questions; ranking rewards chunks that match more/rarer terms. Stop words of
    every configured language are dropped first (PostgreSQL's `indonesian`
    configuration has no stop-word list of its own).
    """
    return " | ".join(query_terms(query, languages))


def multilingual_tsquery(query: str, languages: list[str]):
    """One tsquery per language, OR-ed together, so a question matches chunks
    indexed in any language: 'tahunan' is stemmed to 'tahun' for Indonesian
    chunks and kept as-is for English ones."""
    if isinstance(languages, str):  # tolerate a single configuration name
        languages = [languages]
    text_query = to_or_tsquery_text(query, languages)
    if not text_query:
        return None
    combined = None
    for language in languages:
        part = func.to_tsquery(_regconfig(language), text_query)
        combined = part if combined is None else combined.op("||")(part)
    return combined


async def keyword_search(
    session: AsyncSession,
    agent_id: uuid.UUID,
    query: str,
    limit: int,
    languages: list[str],
) -> list[ChunkHit]:
    tsquery = multilingual_tsquery(query, languages)
    if tsquery is None:
        return []
    rank = func.ts_rank_cd(Chunk.tsv, tsquery, 32)  # 32: rank / (rank + 1)
    rows = await session.execute(
        _base_query(agent_id)
        .add_columns(rank.label("rank"))
        .where(Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    return [_hit(r, r.rank) for r in rows]


async def unknown_text_search_configs(session: AsyncSession, languages: list[str]) -> list[str]:
    """Configured languages that this PostgreSQL server does not provide."""
    rows = await session.scalars(text("SELECT cfgname FROM pg_ts_config"))
    available = set(rows)
    return [lang for lang in languages if lang not in available]


async def count_ready_chunks(session: AsyncSession, agent_id: uuid.UUID) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.agent_id == agent_id, Document.status == DocumentStatus.READY)
        )
        or 0
    )


async def verify_text_search_configs(languages: list[str]) -> None:
    """Fail fast at startup when FTS_LANGUAGES names an unknown configuration."""
    from app.db.session import session_scope

    async with session_scope() as session:
        unknown = await unknown_text_search_configs(session, languages)
    if unknown:
        raise RuntimeError(
            f"FTS_LANGUAGES contains {', '.join(unknown)}, which PostgreSQL does not provide. "
            "Use built-in configurations such as english, indonesian or simple "
            "(see: SELECT cfgname FROM pg_ts_config)."
        )
