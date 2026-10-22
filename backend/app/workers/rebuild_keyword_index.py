"""Rebuild the keyword (full-text) index of existing chunks.

Run after changing FTS_LANGUAGES, or once after upgrading to per-language
indexing, so documents ingested earlier get the right language rules:

    docker compose exec api python -m app.workers.rebuild_keyword_index

Only the PostgreSQL `tsvector` column is recomputed (language detection +
stemming). Embeddings are untouched, so this costs nothing and takes seconds.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from sqlalchemy import cast, func, literal, select, update
from sqlalchemy.dialects.postgresql import REGCONFIG

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.models import Chunk, Document
from app.db.session import session_scope
from app.ingestion.chunker import TextChunk
from app.ingestion.language import detect_language
from app.ingestion.pipeline import contextual_header
from app.repositories.chunks import verify_text_search_configs

logger = get_logger("app.rebuild_keyword_index")


async def rebuild() -> Counter:
    settings = get_settings()
    await verify_text_search_configs(settings.fts_languages)
    counts: Counter = Counter()
    async with session_scope() as session:
        rows = await session.execute(
            select(
                Chunk.id,
                Chunk.content,
                Chunk.page_number,
                Chunk.section,
                Chunk.meta,
                Document.filename,
            ).join(Document, Document.id == Chunk.document_id)
        )
        for row in rows:
            language = detect_language(row.content, settings.fts_languages)
            header = contextual_header(
                row.filename, TextChunk(0, row.content, 0, row.page_number, row.section)
            )
            await session.execute(
                update(Chunk)
                .where(Chunk.id == row.id)
                .values(
                    tsv=func.to_tsvector(
                        cast(literal(language), REGCONFIG), f"{header}\n\n{row.content}"
                    ),
                    meta={**(row.meta or {}), "language": language},
                )
            )
            counts[language] += 1
    return counts


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, "console")
    counts = await rebuild()
    total = sum(counts.values())
    summary = ", ".join(f"{lang}: {n}" for lang, n in counts.most_common()) or "none"
    print(f"Rebuilt the keyword index for {total} chunks ({summary}).")


if __name__ == "__main__":
    asyncio.run(_main())
