"""Ingestion pipeline: extract -> (OCR) -> chunk -> embed.

The pipeline is deliberately free of database code: it turns a file into a
list of `PreparedChunk`s. The worker (app/workers/ingestion.py) owns the
document's lifecycle and persistence. This keeps the pipeline easy to unit
test with a fake embedder.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.db.models import DocumentStage
from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunker import Chunker, TextChunk
from app.ingestion.errors import NoExtractableTextError, UnsupportedFileError
from app.ingestion.extractors.base import ExtractedDocument, Extractor
from app.ingestion.extractors.docx import DocxExtractor
from app.ingestion.extractors.pdf import PdfExtractor
from app.ingestion.language import detect_language

StageCallback = Callable[[DocumentStage], Awaitable[None]]


@dataclass(slots=True)
class PreparedChunk:
    chunk: TextChunk
    # Text that is embedded and full-text indexed: the chunk prefixed with a
    # short header naming its document and location ("contextual header").
    indexed_text: str
    embedding: list[float]
    # PostgreSQL text-search configuration used for this chunk's keyword index.
    language: str = "english"


@dataclass(slots=True)
class IngestionResult:
    chunks: list[PreparedChunk] = field(default_factory=list)
    page_count: int | None = None
    ocr_page_count: int = 0
    char_count: int = 0
    embedding_tokens: int = 0
    embedding_model: str = ""
    embedding_provider: str = ""


def contextual_header(filename: str, chunk: TextChunk) -> str:
    location = []
    if chunk.section:
        location.append(f"Section: {chunk.section}")
    if chunk.page_number:
        location.append(f"Page {chunk.page_number}")
    return f"Document: {filename}" + (f" | {' | '.join(location)}" if location else "")


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunker: Chunker,
        extractors: dict[str, Extractor],
        languages: list[str] | None = None,
    ) -> None:
        self.embedder = embedder
        self.chunker = chunker
        self.extractors = extractors
        self.languages = languages or ["english"]

    @classmethod
    def from_settings(cls, settings: Settings, embedder: EmbeddingProvider) -> IngestionPipeline:
        return cls(
            embedder=embedder,
            chunker=Chunker(settings.chunk_size_tokens, settings.chunk_overlap_tokens),
            extractors={
                "pdf": PdfExtractor(
                    ocr_enabled=settings.ocr_enabled,
                    ocr_languages=settings.ocr_languages,
                    ocr_dpi=settings.ocr_dpi,
                    ocr_min_chars=settings.ocr_min_chars_per_page,
                ),
                "docx": DocxExtractor(),
            },
            languages=settings.fts_languages,
        )

    async def run(
        self,
        path: Path,
        file_type: str,
        filename: str,
        on_stage: StageCallback | None = None,
    ) -> IngestionResult:
        async def stage(s: DocumentStage) -> None:
            if on_stage:
                await on_stage(s)

        extractor = self.extractors.get(file_type)
        if extractor is None:
            raise UnsupportedFileError(f"No extractor available for '{file_type}' files.")

        await stage(DocumentStage.EXTRACTING)
        # Parsing/OCR is CPU-bound and blocking: keep it off the event loop.
        extracted: ExtractedDocument = await asyncio.to_thread(extractor.extract, path)
        if not extracted.blocks or extracted.char_count == 0:
            raise NoExtractableTextError(
                f"'{filename}' contains no extractable text. The document may be empty, "
                "contain only images, or use an unsupported encoding."
            )

        await stage(DocumentStage.CHUNKING)
        chunks = await asyncio.to_thread(self.chunker.chunk, extracted.blocks)

        await stage(DocumentStage.EMBEDDING)
        indexed = [f"{contextual_header(filename, c)}\n\n{c.text}" for c in chunks]
        embedded = await self.embedder.embed(indexed)

        return IngestionResult(
            chunks=[
                PreparedChunk(
                    chunk=c,
                    indexed_text=t,
                    embedding=v,
                    language=detect_language(c.text, self.languages),
                )
                for c, t, v in zip(chunks, indexed, embedded.vectors, strict=True)
            ],
            page_count=extracted.page_count,
            ocr_page_count=extracted.ocr_page_count,
            char_count=extracted.char_count,
            embedding_tokens=embedded.prompt_tokens,
            embedding_model=embedded.model,
            embedding_provider=embedded.provider,
        )
