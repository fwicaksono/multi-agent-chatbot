from pathlib import Path

import pytest

from app.db.models import DocumentStage
from app.ingestion.errors import CorruptedFileError, NoExtractableTextError, UnsupportedFileError
from app.ingestion.extractors.base import ExtractedDocument, TextBlock
from app.ingestion.pipeline import IngestionPipeline, contextual_header
from app.llm.errors import ProviderRateLimitError


@pytest.fixture
def pipeline(settings, fake_embedder) -> IngestionPipeline:
    return IngestionPipeline.from_settings(settings, fake_embedder)


async def test_pdf_end_to_end(pipeline, fake_embedder, sample_data: Path):
    stages: list[DocumentStage] = []

    async def on_stage(stage):
        stages.append(stage)

    result = await pipeline.run(
        sample_data / "hr" / "employee_handbook.pdf", "pdf", "employee_handbook.pdf", on_stage
    )

    assert stages == [DocumentStage.EXTRACTING, DocumentStage.CHUNKING, DocumentStage.EMBEDDING]
    assert result.page_count == 3
    assert len(result.chunks) >= 3
    assert {c.chunk.page_number for c in result.chunks} == {1, 2, 3}
    assert all(len(c.embedding) == 8 for c in result.chunks)
    assert result.embedding_tokens > 0
    # The indexed text carries a contextual header naming document and page.
    first = result.chunks[0]
    assert first.indexed_text.startswith("Document: employee_handbook.pdf | Page 1")
    assert first.indexed_text.endswith(first.chunk.text)
    # Everything was embedded, in one batch for a small document.
    assert sum(len(batch) for batch in fake_embedder.calls) == len(result.chunks)


async def test_docx_end_to_end(pipeline, sample_data: Path):
    result = await pipeline.run(
        sample_data / "legal" / "service_agreement_template.docx", "docx", "msa.docx"
    )
    assert result.page_count is None
    assert all(c.chunk.section for c in result.chunks)
    termination = [c for c in result.chunks if "90 days" in c.chunk.text]
    assert termination and "Termination for Convenience" in termination[0].chunk.section
    assert "Section: " in termination[0].indexed_text


async def test_corrupted_file_propagates_user_facing_error(pipeline, sample_data: Path):
    with pytest.raises(CorruptedFileError):
        await pipeline.run(sample_data / "invalid" / "corrupted.pdf", "pdf", "corrupted.pdf")


async def test_document_without_text_fails_clearly(settings, fake_embedder, tmp_path):
    class EmptyExtractor:
        def extract(self, path):
            return ExtractedDocument(blocks=[], page_count=1)

    pipeline = IngestionPipeline.from_settings(settings, fake_embedder)
    pipeline.extractors["pdf"] = EmptyExtractor()
    with pytest.raises(NoExtractableTextError, match="contains no extractable text"):
        await pipeline.run(tmp_path / "x.pdf", "pdf", "x.pdf")
    assert fake_embedder.calls == []  # never pays for embeddings on empty docs


async def test_unsupported_type(pipeline, tmp_path):
    with pytest.raises(UnsupportedFileError):
        await pipeline.run(tmp_path / "x.txt", "txt", "x.txt")


async def test_embedding_failures_bubble_up_for_retry(settings, fake_embedder, sample_data):
    async def boom(texts):
        raise ProviderRateLimitError("slow down", provider="fake")

    fake_embedder._embed_batch = boom
    pipeline = IngestionPipeline.from_settings(settings, fake_embedder)
    with pytest.raises(ProviderRateLimitError):
        await pipeline.run(sample_data / "hr" / "employee_handbook.pdf", "pdf", "h.pdf")


async def test_embeddings_are_batched(settings, fake_embedder):
    settings.embedding_batch_size = 2

    class ManyPages:
        def extract(self, path):
            return ExtractedDocument(
                blocks=[TextBlock(f"Page {i} text.", page_number=i) for i in range(1, 6)],
                page_count=5,
            )

    pipeline = IngestionPipeline.from_settings(settings, fake_embedder)
    pipeline.embedder.batch_size = 2
    pipeline.extractors["pdf"] = ManyPages()
    result = await pipeline.run(Path("unused.pdf"), "pdf", "many.pdf")
    assert len(result.chunks) == 5
    assert [len(b) for b in fake_embedder.calls] == [2, 2, 1]


def test_contextual_header_formats():
    from app.ingestion.chunker import TextChunk

    pdf_chunk = TextChunk(0, "x", 1, page_number=4, section=None)
    docx_chunk = TextChunk(0, "x", 1, page_number=None, section="A > B")
    assert contextual_header("f.pdf", pdf_chunk) == "Document: f.pdf | Page 4"
    assert contextual_header("f.docx", docx_chunk) == "Document: f.docx | Section: A > B"
