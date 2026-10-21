import pytest

from app.ingestion.chunker import Chunker, count_tokens
from app.ingestion.extractors.base import TextBlock


def _paragraphs(n: int, words: int = 30) -> str:
    return "\n\n".join(
        f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(words)) + "." for i in range(n)
    )


def test_chunks_respect_size_limit():
    chunker = Chunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk([TextBlock(_paragraphs(12), page_number=1)])
    assert len(chunks) > 1
    # Small slack for the "\n\n" joiners between units.
    assert all(c.token_count <= 100 + 5 for c in chunks)


def test_chunks_never_cross_page_or_section_boundaries():
    blocks = [
        TextBlock(_paragraphs(3), page_number=1),
        TextBlock(_paragraphs(3), page_number=2),
        TextBlock("Short clause.", section="4. Termination"),
    ]
    chunks = Chunker(chunk_size=200, overlap=20).chunk(blocks)
    pages = {c.page_number for c in chunks if c.page_number}
    assert pages == {1, 2}
    for c in chunks:
        if c.page_number == 1:
            assert "Paragraph 0." in c.text or "word" in c.text
    assert chunks[-1].section == "4. Termination"
    assert chunks[-1].page_number is None
    # Indexes are sequential across the whole document.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    chunker = Chunker(chunk_size=80, overlap=40)
    text = "\n\n".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunker.chunk([TextBlock(text, page_number=1)])
    assert len(chunks) >= 2
    first_units = chunks[0].text.split("\n\n")
    second_units = chunks[1].text.split("\n\n")
    # The next chunk starts with the trailing sentences of the previous one.
    carried = [u for u in second_units if u in first_units]
    assert carried and carried == first_units[-len(carried) :]
    assert second_units[0] == carried[0]
    assert sum(count_tokens(u) for u in carried) <= 40


def test_overlong_sentence_is_hard_split():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = " ".join(f"token{i}" for i in range(400))  # no punctuation at all
    chunks = chunker.chunk([TextBlock(text, page_number=3)])
    assert len(chunks) > 3
    assert all(count_tokens(c.text) <= 50 + 2 for c in chunks)
    assert all(c.page_number == 3 for c in chunks)


def test_empty_blocks_produce_no_chunks():
    assert Chunker().chunk([TextBlock("   \n\n  ", page_number=1)]) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        Chunker(chunk_size=size, overlap=overlap)


def test_ocr_flag_propagates():
    chunks = Chunker().chunk([TextBlock("Scanned text here.", page_number=1, ocr=True)])
    assert chunks[0].ocr is True
