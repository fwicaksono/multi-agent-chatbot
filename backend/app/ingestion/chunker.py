"""Structure-aware, token-based chunking.

Strategy (see README "Chunking"):

1. Chunks never cross a block boundary. A block is a PDF page or a DOCX
   section, so every chunk maps to exactly one page/section and citations are
   precise.
2. Within a block, text is split on paragraphs, then sentences, then (as a
   last resort) raw token windows, and greedily packed up to `chunk_size`
   tokens.
3. Consecutive chunks of the same block overlap by up to `overlap` tokens
   (whole sentences/paragraphs) so facts spanning a boundary stay retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from app.ingestion.extractors.base import TextBlock

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(\[])|\n")


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text, disallowed_special=()))


@dataclass(slots=True)
class TextChunk:
    index: int
    text: str
    token_count: int
    page_number: int | None
    section: str | None
    ocr: bool = False


class Chunker:
    def __init__(self, chunk_size: int = 450, overlap: int = 60) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, blocks: list[TextBlock]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for block in blocks:
            for text in self._split_block(block.text):
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        text=text,
                        token_count=count_tokens(text),
                        page_number=block.page_number,
                        section=block.section,
                        ocr=block.ocr,
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    def _units(self, text: str) -> list[str]:
        """Break text into units no larger than chunk_size tokens."""
        units: list[str] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if count_tokens(paragraph) <= self.chunk_size:
                units.append(paragraph)
                continue
            for sentence in _SENTENCE_SPLIT.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if count_tokens(sentence) <= self.chunk_size:
                    units.append(sentence)
                else:
                    units.extend(self._hard_split(sentence))
        return units

    def _hard_split(self, text: str) -> list[str]:
        enc = get_encoding()
        tokens = enc.encode(text, disallowed_special=())
        step = self.chunk_size - self.overlap
        return [enc.decode(tokens[i : i + self.chunk_size]) for i in range(0, len(tokens), step)]

    def _split_block(self, text: str) -> list[str]:
        units = self._units(text)
        sizes = [count_tokens(u) for u in units]
        chunks: list[str] = []
        current: list[int] = []  # indexes into units
        current_tokens = 0

        for i, size in enumerate(sizes):
            if current and current_tokens + size > self.chunk_size:
                chunks.append("\n\n".join(units[j] for j in current))
                # Start the next chunk with trailing units that fit in the overlap.
                carried: list[int] = []
                carried_tokens = 0
                for j in reversed(current):
                    if carried_tokens + sizes[j] > self.overlap:
                        break
                    carried.insert(0, j)
                    carried_tokens += sizes[j]
                if carried_tokens + size > self.chunk_size:
                    carried, carried_tokens = [], 0
                current, current_tokens = carried, carried_tokens
            current.append(i)
            current_tokens += size

        if current:
            chunks.append("\n\n".join(units[j] for j in current))
        return chunks
