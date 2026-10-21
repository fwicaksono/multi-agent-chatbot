"""Common types for text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class TextBlock:
    """A run of text with its location, used for citations.

    PDFs are located by `page_number`; DOCX files have no stable pages, so
    they are located by `section` (the heading path, e.g. "4. Termination").
    """

    text: str
    page_number: int | None = None
    section: str | None = None
    ocr: bool = False


@dataclass(slots=True)
class ExtractedDocument:
    blocks: list[TextBlock] = field(default_factory=list)
    page_count: int | None = None
    ocr_page_count: int = 0

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class Extractor(Protocol):
    def extract(self, path: Path) -> ExtractedDocument: ...


_WS = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def normalise_text(text: str) -> str:
    """Tidy extracted text without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words hyphenated across lines
    text = "\n".join(_WS.sub(" ", line).strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()
