"""DOCX text extraction that preserves heading structure and tables.

Word documents have no fixed pages, so every block is tagged with its heading
path (e.g. "Service Agreement > 4. Termination") which is used as the citation
location. Tables are flattened row by row as "Header: value" pairs so each
row stays meaningful after chunking.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.errors import CorruptedFileError
from app.ingestion.extractors.base import ExtractedDocument, TextBlock, normalise_text
from app.ingestion.validation import check_docx_archive

_MAX_SECTION_DEPTH = 3


def _heading_level(paragraph: Paragraph) -> int | None:
    style = (paragraph.style.name if paragraph.style is not None else "") or ""
    if style == "Title":
        return 0
    if style.startswith("Heading"):
        suffix = style.removeprefix("Heading").strip()
        return int(suffix) if suffix.isdigit() else 1
    return None


def _table_text(table: Table) -> str:
    rows = [[normalise_text(cell.text) for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if not body:
        return " | ".join(header)
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row, strict=False) if v]
        lines.append("; ".join(pairs))
    return "\n".join(lines)


class DocxExtractor:
    def extract(self, path: Path) -> ExtractedDocument:
        check_docx_archive(path, "This Word document")  # also guards re-processing
        try:
            document: DocxDocument = docx.Document(str(path))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise CorruptedFileError(
                "The file could not be read as a Word (.docx) document. It may be corrupted, "
                "password-protected, or an older .doc file renamed to .docx. Re-save it as "
                ".docx and try again."
            ) from exc

        result = ExtractedDocument(page_count=None)
        headings: dict[int, str] = {}
        buffer: list[str] = []

        def section() -> str | None:
            # The document title (level 0) is only used until a real heading appears.
            levels = sorted(lvl for lvl in headings if lvl > 0) or sorted(headings)
            parts = [headings[level] for level in levels]
            return " > ".join(parts[-_MAX_SECTION_DEPTH:]) if parts else None

        def flush() -> None:
            text = normalise_text("\n\n".join(buffer))
            if text:
                result.blocks.append(TextBlock(text, section=section()))
            buffer.clear()

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                level = _heading_level(item)
                if level is not None:
                    flush()
                    headings = {lvl: h for lvl, h in headings.items() if lvl < level}
                    headings[level] = text
                    continue
                buffer.append(text)
            elif isinstance(item, Table):
                table_text = _table_text(item)
                if table_text:
                    buffer.append(table_text)
        flush()
        return result
