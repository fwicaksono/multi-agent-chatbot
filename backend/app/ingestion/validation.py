"""Up-front validation of uploaded files (before they are queued).

Cheap checks run synchronously in the upload request so obviously bad files
are rejected immediately with a clear reason; deeper problems (corrupted
internals, scanned PDFs) are detected by the worker and reported on the
document's status.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePath

from app.ingestion.errors import EmptyFileError, FileTooLargeError, UnsupportedFileError


@dataclass(frozen=True, slots=True)
class FileKind:
    file_type: str
    content_type: str


_KINDS = {
    ".pdf": FileKind("pdf", "application/pdf"),
    ".docx": FileKind(
        "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
}


# A .docx is a ZIP archive. These limits stop "zip bombs" (a few MB on disk,
# hundreds of MB unpacked) that would tie up the worker for many minutes.
# Very large real documents stay far below them.
MAX_DOCX_UNPACKED_BYTES = 200 * 1024 * 1024
MAX_DOCX_XML_BYTES = 60 * 1024 * 1024


def check_docx_archive(source: bytes | Path, name: str) -> None:
    """Reject archives that unpack to an unreasonable size (reads only the ZIP index)."""
    try:
        with zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source) as z:
            entries = z.infolist()
    except (zipfile.BadZipFile, OSError):
        return  # not a readable archive: the extractor reports it as corrupted
    total = sum(e.file_size for e in entries)
    xml = sum(e.file_size for e in entries if e.filename.endswith((".xml", ".rels")))
    if total > MAX_DOCX_UNPACKED_BYTES or xml > MAX_DOCX_XML_BYTES:
        raise FileTooLargeError(
            f"{name} unpacks to {total / 1_048_576:.0f} MB, more than this server accepts "
            f"({MAX_DOCX_UNPACKED_BYTES // 1_048_576} MB, of which at most "
            f"{MAX_DOCX_XML_BYTES // 1_048_576} MB of text). Split the document into smaller files."
        )


def validate_upload(filename: str, data: bytes, max_bytes: int) -> FileKind:
    suffix = PurePath(filename).suffix.lower()
    kind = _KINDS.get(suffix)
    if kind is None:
        if suffix == ".doc":
            raise UnsupportedFileError(
                f"'{filename}' is a legacy .doc file. Open it in Word and save as .docx."
            )
        raise UnsupportedFileError(
            f"'{filename}' is not supported. Upload PDF (.pdf) or Word (.docx) files."
        )
    if not data:
        raise EmptyFileError(f"'{filename}' is empty (0 bytes).")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(data) / 1_048_576:.1f} MB; the limit is "
            f"{max_bytes / 1_048_576:.0f} MB."
        )
    # Magic-byte sniffing: catches renamed files early.
    if kind.file_type == "pdf" and b"%PDF-" not in data[:1024]:
        raise UnsupportedFileError(
            f"'{filename}' has a .pdf extension but is not a PDF file (it may be corrupted)."
        )
    if kind.file_type == "docx" and not data.startswith(b"PK"):
        raise UnsupportedFileError(
            f"'{filename}' has a .docx extension but is not a valid Word document "
            "(it may be corrupted or a renamed .doc file)."
        )
    if kind.file_type == "docx":
        check_docx_archive(data, f"'{filename}'")
    return kind
