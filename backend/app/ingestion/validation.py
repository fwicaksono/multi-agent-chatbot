"""Up-front validation of uploaded files (before they are queued).

Cheap checks run synchronously in the upload request so obviously bad files
are rejected immediately with a clear reason; deeper problems (corrupted
internals, scanned PDFs) are detected by the worker and reported on the
document's status.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

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
    return kind
