"""User-facing ingestion failures.

Each error has a stable `code` (stored on the document row) and a message that
is shown verbatim in the UI, so it should tell the user what went wrong and
what they can do about it.
"""

from __future__ import annotations

from app.core.errors import AppError


class IngestionError(AppError):
    status_code = 422
    code = "ingestion_failed"


class UnsupportedFileError(IngestionError):
    code = "unsupported_file_type"


class EmptyFileError(IngestionError):
    code = "empty_file"


class FileTooLargeError(IngestionError):
    status_code = 413
    code = "file_too_large"


class CorruptedFileError(IngestionError):
    code = "corrupted_file"


class EncryptedFileError(IngestionError):
    code = "encrypted_file"


class NoExtractableTextError(IngestionError):
    code = "no_extractable_text"


class ScannedPdfError(IngestionError):
    code = "scanned_pdf_no_ocr"
