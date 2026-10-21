import pytest

from app.ingestion.errors import EmptyFileError, FileTooLargeError, UnsupportedFileError
from app.ingestion.validation import validate_upload

MB = 1024 * 1024


def test_accepts_pdf_and_docx():
    assert validate_upload("a.PDF", b"%PDF-1.7 ...", MB).file_type == "pdf"
    assert validate_upload("b.docx", b"PK\x03\x04...", MB).file_type == "docx"


@pytest.mark.parametrize(
    "name,data,error,fragment",
    [
        ("notes.txt", b"hello", UnsupportedFileError, "not supported"),
        ("old.doc", b"\xd0\xcf\x11\xe0", UnsupportedFileError, "save as .docx"),
        ("empty.pdf", b"", EmptyFileError, "empty"),
        ("fake.pdf", b"<html>not a pdf</html>", UnsupportedFileError, "not a PDF"),
        ("fake.docx", b"%PDF-1.4", UnsupportedFileError, "not a valid Word"),
    ],
)
def test_rejects_bad_uploads_with_clear_messages(name, data, error, fragment):
    with pytest.raises(error) as exc:
        validate_upload(name, data, MB)
    assert fragment in exc.value.message
    assert name in exc.value.message


def test_rejects_oversized_files():
    with pytest.raises(FileTooLargeError, match="limit is 1 MB"):
        validate_upload("big.pdf", b"%PDF-" + b"0" * (MB + 10), MB)
