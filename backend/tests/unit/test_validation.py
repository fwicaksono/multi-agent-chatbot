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


def _docx_with_xml_of(size: int) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "<w:p>" + "a" * size + "</w:p>")
    return buf.getvalue()


def test_zip_bombs_are_rejected_at_upload(monkeypatch):
    """Regression: a 1.4 MB .docx unpacking to 350 MB kept the worker busy for >10 minutes."""
    from app.ingestion import validation

    monkeypatch.setattr(validation, "MAX_DOCX_XML_BYTES", 1024 * 1024)
    bomb = _docx_with_xml_of(3 * 1024 * 1024)  # tiny on disk thanks to compression
    assert len(bomb) < 50_000
    with pytest.raises(FileTooLargeError, match="unpacks to"):
        validate_upload("bomb.docx", bomb, MB * 25)
    # A normal document passes.
    assert validate_upload("ok.docx", _docx_with_xml_of(1000), MB).file_type == "docx"


def test_extractor_also_refuses_zip_bombs(monkeypatch, tmp_path):
    from app.ingestion import validation
    from app.ingestion.extractors.docx import DocxExtractor

    monkeypatch.setattr(validation, "MAX_DOCX_XML_BYTES", 1024 * 1024)
    path = tmp_path / "bomb.docx"
    path.write_bytes(_docx_with_xml_of(3 * 1024 * 1024))
    with pytest.raises(FileTooLargeError):
        DocxExtractor().extract(path)
