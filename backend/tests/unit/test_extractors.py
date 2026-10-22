from pathlib import Path

import pytest

from app.ingestion.errors import CorruptedFileError, ScannedPdfError
from app.ingestion.extractors import pdf as pdf_module
from app.ingestion.extractors.base import normalise_text
from app.ingestion.extractors.docx import DocxExtractor
from app.ingestion.extractors.pdf import PdfExtractor


def test_native_pdf_is_extracted_per_page(sample_data: Path):
    doc = PdfExtractor(ocr_enabled=False).extract(sample_data / "hr" / "employee_handbook.pdf")
    assert doc.page_count == 3
    assert [b.page_number for b in doc.blocks] == [1, 2, 3]
    assert doc.ocr_page_count == 0
    page2 = doc.blocks[1].text
    assert "18 working days of paid annual leave" in page2
    assert "Annual Leave" in page2


def test_docx_keeps_heading_path_and_tables(sample_data: Path):
    doc = DocxExtractor().extract(sample_data / "legal" / "service_agreement_template.docx")
    sections = [b.section for b in doc.blocks]
    assert any(
        s and s.endswith("4. Termination > 4.1 Termination for Convenience") for s in sections
    )
    fees = next(b for b in doc.blocks if b.section and "3. Fees and Payment" in b.section)
    # Table rows are flattened into "Header: value" pairs.
    assert "Service tier: Enterprise" in fees.text
    assert "Monthly fee: IDR 150,000,000" in fees.text
    assert all(b.page_number is None for b in doc.blocks)


def test_corrupted_pdf_raises_clear_error(sample_data: Path):
    with pytest.raises(CorruptedFileError, match="could not be read as a PDF"):
        PdfExtractor().extract(sample_data / "invalid" / "corrupted.pdf")


def test_corrupted_docx_raises_clear_error(sample_data: Path):
    with pytest.raises(CorruptedFileError, match="Word"):
        DocxExtractor().extract(sample_data / "invalid" / "corrupted.docx")


def test_scanned_pdf_without_ocr_explains_how_to_fix(sample_data: Path):
    with pytest.raises(ScannedPdfError) as exc:
        PdfExtractor(ocr_enabled=False).extract(sample_data / "legal" / "scanned_nda.pdf")
    assert "scanned" in exc.value.message
    assert "OCR_ENABLED" in exc.value.message


def test_scanned_pdf_uses_ocr_when_available(sample_data: Path, monkeypatch):
    monkeypatch.setattr(pdf_module, "tesseract_available", lambda: True)
    monkeypatch.setattr(
        pdf_module,
        "ocr_image",
        lambda image, lang: "MUTUAL NON-DISCLOSURE AGREEMENT\nDuration: 3 years",
    )
    doc = PdfExtractor(ocr_enabled=True).extract(sample_data / "legal" / "scanned_nda.pdf")
    assert doc.ocr_page_count == 1
    assert doc.blocks[0].ocr is True
    assert "3 years" in doc.blocks[0].text


def test_blank_pdf_with_failed_ocr_yields_no_blocks(sample_data: Path, monkeypatch):
    monkeypatch.setattr(pdf_module, "tesseract_available", lambda: True)
    monkeypatch.setattr(pdf_module, "ocr_image", lambda image, lang: "")
    doc = PdfExtractor(ocr_enabled=True).extract(sample_data / "invalid" / "blank.pdf")
    assert doc.blocks == []  # the pipeline turns this into NoExtractableTextError


def test_normalise_text_joins_hyphenation_and_collapses_whitespace():
    raw = "Employ-\nees   get\t18 days.\r\n\n\n\nNew  paragraph"
    assert normalise_text(raw) == "Employees get 18 days.\n\nNew paragraph"


def test_parallel_pdf_extraction_is_thread_safe(sample_data: Path):
    """Regression: PDFium is not thread-safe; the worker extracts in threads.

    Without the module-level lock this crashed the process (SIGSEGV) as soon as
    two PDFs were extracted at the same time.
    """
    from concurrent.futures import ThreadPoolExecutor

    path = sample_data / "hr" / "employee_handbook.pdf"
    extractor = PdfExtractor(ocr_enabled=False)
    expected = [b.text for b in extractor.extract(path).blocks]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: [b.text for b in extractor.extract(path).blocks], range(80))
        )
    assert all(r == expected for r in results)
