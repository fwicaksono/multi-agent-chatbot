"""PDF text extraction with per-page OCR fallback.

Text is extracted page by page with PDFium (pypdfium2, permissive licence).
Pages that yield (almost) no text are treated as scanned images: they are
rendered to a bitmap and passed through Tesseract OCR when it is enabled.

Thread safety: PDFium is not thread-safe -- calling it from two threads at
once (even on different documents) can crash the process. The worker runs
extractions in threads, so every PDFium call goes through `_PDFIUM_LOCK`.
The slow part, OCR, runs outside the lock on a copied image, so concurrent
jobs still overlap.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from app.core.logging import get_logger
from app.ingestion.errors import CorruptedFileError, EncryptedFileError, ScannedPdfError
from app.ingestion.extractors.base import ExtractedDocument, TextBlock, normalise_text
from app.ingestion.extractors.ocr import ocr_image, tesseract_available

logger = get_logger(__name__)

_PDFIUM_LOCK = threading.Lock()


class PdfExtractor:
    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_languages: str = "eng",
        ocr_dpi: int = 300,
        ocr_min_chars: int = 25,
        ocr_max_pages: int = 300,
    ) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.ocr_min_chars = ocr_min_chars
        self.ocr_max_pages = ocr_max_pages

    def extract(self, path: Path) -> ExtractedDocument:
        with _PDFIUM_LOCK:
            try:
                pdf = pdfium.PdfDocument(str(path))
            except pdfium.PdfiumError as exc:
                if "password" in str(exc).lower():
                    raise EncryptedFileError(
                        "This PDF is password-protected. Remove the password and upload it again."
                    ) from exc
                raise CorruptedFileError(
                    "The file could not be read as a PDF. It may be corrupted, truncated, "
                    "or not actually a PDF. Re-export or re-download it and try again."
                ) from exc

        try:
            return self._extract_pages(pdf)
        finally:
            with _PDFIUM_LOCK:
                pdf.close()

    def _extract_pages(self, pdf: pdfium.PdfDocument) -> ExtractedDocument:
        with _PDFIUM_LOCK:
            page_count = len(pdf)
        result = ExtractedDocument(page_count=page_count)
        image_only_pages: list[int] = []
        use_ocr = self.ocr_enabled and tesseract_available()
        ocr_attempts = ocr_skipped = 0

        for index in range(page_count):
            page_number = index + 1
            can_ocr = use_ocr and ocr_attempts < self.ocr_max_pages
            text, image = self._read_page(pdf, index, render=can_ocr)

            if len(text) < self.ocr_min_chars:
                if use_ocr and not can_ocr:
                    ocr_skipped += 1
                elif use_ocr and image is not None:
                    ocr_attempts += 1
                    ocr_text = self._ocr(image, page_number)
                    if len(ocr_text) > len(text):
                        result.ocr_page_count += 1
                        result.blocks.append(TextBlock(ocr_text, page_number=page_number, ocr=True))
                        continue
                elif not use_ocr:
                    image_only_pages.append(page_number)

            if text:
                result.blocks.append(TextBlock(text, page_number=page_number))

        if ocr_skipped:
            result.notes.append(
                f"OCR read the first {self.ocr_max_pages} scanned pages; {ocr_skipped} more "
                "scanned pages were not read (OCR_MAX_PAGES). Split the PDF to index them."
            )
            logger.warning("ocr.page_cap_reached", extra={"skipped_pages": ocr_skipped})
        if not result.blocks and image_only_pages:
            reason = (
                "OCR is disabled (set OCR_ENABLED=true)"
                if not self.ocr_enabled
                else "the OCR engine (Tesseract) is not installed on the server"
            )
            raise ScannedPdfError(
                "This PDF appears to be a scanned document: its pages are images with no "
                f"selectable text, and {reason}. Enable OCR or upload a text-based PDF."
            )
        return result

    def _read_page(
        self, pdf: pdfium.PdfDocument, index: int, render: bool
    ) -> tuple[str, Image.Image | None]:
        """Extract a page's text and, if it has (almost) none, a greyscale render.

        Holds the PDFium lock only for the PDFium calls. The rendered image is
        converted (which copies it), so it no longer references PDFium memory.
        """
        with _PDFIUM_LOCK:
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = normalise_text(textpage.get_text_range())
                finally:
                    textpage.close()
                image = None
                if render and len(text) < self.ocr_min_chars:
                    bitmap = page.render(scale=self.ocr_dpi / 72)
                    try:
                        image = bitmap.to_pil().convert("L")
                    finally:
                        bitmap.close()
            finally:
                page.close()
        return text, image

    def _ocr(self, image: Image.Image, page_number: int) -> str:
        try:
            text = normalise_text(ocr_image(image, self.ocr_languages))
        except Exception as exc:  # OCR must never crash ingestion of other pages
            logger.warning(
                "ocr.page_failed", extra={"page": page_number, "error": type(exc).__name__}
            )
            return ""
        logger.info("ocr.page_done", extra={"page": page_number, "chars": len(text)})
        return text
