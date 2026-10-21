"""PDF text extraction with per-page OCR fallback.

Text is extracted page by page with PDFium (pypdfium2, permissive licence).
Pages that yield (almost) no text are treated as scanned images: they are
rendered to a bitmap and passed through Tesseract OCR when it is enabled.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from app.core.logging import get_logger
from app.ingestion.errors import CorruptedFileError, EncryptedFileError, ScannedPdfError
from app.ingestion.extractors.base import ExtractedDocument, TextBlock, normalise_text
from app.ingestion.extractors.ocr import ocr_image, tesseract_available

logger = get_logger(__name__)


class PdfExtractor:
    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_languages: str = "eng",
        ocr_dpi: int = 300,
        ocr_min_chars: int = 25,
    ) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages
        self.ocr_dpi = ocr_dpi
        self.ocr_min_chars = ocr_min_chars

    def extract(self, path: Path) -> ExtractedDocument:
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
            pdf.close()

    def _extract_pages(self, pdf: pdfium.PdfDocument) -> ExtractedDocument:
        result = ExtractedDocument(page_count=len(pdf))
        image_only_pages: list[int] = []
        use_ocr = self.ocr_enabled and tesseract_available()

        for index in range(len(pdf)):
            page_number = index + 1
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                text = normalise_text(textpage.get_text_range())
                textpage.close()

                if len(text) < self.ocr_min_chars:
                    if use_ocr:
                        ocr_text = self._ocr_page(page, page_number)
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            result.ocr_page_count += 1
                            result.blocks.append(TextBlock(text, page_number=page_number, ocr=True))
                            continue
                    else:
                        image_only_pages.append(page_number)
            finally:
                page.close()

            if text:
                result.blocks.append(TextBlock(text, page_number=page_number))

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

    def _ocr_page(self, page: pdfium.PdfPage, page_number: int) -> str:
        bitmap = page.render(scale=self.ocr_dpi / 72)
        try:
            image = bitmap.to_pil()
            text = normalise_text(ocr_image(image, self.ocr_languages))
        except Exception as exc:  # OCR must never crash ingestion of other pages
            logger.warning(
                "ocr.page_failed", extra={"page": page_number, "error": type(exc).__name__}
            )
            return ""
        finally:
            bitmap.close()
        logger.info("ocr.page_done", extra={"page": page_number, "chars": len(text)})
        return text
