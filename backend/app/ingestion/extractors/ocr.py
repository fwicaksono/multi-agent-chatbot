"""OCR for image-only PDF pages, via Tesseract."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageOps

from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # TesseractNotFoundError, OSError, ImportError
        logger.warning("ocr.tesseract_unavailable")
        return False


def ocr_image(image: Image.Image, languages: str) -> str:
    import pytesseract

    # Greyscale + autocontrast noticeably helps on low-contrast scans.
    prepared = ImageOps.autocontrast(image.convert("L"))
    # psm 3 = fully automatic page segmentation (default for full pages).
    return pytesseract.image_to_string(prepared, lang=languages, config="--psm 3")
