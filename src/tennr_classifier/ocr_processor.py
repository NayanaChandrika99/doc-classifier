"""OCR processing utilities for the Tennr classifier pipeline."""

from __future__ import annotations

import importlib
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import pytesseract
from pytesseract import Output

from config import Settings
from .logging_utils import get_logger
from .pipeline import OCRResult, OCRWord, PageData

logger = get_logger(__name__)

class OCRProcessingError(RuntimeError):
    """Raised when OCR fails for a given page."""


class OCRProcessor:
    """Run OCR on rendered page images using configurable backends."""

    def __init__(
        self,
        settings: Settings,
        backend: Optional[str] = None,
        custom_callable: Optional[Callable[[Image.Image], Tuple[str, List[OCRWord]]]] = None,
    ):
        self.settings = settings
        self.backend = backend or settings.ocr_backend
        self.custom_callable = custom_callable or self._load_custom_callable()

    def process_pages(self, pages: Sequence[PageData]) -> List[OCRResult]:
        """Process multiple pages sequentially."""
        results: List[OCRResult] = []
        for page in pages:
            results.append(self.process_page(page))
        return results

    def process_page(self, page: PageData) -> OCRResult:
        """Run OCR on a single page image."""
        if not page.image_path.exists():
            raise OCRProcessingError(f"Image for page {page.index} not found: {page.image_path}")

        image = Image.open(page.image_path)
        try:
            if self.custom_callable:
                text, words_iter = self.custom_callable(image)
            elif self.backend == "tesseract":
                text, words_iter = self._run_tesseract(image)
            elif self.backend == "olmocr":
                raise OCRProcessingError(
                    "olmOCR backend requires TENNR_OLMOCR_HANDLER or a custom callable passed to OCRProcessor."
                )
            else:
                raise OCRProcessingError(f"Unsupported OCR backend '{self.backend}'.")
        finally:
            image.close()

        words = list(words_iter)
        avg_conf = self._calculate_average_confidence(words)
        if avg_conf is not None and avg_conf < self.settings.ocr_warn_threshold:
            logger.warning(
                "Low OCR confidence for page %s (avg=%.2f). Adjust DPI or review the source PDF.",
                page.index,
                avg_conf,
            )

        return OCRResult(page_index=page.index, text=text, words=words, average_confidence=avg_conf)

    def _run_tesseract(self, image: Image.Image) -> Tuple[str, List[OCRWord]]:
        try:
            text = pytesseract.image_to_string(image)
            data = pytesseract.image_to_data(image, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRProcessingError(
                "Tesseract binary not found. Install via 'brew install tesseract' or configure a custom OCR backend."
            ) from exc

        words: List[OCRWord] = []
        for word, left, top, width, height, confidence in zip(
            data["text"],
            data["left"],
            data["top"],
            data["width"],
            data["height"],
            data["conf"],
        ):
            if not word.strip():
                continue
            try:
                conf_value = float(confidence)
            except ValueError:
                conf_value = -1.0
            if conf_value < 0:
                continue
            normalized_conf = min(max(conf_value / 100.0, 0.0), 1.0)
            words.append(
                OCRWord(
                    text=word,
                    bbox=(int(left), int(top), int(width), int(height)),
                    confidence=normalized_conf,
                )
            )
        return text, words

    def _calculate_average_confidence(self, words: Iterable[OCRWord]) -> Optional[float]:
        confidences = [word.confidence for word in words if word.confidence is not None]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def _load_custom_callable(self) -> Optional[Callable[[Image.Image], Tuple[str, List[OCRWord]]]]:
        backend_path = self.settings.ocr_backend
        if backend_path == "tesseract":
            return None
        if backend_path == "olmocr":
            handler_path = self.settings.olmocr_handler
            if not handler_path:
                logger.warning(
                    "olmOCR backend selected but TENNR_OLMOCR_HANDLER is not set. Provide a custom callable path."
                )
                return None
            backend_path = handler_path
        try:
            module_name, func_name = backend_path.rsplit(":", 1)
        except ValueError:
            raise OCRProcessingError(
                "Custom OCR backend must be specified as 'module:function'."
            ) from None

        module = importlib.import_module(module_name)
        callable_obj = getattr(module, func_name, None)
        if callable_obj is None:
            raise OCRProcessingError(f"Function '{func_name}' not found in module '{module_name}'.")
        return callable_obj
