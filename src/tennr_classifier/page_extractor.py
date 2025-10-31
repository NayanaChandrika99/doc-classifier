"""PDF page extraction utilities for the Tennr classifier pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path

from config import Settings
from .logging_utils import get_logger
from .pipeline import PageData

logger = get_logger(__name__)


class PageExtractionError(RuntimeError):
    """Raised when the PDF page extraction process fails."""


class PageExtractor:
    """Render PDF pages to images and provide metadata for downstream OCR."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def extract_pages(self, pdf_path: Path, *, persist_images: Optional[bool] = None) -> List[PageData]:
        """
        Render the PDF and return metadata for each page.

        Args:
            pdf_path: Location of the multi-page PDF.
            persist_images: Override whether rendered images remain on disk.

        Returns:
            List of PageData entries ordered by page index.
        """
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise PageExtractionError(f"PDF not found: {pdf_path}")

        persist = self.settings.persist_page_images if persist_images is None else persist_images
        output_dir = self.settings.page_image_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            pages = self._render_with_pymupdf(pdf_path, output_dir, persist)
            logger.info("Rendered %s pages using PyMuPDF", len(pages))
            return pages
        except Exception as primary_error:  # pylint: disable=broad-except
            logger.warning("PyMuPDF rendering failed (%s). Falling back to pdf2image.", primary_error)
            try:
                pages = self._render_with_pdf2image(pdf_path, output_dir, persist)
                logger.info("Rendered %s pages using pdf2image fallback", len(pages))
                return pages
            except Exception as fallback_error:  # pylint: disable=broad-except
                raise PageExtractionError(
                    f"Unable to render PDF with PyMuPDF or pdf2image: {fallback_error}"
                ) from fallback_error

    def _render_with_pymupdf(
        self, pdf_path: Path, output_dir: Path, persist: bool
    ) -> List[PageData]:
        pages: List[PageData] = []
        zoom = self.settings.pdf_render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(pdf_path) as document:
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix)
                mode = "RGBA" if pixmap.alpha else "RGB"
                image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
                if pixmap.alpha:
                    image = image.convert("RGB")
                image_path = self._persist_image(image, output_dir, page_index, persist)
                pages.append(PageData(page_index, image_path, image.width, image.height, persisted=persist))
        return pages

    def _render_with_pdf2image(
        self, pdf_path: Path, output_dir: Path, persist: bool
    ) -> List[PageData]:
        images = convert_from_path(pdf_path, dpi=self.settings.pdf_render_dpi)
        pages: List[PageData] = []
        for page_index, image in enumerate(images):
            image_path = self._persist_image(image, output_dir, page_index, persist)
            pages.append(PageData(page_index, image_path, image.width, image.height, persisted=persist))
        return pages

    def _persist_image(self, image: Image.Image, output_dir: Path, page_index: int, persist: bool) -> Path:
        if persist:
            filename = output_dir / f"page_{page_index:04d}.png"
            image.save(filename, format="PNG")
            return filename

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            dir=output_dir,
            delete=False,
        ) as temp_file:
            image.save(temp_file, format="PNG")
            return Path(temp_file.name)
