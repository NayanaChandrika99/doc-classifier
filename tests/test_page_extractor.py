"""Tests for PDF page extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from config import Settings
from src.tennr_classifier.page_extractor import PageData, PageExtractionError, PageExtractor


def _build_sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=letter)
    pdf_canvas.drawString(72, 720, "Patient: John Doe")
    pdf_canvas.showPage()
    pdf_canvas.drawString(72, 720, "Patient: Jane Doe")
    pdf_canvas.save()
    return pdf_path


def _settings(tmp_path: Path, persist: bool = True) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        temp_dir=tmp_path / "tmp",
        persist_page_images=persist,
        pdf_render_dpi=72,
    )
    settings.ensure_directories()
    return settings


def test_extract_pages_with_pymupdf(tmp_path):
    pdf_path = _build_sample_pdf(tmp_path)
    settings = _settings(tmp_path)
    extractor = PageExtractor(settings)

    pages = extractor.extract_pages(pdf_path)

    assert len(pages) == 2
    for idx, page in enumerate(pages):
        assert page.index == idx
        assert page.image_path.exists()
        assert page.width > 0 and page.height > 0
        assert page.persisted is True


def test_extract_pages_falls_back_to_pdf2image(tmp_path, monkeypatch):
    pdf_path = _build_sample_pdf(tmp_path)
    settings = _settings(tmp_path, persist=False)
    extractor = PageExtractor(settings)

    def fail(*args, **kwargs):  # pragma: no cover - failure path
        raise RuntimeError("boom")

    called = {}

    def fake_fallback(pdf_path: Path, output_dir: Path, persist: bool):
        called["used"] = True
        return [PageData(index=0, image_path=output_dir / "page_0000.png", width=100, height=100, persisted=persist)]

    monkeypatch.setattr(extractor, "_render_with_pymupdf", fail)
    monkeypatch.setattr(extractor, "_render_with_pdf2image", fake_fallback)

    pages = extractor.extract_pages(pdf_path)

    assert called["used"] is True
    assert pages[0].persisted is False


def test_extract_pages_missing_pdf(tmp_path):
    settings = _settings(tmp_path)
    extractor = PageExtractor(settings)
    with pytest.raises(PageExtractionError):
        extractor.extract_pages(tmp_path / "missing.pdf")
