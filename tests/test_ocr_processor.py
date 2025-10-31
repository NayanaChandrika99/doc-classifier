"""Tests for OCR processing."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image

from config import Settings
from src.tennr_classifier.ocr_processor import OCRProcessingError, OCRProcessor
from src.tennr_classifier.pipeline import OCRWord, PageData


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        temp_dir=tmp_path / "tmp",
        log_level="DEBUG",
        ocr_warn_threshold=0.95,
    )
    settings.ensure_directories()
    return settings


def _make_image(tmp_path: Path) -> PageData:
    image_path = tmp_path / "page.png"
    image = Image.new("RGB", (32, 32), color="white")
    image.save(image_path)
    return PageData(index=0, image_path=image_path, width=32, height=32)


def test_ocr_processor_uses_tesseract(monkeypatch, tmp_path, caplog):
    page = _make_image(tmp_path)
    settings = _settings(tmp_path)

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda image: "Patient John Doe",
    )

    def fake_image_to_data(image, output_type):
        return {
            "text": ["Patient", "John", "Doe"],
            "left": [0, 10, 20],
            "top": [0, 0, 0],
            "width": [9, 8, 8],
            "height": [10, 10, 10],
            "conf": ["96", "92", "89"],
        }

    monkeypatch.setattr("pytesseract.image_to_data", fake_image_to_data)

    processor = OCRProcessor(settings)

    with caplog.at_level(logging.WARNING):
        result = processor.process_page(page)

    assert "Patient John Doe" in result.text
    assert len(result.words) == 3
    assert result.average_confidence is not None
    assert any(word.text == "John" for word in result.words)
    # Confidence is ~0.92, so threshold 0.95 should trigger warning
    assert any("Low OCR confidence" in record.message for record in caplog.records)


def test_ocr_processor_requires_custom_callable_for_olmocr(tmp_path):
    page = _make_image(tmp_path)
    settings = _settings(tmp_path)
    settings.ocr_backend = "olmocr"
    processor = OCRProcessor(settings)

    with pytest.raises(OCRProcessingError):
        processor.process_page(page)


def test_ocr_processor_custom_callable(tmp_path):
    page = _make_image(tmp_path)
    settings = _settings(tmp_path)
    settings.ocr_backend = "custom"

    def custom_callable(image):
        words = [
            OCRWord(text="Custom", bbox=(0, 0, 10, 10), confidence=0.8),
        ]
        return "Custom", words

    processor = OCRProcessor(settings, custom_callable=custom_callable)
    result = processor.process_page(page)

    assert result.text == "Custom"
    assert len(result.words) == 1
    assert result.words[0].confidence == 0.8
