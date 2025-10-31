"""Tests for regex-based identifier extraction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from config import Settings
from src.tennr_classifier.entity_extractor import EntityExtractor
from src.tennr_classifier.pipeline import IdentifierMatch, OCRResult, OCRWord


def _make_word(text: str, x: int) -> OCRWord:
    return OCRWord(text=text, bbox=(x, 0, 10, 10), confidence=0.9)


def _settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        temp_dir=tmp_path / "tmp",
    )
    settings.ensure_directories()
    return settings


def test_extract_page_returns_identifiers(tmp_path):
    settings = _settings(tmp_path)
    extractor = EntityExtractor(settings)

    text = "Patient: John Doe\nMRN: 123456\nDOB: 01/15/1990"
    words = [
        _make_word("Patient", 0),
        _make_word("John", 10),
        _make_word("Doe", 20),
        _make_word("MRN", 30),
        _make_word("123456", 40),
        _make_word("DOB", 50),
        _make_word("01/15/1990", 60),
    ]
    ocr = OCRResult(page_index=0, text=text, words=words)

    entities = extractor.extract_page(ocr)
    values = {match.kind: match.value for match in entities.identifiers}

    assert values["name"].endswith("John Doe")
    assert values["mrn"].endswith("123456")
    assert values["dob"].endswith("01/15/1990")


def test_extract_page_filters_low_confidence(tmp_path):
    settings = _settings(tmp_path)
    settings.entity_min_confidence = 0.9
    extractor = EntityExtractor(settings)

    text = "MRN: 12"
    words = [_make_word("MRN", 0), _make_word("12", 10)]
    ocr = OCRResult(page_index=0, text=text, words=words)

    entities = extractor.extract_page(ocr)
    assert not entities.identifiers


def test_extract_page_uses_regex_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TENNR_REGEX_NAME", "(?P<name>Jane Doe)")
    monkeypatch.setenv("TENNR_ENTITY_MIN_CONFIDENCE", "0.1")
    settings = _settings(tmp_path)
    extractor = EntityExtractor(settings)

    text = "Name: Jane Doe"
    words = [_make_word("Jane", 0), _make_word("Doe", 10)]
    ocr = OCRResult(page_index=0, text=text, words=words)

    entities = extractor.extract_page(ocr)
    assert any(match.value == "Jane Doe" for match in entities.identifiers)


def test_extract_document_handles_multiple_pages(tmp_path):
    settings = _settings(tmp_path)
    extractor = EntityExtractor(settings)

    result1 = OCRResult(
        page_index=0,
        text="Patient: Alice Smith\nMRN: ABC123",
        words=[_make_word("Alice", 0), _make_word("Smith", 10), _make_word("ABC123", 20)],
    )
    result2 = OCRResult(
        page_index=1,
        text="DOB: 03/04/1985",
        words=[_make_word("DOB", 0), _make_word("03/04/1985", 10)],
    )

    pages = extractor.extract_document([result1, result2])
    assert len(pages) == 2
    assert any(match.kind == "mrn" for match in pages[0].identifiers)
    assert any(match.kind == "dob" for match in pages[1].identifiers)
