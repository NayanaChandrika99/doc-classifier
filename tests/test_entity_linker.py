"""Tests for patient entity linker clustering."""

from __future__ import annotations

from config import Settings
from src.tennr_classifier.entity_linker import PatientEntityLinker
from src.tennr_classifier.pipeline import IdentifierMatch, PageEntities


def _identifier(kind: str, value: str, confidence: float = 0.9, indices=None) -> IdentifierMatch:
    return IdentifierMatch(kind=kind, value=value, confidence=confidence, bbox=(0, 0, 1, 1), word_indices=indices or [])


def _page(page_index: int, *identifiers: IdentifierMatch) -> PageEntities:
    return PageEntities(page_index=page_index, identifiers=list(identifiers))


def _linker() -> PatientEntityLinker:
    settings = Settings()
    settings.name_match_threshold = 0.8
    settings.mrn_match_threshold = 0.95
    settings.dob_match_threshold = 1.0
    settings.phone_match_threshold = 0.95
    settings.linker_strict_mode = True
    return PatientEntityLinker(settings)


def test_linker_merges_pages_with_same_mrn():
    linker = _linker()
    pages = [
        _page(0, _identifier("mrn", "12345"), _identifier("name", "John Doe")),
        _page(1, _identifier("mrn", "12345"), _identifier("dob", "01/02/1990")),
    ]

    linked = linker.link(pages)

    assert len(linked) == 1
    patient = linked[0]
    assert patient.patient_id == "patient_001"
    assert set(patient.pages) == {0, 1}
    assert any(li.kind == "mrn" and li.value.endswith("12345") for li in patient.identifiers)


def test_linker_separates_different_mrn():
    linker = _linker()
    pages = [
        _page(0, _identifier("mrn", "12345"), _identifier("name", "John Doe")),
        _page(1, _identifier("mrn", "67890"), _identifier("name", "John Doe")),
    ]

    linked = linker.link(pages)

    assert len(linked) == 2
    assert linked[0].patient_id != linked[1].patient_id


def test_linker_uses_name_and_dob_when_mrn_missing():
    linker = _linker()
    pages = [
        _page(0, _identifier("name", "Alice Smith"), _identifier("dob", "03/04/1985")),
        _page(1, _identifier("name", "Alicia Smith"), _identifier("dob", "03/04/1985")),
    ]

    linked = linker.link(pages)
    assert len(linked) == 1


def test_linker_handles_conflicting_identifiers():
    linker = _linker()
    pages = [
        _page(0, _identifier("name", "Bob Brown"), _identifier("dob", "02/02/1990")),
        _page(1, _identifier("name", "Bob Brown"), _identifier("dob", "01/01/1970")),
    ]

    linked = linker.link(pages)
    assert len(linked) == 2


def test_linker_multiple_mrns_on_same_page_create_separate_clusters():
    linker = _linker()
    pages = [
        _page(0, _identifier("mrn", "12345"), _identifier("mrn", "67890")),
    ]

    linked = linker.link(pages)
    assert len(linked) == 2
