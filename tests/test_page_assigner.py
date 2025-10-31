"""Tests for page assignment logic."""

from __future__ import annotations

from config import Settings
from src.tennr_classifier.page_assigner import PageAssigner
from src.tennr_classifier.pipeline import (
    AssignmentReason,
    IdentifierMatch,
    LinkedIdentifier,
    LinkedPatient,
    PageEntities,
)


def _identifier(kind: str, value: str, confidence: float = 0.9) -> IdentifierMatch:
    return IdentifierMatch(kind=kind, value=value, confidence=confidence, bbox=(0, 0, 1, 1), word_indices=[])


def _linked_patient(patient_id: str, **identifiers: str) -> LinkedPatient:
    linked = [
        LinkedIdentifier(kind=kind, value=value, confidence=0.9, sources=[(0, [])])
        for kind, value in identifiers.items()
    ]
    return LinkedPatient(patient_id=patient_id, identifiers=linked, pages=[0], score=0.9)


def _settings() -> Settings:
    settings = Settings()
    settings.assign_min_confidence = 0.4
    settings.assign_ambiguity_margin = 0.05
    settings.assign_allow_unassigned = True
    settings.assign_name_weight = 0.3
    settings.assign_mrn_weight = 0.5
    settings.assign_dob_weight = 0.2
    settings.assign_phone_weight = 0.0
    return settings


def test_assigner_assigns_clear_mrn_match():
    settings = _settings()
    assigner = PageAssigner(settings)

    page = PageEntities(page_index=0, identifiers=[_identifier("mrn", "12345"), _identifier("name", "John Doe")])
    patient = _linked_patient("patient_001", mrn="12345", name="John Doe")

    summary = assigner.assign_pages([page], [patient])

    assert summary.assignments[0].patient_id == "patient_001"
    assert summary.assignments[0].manual_review is False
    assert summary.unassigned_pages == []


def test_assigner_marks_ambiguous_when_scores_close():
    settings = _settings()
    settings.assign_ambiguity_margin = 0.1
    assigner = PageAssigner(settings)

    page = PageEntities(page_index=0, identifiers=[_identifier("name", "Alice"), _identifier("dob", "01/01/1980")])
    patient_a = _linked_patient("patient_001", name="Alice", dob="01/01/1980")
    patient_b = _linked_patient("patient_002", name="Alice", dob="01/01/1980")

    summary = assigner.assign_pages([page], [patient_a, patient_b])

    assignment = summary.assignments[0]
    assert assignment.manual_review is True
    assert assignment.patient_id == "patient_001" or assignment.patient_id == "patient_002"
    assert summary.ambiguous_pages == [0]


def test_assigner_handles_unassigned_page():
    settings = _settings()
    assigner = PageAssigner(settings)

    page = PageEntities(page_index=0, identifiers=[_identifier("name", "Unknown")])
    patient = _linked_patient("patient_001", name="John Doe")

    summary = assigner.assign_pages([page], [patient])

    assert summary.assignments[0].patient_id is None
    assert summary.assignments[0].manual_review is True
    assert summary.unassigned_pages == [0]


def test_assigner_respects_weights():
    settings = _settings()
    settings.assign_name_weight = 0.8
    settings.assign_mrn_weight = 0.1
    settings.assign_dob_weight = 0.1
    assigner = PageAssigner(settings)

    page = PageEntities(page_index=0, identifiers=[_identifier("name", "Bob Brown")])
    patient_a = _linked_patient("patient_001", name="Bob Brown")
    patient_b = _linked_patient("patient_002", name="Rob Brown")

    summary = assigner.assign_pages([page], [patient_a, patient_b])
    assert summary.assignments[0].patient_id == "patient_001"
