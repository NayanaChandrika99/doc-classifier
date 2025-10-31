"""Assign OCR pages to linked patient clusters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from config import Settings
from .fuzzy_matcher import FuzzyMatcher
from .logging_utils import get_logger
from .pipeline import (
    AssignmentReason,
    DocumentAssignmentSummary,
    LinkedIdentifier,
    LinkedPatient,
    PageAssignment,
    PageEntities,
)

logger = get_logger(__name__)


@dataclass
class AssignmentScore:
    patient_id: str
    score: float
    reasons: List[AssignmentReason]


class PageAssigner:
    """Calculate ownership of pages based on linked patient identifiers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.matcher = FuzzyMatcher(settings)

    def assign_pages(
        self,
        pages: Sequence[PageEntities],
        patients: Sequence[LinkedPatient],
    ) -> DocumentAssignmentSummary:
        assignments: List[PageAssignment] = []
        unassigned: List[int] = []
        ambiguous: List[int] = []

        for page in pages:
            result = self._assign_single_page(page, patients)
            assignments.append(result)
            if result.patient_id is None:
                unassigned.append(page.page_index)
            elif result.manual_review:
                ambiguous.append(page.page_index)

        summary = DocumentAssignmentSummary(
            assignments=assignments,
            unassigned_pages=unassigned,
            ambiguous_pages=ambiguous,
        )
        logger.info(
            "Page assignment complete: %s assigned, %s unassigned, %s ambiguous",
            len(assignments) - len(unassigned),
            len(unassigned),
            len(ambiguous),
        )
        return summary

    def _assign_single_page(
        self,
        page: PageEntities,
        patients: Sequence[LinkedPatient],
    ) -> PageAssignment:
        scores = [self._score_page(page, patient) for patient in patients]
        scores = [score for score in scores if score.score > 0]
        if not scores:
            return PageAssignment(
                page_index=page.page_index,
                patient_id=None,
                confidence=0.0,
                reasons=[],
                manual_review=True,
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        top = scores[0]
        confidence = top.score
        manual_review = False

        if len(scores) > 1:
            margin = self.settings.assign_ambiguity_margin
            if top.score - scores[1].score <= margin:
                manual_review = True

        if confidence < self.settings.assign_min_confidence:
            return PageAssignment(
                page_index=page.page_index,
                patient_id=None if self.settings.assign_allow_unassigned else top.patient_id,
                confidence=confidence,
                reasons=top.reasons if not self.settings.assign_allow_unassigned else [],
                manual_review=True,
            )

        return PageAssignment(
            page_index=page.page_index,
            patient_id=top.patient_id,
            confidence=confidence,
            reasons=top.reasons,
            manual_review=manual_review,
        )

    def _score_page(self, page: PageEntities, patient: LinkedPatient) -> AssignmentScore:
        reasons: List[AssignmentReason] = []
        score = 0.0
        weights = self._weights()

        patient_lookup: Dict[str, List[LinkedIdentifier]] = {}
        for identifier in patient.identifiers:
            patient_lookup.setdefault(identifier.kind, []).append(identifier)

        for identifier in page.identifiers:
            matches = patient_lookup.get(identifier.kind, [])
            best = 0.0
            best_value = None
            for candidate in matches:
                current = self._compare(identifier.kind, identifier.value, candidate.value)
                if current > best:
                    best = current
                    best_value = candidate.value
            if best > 0:
                weight = weights.get(identifier.kind, 0.0)
                contribution = best * weight
                score += contribution
                reasons.append(AssignmentReason(kind=identifier.kind, value=best_value or identifier.value, score=round(contribution, 3)))

        score = min(score, 1.0)
        return AssignmentScore(patient_id=patient.patient_id, score=round(score, 3), reasons=reasons)

    def _weights(self) -> Dict[str, float]:
        weights = {
            "mrn": self.settings.assign_mrn_weight,
            "dob": self.settings.assign_dob_weight,
            "phone": self.settings.assign_phone_weight,
            "name": self.settings.assign_name_weight,
        }
        total = sum(weights.values())
        if total == 0:
            return weights
        return {key: value / total for key, value in weights.items()}

    def _compare(self, kind: str, a: str, b: str) -> float:
        if kind == "mrn":
            return self.matcher.score_mrn(a, b)
        if kind == "dob":
            return self.matcher.score_dob(a, b)
        if kind == "phone":
            return self.matcher.score_phone(a, b)
        if kind == "name":
            return self.matcher.score_name(a, b)
        return 0.0
