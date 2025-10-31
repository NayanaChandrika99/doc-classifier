"""Entity linking to cluster page-level identifiers into patient profiles."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from config import Settings
from .fuzzy_matcher import FuzzyMatcher
from .logging_utils import get_logger
from .pipeline import IdentifierMatch, LinkedIdentifier, LinkedPatient, PageEntities

logger = get_logger(__name__)


@dataclass
class ClusterCandidate:
    patient_id: str
    identifiers: Dict[str, LinkedIdentifier]
    pages: List[int]
    score: float


class PatientEntityLinker:
    """Link identifiers across pages into patient-level clusters."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.matcher = FuzzyMatcher(settings)
        self._next_patient = itertools.count(1)

    def link(self, pages: Sequence[PageEntities]) -> List[LinkedPatient]:
        clusters: List[ClusterCandidate] = []
        anchor_kinds = {"mrn"}
        supportive_kinds = {"name", "dob", "phone"}

        for page in pages:
            indexed_identifiers = list(enumerate(page.identifiers))
            anchor_positions: List[Tuple[int, ClusterCandidate]] = []

            for idx, identifier in indexed_identifiers:
                preferred: Optional[ClusterCandidate] = None
                force_attach = False

                if anchor_positions and identifier.kind in supportive_kinds:
                    preferred = self._nearest_anchor(idx, anchor_positions)
                    force_attach = True

                cluster = self._assign_identifier(
                    clusters,
                    identifier,
                    page.page_index,
                    preferred=preferred,
                    force_attach=force_attach,
                )

                if identifier.kind in anchor_kinds:
                    anchor_positions.append((idx, cluster))
                elif identifier.kind in supportive_kinds:
                    # If we created a new cluster (no preferred anchor matched), track it for later identifiers.
                    if preferred is None and cluster not in [c for _, c in anchor_positions]:
                        anchor_positions.append((idx, cluster))

        return [self._to_linked_patient(cluster) for cluster in clusters]

    def _assign_identifier(
        self,
        clusters: List[ClusterCandidate],
        identifier: IdentifierMatch,
        page_index: int,
        *,
        preferred: ClusterCandidate | None = None,
        force_attach: bool = False,
    ) -> ClusterCandidate:
        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = self._match_identifier(cluster, identifier)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        threshold = self._threshold_for(identifier.kind)
        if best_cluster and best_score >= threshold:
            logger.debug(
                "Merging %s '%s' into patient %s (score=%.2f)",
                identifier.kind,
                identifier.value,
                best_cluster.patient_id,
                best_score,
            )
            self._merge_identifier(best_cluster, identifier, page_index, best_score)
            return best_cluster

        if preferred and force_attach:
            logger.debug(
                "Attaching %s '%s' to preferred cluster %s from same page",
                identifier.kind,
                identifier.value,
                preferred.patient_id,
            )
            self._merge_identifier(preferred, identifier, page_index, best_score)
            return preferred

        if (
            identifier.kind == "name"
            and best_cluster
            and best_score >= self.settings.name_match_threshold
            and self.settings.linker_strict_mode
        ):
            logger.debug(
                "Name-only match below threshold ignored in strict mode for %s '%s' (score=%.2f)",
                identifier.kind,
                identifier.value,
                best_score,
            )
            return best_cluster if best_cluster else preferred or self._create_cluster(clusters, identifier, page_index)

        if best_cluster and best_score >= threshold:
            self._merge_identifier(best_cluster, identifier, page_index, best_score)
            return best_cluster

        return self._create_cluster(clusters, identifier, page_index)

    def _create_cluster(
        self,
        clusters: List[ClusterCandidate],
        identifier: IdentifierMatch,
        page_index: int,
    ) -> ClusterCandidate:
        patient_id = f"patient_{next(self._next_patient):03d}"
        logger.debug(
            "Creating new cluster %s for %s '%s'",
            patient_id,
            identifier.kind,
            identifier.value,
        )
        cluster = ClusterCandidate(
            patient_id=patient_id,
            identifiers={
                identifier.kind: LinkedIdentifier(
                    kind=identifier.kind,
                    value=identifier.value,
                    confidence=identifier.confidence,
                    sources=[(page_index, identifier.word_indices)],
                )
            },
            pages=[page_index],
            score=identifier.confidence,
        )
        clusters.append(cluster)
        return cluster

    @staticmethod
    def _nearest_anchor(
        identifier_index: int,
        anchor_positions: List[Tuple[int, ClusterCandidate]],
    ) -> ClusterCandidate:
        preceding = [item for item in anchor_positions if item[0] <= identifier_index]
        if preceding:
            # Use the nearest preceding anchor (highest index <= identifier_index).
            nearest = max(preceding, key=lambda item: item[0])
            return nearest[1]

        # No preceding anchors; fall back to the earliest following anchor.
        following = min(anchor_positions, key=lambda item: item[0])
        return following[1]

    def _match_identifier(self, cluster: ClusterCandidate, identifier: IdentifierMatch) -> float:
        linked = cluster.identifiers.get(identifier.kind)
        if not linked:
            return 0.0
        if identifier.kind == "name":
            return self.matcher.score_name(linked.value, identifier.value)
        if identifier.kind == "mrn":
            return self.matcher.score_mrn(linked.value, identifier.value)
        if identifier.kind == "dob":
            return self.matcher.score_dob(linked.value, identifier.value)
        if identifier.kind == "phone":
            return self.matcher.score_phone(linked.value, identifier.value)
        return 0.0

    def _threshold_for(self, kind: str) -> float:
        mapping = {
            "name": self.settings.name_match_threshold,
            "mrn": self.settings.mrn_match_threshold,
            "dob": self.settings.dob_match_threshold,
            "phone": self.settings.phone_match_threshold,
        }
        return mapping.get(kind, 0.8)

    def _merge_identifier(
        self,
        cluster: ClusterCandidate,
        identifier: IdentifierMatch,
        page_index: int,
        score: float,
    ) -> None:
        existing = cluster.identifiers.get(identifier.kind)
        if not existing:
            cluster.identifiers[identifier.kind] = LinkedIdentifier(
                kind=identifier.kind,
                value=identifier.value,
                confidence=identifier.confidence,
                sources=[(page_index, identifier.word_indices)],
            )
        else:
            if score > existing.confidence:
                existing.value = identifier.value
            existing.confidence = max(existing.confidence, identifier.confidence)
            existing.sources.append((page_index, identifier.word_indices))
        if page_index not in cluster.pages:
            cluster.pages.append(page_index)
        cluster.score = max(cluster.score, identifier.confidence)

    def _to_linked_patient(self, cluster: ClusterCandidate) -> LinkedPatient:
        identifiers = sorted(cluster.identifiers.values(), key=lambda ident: ident.kind)
        identifiers_list = [
            LinkedIdentifier(
                kind=ident.kind,
                value=ident.value,
                confidence=round(ident.confidence, 3),
                sources=ident.sources,
            )
            for ident in identifiers
        ]
        return LinkedPatient(
            patient_id=cluster.patient_id,
            identifiers=identifiers_list,
            pages=sorted(cluster.pages),
            score=round(cluster.score, 3),
        )
