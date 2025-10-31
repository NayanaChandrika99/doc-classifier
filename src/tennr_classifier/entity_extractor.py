"""Regex-based patient identifier extraction from OCR results."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from config import Settings
from .logging_utils import get_logger
from .pipeline import IdentifierMatch, OCRResult, OCRWord, PageEntities

logger = get_logger(__name__)


@dataclass
class MatchCandidate:
    kind: str
    value: str
    confidence: float
    word_indices: List[int]


class EntityExtractor:
    """Extract patient identifiers from OCR output using regex heuristics."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._name_pattern = settings.compiled_pattern("name")
        self._mrn_pattern = settings.compiled_pattern("mrn")
        self._dob_pattern = settings.compiled_pattern("dob")
        self._phone_pattern = settings.compiled_pattern("phone")

    def extract_page(self, ocr_result: OCRResult) -> PageEntities:
        matches = self._collect_matches(ocr_result)
        identifiers = [self._build_identifier(ocr_result, candidate) for candidate in matches]
        logger.debug("Page %s: extracted %s identifiers", ocr_result.page_index, len(identifiers))
        return PageEntities(page_index=ocr_result.page_index, identifiers=identifiers)

    def extract_document(self, results: Sequence[OCRResult]) -> List[PageEntities]:
        return [self.extract_page(result) for result in results]

    def _collect_matches(self, ocr_result: OCRResult) -> List[MatchCandidate]:
        text = ocr_result.text
        matches: List[MatchCandidate] = []
        matches.extend(self._match_pattern("name", self._name_pattern, text, ocr_result))
        matches.extend(self._match_pattern("mrn", self._mrn_pattern, text, ocr_result))
        matches.extend(self._match_pattern("dob", self._dob_pattern, text, ocr_result))
        matches.extend(self._match_pattern("phone", self._phone_pattern, text, ocr_result))
        return self._deduplicate(matches)

    def _match_pattern(
        self,
        kind: str,
        pattern: re.Pattern[str],
        text: str,
        ocr_result: OCRResult,
    ) -> List[MatchCandidate]:
        candidates: List[MatchCandidate] = []
        for match in pattern.finditer(text):
            group_value = match.group(kind) if kind in match.groupdict() else match.group(0)
            if not group_value:
                continue
            truncated_value = self._truncate_value(group_value)
            normalized_value = self._normalize_value(kind, truncated_value)
            span_end = match.start() + len(truncated_value)
            word_indices = self._map_to_words(ocr_result.words, match.start(), span_end, text)
            confidence = self._confidence(kind, normalized_value, word_indices, ocr_result.words)
            if confidence < self.settings.entity_min_confidence:
                logger.debug(
                    "Discarding %s match '%s' due to low confidence %.2f", kind, normalized_value, confidence
                )
                continue
            candidates.append(MatchCandidate(kind, normalized_value, confidence, word_indices))
        return candidates

    def _map_to_words(
        self,
        words: Sequence[OCRWord],
        start: int,
        end: int,
        full_text: str,
    ) -> List[int]:
        indices: List[int] = []
        cursor = 0
        for idx, word in enumerate(words):
            word_text = word.text
            if not word_text:
                continue
            lower_text = full_text.lower()
            try:
                word_start = lower_text.index(word_text.lower(), cursor)
            except ValueError:
                continue
            word_end = word_start + len(word_text)
            cursor = word_end
            if word_end < start or word_start > end:
                continue
            indices.append(idx)
        return indices

    def _confidence(
        self,
        kind: str,
        value: str,
        word_indices: List[int],
        words: Sequence[OCRWord],
    ) -> float:
        base = 0.6
        if kind == "name":
            base += 0.1
        if kind == "mrn":
            base += min(len(value) / 12.0, 0.2)
        if kind == "dob":
            base += 0.1
        if not word_indices:
            return base
        avg_word_conf = sum(words[i].confidence for i in word_indices) / len(word_indices)
        return min(1.0, base + (avg_word_conf * 0.2))

    def _truncate_value(self, value: str) -> str:
        for sep in ("\n", "\r"):
            if sep in value:
                return value.split(sep)[0]
        return value

    def _normalize_value(self, kind: str, value: str) -> str:
        value = value.strip()
        if kind == "mrn":
            return re.sub(r"[^A-Za-z0-9]", "", value).upper()
        if kind == "dob":
            return value.replace(".", "/")
        if kind == "name":
            tokens = [token.strip(",:") for token in value.split() if token.strip(",:")]
            return " ".join(part.capitalize() for part in tokens)
        return value

    def _build_identifier(self, ocr_result: OCRResult, candidate: MatchCandidate) -> IdentifierMatch:
        bbox = self._aggregate_bbox([ocr_result.words[i] for i in candidate.word_indices], candidate.kind)
        return IdentifierMatch(
            kind=candidate.kind,
            value=candidate.value,
            confidence=round(candidate.confidence, 3),
            bbox=bbox,
            word_indices=candidate.word_indices,
        )

    def _aggregate_bbox(self, words: Iterable[OCRWord], kind: str) -> tuple[int, int, int, int]:
        xs: List[int] = []
        ys: List[int] = []
        xe: List[int] = []
        ye: List[int] = []
        for word in words:
            x, y, w, h = word.bbox
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys))

    def _deduplicate(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        unique: dict[tuple[str, str], MatchCandidate] = {}
        for candidate in candidates:
            key = (candidate.kind, candidate.value)
            existing = unique.get(key)
            if not existing or existing.confidence < candidate.confidence:
                unique[key] = candidate
        return list(unique.values())
