"""Fuzzy matching utilities for patient identifier comparison."""

from __future__ import annotations

import re
from typing import Callable

from config import Settings

try:  # pragma: no cover - import guard
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback when rapidfuzz missing
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100

        token_sort_ratio = ratio
        partial_ratio = ratio

    fuzz = _FallbackFuzz()  # type: ignore


class FuzzyMatcher:
    """Score similarity between identifier values using configurable heuristics."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def score_name(self, a: str, b: str) -> float:
        a_clean, b_clean = self._normalize_name(a), self._normalize_name(b)
        return self._scale_score(fuzz.token_sort_ratio(a_clean, b_clean))

    def score_mrn(self, a: str, b: str) -> float:
        a_norm, b_norm = self._normalize_mrn(a), self._normalize_mrn(b)
        if not a_norm or not b_norm:
            return 0.0
        base = 1.0 if a_norm == b_norm else self._scale_score(fuzz.ratio(a_norm, b_norm))
        return base

    def score_dob(self, a: str, b: str) -> float:
        return 1.0 if self._normalize_dob(a) == self._normalize_dob(b) else 0.0

    def score_phone(self, a: str, b: str) -> float:
        a_norm, b_norm = self._normalize_phone(a), self._normalize_phone(b)
        if not a_norm or not b_norm:
            return 0.0
        return 1.0 if a_norm == b_norm else self._scale_score(fuzz.partial_ratio(a_norm, b_norm))

    @staticmethod
    def _scale_score(raw: float) -> float:
        return max(0.0, min(raw / 100.0, 1.0))

    @staticmethod
    def _normalize_name(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9\s]", "", value)
        return re.sub(r"\s+", " ", value)

    @staticmethod
    def _normalize_mrn(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    @staticmethod
    def _normalize_dob(value: str) -> str:
        value = value.replace("-", "/").replace(".", "/")
        parts = value.split("/")
        if len(parts) != 3:
            return value
        month, day, year = parts
        if len(year) == 2:
            year = "20" + year if int(year) < 50 else "19" + year
        return f"{month.zfill(2)}/{day.zfill(2)}/{year.zfill(4)}"

    @staticmethod
    def _normalize_phone(value: str) -> str:
        digits = re.sub(r"\D", "", value)
        return digits[-10:] if len(digits) >= 10 else digits
