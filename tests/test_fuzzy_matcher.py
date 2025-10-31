"""Tests for fuzzy matcher scoring functions."""

from __future__ import annotations

import math

from config import Settings
from src.tennr_classifier.fuzzy_matcher import FuzzyMatcher


def _matcher() -> FuzzyMatcher:
    return FuzzyMatcher(Settings())


def test_score_name_exact():
    matcher = _matcher()
    assert math.isclose(matcher.score_name("John Doe", "John Doe"), 1.0)


def test_score_name_insensitive():
    matcher = _matcher()
    assert matcher.score_name("JOHN DOE", "john doe") > 0.9


def test_score_name_swapped_tokens():
    matcher = _matcher()
    assert matcher.score_name("Doe, John", "John Doe") > 0.9


def test_score_mrn_exact():
    matcher = _matcher()
    assert matcher.score_mrn("123-45", "12345") == 1.0


def test_score_mrn_partial():
    matcher = _matcher()
    assert matcher.score_mrn("12345", "12340") < 1.0


def test_score_dob_normalization():
    matcher = _matcher()
    assert matcher.score_dob("01-02-1990", "01/02/1990") == 1.0


def test_score_phone_equivalence():
    matcher = _matcher()
    assert matcher.score_phone("(555) 111-2222", "5551112222") == 1.0
