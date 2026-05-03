from __future__ import annotations

from core.coarse_crnn.evaluate import _boundary_errors, _cv_transition_errors, _vowel_onset_errors
from core.coarse_crnn.types import PhoneSegment, Segment


def test_boundary_errors_compare_matched_phone_edges():
    gold = [
        Segment("k", 0.10, 0.16),
        Segment("a", 0.16, 0.40),
    ]
    pred = [
        PhoneSegment("k", 0.11, 0.15, "C_OBS", 0.9),
        PhoneSegment("a", 0.17, 0.42, "V", 0.8),
    ]

    errors = _boundary_errors(gold, pred)

    assert [round(value * 1000.0) for value in errors] == [10, 10, 10, 20]


def test_oto_anchor_metrics_focus_on_vowel_onsets_and_cv_transitions():
    gold = [
        Segment("k", 0.10, 0.16),
        Segment("a", 0.16, 0.40),
        Segment("n", 0.40, 0.47),
        Segment("i", 0.47, 0.62),
    ]
    pred = [
        PhoneSegment("k", 0.10, 0.15, "C_OBS", 0.9),
        PhoneSegment("a", 0.18, 0.39, "V", 0.8),
        PhoneSegment("n", 0.39, 0.45, "C_SON", 0.7),
        PhoneSegment("i", 0.50, 0.62, "V", 0.8),
    ]

    vowel = _vowel_onset_errors(gold, pred)
    cv = _cv_transition_errors(gold, pred)

    assert [round(value * 1000.0) for value in vowel] == [20, 30]
    assert [round(value * 1000.0) for value in cv] == [20, 30]
