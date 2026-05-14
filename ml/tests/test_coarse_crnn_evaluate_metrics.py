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


def test_oto_eval_position_bad_rate_uses_full_placement_params():
    from core.coarse_crnn.oto_evaluate import _is_position_bad, _position_bad_rate
    from ml.scripts.coarse_crnn.evaluate_oto_runtime import _summarize_variant

    good = {
        "param_abs_errors_ms": {
            "offset": 20.0,
            "consonant": 40.0,
            "cutoff_abs": 30.0,
            "preutterance": 25.0,
            "overlap": 300.0,
        },
        "anchor_abs_errors_ms": {
            "offset": 20.0,
            "overlap": 300.0,
            "preutterance": 25.0,
            "consonant": 40.0,
            "cutoff": 30.0,
        },
        "hard_failure": False,
    }
    bad = {
        "param_abs_errors_ms": {
            "offset": 260.0,
            "consonant": 20.0,
            "cutoff_abs": 20.0,
            "preutterance": 20.0,
            "overlap": 0.0,
        },
        "anchor_abs_errors_ms": {
            "offset": 260.0,
            "overlap": 0.0,
            "preutterance": 20.0,
            "consonant": 20.0,
            "cutoff": 20.0,
        },
        "hard_failure": False,
    }

    assert not _is_position_bad(good["param_abs_errors_ms"], 250.0)
    assert _is_position_bad(bad["param_abs_errors_ms"], 250.0)
    assert _position_bad_rate([good, bad], 250.0) == 0.5
    assert _summarize_variant([good, bad])["position_bad_rate_250ms"] == 0.5
