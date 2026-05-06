from __future__ import annotations

from core.coarse_crnn.oto_evaluate import _aggregate_by_language_format_role
from ml.scripts.coarse_crnn.gate_oto_eval_slices import DEFAULT_GATE_SPECS, gate_eval_slices


def test_aggregate_by_language_format_role_groups_eval_rows():
    rows = [
        _row("korean", "cvvc", "vc", pre=40.0, offset=100.0, cutoff=300.0),
        _row("korean", "cvvc", "vc", pre=80.0, offset=200.0, cutoff=500.0),
        _row("japanese", "cvvc", "vc", pre=20.0, offset=60.0, cutoff=120.0),
    ]

    result = _aggregate_by_language_format_role(rows)

    assert result["korean|cvvc|vc"]["files"] == 2
    assert result["korean|cvvc|vc"]["preutterance_acc_50ms"] == 0.5
    assert result["korean|cvvc|vc"]["offset_mae_ms"] == 150.0
    assert result["japanese|cvvc|vc"]["files"] == 1


def test_gate_eval_slices_applies_language_format_role_specs():
    payload = {
        "files": [
            _row("korean", "cvvc", "vc", pre=80.0, offset=900.0, cutoff=1100.0, hard=True),
            _row("korean", "cvvc", "vc", pre=120.0, offset=800.0, cutoff=1000.0),
            _row("japanese", "cvvc", "vc", pre=20.0, offset=100.0, cutoff=200.0),
            _row("japanese", "cvvc", "vc", pre=30.0, offset=120.0, cutoff=220.0),
            _row("korean", "cvc", "cv", pre=35.0, offset=100.0, cutoff=200.0),
            _row("korean", "cvc", "cv", pre=40.0, offset=110.0, cutoff=210.0),
        ]
    }
    specs = [
        {
            "pattern": "korean/cvvc/*",
            "min_files": 2,
            "min_preutterance_acc_50ms": 0.5,
            "max_preutterance_mae_ms": 100.0,
            "max_offset_mae_ms": 500.0,
            "max_cutoff_abs_mae_ms": 800.0,
            "max_hard_failure_rate": 0.25,
            "max_syllable_shift_rate": 0.25,
            "recommendation": "ko_cvvc_test_fallback",
        },
        {
            "pattern": "japanese/cvvc/*",
            "min_files": 2,
            "min_preutterance_acc_50ms": 0.5,
            "max_preutterance_mae_ms": 100.0,
            "max_offset_mae_ms": 500.0,
            "max_cutoff_abs_mae_ms": 800.0,
            "max_hard_failure_rate": 0.25,
            "max_syllable_shift_rate": 0.25,
        },
        {
            "pattern": "korean/cvc/*",
            "min_files": 2,
            "min_preutterance_acc_50ms": 0.5,
            "max_preutterance_mae_ms": 100.0,
            "max_offset_mae_ms": 500.0,
            "max_cutoff_abs_mae_ms": 800.0,
            "max_hard_failure_rate": 0.25,
            "max_syllable_shift_rate": 0.25,
        },
    ]

    report = gate_eval_slices(payload, specs=specs)

    assert report["matched_count"] == 3
    assert report["failed_count"] == 1
    assert report["failed_slices"][0]["key"] == "korean/cvvc/vc"
    assert "offset_mae_ms" in report["failed_slices"][0]["failed_checks"]
    assert "ko_cvvc_test_fallback" in report["failed_slices"][0]["fallback_tags"]


def test_gate_eval_slices_flags_korean_cvvc_syllable_shift():
    payload = {
        "files": [
            _row("korean", "cvvc", "vc", pre=35.0, offset=100.0, cutoff=300.0, shift=True),
            _row("korean", "cvvc", "vc", pre=40.0, offset=120.0, cutoff=320.0, shift=True),
            _row("korean", "cvvc", "vc", pre=45.0, offset=130.0, cutoff=330.0),
        ]
    }

    report = gate_eval_slices(
        payload,
        specs=[
            {
                "pattern": "korean/cvvc/vc",
                "min_files": 3,
                "min_preutterance_acc_50ms": 0.5,
                "max_preutterance_mae_ms": 120.0,
                "max_offset_mae_ms": 300.0,
                "max_cutoff_abs_mae_ms": 500.0,
                "max_syllable_shift_rate": 0.50,
                "recommendation": "korean_cvvc_vc_activity_fallback",
            }
        ],
    )

    failed = report["failed_slices"][0]
    assert failed["key"] == "korean/cvvc/vc"
    assert failed["syllable_shift_rate"] == 2 / 3
    assert "syllable_shift_rate" in failed["failed_checks"]
    assert "syllable_shift_guard" in failed["fallback_tags"]
    assert "korean_cvvc_vc_activity_fallback" in failed["fallback_tags"]


def test_builtin_gate_specs_prefer_korean_cvvc_role_specific_patterns():
    payload = {
        "by_language_format_role": {
            "korean|cvvc|v-cv": {
                "files": 30,
                "preutterance_acc_50ms": 0.10,
                "preutterance_mae_ms": 260.0,
                "offset_mae_ms": 950.0,
                "cutoff_abs_mae_ms": 1800.0,
                "hard_failure_rate": 0.0,
                "fn_voiced_miss_rate": 0.70,
                "syllable_shift_rate": 0.70,
            }
        }
    }

    report = gate_eval_slices(payload, specs=list(DEFAULT_GATE_SPECS))

    failed = report["failed_slices"][0]
    assert failed["pattern"] == "korean/cvvc/v-cv"
    assert "korean_cvvc_v_cv_bridge_guard" in failed["fallback_tags"]


def _row(
    language: str,
    format_type: str,
    alias_role: str,
    *,
    pre: float,
    offset: float,
    cutoff: float,
    hard: bool = False,
    shift: bool = False,
    voiced_miss: bool = False,
) -> dict:
    return {
        "language": language,
        "format_type": format_type,
        "alias_role": alias_role,
        "hard_failure": hard,
        "fn_voiced_miss": voiced_miss,
        "syllable_shift_flag": shift,
        "param_abs_errors_ms": {
            "preutterance": pre,
            "offset": offset,
            "consonant": 50.0,
            "cutoff_abs": cutoff,
            "overlap": 20.0,
        },
    }
