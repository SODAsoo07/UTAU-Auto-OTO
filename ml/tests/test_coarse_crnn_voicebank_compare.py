from __future__ import annotations

from ml.scripts.coarse_crnn.compare_oto_voicebank_diagnostics import compare_voicebanks


def test_compare_voicebanks_summarizes_rates_and_deltas():
    result = compare_voicebanks(
        [
            {
                "label": "song",
                "diagnostic_path": "song_diagnostic.json",
                "eval_path": "song_eval.json",
                "diagnostic": _diagnostic(rows=10, blank_rate=0.2, order_rate=0.1),
                "eval": _eval(rows=10, pre_acc=0.25, hard_fail=0.1, pre_mae=200.0),
            },
            {
                "label": "sato",
                "diagnostic_path": "sato_diagnostic.json",
                "eval_path": "sato_eval.json",
                "diagnostic": _diagnostic(rows=8, blank_rate=0.0, order_rate=0.05),
                "eval": _eval(rows=8, pre_acc=0.5, hard_fail=0.0, pre_mae=100.0),
            },
        ],
        top_contexts=1,
    )

    assert result["case_count"] == 2
    assert result["baseline_label"] == "song"
    assert result["rows"][0]["blank_alias_rate"] == 0.2
    assert result["rows"][0]["weak_contexts"][0]["key"] == "cvvc|vc|multi"
    assert result["rows"][1]["delta_vs_first"]["preutterance_acc_50ms"] == 0.25
    assert result["rows"][1]["delta_vs_first"]["param_mae_ms"]["preutterance"] == -100.0
    assert "Best preutterance" in result["notes"][0]


def _diagnostic(*, rows: int, blank_rate: float, order_rate: float) -> dict:
    return {
        "manifest_rows": rows,
        "eval_matched_rows": rows,
        "target_summary": {"blank_alias_rate": blank_rate},
        "row_order_summary": {
            "monotonic_violation_rate": order_rate,
            "abs_offset_vs_row_ratio_ms": {"mean": 100.0, "p90": 200.0},
        },
    }


def _eval(*, rows: int, pre_acc: float, hard_fail: float, pre_mae: float) -> dict:
    return {
        "evaluated_items": rows,
        "preutterance_acc_50ms": pre_acc,
        "hard_failure_rate": hard_fail,
        "fn_voiced_miss_rate": 0.2,
        "onset_lag_p90_ms": 300.0,
        "syllable_shift_rate": 0.4,
        "param_mae_ms": {
            "offset": 300.0,
            "preutterance": pre_mae,
            "consonant": 120.0,
            "cutoff_abs": 400.0,
            "overlap": 80.0,
        },
        "by_alias_context": {
            "cvvc|cv|multi": {
                "files": 5,
                "preutterance_acc_50ms": 0.9,
                "preutterance_mae_ms": 30.0,
                "offset_mae_ms": 40.0,
                "consonant_mae_ms": 50.0,
                "cutoff_abs_mae_ms": 60.0,
                "overlap_mae_ms": 20.0,
            },
            "cvvc|vc|multi": {
                "files": 5,
                "preutterance_acc_50ms": 0.1,
                "preutterance_mae_ms": 300.0,
                "offset_mae_ms": 400.0,
                "consonant_mae_ms": 100.0,
                "cutoff_abs_mae_ms": 500.0,
                "overlap_mae_ms": 90.0,
            },
        },
    }
