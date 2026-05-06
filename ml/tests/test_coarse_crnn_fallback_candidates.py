from __future__ import annotations

from ml.scripts.coarse_crnn.analyze_oto_fallback_candidates import analyze_fallback_candidates


def test_fallback_candidate_analysis_scores_recall_and_effective_metrics():
    payload = {
        "files": [
            {
                "language": "korean",
                "format_type": "cvvc",
                "alias_role": "vc",
                "confidence": 0.40,
                "predicted_error_ms": 140.0,
                "low_confidence": True,
                "hard_failure": False,
                "fn_voiced_miss": False,
                "param_abs_errors_ms": {
                    "preutterance": 120.0,
                    "offset": 800.0,
                    "consonant": 30.0,
                    "cutoff_abs": 950.0,
                    "overlap": 20.0,
                },
            },
            {
                "language": "korean",
                "format_type": "cvvc",
                "alias_role": "cv",
                "confidence": 0.80,
                "predicted_error_ms": 30.0,
                "low_confidence": False,
                "hard_failure": False,
                "fn_voiced_miss": False,
                "param_abs_errors_ms": {
                    "preutterance": 20.0,
                    "offset": 40.0,
                    "consonant": 10.0,
                    "cutoff_abs": 50.0,
                    "overlap": 5.0,
                },
            },
            {
                "language": "japanese",
                "format_type": "cvvc",
                "alias_role": "vc",
                "confidence": 0.20,
                "predicted_error_ms": 200.0,
                "low_confidence": True,
                "hard_failure": True,
                "fn_voiced_miss": True,
                "param_abs_errors_ms": {
                    "preutterance": 300.0,
                    "offset": 1200.0,
                    "consonant": 100.0,
                    "cutoff_abs": 1500.0,
                    "overlap": 60.0,
                },
            },
        ]
    }

    report = analyze_fallback_candidates(payload, language="korean", format_type="cvvc")
    low_conf = next(row for row in report["candidates"] if row["name"] == "low_confidence")

    assert report["rows"] == 2
    assert report["bad_rows"] == 1
    assert low_conf["bad_recall"] == 1.0
    assert low_conf["good_keep_rate"] == 1.0
    assert low_conf["effective_preutterance_acc_50ms"] == 1.0
    assert low_conf["effective_offset_mae_ms"] == 20.0
