from __future__ import annotations

from ml.scripts.coarse_crnn.analyze_oto_voicebank_diagnostics import analyze_voicebank


def test_analyze_voicebank_matches_eval_rows_and_detects_order_issues():
    manifest = [
        _manifest_row("vb", "a.wav", "a", 0, 3, 0.0, 100.0),
        _manifest_row("vb", "a.wav", "b", 1, 3, 0.33, 80.0),
        _manifest_row("vb", "a.wav", "", 2, 3, 0.66, 900.0),
        _manifest_row("other", "b.wav", "x", 0, 1, 0.0, 10.0),
    ]
    eval_rows = [
        {
            "voicebank_id": "vb",
            "format_type": "cvvc",
            "audio": "a.wav",
            "alias": "b",
            "confidence": 0.2,
            "low_confidence": True,
            "hard_failure": True,
            "param_abs_errors_ms": {
                "offset": 50.0,
                "preutterance": 60.0,
                "overlap": 10.0,
                "consonant": 20.0,
                "cutoff_abs": 500.0,
            },
        }
    ]

    result = analyze_voicebank(manifest, eval_rows=eval_rows, voicebank_id="vb", format_type="cvvc")

    assert result["manifest_rows"] == 3
    assert result["eval_matched_rows"] == 1
    assert result["row_order_summary"]["monotonic_violations"] == 1
    assert result["eval_error_summary"]["eval_hard_failure_rate"] == 1.0
    assert result["abnormal_rows"][0]["alias"] == ""
    assert result["diagnostic_eval_manifest_rows"][0]["alias"] == "b"


def _manifest_row(
    voicebank: str,
    audio: str,
    alias: str,
    row_index: int,
    row_count: int,
    ratio: float,
    offset: float,
) -> dict:
    return {
        "voicebank_id": voicebank,
        "language": "korean",
        "format_type": "cvvc",
        "audio": audio,
        "wav": audio,
        "alias": alias,
        "alias_type": "cv",
        "transition_type": "multi",
        "alias_role": "cv",
        "row_index_in_wav": row_index,
        "file_row_count": row_count,
        "row_ratio_in_wav": ratio,
        "duration_ms": 1000.0,
        "target_offset_ms": offset,
        "target_preutterance_ms": 100.0,
        "target_overlap_ms": 50.0,
        "target_consonant_ms": 100.0,
        "target_cutoff_abs_ms": offset + 200.0,
        "quality_reason": "ok",
    }
