import json
from pathlib import Path

import pytest

from universal_auto_oto.confidence import (
    ConfidenceCalibrationError,
    audit_confidence_calibration_readiness,
    fit_confidence_calibration,
    load_confidence_profile,
    predict_risks,
    runtime_confidence_features,
)


def _write_examples(path: Path, *, leak: bool = False) -> None:
    rows = []
    splits = (("train", 120, 0), ("validation", 40, 1000), ("locked_test", 20, 2000))
    for split, count, offset in splits:
        for index in range(count):
            raw_confidence = (index % 25) / 24.0
            warning_count = float(index % 3)
            error_ms = (1.0 - raw_confidence) * 330.0 + warning_count * 12.0
            speaker = f"{split}_speaker_{index % 4}"
            if leak and split == "validation" and index == 0:
                speaker = "train_speaker_0"
            rows.append(
                {
                    "schema_version": 1,
                    "example_id": f"e{offset + index}",
                    "speaker_id": speaker,
                    "bank_id": f"{speaker}_bank",
                    "group": "korean/cv",
                    "split": split,
                    "label_source": "manual_gold",
                    "raw_confidence": raw_confidence,
                    "warning_count": warning_count,
                    "constraint_repair_count": float(index % 2),
                    "duration_z_abs": float(index % 5) / 2.0,
                    "boundary_margin": raw_confidence - 0.5,
                    "error_ms": error_ms,
                    "one_slot_shift": raw_confidence < 0.35,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_confidence_calibration_fits_manual_gold_and_predicts_risk(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    profile_path = tmp_path / "confidence.json"
    _write_examples(examples)
    report = fit_confidence_calibration(
        examples,
        profile_path,
        minimum_train_examples=100,
        minimum_validation_examples=20,
    )
    assert report["train_examples"] == 120
    profile = load_confidence_profile(profile_path)
    risky = predict_risks(
        profile,
        runtime_confidence_features(raw_confidence=0.1, warning_count=2, boundary_margin=-0.4),
    )
    safe = predict_risks(
        profile,
        runtime_confidence_features(raw_confidence=0.9, warning_count=0, boundary_margin=0.4),
    )
    assert risky["error_gt_80ms"] > safe["error_gt_80ms"]
    assert risky["error_gt_200ms"] > safe["error_gt_200ms"]
    assert risky["one_slot_shift"] > safe["one_slot_shift"]


def test_confidence_calibration_rejects_speaker_split_leakage(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_examples(examples, leak=True)
    with pytest.raises(ConfidenceCalibrationError, match="speaker split leakage"):
        fit_confidence_calibration(examples, tmp_path / "out.json")


def test_confidence_readiness_reports_support_without_fitting(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    report_path = tmp_path / "readiness.json"
    _write_examples(examples)
    report = audit_confidence_calibration_readiness(examples, report_path)
    assert report["ready_to_fit"] is True
    assert report["fit_performed"] is False
    assert report["counts"] == {
        "train": 120,
        "validation": 40,
        "locked_test_excluded_from_fit": 20,
    }
    assert report_path.is_file()
