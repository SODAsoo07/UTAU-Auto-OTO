import json
from pathlib import Path

import pytest

from universal_auto_oto.renderer_calibration import (
    ANCHOR_FIELDS,
    apply_renderer_calibration,
    build_renderer_calibration_examples,
    evaluate_renderer_calibration_holdout,
    fit_renderer_calibration,
    load_renderer_calibration,
    select_renderer_calibration,
)


def _write_examples(path: Path) -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "split": "train",
                "reference_kind": "manual_oto_anchor_gold",
                "source_row_report_sha256": "b" * 64,
                "template_id": "cv_hard",
                "bank_id": "bank_a" if index < 8 else "bank_b",
                "format_type": "korean_cvvc",
                "residuals_ms": {field: 10.0 + index for field in ANCHOR_FIELDS},
            }
        )
    rows.append(
        {
            "split": "locked_test",
            "reference_kind": "manual_oto_anchor_gold",
            "template_id": "cv_hard",
            "bank_id": "locked_bank",
            "format_type": "korean_cvvc",
            "residuals_ms": {field: 9999.0 for field in ANCHOR_FIELDS},
        }
    )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_fit_is_train_only_hierarchical_and_runtime_selects_bank(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    profile_path = tmp_path / "renderer_profile.json"
    _write_examples(examples)
    result = fit_renderer_calibration(examples, profile_path, minimum_support=4, shrinkage_strength=4.0)
    assert result["training_splits_used"] == ["train"]
    assert result["validation_labels_used"] is False
    assert result["locked_test_labels_used"] is False
    assert result["split_counts"] == {"locked_test": 1, "train": 12}
    assert result["used_train_examples"] == 12
    assert "bank_template|locked_bank|cv_hard" not in result["entries"]
    assert result["entries"]["format_template|korean_cvvc|cv_hard"]["parent_key"] == "template|cv_hard"
    assert result["entries"]["bank_template|bank_a|cv_hard"]["parent_key"] == (
        "format_template|korean_cvvc|cv_hard"
    )

    profile = load_renderer_calibration(profile_path)
    selection = select_renderer_calibration(
        profile,
        template_id="cv_hard",
        bank_id="bank_a",
        format_type="korean_cvvc",
    )
    assert selection.scope == "bank_template"
    assert selection.support == 8
    assert selection.shrinkage_weight == pytest.approx(8 / 12)
    assert set(selection.statistics) == {
        "raw_median_ms", "mad_ms", "p10_ms", "p25_ms", "p75_ms", "p90_ms", "outlier_rate"
    }
    assert set(selection.statistics["mad_ms"]) == set(ANCHOR_FIELDS)
    anchors = {field: 100.0 for field in ANCHOR_FIELDS}
    calibrated = apply_renderer_calibration(anchors, selection)
    assert all(calibrated[field] > 100.0 for field in ANCHOR_FIELDS)
    assert len(profile.sha256) == 64
    assert len(profile.training_input_sha256) == 64
    assert profile.training_bank_ids == frozenset({"bank_a", "bank_b"})
    assert profile.training_format_types == frozenset({"korean_cvvc"})


def test_fit_rejects_insufficient_train_support(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps(
            {
                "split": "train",
                "reference_kind": "manual_oto_anchor_gold",
                "template_id": "cv_hard",
                "bank_id": "bank",
                "format_type": "cv",
                "residuals_ms": {field: 1.0 for field in ANCHOR_FIELDS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="support is insufficient"):
        fit_renderer_calibration(examples, tmp_path / "profile.json", minimum_support=2)


def test_fit_rejects_unreviewed_reference_kind(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps(
            {
                "split": "train",
                "reference_kind": "source_oto_unverified",
                "template_id": "cv_hard",
                "bank_id": "bank",
                "format_type": "cv",
                "residuals_ms": {field: 1.0 for field in ANCHOR_FIELDS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported reference_kind"):
        fit_renderer_calibration(examples, tmp_path / "profile.json", minimum_support=1)


def test_expression_and_pitch_profiles_override_broader_bank_scope(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    rows = []
    for index in range(4):
        rows.append(
            {
                "split": "train",
                "reference_kind": "manual_oto_anchor_gold",
                "template_id": "cv_soft",
                "bank_id": "bank",
                "format_type": "korean_cvvc",
                "expression_profile": "power",
                "pitch_bucket": "high",
                "residuals_ms": {field: 10.0 + index for field in ANCHOR_FIELDS},
            }
        )
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    fit_renderer_calibration(examples, profile_path, minimum_support=4, shrinkage_strength=0.0)
    raw_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert raw_profile["entries"]["expression_template|power|cv_soft"]["parent_key"] == (
        "bank_template|bank|cv_soft"
    )
    assert raw_profile["entries"]["pitch_template|high|cv_soft"]["parent_key"] == (
        "expression_template|power|cv_soft"
    )
    profile = load_renderer_calibration(profile_path)
    pitch = select_renderer_calibration(
        profile,
        template_id="cv_soft",
        bank_id="bank",
        format_type="korean_cvvc",
        expression_profile="power",
        pitch_bucket="high",
    )
    expression = select_renderer_calibration(
        profile,
        template_id="cv_soft",
        bank_id="bank",
        format_type="korean_cvvc",
        expression_profile="power",
    )
    assert pitch.scope == "pitch_template"
    assert expression.scope == "expression_template"


def test_reviewed_anchors_are_bound_to_row_report_and_converted_to_residuals(tmp_path: Path) -> None:
    row_report = tmp_path / "row_report.jsonl"
    raw_anchors = {field: 100.0 + index for index, field in enumerate(ANCHOR_FIELDS)}
    row_report.write_text(
        json.dumps(
            {
                "recording_wav": "wav/a.wav",
                "alias": "a",
                "template_id": "cv_hard",
                "alias_type": {"structural_role": "CV", "acoustic_subtype": "hard"},
                "raw_anchors": raw_anchors,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    references = tmp_path / "reviewed.jsonl"
    references.write_text(
        json.dumps(
            {
                "row_index": 0,
                "wav": "wav/a.wav",
                "alias": "a",
                "split": "train",
                "reference_kind": "manual_oto_convention_reviewed",
                "metadata": {"bank_id": "bank_a", "format_type": "korean_cvvc"},
                "reviewed_anchors": {field: value + 3.0 for field, value in raw_anchors.items()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "examples.jsonl"
    report = build_renderer_calibration_examples(row_report, references, output)
    example = json.loads(output.read_text(encoding="utf-8"))
    assert report["example_count"] == 1
    assert report["release_promotion_evidence"] is False
    assert example["residuals_ms"] == {field: 3.0 for field in ANCHOR_FIELDS}
    assert example["source_row_report_sha256"] == report["row_report_sha256"]


def test_reviewed_anchor_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    row_report = tmp_path / "row_report.jsonl"
    row_report.write_text(
        json.dumps(
            {
                "recording_wav": "a.wav",
                "alias": "a",
                "template_id": "cv",
                "raw_anchors": {field: 1.0 for field in ANCHOR_FIELDS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    references = tmp_path / "reviewed.jsonl"
    references.write_text(
        json.dumps(
            {
                "row_index": 0,
                "alias": "different",
                "split": "train",
                "reference_kind": "manual_oto_anchor_gold",
                "metadata": {"bank_id": "bank", "format_type": "korean_cvvc"},
                "reviewed_anchors": {field: 2.0 for field in ANCHOR_FIELDS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="alias does not match"):
        build_renderer_calibration_examples(row_report, references, tmp_path / "examples.jsonl")


def test_holdout_evaluation_improves_unseen_bank_without_label_leakage(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    rows = []
    for index in range(4):
        raw = {field: 100.0 + index for field in ANCHOR_FIELDS}
        reviewed = {field: value + 5.0 for field, value in raw.items()}
        rows.append(
            {
                "split": "train",
                "reference_kind": "manual_oto_anchor_gold",
                "template_id": "cv_hard",
                "bank_id": "train_bank",
                "format_type": "korean_cvvc",
                "raw_anchors_ms": raw,
                "reviewed_anchors_ms": reviewed,
                "residuals_ms": {field: 5.0 for field in ANCHOR_FIELDS},
            }
        )
    for index in range(2):
        raw = {field: 200.0 + index for field in ANCHOR_FIELDS}
        rows.append(
            {
                "split": "validation",
                "reference_kind": "manual_oto_anchor_gold",
                "source_row_report_sha256": "b" * 64,
                "template_id": "cv_hard",
                "bank_id": "heldout_bank",
                "format_type": "korean_cvvc",
                "raw_anchors_ms": raw,
                "reviewed_anchors_ms": {field: value + 5.0 for field, value in raw.items()},
                "residuals_ms": {field: 5.0 for field in ANCHOR_FIELDS},
            }
        )
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    fit_renderer_calibration(examples, profile, minimum_support=4, shrinkage_strength=0.0)
    result = evaluate_renderer_calibration_holdout(
        examples,
        profile,
        tmp_path / "evaluation.json",
        minimum_examples=2,
    )
    assert result["passed"] is True
    assert result["baseline_mae_ms"] == pytest.approx(5.0)
    assert result["candidate_mae_ms"] == pytest.approx(0.0)
    assert result["baseline_template_residual_median_ms"] == pytest.approx(5.0)
    assert result["candidate_template_residual_median_ms"] == pytest.approx(0.0)
    assert result["template_residual_median_reduced"] is True
    assert result["large_repair_not_increased"] is True
    assert result["baseline_manual_edit_distance_ms"] == pytest.approx(10.0)
    assert result["candidate_manual_edit_distance_ms"] == pytest.approx(0.0)
    assert result["baseline_oto_field_mae_ms"] == {
        "offset": pytest.approx(5.0),
        "overlap": pytest.approx(0.0),
        "preutterance": pytest.approx(0.0),
        "consonant": pytest.approx(0.0),
        "cutoff_abs": pytest.approx(5.0),
    }
    assert result["selection_scope_counts"] == {"format_template": 2}
    assert result["heldout_bank_leakage_count"] == 0
    assert result["source_row_report_sha256s"] == ["b" * 64]
    assert result["release_promotion_evidence"] is False


def test_holdout_evaluation_rejects_seen_bank_profile_selection(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    rows = []
    for split in ("train", "train", "validation"):
        raw = {field: 100.0 for field in ANCHOR_FIELDS}
        rows.append(
            {
                "split": split,
                "reference_kind": "manual_oto_anchor_gold",
                "source_row_report_sha256": "b" * 64,
                "template_id": "cv_hard",
                "bank_id": "same_bank",
                "format_type": "korean_cvvc",
                "raw_anchors_ms": raw,
                "reviewed_anchors_ms": {field: 102.0 for field in ANCHOR_FIELDS},
                "residuals_ms": {field: 2.0 for field in ANCHOR_FIELDS},
            }
        )
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    fit_renderer_calibration(examples, profile, minimum_support=2, shrinkage_strength=0.0)
    result = evaluate_renderer_calibration_holdout(
        examples,
        profile,
        tmp_path / "evaluation.json",
        minimum_examples=1,
    )
    assert result["passed"] is False
    assert result["heldout_bank_leakage_count"] == 1
    assert result["checks"]["heldout_bank_isolation"] is False


def test_holdout_rejects_seen_bank_even_when_profile_backoff_hides_bank_scope(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    rows = []
    for index, bank_id in enumerate(("seen_bank", "other_bank", "other_bank")):
        raw = {field: 100.0 + index for field in ANCHOR_FIELDS}
        rows.append({
            "split": "train",
            "reference_kind": "manual_oto_anchor_gold",
            "source_row_report_sha256": "c" * 64,
            "template_id": "cv_hard",
            "bank_id": bank_id,
            "format_type": "korean_cvvc",
            "raw_anchors_ms": raw,
            "reviewed_anchors_ms": {field: value + 3.0 for field, value in raw.items()},
            "residuals_ms": {field: 3.0 for field in ANCHOR_FIELDS},
        })
    raw = {field: 200.0 for field in ANCHOR_FIELDS}
    rows.append({
        "split": "validation",
        "reference_kind": "manual_oto_anchor_gold",
        "source_row_report_sha256": "c" * 64,
        "template_id": "cv_hard",
        "bank_id": "seen_bank",
        "format_type": "korean_cvvc",
        "raw_anchors_ms": raw,
        "reviewed_anchors_ms": {field: value + 3.0 for field, value in raw.items()},
        "residuals_ms": {field: 3.0 for field in ANCHOR_FIELDS},
    })
    examples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    profile = tmp_path / "profile.json"
    fit_renderer_calibration(examples, profile, minimum_support=3, shrinkage_strength=0.0)
    result = evaluate_renderer_calibration_holdout(
        examples,
        profile,
        tmp_path / "evaluation.json",
        minimum_examples=1,
    )
    assert result["selection_scope_counts"] != {"bank_template": 1}
    assert result["heldout_bank_leakage_count"] == 1
    assert result["passed"] is False
