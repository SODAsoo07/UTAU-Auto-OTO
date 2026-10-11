import hashlib
import json
from pathlib import Path

import pytest

from universal_auto_oto.alias_type_confidence import (
    evaluate_alias_type_confidence,
    fit_alias_type_confidence,
    load_alias_type_confidence,
    predict_alias_type_confidence,
)


def _type() -> dict[str, object]:
    return {
        "structural_role": "CV",
        "acoustic_subtype": "hard",
        "transition_type": "plain",
        "edge_role": "isolated",
        "confidence": 1.0,
    }


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    row_report = tmp_path / "rows.jsonl"
    rows = [{"alias_type": _type()} for _ in range(24)]
    row_report.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    row_sha = hashlib.sha256(row_report.read_bytes()).hexdigest()
    gold = tmp_path / "gold.jsonl"
    gold_rows = [
        {
            "reference_kind": "manual_alias_type_gold",
            "row_report_sha256": row_sha,
            "row_index": index,
            "evaluation_split": "train" if index < 20 else "validation",
            "bank_id": "train_bank" if index < 20 else "validation_bank",
            "root_speaker_id": "train_speaker" if index < 20 else "validation_speaker",
            "alias_type": {key: value for key, value in _type().items() if key != "confidence"},
        }
        for index in range(24)
    ]
    gold.write_text("\n".join(json.dumps(row) for row in gold_rows) + "\n", encoding="utf-8")
    return row_report, gold


def test_alias_type_confidence_fit_is_train_only_and_validation_is_separate(tmp_path: Path) -> None:
    row_report, gold = _dataset(tmp_path)
    profile_path = tmp_path / "profile.json"
    result = fit_alias_type_confidence(row_report, gold, profile_path, minimum_support=20)
    assert result["training_splits_used"] == ["train"]
    assert result["validation_labels_used"] is False
    assert result["locked_test_labels_used"] is False
    assert result["train_example_count"] == 20
    profile = load_alias_type_confidence(profile_path)
    probability, key, support = predict_alias_type_confidence(profile, _type())
    assert probability == pytest.approx(1.0)
    assert key == "CV|hard"
    assert support == 20
    evaluation = evaluate_alias_type_confidence(
        row_report,
        gold,
        profile_path,
        tmp_path / "evaluation.json",
        minimum_examples=4,
    )
    assert evaluation["passed"] is True
    assert evaluation["accuracy"] == 1.0
    assert evaluation["calibrated_brier"] == 0.0
    assert evaluation["identity_leakage_count"] == 0


def test_locked_alias_type_confidence_evaluation_requires_explicit_unlock(tmp_path: Path) -> None:
    row_report, gold = _dataset(tmp_path)
    profile_path = tmp_path / "profile.json"
    fit_alias_type_confidence(row_report, gold, profile_path, minimum_support=20)
    with pytest.raises(ValueError, match="allow_locked_test"):
        evaluate_alias_type_confidence(
            row_report,
            gold,
            profile_path,
            tmp_path / "evaluation.json",
            split="locked_test",
        )


@pytest.mark.parametrize("mutation", ["wrong_hash", "wrong_kind", "duplicate"])
def test_alias_type_confidence_holdout_rejects_unbound_or_duplicate_gold(
    tmp_path: Path,
    mutation: str,
) -> None:
    row_report, gold = _dataset(tmp_path)
    profile_path = tmp_path / "profile.json"
    fit_alias_type_confidence(row_report, gold, profile_path, minimum_support=20)
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    validation = next(row for row in rows if row["evaluation_split"] == "validation")
    if mutation == "wrong_hash":
        validation["row_report_sha256"] = "0" * 64
    elif mutation == "wrong_kind":
        validation["reference_kind"] = "phone_gold"
    else:
        rows.append(dict(validation))
    gold.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different row report|reference_kind|duplicated"):
        evaluate_alias_type_confidence(
            row_report,
            gold,
            profile_path,
            tmp_path / "evaluation.json",
            minimum_examples=1,
        )


def test_alias_type_confidence_holdout_rejects_bank_or_speaker_leakage(tmp_path: Path) -> None:
    row_report, gold = _dataset(tmp_path)
    profile_path = tmp_path / "profile.json"
    fit_alias_type_confidence(row_report, gold, profile_path, minimum_support=20)
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["evaluation_split"] == "validation":
            row["bank_id"] = "train_bank"
    gold.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = evaluate_alias_type_confidence(
        row_report,
        gold,
        profile_path,
        tmp_path / "evaluation.json",
        minimum_examples=4,
    )
    assert result["passed"] is False
    assert result["leaked_bank_ids"] == ["train_bank"]
