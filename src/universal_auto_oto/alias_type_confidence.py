from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .promotion_gate import write_deterministic_json


PROFILE_KIND = "alias_type_empirical_confidence_v1"


@dataclass(frozen=True)
class AliasTypeConfidenceProfile:
    path: Path
    sha256: str
    profile_id: str
    row_report_sha256: str
    global_probability: float
    entries: Mapping[str, Mapping[str, object]]
    training_bank_ids: frozenset[str]
    training_root_speaker_ids: frozenset[str]


def fit_alias_type_confidence(
    row_report_jsonl: Path,
    alias_type_gold_jsonl: Path,
    output_json: Path,
    *,
    minimum_support: int = 20,
    shrinkage_strength: float = 20.0,
) -> dict[str, object]:
    if minimum_support < 1:
        raise ValueError("minimum_support must be >= 1")
    if not math.isfinite(shrinkage_strength) or shrinkage_strength < 0.0:
        raise ValueError("shrinkage_strength must be finite and >= 0")
    row_path = row_report_jsonl.resolve()
    gold_path = alias_type_gold_jsonl.resolve()
    rows = _load_jsonl(row_path)
    gold = _load_jsonl(gold_path)
    row_sha = _sha256(row_path)
    outcomes: list[float] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)
    seen_train_rows: set[int] = set()
    training_bank_ids: set[str] = set()
    training_root_speaker_ids: set[str] = set()
    for index, item in enumerate(gold):
        split = str(item.get("evaluation_split", "missing")).strip().casefold() or "missing"
        split_counts[split] += 1
        if split != "train":
            continue
        if item.get("reference_kind") != "manual_alias_type_gold":
            raise ValueError(f"Gold line {index + 1} has invalid reference_kind")
        if item.get("row_report_sha256") != row_sha:
            raise ValueError(f"Gold line {index + 1} is bound to a different row report")
        row_index = _row_index(item, len(rows), f"Gold line {index + 1}")
        if row_index in seen_train_rows:
            raise ValueError(f"AliasType confidence train row_index is duplicated: {row_index}")
        seen_train_rows.add(row_index)
        training_bank_ids.add(_nonempty_text(item.get("bank_id"), f"Gold line {index + 1}.bank_id"))
        training_root_speaker_ids.add(
            _nonempty_text(item.get("root_speaker_id"), f"Gold line {index + 1}.root_speaker_id")
        )
        predicted = _alias_type(rows[row_index].get("alias_type"), f"row_report[{row_index}].alias_type")
        expected = _alias_type(item.get("alias_type"), f"Gold line {index + 1}.alias_type")
        correct = float(predicted == expected)
        outcomes.append(correct)
        grouped[_signature(predicted)].append(correct)
    if len(outcomes) < minimum_support:
        raise ValueError(f"AliasType confidence train support is insufficient: {len(outcomes)} < {minimum_support}")
    global_probability = float(np.mean(outcomes))
    entries = {}
    for key, values in sorted(grouped.items()):
        support = len(values)
        if support < minimum_support:
            continue
        empirical = float(np.mean(values))
        weight = support / (support + shrinkage_strength) if shrinkage_strength else 1.0
        entries[key] = {
            "support": support,
            "correct_count": int(sum(values)),
            "empirical_accuracy": empirical,
            "shrinkage_weight": weight,
            "calibrated_probability": weight * empirical + (1.0 - weight) * global_probability,
        }
    source_hash = hashlib.sha256(row_path.read_bytes() + b"\0" + gold_path.read_bytes()).hexdigest()
    result: dict[str, object] = {
        "schema_version": 1,
        "profile_kind": PROFILE_KIND,
        "profile_id": f"alias-type-confidence-{source_hash[:12]}",
        "row_report": str(row_path),
        "row_report_sha256": row_sha,
        "gold": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "minimum_support": minimum_support,
        "shrinkage_strength": shrinkage_strength,
        "training_splits_used": ["train"],
        "validation_labels_used": False,
        "locked_test_labels_used": False,
        "train_example_count": len(outcomes),
        "training_bank_ids": sorted(training_bank_ids),
        "training_root_speaker_ids": sorted(training_root_speaker_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "global_probability": global_probability,
        "entries": entries,
    }
    write_deterministic_json(output_json.resolve(), result)
    return result


def load_alias_type_confidence(path: Path) -> AliasTypeConfidenceProfile:
    resolved = path.resolve()
    raw = _load_mapping(resolved)
    if raw.get("schema_version") != 1 or raw.get("profile_kind") != PROFILE_KIND:
        raise ValueError(f"unsupported AliasType confidence profile: {resolved}")
    global_probability = _probability(raw.get("global_probability"), "global_probability")
    row_sha = str(raw.get("row_report_sha256", ""))
    if not _is_sha(row_sha):
        raise ValueError("AliasType confidence row_report_sha256 is invalid")
    entries = raw.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("AliasType confidence entries must be a mapping")
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ValueError("AliasType confidence entry is invalid")
        _probability(entry.get("calibrated_probability"), f"entries.{key}.calibrated_probability")
        if int(entry.get("support", 0) or 0) < 1:
            raise ValueError(f"AliasType confidence entry support is invalid: {key}")
    training_bank_ids = _nonempty_string_set(raw.get("training_bank_ids"), "training_bank_ids")
    training_root_speaker_ids = _nonempty_string_set(
        raw.get("training_root_speaker_ids"),
        "training_root_speaker_ids",
    )
    return AliasTypeConfidenceProfile(
        path=resolved,
        sha256=_sha256(resolved),
        profile_id=str(raw.get("profile_id", "")),
        row_report_sha256=row_sha,
        global_probability=global_probability,
        entries=entries,
        training_bank_ids=training_bank_ids,
        training_root_speaker_ids=training_root_speaker_ids,
    )


def predict_alias_type_confidence(
    profile: AliasTypeConfidenceProfile,
    alias_type: Mapping[str, object],
) -> tuple[float, str, int]:
    normalized = _alias_type(alias_type, "alias_type")
    key = _signature(normalized)
    entry = profile.entries.get(key)
    if isinstance(entry, Mapping):
        return (
            _probability(entry.get("calibrated_probability"), f"entries.{key}.calibrated_probability"),
            key,
            int(entry.get("support", 0) or 0),
        )
    return profile.global_probability, "global", 0


def evaluate_alias_type_confidence(
    row_report_jsonl: Path,
    alias_type_gold_jsonl: Path,
    profile_json: Path,
    output_json: Path,
    *,
    split: str = "validation",
    minimum_examples: int = 20,
    allow_locked_test: bool = False,
) -> dict[str, object]:
    normalized_split = split.strip().casefold()
    if normalized_split not in {"validation", "locked_test"}:
        raise ValueError("AliasType confidence evaluation split must be validation or locked_test")
    if normalized_split == "locked_test" and not allow_locked_test:
        raise ValueError("locked_test AliasType confidence evaluation requires explicit allow_locked_test=True")
    if minimum_examples < 1:
        raise ValueError("minimum_examples must be >= 1")
    row_path = row_report_jsonl.resolve()
    gold_path = alias_type_gold_jsonl.resolve()
    rows = _load_jsonl(row_path)
    gold = _load_jsonl(gold_path)
    profile = load_alias_type_confidence(profile_json.resolve())
    if profile.row_report_sha256 != _sha256(row_path):
        raise ValueError("AliasType confidence profile is bound to a different row report")
    outcomes = []
    raw_probabilities = []
    calibrated_probabilities = []
    result_rows = []
    seen_rows: set[int] = set()
    row_sha = _sha256(row_path)
    evaluated_bank_ids: set[str] = set()
    evaluated_root_speaker_ids: set[str] = set()
    for index, item in enumerate(gold):
        if str(item.get("evaluation_split", "")).strip().casefold() != normalized_split:
            continue
        if item.get("reference_kind") != "manual_alias_type_gold":
            raise ValueError(f"Gold line {index + 1} has invalid reference_kind")
        if item.get("row_report_sha256") != row_sha:
            raise ValueError(f"Gold line {index + 1} is bound to a different row report")
        row_index = _row_index(item, len(rows), f"Gold line {index + 1}")
        if row_index in seen_rows:
            raise ValueError(f"AliasType confidence {normalized_split} row_index is duplicated: {row_index}")
        seen_rows.add(row_index)
        bank_id = _nonempty_text(item.get("bank_id"), f"Gold line {index + 1}.bank_id")
        root_speaker_id = _nonempty_text(
            item.get("root_speaker_id"),
            f"Gold line {index + 1}.root_speaker_id",
        )
        evaluated_bank_ids.add(bank_id)
        evaluated_root_speaker_ids.add(root_speaker_id)
        predicted = _alias_type(rows[row_index].get("alias_type"), f"row_report[{row_index}].alias_type")
        expected = _alias_type(item.get("alias_type"), f"Gold line {index + 1}.alias_type")
        outcome = float(predicted == expected)
        raw_probability = _probability(
            _mapping(rows[row_index].get("alias_type"), "alias_type").get("confidence", 0.0),
            "alias_type.confidence",
        )
        calibrated, selection_key, support = predict_alias_type_confidence(profile, predicted)
        outcomes.append(outcome)
        raw_probabilities.append(raw_probability)
        calibrated_probabilities.append(calibrated)
        result_rows.append(
            {
                "row_index": row_index,
                "correct": bool(outcome),
                "raw_confidence": raw_probability,
                "calibrated_confidence": calibrated,
                "selection_key": selection_key,
                "support": support,
                "bank_id": bank_id,
                "root_speaker_id": root_speaker_id,
            }
        )
    example_count = len(outcomes)
    raw_brier = _brier(outcomes, raw_probabilities)
    calibrated_brier = _brier(outcomes, calibrated_probabilities)
    accuracy = float(np.mean(outcomes)) if outcomes else None
    leaked_bank_ids = sorted(evaluated_bank_ids & profile.training_bank_ids)
    leaked_root_speaker_ids = sorted(
        evaluated_root_speaker_ids & profile.training_root_speaker_ids
    )
    no_identity_leakage = not leaked_bank_ids and not leaked_root_speaker_ids
    passed = bool(
        example_count >= minimum_examples
        and accuracy is not None
        and accuracy > 0.95
        and calibrated_brier is not None
        and raw_brier is not None
        and calibrated_brier <= raw_brier + 1e-12
        and no_identity_leakage
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "alias_type_confidence_holdout_evaluation",
        "split": normalized_split,
        "passed": passed,
        "minimum_examples": minimum_examples,
        "example_count": example_count,
        "accuracy": accuracy,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "evaluated_bank_ids": sorted(evaluated_bank_ids),
        "evaluated_root_speaker_ids": sorted(evaluated_root_speaker_ids),
        "training_bank_ids": sorted(profile.training_bank_ids),
        "training_root_speaker_ids": sorted(profile.training_root_speaker_ids),
        "leaked_bank_ids": leaked_bank_ids,
        "leaked_root_speaker_ids": leaked_root_speaker_ids,
        "identity_leakage_count": len(leaked_bank_ids) + len(leaked_root_speaker_ids),
        "row_report": str(row_path),
        "row_report_sha256": row_sha,
        "gold": str(gold_path),
        "gold_sha256": _sha256(gold_path),
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "profile_training_row_report_sha256": profile.row_report_sha256,
        "validation_labels_used_for_fit": False,
        "locked_test_labels_used_for_fit": False,
        "release_promotion_evidence": False,
        "rows": result_rows,
    }
    write_deterministic_json(output_json.resolve(), result)
    return result


def _brier(outcomes: Sequence[float], probabilities: Sequence[float]) -> float | None:
    if not outcomes:
        return None
    return float(np.mean([(probability - outcome) ** 2 for outcome, probability in zip(outcomes, probabilities)]))


def _alias_type(value: object, path: str) -> Mapping[str, str]:
    raw = _mapping(value, path)
    fields = ("structural_role", "acoustic_subtype", "transition_type", "edge_role")
    result = {field: str(raw.get(field, "")) for field in fields}
    if any(not result[field] for field in fields):
        raise ValueError(f"{path} is incomplete")
    return result


def _signature(alias_type: Mapping[str, str]) -> str:
    return f"{alias_type['structural_role']}|{alias_type['acoustic_subtype']}"


def _row_index(value: Mapping[str, object], row_count: int, path: str) -> int:
    row_index = value.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < row_count:
        raise ValueError(f"{path} has invalid row_index")
    return row_index


def _probability(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between 0 and 1")
    return result


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _nonempty_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _nonempty_string_set(value: object, path: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    result = frozenset(_nonempty_text(item, f"{path}[]") for item in value)
    if not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        result.append(_mapping(json.loads(line), f"{path} line {line_number}"))
    if not result:
        raise ValueError(f"{path} has no rows")
    return result


def _load_mapping(path: Path) -> Mapping[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.casefold())
