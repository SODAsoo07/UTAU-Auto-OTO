from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .annotations import AnnotationDocument, AnnotationMarker, load_annotation
from .oto_io import write_lines_atomic
from .project import ProjectSpec, load_project


CONFIDENCE_PROFILE_SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "raw_confidence",
    "warning_count",
    "constraint_repair_count",
    "duration_z_abs",
    "boundary_margin",
)
TARGETS = ("error_gt_80ms", "error_gt_200ms", "one_slot_shift")
SPLITS = {"train", "validation", "locked_test"}


class ConfidenceCalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class ConfidenceExample:
    example_id: str
    speaker_id: str
    bank_id: str
    group: str
    split: str
    label_source: str
    raw_confidence: float
    warning_count: float
    constraint_repair_count: float
    duration_z_abs: float
    boundary_margin: float
    error_ms: float
    one_slot_shift: bool

    def features(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)

    def target(self, name: str) -> float:
        if name == "error_gt_80ms":
            return float(self.error_ms > 80.0)
        if name == "error_gt_200ms":
            return float(self.error_ms > 200.0)
        if name == "one_slot_shift":
            return float(self.one_slot_shift)
        raise KeyError(name)


def build_confidence_examples(
    project: ProjectSpec,
    alignment_report: Path,
    annotations_dir: Path,
    output_jsonl: Path,
) -> dict[str, object]:
    """Build recording-level calibration targets from predictions and manual Gold.

    The target error is the worst matched landmark error in a recording.  This
    intentionally favors false-clean safety over a pleasant average error.
    """
    report = json.loads(alignment_report.resolve().read_text(encoding="utf-8"))
    if not isinstance(report, Mapping) or report.get("project_sha256") != project.project_sha256:
        raise ConfidenceCalibrationError("alignment report와 project hash가 일치하지 않습니다")
    raw_rows = report.get("rows")
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise ConfidenceCalibrationError("alignment report rows가 없습니다")
    prediction_by_index = {
        int(row["recording_index"]): row
        for row in raw_rows
        if isinstance(row, Mapping) and "recording_index" in row
    }
    annotation_by_wav = {}
    for path in sorted(annotations_dir.resolve().glob("*.annotation.json")):
        annotation = load_annotation(path, project=project)
        if annotation.label_source != "manual_gold":
            continue
        if annotation.wav in annotation_by_wav:
            raise ConfidenceCalibrationError(f"manual_gold annotation이 중복되었습니다: {annotation.wav}")
        annotation_by_wav[annotation.wav] = annotation
    rows: list[dict[str, object]] = []
    skipped_missing_gold = 0
    for recording_index, recording in enumerate(project.recordings):
        annotation = annotation_by_wav.get(recording.wav)
        prediction_row = prediction_by_index.get(recording_index)
        if annotation is None:
            skipped_missing_gold += 1
            continue
        if prediction_row is None:
            raise ConfidenceCalibrationError(f"alignment report에 recording_index가 없습니다: {recording_index}")
        metadata = recording.metadata
        speaker_id = str(metadata.get("root_speaker_id") or metadata.get("speaker_id") or "").strip()
        bank_id = str(metadata.get("bank_id") or "").strip()
        split = str(metadata.get("split") or "").strip()
        if not speaker_id or not bank_id:
            raise ConfidenceCalibrationError(f"{recording.wav}: root_speaker_id와 bank_id가 필요합니다")
        if split not in SPLITS:
            raise ConfidenceCalibrationError(f"{recording.wav}: split은 train/validation/locked_test여야 합니다")
        timeline_path = Path(str(prediction_row.get("timeline", "")))
        if not timeline_path.is_absolute():
            timeline_path = alignment_report.resolve().parent / timeline_path
        timeline = json.loads(timeline_path.resolve().read_text(encoding="utf-8"))
        if not isinstance(timeline, Mapping) or timeline.get("schema_version") != 2:
            raise ConfidenceCalibrationError(f"LandmarkTimeline v2가 아닙니다: {timeline_path}")
        if timeline.get("wav_sha256") != annotation.wav_sha256:
            raise ConfidenceCalibrationError(f"timeline/annotation WAV hash가 다릅니다: {recording.wav}")
        error_ms, one_slot_shift, matched_landmarks, missing_landmarks = _recording_landmark_error(
            timeline,
            annotation.markers,
        )
        runtime_features = _timeline_runtime_features(project, timeline)
        roles = sorted({alias.role for alias in recording.aliases})
        rows.append(
            {
                "schema_version": 1,
                "example_id": f"recording_{recording_index:06d}_{annotation.annotation_id}",
                "speaker_id": speaker_id,
                "bank_id": bank_id,
                "group": f"{metadata.get('language_hint', project.language_hint)}/{'_'.join(roles) or 'unknown'}",
                "split": split,
                "label_source": "manual_gold",
                **runtime_features,
                "warning_count": runtime_features["warning_count"] + float(missing_landmarks),
                "error_ms": error_ms,
                "one_slot_shift": one_slot_shift,
                "matched_landmark_count": matched_landmarks,
                "missing_landmark_count": missing_landmarks,
            }
        )
    if not rows:
        raise ConfidenceCalibrationError("manual_gold와 prediction이 함께 있는 recording이 없습니다")
    # The trainer contract is intentionally smaller; provenance-only counts are
    # reported separately rather than written as unknown training fields.
    training_rows = [
        {key: value for key, value in row.items() if key not in {"matched_landmark_count", "missing_landmark_count"}}
        for row in rows
    ]
    _validate_speaker_splits(tuple(_example_from_mapping(row) for row in training_rows))
    write_lines_atomic(
        output_jsonl.resolve(),
        (json.dumps(row, ensure_ascii=False, sort_keys=True) for row in training_rows),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_jsonl.exists(),
    )
    return {
        "examples": str(output_jsonl.resolve()),
        "examples_sha256": _sha256_file(output_jsonl),
        "example_count": len(rows),
        "skipped_missing_gold": skipped_missing_gold,
        "matched_landmarks": sum(int(row["matched_landmark_count"]) for row in rows),
        "missing_landmarks": sum(int(row["missing_landmark_count"]) for row in rows),
        "split_counts": _counts(str(row["split"]) for row in rows),
        "label_source": "manual_gold",
        "target_aggregation": "maximum_absolute_landmark_error_per_recording",
    }


def build_confidence_examples_from_candidate_items(
    candidate_items_csv: Path,
    output_jsonl: Path,
) -> dict[str, object]:
    """Build one global calibration set from a multi-project candidate manifest.

    ``align-gold-manifest`` writes one row per recording with explicit project,
    annotation, and timeline paths.  Consuming that manifest avoids weakening
    the speaker-split check by fitting separate profiles for each inventory.
    Every path and hash is verified before a training row is emitted.
    """
    source_path = candidate_items_csv.resolve()
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    required = {
        "recording_id",
        "bank_id",
        "root_speaker_id",
        "split",
        "recording_index",
        "project",
        "project_sha256",
        "annotation",
        "annotation_sha256",
        "timeline",
        "timeline_sha256",
        "wav_sha256",
    }
    missing_columns = sorted(required - fieldnames)
    if missing_columns:
        raise ConfidenceCalibrationError(
            f"candidate items CSV missing columns: {', '.join(missing_columns)}"
        )
    if not raw_rows:
        raise ConfidenceCalibrationError("candidate items CSV has no rows")

    project_cache: dict[Path, ProjectSpec] = {}
    rows: list[dict[str, object]] = []
    seen_examples: set[str] = set()
    seen_recordings: set[tuple[Path, int]] = set()
    matched_landmarks = 0
    missing_landmarks = 0
    project_counts: dict[str, int] = {}
    for csv_row_number, raw in enumerate(raw_rows, 2):
        project_path = _manifest_path(raw["project"], source_path.parent, f"row {csv_row_number}.project")
        project = project_cache.get(project_path)
        if project is None:
            project = load_project(project_path)
            project_cache[project_path] = project
        if raw["project_sha256"].strip() != project.project_sha256:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: project hash mismatch")
        try:
            recording_index = int(raw["recording_index"])
        except (TypeError, ValueError) as exc:
            raise ConfidenceCalibrationError(
                f"row {csv_row_number}: recording_index must be an integer"
            ) from exc
        if recording_index < 0 or recording_index >= len(project.recordings):
            raise ConfidenceCalibrationError(f"row {csv_row_number}: recording_index out of range")
        recording_key = (project_path, recording_index)
        if recording_key in seen_recordings:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: duplicate project recording")
        seen_recordings.add(recording_key)
        recording = project.recordings[recording_index]
        metadata = recording.metadata
        speaker_id = str(metadata.get("root_speaker_id") or metadata.get("speaker_id") or "").strip()
        bank_id = str(metadata.get("bank_id") or "").strip()
        split = str(metadata.get("split") or "").strip()
        for column, actual in (
            ("root_speaker_id", speaker_id),
            ("bank_id", bank_id),
            ("split", split),
        ):
            if raw[column].strip() != actual:
                raise ConfidenceCalibrationError(f"row {csv_row_number}: {column} mismatch")
        if split not in SPLITS:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: unsupported split {split!r}")

        annotation_path = _manifest_path(
            raw["annotation"], source_path.parent, f"row {csv_row_number}.annotation"
        )
        timeline_path = _manifest_path(
            raw["timeline"], source_path.parent, f"row {csv_row_number}.timeline"
        )
        if _sha256_file(annotation_path) != raw["annotation_sha256"].strip():
            raise ConfidenceCalibrationError(f"row {csv_row_number}: annotation hash mismatch")
        if _sha256_file(timeline_path) != raw["timeline_sha256"].strip():
            raise ConfidenceCalibrationError(f"row {csv_row_number}: timeline hash mismatch")
        annotation = load_annotation(annotation_path, project=project)
        if annotation.label_source != "manual_gold":
            raise ConfidenceCalibrationError(f"row {csv_row_number}: annotation is not manual_gold")
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        if not isinstance(timeline, Mapping) or timeline.get("schema_version") != 2:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: expected LandmarkTimeline v2")
        expected_wav_sha256 = raw["wav_sha256"].strip()
        if annotation.wav_sha256 != expected_wav_sha256 or timeline.get("wav_sha256") != expected_wav_sha256:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: WAV hash mismatch")

        recording_id = raw["recording_id"].strip()
        if not recording_id:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: recording_id is empty")
        example_id = f"recording_{recording_id}"
        if example_id in seen_examples:
            raise ConfidenceCalibrationError(f"row {csv_row_number}: duplicate recording_id {recording_id}")
        seen_examples.add(example_id)
        row, matched, missing = _build_confidence_row(
            project,
            recording_index,
            annotation,
            timeline,
            example_id=example_id,
        )
        rows.append(row)
        matched_landmarks += matched
        missing_landmarks += missing
        project_counts[project.project_sha256] = project_counts.get(project.project_sha256, 0) + 1

    examples = tuple(_example_from_mapping(row) for row in rows)
    _validate_speaker_splits(examples)
    write_lines_atomic(
        output_jsonl.resolve(),
        (json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_jsonl.exists(),
    )
    return {
        "examples": str(output_jsonl.resolve()),
        "examples_sha256": _sha256_file(output_jsonl),
        "example_count": len(rows),
        "matched_landmarks": matched_landmarks,
        "missing_landmarks": missing_landmarks,
        "split_counts": _counts(example.split for example in examples),
        "speaker_counts_by_split": {
            split_name: len({example.speaker_id for example in examples if example.split == split_name})
            for split_name in sorted(SPLITS)
        },
        "project_counts": dict(sorted(project_counts.items())),
        "label_source": "manual_gold",
        "target_aggregation": "maximum_absolute_landmark_error_per_recording",
        "source_candidate_items_csv": str(source_path),
        "source_candidate_items_sha256": _sha256_file(source_path),
    }


def _build_confidence_row(
    project: ProjectSpec,
    recording_index: int,
    annotation: AnnotationDocument,
    timeline: Mapping[str, object],
    *,
    example_id: str,
) -> tuple[dict[str, object], int, int]:
    recording = project.recordings[recording_index]
    metadata = recording.metadata
    speaker_id = str(metadata.get("root_speaker_id") or metadata.get("speaker_id") or "").strip()
    bank_id = str(metadata.get("bank_id") or "").strip()
    split = str(metadata.get("split") or "").strip()
    if not speaker_id or not bank_id:
        raise ConfidenceCalibrationError(f"{recording.wav}: root_speaker_id and bank_id are required")
    if split not in SPLITS:
        raise ConfidenceCalibrationError(
            f"{recording.wav}: split must be train, validation, or locked_test"
        )
    if annotation.wav != recording.wav:
        raise ConfidenceCalibrationError(f"{recording.wav}: annotation WAV path mismatch")
    if timeline.get("wav_sha256") != annotation.wav_sha256:
        raise ConfidenceCalibrationError(f"{recording.wav}: timeline/annotation WAV hash mismatch")
    error_ms, one_slot_shift, matched_landmarks, missing_landmarks = _recording_landmark_error(
        timeline,
        annotation.markers,
    )
    runtime_features = _timeline_runtime_features(project, timeline)
    roles = sorted({alias.role for alias in recording.aliases})
    return (
        {
            "schema_version": 1,
            "example_id": example_id,
            "speaker_id": speaker_id,
            "bank_id": bank_id,
            "group": f"{metadata.get('language_hint', project.language_hint)}/{'_'.join(roles) or 'unknown'}",
            "split": split,
            "label_source": "manual_gold",
            **runtime_features,
            "warning_count": runtime_features["warning_count"] + float(missing_landmarks),
            "error_ms": error_ms,
            "one_slot_shift": one_slot_shift,
        },
        matched_landmarks,
        missing_landmarks,
    )


def _manifest_path(value: object, base: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ConfidenceCalibrationError(f"{label}: path is empty")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise ConfidenceCalibrationError(f"{label}: file does not exist: {resolved}")
    return resolved


def load_confidence_examples(path: Path) -> tuple[ConfidenceExample, ...]:
    examples: list[ConfidenceExample] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.resolve().read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ConfidenceCalibrationError(f"line {line_number}: JSON 오류: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfidenceCalibrationError(f"line {line_number}: mapping이어야 합니다")
        allowed = {
            "schema_version",
            "example_id",
            "speaker_id",
            "bank_id",
            "group",
            "split",
            "label_source",
            *FEATURE_NAMES,
            "error_ms",
            "one_slot_shift",
        }
        unknown = sorted(set(raw) - allowed)
        missing = sorted(allowed - set(raw))
        if unknown or missing:
            raise ConfidenceCalibrationError(
                f"line {line_number}: unknown={unknown}, missing={missing}"
            )
        if raw["schema_version"] != 1:
            raise ConfidenceCalibrationError(f"line {line_number}: schema_version=1이 필요합니다")
        example_id = _string(raw["example_id"], f"line {line_number}.example_id")
        if example_id in seen_ids:
            raise ConfidenceCalibrationError(f"line {line_number}: 중복 example_id입니다: {example_id}")
        seen_ids.add(example_id)
        split = _choice(raw["split"], SPLITS, f"line {line_number}.split")
        label_source = _string(raw["label_source"], f"line {line_number}.label_source")
        if label_source != "manual_gold":
            raise ConfidenceCalibrationError(
                f"line {line_number}: confidence 정답은 manual_gold만 허용합니다"
            )
        examples.append(
            ConfidenceExample(
                example_id=example_id,
                speaker_id=_string(raw["speaker_id"], f"line {line_number}.speaker_id"),
                bank_id=_string(raw["bank_id"], f"line {line_number}.bank_id"),
                group=_string(raw["group"], f"line {line_number}.group"),
                split=split,
                label_source=label_source,
                raw_confidence=_number(raw["raw_confidence"], f"line {line_number}.raw_confidence", 0.0, 1.0),
                warning_count=_number(raw["warning_count"], f"line {line_number}.warning_count", 0.0, None),
                constraint_repair_count=_number(raw["constraint_repair_count"], f"line {line_number}.constraint_repair_count", 0.0, None),
                duration_z_abs=_number(raw["duration_z_abs"], f"line {line_number}.duration_z_abs", 0.0, None),
                boundary_margin=_number(raw["boundary_margin"], f"line {line_number}.boundary_margin", -100.0, 100.0),
                error_ms=_number(raw["error_ms"], f"line {line_number}.error_ms", 0.0, None),
                one_slot_shift=_boolean(raw["one_slot_shift"], f"line {line_number}.one_slot_shift"),
            )
        )
    if not examples:
        raise ConfidenceCalibrationError("confidence calibration example이 없습니다")
    _validate_speaker_splits(examples)
    return tuple(examples)


def fit_confidence_calibration(
    examples_path: Path,
    output_json: Path,
    *,
    minimum_train_examples: int = 100,
    minimum_validation_examples: int = 20,
    l2: float = 0.05,
) -> dict[str, object]:
    examples = load_confidence_examples(examples_path)
    train = tuple(example for example in examples if example.split == "train")
    validation = tuple(example for example in examples if example.split == "validation")
    locked = tuple(example for example in examples if example.split == "locked_test")
    if len(train) < minimum_train_examples:
        raise ConfidenceCalibrationError(
            f"train example이 부족합니다: {len(train)} < {minimum_train_examples}"
        )
    if len(validation) < minimum_validation_examples:
        raise ConfidenceCalibrationError(
            f"validation example이 부족합니다: {len(validation)} < {minimum_validation_examples}"
        )
    x_train = np.asarray([example.features() for example in train], dtype=np.float64)
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    x_train_normalized = (x_train - means) / scales
    target_profiles: dict[str, object] = {}
    for target in TARGETS:
        y_train = np.asarray([example.target(target) for example in train], dtype=np.float64)
        positive_count = int(np.sum(y_train))
        negative_count = len(y_train) - positive_count
        if positive_count < 10 or negative_count < 10:
            raise ConfidenceCalibrationError(
                f"{target}: train positive/negative가 각각 10개 이상 필요합니다 "
                f"(positive={positive_count}, negative={negative_count})"
            )
        coefficients = _fit_logistic(x_train_normalized, y_train, l2=l2)
        target_profiles[target] = {
            "coefficients": coefficients.tolist(),
            "train": _risk_metrics(train, target, coefficients, means, scales),
            "validation": _risk_metrics(validation, target, coefficients, means, scales),
            "locked_test": _risk_metrics(locked, target, coefficients, means, scales) if locked else None,
            "train_positive_count": positive_count,
            "train_negative_count": negative_count,
        }
    source_sha256 = _sha256_file(examples_path)
    profile = {
        "schema_version": CONFIDENCE_PROFILE_SCHEMA_VERSION,
        "profile_kind": "landmark_risk_platt_v1",
        "feature_names": list(FEATURE_NAMES),
        "normalization": {"mean": means.tolist(), "scale": scales.tolist()},
        "targets": target_profiles,
        "training": {
            "source": str(examples_path.resolve()),
            "source_sha256": source_sha256,
            "label_source": "manual_gold",
            "split_unit": "root_speaker_id",
            "train_examples": len(train),
            "validation_examples": len(validation),
            "locked_test_examples": len(locked),
            "group_counts": _counts(example.group for example in examples),
            "speaker_counts_by_split": {
                split: len({example.speaker_id for example in examples if example.split == split})
                for split in sorted(SPLITS)
            },
            "l2": l2,
        },
    }
    text = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        output_json.resolve(),
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_json.exists(),
    )
    return {
        "profile": str(output_json.resolve()),
        "profile_sha256": _sha256_file(output_json),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "locked_test_examples": len(locked),
        "targets": target_profiles,
    }


def audit_confidence_calibration_readiness(
    examples_path: Path,
    output_json: Path,
    *,
    minimum_train_examples: int = 100,
    minimum_validation_examples: int = 20,
    minimum_train_class_examples: int = 10,
) -> dict[str, object]:
    """Write a non-mutating readiness decision before confidence fitting."""
    examples = load_confidence_examples(examples_path)
    train = tuple(example for example in examples if example.split == "train")
    validation = tuple(example for example in examples if example.split == "validation")
    locked = tuple(example for example in examples if example.split == "locked_test")
    blockers: list[str] = []
    if len(train) < minimum_train_examples:
        blockers.append(f"train_examples:{len(train)}<{minimum_train_examples}")
    if len(validation) < minimum_validation_examples:
        blockers.append(f"validation_examples:{len(validation)}<{minimum_validation_examples}")
    target_counts: dict[str, object] = {}
    for target in TARGETS:
        train_positive = sum(int(example.target(target)) for example in train)
        train_negative = len(train) - train_positive
        validation_positive = sum(int(example.target(target)) for example in validation)
        validation_negative = len(validation) - validation_positive
        if train_positive < minimum_train_class_examples:
            blockers.append(
                f"{target}.train_positive:{train_positive}<{minimum_train_class_examples}"
            )
        if train_negative < minimum_train_class_examples:
            blockers.append(
                f"{target}.train_negative:{train_negative}<{minimum_train_class_examples}"
            )
        target_counts[target] = {
            "train_positive": train_positive,
            "train_negative": train_negative,
            "validation_positive": validation_positive,
            "validation_negative": validation_negative,
        }
    payload = {
        "schema_version": 1,
        "purpose": "confidence_calibration_readiness",
        "ready_to_fit": not blockers,
        "blockers": blockers,
        "requirements": {
            "minimum_train_examples": minimum_train_examples,
            "minimum_validation_examples": minimum_validation_examples,
            "minimum_train_positive_and_negative_per_target": minimum_train_class_examples,
        },
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "locked_test_excluded_from_fit": len(locked),
        },
        "target_counts": target_counts,
        "speaker_counts_by_split": {
            split_name: len({example.speaker_id for example in examples if example.split == split_name})
            for split_name in sorted(SPLITS)
        },
        "source_examples": str(examples_path.resolve()),
        "source_examples_sha256": _sha256_file(examples_path),
        "fit_performed": False,
        "release_promotion_evidence": False,
        "recommended_action": (
            "fit the train-only confidence profile and evaluate on validation"
            if not blockers
            else "improve development alignment and expand speaker-isolated manual Gold; do not tune on locked_test"
        ),
    }
    write_lines_atomic(
        output_json.resolve(),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_json.exists(),
    )
    return {
        **payload,
        "report": str(output_json.resolve()),
        "report_sha256": _sha256_file(output_json),
    }


def load_confidence_profile(path: Path) -> Mapping[str, object]:
    profile = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise ConfidenceCalibrationError("confidence profile root는 mapping이어야 합니다")
    if profile.get("schema_version") != CONFIDENCE_PROFILE_SCHEMA_VERSION:
        raise ConfidenceCalibrationError("지원하지 않는 confidence profile schema_version입니다")
    if profile.get("profile_kind") != "landmark_risk_platt_v1":
        raise ConfidenceCalibrationError("지원하지 않는 confidence profile_kind입니다")
    if tuple(profile.get("feature_names", ())) != FEATURE_NAMES:
        raise ConfidenceCalibrationError("confidence profile feature_names가 runtime과 다릅니다")
    normalization = profile.get("normalization")
    targets = profile.get("targets")
    if not isinstance(normalization, Mapping) or not isinstance(targets, Mapping):
        raise ConfidenceCalibrationError("confidence profile normalization/targets가 없습니다")
    for target in TARGETS:
        if target not in targets:
            raise ConfidenceCalibrationError(f"confidence profile target이 없습니다: {target}")
    return profile


def predict_risks(
    profile: Mapping[str, object],
    features: Mapping[str, float],
) -> dict[str, float]:
    if tuple(profile.get("feature_names", ())) != FEATURE_NAMES:
        raise ConfidenceCalibrationError("confidence profile feature_names가 runtime과 다릅니다")
    vector = np.asarray([float(features[name]) for name in FEATURE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ConfidenceCalibrationError("confidence runtime feature가 finite가 아닙니다")
    normalization = profile["normalization"]
    means = np.asarray(normalization["mean"], dtype=np.float64)
    scales = np.asarray(normalization["scale"], dtype=np.float64)
    normalized = (vector - means) / scales
    targets = profile["targets"]
    result: dict[str, float] = {}
    for target in TARGETS:
        coefficients = np.asarray(targets[target]["coefficients"], dtype=np.float64)
        result[target] = float(_sigmoid(coefficients[0] + normalized @ coefficients[1:]))
    return result


def runtime_confidence_features(
    *,
    raw_confidence: float,
    warning_count: int,
    constraint_repair_count: int = 0,
    duration_z_abs: float = 0.0,
    boundary_margin: float = 0.0,
) -> dict[str, float]:
    return {
        "raw_confidence": float(raw_confidence),
        "warning_count": float(warning_count),
        "constraint_repair_count": float(constraint_repair_count),
        "duration_z_abs": float(duration_z_abs),
        "boundary_margin": float(boundary_margin),
    }


def _recording_landmark_error(
    timeline: Mapping[str, object],
    gold_markers: Sequence[AnnotationMarker],
) -> tuple[float, bool, int, int]:
    raw_landmarks = timeline.get("landmarks", [])
    if isinstance(raw_landmarks, (str, bytes)) or not isinstance(raw_landmarks, Sequence):
        raise ConfidenceCalibrationError("timeline.landmarks가 없습니다")
    predictions = [item for item in raw_landmarks if isinstance(item, Mapping)]
    errors: list[float] = []
    missing = 0
    one_slot_shift = False
    for marker in gold_markers:
        if marker.label in {"usable_start", "usable_end"}:
            continue
        phone_index = marker.right_phone_index if marker.label == "phone_boundary" else marker.phone_index
        if phone_index is None:
            continue
        same_label = [item for item in predictions if str(item.get("label", "")) == marker.label]
        correct = [item for item in same_label if item.get("phone_index") == phone_index]
        if not correct:
            missing += 1
            continue
        correct_error = min(abs(float(item.get("time_ms", 0.0)) - marker.time_ms) for item in correct)
        errors.append(correct_error)
        adjacent = [
            item
            for item in same_label
            if isinstance(item.get("phone_index"), int)
            and abs(int(item["phone_index"]) - phone_index) == 1
        ]
        if adjacent:
            adjacent_error = min(abs(float(item.get("time_ms", 0.0)) - marker.time_ms) for item in adjacent)
            if adjacent_error + 20.0 < correct_error and adjacent_error <= 80.0:
                one_slot_shift = True
    duration_ms = float(timeline.get("duration_ms", 0.0) or 0.0)
    if missing:
        errors.append(max(200.001, duration_ms))
    if not errors:
        raise ConfidenceCalibrationError("confidence target으로 비교할 수동 landmark가 없습니다")
    return max(errors), one_slot_shift, len(errors) - (1 if missing else 0), missing


def _timeline_runtime_features(project: ProjectSpec, timeline: Mapping[str, object]) -> dict[str, float]:
    raw_phone_segments = timeline.get("phone_segments", [])
    if isinstance(raw_phone_segments, (str, bytes)) or not isinstance(raw_phone_segments, Sequence):
        raise ConfidenceCalibrationError("timeline.phone_segments가 없습니다")
    duration_z_abs = 0.0
    state_confidences: list[float] = []
    for raw_phone in raw_phone_segments:
        if not isinstance(raw_phone, Mapping):
            continue
        phone_id = str(raw_phone.get("phone_id", ""))
        definition = project.inventory.get(phone_id)
        if definition is None:
            raise ConfidenceCalibrationError(f"timeline phone이 inventory에 없습니다: {phone_id}")
        definitions = {state.id: state for state in definition.states}
        raw_states = raw_phone.get("state_segments", [])
        if isinstance(raw_states, (str, bytes)) or not isinstance(raw_states, Sequence):
            continue
        for raw_state in raw_states:
            if not isinstance(raw_state, Mapping):
                continue
            confidence = float(raw_state.get("confidence", 0.0) or 0.0)
            state_confidences.append(confidence)
            state_definition = definitions.get(str(raw_state.get("state_id", "")))
            if state_definition is None:
                continue
            mode = state_definition.mode_ms
            if mode is None:
                mode = state_definition.min_ms + 0.5 * (state_definition.max_ms - state_definition.min_ms)
            sigma = max(10.0, (state_definition.max_ms - state_definition.min_ms) / 2.5)
            duration = float(raw_state.get("end_ms", 0.0)) - float(raw_state.get("start_ms", 0.0))
            duration_z_abs = max(duration_z_abs, abs((duration - mode) / sigma))
    raw_warnings = timeline.get("warnings", [])
    warning_count = len(raw_warnings) if isinstance(raw_warnings, Sequence) and not isinstance(raw_warnings, (str, bytes)) else 0
    raw_confidence = float(timeline.get("confidence", 0.0) or 0.0)
    min_state_confidence = min(state_confidences, default=raw_confidence)
    return runtime_confidence_features(
        raw_confidence=raw_confidence,
        warning_count=warning_count,
        duration_z_abs=duration_z_abs,
        boundary_margin=min_state_confidence - 0.5,
    )


def _example_from_mapping(raw: Mapping[str, object]) -> ConfidenceExample:
    return ConfidenceExample(
        example_id=str(raw["example_id"]),
        speaker_id=str(raw["speaker_id"]),
        bank_id=str(raw["bank_id"]),
        group=str(raw["group"]),
        split=str(raw["split"]),
        label_source=str(raw["label_source"]),
        raw_confidence=float(raw["raw_confidence"]),
        warning_count=float(raw["warning_count"]),
        constraint_repair_count=float(raw["constraint_repair_count"]),
        duration_z_abs=float(raw["duration_z_abs"]),
        boundary_margin=float(raw["boundary_margin"]),
        error_ms=float(raw["error_ms"]),
        one_slot_shift=bool(raw["one_slot_shift"]),
    )


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ coefficients
        probabilities = _sigmoid(logits)
        eps = 1e-9
        loss = -float(np.mean(y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps)))
        penalty = 0.5 * l2 * float(np.sum(coefficients[1:] ** 2))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[1:] += l2 * coefficients[1:]
        return loss + penalty, gradient

    initial = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    initial[0] = np.log(prevalence / (1.0 - prevalence))
    result = minimize(
        lambda coefficients: objective(coefficients)[0],
        initial,
        jac=lambda coefficients: objective(coefficients)[1],
        method="L-BFGS-B",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ConfidenceCalibrationError(f"confidence logistic fitting 실패: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _risk_metrics(
    examples: Sequence[ConfidenceExample],
    target: str,
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> dict[str, float | int]:
    if not examples:
        return {"count": 0, "positive_count": 0, "brier": 0.0, "ece": 0.0}
    x = np.asarray([example.features() for example in examples], dtype=np.float64)
    y = np.asarray([example.target(target) for example in examples], dtype=np.float64)
    normalized = (x - means) / scales
    probabilities = _sigmoid(coefficients[0] + normalized @ coefficients[1:])
    return {
        "count": len(examples),
        "positive_count": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "mean_probability": float(np.mean(probabilities)),
        "brier": float(np.mean((probabilities - y) ** 2)),
        "ece": _expected_calibration_error(probabilities, y),
    }


def _expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    total = len(probabilities)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        count = int(np.sum(selected))
        if count:
            error += count / total * abs(float(np.mean(probabilities[selected])) - float(np.mean(targets[selected])))
    return float(error)


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _validate_speaker_splits(examples: Sequence[ConfidenceExample]) -> None:
    speaker_splits: dict[str, set[str]] = {}
    bank_splits: dict[str, set[str]] = {}
    for example in examples:
        speaker_splits.setdefault(example.speaker_id, set()).add(example.split)
        bank_splits.setdefault(example.bank_id, set()).add(example.split)
    leaked_speakers = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    leaked_banks = sorted(bank for bank, splits in bank_splits.items() if len(splits) > 1)
    if leaked_speakers:
        raise ConfidenceCalibrationError(f"root speaker split leakage: {', '.join(leaked_speakers[:10])}")
    if leaked_banks:
        raise ConfidenceCalibrationError(f"bank split leakage: {', '.join(leaked_banks[:10])}")


def _counts(values: Sequence[str] | object) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfidenceCalibrationError(f"{path}: 비어 있지 않은 문자열이어야 합니다")
    return value.strip()


def _choice(value: object, choices: set[str], path: str) -> str:
    text = _string(value, path)
    if text not in choices:
        raise ConfidenceCalibrationError(f"{path}: 지원값은 {', '.join(sorted(choices))}입니다")
    return text


def _number(value: object, path: str, minimum: float, maximum: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfidenceCalibrationError(f"{path}: 숫자여야 합니다")
    number = float(value)
    if not np.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise ConfidenceCalibrationError(f"{path}: 범위 밖 숫자입니다")
    return number


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfidenceCalibrationError(f"{path}: boolean이어야 합니다")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
