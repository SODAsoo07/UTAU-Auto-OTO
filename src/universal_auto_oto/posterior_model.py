from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .acoustic_evidence import AcousticEvidence, SUPPORTED_FEATURE_BACKENDS, extract_acoustic_evidence
from .annotations import AnnotationDocument, load_annotation
from .audio import file_sha256
from .oto_io import write_lines_atomic
from .project import ProjectSpec, load_project


POSTERIOR_MODEL_SCHEMA_VERSION = 1
POSTERIOR_MODEL_KIND = "manual_gold_acoustic_posterior_v1"
POSTERIOR_FEATURE_NAMES = (
    "energy",
    "flux",
    "periodicity",
    "high_band_energy",
    "zcr",
    "spectral_stability",
    "energy_rise",
    "periodicity_rise",
)
POSTERIOR_HEADS = (
    "boundary",
    "transition",
    "vowel_like",
    "consonant_like",
    "periodicity",
    "aperiodic",
)
VALID_SPLITS = {"train", "validation", "locked_test"}
SPECIAL_MANUAL_PHONE_TARGET_POLICY = {
    "sil": "neutral_for_phone_class_and_voicing_heads",
    "vf": "vocal_fry_neutral_until_dedicated_head_has_sufficient_gold",
    "exh": "vowel_final_exhalation_filename_uppercase_R_aperiodic_positive",
}


class PosteriorModelError(ValueError):
    pass


def audit_posterior_training_manifest(
    items_csv: Path,
    *,
    output_json: Path | None = None,
    minimum_train_recordings: int = 20,
    minimum_validation_recordings: int = 5,
) -> dict[str, object]:
    """Report Gold/split/head readiness without training or promoting labels."""

    items_path = items_csv.resolve()
    with items_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PosteriorModelError(f"training items CSV is empty: {items_path}")
    split_counts = {split: 0 for split in sorted(VALID_SPLITS)}
    gold_split_counts = {split: 0 for split in sorted(VALID_SPLITS)}
    scope_counts: dict[str, int] = {}
    head_recording_support = {head: 0 for head in POSTERIOR_HEADS}
    missing_speaker: list[str] = []
    invalid_split: list[str] = []
    not_gold: list[str] = []
    invalid_items: list[dict[str, str]] = []
    speaker_splits: dict[str, set[str]] = {}
    bank_splits: dict[str, set[str]] = {}
    projects: dict[Path, ProjectSpec] = {}
    feature_backend_counts: dict[str, int] = {}
    manual_gold_backend_split_counts: dict[str, dict[str, int]] = {}
    gold_count = 0
    for row in rows:
        bank_id = str(row.get("bank_id", "")).strip()
        recording_index_text = str(row.get("recording_index", "")).strip()
        item_id = f"{bank_id}/{recording_index_text}"
        speaker = str(row.get("root_speaker_id", "")).strip()
        split = str(row.get("split") or row.get("evaluation_split") or "").strip()
        if not speaker:
            missing_speaker.append(item_id)
        if split not in VALID_SPLITS:
            invalid_split.append(item_id)
        else:
            split_counts[split] += 1
            if speaker:
                speaker_splits.setdefault(speaker, set()).add(split)
            if bank_id:
                bank_splits.setdefault(bank_id, set()).add(split)
        try:
            project_path = Path(str(row.get("project", "")).strip()).resolve()
            annotation_path = Path(str(row.get("annotation", "")).strip()).resolve()
            project = projects.get(project_path)
            if project is None:
                project = load_project(project_path)
                projects[project_path] = project
            expected_project_hash = str(row.get("project_sha256", "")).strip()
            if expected_project_hash and expected_project_hash != project.project_sha256:
                raise PosteriorModelError("project semantic hash mismatch")
            expected_annotation_hash = str(row.get("annotation_sha256", "")).strip()
            if expected_annotation_hash and expected_annotation_hash != file_sha256(annotation_path):
                raise PosteriorModelError("annotation file hash mismatch")
            annotation = load_annotation(annotation_path)
            recording_index = int(recording_index_text)
            if not 0 <= recording_index < len(project.recordings):
                raise PosteriorModelError("recording_index outside project")
            recording = project.recordings[recording_index]
            if annotation.inventory_sha256 != project.inventory_sha256:
                raise PosteriorModelError("annotation inventory hash mismatch")
            if recording.wav != annotation.wav or file_sha256(project.wav_path(recording)) != annotation.wav_sha256:
                raise PosteriorModelError("recording and annotation WAV mismatch")
        except Exception as error:
            invalid_items.append({"item": item_id, "error": str(error)})
            continue
        feature_backend = project.alignment.feature_backend
        feature_backend_counts[feature_backend] = feature_backend_counts.get(feature_backend, 0) + 1
        if annotation.label_source != "manual_gold":
            not_gold.append(item_id)
            continue
        gold_count += 1
        if split in VALID_SPLITS:
            gold_split_counts[split] += 1
            backend_splits = manual_gold_backend_split_counts.setdefault(
                feature_backend, {name: 0 for name in sorted(VALID_SPLITS)}
            )
            backend_splits[split] += 1
        scope = _annotation_scope(annotation)
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        for head in POSTERIOR_HEADS:
            if head != "transition" or any(
                marker.label in {"vowel_transition", "subphone_transition", "voicing_onset"}
                for marker in annotation.markers
            ):
                head_recording_support[head] += 1

    leaking_speakers = sorted(key for key, splits in speaker_splits.items() if len(splits) > 1)
    leaking_banks = sorted(key for key, splits in bank_splits.items() if len(splits) > 1)
    blockers = {
        "invalid_items": invalid_items,
        "not_manual_gold_count": len(not_gold),
        "not_manual_gold_items": not_gold,
        "missing_root_speaker_count": len(missing_speaker),
        "missing_root_speaker_items": missing_speaker,
        "invalid_split_count": len(invalid_split),
        "invalid_split_items": invalid_split,
        "train_shortfall": max(0, minimum_train_recordings - gold_split_counts["train"]),
        "validation_shortfall": max(0, minimum_validation_recordings - gold_split_counts["validation"]),
        "speaker_leakage": leaking_speakers,
        "bank_leakage": leaking_banks,
    }
    ready = not any(
        (
            invalid_items,
            not_gold,
            missing_speaker,
            invalid_split,
            blockers["train_shortfall"],
            blockers["validation_shortfall"],
            leaking_speakers,
            leaking_banks,
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "posterior_training_readiness_and_phone_gold_head_coverage",
        "items_csv": str(items_path),
        "items_csv_sha256": file_sha256(items_path),
        "item_count": len(rows),
        "manual_gold_count": gold_count,
        "split_counts": split_counts,
        "manual_gold_split_counts": gold_split_counts,
        "feature_backend_counts": dict(sorted(feature_backend_counts.items())),
        "manual_gold_backend_split_counts": dict(sorted(manual_gold_backend_split_counts.items())),
        "annotation_scope_counts": dict(sorted(scope_counts.items())),
        "head_recording_support": head_recording_support,
        "expected_runtime_fallback_heads": [
            head for head, count in head_recording_support.items() if count == 0
        ],
        "special_manual_phone_target_policy": SPECIAL_MANUAL_PHONE_TARGET_POLICY,
        "minimum_train_recordings": minimum_train_recordings,
        "minimum_validation_recordings": minimum_validation_recordings,
        "blockers": blockers,
        "ready_for_training": ready,
        "automatic_gold_created": False,
        "note": (
            "Transition head has manually confirmed transition-marker support."
            if head_recording_support["transition"]
            else "Phone-only Gold trains boundary and phone-class heads; transition remains heuristic without manual transition markers."
        ),
    }
    if output_json is not None:
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        write_lines_atomic(
            output_json.resolve(),
            text.splitlines(),
            encoding="utf-8",
            newline="\n",
            backup_existing=output_json.exists(),
        )
        report["output_json"] = str(output_json.resolve())
    return report


@dataclass(frozen=True)
class PosteriorTrainingItem:
    project: ProjectSpec
    annotation_path: Path
    annotation: AnnotationDocument
    recording_index: int
    speaker_id: str
    bank_id: str
    split: str


def train_posterior_model(
    project: ProjectSpec,
    annotations_dir: Path,
    output_model: Path,
    *,
    minimum_train_recordings: int = 20,
    minimum_validation_recordings: int = 5,
    l2: float = 0.05,
    posterior_blend: float = 0.75,
) -> dict[str, object]:
    if not 0.0 < posterior_blend <= 1.0:
        raise PosteriorModelError("posterior_blend는 0보다 크고 1 이하여야 합니다")
    items = _load_training_items(project, annotations_dir)
    train_items = tuple(item for item in items if item.split == "train")
    validation_items = tuple(item for item in items if item.split == "validation")
    locked_items = tuple(item for item in items if item.split == "locked_test")
    if len(train_items) < minimum_train_recordings:
        raise PosteriorModelError(
            f"manual_gold train 녹음이 부족합니다: {len(train_items)} < {minimum_train_recordings}"
        )
    if len(validation_items) < minimum_validation_recordings:
        raise PosteriorModelError(
            f"manual_gold validation 녹음이 부족합니다: {len(validation_items)} < {minimum_validation_recordings}"
        )
    _validate_speaker_splits(items)

    split_arrays: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    item_groups = {
        "train": train_items,
        "validation": validation_items,
        "locked_test": locked_items,
    }
    for split, split_items in item_groups.items():
        if split_items:
            split_arrays[split] = _compile_examples(project, split_items)
    x_train, y_train = split_arrays["train"]
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    normalized_train = (x_train - means) / scales
    heads: dict[str, object] = {}
    for head in POSTERIOR_HEADS:
        target = y_train[head]
        labelled = np.isfinite(target)
        labelled_target = target[labelled]
        if not np.any(labelled):
            heads[head] = {
                "trained": False,
                "reason": "manual_gold_head_unlabelled",
                "coefficients": np.zeros(len(POSTERIOR_FEATURE_NAMES) + 1, dtype=np.float64).tolist(),
                "train": None,
                "validation": None,
                "locked_test": None,
                "train_labelled_frames": 0,
                "train_positive_frames": 0,
                "train_negative_frames": 0,
            }
            continue
        labelled_train = normalized_train[labelled]
        positives = int(np.sum(labelled_target >= 0.5))
        negatives = len(labelled_target) - positives
        if positives < 20 or negatives < 20:
            raise PosteriorModelError(
                f"{head}: 학습 positive/negative frame이 부족합니다 (positive={positives}, negative={negatives})"
            )
        coefficients = _fit_logistic(labelled_train, labelled_target, l2=l2)
        heads[head] = {
            "trained": True,
            "coefficients": coefficients.tolist(),
            "train": _metrics(x_train, target, coefficients, means, scales),
            "validation": _metrics(
                split_arrays["validation"][0],
                split_arrays["validation"][1][head],
                coefficients,
                means,
                scales,
            ),
            "locked_test": (
                _metrics(
                    split_arrays["locked_test"][0],
                    split_arrays["locked_test"][1][head],
                    coefficients,
                    means,
                    scales,
                )
                if "locked_test" in split_arrays
                else None
            ),
            "train_labelled_frames": len(labelled_target),
            "train_positive_frames": positives,
            "train_negative_frames": negatives,
        }
    annotation_hashes = {
        str(item.annotation_path.resolve()): _sha256_file(item.annotation_path)
        for item in items
    }
    scope_counts: dict[str, int] = {}
    for item in items:
        scope = _annotation_scope(item.annotation)
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
    head_label_coverage = _head_label_coverage(split_arrays)
    training_input_sha256 = _training_input_sha256(
        items,
        feature_backend=project.alignment.feature_backend,
        l2=l2,
        posterior_blend=posterior_blend,
    )
    model = {
        "schema_version": POSTERIOR_MODEL_SCHEMA_VERSION,
        "model_kind": POSTERIOR_MODEL_KIND,
        "feature_backend": project.alignment.feature_backend,
        "feature_names": list(POSTERIOR_FEATURE_NAMES),
        "heads": heads,
        "normalization": {"mean": means.tolist(), "scale": scales.tolist()},
        "runtime": {
            "posterior_blend": posterior_blend,
            "heuristic_residual_blend": 1.0 - posterior_blend,
        },
        "training": {
            "label_source": "manual_gold",
            "derived_frame_targets": True,
            "special_manual_phone_target_policy": SPECIAL_MANUAL_PHONE_TARGET_POLICY,
            "split_unit": "root_speaker_id",
            "project": str(project.path),
            "project_sha256": project.project_sha256,
            "inventory_sha256": project.inventory_sha256,
            "train_recordings": len(train_items),
            "validation_recordings": len(validation_items),
            "locked_test_recordings": len(locked_items),
            "speaker_counts": {
                split: len({item.speaker_id for item in split_items})
                for split, split_items in item_groups.items()
            },
            "annotation_sha256": annotation_hashes,
            "annotation_scope_counts": dict(sorted(scope_counts.items())),
            "head_label_coverage": head_label_coverage,
            "training_input_sha256": training_input_sha256,
            "reproducibility": {
                "deterministic": True,
                "optimizer": "L-BFGS-B",
                "initial_coefficients": "all_zero",
                "random_seed_used": False,
                "serialization": "sorted_keys_utf8_lf",
            },
            "l2": l2,
        },
    }
    text = json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        output_model.resolve(),
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_model.exists(),
    )
    return {
        "model": str(output_model.resolve()),
        "model_sha256": _sha256_file(output_model),
        "model_kind": POSTERIOR_MODEL_KIND,
        "train_recordings": len(train_items),
        "validation_recordings": len(validation_items),
        "locked_test_recordings": len(locked_items),
        "heads": heads,
        "label_source": "manual_gold",
        "annotation_scope_counts": dict(sorted(scope_counts.items())),
        "head_label_coverage": head_label_coverage,
        "training_input_sha256": training_input_sha256,
    }


def train_posterior_model_from_manifest(
    items_csv: Path,
    output_model: Path,
    *,
    feature_backend: str | None = None,
    minimum_train_recordings: int = 20,
    minimum_validation_recordings: int = 5,
    l2: float = 0.05,
    posterior_blend: float = 0.75,
) -> dict[str, object]:
    if not 0.0 < posterior_blend <= 1.0:
        raise PosteriorModelError("posterior_blend must be greater than 0 and at most 1")
    items_path = items_csv.resolve()
    all_items = _load_manifest_training_items(items_path)
    available_backends = sorted({item.project.alignment.feature_backend for item in all_items})
    selected_backend = str(feature_backend or "").strip()
    if not selected_backend:
        if len(available_backends) != 1:
            raise PosteriorModelError(
                f"multiple feature backends require --feature-backend: {available_backends}"
            )
        selected_backend = available_backends[0]
    if selected_backend not in available_backends:
        raise PosteriorModelError(
            f"no manual_gold items for feature backend {selected_backend}; available={available_backends}"
        )
    items = tuple(
        item for item in all_items if item.project.alignment.feature_backend == selected_backend
    )
    train_items = tuple(item for item in items if item.split == "train")
    validation_items = tuple(item for item in items if item.split == "validation")
    locked_items = tuple(item for item in items if item.split == "locked_test")
    if len(train_items) < minimum_train_recordings:
        raise PosteriorModelError(
            f"manual_gold train recordings are insufficient for {selected_backend}: "
            f"{len(train_items)} < {minimum_train_recordings}"
        )
    if len(validation_items) < minimum_validation_recordings:
        raise PosteriorModelError(
            f"manual_gold validation recordings are insufficient for {selected_backend}: "
            f"{len(validation_items)} < {minimum_validation_recordings}"
        )
    _validate_speaker_splits(items)
    item_groups = {
        "train": train_items,
        "validation": validation_items,
        "locked_test": locked_items,
    }
    split_arrays: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for split, split_items in item_groups.items():
        if split_items:
            split_arrays[split] = _compile_multi_project_examples(split_items)
    x_train, y_train = split_arrays["train"]
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales < 1e-6, 1.0, scales)
    normalized_train = (x_train - means) / scales
    heads: dict[str, object] = {}
    for head in POSTERIOR_HEADS:
        target = y_train[head]
        labelled = np.isfinite(target)
        labelled_target = target[labelled]
        if not np.any(labelled):
            heads[head] = {
                "trained": False,
                "reason": "manual_gold_head_unlabelled",
                "coefficients": np.zeros(len(POSTERIOR_FEATURE_NAMES) + 1, dtype=np.float64).tolist(),
                "train": None,
                "validation": None,
                "locked_test": None,
                "train_labelled_frames": 0,
                "train_positive_frames": 0,
                "train_negative_frames": 0,
            }
            continue
        positives = int(np.sum(labelled_target >= 0.5))
        negatives = len(labelled_target) - positives
        if positives < 20 or negatives < 20:
            raise PosteriorModelError(
                f"{head}: insufficient positive/negative frames (positive={positives}, negative={negatives})"
            )
        coefficients = _fit_logistic(normalized_train[labelled], labelled_target, l2=l2)
        heads[head] = {
            "trained": True,
            "coefficients": coefficients.tolist(),
            "train": _metrics(x_train, target, coefficients, means, scales),
            "validation": _metrics(
                split_arrays["validation"][0],
                split_arrays["validation"][1][head],
                coefficients,
                means,
                scales,
            ),
            "locked_test": (
                _metrics(
                    split_arrays["locked_test"][0],
                    split_arrays["locked_test"][1][head],
                    coefficients,
                    means,
                    scales,
                )
                if "locked_test" in split_arrays
                else None
            ),
            "train_labelled_frames": len(labelled_target),
            "train_positive_frames": positives,
            "train_negative_frames": negatives,
        }
    inventory_hashes = sorted({item.project.inventory_sha256 for item in items})
    project_hashes = sorted({item.project.project_sha256 for item in items})
    scope_counts: dict[str, int] = {}
    for item in items:
        scope = _annotation_scope(item.annotation)
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
    coverage = _head_label_coverage(split_arrays)
    training_input_sha256 = _training_input_sha256(
        items,
        feature_backend=selected_backend,
        l2=l2,
        posterior_blend=posterior_blend,
    )
    model = {
        "schema_version": POSTERIOR_MODEL_SCHEMA_VERSION,
        "model_kind": POSTERIOR_MODEL_KIND,
        "feature_backend": selected_backend,
        "feature_names": list(POSTERIOR_FEATURE_NAMES),
        "heads": heads,
        "normalization": {"mean": means.tolist(), "scale": scales.tolist()},
        "runtime": {
            "posterior_blend": posterior_blend,
            "heuristic_residual_blend": 1.0 - posterior_blend,
        },
        "training": {
            "label_source": "manual_gold",
            "derived_frame_targets": True,
            "special_manual_phone_target_policy": SPECIAL_MANUAL_PHONE_TARGET_POLICY,
            "split_unit": "root_speaker_id",
            "source": {
                "kind": "speaker_locked_items_manifest",
                "items_csv": str(items_path),
                "items_csv_sha256": file_sha256(items_path),
            },
            "project_sha256": project_hashes[0] if len(project_hashes) == 1 else None,
            "project_sha256_allowlist": project_hashes,
            "inventory_sha256": inventory_hashes[0] if len(inventory_hashes) == 1 else None,
            "inventory_sha256_allowlist": inventory_hashes,
            "generic_acoustic_heads": len(inventory_hashes) > 1,
            "train_recordings": len(train_items),
            "validation_recordings": len(validation_items),
            "locked_test_recordings": len(locked_items),
            "speaker_counts": {
                split: len({item.speaker_id for item in split_items})
                for split, split_items in item_groups.items()
            },
            "annotation_sha256": {
                f"{item.bank_id}/{item.recording_index:05d}": _sha256_file(item.annotation_path)
                for item in items
            },
            "annotation_scope_counts": dict(sorted(scope_counts.items())),
            "head_label_coverage": coverage,
            "training_input_sha256": training_input_sha256,
            "reproducibility": {
                "deterministic": True,
                "optimizer": "L-BFGS-B",
                "initial_coefficients": "all_zero",
                "random_seed_used": False,
                "serialization": "sorted_keys_utf8_lf",
            },
            "l2": l2,
        },
    }
    text = json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        output_model.resolve(),
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=output_model.exists(),
    )
    return {
        "model": str(output_model.resolve()),
        "model_sha256": _sha256_file(output_model),
        "model_kind": POSTERIOR_MODEL_KIND,
        "feature_backend": selected_backend,
        "inventory_sha256_allowlist": inventory_hashes,
        "train_recordings": len(train_items),
        "validation_recordings": len(validation_items),
        "locked_test_recordings": len(locked_items),
        "heads": heads,
        "label_source": "manual_gold",
        "annotation_scope_counts": dict(sorted(scope_counts.items())),
        "head_label_coverage": coverage,
        "training_input_sha256": training_input_sha256,
    }


def load_posterior_model(
    path: Path,
    *,
    expected_feature_backend: str | None = None,
) -> Mapping[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PosteriorModelError("posterior model root는 mapping이어야 합니다")
    if payload.get("schema_version") != POSTERIOR_MODEL_SCHEMA_VERSION:
        raise PosteriorModelError("지원하지 않는 posterior model schema_version입니다")
    if payload.get("model_kind") != POSTERIOR_MODEL_KIND:
        raise PosteriorModelError("지원하지 않는 posterior model_kind입니다")
    model_backend = payload.get("feature_backend")
    if model_backend not in SUPPORTED_FEATURE_BACKENDS:
        raise PosteriorModelError("posterior model feature backend가 runtime과 다릅니다")
    if expected_feature_backend is not None and model_backend != expected_feature_backend:
        raise PosteriorModelError("posterior model feature backend이 project와 다릅니다")
    if tuple(payload.get("feature_names", ())) != POSTERIOR_FEATURE_NAMES:
        raise PosteriorModelError("posterior model feature_names가 runtime과 다릅니다")
    heads = payload.get("heads")
    normalization = payload.get("normalization")
    runtime = payload.get("runtime")
    if not isinstance(heads, Mapping) or not isinstance(normalization, Mapping) or not isinstance(runtime, Mapping):
        raise PosteriorModelError("posterior model heads/normalization/runtime이 없습니다")
    for head in POSTERIOR_HEADS:
        definition = heads.get(head)
        if not isinstance(definition, Mapping):
            raise PosteriorModelError(f"posterior head가 없습니다: {head}")
        coefficients = np.asarray(definition.get("coefficients"), dtype=np.float64)
        if coefficients.shape != (len(POSTERIOR_FEATURE_NAMES) + 1,) or not np.all(np.isfinite(coefficients)):
            raise PosteriorModelError(f"posterior coefficient가 유효하지 않습니다: {head}")
    blend = float(runtime.get("posterior_blend", -1.0))
    if not 0.0 < blend <= 1.0:
        raise PosteriorModelError("posterior model runtime blend가 유효하지 않습니다")
    return payload


def apply_posterior_model(
    evidence: AcousticEvidence,
    model: Mapping[str, object],
    *,
    model_path: Path,
) -> AcousticEvidence:
    if model.get("feature_backend") != evidence.metadata.get("feature_backend"):
        raise PosteriorModelError("posterior model feature backend이 acoustic evidence와 다릅니다")
    features = np.column_stack([evidence.tracks[name] for name in POSTERIOR_FEATURE_NAMES]).astype(np.float64)
    normalization = model["normalization"]
    means = np.asarray(normalization["mean"], dtype=np.float64)
    scales = np.asarray(normalization["scale"], dtype=np.float64)
    if means.shape != (len(POSTERIOR_FEATURE_NAMES),) or scales.shape != means.shape or np.any(scales <= 0.0):
        raise PosteriorModelError("posterior normalization 값이 유효하지 않습니다")
    normalized = (features - means) / scales
    blend = float(model["runtime"]["posterior_blend"])
    tracks = dict(evidence.tracks)
    for head in POSTERIOR_HEADS:
        definition = model["heads"][head]
        if definition.get("trained", True) is False:
            continue
        coefficients = np.asarray(definition["coefficients"], dtype=np.float64)
        learned = _sigmoid(coefficients[0] + normalized @ coefficients[1:])
        heuristic = tracks[head]
        tracks[head] = np.clip(blend * learned + (1.0 - blend) * heuristic, 0.0, 1.0)
    model_sha256 = _sha256_file(model_path)
    return replace(
        evidence,
        tracks=tracks,
        metadata={
            **dict(evidence.metadata),
            "posterior_model": {
                "model_kind": POSTERIOR_MODEL_KIND,
                "model_path": str(model_path.resolve()),
                "model_sha256": model_sha256,
                "posterior_blend": blend,
                "manual_gold_trained": True,
            },
        },
    )


def _load_training_items(project: ProjectSpec, annotations_dir: Path) -> tuple[PosteriorTrainingItem, ...]:
    paths = sorted(annotations_dir.resolve().glob("*.annotation.json"))
    if not paths:
        raise PosteriorModelError(f"annotation 파일이 없습니다: {annotations_dir.resolve()}")
    wav_to_index = {recording.wav: index for index, recording in enumerate(project.recordings)}
    items: list[PosteriorTrainingItem] = []
    seen_wavs: set[str] = set()
    for path in paths:
        annotation = load_annotation(path)
        if annotation.label_source != "manual_gold":
            continue
        if annotation.wav in seen_wavs:
            raise PosteriorModelError(f"동일 WAV의 manual_gold annotation이 중복되었습니다: {annotation.wav}")
        seen_wavs.add(annotation.wav)
        if annotation.wav not in wav_to_index:
            raise PosteriorModelError(f"annotation WAV가 project에 없습니다: {annotation.wav}")
        index = wav_to_index[annotation.wav]
        recording = project.recordings[index]
        if annotation.inventory_sha256 != project.inventory_sha256:
            raise PosteriorModelError(f"{annotation.wav}: annotation inventory hash mismatch")
        if file_sha256(project.wav_path(recording)) != annotation.wav_sha256:
            raise PosteriorModelError(f"{annotation.wav}: annotation WAV hash mismatch")
        metadata = recording.metadata
        speaker_id = str(metadata.get("root_speaker_id") or metadata.get("speaker_id") or "").strip()
        bank_id = str(metadata.get("bank_id") or "").strip()
        split = str(metadata.get("split") or "").strip()
        if not speaker_id or not bank_id:
            raise PosteriorModelError(f"{recording.wav}: metadata.root_speaker_id와 bank_id가 필요합니다")
        if split not in VALID_SPLITS:
            raise PosteriorModelError(f"{recording.wav}: metadata.split은 train/validation/locked_test여야 합니다")
        items.append(PosteriorTrainingItem(project, path, annotation, index, speaker_id, bank_id, split))
    if not items:
        raise PosteriorModelError("manual_gold annotation이 없습니다")
    return tuple(items)


def _validate_speaker_splits(items: Sequence[PosteriorTrainingItem]) -> None:
    speaker_splits: dict[str, set[str]] = {}
    bank_splits: dict[str, set[str]] = {}
    for item in items:
        speaker_splits.setdefault(item.speaker_id, set()).add(item.split)
        bank_splits.setdefault(item.bank_id, set()).add(item.split)
    leaking_speakers = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    leaking_banks = sorted(bank for bank, splits in bank_splits.items() if len(splits) > 1)
    if leaking_speakers or leaking_banks:
        raise PosteriorModelError(
            f"split leakage가 있습니다: speakers={leaking_speakers[:5]}, banks={leaking_banks[:5]}"
        )


def _compile_examples(
    project: ProjectSpec,
    items: Sequence[PosteriorTrainingItem],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    feature_rows: list[np.ndarray] = []
    targets: dict[str, list[np.ndarray]] = {head: [] for head in POSTERIOR_HEADS}
    for item in items:
        recording = project.recordings[item.recording_index]
        evidence = extract_acoustic_evidence(
            project.wav_path(recording),
            internal_sample_rate=project.audio.internal_sample_rate,
            convert_to_mono=project.audio.convert_to_mono,
            resampler=project.audio.resampler,
            feature_backend=project.alignment.feature_backend,
            evidence_model=(project.root / project.alignment.evidence_model).resolve() if project.alignment.evidence_model else None,
            evidence_provider=project.alignment.evidence_provider,
        )
        feature_rows.append(np.column_stack([evidence.tracks[name] for name in POSTERIOR_FEATURE_NAMES]))
        item_targets = _frame_targets(project, recording, item.annotation, evidence.frame_times_ms)
        for head in POSTERIOR_HEADS:
            targets[head].append(item_targets[head])
    return (
        np.concatenate(feature_rows, axis=0).astype(np.float64),
        {head: np.concatenate(values).astype(np.float64) for head, values in targets.items()},
    )


def _compile_multi_project_examples(
    items: Sequence[PosteriorTrainingItem],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    feature_rows: list[np.ndarray] = []
    targets: dict[str, list[np.ndarray]] = {head: [] for head in POSTERIOR_HEADS}
    for item in sorted(items, key=lambda value: (value.bank_id, value.recording_index, value.annotation.wav_sha256)):
        features, item_targets = _compile_examples(item.project, (item,))
        feature_rows.append(features)
        for head in POSTERIOR_HEADS:
            targets[head].append(item_targets[head])
    return (
        np.concatenate(feature_rows, axis=0).astype(np.float64),
        {head: np.concatenate(values).astype(np.float64) for head, values in targets.items()},
    )


def _load_manifest_training_items(items_csv: Path) -> tuple[PosteriorTrainingItem, ...]:
    with items_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PosteriorModelError(f"training items CSV is empty: {items_csv}")
    projects: dict[Path, ProjectSpec] = {}
    items: list[PosteriorTrainingItem] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        project_path = Path(str(row.get("project", "")).strip()).resolve()
        annotation_path = Path(str(row.get("annotation", "")).strip()).resolve()
        if not project_path.is_file() or not annotation_path.is_file():
            raise PosteriorModelError(f"manifest project/annotation file is missing: {project_path}, {annotation_path}")
        project = projects.get(project_path)
        if project is None:
            project = load_project(project_path)
            projects[project_path] = project
        expected_project_hash = str(row.get("project_sha256", "")).strip()
        if expected_project_hash and expected_project_hash != project.project_sha256:
            raise PosteriorModelError(f"project semantic hash mismatch: {project_path}")
        expected_annotation_hash = str(row.get("annotation_sha256", "")).strip()
        if expected_annotation_hash and expected_annotation_hash != file_sha256(annotation_path):
            raise PosteriorModelError(f"annotation hash mismatch: {annotation_path}")
        annotation = load_annotation(annotation_path)
        if annotation.label_source != "manual_gold":
            continue
        recording_index = int(str(row.get("recording_index", "")).strip())
        if not 0 <= recording_index < len(project.recordings):
            raise PosteriorModelError(f"recording_index outside project: {recording_index}")
        recording = project.recordings[recording_index]
        if annotation.inventory_sha256 != project.inventory_sha256:
            raise PosteriorModelError(f"manifest annotation inventory mismatch: {annotation_path}")
        if recording.wav != annotation.wav or file_sha256(project.wav_path(recording)) != annotation.wav_sha256:
            raise PosteriorModelError(f"manifest recording/annotation mismatch: {annotation_path}")
        bank_id = str(row.get("bank_id", "")).strip()
        speaker_id = str(row.get("root_speaker_id", "")).strip()
        split = str(row.get("split") or row.get("evaluation_split") or "").strip()
        if not bank_id or not speaker_id:
            raise PosteriorModelError(f"manifest bank_id/root_speaker_id is empty: {annotation_path}")
        if split not in VALID_SPLITS:
            raise PosteriorModelError(f"manifest split is invalid: {split}")
        identity = (bank_id, annotation.wav_sha256)
        if identity in seen:
            raise PosteriorModelError(f"duplicate manifest training item: {identity}")
        seen.add(identity)
        items.append(
            PosteriorTrainingItem(
                project,
                annotation_path,
                annotation,
                recording_index,
                speaker_id,
                bank_id,
                split,
            )
        )
    if not items:
        raise PosteriorModelError("manual_gold annotations are required")
    return tuple(items)


def _frame_targets(project, recording, annotation, frame_times_ms: np.ndarray) -> dict[str, np.ndarray]:
    phones = tuple(annotation.phones)
    marker_times: dict[str, list[float]] = {}
    usable_start = 0.0
    usable_end = float(frame_times_ms[-1]) if len(frame_times_ms) else 0.0
    boundaries: dict[int, float] = {}
    for marker in annotation.markers:
        marker_times.setdefault(marker.label, []).append(marker.time_ms)
        if marker.label == "usable_start":
            usable_start = marker.time_ms
        elif marker.label == "usable_end":
            usable_end = marker.time_ms
        elif marker.label == "phone_boundary" and marker.right_phone_index is not None:
            boundaries[marker.right_phone_index] = marker.time_ms
    if len(phones) > 1 and any(index not in boundaries for index in range(1, len(phones))):
        raise PosteriorModelError(f"{recording.wav}: 모든 내부 phone_boundary를 수동 Gold로 표시해야 합니다")
    edges = [usable_start, *(boundaries[index] for index in range(1, len(phones))), usable_end]
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise PosteriorModelError(f"{recording.wav}: phone boundary 순서가 유효하지 않습니다")
    active = (frame_times_ms >= usable_start) & (frame_times_ms <= usable_end)
    result = {head: np.zeros(len(frame_times_ms), dtype=np.float64) for head in POSTERIOR_HEADS}
    result["boundary"] = _event_target(
        frame_times_ms,
        [
            usable_start,
            *boundaries.values(),
            usable_end,
            *marker_times.get("consonant_onset", []),
            *marker_times.get("release", []),
        ],
    )
    transition_events = [
        *marker_times.get("vowel_transition", []),
        *marker_times.get("subphone_transition", []),
        *marker_times.get("voicing_onset", []),
    ]
    transition_labelled = bool(transition_events)
    result["transition"] = (
        _event_target(frame_times_ms, transition_events)
        if transition_labelled
        else np.full(len(frame_times_ms), np.nan, dtype=np.float64)
    )
    for phone_index, phone_id in enumerate(phones):
        mask = (frame_times_ms >= edges[phone_index]) & (frame_times_ms < edges[phone_index + 1])
        special = phone_id.casefold()
        if special == "sil":
            for head in ("vowel_like", "consonant_like", "periodicity", "aperiodic"):
                result[head][mask] = np.nan
            continue
        if special == "vf":
            # Vocal fry is intentionally neutral for the existing generic class
            # heads until a dedicated fry head has enough Gold support.
            for head in ("vowel_like", "consonant_like", "periodicity", "aperiodic"):
                result[head][mask] = np.nan
            continue
        if special == "exh":
            # Vowel-final exhalation (filename uppercase R) is aperiodic but is
            # not treated as an ordinary consonant.
            for head in ("vowel_like", "consonant_like", "periodicity"):
                result[head][mask] = np.nan
            result["aperiodic"][mask] = 1.0
            continue
        definition = project.inventory.get(phone_id)
        if definition is None:
            raise PosteriorModelError(
                f"{recording.wav}: manual Gold phone is missing from inventory: {phone_id}"
            )
        features = definition.features
        vowel = features.get("syllabic") is True or features.get("vowel_like") is True
        voiced = features.get("voiced") is True or features.get("periodic") is True
        unvoiced = features.get("voiced") is False
        if vowel:
            result["vowel_like"][mask] = 1.0
        else:
            result["consonant_like"][mask] = 1.0
        if voiced:
            result["periodicity"][mask] = 1.0
        if unvoiced:
            result["aperiodic"][mask] = 1.0
    for head in POSTERIOR_HEADS:
        if head != "transition" or transition_labelled:
            result[head][~active] = 0.0
    return result


def _annotation_scope(annotation: AnnotationDocument) -> str:
    labels = {marker.label for marker in annotation.markers}
    detailed = labels - {"usable_start", "usable_end", "phone_boundary"}
    return "full_landmark" if detailed else "phone_boundary_only"


def _head_label_coverage(
    split_arrays: Mapping[str, tuple[np.ndarray, Mapping[str, np.ndarray]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for head in POSTERIOR_HEADS:
        split_result: dict[str, object] = {}
        for split in ("train", "validation", "locked_test"):
            if split not in split_arrays:
                split_result[split] = None
                continue
            target = split_arrays[split][1][head]
            labelled = int(np.sum(np.isfinite(target)))
            split_result[split] = {
                "total_frames": len(target),
                "labelled_frames": labelled,
                "coverage": labelled / len(target) if len(target) else 0.0,
            }
        result[head] = split_result
    return result


def _training_input_sha256(
    items: Sequence[PosteriorTrainingItem],
    *,
    feature_backend: str,
    l2: float,
    posterior_blend: float,
) -> str:
    records = [
        {
            "annotation_sha256": _sha256_file(item.annotation_path),
            "bank_id": item.bank_id,
            "inventory_sha256": item.project.inventory_sha256,
            "project_sha256": item.project.project_sha256,
            "recording_index": item.recording_index,
            "speaker_id": item.speaker_id,
            "split": item.split,
            "wav_sha256": item.annotation.wav_sha256,
        }
        for item in items
    ]
    records.sort(key=lambda item: (str(item["bank_id"]), int(item["recording_index"]), str(item["wav_sha256"])))
    payload = {
        "feature_backend": feature_backend,
        "feature_names": POSTERIOR_FEATURE_NAMES,
        "items": records,
        "l2": l2,
        "posterior_blend": posterior_blend,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_target(frame_times_ms: np.ndarray, events: Sequence[float], radius_ms: float = 12.0) -> np.ndarray:
    target = np.zeros(len(frame_times_ms), dtype=np.float64)
    for event in events:
        target = np.maximum(target, np.exp(-0.5 * ((frame_times_ms - float(event)) / radius_ms) ** 2))
    return target


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float) -> np.ndarray:
    positives = max(float(np.sum(y)), 1.0)
    negatives = max(float(np.sum(1.0 - y)), 1.0)
    weights = np.where(y >= 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = coefficients[0] + x @ coefficients[1:]
        probability = _sigmoid(logits)
        epsilon = 1e-9
        loss = -np.mean(weights * (y * np.log(probability + epsilon) + (1.0 - y) * np.log(1.0 - probability + epsilon)))
        loss += 0.5 * l2 * float(coefficients[1:] @ coefficients[1:])
        residual = weights * (probability - y) / len(y)
        gradient = np.concatenate(([np.sum(residual)], x.T @ residual + l2 * coefficients[1:]))
        return float(loss), gradient

    result = minimize(
        lambda coefficients: objective(coefficients),
        np.zeros(x.shape[1] + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise PosteriorModelError(f"posterior logistic 학습 실패: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _metrics(x, y, coefficients, means, scales) -> dict[str, float | int]:
    labelled = np.isfinite(y)
    x = x[labelled]
    y = y[labelled]
    if not len(y):
        return {"frames": 0, "brier": 0.0, "balanced_accuracy": 0.0, "positive_rate": 0.0}
    probability = _sigmoid(coefficients[0] + ((x - means) / scales) @ coefficients[1:])
    predicted = probability >= 0.5
    positive = y >= 0.5
    negative = ~positive
    true_positive_rate = float(np.mean(predicted[positive])) if np.any(positive) else 0.0
    true_negative_rate = float(np.mean(~predicted[negative])) if np.any(negative) else 0.0
    return {
        "frames": len(y),
        "brier": float(np.mean((probability - y) ** 2)),
        "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "positive_rate": float(np.mean(y)),
    }


def _sigmoid(value):
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
