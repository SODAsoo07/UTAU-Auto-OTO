from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .oto_io import write_lines_atomic


ANCHOR_FIELDS = ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
OTO_METRIC_FIELDS = ("offset", "overlap", "preutterance", "consonant", "cutoff_abs")
PROFILE_KIND = "renderer_landmark_residual_v1"
ALLOWED_REFERENCE_KINDS = frozenset({"manual_oto_anchor_gold", "manual_oto_convention_reviewed"})


@dataclass(frozen=True)
class RendererCalibrationSelection:
    scope: str
    key: str
    support: int
    residuals_ms: Mapping[str, float]
    shrinkage_weight: float
    statistics: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class RendererCalibrationProfile:
    path: Path
    sha256: str
    profile_id: str
    minimum_support: int
    entries: Mapping[str, Mapping[str, object]]
    training_input_sha256: str
    training_bank_ids: frozenset[str]
    training_format_types: frozenset[str]


def build_renderer_calibration_examples(
    row_report_jsonl: Path,
    reviewed_anchors_jsonl: Path,
    output_jsonl: Path,
) -> dict[str, object]:
    """Bind reviewed absolute OTO anchors to the renderer inputs they correct."""
    row_report_path = row_report_jsonl.resolve()
    references_path = reviewed_anchors_jsonl.resolve()
    rows = _load_jsonl_mappings(row_report_path)
    references = _load_jsonl_mappings(references_path)
    by_index = {index: row for index, row in enumerate(rows)}
    if len(by_index) != len(rows):  # pragma: no cover - enumerate is unique by construction
        raise ValueError("row report indices are not unique")
    output_rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for reference_number, reference in enumerate(references, start=1):
        raw_index = reference.get("row_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index not in by_index:
            raise ValueError(f"reviewed anchor line {reference_number} has invalid row_index={raw_index!r}")
        if raw_index in seen:
            raise ValueError(f"duplicate reviewed row_index: {raw_index}")
        seen.add(raw_index)
        reference_kind = str(reference.get("reference_kind", "")).strip()
        if reference_kind not in ALLOWED_REFERENCE_KINDS:
            raise ValueError(f"reviewed anchor line {reference_number} has unsupported reference_kind")
        split = str(reference.get("split", "")).strip().casefold()
        if split not in {"train", "validation", "locked_test"}:
            raise ValueError(f"reviewed anchor line {reference_number} has invalid split={split!r}")
        rendered = by_index[raw_index]
        for identity_field, rendered_field in (("alias", "alias"), ("wav", "recording_wav")):
            supplied = reference.get(identity_field)
            if supplied is not None and str(supplied) != str(rendered.get(rendered_field, rendered.get(identity_field, ""))):
                raise ValueError(
                    f"reviewed anchor line {reference_number} {identity_field} does not match row report"
                )
        raw_anchors = rendered.get("raw_anchors")
        reviewed_anchors = reference.get("reviewed_anchors")
        raw_values = _anchor_mapping(raw_anchors, f"row_report[{raw_index}].raw_anchors")
        reviewed_values = _anchor_mapping(
            reviewed_anchors,
            f"reviewed_anchors[{reference_number}].reviewed_anchors",
        )
        alias_type = rendered.get("alias_type")
        alias_type_mapping = alias_type if isinstance(alias_type, Mapping) else {}
        metadata = reference.get("metadata")
        metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
        output_rows.append(
            {
                "schema_version": 1,
                "split": split,
                "reference_kind": reference_kind,
                "row_index": raw_index,
                "wav": str(rendered.get("recording_wav") or rendered.get("wav") or ""),
                "alias": str(rendered.get("alias", "")),
                "template_id": _nonempty(rendered.get("template_id"), f"row_report[{raw_index}].template_id"),
                "structural_role": str(alias_type_mapping.get("structural_role", "UNKNOWN")),
                "acoustic_subtype": str(alias_type_mapping.get("acoustic_subtype", "unknown")),
                "bank_id": _nonempty(metadata_mapping.get("bank_id"), f"reviewed line {reference_number}.metadata.bank_id"),
                "format_type": _nonempty(
                    metadata_mapping.get("format_type"),
                    f"reviewed line {reference_number}.metadata.format_type",
                ),
                **(
                    {"expression_profile": str(metadata_mapping["expression_profile"]).strip()}
                    if _optional_nonempty(metadata_mapping.get("expression_profile"))
                    else {}
                ),
                **(
                    {"pitch_bucket": str(metadata_mapping["pitch_bucket"]).strip()}
                    if _optional_nonempty(metadata_mapping.get("pitch_bucket"))
                    else {}
                ),
                "raw_anchors_ms": raw_values,
                "reviewed_anchors_ms": reviewed_values,
                "residuals_ms": {
                    field: reviewed_values[field] - raw_values[field]
                    for field in ANCHOR_FIELDS
                },
                "source_row_report_sha256": hashlib.sha256(row_report_path.read_bytes()).hexdigest(),
                "source_reviewed_anchors_sha256": hashlib.sha256(references_path.read_bytes()).hexdigest(),
            }
        )
    target = output_jsonl.resolve()
    write_lines_atomic(
        target,
        (json.dumps(row, ensure_ascii=False, sort_keys=True) for row in output_rows),
        encoding="utf-8",
        newline="\n",
        backup_existing=target.exists(),
    )
    return {
        "schema_version": 1,
        "purpose": "renderer_calibration_examples",
        "example_count": len(output_rows),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in output_rows).items())),
        "row_report_sha256": hashlib.sha256(row_report_path.read_bytes()).hexdigest(),
        "reviewed_anchors_sha256": hashlib.sha256(references_path.read_bytes()).hexdigest(),
        "output": str(target),
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "release_promotion_evidence": False,
    }


def evaluate_renderer_calibration_holdout(
    examples_jsonl: Path,
    profile_json: Path,
    output_json: Path,
    *,
    split: str = "validation",
    correction_threshold_ms: float = 20.0,
    minimum_examples: int = 10,
    allow_locked_test: bool = False,
) -> dict[str, object]:
    normalized_split = split.strip().casefold()
    if normalized_split not in {"validation", "locked_test"}:
        raise ValueError("renderer holdout evaluation split must be validation or locked_test")
    if normalized_split == "locked_test" and not allow_locked_test:
        raise ValueError("locked_test evaluation requires explicit allow_locked_test=True")
    if minimum_examples < 1:
        raise ValueError("minimum_examples must be >= 1")
    if not math.isfinite(correction_threshold_ms) or correction_threshold_ms <= 0.0:
        raise ValueError("correction_threshold_ms must be finite and > 0")
    examples_path = examples_jsonl.resolve()
    profile_path = profile_json.resolve()
    profile = load_renderer_calibration(profile_path)
    all_rows = _load_jsonl_mappings(examples_path)
    training_bank_ids = {
        _nonempty(row.get("bank_id"), f"train examples[{index}].bank_id")
        for index, row in enumerate(all_rows)
        if str(row.get("split", "")).strip().casefold() == "train"
    }
    rows = [
        row
        for row in all_rows
        if str(row.get("split", "")).strip().casefold() == normalized_split
    ]
    row_metrics = []
    baseline_errors: list[float] = []
    candidate_errors: list[float] = []
    baseline_errors_by_field: dict[str, list[float]] = {field: [] for field in ANCHOR_FIELDS}
    candidate_errors_by_field: dict[str, list[float]] = {field: [] for field in ANCHOR_FIELDS}
    baseline_oto_errors_by_field: dict[str, list[float]] = {field: [] for field in OTO_METRIC_FIELDS}
    candidate_oto_errors_by_field: dict[str, list[float]] = {field: [] for field in OTO_METRIC_FIELDS}
    leakage_rows = []
    baseline_correction_rows = 0
    candidate_correction_rows = 0
    baseline_large_repair_rows = 0
    candidate_large_repair_rows = 0
    selection_counts: Counter[str] = Counter()
    source_row_report_sha256s: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("reference_kind", "")).strip() not in ALLOWED_REFERENCE_KINDS:
            raise ValueError(f"examples[{index}] has unsupported reference_kind")
        source_row_sha = str(row.get("source_row_report_sha256", "")).strip().casefold()
        if len(source_row_sha) != 64 or any(character not in "0123456789abcdef" for character in source_row_sha):
            raise ValueError(f"examples[{index}].source_row_report_sha256 must be SHA-256")
        source_row_report_sha256s.add(source_row_sha)
        raw = _anchor_mapping(row.get("raw_anchors_ms"), f"examples[{index}].raw_anchors_ms")
        reviewed = _anchor_mapping(
            row.get("reviewed_anchors_ms"),
            f"examples[{index}].reviewed_anchors_ms",
        )
        template_id = _nonempty(row.get("template_id"), f"examples[{index}].template_id")
        bank_id = _nonempty(row.get("bank_id"), f"examples[{index}].bank_id")
        format_type = _nonempty(row.get("format_type"), f"examples[{index}].format_type")
        selection = select_renderer_calibration(
            profile,
            template_id=template_id,
            bank_id=bank_id,
            format_type=format_type,
            expression_profile=_optional_nonempty(row.get("expression_profile")),
            pitch_bucket=_optional_nonempty(row.get("pitch_bucket")),
        )
        predicted = apply_renderer_calibration(raw, selection)
        baseline_by_field = {field: abs(raw[field] - reviewed[field]) for field in ANCHOR_FIELDS}
        candidate_by_field = {field: abs(predicted[field] - reviewed[field]) for field in ANCHOR_FIELDS}
        baseline_errors.extend(baseline_by_field.values())
        candidate_errors.extend(candidate_by_field.values())
        for field in ANCHOR_FIELDS:
            baseline_errors_by_field[field].append(baseline_by_field[field])
            candidate_errors_by_field[field].append(candidate_by_field[field])
        reviewed_oto = _oto_metric_values(reviewed)
        raw_oto = _oto_metric_values(raw)
        predicted_oto = _oto_metric_values(predicted)
        for field in OTO_METRIC_FIELDS:
            baseline_oto_errors_by_field[field].append(abs(raw_oto[field] - reviewed_oto[field]))
            candidate_oto_errors_by_field[field].append(abs(predicted_oto[field] - reviewed_oto[field]))
        baseline_needs_correction = max(baseline_by_field.values()) > correction_threshold_ms
        candidate_needs_correction = max(candidate_by_field.values()) > correction_threshold_ms
        baseline_correction_rows += int(baseline_needs_correction)
        candidate_correction_rows += int(candidate_needs_correction)
        baseline_large_repair = _anchor_order_repair_delta(raw) > 20.0
        candidate_large_repair = _anchor_order_repair_delta(predicted) > 20.0
        baseline_large_repair_rows += int(baseline_large_repair)
        candidate_large_repair_rows += int(candidate_large_repair)
        selection_counts[selection.scope] += 1
        leakage = bank_id in training_bank_ids
        if leakage:
            leakage_rows.append(index)
        row_metrics.append(
            {
                "example_index": index,
                "row_index": row.get("row_index"),
                "bank_id": bank_id,
                "format_type": format_type,
                "template_id": template_id,
                "selection_scope": selection.scope,
                "selection_key": selection.key,
                "support": selection.support,
                "heldout_bank_leakage": leakage,
                "baseline_mae_ms": float(np.mean(list(baseline_by_field.values()))),
                "candidate_mae_ms": float(np.mean(list(candidate_by_field.values()))),
                "baseline_needs_correction": baseline_needs_correction,
                "candidate_needs_correction": candidate_needs_correction,
                "baseline_large_repair": baseline_large_repair,
                "candidate_large_repair": candidate_large_repair,
            }
        )
    example_count = len(rows)
    baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
    candidate_mae = float(np.mean(candidate_errors)) if candidate_errors else None
    baseline_residual_median = float(np.median(baseline_errors)) if baseline_errors else None
    candidate_residual_median = float(np.median(candidate_errors)) if candidate_errors else None
    residual_median_reduced = bool(
        baseline_residual_median is not None
        and candidate_residual_median is not None
        and candidate_residual_median < baseline_residual_median - 1e-12
    )
    enough_examples = example_count >= minimum_examples
    no_bank_leakage = not leakage_rows
    large_repair_not_increased = candidate_large_repair_rows <= baseline_large_repair_rows
    not_worse = bool(
        baseline_mae is not None
        and candidate_mae is not None
        and candidate_mae <= baseline_mae + 1e-9
        and candidate_correction_rows <= baseline_correction_rows
    )
    passed = (
        enough_examples
        and no_bank_leakage
        and not_worse
        and residual_median_reduced
        and large_repair_not_increased
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "renderer_calibration_holdout_evaluation",
        "split": normalized_split,
        "passed": passed,
        "minimum_examples": minimum_examples,
        "example_count": example_count,
        "correction_threshold_ms": correction_threshold_ms,
        "baseline_mae_ms": baseline_mae,
        "candidate_mae_ms": candidate_mae,
        "baseline_anchor_mae_ms": {
            field: float(np.mean(values)) if values else None
            for field, values in baseline_errors_by_field.items()
        },
        "candidate_anchor_mae_ms": {
            field: float(np.mean(values)) if values else None
            for field, values in candidate_errors_by_field.items()
        },
        "baseline_oto_field_mae_ms": {
            field: float(np.mean(values)) if values else None
            for field, values in baseline_oto_errors_by_field.items()
        },
        "candidate_oto_field_mae_ms": {
            field: float(np.mean(values)) if values else None
            for field, values in candidate_oto_errors_by_field.items()
        },
        "baseline_template_residual_median_ms": baseline_residual_median,
        "candidate_template_residual_median_ms": candidate_residual_median,
        "template_residual_median_reduced": residual_median_reduced,
        "baseline_manual_edit_distance_ms": (
            float(sum(sum(values) for values in baseline_oto_errors_by_field.values()) / example_count)
            if example_count else None
        ),
        "candidate_manual_edit_distance_ms": (
            float(sum(sum(values) for values in candidate_oto_errors_by_field.values()) / example_count)
            if example_count else None
        ),
        "mae_delta_ms": (
            candidate_mae - baseline_mae
            if baseline_mae is not None and candidate_mae is not None
            else None
        ),
        "baseline_correction_row_ratio": baseline_correction_rows / example_count if example_count else None,
        "candidate_correction_row_ratio": candidate_correction_rows / example_count if example_count else None,
        "baseline_large_repair_rate": baseline_large_repair_rows / example_count if example_count else None,
        "candidate_large_repair_rate": candidate_large_repair_rows / example_count if example_count else None,
        "large_repair_not_increased": large_repair_not_increased,
        "selection_scope_counts": dict(sorted(selection_counts.items())),
        "training_bank_ids": sorted(training_bank_ids),
        "evaluated_bank_ids": sorted({_nonempty(row.get("bank_id"), "evaluated bank_id") for row in rows}),
        "heldout_bank_leakage_count": len(leakage_rows),
        "heldout_bank_leakage_rows": leakage_rows,
        "checks": {
            "minimum_examples": enough_examples,
            "heldout_bank_isolation": no_bank_leakage,
            "candidate_not_worse": not_worse,
            "template_residual_median_reduced": residual_median_reduced,
            "large_repair_not_increased": large_repair_not_increased,
        },
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "training_input_sha256": profile.training_input_sha256,
        "examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "source_row_report_sha256s": sorted(source_row_report_sha256s),
        "validation_labels_used_for_fit": False,
        "locked_test_labels_used_for_fit": False,
        "release_promotion_evidence": False,
        "rows": row_metrics,
    }
    target = output_json.resolve()
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        target,
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=target.exists(),
    )
    return result


def fit_renderer_calibration(
    examples_jsonl: Path,
    output_json: Path,
    *,
    minimum_support: int = 20,
    shrinkage_strength: float = 20.0,
) -> dict[str, object]:
    if minimum_support < 1:
        raise ValueError("minimum_support must be >= 1")
    if not math.isfinite(shrinkage_strength) or shrinkage_strength < 0.0:
        raise ValueError("shrinkage_strength must be finite and >= 0")
    source = examples_jsonl.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    groups: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    parent_candidates: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    split_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    used_examples = 0
    training_bank_ids: set[str] = set()
    training_format_types: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"renderer calibration example line {line_number} must be a mapping")
        split = str(raw.get("split", "")).strip().casefold()
        split_counts[split or "missing"] += 1
        if split != "train":
            continue
        reference_kind = str(raw.get("reference_kind", "")).strip()
        reference_counts[reference_kind or "missing"] += 1
        if reference_kind not in ALLOWED_REFERENCE_KINDS:
            raise ValueError(
                f"renderer calibration example line {line_number} has unsupported reference_kind={reference_kind!r}"
            )
        template_id = _nonempty(raw.get("template_id"), f"line {line_number}.template_id")
        bank_id = _nonempty(raw.get("bank_id"), f"line {line_number}.bank_id")
        format_type = _nonempty(raw.get("format_type"), f"line {line_number}.format_type")
        training_bank_ids.add(bank_id)
        training_format_types.add(format_type)
        expression_profile = _optional_nonempty(raw.get("expression_profile"))
        pitch_bucket = _optional_nonempty(raw.get("pitch_bucket"))
        residuals_raw = raw.get("residuals_ms")
        if not isinstance(residuals_raw, Mapping) or set(residuals_raw) != set(ANCHOR_FIELDS):
            raise ValueError(f"renderer calibration example line {line_number} residual fields are invalid")
        residuals: dict[str, float] = {}
        for field in ANCHOR_FIELDS:
            value = residuals_raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"renderer calibration example line {line_number} {field} must be finite")
            residuals[field] = float(value)
        template_key = f"template|{template_id}"
        format_key = f"format_template|{format_type}|{template_id}"
        bank_key = f"bank_template|{bank_id}|{template_id}"
        groups["global"].append(residuals)
        groups[template_key].append(residuals)
        groups[format_key].append(residuals)
        groups[bank_key].append(residuals)
        parent_candidates[format_key]["template"].add(template_key)
        parent_candidates[bank_key]["format"].add(format_key)
        parent_candidates[bank_key]["template"].add(template_key)
        if expression_profile:
            expression_key = f"expression_template|{expression_profile}|{template_id}"
            groups[expression_key].append(residuals)
            parent_candidates[expression_key]["bank"].add(bank_key)
            parent_candidates[expression_key]["format"].add(format_key)
            parent_candidates[expression_key]["template"].add(template_key)
        if pitch_bucket:
            pitch_key = f"pitch_template|{pitch_bucket}|{template_id}"
            groups[pitch_key].append(residuals)
            if expression_profile:
                parent_candidates[pitch_key]["expression"].add(expression_key)
            parent_candidates[pitch_key]["bank"].add(bank_key)
            parent_candidates[pitch_key]["format"].add(format_key)
            parent_candidates[pitch_key]["template"].add(template_key)
        used_examples += 1

    if used_examples < minimum_support:
        raise ValueError(
            f"renderer calibration train support is insufficient: {used_examples} < {minimum_support}"
        )

    entries: dict[str, dict[str, object]] = {}
    ordered_keys = sorted(groups, key=_scope_order)
    for key in ordered_keys:
        values = groups[key]
        support = len(values)
        if support < minimum_support:
            continue
        scope = key.split("|", 1)[0]
        parent_key = _parent_key(key, entries, parent_candidates)
        parent = entries.get(parent_key) if parent_key else None
        raw_median = _field_stat(values, np.median)
        if parent is None:
            residuals = raw_median
            weight = 1.0
        else:
            weight = support / (support + shrinkage_strength) if shrinkage_strength > 0.0 else 1.0
            parent_residuals = parent["residuals_ms"]
            assert isinstance(parent_residuals, Mapping)
            residuals = {
                field: weight * raw_median[field] + (1.0 - weight) * float(parent_residuals[field])
                for field in ANCHOR_FIELDS
            }
        arrays = {field: np.asarray([row[field] for row in values], dtype=np.float64) for field in ANCHOR_FIELDS}
        medians = {field: float(np.median(array)) for field, array in arrays.items()}
        mads = {
            field: float(np.median(np.abs(array - medians[field])))
            for field, array in arrays.items()
        }
        entries[key] = {
            "scope": scope,
            "support": support,
            "parent_key": parent_key,
            "shrinkage_weight": float(weight),
            "residuals_ms": residuals,
            "raw_median_ms": raw_median,
            "mad_ms": mads,
            "p10_ms": {field: float(np.percentile(array, 10.0)) for field, array in arrays.items()},
            "p25_ms": {field: float(np.percentile(array, 25.0)) for field, array in arrays.items()},
            "p75_ms": {field: float(np.percentile(array, 75.0)) for field, array in arrays.items()},
            "p90_ms": {field: float(np.percentile(array, 90.0)) for field, array in arrays.items()},
            "outlier_rate": {
                field: float(
                    np.mean(
                        np.abs(array - medians[field])
                        > max(20.0, 4.4478 * mads[field])
                    )
                )
                for field, array in arrays.items()
            },
        }

    profile_id = f"renderer-calibration-{hashlib.sha256(source.read_bytes()).hexdigest()[:12]}"
    result: dict[str, object] = {
        "schema_version": 1,
        "profile_kind": PROFILE_KIND,
        "profile_id": profile_id,
        "minimum_support": minimum_support,
        "shrinkage_strength": shrinkage_strength,
        "training_splits_used": ["train"],
        "validation_labels_used": False,
        "locked_test_labels_used": False,
        "gold_timing_used_at_inference": False,
        "training_input": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "split_counts": dict(sorted(split_counts.items())),
        "reference_kind_counts_train": dict(sorted(reference_counts.items())),
        "used_train_examples": used_examples,
        "training_bank_ids": sorted(training_bank_ids),
        "training_format_types": sorted(training_format_types),
        "entries": entries,
    }
    target = output_json.resolve()
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(
        target,
        text.splitlines(),
        encoding="utf-8",
        newline="\n",
        backup_existing=target.exists(),
    )
    return result


def load_renderer_calibration(path: Path) -> RendererCalibrationProfile:
    resolved = path.resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1 or raw.get("profile_kind") != PROFILE_KIND:
        raise ValueError(f"unsupported renderer calibration profile: {resolved}")
    entries = raw.get("entries")
    if not isinstance(entries, Mapping) or "global" not in entries:
        raise ValueError(f"renderer calibration profile has no global entry: {resolved}")
    minimum_support = raw.get("minimum_support")
    if isinstance(minimum_support, bool) or not isinstance(minimum_support, int) or minimum_support < 1:
        raise ValueError(f"renderer calibration minimum_support is invalid: {resolved}")
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ValueError(f"renderer calibration entry is invalid: {key!r}")
        support = entry.get("support")
        residuals = entry.get("residuals_ms")
        if isinstance(support, bool) or not isinstance(support, int) or support < minimum_support:
            raise ValueError(f"renderer calibration entry support is invalid: {key}")
        if not isinstance(residuals, Mapping) or set(residuals) != set(ANCHOR_FIELDS):
            raise ValueError(f"renderer calibration entry residuals are invalid: {key}")
        if any(
            isinstance(residuals[field], bool)
            or not isinstance(residuals[field], (int, float))
            or not math.isfinite(float(residuals[field]))
            for field in ANCHOR_FIELDS
        ):
            raise ValueError(f"renderer calibration entry residual is not finite: {key}")
    training_input = raw.get("training_input")
    training_sha = str(training_input.get("sha256", "")) if isinstance(training_input, Mapping) else ""
    if len(training_sha) != 64:
        raise ValueError(f"renderer calibration training input hash is invalid: {resolved}")
    training_bank_ids = _nonempty_string_set(raw.get("training_bank_ids"), "training_bank_ids")
    training_format_types = _nonempty_string_set(raw.get("training_format_types"), "training_format_types")
    return RendererCalibrationProfile(
        path=resolved,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
        profile_id=str(raw.get("profile_id", "")),
        minimum_support=minimum_support,
        entries=entries,
        training_input_sha256=training_sha,
        training_bank_ids=training_bank_ids,
        training_format_types=training_format_types,
    )


def select_renderer_calibration(
    profile: RendererCalibrationProfile,
    *,
    template_id: str,
    bank_id: str | None,
    format_type: str | None,
    expression_profile: str | None = None,
    pitch_bucket: str | None = None,
) -> RendererCalibrationSelection:
    candidates = []
    if pitch_bucket:
        candidates.append(f"pitch_template|{pitch_bucket}|{template_id}")
    if expression_profile:
        candidates.append(f"expression_template|{expression_profile}|{template_id}")
    if bank_id:
        candidates.append(f"bank_template|{bank_id}|{template_id}")
    if format_type:
        candidates.append(f"format_template|{format_type}|{template_id}")
    candidates.extend((f"template|{template_id}", "global"))
    for key in candidates:
        entry = profile.entries.get(key)
        if not isinstance(entry, Mapping):
            continue
        residuals = entry["residuals_ms"]
        assert isinstance(residuals, Mapping)
        return RendererCalibrationSelection(
            scope=str(entry["scope"]),
            key=key,
            support=int(entry["support"]),
            residuals_ms={field: float(residuals[field]) for field in ANCHOR_FIELDS},
            shrinkage_weight=float(entry.get("shrinkage_weight", 1.0)),
            statistics={
                name: {field: float(values[field]) for field in ANCHOR_FIELDS}
                for name in (
                    "raw_median_ms",
                    "mad_ms",
                    "p10_ms",
                    "p25_ms",
                    "p75_ms",
                    "p90_ms",
                    "outlier_rate",
                )
                if isinstance((values := entry.get(name)), Mapping)
                and set(values) == set(ANCHOR_FIELDS)
            },
        )
    raise ValueError("renderer calibration profile has no applicable entry")


def apply_renderer_calibration(
    anchors: Mapping[str, float], selection: RendererCalibrationSelection
) -> dict[str, float]:
    if set(anchors) != set(ANCHOR_FIELDS):
        raise ValueError("renderer calibration anchor set is invalid")
    return {field: float(anchors[field]) + float(selection.residuals_ms[field]) for field in ANCHOR_FIELDS}


def _field_stat(rows: Sequence[Mapping[str, float]], function) -> dict[str, float]:
    return {
        field: float(function(np.asarray([row[field] for row in rows], dtype=np.float64)))
        for field in ANCHOR_FIELDS
    }


def _scope_order(key: str) -> tuple[int, str]:
    scope = key.split("|", 1)[0]
    return (
        {
            "global": 0,
            "template": 1,
            "format_template": 2,
            "bank_template": 3,
            "expression_template": 4,
            "pitch_template": 5,
        }[scope],
        key,
    )


def _parent_key(
    key: str,
    entries: Mapping[str, object],
    parent_candidates: Mapping[str, Mapping[str, set[str]]] | None = None,
) -> str | None:
    parts = key.split("|")
    if parts[0] == "global":
        return None
    if parts[0] == "template":
        return "global"
    candidates = parent_candidates.get(key, {}) if parent_candidates is not None else {}
    for level in ("expression", "bank", "format", "template"):
        available = sorted(candidate for candidate in candidates.get(level, set()) if candidate in entries)
        if len(available) == 1:
            return available[0]
    if parts[0] == "format_template":
        return f"template|{parts[2]}" if f"template|{parts[2]}" in entries else "global"
    if parts[0] in {"bank_template", "expression_template", "pitch_template"}:
        return f"template|{parts[2]}" if f"template|{parts[2]}" in entries else "global"
    raise ValueError(f"unknown renderer calibration scope: {key}")


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _optional_nonempty(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nonempty_string_set(value: object, path: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    result = frozenset(_nonempty(item, f"{path}[]") for item in value)
    if not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _load_jsonl_mappings(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} line {line_number} must be a mapping")
        rows.append(raw)
    return rows


def _anchor_mapping(value: object, path: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(ANCHOR_FIELDS):
        raise ValueError(f"{path} must contain exactly {', '.join(ANCHOR_FIELDS)}")
    result = {}
    for field in ANCHOR_FIELDS:
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"{path}.{field} must be finite")
        result[field] = float(raw)
    return result


def _anchor_order_repair_delta(values: Mapping[str, float]) -> float:
    repaired = float(values[ANCHOR_FIELDS[0]])
    largest = 0.0
    for field in ANCHOR_FIELDS[1:]:
        original = float(values[field])
        repaired = max(repaired, original)
        largest = max(largest, abs(repaired - original))
    return largest


def _oto_metric_values(values: Mapping[str, float]) -> dict[str, float]:
    offset = float(values["offset_abs"])
    return {
        "offset": offset,
        "overlap": float(values["overlap_abs"]) - offset,
        "preutterance": float(values["pre_abs"]) - offset,
        "consonant": float(values["fixed_abs"]) - offset,
        "cutoff_abs": float(values["cutoff_abs"]),
    }
