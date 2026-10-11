from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import file_sha256
from .evaluation import WavDirectoryIndex, _read_timing_records, _record_key
from .oto_io import write_lines_atomic


ANCHOR_FIELDS = ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")


def evaluate_dataset_manifest(
    manifest_path: Path,
    *,
    candidate_field: str = "auto_oto",
    output_json: Path | None = None,
    language: str | None = None,
    format_type: str | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest의 items가 배열이 아닙니다")

    aggregate_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    role_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    group_role_errors: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    pair_results: list[dict[str, Any]] = []
    recording_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    wav_index_cache: dict[str, WavDirectoryIndex] = {}
    selected = 0
    for item_index, item in enumerate(items):
        item_language = str(item.get("language", "unknown"))
        item_format = str(item.get("format_type", "unknown"))
        if language and item_language.casefold() != language.casefold():
            continue
        if format_type and item_format.casefold() != format_type.casefold():
            continue
        selected += 1
        group = f"{item_language}/{item_format}"
        work_dir = Path(str(item.get("work_dir") or item.get("stage_root") or ""))
        reference_path = Path(str(item.get("manual_oto") or ""))
        candidate_path = Path(str(item.get(candidate_field) or ""))
        try:
            if not reference_path.is_file():
                raise FileNotFoundError(f"manual_oto가 없습니다: {reference_path}")
            if not candidate_path.is_file():
                raise FileNotFoundError(f"candidate OTO가 없습니다: {candidate_path}")
            work_key = str(work_dir.resolve()).casefold()
            wav_index = wav_index_cache.get(work_key)
            if wav_index is None:
                wav_index = WavDirectoryIndex(work_dir)
                wav_index_cache[work_key] = wav_index
            candidate_issues: list[dict[str, Any]] = []
            reference_issues: list[dict[str, Any]] = []
            candidate = _read_timing_records(
                candidate_path,
                wav_index.duration,
                identity_resolver=wav_index.identity,
                strict=False,
                issues=candidate_issues,
            )
            reference_parsed = _read_timing_records(
                reference_path,
                wav_index.duration,
                identity_resolver=wav_index.identity,
                strict=False,
                issues=reference_issues,
            )
            reference = [row for row in reference_parsed if not _is_unset_reference(row)]
            reference_unset_rows = len(reference_parsed) - len(reference)
            candidate_map = {_record_key(row): row for row in candidate}
            reference_map = {_record_key(row): row for row in reference}
            candidate_roles = _load_candidate_roles(candidate_path, wav_index)
            shared_keys = set(candidate_map) & set(reference_map)
            pair_errors: dict[str, list[float]] = defaultdict(list)
            pair_role_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            recording_errors: dict[str, dict[str, Any]] = {}
            for key in shared_keys:
                predicted = candidate_map[key]
                target = reference_map[key]
                role = candidate_roles.get(key) or _fallback_role(predicted.alias, item_format)
                wav_identity = key[0]
                recording = recording_errors.setdefault(
                    wav_identity,
                    {
                        "wav": predicted.wav,
                        "wav_path": str(wav_index.resolve(predicted.wav)),
                        "roles": Counter(),
                        "fields": defaultdict(list),
                        "matched_rows": 0,
                    },
                )
                recording["roles"][role] += 1
                recording["matched_rows"] += 1
                row_errors: dict[str, float] = {}
                for field in ANCHOR_FIELDS:
                    error = abs(float(getattr(predicted, field)) - float(getattr(target, field)))
                    row_errors[field] = error
                    recording["fields"][field].append(error)
                    pair_errors[field].append(error)
                    pair_role_errors[role][field].append(error)
                    aggregate_errors[group][field].append(error)
                    aggregate_errors["overall"][field].append(error)
                    role_errors[role][field].append(error)
                    group_role_errors[group][role][field].append(error)
                worst_rows.append(
                    {
                        "item_index": item_index,
                        "language": item_language,
                        "format_type": item_format,
                        "work_dir": str(work_dir.resolve()),
                        "wav": predicted.wav,
                        "alias": predicted.alias,
                        "occurrence": predicted.occurrence,
                        "role": role,
                        "max_anchor_error_ms": max(row_errors.values()),
                        "pre_signed_error_ms": predicted.pre_abs - target.pre_abs,
                        "errors_ms": row_errors,
                    }
                )
            for wav_identity, recording in sorted(recording_errors.items()):
                fields = recording["fields"]
                pooled_recording = [error for field in ANCHOR_FIELDS for error in fields.get(field, [])]
                pre_summary = _summarize(fields.get("pre_abs", []))
                recording_results.append(
                    {
                        "item_index": item_index,
                        "language": item_language,
                        "format_type": item_format,
                        "group": group,
                        "work_dir": str(work_dir.resolve()),
                        "bank_id": str(item.get("bank_id") or work_dir.resolve()),
                        "speaker_id": str(item.get("root_speaker_id") or item.get("speaker_id") or ""),
                        "speaker_metadata_status": (
                            "present" if item.get("root_speaker_id") or item.get("speaker_id") else "missing"
                        ),
                        "candidate_oto": str(candidate_path.resolve()),
                        "wav": recording["wav"],
                        "wav_path": recording["wav_path"],
                        "wav_identity": wav_identity,
                        "matched_rows": recording["matched_rows"],
                        "roles": dict(sorted(recording["roles"].items())),
                        "pooled_anchor_mae_ms": float(np.mean(pooled_recording)) if pooled_recording else 0.0,
                        "max_anchor_error_ms": float(max(pooled_recording)) if pooled_recording else 0.0,
                        "pre_mae_ms": pre_summary["mae"],
                        "pre_gt200_count": pre_summary["gt200_count"],
                        "pre_gt200_rate": pre_summary["gt200_rate"],
                    }
                )
            pair_results.append(
                {
                    "item_index": item_index,
                    "language": item_language,
                    "format_type": item_format,
                    "work_dir": str(work_dir.resolve()),
                    "reference_oto": str(reference_path.resolve()),
                    "candidate_oto": str(candidate_path.resolve()),
                    "candidate_rows": len(candidate),
                    "reference_rows": len(reference),
                    "reference_rows_parsed": len(reference_parsed),
                    "reference_unset_rows": reference_unset_rows,
                    "candidate_parse_issues": candidate_issues,
                    "reference_parse_issues": reference_issues,
                    "matched_rows": len(shared_keys),
                    "candidate_match_rate": len(shared_keys) / max(1, len(candidate_map)),
                    "reference_coverage": len(shared_keys) / max(1, len(reference_map)),
                    "metrics_ms": {field: _summarize(values) for field, values in pair_errors.items()},
                    "metrics_by_role": {
                        role: {field: _summarize(values) for field, values in fields.items()}
                        for role, fields in sorted(pair_role_errors.items())
                    },
                }
            )
        except Exception as error:
            failures.append(
                {
                    "item_index": item_index,
                    "language": item_language,
                    "format_type": item_format,
                    "work_dir": str(work_dir),
                    "reference_oto": str(reference_path),
                    "candidate_oto": str(candidate_path),
                    "error": str(error),
                }
            )

    metrics_by_group = {
        group: {field: _summarize(values) for field, values in fields.items()}
        for group, fields in sorted(aggregate_errors.items())
    }
    overall_fields = aggregate_errors.get("overall", {})
    pooled = [error for field in ANCHOR_FIELDS for error in overall_fields.get(field, [])]
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "reference_field": "manual_oto",
        "candidate_field": candidate_field,
        "filters": {"language": language, "format_type": format_type},
        "selected_items": selected,
        "evaluated_items": len(pair_results),
        "failed_items": len(failures),
        "matched_rows": sum(item["matched_rows"] for item in pair_results),
        "reference_rows": sum(item["reference_rows"] for item in pair_results),
        "reference_rows_parsed": sum(item["reference_rows_parsed"] for item in pair_results),
        "reference_unset_rows": sum(item["reference_unset_rows"] for item in pair_results),
        "candidate_parse_issues": sum(len(item["candidate_parse_issues"]) for item in pair_results),
        "reference_parse_issues": sum(len(item["reference_parse_issues"]) for item in pair_results),
        "candidate_rows": sum(item["candidate_rows"] for item in pair_results),
        "anchor_mae_ms": float(np.mean(pooled)) if pooled else None,
        "metrics_by_group": metrics_by_group,
        "metrics_by_role": {
            role: {field: _summarize(values) for field, values in fields.items()}
            for role, fields in sorted(role_errors.items())
        },
        "metrics_by_group_role": {
            group: {
                role: {field: _summarize(values) for field, values in fields.items()}
                for role, fields in sorted(roles.items())
            }
            for group, roles in sorted(group_role_errors.items())
        },
        "catastrophic_pre_error_gt_200ms": _threshold_summary(
            overall_fields.get("pre_abs", []), 200.0
        ),
        "worst_rows": sorted(worst_rows, key=lambda row: row["max_anchor_error_ms"], reverse=True)[:100],
        "recording_results": sorted(
            recording_results,
            key=lambda row: (-row["pooled_anchor_mae_ms"], row["item_index"], row["wav_identity"]),
        ),
        "pair_results": pair_results,
        "failures": failures,
        "evaluation_policy": {
            "gold_timing_used_for_generation": False,
            "matching_key": "resolved_wav_path_alias_occurrence",
            "negative_cutoff": "offset_plus_absolute_cutoff",
            "blank_numeric_field": 0.0,
            "all_zero_reference_row": "excluded_as_unset",
            "row_parse_or_missing_wav_error": "excluded_and_reported",
        },
    }
    if output_json:
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        write_lines_atomic(output_json.resolve(), payload.splitlines(), encoding="utf-8", newline="\n")
    return result


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0, "mae": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0,
            "gt50_count": 0, "gt50_rate": 0.0, "gt80_count": 0, "gt80_rate": 0.0,
            "gt100_count": 0, "gt100_rate": 0.0, "gt200_count": 0, "gt200_rate": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    summary = {
        "count": len(values),
        "mae": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }
    for threshold in (50, 80, 100, 200):
        count = int(np.sum(array > threshold))
        summary[f"gt{threshold}_count"] = count
        summary[f"gt{threshold}_rate"] = count / len(array)
    return summary


def _is_unset_reference(row) -> bool:
    return all(
        abs(float(getattr(row, field))) < 1e-9
        for field in ANCHOR_FIELDS
    )


def _threshold_summary(values: list[float], threshold_ms: float) -> dict[str, float | int]:
    count = sum(float(value) > threshold_ms for value in values)
    return {"threshold_ms": threshold_ms, "count": count, "rate": count / max(1, len(values))}


def _load_candidate_roles(candidate_path: Path, wav_index: WavDirectoryIndex) -> dict[tuple[str, str, int], str]:
    row_report = candidate_path.parent / "row_report.jsonl"
    if not row_report.is_file():
        return {}
    occurrences: Counter[tuple[str, str]] = Counter()
    roles: dict[tuple[str, str, int], str] = {}
    for raw in row_report.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            wav_key = wav_index.identity(str(row.get("wav", "")))
            alias = str(row.get("alias", ""))
            base_key = (wav_key, alias)
            occurrence = occurrences[base_key]
            occurrences[base_key] += 1
            roles[(wav_key, alias, occurrence)] = str(row.get("role") or "other").casefold()
        except Exception:
            continue
    return roles


def _fallback_role(alias: str, format_type: str) -> str:
    value = alias.strip().casefold()
    parts = value.split()
    if value.startswith("br"):
        return "breath"
    if value.startswith("-"):
        return "cv_head"
    if len(parts) >= 2 and parts[-1] == "r":
        return "release"
    if len(parts) >= 2:
        return "vcv" if format_type.casefold() == "vcv" else "vc_or_vv"
    return "cv"
