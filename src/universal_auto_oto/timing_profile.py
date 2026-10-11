from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import file_sha256
from .dataset_evaluation import ANCHOR_FIELDS, _is_unset_reference
from .evaluation import WavDirectoryIndex, _read_timing_records, _record_key
from .oto_io import write_lines_atomic
from .planning import normalize_wav_key


BUILTIN_PROFILE_ID = "builtin:cv-cutoff-v1"
BUILTIN_VALIDATED_PROFILES: dict[str, dict[str, Any]] = {
    "japanese/cv/cv": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.7026669929043308}},
    "japanese/cv/cv_head": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.6105139254244846}},
    "japanese/cv/vc": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 1.2250250762705364}},
    "korean/cv/cv": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.6558131455027738}},
    "korean/cv/cv_head": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.3163028733572981}},
    "korean/cv/release": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.897744741935145}},
    "korean/cv/vc": {"apply_fields": ["cutoff_abs"], "ratios": {"cutoff_abs": 0.44626124524691085}},
}


def fit_timing_profile(
    candidate_manifests: list[Path],
    output_json: Path,
    *,
    minimum_support: int = 20,
) -> dict[str, Any]:
    samples: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    issue_counts: Counter[str] = Counter()
    source_items = 0
    matched_rows = 0
    source_manifest_records: list[dict[str, str]] = []

    for candidate_manifest_path in candidate_manifests:
        candidate_manifest_path = candidate_manifest_path.resolve()
        source_manifest_records.append(
            {"path": str(candidate_manifest_path), "sha256": file_sha256(candidate_manifest_path)}
        )
        payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"candidate manifest items가 배열이 아닙니다: {candidate_manifest_path}")
        for item in items:
            candidate_oto = Path(str(item.get("candidate_oto") or ""))
            manual_oto = Path(str(item.get("manual_oto") or ""))
            work_dir = Path(str(item.get("work_dir") or item.get("stage_root") or ""))
            row_report = candidate_oto.parent / "row_report.jsonl"
            timeline_dir = candidate_oto.parent / "timelines"
            if not candidate_oto.is_file() or not manual_oto.is_file() or not row_report.is_file() or not timeline_dir.is_dir():
                issue_counts["missing_candidate_artifact"] += 1
                continue
            source_items += 1
            wav_index = WavDirectoryIndex(work_dir)
            reference_issues: list[dict[str, Any]] = []
            reference = _read_timing_records(
                manual_oto,
                wav_index.duration,
                identity_resolver=wav_index.identity,
                strict=False,
                issues=reference_issues,
            )
            issue_counts["reference_parse_issue"] += len(reference_issues)
            reference_map = {
                _record_key(row): row
                for row in reference
                if not _is_unset_reference(row)
            }
            timelines: dict[str, dict[str, Any]] = {}
            for timeline_path in timeline_dir.glob("*.timeline.json"):
                timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
                timelines[normalize_wav_key(str(timeline.get("wav", "")))] = timeline
            occurrences: Counter[tuple[str, str]] = Counter()
            for raw in row_report.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                wav_name = str(row.get("wav", ""))
                alias = str(row.get("alias", ""))
                try:
                    wav_key = wav_index.identity(wav_name)
                except Exception:
                    issue_counts["candidate_wav_resolution_issue"] += 1
                    continue
                base_key = (wav_key, alias)
                occurrence = occurrences[base_key]
                occurrences[base_key] += 1
                reference_row = reference_map.get((wav_key, alias, occurrence))
                if reference_row is None:
                    continue
                timeline = timelines.get(normalize_wav_key(wav_name))
                if not timeline:
                    issue_counts["timeline_missing"] += 1
                    continue
                slots = timeline.get("slots") or []
                slot_index = int(row.get("slot_index", 0))
                if not 0 <= slot_index < len(slots):
                    issue_counts["slot_index_invalid"] += 1
                    continue
                current = slots[slot_index]
                following = slots[min(slot_index + 1, len(slots) - 1)]
                base = float(current["start_ms"])
                if slot_index + 1 < len(slots):
                    unit = float(following["start_ms"]) - base
                else:
                    unit = float(current["end_ms"]) - base
                if unit < 20.0:
                    issue_counts["slot_unit_too_short"] += 1
                    continue
                ratios = {
                    field: (float(getattr(reference_row, field)) - base) / unit
                    for field in ANCHOR_FIELDS
                }
                if any(not np.isfinite(value) or value < -2.0 or value > 4.0 for value in ratios.values()):
                    issue_counts["ratio_outlier"] += 1
                    continue
                key = (
                    str(item.get("language", "unknown")).casefold(),
                    str(item.get("format_type", "unknown")).casefold(),
                    str(row.get("role", "unknown")).casefold(),
                )
                for field, value in ratios.items():
                    samples[key][field].append(value)
                matched_rows += 1

    profiles: dict[str, Any] = {}
    for (language, format_type, role), fields in sorted(samples.items()):
        support = min((len(values) for values in fields.values()), default=0)
        if support < minimum_support:
            continue
        profile_key = f"{language}/{format_type}/{role}"
        profiles[profile_key] = {
            "language": language,
            "format_type": format_type,
            "role": role,
            "support": support,
            "ratios": {field: float(np.median(values)) for field, values in fields.items()},
            "ratio_p10": {field: float(np.percentile(values, 10.0)) for field, values in fields.items()},
            "ratio_p90": {field: float(np.percentile(values, 90.0)) for field, values in fields.items()},
        }
    result = {
        "schema_version": 1,
        "profile_kind": "manual_gold_trained_normalized_slot_template",
        "gold_timing_used_at_inference": False,
        "minimum_support": minimum_support,
        "source_items": source_items,
        "matched_training_rows": matched_rows,
        "source_candidate_manifests": source_manifest_records,
        "issue_counts": dict(sorted(issue_counts.items())),
        "profiles": profiles,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    write_lines_atomic(output_json.resolve(), text.splitlines(), encoding="utf-8", newline="\n", backup_existing=output_json.exists())
    return result


def load_timing_profile(
    path: Path | None,
    *,
    include_builtin: bool = True,
) -> tuple[dict[str, Any], str | None]:
    loaded: dict[str, Any] = dict(BUILTIN_VALIDATED_PROFILES) if include_builtin else {}
    if path is None:
        return loaded, None
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"timing profile의 profiles가 객체가 아닙니다: {resolved}")
    loaded.update(profiles)
    return loaded, file_sha256(resolved)


def select_timing_profile(
    profiles: dict[str, Any],
    *,
    language: str | None,
    format_type: str | None,
    role: str,
) -> dict[str, Any] | None:
    key = f"{(language or '').casefold()}/{(format_type or '').casefold()}/{role.casefold()}"
    entry = profiles.get(key)
    return entry if isinstance(entry, dict) else None
