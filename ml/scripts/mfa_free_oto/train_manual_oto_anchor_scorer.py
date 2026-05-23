from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.manual_oto_decoder import (
    CandidateOption,
    build_joint_anchor_options,
    decode_joint_anchor_lattice,
    decode_monotonic_candidate_indices,
)
from core.mfa_free_oto.manual_oto_candidates import (
    ALIAS_FAMILY_VALUES,
    FORMAT_VALUES,
    LANGUAGE_VALUES,
    ROLE_VALUES,
    SCORABLE_ANCHORS,
    ManualOtoCandidateTracks,
    extract_manual_oto_candidate_tracks,
    manual_oto_alias_family,
    manual_oto_candidate_feature_names,
    manual_oto_candidate_features,
)
from core.mfa_free_oto.vowel_island import (
    SlotIslandAssignment,
    VowelIslandDecode,
    VowelIsland,
    assign_slots_to_islands,
    assignment_is_safe,
    extract_vowel_islands,
    fit_islands_to_slot_count,
    island_overlay_events,
    local_offset_from_preutterance,
)
from core.model_context.filename import canonicalize_order_token, parse_filename_context
from core.model_context.manual_oto_anchor import manual_oto_alias_order_terms


DEFAULT_SPLIT_DIR = r"ml_workspace\manual_oto_anchor\splits"
DEFAULT_OUT_DIR = r"ml_workspace\manual_oto_anchor\scorer"


def _load_rows(
    path: Path,
    *,
    max_rows: int | None = None,
    max_wavs: int | None = None,
    sample_wavs: bool = False,
    seed: int = 20260522,
) -> list[dict[str, object]]:
    if sample_wavs and max_wavs is not None:
        all_rows: list[dict[str, object]] = []
        by_wav: dict[str, list[dict[str, object]]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                wav = str(row.get("wav_path", "") or "")
                by_wav[wav].append(row)
        wavs = sorted(by_wav)
        rng = random.Random(int(seed))
        rng.shuffle(wavs)
        keep = set(wavs[: max(0, int(max_wavs))])
        for wav in sorted(keep):
            all_rows.extend(by_wav[wav])
        if max_rows is not None:
            all_rows = all_rows[: max(0, int(max_rows))]
        return all_rows

    rows: list[dict[str, object]] = []
    seen_wavs: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            wav = str(row.get("wav_path", "") or "")
            if max_wavs is not None and wav not in seen_wavs and len(seen_wavs) >= max_wavs:
                continue
            seen_wavs.add(wav)
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _anchors_arg(value: str) -> tuple[str, ...]:
    if not value:
        return SCORABLE_ANCHORS
    anchors = tuple(item.strip() for item in value.split(",") if item.strip())
    bad = [anchor for anchor in anchors if anchor not in SCORABLE_ANCHORS]
    if bad:
        raise ValueError(f"Unsupported anchors for this scorer: {bad}; allowed={SCORABLE_ANCHORS}")
    return anchors


def _thresholds_arg(value: str) -> tuple[float, ...]:
    if not value.strip():
        return (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
    thresholds: list[float] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        threshold = float(text)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Gate threshold must be in [0, 1], got {threshold!r}.")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one gate threshold is required.")
    return tuple(sorted(set(thresholds)))


def _thresholds_by_anchor_arg(value: str) -> dict[str, float]:
    if not value.strip():
        return {}
    thresholds: dict[str, float] = {}
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("Anchor gate threshold entries must use anchor=value.")
        anchor, raw_threshold = (part.strip() for part in text.split("=", 1))
        if anchor not in SCORABLE_ANCHORS:
            raise ValueError(f"Unsupported anchor threshold key: {anchor!r}; allowed={SCORABLE_ANCHORS}")
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Gate threshold must be in [0, 1], got {threshold!r}.")
        thresholds[anchor] = threshold
    return thresholds


def _threshold_for_anchor(anchor: str, default: float, thresholds_by_anchor: dict[str, float] | None) -> float:
    if not thresholds_by_anchor:
        return float(default)
    return float(thresholds_by_anchor.get(anchor, default))


def _target_ms(row: dict[str, object], anchor: str) -> float | None:
    anchors = row.get("anchors_abs_ms") or {}
    if not isinstance(anchors, dict) or anchor not in anchors:
        return None
    try:
        value = float(anchors[anchor])
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _relative_gap_bucket_keys(row: dict[str, object]) -> list[str]:
    language = str(row.get("language", "") or "").strip().lower() or "*"
    format_type = str(row.get("format_type", "") or "").strip().lower() or "*"
    role = str(row.get("alias_role", "") or "").strip().lower() or "*"
    family = manual_oto_alias_family(row.get("alias", ""))
    bank = str(row.get("voicebank_id", "") or "").strip().lower() or "*"
    return [
        f"{language}|{format_type}|{role}|{family}|{bank}",
        f"{language}|{format_type}|{role}|*|{bank}",
        f"{language}|{format_type}|{role}|{family}",
        f"{language}|{format_type}|{role}|*",
        f"{format_type}|{role}|{family}|{bank}",
        f"{format_type}|{role}|*|{bank}",
        f"{format_type}|{role}|*",
        f"{role}|*",
        "*",
    ]


def _relative_prior_stats(values: list[float], *, ratio: bool = False) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float32)
    if not arr.size:
        return {
            "n": 0,
            "median": None,
            "p10": None,
            "p90": None,
            "iqr": None,
            "p90_p10": None,
            "mad": None,
            "outlier_rate": 1.0,
            "stable": False,
        }
    med = float(np.median(arr))
    p10 = float(np.percentile(arr, 10.0))
    p25 = float(np.percentile(arr, 25.0))
    p75 = float(np.percentile(arr, 75.0))
    p90 = float(np.percentile(arr, 90.0))
    iqr = float(p75 - p25)
    spread = float(p90 - p10)
    mad = float(np.median(np.abs(arr - med)))
    if ratio:
        outlier_band = 0.20
        stable = arr.size >= 10 and iqr <= 0.18 and spread <= 0.40 and float(np.mean(np.abs(arr - med) > outlier_band)) <= 0.25
    else:
        outlier_band = 80.0
        stable = arr.size >= 12 and iqr <= 70.0 and spread <= 160.0 and float(np.mean(np.abs(arr - med) > outlier_band)) <= 0.25
    return {
        "n": int(arr.size),
        "median": med,
        "p10": p10,
        "p90": p90,
        "iqr": iqr,
        "p90_p10": spread,
        "mad": mad,
        "outlier_rate": float(np.mean(np.abs(arr - med) > outlier_band)),
        "stable": bool(stable),
    }


def _fit_relative_anchor_priors(rows: list[dict[str, object]]) -> dict[str, object]:
    kinds = (
        "offset_from_preutterance",
        "overlap_from_preutterance",
        "fixed_end_from_preutterance",
        "overlap_ratio_from_offset_preutterance",
    )
    samples: dict[str, dict[str, list[float]]] = {kind: defaultdict(list) for kind in kinds}
    for row in rows:
        pre = _target_ms(row, "preutterance")
        if pre is None:
            continue
        targets = {
            "offset_from_preutterance": _target_ms(row, "offset"),
            "overlap_from_preutterance": _target_ms(row, "overlap"),
            "fixed_end_from_preutterance": _target_ms(row, "fixed_end"),
        }
        for kind, target in targets.items():
            if target is None:
                continue
            if kind == "fixed_end_from_preutterance":
                gap = float(target) - float(pre)
            else:
                gap = float(pre) - float(target)
            if not np.isfinite(gap) or gap < -50.0 or gap > 2000.0:
                continue
            for key in _relative_gap_bucket_keys(row):
                samples[kind][key].append(float(gap))
        offset = _target_ms(row, "offset")
        overlap = _target_ms(row, "overlap")
        if offset is not None and overlap is not None:
            span = float(pre) - float(offset)
            if span > 8.0:
                ratio = (float(overlap) - float(offset)) / span
                if np.isfinite(ratio) and -0.10 <= ratio <= 1.25:
                    for key in _relative_gap_bucket_keys(row):
                        samples["overlap_ratio_from_offset_preutterance"][key].append(float(ratio))

    buckets: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    stats: dict[str, dict[str, dict[str, object]]] = {}
    for kind, by_key in samples.items():
        buckets[kind] = {key: float(median(values)) for key, values in by_key.items() if values}
        counts[kind] = {key: int(len(values)) for key, values in by_key.items() if values}
        stats[kind] = {
            key: _relative_prior_stats(values, ratio=kind == "overlap_ratio_from_offset_preutterance")
            for key, values in by_key.items()
            if values
        }
    return {
        "schema_version": "relative_anchor_gap_priors_v3",
        "buckets": buckets,
        "counts": counts,
        "stats": stats,
    }


def _relative_anchor_prior_record(
    priors: dict[str, object] | None,
    kind: str,
    row: dict[str, object],
    *,
    require_stable: bool = False,
) -> dict[str, object] | None:
    if not isinstance(priors, dict):
        return None
    buckets = priors.get("buckets")
    if not isinstance(buckets, dict):
        return None
    by_kind = buckets.get(kind)
    if not isinstance(by_kind, dict):
        return None
    stats_by_kind = priors.get("stats")
    kind_stats = stats_by_kind.get(kind) if isinstance(stats_by_kind, dict) else {}
    if not isinstance(kind_stats, dict):
        kind_stats = {}
    first_valid: dict[str, object] | None = None
    for key in _relative_gap_bucket_keys(row):
        value = by_kind.get(key)
        if value is None:
            continue
        try:
            gap = float(value)
        except Exception:
            continue
        if not np.isfinite(gap):
            continue
        stat = kind_stats.get(key)
        if not isinstance(stat, dict):
            stat = {}
        record = {
            "kind": kind,
            "key": key,
            "value": gap,
            "stats": stat,
            "stable": bool(stat.get("stable", False)),
        }
        if first_valid is None:
            first_valid = record
        if bool(require_stable):
            return record if bool(record["stable"]) else None
        if not require_stable:
            return record
    if require_stable:
        return None
    return first_valid


def _relative_anchor_gap_ms(
    priors: dict[str, object] | None,
    kind: str,
    row: dict[str, object],
    *,
    require_stable: bool = False,
) -> float | None:
    record = _relative_anchor_prior_record(priors, kind, row, require_stable=bool(require_stable))
    if record is None:
        return None
    try:
        gap = float(record.get("value"))
    except Exception:
        return None
    if np.isfinite(gap):
        return gap
    return None


def _relative_anchor_ratio(
    priors: dict[str, object] | None,
    kind: str,
    row: dict[str, object],
    *,
    require_stable: bool = False,
) -> float | None:
    value = _relative_anchor_gap_ms(priors, kind, row, require_stable=bool(require_stable))
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _relative_prior_bucket_summary(priors: dict[str, object]) -> dict[str, object]:
    buckets = priors.get("buckets", {})
    stats = priors.get("stats", {})
    if not isinstance(buckets, dict):
        buckets = {}
    if not isinstance(stats, dict):
        stats = {}
    stable_counts: dict[str, int] = {}
    for kind, by_key in stats.items():
        if not isinstance(by_key, dict):
            continue
        stable_counts[str(kind)] = int(
            sum(1 for item in by_key.values() if isinstance(item, dict) and bool(item.get("stable", False)))
        )
    return {
        "schema_version": str(priors.get("schema_version", "")),
        "bucket_counts": {
            kind: len(values) if isinstance(values, dict) else 0
            for kind, values in dict(buckets).items()
        },
        "stable_bucket_counts": stable_counts,
    }


def _group_rows_by_wav(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_wav: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_wav[str(row.get("wav_path", "") or "")].append(row)
    return by_wav


def _centered_slot_pos(row: dict[str, object]) -> float:
    try:
        slot_index = int(row.get("slot_index", 0) or 0)
        slot_count = int(row.get("slot_count", 1) or 1)
    except Exception:
        return float(row.get("slot_pos_norm", 0.0) or 0.0)
    if slot_count <= 1:
        return 0.5
    return min(0.98, max(0.02, (slot_index + 0.5) / max(1.0, float(slot_count))))


def _row_with_slot_pos(row: dict[str, object], slot_pos_norm: float) -> dict[str, object]:
    current = row.get("slot_pos_norm")
    try:
        if abs(float(current) - float(slot_pos_norm)) <= 1e-9:
            return row
    except Exception:
        pass
    out = dict(row)
    out["slot_pos_norm"] = min(0.98, max(0.02, float(slot_pos_norm)))
    return out


def _filename_tokens_for_row(row: dict[str, object]) -> list[str]:
    raw = row.get("filename_canonical_tokens") or []
    if isinstance(raw, list) and raw:
        return [canonicalize_order_token(item) for item in raw if canonicalize_order_token(item)]
    wav_name = str(row.get("wav_name", "") or Path(str(row.get("wav_path", "") or "")).name)
    language = str(row.get("language", "") or "")
    context = parse_filename_context(wav_name, language=language, format_type=str(row.get("format_type", "") or ""))
    return [canonicalize_order_token(item) for item in context.canonical_tokens if canonicalize_order_token(item)]


def _infer_filename_slot_pos(
    row: dict[str, object],
    tokens: list[str],
    *,
    require_primary_transition_token: bool = False,
) -> float | None:
    if not tokens:
        return None
    terms = manual_oto_alias_order_terms(
        row.get("alias", ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias_role=str(row.get("alias_role", "") or ""),
    )
    if not terms:
        return None
    fallback_pos = _centered_slot_pos(row)
    role = str(row.get("alias_role", "") or "").strip().lower()
    primary_term = terms[0] if terms else ""
    primary_best_score = 0
    best_idx: int | None = None
    best_score = -1
    for idx, token in enumerate(tokens):
        score = 0
        for term_idx, term in enumerate(terms):
            if not term:
                continue
            if term == token:
                score += 4
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 4)
            elif token.startswith(term) or term.startswith(token):
                score += 2
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 2)
            elif term in token or token in term:
                score += 1
                if term_idx == 0:
                    primary_best_score = max(primary_best_score, 1)
        if score > best_score:
            best_score = score
            best_idx = idx
        elif score == best_score and best_idx is not None:
            current_pos = (idx + 0.5) / max(1.0, float(len(tokens)))
            best_pos = (best_idx + 0.5) / max(1.0, float(len(tokens)))
            if abs(current_pos - fallback_pos) < abs(best_pos - fallback_pos):
                best_idx = idx
    if best_idx is None or best_score <= 0:
        return None
    if bool(require_primary_transition_token) and role in {"vc", "vcv", "vv"} and primary_term and primary_best_score <= 0:
        return None
    return min(0.98, max(0.02, (best_idx + 0.5) / max(1.0, float(len(tokens)))))


def _filename_ordered_rows(
    wav_rows: list[dict[str, object]],
    *,
    require_primary_transition_token: bool = False,
) -> list[tuple[dict[str, object], float]]:
    if not wav_rows:
        return []
    tokens = _filename_tokens_for_row(wav_rows[0])
    ordered: list[tuple[dict[str, object], float, int, bool]] = []
    inferred = 0
    for input_idx, row in enumerate(wav_rows):
        pos = _infer_filename_slot_pos(
            row,
            tokens,
            require_primary_transition_token=bool(require_primary_transition_token),
        )
        if pos is not None:
            inferred += 1
        else:
            pos = _centered_slot_pos(row)
        ordered.append((row, float(pos), input_idx, pos is not None))
    if inferred >= 2:
        ordered.sort(key=lambda item: (item[1], item[2]))
    return [(row, pos) for row, pos, _input_idx, _ok in ordered]


def _row_slot_ordered_rows(wav_rows: list[dict[str, object]]) -> list[tuple[dict[str, object], float]]:
    ordered: list[tuple[dict[str, object], float, int]] = []
    for input_idx, row in enumerate(wav_rows):
        ordered.append((row, _centered_slot_pos(row), input_idx))
    ordered.sort(
        key=lambda item: (
            int(item[0].get("slot_index", item[2]) or item[2]),
            item[2],
        )
    )
    return [(row, pos) for row, pos, _input_idx in ordered]


def _vowel_island_slot_count(wav_rows: list[dict[str, object]]) -> int:
    if not wav_rows:
        return 1
    max_count = 1
    max_index = 0
    for row in wav_rows:
        try:
            max_count = max(max_count, int(row.get("slot_count", 1) or 1))
            max_index = max(max_index, int(row.get("slot_index", 0) or 0))
        except Exception:
            continue
    manifest_count = max(1, max(max_count, max_index + 1))
    tokens = _filename_tokens_for_row(wav_rows[0])
    format_type = str(wav_rows[0].get("format_type", "") or "").strip().lower()
    if format_type in {"cv", "cvvc", "cvc"} and len(tokens) == manifest_count + 1:
        inferred_token_indices: list[int] = []
        for row in wav_rows:
            pos = _infer_filename_slot_pos(row, tokens)
            if pos is None:
                continue
            idx = int(round(float(pos) * float(len(tokens)) - 0.5))
            if 0 <= idx < len(tokens):
                inferred_token_indices.append(idx)
        if inferred_token_indices and min(inferred_token_indices) >= 1:
            return len(tokens)
    if manifest_count >= 2:
        return manifest_count
    if len(tokens) >= 2:
        return len(tokens)
    return manifest_count


def _row_vowel_island_slot_index(row: dict[str, object], slot_count: int) -> int:
    count = max(1, int(slot_count))
    tokens = _filename_tokens_for_row(row)
    try:
        row_count = max(1, int(row.get("slot_count", count) or count))
        row_index = int(row.get("slot_index", 0) or 0)
        if count != row_count and len(tokens) == count:
            pos = _infer_filename_slot_pos(row, tokens)
            if pos is not None:
                idx = int(round(float(pos) * float(count) - 0.5))
                return max(0, min(count - 1, idx))
        if 0 <= row_index < row_count:
            pos = (float(row_index) + 0.5) / max(1.0, float(row_count))
            idx = int(round(pos * float(count) - 0.5))
            return max(0, min(count - 1, idx))
    except Exception:
        pass
    pos = _infer_filename_slot_pos(row, tokens)
    if pos is None:
        pos = _centered_slot_pos(row)
    idx = int(round(float(pos) * float(count) - 0.5))
    return max(0, min(count - 1, idx))


def _default_fixed_gap_ms(row: dict[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    fmt = str(row.get("format_type", "") or "").strip().lower()
    if role in {"vc", "vv"}:
        return 95.0
    if role == "vcv":
        return 115.0
    if fmt in {"cv", "cvc"}:
        return 170.0
    return 135.0


def _island_anchor_bucket_keys(row: dict[str, object], anchor: str) -> list[str]:
    language = str(row.get("language", "") or "").strip().lower() or "*"
    format_type = str(row.get("format_type", "") or "").strip().lower() or "*"
    role = str(row.get("alias_role", "") or "").strip().lower() or "*"
    family = manual_oto_alias_family(row.get("alias", ""))
    anchor = str(anchor)
    return [
        f"{language}|{format_type}|{role}|{family}|{anchor}",
        f"{language}|{format_type}|{role}|*|{anchor}",
        f"{format_type}|{role}|*|{anchor}",
        f"{role}|*|{anchor}",
        f"*|{anchor}",
    ]


def _fit_island_anchor_position_priors(
    rows: list[dict[str, object]],
    *,
    anchors: tuple[str, ...],
    encoder: str,
    min_score: float,
    top_k_fallback: int,
) -> dict[str, object]:
    cache: dict[str, ManualOtoCandidateTracks] = {}
    samples: dict[str, list[float]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for wav_path, wav_rows in _group_rows_by_wav(rows).items():
        try:
            tracks = _extract_cached(
                cache,
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception:
            stats["wav_failures"] += 1
            continue
        slot_count = _vowel_island_slot_count(wav_rows)
        raw_islands = extract_vowel_islands(tracks)
        islands = fit_islands_to_slot_count(tracks, raw_islands, slot_count=slot_count)
        decode = assign_slots_to_islands(islands, slot_count=slot_count, duration_ms=float(tracks.duration_ms))
        stats["wavs"] += 1
        if len(raw_islands) == int(slot_count):
            stats["raw_count_match_wavs"] += 1
        else:
            stats["raw_count_mismatch_wavs"] += 1
            continue
        for row in wav_rows:
            assignment = _island_assignment_for_row(row, decode.assignments, slot_count)
            island = _island_for_assignment(assignment, decode.islands)
            if island is None:
                stats["missing_island"] += 1
                continue
            span = max(20.0, float(island.end_ms) - float(island.start_ms))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                ratio = (float(target) - float(island.start_ms)) / span
                if not np.isfinite(ratio) or ratio < -2.0 or ratio > 3.0:
                    stats[f"{anchor}:ratio_outlier"] += 1
                    continue
                for key in _island_anchor_bucket_keys(row, anchor):
                    samples[key].append(float(ratio))
                stats[f"{anchor}:samples"] += 1
    buckets: dict[str, float] = {}
    counts: dict[str, int] = {}
    for key, values in samples.items():
        if not values:
            continue
        buckets[key] = float(median(values))
        counts[key] = int(len(values))
    return {
        "schema_version": "island_anchor_position_priors_v1",
        "buckets": buckets,
        "counts": counts,
        "stats": dict(stats),
    }


def _island_anchor_ratio(
    priors: dict[str, object] | None,
    row: dict[str, object],
    anchor: str,
) -> float | None:
    if not isinstance(priors, dict):
        return None
    buckets = priors.get("buckets")
    counts = priors.get("counts")
    if not isinstance(buckets, dict):
        return None
    if not isinstance(counts, dict):
        counts = {}
    for key in _island_anchor_bucket_keys(row, anchor):
        if int(counts.get(key, 0) or 0) < 6:
            continue
        value = buckets.get(key)
        if value is None:
            continue
        try:
            ratio = float(value)
        except Exception:
            continue
        if np.isfinite(ratio):
            return ratio
    return None


def _island_assignment_for_row(
    row: dict[str, object],
    assignments: tuple[SlotIslandAssignment, ...],
    slot_count: int,
) -> SlotIslandAssignment | None:
    if not assignments:
        return None
    idx = _row_vowel_island_slot_index(row, slot_count)
    if idx >= len(assignments):
        return None
    return assignments[idx]


def _island_for_assignment(
    assignment: SlotIslandAssignment | None,
    islands: tuple[VowelIsland, ...],
) -> VowelIsland | None:
    if assignment is None or assignment.island_index is None:
        return None
    idx = int(assignment.island_index)
    if idx < 0 or idx >= len(islands):
        return None
    return islands[idx]


def _candidate_near_expected_ms(
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    expected_ms: float,
    *,
    window_ms: float,
) -> float:
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size == 0:
        return max(0.0, float(expected_ms))
    duration_ms = max(1.0, float(tracks.duration_ms or 1.0))
    expected = min(duration_ms, max(0.0, float(expected_ms)))
    indices = [int(idx) for idx in tracks.candidate_indices.get(anchor, ()) if 0 <= int(idx) < int(times.size)]
    if not indices:
        idx = int(np.argmin(np.abs(times - expected)))
        return float(times[idx])
    idxs = np.asarray(indices, dtype=np.int32)
    distances = np.abs(times[idxs] - expected)
    local_mask = distances <= float(window_ms)
    if np.any(local_mask):
        idxs = idxs[local_mask]
        distances = distances[local_mask]
    score_track = np.asarray(tracks.anchor_scores.get(anchor, np.zeros((times.size,), dtype=np.float32)), dtype=np.float32)
    if score_track.size != times.size:
        score_track = np.resize(score_track, (times.size,)).astype(np.float32)
    local_scores = score_track[idxs]
    score = 0.25 * _unit_vector(local_scores) - 0.75 * np.clip(distances / max(1.0, float(window_ms)), 0.0, 2.0)
    best = int(idxs[int(np.argmax(score))])
    return float(times[best])


def _vcv_initial_cv_preutterance_policy(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    island: VowelIsland | None,
    fallback_pred_ms: float,
    fallback_safe_prob: float | None,
    duration_ms: float,
    slot_count: int,
    slot_index: int,
) -> tuple[float | None, str, str]:
    format_type = str(row.get("format_type", "") or "").strip().lower()
    role = str(row.get("alias_role", "") or "").strip().lower()
    if format_type != "vcv" or role != "cv":
        return None, "", "not_vcv_initial_cv"
    if float(fallback_pred_ms) <= 150.0:
        return float(fallback_pred_ms), "vcv_initial_cv_early_constrained", ""
    local_pre_pred: float | None = None
    local_pre_score = 0.0
    if island is not None:
        local_pre_pred = _candidate_near_expected_ms(
            tracks,
            "preutterance",
            float(island.start_ms),
            window_ms=160.0,
        )
        local_pre_score = _nearest_track_value(
            tracks,
            tracks.anchor_scores.get("preutterance", ()),
            float(local_pre_pred),
        )
    local_delta = (
        _slot_shift_delta(
            row,
            pred_ms=float(local_pre_pred),
            duration_ms=duration_ms,
            slot_count=slot_count,
            slot_index=slot_index,
        )
        if local_pre_pred is not None
        else None
    )
    if (
        island is not None
        and local_pre_pred is not None
        and local_delta == 0
        and float(local_pre_score) >= 0.95
        and float(local_pre_pred) <= float(island.nucleus_ms) + 20.0
    ):
        return float(local_pre_pred), f"vcv_initial_cv_island_start:{local_pre_score:.3f}", ""
    if (
        island is not None
        and fallback_safe_prob is not None
        and float(fallback_safe_prob) >= 0.99
        and float(fallback_pred_ms) - float(island.start_ms) <= 130.0
        and float(fallback_pred_ms) <= float(island.nucleus_ms) + 20.0
    ):
        return float(fallback_pred_ms), f"vcv_initial_cv_constrained:{fallback_safe_prob:.3f}", ""
    return None, "", "vcv_initial_cv_late_preutterance"


def _vcv_sonorant_preutterance_policy(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    island: VowelIsland | None,
    fallback_pred_ms: float,
    fallback_safe_prob: float | None,
    duration_ms: float,
    slot_count: int,
    slot_index: int,
) -> tuple[float | None, str, str]:
    format_type = str(row.get("format_type", "") or "").strip().lower()
    role = str(row.get("alias_role", "") or "").strip().lower()
    leading = str(row.get("alias_norm") or row.get("alias") or "").strip().lower().split()
    leading_token = leading[0].strip("-_") if leading else ""
    if format_type != "vcv" or role == "cv" or leading_token not in {"l", "r", "m", "n", "ng"}:
        return None, "", "not_vcv_sonorant"
    safe_prob = float(fallback_safe_prob) if fallback_safe_prob is not None else 0.0
    fallback_delta = _slot_shift_delta(
        row,
        pred_ms=float(fallback_pred_ms),
        duration_ms=duration_ms,
        slot_count=slot_count,
        slot_index=slot_index,
    )
    if fallback_delta != 0:
        return None, "", f"vcv_sonorant_slot_not_exact:{_slot_shift_bucket(fallback_delta)}"
    if island is None:
        if safe_prob >= 0.98:
            return float(fallback_pred_ms), f"vcv_sonorant_safe_constrained:{safe_prob:.3f}", ""
        return None, "", "vcv_sonorant_missing_island"

    span_ms = max(1.0, float(island.end_ms) - float(island.start_ms))
    if (
        leading_token in {"n", "ng"}
        and float(island.confidence) >= 0.90
        and safe_prob >= 0.90
    ):
        nucleus_pred = _candidate_near_expected_ms(
            tracks,
            "preutterance",
            float(island.nucleus_ms),
            window_ms=180.0,
        )
        nucleus_score = _nearest_track_value(
            tracks,
            tracks.anchor_scores.get("preutterance", ()),
            float(nucleus_pred),
        )
        nucleus_delta = _slot_shift_delta(
            row,
            pred_ms=float(nucleus_pred),
            duration_ms=duration_ms,
            slot_count=slot_count,
            slot_index=slot_index,
        )
        if (
            nucleus_delta == 0
            and float(nucleus_score) >= 0.95
            and abs(float(nucleus_pred) - float(fallback_pred_ms)) <= 90.0
        ):
            return float(nucleus_pred), f"vcv_sonorant_nasal_nucleus:{nucleus_score:.3f}", ""

    fallback_outside_island = (
        float(fallback_pred_ms) <= float(island.start_ms) or float(fallback_pred_ms) >= float(island.end_ms)
    )
    if safe_prob >= 0.98 and (float(island.confidence) >= 0.85 or fallback_outside_island):
        return float(fallback_pred_ms), f"vcv_sonorant_safe_constrained:{safe_prob:.3f}", ""
    if safe_prob >= 0.95 and fallback_outside_island:
        return float(fallback_pred_ms), f"vcv_sonorant_outside_island:{safe_prob:.3f}", ""
    if safe_prob >= 0.95:
        r33_pred = _candidate_near_expected_ms(
            tracks,
            "preutterance",
            float(island.start_ms) + 0.33 * span_ms,
            window_ms=180.0,
        )
        r33_score = _nearest_track_value(
            tracks,
            tracks.anchor_scores.get("preutterance", ()),
            float(r33_pred),
        )
        if abs(float(r33_pred) - float(fallback_pred_ms)) <= 35.0 and float(r33_score) >= 0.95:
            return float(fallback_pred_ms), f"vcv_sonorant_r33_agree:{r33_score:.3f}", ""
    return None, "", "vcv_sonorant_transition_review"


def _slot_exact_island_start_preutterance_policy(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    island: VowelIsland | None,
    fallback_pred_ms: float,
    fallback_safe_prob: float | None,
    duration_ms: float,
    slot_count: int,
    slot_index: int,
) -> tuple[float | None, str, str]:
    role = str(row.get("alias_role", "") or "").strip().lower()
    format_type = str(row.get("format_type", "") or "").strip().lower()
    if role not in {"cv", "vcv"}:
        return None, "", "not_slot_exact_island_start_preutterance"
    leading = str(row.get("alias_norm") or row.get("alias") or "").strip().lower().split()
    leading_token = leading[0].strip("-_") if leading else ""
    if format_type == "vcv" and role != "cv" and leading_token in {"l", "r", "m", "n", "ng"}:
        return None, "", "not_slot_exact_island_start_preutterance"
    if island is None:
        return None, "", "slot_exact_island_start_missing_island"
    safe_prob = float(fallback_safe_prob) if fallback_safe_prob is not None else 0.0
    if safe_prob < 0.84:
        return None, "", "slot_exact_island_start_low_margin"
    fallback_delta = _slot_shift_delta(
        row,
        pred_ms=float(fallback_pred_ms),
        duration_ms=duration_ms,
        slot_count=slot_count,
        slot_index=slot_index,
    )
    if fallback_delta != 0:
        return None, "", f"slot_exact_island_start_fallback_not_exact:{_slot_shift_bucket(fallback_delta)}"
    if float(island.confidence) < 0.88:
        return None, "", "slot_exact_island_start_low_island_confidence"
    local_pred = _candidate_near_expected_ms(
        tracks,
        "preutterance",
        float(island.start_ms),
        window_ms=140.0,
    )
    local_delta = _slot_shift_delta(
        row,
        pred_ms=float(local_pred),
        duration_ms=duration_ms,
        slot_count=slot_count,
        slot_index=slot_index,
    )
    if local_delta != 0:
        return None, "", f"slot_exact_island_start_local_not_exact:{_slot_shift_bucket(local_delta)}"
    local_score = _nearest_track_value(
        tracks,
        tracks.anchor_scores.get("preutterance", ()),
        float(local_pred),
    )
    fallback_score = _nearest_track_value(
        tracks,
        tracks.anchor_scores.get("preutterance", ()),
        float(fallback_pred_ms),
    )
    boundary_floor = float(island.start_ms) - 80.0
    boundary_ceiling = float(island.nucleus_ms) + (45.0 if role == "vcv" or format_type == "vcv" else 30.0)
    if not (boundary_floor <= float(local_pred) <= boundary_ceiling):
        return None, "", "slot_exact_island_start_local_outside_boundary_window"
    if abs(float(local_pred) - float(fallback_pred_ms)) > (85.0 if role == "vcv" or format_type == "vcv" else 65.0):
        return None, "", "slot_exact_island_start_disagreement"
    if float(local_score) < 0.95:
        return None, "", "slot_exact_island_start_low_local_score"
    if float(fallback_score) >= float(local_score) + 0.03 and boundary_floor <= float(fallback_pred_ms) <= boundary_ceiling:
        return float(fallback_pred_ms), f"slot_exact_constrained_preutterance:{safe_prob:.3f}", ""
    return float(local_pred), f"slot_exact_island_start_preutterance:{local_score:.3f}", ""


def _unit_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _predict_island_anchor_ms(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    anchor: str,
    island: VowelIsland,
    relative_anchor_priors: dict[str, object] | None,
    island_anchor_priors: dict[str, object] | None = None,
) -> float | None:
    pre_ms = max(0.0, float(island.start_ms))
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    ratio = _island_anchor_ratio(island_anchor_priors, row, anchor)
    if ratio is not None:
        span_ms = max(20.0, float(island.end_ms) - float(island.start_ms))
        expected = float(island.start_ms) + float(ratio) * span_ms
        window = 90.0 if anchor in {"preutterance", "fixed_end"} else 130.0
        return min(duration_ms, max(0.0, _candidate_near_expected_ms(tracks, anchor, expected, window_ms=window)))
    if anchor == "preutterance":
        return min(duration_ms, pre_ms)
    if anchor == "fixed_end":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "fixed_end_from_preutterance", row)
        if gap is None:
            gap = _default_fixed_gap_ms(row)
        return min(duration_ms, max(pre_ms + 20.0, min(float(island.end_ms), pre_ms + float(gap))))
    if anchor == "overlap":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "overlap_from_preutterance", row)
        if gap is None:
            return None
        return min(duration_ms, max(0.0, pre_ms - float(gap)))
    if anchor == "offset":
        gap = _relative_anchor_gap_ms(relative_anchor_priors, "offset_from_preutterance", row)
        if gap is None:
            return None
        return local_offset_from_preutterance(tracks, preutterance_ms=pre_ms, expected_gap_ms=float(gap))
    return None


def _supervised_safe_probability(
    row: dict[str, object],
    anchor: str,
    prediction: dict[str, object] | None,
    *,
    gate_model: object | None,
    filename_pred: dict[str, object] | None,
    joint_pred: dict[str, object] | None,
    require_primary_transition_token: bool,
) -> float | None:
    if gate_model is None or prediction is None:
        return None
    filename_tokens = _filename_tokens_for_row(row)
    filename_slot_pos = _infer_filename_slot_pos(
        row,
        filename_tokens,
        require_primary_transition_token=bool(require_primary_transition_token),
    )
    gate_features = _gate_features(
        row,
        anchor,
        prediction,
        filename_pred=filename_pred,
        joint_pred=joint_pred,
        filename_slot_pos=filename_slot_pos,
        filename_tokens=filename_tokens,
    )
    try:
        return float(gate_model.predict_proba(gate_features.reshape(1, -1))[0, 1])
    except Exception:
        try:
            return float(gate_model.predict(gate_features.reshape(1, -1))[0])
        except Exception:
            return None


def _routed_preutterance_decision(
    row: dict[str, object],
    *,
    constrained_predictions: dict[tuple[int, str], dict[str, object]],
    filename_slot_predictions: dict[tuple[int, str], dict[str, object]],
    joint_predictions: dict[tuple[int, str], dict[str, object]],
    supervised_gate_models: dict[str, object] | None,
    supervised_gate_threshold: float,
    supervised_gate_thresholds_by_anchor: dict[str, float] | None,
    require_primary_transition_token: bool,
    use_heuristic_gate: bool,
    gate_slot_disagreement: float,
    gate_pred_disagreement_ms: float,
    gate_joint_disagreement_ms: float,
    gate_order_disagreement: float,
    gate_joint_score_min: float,
    gate_review_mojibake: bool,
    gate_review_duplicate_tokens: bool,
) -> tuple[float | None, float | None, tuple[str, ...]]:
    primary = constrained_predictions.get((id(row), "preutterance"))
    filename_tokens = _filename_tokens_for_row(row)
    filename_slot_pos = _infer_filename_slot_pos(
        row,
        filename_tokens,
        require_primary_transition_token=bool(require_primary_transition_token),
    )
    if bool(use_heuristic_gate):
        reasons = _gate_constrained_prediction(
            row,
            "preutterance",
            primary,
            filename_pred=filename_slot_predictions.get((id(row), "preutterance")),
            joint_pred=joint_predictions.get((id(row), "preutterance")),
            filename_slot_pos=filename_slot_pos,
            filename_tokens=filename_tokens,
            slot_disagreement_threshold=float(gate_slot_disagreement),
            pred_disagreement_ms=float(gate_pred_disagreement_ms),
            joint_disagreement_ms=float(gate_joint_disagreement_ms),
            order_disagreement_threshold=float(gate_order_disagreement),
            joint_score_min=float(gate_joint_score_min),
            review_mojibake=bool(gate_review_mojibake),
            review_duplicate_tokens=bool(gate_review_duplicate_tokens),
        )
    else:
        reasons = ["missing_constrained_prediction"] if primary is None else []

    gate_model = (supervised_gate_models or {}).get("preutterance")
    safe_prob = _supervised_safe_probability(
        row,
        "preutterance",
        primary,
        gate_model=gate_model,
        filename_pred=filename_slot_predictions.get((id(row), "preutterance")),
        joint_pred=joint_predictions.get((id(row), "preutterance")),
        require_primary_transition_token=bool(require_primary_transition_token),
    )
    if supervised_gate_models:
        threshold = _threshold_for_anchor("preutterance", float(supervised_gate_threshold), supervised_gate_thresholds_by_anchor)
        if gate_model is None:
            reasons.append("missing_supervised_gate_model")
        elif safe_prob is None or safe_prob < float(threshold):
            reasons.append("supervised_failure_risk")

    if reasons or primary is None:
        return None, safe_prob, tuple(dict.fromkeys(reasons or ["missing_constrained_prediction"]))
    return float(primary.get("pred_ms", 0.0) or 0.0), safe_prob, ()


def _dependent_preutterance_ms(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    constrained_predictions: dict[tuple[int, str], dict[str, object]],
    filename_slot_predictions: dict[tuple[int, str], dict[str, object]],
    joint_predictions: dict[tuple[int, str], dict[str, object]],
    supervised_gate_models: dict[str, object] | None,
    supervised_gate_threshold: float,
    supervised_gate_thresholds_by_anchor: dict[str, float] | None,
    require_primary_transition_token: bool,
    fallback_pre_ms: float | None = None,
    slot_count: int | None = None,
    slot_index: int | None = None,
) -> tuple[float | None, str, float | None]:
    primary = constrained_predictions.get((id(row), "preutterance"))
    threshold = _threshold_for_anchor("preutterance", float(supervised_gate_threshold), supervised_gate_thresholds_by_anchor)
    safe_prob = _supervised_safe_probability(
        row,
        "preutterance",
        primary,
        gate_model=(supervised_gate_models or {}).get("preutterance"),
        filename_pred=filename_slot_predictions.get((id(row), "preutterance")),
        joint_pred=joint_predictions.get((id(row), "preutterance")),
        require_primary_transition_token=bool(require_primary_transition_token),
    )
    if primary is not None:
        primary_ms = float(primary.get("pred_ms", 0.0) or 0.0)
        if safe_prob is not None and safe_prob >= float(threshold):
            return primary_ms, "gated_constrained_preutterance", safe_prob
        primary_delta = _slot_shift_delta(
            row,
            pred_ms=primary_ms,
            duration_ms=float(tracks.duration_ms),
            slot_count=slot_count,
            slot_index=slot_index,
        )
        if safe_prob is None and primary_delta == 0:
            return primary_ms, "slot_exact_constrained_preutterance", safe_prob
        return primary_ms, "fallback_constrained_preutterance", safe_prob
    if fallback_pre_ms is not None and np.isfinite(float(fallback_pre_ms)):
        return float(fallback_pre_ms), "fallback_vowel_island_preutterance", safe_prob
    return None, "missing_preutterance", safe_prob


def _default_offset_gap_ms(row: dict[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if role in {"v", "vv"}:
        return 70.0
    if role in {"vc", "vcv"} or family in {"vowel_transition", "leading_n"}:
        return 150.0
    if family == "breath_silence":
        return 40.0
    return 120.0


def _dependent_offset_ms(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    relative_anchor_priors: dict[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "offset_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is None:
        if bool(require_stable_prior):
            return None
        gap = _default_offset_gap_ms(row)
    expected = float(preutterance_ms) - max(0.0, float(gap))
    window = max(55.0, min(190.0, abs(float(gap)) * 0.75 + 35.0))
    min_gap = 8.0 if str(row.get("alias_role", "") or "").strip().lower() in {"v", "vv"} else 16.0
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size:
        start = max(0.0, expected - window)
        end = min(float(preutterance_ms) - min_gap, expected + window)
        mask = (times >= start) & (times <= end)
        if np.any(mask):
            idxs = np.flatnonzero(mask)
            size = int(times.size)
            offset_score = _unit_vector(
                np.asarray(tracks.anchor_scores.get("offset", np.zeros((size,), dtype=np.float32)), dtype=np.float32)
            )
            transition = _unit_vector(np.asarray(tracks.tracks.get("transition", np.zeros((size,), dtype=np.float32)), dtype=np.float32))
            activity_edge = _unit_vector(
                np.asarray(tracks.tracks.get("activity_edge", np.zeros((size,), dtype=np.float32)), dtype=np.float32)
            )
            distance = np.abs(times[idxs] - expected) / max(1.0, window)
            score = (
                0.08 * offset_score[idxs]
                + 0.06 * transition[idxs]
                + 0.04 * activity_edge[idxs]
                - 1.00 * np.clip(distance, 0.0, 2.0)
            )
            pred = float(times[int(idxs[int(np.argmax(score))])])
        else:
            pred = expected
    else:
        pred = expected
    return min(duration_ms, max(0.0, min(float(pred), float(preutterance_ms) - min_gap)))


def _default_overlap_ratio(row: dict[str, object]) -> float:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    if role in {"v", "vv"}:
        return 0.50
    if role in {"vc", "vcv"} or family in {"vowel_transition", "leading_n"}:
        return 1.0 / 3.0
    if family == "breath_silence":
        return 0.0
    return 1.0 / 3.0


def _overlap_ratio_candidate_keys(row: dict[str, object]) -> list[tuple[str, int, bool]]:
    language = str(row.get("language", "") or "").strip().lower() or "*"
    format_type = str(row.get("format_type", "") or "").strip().lower() or "*"
    role = str(row.get("alias_role", "") or "").strip().lower() or "*"
    family = manual_oto_alias_family(row.get("alias", ""))
    bank = str(row.get("voicebank_id", "") or "").strip().lower() or "*"
    return [
        (f"{language}|{format_type}|{role}|{family}|{bank}", 8, True),
        (f"{language}|{format_type}|{role}|*|{bank}", 10, True),
        (f"{format_type}|{role}|{family}|{bank}", 8, True),
        (f"{format_type}|{role}|*|{bank}", 10, True),
        (f"{language}|{format_type}|{role}|{family}", 10, False),
        (f"{language}|{format_type}|{role}|*", 14, False),
        (f"{format_type}|{role}|*", 14, False),
        (f"{role}|*", 18, False),
        ("*", 24, False),
    ]


def _overlap_span_ratio_prior(
    priors: dict[str, object] | None,
    row: dict[str, object],
) -> tuple[float, str, dict[str, object]]:
    default_ratio = _default_overlap_ratio(row)
    if not isinstance(priors, dict):
        return default_ratio, "default_overlap_ratio", {}
    buckets = priors.get("buckets")
    stats = priors.get("stats")
    if not isinstance(buckets, dict) or not isinstance(stats, dict):
        return default_ratio, "default_overlap_ratio", {}
    by_kind = buckets.get("overlap_ratio_from_offset_preutterance")
    stats_by_kind = stats.get("overlap_ratio_from_offset_preutterance")
    if not isinstance(by_kind, dict) or not isinstance(stats_by_kind, dict):
        return default_ratio, "default_overlap_ratio", {}
    for key, min_count, require_stable_if_bank in _overlap_ratio_candidate_keys(row):
        value = by_kind.get(key)
        stat = stats_by_kind.get(key)
        if value is None or not isinstance(stat, dict):
            continue
        try:
            ratio = float(value)
            n = int(stat.get("n", 0) or 0)
        except Exception:
            continue
        if not np.isfinite(ratio) or n < int(min_count):
            continue
        if require_stable_if_bank and not bool(stat.get("stable", False)):
            continue
        blended = (0.65 * ratio) + (0.35 * default_ratio)
        return min(0.92, max(0.02, float(blended))), key, stat
    return default_ratio, "default_overlap_ratio", {}


def _overlap_span_ratio_ms(
    row: dict[str, object],
    *,
    offset_ms: float,
    preutterance_ms: float,
    relative_anchor_priors: dict[str, object] | None,
) -> tuple[float, str, dict[str, object]]:
    offset = float(offset_ms)
    pre = max(offset, float(preutterance_ms))
    span = max(0.0, pre - offset)
    ratio, source, stat = _overlap_span_ratio_prior(relative_anchor_priors, row)
    pred = offset + ratio * span
    return min(pre, max(offset, float(pred))), source, stat


def _safe_overlap_gap_config(row: dict[str, object], span_ms: float) -> tuple[float, float, float, str]:
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    span = max(0.0, float(span_ms))
    if family == "breath_silence":
        min_gap, role_gap, ratio = 0.0, 0.0, 0.0
        label = "breath_silence"
    elif role in {"vc", "vcv"} or family in {"leading_n", "vowel_transition"}:
        min_gap, role_gap, ratio = 30.0, 85.0, 0.50
        label = "transition"
    elif role in {"v", "vv"}:
        min_gap, role_gap, ratio = 20.0, 55.0, 0.50
        label = "vowel"
    else:
        min_gap, role_gap, ratio = 18.0, 45.0, 0.38
        label = "cv"
    if span <= 0.0:
        return 0.0, 0.0, 0.0, label
    target_gap = min(float(role_gap), span * float(ratio))
    max_gap = min(span, max(float(min_gap), max(float(role_gap), span * 0.72)))
    min_gap = min(float(min_gap), max_gap)
    gap = min(max_gap, max(min_gap, target_gap))
    return float(gap), float(min_gap), float(max_gap), label


def _safe_overlap_ms(
    row: dict[str, object],
    *,
    offset_ms: float,
    preutterance_ms: float,
    preferred_overlap_ms: float | None = None,
) -> tuple[float, dict[str, object]]:
    offset = float(offset_ms)
    pre = max(offset, float(preutterance_ms))
    span = max(0.0, pre - offset)
    gap, min_gap, max_gap, label = _safe_overlap_gap_config(row, span)
    if preferred_overlap_ms is None or not np.isfinite(float(preferred_overlap_ms)):
        preferred_overlap = pre - gap
    else:
        preferred_overlap = min(pre, max(offset, float(preferred_overlap_ms)))
    preferred_gap = max(0.0, pre - preferred_overlap)
    clamped_gap = min(max_gap, max(min_gap, preferred_gap))
    overlap = min(pre, max(offset, pre - clamped_gap))
    return overlap, {
        "policy": label,
        "span_ms": float(span),
        "gap_ms": float(pre - overlap),
        "preferred_gap_ms": float(preferred_gap),
        "min_gap_ms": float(min_gap),
        "max_gap_ms": float(max_gap),
        "clamped": bool(abs(float(pre - overlap) - preferred_gap) > 1e-6),
        "ratio": float((overlap - offset) / span) if span > 0.0 else 0.0,
    }


def _record_overlap_safety(
    records: list[dict[str, object]],
    *,
    series: str,
    row: dict[str, object],
    offset_ms: float | None,
    overlap_ms: float | None,
    preutterance_ms: float | None,
    fixed_end_ms: float | None,
    accepted: bool,
    policy: str = "",
) -> None:
    if offset_ms is None or overlap_ms is None or preutterance_ms is None:
        records.append(
            {
                "series": series,
                "accepted": bool(accepted),
                "safe": False,
                "order_valid": False,
                "gap_ok": False,
                "ratio_ok": False,
                "missing": True,
                "policy": policy or "missing",
                "format_type": str(row.get("format_type", "") or "<unknown>"),
                "alias_role": str(row.get("alias_role", "") or "<unknown>"),
                "alias_family": manual_oto_alias_family(row.get("alias", "")),
            }
        )
        return
    offset = float(offset_ms)
    overlap = float(overlap_ms)
    pre = float(preutterance_ms)
    fixed = float(fixed_end_ms) if fixed_end_ms is not None else pre
    span = max(0.0, pre - offset)
    gap = max(0.0, pre - overlap)
    _, min_gap, max_gap, default_policy = _safe_overlap_gap_config(row, span)
    ratio = (overlap - offset) / span if span > 0.0 else 0.0
    order_valid = bool(offset <= overlap <= pre <= fixed)
    gap_ok = bool(min_gap - 1e-6 <= gap <= max_gap + 1e-6)
    ratio_ok = bool(0.0 <= ratio <= 1.0)
    records.append(
        {
            "series": series,
            "accepted": bool(accepted),
            "safe": bool(order_valid and gap_ok and ratio_ok),
            "order_valid": order_valid,
            "gap_ok": gap_ok,
            "ratio_ok": ratio_ok,
            "missing": False,
            "policy": policy or default_policy,
            "format_type": str(row.get("format_type", "") or "<unknown>"),
            "alias_role": str(row.get("alias_role", "") or "<unknown>"),
            "alias_family": manual_oto_alias_family(row.get("alias", "")),
            "gap_ms": float(gap),
            "span_ms": float(span),
            "ratio": float(ratio),
            "min_gap_ms": float(min_gap),
            "max_gap_ms": float(max_gap),
        }
    )


def _dependent_overlap_ms(
    row: dict[str, object],
    *,
    offset_ms: float,
    preutterance_ms: float,
    relative_anchor_priors: dict[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    offset = float(offset_ms)
    pre = max(offset, float(preutterance_ms))
    span = max(0.0, pre - offset)
    ratio = _relative_anchor_ratio(
        relative_anchor_priors,
        "overlap_ratio_from_offset_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if ratio is None:
        ratio = _default_overlap_ratio(row)
    ratio = min(1.0, max(0.0, float(ratio)))
    ratio_pred = offset + ratio * span
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "overlap_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is not None:
        gap_pred = pre - float(gap)
        pred = (0.10 * ratio_pred) + (0.90 * gap_pred)
    elif bool(require_stable_prior):
        return None
    else:
        pred = ratio_pred
    return min(pre, max(offset, float(pred)))


def _dependent_fixed_end_ms(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    relative_anchor_priors: dict[str, object] | None,
    require_stable_prior: bool = False,
) -> float | None:
    duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    gap = _relative_anchor_gap_ms(
        relative_anchor_priors,
        "fixed_end_from_preutterance",
        row,
        require_stable=bool(require_stable_prior),
    )
    if gap is None:
        if bool(require_stable_prior):
            return None
        gap = _default_fixed_gap_ms(row)
    expected = min(duration_ms, max(0.0, float(preutterance_ms) + float(gap)))
    window = max(55.0, min(240.0, abs(float(gap)) * 0.70 + 45.0))
    pred = _candidate_near_expected_ms(tracks, "fixed_end", expected, window_ms=window)
    return min(duration_ms, max(float(preutterance_ms) + 8.0, float(pred)))


def _island_row_safe(
    assignment: SlotIslandAssignment | None,
    island: VowelIsland | None,
    predictions: dict[str, float],
) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    if not assignment_is_safe(assignment, island):
        warnings.append("unsafe_island_assignment")
    if island is None:
        warnings.append("missing_island")
    else:
        pre = predictions.get("preutterance")
        if pre is None or not (float(island.start_ms) - 25.0 <= float(pre) <= float(island.end_ms) + 5.0):
            warnings.append("preutterance_outside_island")
    offset = predictions.get("offset")
    overlap = predictions.get("overlap")
    pre = predictions.get("preutterance")
    fixed = predictions.get("fixed_end")
    if None not in (offset, overlap, pre, fixed):
        if not (float(offset) <= float(overlap) <= float(pre) <= float(fixed)):
            warnings.append("anchor_order_violation")
    return not warnings, tuple(warnings)


def _island_prediction_context(
    decode: object,
    assignment: SlotIslandAssignment | None,
    island: VowelIsland | None,
    *,
    slot_index: int,
    slot_count: int,
    safety_warnings: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "island_slot_index": int(slot_index),
        "island_slot_count": int(slot_count),
        "island_index": None if assignment is None else assignment.island_index,
        "island_score": None if assignment is None else float(assignment.score),
        "island_margin": None if assignment is None else float(assignment.margin),
        "island_assignment_warnings": [] if assignment is None else list(assignment.warnings),
        "island_safety_warnings": list(safety_warnings),
        "island": None
        if island is None
        else {
            "start_ms": float(island.start_ms),
            "nucleus_ms": float(island.nucleus_ms),
            "end_ms": float(island.end_ms),
            "confidence": float(island.confidence),
            "left_valley_ms": float(island.left_valley_ms),
            "right_valley_ms": float(island.right_valley_ms),
        },
        "decode_warnings": list(getattr(decode, "warnings", ()) or ()),
    }


def _classify_island_failure(anchor: str, error_ms: float, island_context: dict[str, object]) -> str:
    warnings = set()
    for key in ("decode_warnings", "island_assignment_warnings", "island_safety_warnings"):
        values = island_context.get(key, [])
        if isinstance(values, list):
            warnings.update(str(item) for item in values)
    if "island_count_mismatch" in warnings:
        return "island_split_merge_error"
    if "low_dp_margin" in warnings:
        return "low_dp_margin"
    if "missing_island" in warnings or "missing_island_prediction" in warnings:
        return "slot_mismatch"
    if anchor == "offset" and float(error_ms) > 100.0:
        return "offset_window_miss"
    if "anchor_order_violation" in warnings:
        return "slot_mismatch"
    if float(error_ms) > 300.0:
        return "slot_mismatch"
    return "boundary_pick_error"


def _slot_shift_delta(
    row: dict[str, object],
    *,
    pred_ms: float,
    duration_ms: float,
    slot_count: int | None = None,
    slot_index: int | None = None,
) -> int | None:
    try:
        count = int(slot_count if slot_count is not None else row.get("slot_count", 1) or 1)
        target_idx = int(slot_index if slot_index is not None else row.get("slot_index", 0) or 0)
    except Exception:
        return None
    if count <= 1:
        return 0
    duration = max(1.0, float(duration_ms))
    pred_pos = min(0.999, max(0.0, float(pred_ms) / duration))
    pred_idx = int(round(pred_pos * float(count) - 0.5))
    pred_idx = max(0, min(count - 1, pred_idx))
    target_idx = max(0, min(count - 1, target_idx))
    return int(pred_idx - target_idx)


def _record_slot_shift(
    store: dict[str, dict[str, list[int]]],
    series: str,
    anchor: str,
    row: dict[str, object],
    *,
    pred_ms: float,
    duration_ms: float,
    slot_count: int | None = None,
    slot_index: int | None = None,
) -> None:
    delta = _slot_shift_delta(row, pred_ms=pred_ms, duration_ms=duration_ms, slot_count=slot_count, slot_index=slot_index)
    if delta is not None:
        store[series][anchor].append(int(delta))


def _slot_shift_bucket(delta: int | None) -> str:
    if delta is None:
        return "unknown"
    if int(delta) == 0:
        return "exact"
    if abs(int(delta)) == 1:
        return "one_step"
    return "multi_step"


def _slot_expected_anchor_ms(
    row: dict[str, object],
    anchor: str,
    *,
    duration_ms: float,
    slot_count: int,
    slot_index: int,
) -> float:
    count = max(1, int(slot_count))
    idx = max(0, min(count - 1, int(slot_index)))
    duration = max(1.0, float(duration_ms))
    if count <= 1:
        return duration * 0.5
    span = duration / float(count)
    center = (float(idx) + 0.5) * span
    role = str(row.get("alias_role", "") or "").strip().lower()
    family = manual_oto_alias_family(row.get("alias", ""))
    shifts = {
        "offset": -0.22,
        "overlap": -0.15,
        "preutterance": -0.05,
        "fixed_end": 0.08,
    }
    if role in {"vc", "vcv", "vv"} or family == "vowel_transition":
        shifts = {
            "offset": -0.28,
            "overlap": -0.20,
            "preutterance": -0.08,
            "fixed_end": 0.02,
        }
    if family == "leading_n":
        shifts = {
            "offset": -0.32,
            "overlap": -0.24,
            "preutterance": -0.10,
            "fixed_end": 0.04,
        }
    return min(duration, max(0.0, center + float(shifts.get(anchor, 0.0)) * span))


def _slot_guarded_anchor_ms(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    pred_ms: float,
    *,
    slot_count: int,
    slot_index: int,
) -> float:
    duration = max(1.0, float(tracks.duration_ms or row.get("duration_ms", 1.0) or 1.0))
    delta = _slot_shift_delta(row, pred_ms=pred_ms, duration_ms=duration, slot_count=slot_count, slot_index=slot_index)
    count = max(1, int(slot_count))
    if count <= 1 or delta == 0:
        return float(pred_ms)
    expected = _slot_expected_anchor_ms(row, anchor, duration_ms=duration, slot_count=count, slot_index=slot_index)
    window = max(70.0, min(240.0, 0.42 * duration / float(count)))
    return _candidate_near_expected_ms(tracks, anchor, expected, window_ms=window)


def _summarize_slot_shifts(store: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for series, by_anchor in sorted(store.items()):
        out[series] = {}
        aggregate: list[int] = []
        for anchor, values in sorted(by_anchor.items()):
            arr = np.asarray(values, dtype=np.int32)
            aggregate.extend(int(value) for value in values)
            out[series][anchor] = {
                "total": int(arr.size),
                "exact_slot_rate": float(np.mean(arr == 0)) if arr.size else 0.0,
                "one_step_shift_rate": float(np.mean(np.abs(arr) == 1)) if arr.size else 0.0,
                "multi_step_shift_rate": float(np.mean(np.abs(arr) > 1)) if arr.size else 0.0,
            }
        arr = np.asarray(aggregate, dtype=np.int32)
        out[series]["__all__"] = {
            "total": int(arr.size),
            "exact_slot_rate": float(np.mean(arr == 0)) if arr.size else 0.0,
            "one_step_shift_rate": float(np.mean(np.abs(arr) == 1)) if arr.size else 0.0,
            "multi_step_shift_rate": float(np.mean(np.abs(arr) > 1)) if arr.size else 0.0,
        }
    return out


def _extract_cached(
    cache: dict[str, ManualOtoCandidateTracks],
    wav_path: str,
    *,
    encoder: str,
    min_score: float,
    top_k_fallback: int,
) -> ManualOtoCandidateTracks:
    cached = cache.get(wav_path)
    if cached is not None:
        return cached
    tracks = extract_manual_oto_candidate_tracks(
        wav_path,
        encoder=encoder,
        min_score=float(min_score),
        top_k_fallback=int(top_k_fallback),
    )
    cache[wav_path] = tracks
    return tracks


def _candidate_error(tracks: ManualOtoCandidateTracks, idx: int, target: float) -> float:
    return abs(float(tracks.times_ms[idx]) - float(target))


def _failure_example_base(
    row: dict[str, object],
    *,
    wav_path: str,
    anchor: str,
    target_ms: float,
    pred_ms: float,
    error_ms: float,
    slot_pos_norm: float | None = None,
) -> dict[str, object]:
    return {
        "wav_path": wav_path,
        "voicebank_id": row.get("voicebank_id", ""),
        "wav_name": row.get("wav_name", ""),
        "alias": row.get("alias", ""),
        "alias_role": row.get("alias_role", ""),
        "alias_family": manual_oto_alias_family(row.get("alias", "")),
        "language": row.get("language", ""),
        "format_type": row.get("format_type", ""),
        "slot_index": row.get("slot_index", ""),
        "slot_count": row.get("slot_count", ""),
        "slot_pos_norm": float(slot_pos_norm) if slot_pos_norm is not None else float(_centered_slot_pos(row)),
        "anchor": anchor,
        "target_ms": float(target_ms),
        "pred_ms": float(pred_ms),
        "error_ms": float(error_ms),
        "signed_error_ms": float(pred_ms) - float(target_ms),
    }


def _profile_worst_examples(examples: list[dict[str, object]]) -> dict[str, object]:
    counters: dict[str, Counter[str]] = {
        "by_voicebank": Counter(),
        "by_format": Counter(),
        "by_language": Counter(),
        "by_alias_role": Counter(),
        "by_alias_family": Counter(),
        "by_anchor": Counter(),
        "by_island_failure_reason": Counter(),
    }
    for item in examples:
        counters["by_voicebank"][str(item.get("voicebank_id", "") or "<unknown>")] += 1
        counters["by_format"][str(item.get("format_type", "") or "<unknown>")] += 1
        counters["by_language"][str(item.get("language", "") or "<unknown>")] += 1
        counters["by_alias_role"][str(item.get("alias_role", "") or "<unknown>")] += 1
        counters["by_alias_family"][str(item.get("alias_family", "") or "<unknown>")] += 1
        counters["by_anchor"][str(item.get("anchor", "") or "<unknown>")] += 1
        if item.get("island_failure_reason"):
            counters["by_island_failure_reason"][str(item.get("island_failure_reason"))] += 1
    return {
        key: dict(counter.most_common(12))
        for key, counter in counters.items()
    }


def _record_failure_breakdown(
    records: list[dict[str, object]],
    *,
    series: str,
    anchor: str,
    row: dict[str, object],
    error_ms: float,
    pred_ms: float,
    duration_ms: float,
    dependent_preutterance_source: str,
    slot_count: int | None = None,
    slot_index: int | None = None,
) -> None:
    delta = _slot_shift_delta(
        row,
        pred_ms=float(pred_ms),
        duration_ms=float(duration_ms),
        slot_count=slot_count,
        slot_index=slot_index,
    )
    voicebank_id = str(row.get("voicebank_id", "") or "<unknown>")
    format_type = str(row.get("format_type", "") or "<unknown>")
    records.append(
        {
            "series": str(series),
            "anchor": str(anchor),
            "error_ms": float(error_ms),
            "reject_gt80": bool(float(error_ms) > 80.0),
            "voicebank_id": voicebank_id,
            "format_type": format_type,
            "voicebank_format": f"{voicebank_id}|{format_type}",
            "language": str(row.get("language", "") or "<unknown>"),
            "alias_role": str(row.get("alias_role", "") or "<unknown>"),
            "alias_family": manual_oto_alias_family(row.get("alias", "")),
            "dependent_preutterance_source": str(dependent_preutterance_source or "<unknown>"),
            "slot_shift_bucket": _slot_shift_bucket(delta),
        }
    )


def _summarize_failure_breakdown(records: list[dict[str, object]], *, top_n: int = 80) -> dict[str, object]:
    dimensions = (
        "voicebank_id",
        "format_type",
        "voicebank_format",
        "alias_role",
        "alias_family",
        "dependent_preutterance_source",
        "slot_shift_bucket",
    )
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        series = str(record.get("series", "") or "<unknown>")
        anchor = str(record.get("anchor", "") or "<unknown>")
        error = float(record.get("error_ms", 0.0) or 0.0)
        for dimension in dimensions:
            value = str(record.get(dimension, "") or "<unknown>")
            grouped[(series, anchor, dimension, value)].append(error)

    rows: list[dict[str, object]] = []
    for (series, anchor, dimension, value), values in grouped.items():
        arr = np.asarray(values, dtype=np.float32)
        reject = int(np.sum(arr > 80.0)) if arr.size else 0
        rows.append(
            {
                "series": series,
                "anchor": anchor,
                "dimension": dimension,
                "value": value,
                "total": int(arr.size),
                "reject_gt80": reject,
                "reject_gt80_rate": float(reject / arr.size) if arr.size else 0.0,
                "pass_le10_rate": float(np.mean(arr <= 10.0)) if arr.size else 0.0,
                "recall30": float(np.mean(arr <= 30.0)) if arr.size else 0.0,
                "median_error_ms": float(np.median(arr)) if arr.size else None,
                "p90_error_ms": float(np.percentile(arr, 90.0)) if arr.size else None,
            }
        )

    top_reject = [
        item
        for item in sorted(
            rows,
            key=lambda item: (
                int(item.get("reject_gt80", 0) or 0),
                float(item.get("reject_gt80_rate", 0.0) or 0.0),
                int(item.get("total", 0) or 0),
            ),
            reverse=True,
        )
        if int(item.get("reject_gt80", 0) or 0) > 0
    ][: int(top_n)]
    by_dimension: dict[str, list[dict[str, object]]] = {}
    for dimension in dimensions:
        dimension_rows = [item for item in rows if item.get("dimension") == dimension]
        by_dimension[dimension] = sorted(
            dimension_rows,
            key=lambda item: (
                int(item.get("reject_gt80", 0) or 0),
                float(item.get("reject_gt80_rate", 0.0) or 0.0),
                int(item.get("total", 0) or 0),
            ),
            reverse=True,
        )[: min(int(top_n), 30)]
    return {
        "record_count": int(len(records)),
        "top_reject_gt80": top_reject,
        "by_dimension": by_dimension,
    }


def _summarize_overlap_safety(records: list[dict[str, object]]) -> dict[str, object]:
    def summarize_items(items: list[dict[str, object]]) -> dict[str, object]:
        accepted = [item for item in items if bool(item.get("accepted", False))]
        gaps = [
            float(item.get("gap_ms", 0.0) or 0.0)
            for item in accepted
            if not bool(item.get("missing", False)) and item.get("gap_ms") is not None
        ]
        ratios = [
            float(item.get("ratio", 0.0) or 0.0)
            for item in accepted
            if not bool(item.get("missing", False)) and item.get("ratio") is not None
        ]
        total = len(items)
        accepted_n = len(accepted)
        return {
            "total": int(total),
            "accepted": int(accepted_n),
            "accept_rate": float(accepted_n / total) if total else 0.0,
            "safe": int(sum(1 for item in accepted if bool(item.get("safe", False)))),
            "safe_rate": float(np.mean([bool(item.get("safe", False)) for item in accepted])) if accepted else 0.0,
            "order_valid_rate": float(np.mean([bool(item.get("order_valid", False)) for item in accepted])) if accepted else 0.0,
            "gap_ok_rate": float(np.mean([bool(item.get("gap_ok", False)) for item in accepted])) if accepted else 0.0,
            "ratio_ok_rate": float(np.mean([bool(item.get("ratio_ok", False)) for item in accepted])) if accepted else 0.0,
            "median_gap_ms": float(np.median(np.asarray(gaps, dtype=np.float32))) if gaps else None,
            "p10_gap_ms": float(np.percentile(np.asarray(gaps, dtype=np.float32), 10.0)) if gaps else None,
            "p90_gap_ms": float(np.percentile(np.asarray(gaps, dtype=np.float32), 90.0)) if gaps else None,
            "median_ratio": float(np.median(np.asarray(ratios, dtype=np.float32))) if ratios else None,
        }

    by_series_raw: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_policy_raw: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_role_raw: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        series = str(record.get("series", "") or "<unknown>")
        policy = str(record.get("policy", "") or "<unknown>")
        role = str(record.get("alias_role", "") or "<unknown>")
        by_series_raw[series].append(record)
        by_policy_raw[(series, policy)].append(record)
        by_role_raw[(series, role)].append(record)

    by_policy = [
        {"series": series, "policy": policy, **summarize_items(items)}
        for (series, policy), items in sorted(by_policy_raw.items())
    ]
    by_role = [
        {"series": series, "alias_role": role, **summarize_items(items)}
        for (series, role), items in sorted(by_role_raw.items())
    ]
    return {
        "record_count": int(len(records)),
        "by_series": {
            series: summarize_items(items)
            for series, items in sorted(by_series_raw.items())
        },
        "by_policy": sorted(
            by_policy,
            key=lambda item: (str(item.get("series", "")), -int(item.get("accepted", 0) or 0), str(item.get("policy", ""))),
        ),
        "by_alias_role": sorted(
            by_role,
            key=lambda item: (str(item.get("series", "")), -int(item.get("accepted", 0) or 0), str(item.get("alias_role", ""))),
        ),
    }


def _build_training_arrays(
    rows: list[dict[str, object]],
    *,
    anchors: tuple[str, ...],
    encoder: str,
    min_score: float,
    top_k_fallback: int,
    positive_ms: float,
    negative_min_ms: float,
    max_negatives_per_positive: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    rng = random.Random(int(seed))
    cache: dict[str, ManualOtoCandidateTracks] = {}
    samples: dict[str, list[np.ndarray]] = {anchor: [] for anchor in anchors}
    labels: dict[str, list[int]] = {anchor: [] for anchor in anchors}
    stats: Counter[str] = Counter()
    wav_failures: list[dict[str, str]] = []

    for wav_path, wav_rows in _group_rows_by_wav(rows).items():
        try:
            tracks = _extract_cached(
                cache,
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception as exc:
            stats["wav_failures"] += 1
            if len(wav_failures) < 100:
                wav_failures.append({"wav_path": wav_path, "error": str(exc)})
            continue
        for row in wav_rows:
            feature_row = _row_with_slot_pos(row, _centered_slot_pos(row))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    stats[f"{anchor}:missing_target"] += 1
                    continue
                indices = list(tracks.candidate_indices.get(anchor, ()))
                if not indices:
                    stats[f"{anchor}:no_candidates"] += 1
                    continue
                nearest = min(indices, key=lambda idx: _candidate_error(tracks, idx, target))
                nearest_error = _candidate_error(tracks, nearest, target)
                if nearest_error > float(positive_ms):
                    stats[f"{anchor}:no_positive_within_window"] += 1
                    continue
                samples[anchor].append(manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=nearest))
                labels[anchor].append(1)
                negatives = [idx for idx in indices if _candidate_error(tracks, idx, target) >= float(negative_min_ms)]
                rng.shuffle(negatives)
                for idx in negatives[: max(0, int(max_negatives_per_positive))]:
                    samples[anchor].append(manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx))
                    labels[anchor].append(0)
                stats[f"{anchor}:positives"] += 1
                stats[f"{anchor}:negatives"] += min(len(negatives), max(0, int(max_negatives_per_positive)))

    x_by_anchor: dict[str, np.ndarray] = {}
    y_by_anchor: dict[str, np.ndarray] = {}
    for anchor in anchors:
        if samples[anchor]:
            x_by_anchor[anchor] = np.stack(samples[anchor], axis=0).astype(np.float32)
            y_by_anchor[anchor] = np.asarray(labels[anchor], dtype=np.int64)
        else:
            x_by_anchor[anchor] = np.zeros((0, len(manual_oto_candidate_feature_names())), dtype=np.float32)
            y_by_anchor[anchor] = np.zeros((0,), dtype=np.int64)
    summary = {
        "input_rows": len(rows),
        "wavs_seen": len(_group_rows_by_wav(rows)),
        "wavs_extracted": len(cache),
        "stats": dict(stats),
        "wav_failures": wav_failures,
        "sample_counts": {anchor: int(y_by_anchor[anchor].shape[0]) for anchor in anchors},
        "positive_counts": {anchor: int(np.sum(y_by_anchor[anchor])) for anchor in anchors},
    }
    return x_by_anchor, y_by_anchor, summary


def _build_pairwise_training_arrays(
    rows: list[dict[str, object]],
    *,
    anchors: tuple[str, ...],
    encoder: str,
    min_score: float,
    top_k_fallback: int,
    positive_ms: float,
    negative_min_ms: float,
    max_negatives_per_positive: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    rng = random.Random(int(seed))
    cache: dict[str, ManualOtoCandidateTracks] = {}
    samples: dict[str, list[np.ndarray]] = {anchor: [] for anchor in anchors}
    labels: dict[str, list[int]] = {anchor: [] for anchor in anchors}
    stats: Counter[str] = Counter()
    wav_failures: list[dict[str, str]] = []
    feature_count = len(manual_oto_candidate_feature_names())

    for wav_path, wav_rows in _group_rows_by_wav(rows).items():
        try:
            tracks = _extract_cached(
                cache,
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception as exc:
            stats["wav_failures"] += 1
            if len(wav_failures) < 100:
                wav_failures.append({"wav_path": wav_path, "error": str(exc)})
            continue
        for row in wav_rows:
            feature_row = _row_with_slot_pos(row, _centered_slot_pos(row))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    stats[f"{anchor}:missing_target"] += 1
                    continue
                indices = list(tracks.candidate_indices.get(anchor, ()))
                if not indices:
                    stats[f"{anchor}:no_candidates"] += 1
                    continue
                nearest = min(indices, key=lambda idx: _candidate_error(tracks, idx, target))
                nearest_error = _candidate_error(tracks, nearest, target)
                if nearest_error > float(positive_ms):
                    stats[f"{anchor}:no_positive_within_window"] += 1
                    continue
                positive_features = manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=nearest)
                negatives = [idx for idx in indices if _candidate_error(tracks, idx, target) >= float(negative_min_ms)]
                rng.shuffle(negatives)
                kept_negatives = negatives[: max(0, int(max_negatives_per_positive))]
                for idx in kept_negatives:
                    negative_features = manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                    diff = positive_features - negative_features
                    samples[anchor].append(diff.astype(np.float32))
                    labels[anchor].append(1)
                    samples[anchor].append((-diff).astype(np.float32))
                    labels[anchor].append(0)
                stats[f"{anchor}:positives"] += 1
                stats[f"{anchor}:pairwise_pairs"] += len(kept_negatives)

    x_by_anchor: dict[str, np.ndarray] = {}
    y_by_anchor: dict[str, np.ndarray] = {}
    for anchor in anchors:
        if samples[anchor]:
            x_by_anchor[anchor] = np.stack(samples[anchor], axis=0).astype(np.float32)
            y_by_anchor[anchor] = np.asarray(labels[anchor], dtype=np.int64)
        else:
            x_by_anchor[anchor] = np.zeros((0, feature_count), dtype=np.float32)
            y_by_anchor[anchor] = np.zeros((0,), dtype=np.int64)
    summary = {
        "input_rows": len(rows),
        "wavs_seen": len(_group_rows_by_wav(rows)),
        "wavs_extracted": len(cache),
        "stats": dict(stats),
        "wav_failures": wav_failures,
        "sample_counts": {anchor: int(y_by_anchor[anchor].shape[0]) for anchor in anchors},
        "pair_counts": {anchor: int(y_by_anchor[anchor].shape[0] // 2) for anchor in anchors},
    }
    return x_by_anchor, y_by_anchor, summary


def _build_distance_training_arrays(
    rows: list[dict[str, object]],
    *,
    anchors: tuple[str, ...],
    encoder: str,
    min_score: float,
    top_k_fallback: int,
    distance_sigma_ms: float,
    max_candidates_per_target: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    cache: dict[str, ManualOtoCandidateTracks] = {}
    samples: dict[str, list[np.ndarray]] = {anchor: [] for anchor in anchors}
    labels: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    stats: Counter[str] = Counter()
    wav_failures: list[dict[str, str]] = []
    feature_count = len(manual_oto_candidate_feature_names())
    sigma = max(1.0, float(distance_sigma_ms))

    for wav_path, wav_rows in _group_rows_by_wav(rows).items():
        try:
            tracks = _extract_cached(
                cache,
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception as exc:
            stats["wav_failures"] += 1
            if len(wav_failures) < 100:
                wav_failures.append({"wav_path": wav_path, "error": str(exc)})
            continue
        for row in wav_rows:
            feature_row = _row_with_slot_pos(row, _centered_slot_pos(row))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    stats[f"{anchor}:missing_target"] += 1
                    continue
                indices = list(tracks.candidate_indices.get(anchor, ()))
                if not indices:
                    stats[f"{anchor}:no_candidates"] += 1
                    continue
                if int(max_candidates_per_target) > 0 and len(indices) > int(max_candidates_per_target):
                    anchor_scores = tracks.anchor_scores.get(anchor, np.zeros_like(tracks.times_ms))
                    indices = sorted(
                        indices,
                        key=lambda idx: (
                            _candidate_error(tracks, idx, target),
                            -float(anchor_scores[idx]) if 0 <= idx < anchor_scores.size else 0.0,
                        ),
                    )[: int(max_candidates_per_target)]
                    indices = sorted(indices, key=lambda idx: float(tracks.times_ms[idx]))
                nearest_error = min(_candidate_error(tracks, idx, target) for idx in indices)
                for idx in indices:
                    err = _candidate_error(tracks, idx, target)
                    samples[anchor].append(manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx))
                    labels[anchor].append(float(np.exp(-err / sigma)))
                stats[f"{anchor}:targets"] += 1
                stats[f"{anchor}:candidate_samples"] += len(indices)
                if nearest_error <= 30.0:
                    stats[f"{anchor}:oracle_hit30_in_kept"] += 1

    x_by_anchor: dict[str, np.ndarray] = {}
    y_by_anchor: dict[str, np.ndarray] = {}
    for anchor in anchors:
        if samples[anchor]:
            x_by_anchor[anchor] = np.stack(samples[anchor], axis=0).astype(np.float32)
            y_by_anchor[anchor] = np.asarray(labels[anchor], dtype=np.float32)
        else:
            x_by_anchor[anchor] = np.zeros((0, feature_count), dtype=np.float32)
            y_by_anchor[anchor] = np.zeros((0,), dtype=np.float32)
    summary = {
        "input_rows": len(rows),
        "wavs_seen": len(_group_rows_by_wav(rows)),
        "wavs_extracted": len(cache),
        "distance_sigma_ms": float(distance_sigma_ms),
        "max_candidates_per_target": int(max_candidates_per_target),
        "stats": dict(stats),
        "wav_failures": wav_failures,
        "sample_counts": {anchor: int(y_by_anchor[anchor].shape[0]) for anchor in anchors},
        "label_mean": {anchor: float(np.mean(y_by_anchor[anchor])) if y_by_anchor[anchor].size else 0.0 for anchor in anchors},
    }
    return x_by_anchor, y_by_anchor, summary


def _fit_models(x_by_anchor: dict[str, np.ndarray], y_by_anchor: dict[str, np.ndarray], *, seed: int, max_iter: int) -> dict[str, object]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError("manual OTO anchor scorer training requires scikit-learn.") from exc

    models: dict[str, object] = {}
    for anchor, x in x_by_anchor.items():
        y = y_by_anchor[anchor]
        if x.shape[0] == 0 or len(set(int(value) for value in y.tolist())) < 2:
            continue
        pos = max(1, int(np.sum(y == 1)))
        neg = max(1, int(np.sum(y == 0)))
        weights = np.where(y == 1, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg).astype(np.float32)
        model = HistGradientBoostingClassifier(
            max_iter=int(max_iter),
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=weights)
        models[anchor] = model
    return models


def _fit_pairwise_rankers(
    x_by_anchor: dict[str, np.ndarray],
    y_by_anchor: dict[str, np.ndarray],
    *,
    seed: int,
    max_iter: int,
) -> dict[str, object]:
    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("pairwise manual OTO anchor scorer training requires scikit-learn.") from exc

    models: dict[str, object] = {}
    for anchor, x in x_by_anchor.items():
        y = y_by_anchor[anchor]
        if x.shape[0] == 0 or len(set(int(value) for value in y.tolist())) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=0.0001,
                max_iter=max(1000, int(max_iter) * 20),
                tol=1e-4,
                random_state=int(seed),
                class_weight="balanced",
            ),
        )
        model.fit(x, y)
        models[anchor] = model
    return models


def _fit_distance_rankers(
    x_by_anchor: dict[str, np.ndarray],
    y_by_anchor: dict[str, np.ndarray],
    *,
    seed: int,
    max_iter: int,
) -> dict[str, object]:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("distance manual OTO anchor scorer training requires scikit-learn.") from exc

    models: dict[str, object] = {}
    for anchor, x in x_by_anchor.items():
        y = y_by_anchor[anchor]
        if x.shape[0] == 0 or np.unique(np.round(y, 5)).size < 2:
            continue
        weights = (0.25 + 2.75 * np.asarray(y, dtype=np.float32)).astype(np.float32)
        model = HistGradientBoostingRegressor(
            max_iter=int(max_iter),
            learning_rate=0.07,
            max_leaf_nodes=31,
            l2_regularization=0.03,
            random_state=int(seed),
            loss="squared_error",
        )
        model.fit(x, y, sample_weight=weights)
        models[anchor] = model
    return models


def _fit_failure_gate_models(gate_training: dict[str, object], *, seed: int) -> dict[str, object]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError("supervised failure gate training requires scikit-learn.") from exc

    features_by_anchor = gate_training.get("features_by_anchor") if isinstance(gate_training, dict) else {}
    labels_by_anchor = gate_training.get("labels_by_anchor") if isinstance(gate_training, dict) else {}
    models: dict[str, object] = {}
    if not isinstance(features_by_anchor, dict) or not isinstance(labels_by_anchor, dict):
        return models
    for anchor in SCORABLE_ANCHORS:
        features = features_by_anchor.get(anchor, [])
        labels = labels_by_anchor.get(anchor, [])
        if not features or not labels:
            continue
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or y.size != x.shape[0] or len(set(int(value) for value in y.tolist())) < 2:
            continue
        safe = max(1, int(np.sum(y == 1)))
        hard = max(1, int(np.sum(y == 0)))
        weights = np.where(y == 1, 0.5 * (safe + hard) / safe, 0.5 * (safe + hard) / hard).astype(np.float32)
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=weights)
        models[anchor] = model
    return models


def _fit_local_failure_models(local_training: dict[str, object], *, seed: int) -> dict[str, object]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError as exc:
        raise RuntimeError("local hard-fail gate training requires scikit-learn.") from exc

    features_by_anchor = local_training.get("features_by_anchor") if isinstance(local_training, dict) else {}
    labels_by_anchor = local_training.get("labels_by_anchor") if isinstance(local_training, dict) else {}
    models: dict[str, object] = {}
    if not isinstance(features_by_anchor, dict) or not isinstance(labels_by_anchor, dict):
        return models
    for anchor in SCORABLE_ANCHORS:
        features = features_by_anchor.get(anchor, [])
        labels = labels_by_anchor.get(anchor, [])
        if not features or not labels:
            continue
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or y.size != x.shape[0] or len(set(int(value) for value in y.tolist())) < 2:
            continue
        safe = max(1, int(np.sum(y == 1)))
        hard = max(1, int(np.sum(y == 0)))
        weights = np.where(y == 1, 0.5 * (safe + hard) / safe, 0.5 * (safe + hard) / hard).astype(np.float32)
        model = HistGradientBoostingClassifier(
            max_iter=90,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=weights)
        models[anchor] = model
    return models


def _summarize_gate_threshold_sweep(
    records_by_anchor: dict[str, list[dict[str, object]]],
    *,
    anchors: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> dict[str, list[dict[str, object]]]:
    sweep: dict[str, list[dict[str, object]]] = {}
    normalized_thresholds = tuple(sorted({float(value) for value in thresholds if 0.0 <= float(value) <= 1.0}))
    for anchor in anchors:
        records = records_by_anchor.get(anchor, [])
        total = len(records)
        rows: list[dict[str, object]] = []
        for threshold in normalized_thresholds:
            accepted_errors = [
                float(record["error_ms"])
                for record in records
                if bool(record.get("primary_present"))
                and not bool(record.get("heuristic_rejected"))
                and record.get("safe_prob") is not None
                and float(record.get("safe_prob", 0.0)) >= float(threshold)
                and record.get("error_ms") is not None
            ]
            arr = np.asarray(accepted_errors, dtype=np.float32)
            accepted = int(arr.size)
            rows.append(
                {
                    "threshold": float(threshold),
                    "total": int(total),
                    "accepted": accepted,
                    "reviewed": int(max(0, total - accepted)),
                    "accept_rate": float(accepted / total) if total else 0.0,
                    "review_rate": float(1.0 - (accepted / total)) if total else 0.0,
                    "recall30": float(np.mean(arr <= 30.0)) if accepted else 0.0,
                    "recall60": float(np.mean(arr <= 60.0)) if accepted else 0.0,
                    "pass_le10_rate": float(np.mean(arr <= 10.0)) if accepted else 0.0,
                    "ordinary_30_60_rate": float(np.mean((arr > 30.0) & (arr <= 60.0))) if accepted else 0.0,
                    "review_gt60_rate": float(np.mean(arr > 60.0)) if accepted else 0.0,
                    "warning_60_80_rate": float(np.mean((arr > 60.0) & (arr <= 80.0))) if accepted else 0.0,
                    "reject_gt80_rate": float(np.mean(arr > 80.0)) if accepted else 0.0,
                    "median_error_ms": float(np.median(arr)) if accepted else None,
                    "p90_error_ms": float(np.percentile(arr, 90.0)) if accepted else None,
                    "hard_fail_gt100_rate": float(np.mean(arr > 100.0)) if accepted else 0.0,
                    "max_error_ms": float(np.max(arr)) if accepted else None,
                }
            )
        sweep[anchor] = rows
    return sweep


def _recommend_gate_thresholds(
    sweep: dict[str, list[dict[str, object]]],
    *,
    hard_fail_limit: float = 0.05,
) -> dict[str, dict[str, object] | None]:
    recommendations: dict[str, dict[str, object] | None] = {}
    for anchor, rows in sweep.items():
        eligible = [
            row
            for row in rows
            if int(row.get("accepted", 0) or 0) > 0
            and float(row.get("hard_fail_gt100_rate", 1.0) or 0.0) <= float(hard_fail_limit)
        ]
        if eligible:
            recommendations[anchor] = max(
                eligible,
                key=lambda row: (float(row.get("accept_rate", 0.0) or 0.0), -float(row.get("threshold", 1.0) or 1.0)),
            )
            continue
        nonempty = [row for row in rows if int(row.get("accepted", 0) or 0) > 0]
        if not nonempty:
            recommendations[anchor] = None
            continue
        recommendations[anchor] = min(
            nonempty,
            key=lambda row: (
                float(row.get("hard_fail_gt100_rate", 1.0) or 1.0),
                -float(row.get("accept_rate", 0.0) or 0.0),
            ),
        )
    return recommendations


def _score_candidates(model: object, features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return np.asarray(scores, dtype=np.float32).reshape(-1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        if probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=np.float32)
    scores = model.predict(features)
    return np.asarray(scores, dtype=np.float32)


def _order_norms(tracks: ManualOtoCandidateTracks, anchor: str, indices: list[int]) -> dict[int, float]:
    ordered = sorted(indices, key=lambda idx: float(tracks.times_ms[idx]) if 0 <= idx < tracks.times_ms.size else 0.0)
    return {
        idx: float(order) / max(1.0, float(len(ordered) - 1))
        for order, idx in enumerate(ordered)
    }


def _time_norms(tracks: ManualOtoCandidateTracks, indices: list[int]) -> dict[int, float]:
    duration = max(1.0, float(tracks.duration_ms or 1.0))
    return {
        idx: min(1.0, max(0.0, float(tracks.times_ms[idx]) / duration))
        for idx in indices
        if 0 <= idx < tracks.times_ms.size
    }


def _prediction_record(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    selected: object,
    *,
    anchor: str,
    selected_idx: int,
) -> dict[str, object]:
    return {
        "pred_ms": float(tracks.times_ms[int(selected_idx)]),
        "joint_score": float(getattr(selected, "score", 0.0)),
        "slot_pos_norm": float(getattr(selected, "slot_pos_norm", _centered_slot_pos(row))),
        "time_order_norm": float(getattr(selected, "time_order_norm", 0.0)),
        "anchor": str(anchor),
        "candidate_index": int(selected_idx),
    }


_MOJIBAKE_MARKERS = frozenset("繧縺繝蠑譁荳蜿逕譛髫鬟譖邱")


def _looks_mojibake(value: object) -> bool:
    text = str(value or "")
    if "\ufffd" in text:
        return True
    return sum(1 for ch in text if ch in _MOJIBAKE_MARKERS) >= 2


def _alias_primary_token_count(row: dict[str, object], filename_tokens: list[str]) -> int:
    terms = manual_oto_alias_order_terms(
        row.get("alias", ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias_role=str(row.get("alias_role", "") or ""),
    )
    if not terms or not filename_tokens:
        return 0
    primary = canonicalize_order_token(terms[0])
    if not primary:
        return 0
    count = 0
    for token in filename_tokens:
        current = canonicalize_order_token(token)
        if current == primary or current.startswith(primary) or primary.startswith(current) or primary in current or current in primary:
            count += 1
    return count


def _gate_one_hot(value: object, choices: tuple[str, ...]) -> list[float]:
    text = str(value or "").strip().lower()
    return [1.0 if text == choice else 0.0 for choice in choices]


def _gate_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"anchor={value}" for value in SCORABLE_ANCHORS)
    names.extend(f"role={value}" for value in ROLE_VALUES)
    names.extend(f"format={value}" for value in FORMAT_VALUES)
    names.extend(f"language={value}" for value in LANGUAGE_VALUES)
    names.extend(f"alias_family={value}" for value in ALIAS_FAMILY_VALUES)
    names.extend(
        [
            "slot_index_norm",
            "slot_count_norm",
            "has_filename_tokens",
            "filename_slot_available",
            "slot_gap",
            "filename_pred_available",
            "filename_pred_gap_norm",
            "joint_pred_available",
            "joint_pred_gap_norm",
            "slot_order_gap",
            "joint_score",
            "duplicate_primary_token_count_norm",
            "mojibake_flag",
        ]
    )
    return names


def _gate_features(
    row: dict[str, object],
    anchor: str,
    primary: dict[str, object] | None,
    *,
    filename_pred: dict[str, object] | None,
    joint_pred: dict[str, object] | None,
    filename_slot_pos: float | None,
    filename_tokens: list[str],
) -> np.ndarray:
    try:
        slot_index = float(row.get("slot_index", 0.0) or 0.0)
        slot_count = max(1.0, float(row.get("slot_count", 1.0) or 1.0))
    except Exception:
        slot_index = 0.0
        slot_count = 1.0
    alias_family = manual_oto_alias_family(row.get("alias", ""))
    duration = max(1.0, float(row.get("duration_ms", 1.0) or 1.0))
    if primary is None:
        primary_pred = 0.0
        primary_slot = _centered_slot_pos(row)
        time_order = 0.0
        joint_score = -999.0
    else:
        primary_pred = float(primary.get("pred_ms", 0.0) or 0.0)
        primary_slot = float(primary.get("slot_pos_norm", _centered_slot_pos(row)) or 0.0)
        time_order = float(primary.get("time_order_norm", 0.0) or 0.0)
        joint_score = float(primary.get("joint_score", 0.0) or 0.0)

    if filename_slot_pos is None:
        slot_gap = 1.0
        filename_slot_available = 0.0
    else:
        slot_gap = abs(primary_slot - float(filename_slot_pos))
        filename_slot_available = 1.0

    if filename_pred is None:
        filename_pred_gap = 1.0
        filename_pred_available = 0.0
    else:
        filename_pred_gap = min(1.0, abs(primary_pred - float(filename_pred.get("pred_ms", 0.0) or 0.0)) / duration)
        filename_pred_available = 1.0

    if joint_pred is None:
        joint_pred_gap = 1.0
        joint_pred_available = 0.0
    else:
        joint_pred_gap = min(1.0, abs(primary_pred - float(joint_pred.get("pred_ms", 0.0) or 0.0)) / duration)
        joint_pred_available = 1.0

    values: list[float] = []
    values.extend(_gate_one_hot(anchor, SCORABLE_ANCHORS))
    values.extend(_gate_one_hot(row.get("alias_role", ""), ROLE_VALUES))
    values.extend(_gate_one_hot(row.get("format_type", ""), FORMAT_VALUES))
    values.extend(_gate_one_hot(row.get("language", ""), LANGUAGE_VALUES))
    values.extend(_gate_one_hot(alias_family, ALIAS_FAMILY_VALUES))
    values.extend(
        [
            slot_index / max(1.0, slot_count - 1.0),
            min(slot_count, 64.0) / 64.0,
            1.0 if filename_tokens else 0.0,
            filename_slot_available,
            float(slot_gap),
            filename_pred_available,
            float(filename_pred_gap),
            joint_pred_available,
            float(joint_pred_gap),
            abs(time_order - primary_slot),
            max(-10.0, min(10.0, joint_score)) / 10.0,
            min(4.0, float(_alias_primary_token_count(row, filename_tokens))) / 4.0,
            1.0
            if (
                _looks_mojibake(row.get("alias", ""))
                or _looks_mojibake(row.get("wav_name", ""))
                or _looks_mojibake(row.get("voicebank_id", ""))
            )
            else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _nearest_track_value(tracks: ManualOtoCandidateTracks, values: object, pred_ms: float) -> float:
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if times.size == 0 or arr.size == 0:
        return 0.0
    if arr.size != times.size:
        arr = np.resize(arr, (times.size,)).astype(np.float32)
    idx = int(np.searchsorted(times, float(pred_ms), side="left"))
    if idx <= 0:
        best_idx = 0
    elif idx >= times.size:
        best_idx = int(times.size - 1)
    else:
        left = idx - 1
        best_idx = left if abs(float(times[left]) - float(pred_ms)) <= abs(float(times[idx]) - float(pred_ms)) else idx
    return float(arr[best_idx])


def _local_failure_feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"anchor={value}" for value in SCORABLE_ANCHORS)
    names.extend(f"role={value}" for value in ROLE_VALUES)
    names.extend(f"format={value}" for value in FORMAT_VALUES)
    names.extend(f"language={value}" for value in LANGUAGE_VALUES)
    names.extend(f"alias_family={value}" for value in ALIAS_FAMILY_VALUES)
    names.extend(
        [
            "slot_index_norm",
            "slot_count_norm",
            "duration_norm",
            "pred_norm",
            "expected_slot_pos",
            "pred_slot_gap",
            "slot_delta_abs",
            "slot_delta_sign",
            "pre_safe_prob_available",
            "pre_safe_prob",
            "offset_available",
            "overlap_available",
            "preutterance_available",
            "fixed_end_available",
            "offset_pre_gap_norm",
            "overlap_pre_gap_norm",
            "fixed_pre_gap_norm",
            "overlap_ratio",
            "pred_from_offset_norm",
            "pred_from_pre_norm",
            "safe_overlap_gap_norm",
            "safe_overlap_ratio",
            "safe_overlap_clamped",
            "all_slot_exact",
            "boundary_slot_exact",
            "anchor_score_at_pred",
            "transition_at_pred",
            "nucleus_at_pred",
            "rms_at_pred",
            "activity_edge_at_pred",
            "silence_at_pred",
            "voicing_at_pred",
            "flux_at_pred",
            "onset_at_pred",
            "world_stability_at_pred",
        ]
    )
    return names


def _local_failure_features(
    row: dict[str, object],
    tracks: ManualOtoCandidateTracks,
    anchor: str,
    *,
    pred_ms: float,
    predictions: dict[str, float],
    safe_overlap_context: dict[str, object],
    pre_safe_prob: float | None,
    slot_count: int,
    slot_index: int,
    all_slot_exact: bool,
    boundary_slot_exact: bool,
) -> np.ndarray:
    try:
        count = max(1, int(slot_count))
        index = max(0, min(count - 1, int(slot_index)))
    except Exception:
        count = 1
        index = 0
    duration = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
    pred = min(duration, max(0.0, float(pred_ms)))
    expected_slot_pos = 0.5 if count <= 1 else (float(index) + 0.5) / float(count)
    pred_pos = pred / duration
    delta = _slot_shift_delta(
        row,
        pred_ms=pred,
        duration_ms=duration,
        slot_count=count,
        slot_index=index,
    )
    offset = predictions.get("offset")
    overlap = predictions.get("overlap")
    pre = predictions.get("preutterance")
    fixed = predictions.get("fixed_end")
    offset_f = float(offset) if offset is not None else pred
    pre_f = float(pre) if pre is not None else pred
    span = max(1.0, pre_f - offset_f)
    overlap_f = float(overlap) if overlap is not None else offset_f
    fixed_f = float(fixed) if fixed is not None else pre_f
    alias_family = manual_oto_alias_family(row.get("alias", ""))

    anchor_scores = tracks.anchor_scores.get(anchor, np.zeros((np.asarray(tracks.times_ms).size,), dtype=np.float32))
    values: list[float] = []
    values.extend(_gate_one_hot(anchor, SCORABLE_ANCHORS))
    values.extend(_gate_one_hot(row.get("alias_role", ""), ROLE_VALUES))
    values.extend(_gate_one_hot(row.get("format_type", ""), FORMAT_VALUES))
    values.extend(_gate_one_hot(row.get("language", ""), LANGUAGE_VALUES))
    values.extend(_gate_one_hot(alias_family, ALIAS_FAMILY_VALUES))
    values.extend(
        [
            index / max(1.0, count - 1.0),
            min(count, 64.0) / 64.0,
            min(duration, 10000.0) / 10000.0,
            pred_pos,
            expected_slot_pos,
            abs(pred_pos - expected_slot_pos),
            min(4.0, abs(float(delta if delta is not None else 0))) / 4.0,
            max(-1.0, min(1.0, float(delta if delta is not None else 0))),
            1.0 if pre_safe_prob is not None else 0.0,
            float(pre_safe_prob) if pre_safe_prob is not None and np.isfinite(float(pre_safe_prob)) else 0.0,
            1.0 if offset is not None else 0.0,
            1.0 if overlap is not None else 0.0,
            1.0 if pre is not None else 0.0,
            1.0 if fixed is not None else 0.0,
            max(-1.0, min(1.0, (pre_f - offset_f) / duration)),
            max(-1.0, min(1.0, (pre_f - overlap_f) / duration)),
            max(-1.0, min(1.0, (fixed_f - pre_f) / duration)),
            max(-0.5, min(1.5, (overlap_f - offset_f) / span)),
            max(-1.0, min(1.0, (pred - offset_f) / duration)),
            max(-1.0, min(1.0, (pred - pre_f) / duration)),
            max(0.0, min(1.0, float(safe_overlap_context.get("gap_ms", 0.0) or 0.0) / duration)),
            max(0.0, min(1.0, float(safe_overlap_context.get("ratio", 0.0) or 0.0))),
            1.0 if bool(safe_overlap_context.get("clamped", False)) else 0.0,
            1.0 if bool(all_slot_exact) else 0.0,
            1.0 if bool(boundary_slot_exact) else 0.0,
            _nearest_track_value(tracks, anchor_scores, pred),
            _nearest_track_value(tracks, tracks.tracks.get("transition", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("nucleus", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("rms", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("activity_edge", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("silence", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("voicing", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("flux", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("onset", ()), pred),
            _nearest_track_value(tracks, tracks.tracks.get("world_stability", ()), pred),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _local_failure_safe_probability(model: object | None, features: np.ndarray | None) -> float | None:
    if model is None or features is None:
        return None
    try:
        return float(model.predict_proba(features.reshape(1, -1))[0, 1])
    except Exception:
        try:
            return float(model.predict(features.reshape(1, -1))[0])
        except Exception:
            return None


def _gate_constrained_prediction(
    row: dict[str, object],
    anchor: str,
    primary: dict[str, object] | None,
    *,
    filename_pred: dict[str, object] | None,
    joint_pred: dict[str, object] | None,
    filename_slot_pos: float | None,
    filename_tokens: list[str],
    slot_disagreement_threshold: float,
    pred_disagreement_ms: float,
    joint_disagreement_ms: float,
    order_disagreement_threshold: float,
    joint_score_min: float,
    review_mojibake: bool,
    review_duplicate_tokens: bool,
) -> list[str]:
    if primary is None:
        return ["missing_constrained_prediction"]

    reasons: list[str] = []
    fmt = str(row.get("format_type", "") or "").strip().lower()
    role = str(row.get("alias_role", "") or "").strip().lower()
    try:
        slot_count = int(row.get("slot_count", 1) or 1)
    except Exception:
        slot_count = 1

    complex_format = fmt in {"cvvc", "vcv", "cvc", "vccv", "cv-vc"}
    if complex_format and slot_count > 1 and not filename_tokens:
        reasons.append("missing_filename_tokens")
    if bool(review_mojibake) and (
        _looks_mojibake(row.get("alias", ""))
        or _looks_mojibake(row.get("wav_name", ""))
        or _looks_mojibake(row.get("voicebank_id", ""))
    ):
        reasons.append("mojibake_text")
    if bool(review_duplicate_tokens) and complex_format and _alias_primary_token_count(row, filename_tokens) >= 2:
        reasons.append("duplicate_filename_slot")

    primary_slot = float(primary.get("slot_pos_norm", _centered_slot_pos(row)) or 0.0)
    if float(primary.get("joint_score", 0.0) or 0.0) < float(joint_score_min):
        reasons.append("low_joint_score")
    if filename_slot_pos is not None:
        slot_gap = abs(primary_slot - float(filename_slot_pos))
        if slot_gap > float(slot_disagreement_threshold):
            reasons.append("slot_disagreement")

    primary_pred = float(primary.get("pred_ms", 0.0) or 0.0)
    if filename_pred is not None:
        filename_gap = abs(primary_pred - float(filename_pred.get("pred_ms", 0.0) or 0.0))
        if filename_gap > float(pred_disagreement_ms):
            reasons.append("filename_decoder_disagreement")
    elif complex_format and role in {"vc", "vcv", "vv"}:
        reasons.append("missing_filename_decoder")

    if joint_pred is not None:
        joint_gap = abs(primary_pred - float(joint_pred.get("pred_ms", 0.0) or 0.0))
        if joint_gap > float(joint_disagreement_ms):
            reasons.append("joint_decoder_disagreement")

    order_gap = abs(float(primary.get("time_order_norm", 0.0) or 0.0) - primary_slot)
    if order_gap > float(order_disagreement_threshold):
        reasons.append("slot_order_mismatch")

    return reasons


def _family_anchor_prior_score(
    row: dict[str, object],
    anchor: str,
    *,
    candidate_time_norm: float,
    slot_pos_norm: float,
    weight: float,
) -> float:
    if float(weight) <= 0.0:
        return 0.0
    alias_family = manual_oto_alias_family(row.get("alias", ""))
    role = str(row.get("alias_role", "") or "").strip().lower()
    try:
        slot_count = max(1, int(row.get("slot_count", 1) or 1))
    except Exception:
        slot_count = 1
    slot_span = 1.0 / float(slot_count)
    if slot_count <= 1:
        slot_weight = 0.45
    elif slot_count <= 3:
        slot_weight = 0.75
    else:
        slot_weight = 1.0

    shift_by_anchor = {
        "offset": -0.20,
        "overlap": -0.14,
        "preutterance": -0.04,
        "fixed_end": 0.06,
    }
    window_by_anchor = {
        "offset": 0.16,
        "overlap": 0.15,
        "preutterance": 0.15,
        "fixed_end": 0.18,
    }
    family_weight = 1.0
    if role in {"vc", "vcv", "vv"} or alias_family == "vowel_transition":
        shift_by_anchor = {
            "offset": -0.27,
            "overlap": -0.21,
            "preutterance": -0.09,
            "fixed_end": 0.00,
        }
        window_by_anchor = {
            "offset": 0.18,
            "overlap": 0.17,
            "preutterance": 0.17,
            "fixed_end": 0.20,
        }
        family_weight = 1.15
    if alias_family == "leading_n":
        shift_by_anchor = {
            "offset": -0.32,
            "overlap": -0.24,
            "preutterance": -0.12,
            "fixed_end": 0.02,
        }
        window_by_anchor = {
            "offset": 0.20,
            "overlap": 0.19,
            "preutterance": 0.18,
            "fixed_end": 0.22,
        }
        family_weight = 1.25
    elif alias_family in {"weak_suffix", "power_suffix"}:
        window_by_anchor = {
            "offset": 0.20,
            "overlap": 0.19,
            "preutterance": 0.19,
            "fixed_end": 0.23,
        }
        family_weight = 0.80

    shift = float(shift_by_anchor.get(anchor, 0.0)) * slot_span
    window = max(0.04, float(window_by_anchor.get(anchor, 0.18)) * slot_span)
    expected = min(0.98, max(0.02, float(slot_pos_norm) + shift))
    distance = abs(float(candidate_time_norm) - expected)
    overflow = max(0.0, distance - window)
    if overflow <= 0.0:
        return 0.0
    scaled = overflow / max(0.04, slot_span)
    return -min(3.0, float(weight) * slot_weight * family_weight * scaled)


def evaluate_models(
    rows: list[dict[str, object]],
    *,
    models: dict[str, object],
    anchors: tuple[str, ...],
    encoder: str,
    min_score: float,
    top_k_fallback: int,
    order_window: float = 0.20,
    joint_top_per_anchor: int = 6,
    joint_max_options_per_row: int = 80,
    family_prior_weight: float = 1.0,
    joint_family_prior_weight: float = 0.0,
    require_primary_transition_token: bool = False,
    gate_slot_disagreement: float = 0.24,
    gate_pred_disagreement_ms: float = 320.0,
    gate_joint_disagreement_ms: float = 420.0,
    gate_order_disagreement: float = 0.35,
    gate_joint_score_min: float = 0.0,
    gate_review_mojibake: bool = True,
    gate_review_duplicate_tokens: bool = True,
    use_heuristic_gate: bool = True,
    supervised_gate_models: dict[str, object] | None = None,
    supervised_gate_threshold: float = 0.80,
    supervised_gate_thresholds_by_anchor: dict[str, float] | None = None,
    local_failure_models: dict[str, object] | None = None,
    local_gate_threshold: float = 0.80,
    gate_threshold_sweep: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    relative_anchor_priors: dict[str, object] | None = None,
    island_anchor_priors: dict[str, object] | None = None,
    collect_gate_training: bool = False,
    collect_local_failure_training: bool = False,
) -> dict[str, object]:
    cache: dict[str, ManualOtoCandidateTracks] = {}
    errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    family_prior_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    raw_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    order_prior_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    model_order_window_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    monotonic_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    joint_lattice_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    filename_slot_lattice_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    constrained_slot_lattice_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    gated_constrained_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    offset_gap_model_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    overlap_relative_gap_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    fixed_end_relative_gap_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_dependent_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_dependent_total_by_anchor: Counter[str] = Counter()
    routed_dependent_accept_by_anchor: Counter[str] = Counter()
    routed_dependent_review_by_anchor: Counter[str] = Counter()
    routed_dependent_review_reason_by_anchor: Counter[str] = Counter()
    routed_calibrated_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_calibrated_total_by_anchor: Counter[str] = Counter()
    routed_calibrated_accept_by_anchor: Counter[str] = Counter()
    routed_calibrated_review_by_anchor: Counter[str] = Counter()
    routed_calibrated_review_reason_by_anchor: Counter[str] = Counter()
    routed_overlap_span_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_overlap_span_total_by_anchor: Counter[str] = Counter()
    routed_overlap_span_accept_by_anchor: Counter[str] = Counter()
    routed_overlap_span_review_by_anchor: Counter[str] = Counter()
    routed_overlap_span_review_reason_by_anchor: Counter[str] = Counter()
    routed_safe_overlap_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_safe_overlap_total_by_anchor: Counter[str] = Counter()
    routed_safe_overlap_accept_by_anchor: Counter[str] = Counter()
    routed_safe_overlap_review_by_anchor: Counter[str] = Counter()
    routed_safe_overlap_review_reason_by_anchor: Counter[str] = Counter()
    routed_slot_exact_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_slot_exact_total_by_anchor: Counter[str] = Counter()
    routed_slot_exact_accept_by_anchor: Counter[str] = Counter()
    routed_slot_exact_review_by_anchor: Counter[str] = Counter()
    routed_slot_exact_review_reason_by_anchor: Counter[str] = Counter()
    routed_boundary_slot_exact_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_boundary_slot_exact_total_by_anchor: Counter[str] = Counter()
    routed_boundary_slot_exact_accept_by_anchor: Counter[str] = Counter()
    routed_boundary_slot_exact_review_by_anchor: Counter[str] = Counter()
    routed_boundary_slot_exact_review_reason_by_anchor: Counter[str] = Counter()
    routed_local_gate_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_local_gate_total_by_anchor: Counter[str] = Counter()
    routed_local_gate_accept_by_anchor: Counter[str] = Counter()
    routed_local_gate_review_by_anchor: Counter[str] = Counter()
    routed_local_gate_review_reason_by_anchor: Counter[str] = Counter()
    routed_parameter_policy_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    routed_parameter_policy_total_by_anchor: Counter[str] = Counter()
    routed_parameter_policy_accept_by_anchor: Counter[str] = Counter()
    routed_parameter_policy_review_by_anchor: Counter[str] = Counter()
    routed_parameter_policy_review_reason_by_anchor: Counter[str] = Counter()
    vowel_island_lattice_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    local_offset_from_island_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    overlap_relative_on_island_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    vowel_island_gated_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    vowel_island_total_by_anchor: Counter[str] = Counter()
    vowel_island_accept_by_anchor: Counter[str] = Counter()
    vowel_island_review_by_anchor: Counter[str] = Counter()
    vowel_island_review_reason_by_anchor: Counter[str] = Counter()
    slot_shift_by_series_anchor: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    oracle_errors_by_anchor: dict[str, list[float]] = {anchor: [] for anchor in anchors}
    total_by_anchor: Counter[str] = Counter()
    missing_by_anchor: Counter[str] = Counter()
    gate_total_by_anchor: Counter[str] = Counter()
    gate_accept_by_anchor: Counter[str] = Counter()
    gate_review_by_anchor: Counter[str] = Counter()
    gate_reason_by_anchor: Counter[str] = Counter()
    gate_feature_samples: dict[str, list[np.ndarray]] = {anchor: [] for anchor in anchors}
    gate_feature_labels: dict[str, list[int]] = {anchor: [] for anchor in anchors}
    local_failure_feature_samples: dict[str, list[np.ndarray]] = {anchor: [] for anchor in anchors}
    local_failure_feature_labels: dict[str, list[int]] = {anchor: [] for anchor in anchors}
    gate_sweep_records_by_anchor: dict[str, list[dict[str, object]]] = {anchor: [] for anchor in anchors}
    by_role_anchor_total: Counter[str] = Counter()
    by_role_anchor_hit30: Counter[str] = Counter()
    by_alias_family_anchor_total: Counter[str] = Counter()
    by_alias_family_anchor_hit30: Counter[str] = Counter()
    worst: list[dict[str, object]] = []
    worst_joint: list[dict[str, object]] = []
    worst_filename_slot: list[dict[str, object]] = []
    worst_constrained: list[dict[str, object]] = []
    worst_gated_constrained: list[dict[str, object]] = []
    worst_routed_dependent: list[dict[str, object]] = []
    worst_routed_calibrated: list[dict[str, object]] = []
    worst_routed_overlap_span: list[dict[str, object]] = []
    worst_routed_safe_overlap: list[dict[str, object]] = []
    worst_routed_slot_exact: list[dict[str, object]] = []
    worst_routed_boundary_slot_exact: list[dict[str, object]] = []
    worst_routed_local_gate: list[dict[str, object]] = []
    worst_routed_parameter_policy: list[dict[str, object]] = []
    worst_vowel_island: list[dict[str, object]] = []
    worst_vowel_island_gated: list[dict[str, object]] = []
    failure_breakdown_records: list[dict[str, object]] = []
    overlap_safety_records: list[dict[str, object]] = []
    wav_failures: list[dict[str, str]] = []

    for wav_path, wav_rows in _group_rows_by_wav(rows).items():
        try:
            tracks = _extract_cached(
                cache,
                wav_path,
                encoder=encoder,
                min_score=float(min_score),
                top_k_fallback=int(top_k_fallback),
            )
        except Exception as exc:
            if len(wav_failures) < 100:
                wav_failures.append({"wav_path": wav_path, "error": str(exc)})
            continue
        ordered_wav_rows = _filename_ordered_rows(
            wav_rows,
            require_primary_transition_token=bool(require_primary_transition_token),
        )
        joint_predictions: dict[tuple[int, str], dict[str, object]] = {}
        filename_slot_predictions: dict[tuple[int, str], dict[str, object]] = {}
        constrained_predictions: dict[tuple[int, str], dict[str, object]] = {}
        joint_rows: list[dict[str, object] | None] = []
        joint_options_by_row = []
        if all(anchor in models for anchor in anchors):
            for row, filename_slot_pos in ordered_wav_rows:
                feature_row = _row_with_slot_pos(row, filename_slot_pos)
                options_by_anchor: dict[str, list[CandidateOption]] = {}
                row_has_target = False
                for anchor in anchors:
                    if _target_ms(row, anchor) is not None:
                        row_has_target = True
                    model = models.get(anchor)
                    indices = list(tracks.candidate_indices.get(anchor, ()))
                    if model is None or not indices:
                        continue
                    features = np.stack(
                        [
                            manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                            for idx in indices
                        ],
                        axis=0,
                    ).astype(np.float32)
                    scores = _score_candidates(model, features)
                    order_lookup = _time_norms(tracks, indices)
                    slot_pos = float(filename_slot_pos)
                    adjusted_scores = [
                        float(scores[pos])
                        + _family_anchor_prior_score(
                            row,
                            anchor,
                            candidate_time_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=slot_pos,
                            weight=float(joint_family_prior_weight),
                        )
                        for pos, idx in enumerate(indices)
                    ]
                    options_by_anchor[anchor] = [
                        CandidateOption(
                            candidate_index=int(idx),
                            time_ms=float(tracks.times_ms[idx]),
                            score=float(adjusted_scores[pos]),
                            order_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=slot_pos,
                        )
                        for pos, idx in enumerate(indices)
                    ]
                joint_options = build_joint_anchor_options(
                    options_by_anchor,
                    anchors=anchors,
                    top_per_anchor=int(joint_top_per_anchor),
                    max_options=int(joint_max_options_per_row),
                )
                joint_options_by_row.append(joint_options)
                joint_rows.append(row if row_has_target else None)
            decoded_joint = decode_joint_anchor_lattice(joint_options_by_row)
            for row, selected in zip(joint_rows, decoded_joint, strict=False):
                if row is None or selected is None:
                    continue
                for anchor in anchors:
                    target = _target_ms(row, anchor)
                    selected_idx = selected.anchor_indices.get(anchor)
                    if target is None or selected_idx is None:
                        continue
                    joint_predictions[(id(row), anchor)] = _prediction_record(
                        row,
                        tracks,
                        selected,
                        anchor=anchor,
                        selected_idx=int(selected_idx),
                    )
                    err = float(_candidate_error(tracks, int(selected_idx), target))
                    joint_lattice_errors_by_anchor[anchor].append(err)
                    if err > 60.0 and len(worst_joint) < 200:
                        pred_ms = float(tracks.times_ms[int(selected_idx)])
                        worst_joint.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=pred_ms,
                                    error_ms=err,
                                    slot_pos_norm=float(selected.slot_pos_norm),
                                ),
                                "joint_score": float(selected.score),
                                "time_order_norm": float(selected.time_order_norm),
                            }
                        )

        filename_slot_rows: list[dict[str, object] | None] = []
        filename_slot_options_by_row = []
        if all(anchor in models for anchor in anchors):
            for row, filename_slot_pos in ordered_wav_rows:
                feature_row = _row_with_slot_pos(row, filename_slot_pos)
                options_by_anchor: dict[str, list[CandidateOption]] = {}
                row_has_target = False
                for anchor in anchors:
                    if _target_ms(row, anchor) is not None:
                        row_has_target = True
                    model = models.get(anchor)
                    indices = list(tracks.candidate_indices.get(anchor, ()))
                    if model is None or not indices:
                        continue
                    features = np.stack(
                        [
                            manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                            for idx in indices
                        ],
                        axis=0,
                    ).astype(np.float32)
                    scores = _score_candidates(model, features)
                    order_lookup = _time_norms(tracks, indices)
                    slot_pos = float(filename_slot_pos)
                    options_by_anchor[anchor] = [
                        CandidateOption(
                            candidate_index=int(idx),
                            time_ms=float(tracks.times_ms[idx]),
                            score=float(scores[pos]),
                            order_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=slot_pos,
                        )
                        for pos, idx in enumerate(indices)
                    ]
                filename_slot_options = build_joint_anchor_options(
                    options_by_anchor,
                    anchors=anchors,
                    top_per_anchor=int(joint_top_per_anchor),
                    max_options=int(joint_max_options_per_row),
                    slot_order_penalty=1.8,
                )
                filename_slot_options_by_row.append(filename_slot_options)
                filename_slot_rows.append(row if row_has_target else None)
            decoded_filename_slot = decode_joint_anchor_lattice(
                filename_slot_options_by_row,
                slot_order_penalty=1.4,
                slot_transition_penalty=2.4,
                slot_backward_penalty=4.5,
            )
            for row, selected in zip(filename_slot_rows, decoded_filename_slot, strict=False):
                if row is None or selected is None:
                    continue
                for anchor in anchors:
                    target = _target_ms(row, anchor)
                    selected_idx = selected.anchor_indices.get(anchor)
                    if target is None or selected_idx is None:
                        continue
                    filename_slot_predictions[(id(row), anchor)] = _prediction_record(
                        row,
                        tracks,
                        selected,
                        anchor=anchor,
                        selected_idx=int(selected_idx),
                    )
                    err = float(_candidate_error(tracks, int(selected_idx), target))
                    filename_slot_lattice_errors_by_anchor[anchor].append(err)
                    if err > 60.0 and len(worst_filename_slot) < 200:
                        pred_ms = float(tracks.times_ms[int(selected_idx)])
                        worst_filename_slot.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=pred_ms,
                                    error_ms=err,
                                    slot_pos_norm=float(selected.slot_pos_norm),
                                ),
                                "joint_score": float(selected.score),
                                "time_order_norm": float(selected.time_order_norm),
                            }
                        )

        constrained_rows: list[dict[str, object] | None] = []
        constrained_options_by_row = []
        if all(anchor in models for anchor in anchors):
            for row, row_slot_pos in _row_slot_ordered_rows(wav_rows):
                feature_row = _row_with_slot_pos(row, row_slot_pos)
                options_by_anchor: dict[str, list[CandidateOption]] = {}
                row_has_target = False
                for anchor in anchors:
                    if _target_ms(row, anchor) is not None:
                        row_has_target = True
                    model = models.get(anchor)
                    indices = list(tracks.candidate_indices.get(anchor, ()))
                    if model is None or not indices:
                        continue
                    features = np.stack(
                        [
                            manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                            for idx in indices
                        ],
                        axis=0,
                    ).astype(np.float32)
                    scores = _score_candidates(model, features)
                    order_lookup = _time_norms(tracks, indices)
                    slot_pos = float(row_slot_pos)
                    options_by_anchor[anchor] = [
                        CandidateOption(
                            candidate_index=int(idx),
                            time_ms=float(tracks.times_ms[idx]),
                            score=float(scores[pos]),
                            order_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=slot_pos,
                        )
                        for pos, idx in enumerate(indices)
                    ]
                constrained_options = build_joint_anchor_options(
                    options_by_anchor,
                    anchors=anchors,
                    top_per_anchor=int(joint_top_per_anchor),
                    max_options=int(joint_max_options_per_row),
                    slot_order_penalty=1.8,
                )
                constrained_options_by_row.append(constrained_options)
                constrained_rows.append(row if row_has_target else None)
            decoded_constrained = decode_joint_anchor_lattice(
                constrained_options_by_row,
                slot_order_penalty=1.4,
                slot_transition_penalty=2.4,
                slot_backward_penalty=4.5,
            )
            for row, selected in zip(constrained_rows, decoded_constrained, strict=False):
                if row is None or selected is None:
                    continue
                for anchor in anchors:
                    target = _target_ms(row, anchor)
                    selected_idx = selected.anchor_indices.get(anchor)
                    if target is None or selected_idx is None:
                        continue
                    constrained_predictions[(id(row), anchor)] = _prediction_record(
                        row,
                        tracks,
                        selected,
                        anchor=anchor,
                        selected_idx=int(selected_idx),
                    )
                    err = float(_candidate_error(tracks, int(selected_idx), target))
                    constrained_slot_lattice_errors_by_anchor[anchor].append(err)
                    pred_ms = float(tracks.times_ms[int(selected_idx)])
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_constrained_slot_lattice",
                        anchor,
                        row,
                        pred_ms=pred_ms,
                        duration_ms=float(tracks.duration_ms),
                    )
                    if err > 60.0 and len(worst_constrained) < 200:
                        worst_constrained.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=pred_ms,
                                    error_ms=err,
                                    slot_pos_norm=float(selected.slot_pos_norm),
                                ),
                                "joint_score": float(selected.score),
                                "time_order_norm": float(selected.time_order_norm),
                            }
                        )

        slot_count_for_islands = _vowel_island_slot_count(wav_rows)
        raw_islands = extract_vowel_islands(tracks)
        islands = fit_islands_to_slot_count(
            tracks,
            raw_islands,
            slot_count=slot_count_for_islands,
        )
        island_decode = assign_slots_to_islands(
            islands,
            slot_count=slot_count_for_islands,
            duration_ms=float(tracks.duration_ms),
        )
        if len(raw_islands) != int(slot_count_for_islands):
            raw_count_warning = "island_count_mismatch"
            island_decode = VowelIslandDecode(
                islands=island_decode.islands,
                assignments=tuple(
                    SlotIslandAssignment(
                        slot_index=assignment.slot_index,
                        island_index=assignment.island_index,
                        score=assignment.score,
                        margin=assignment.margin,
                        warnings=tuple(dict.fromkeys((*assignment.warnings, raw_count_warning))),
                    )
                    for assignment in island_decode.assignments
                ),
                score=island_decode.score,
                margin=island_decode.margin,
                warnings=tuple(dict.fromkeys((*island_decode.warnings, raw_count_warning))),
            )
        island_events = island_overlay_events(island_decode)
        for row in wav_rows:
            duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
            slot_idx = _row_vowel_island_slot_index(row, slot_count_for_islands)
            assignment = _island_assignment_for_row(row, island_decode.assignments, slot_count_for_islands)
            island = _island_for_assignment(assignment, island_decode.islands)
            predictions: dict[str, float] = {}
            if island is not None:
                for anchor in anchors:
                    primary_prediction = constrained_predictions.get((id(row), anchor))
                    pred = None
                    apply_slot_guard = True
                    if primary_prediction is not None:
                        pred = float(primary_prediction.get("pred_ms", 0.0) or 0.0)
                        primary_delta = _slot_shift_delta(
                            row,
                            pred_ms=float(pred),
                            duration_ms=float(tracks.duration_ms),
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        if primary_delta == 0:
                            apply_slot_guard = False
                        else:
                            gate_model = (supervised_gate_models or {}).get(anchor)
                            if gate_model is not None:
                                anchor_gate_threshold = _threshold_for_anchor(
                                    anchor,
                                    float(supervised_gate_threshold),
                                    supervised_gate_thresholds_by_anchor,
                                )
                                filename_tokens = _filename_tokens_for_row(row)
                                filename_slot_pos = _infer_filename_slot_pos(
                                    row,
                                    filename_tokens,
                                    require_primary_transition_token=bool(require_primary_transition_token),
                                )
                                gate_features = _gate_features(
                                    row,
                                    anchor,
                                    primary_prediction,
                                    filename_pred=filename_slot_predictions.get((id(row), anchor)),
                                    joint_pred=joint_predictions.get((id(row), anchor)),
                                    filename_slot_pos=filename_slot_pos,
                                    filename_tokens=filename_tokens,
                                )
                                try:
                                    safe_prob = float(gate_model.predict_proba(gate_features.reshape(1, -1))[0, 1])
                                except Exception:
                                    safe_prob = float(gate_model.predict(gate_features.reshape(1, -1))[0])
                                if safe_prob >= float(anchor_gate_threshold):
                                    apply_slot_guard = False
                    if pred is None:
                        pred = _predict_island_anchor_ms(
                            row,
                            tracks,
                            anchor=anchor,
                            island=island,
                            relative_anchor_priors=relative_anchor_priors,
                            island_anchor_priors=island_anchor_priors,
                        )
                    if pred is not None:
                        if apply_slot_guard:
                            pred = _slot_guarded_anchor_ms(
                                row,
                                tracks,
                                anchor,
                                float(pred),
                                slot_count=slot_count_for_islands,
                                slot_index=slot_idx,
                            )
                        predictions[anchor] = float(pred)
            dependent_pre_ms, dependent_pre_source, dependent_pre_safe_prob = _dependent_preutterance_ms(
                row,
                tracks,
                constrained_predictions=constrained_predictions,
                filename_slot_predictions=filename_slot_predictions,
                joint_predictions=joint_predictions,
                supervised_gate_models=supervised_gate_models,
                supervised_gate_threshold=float(supervised_gate_threshold),
                supervised_gate_thresholds_by_anchor=supervised_gate_thresholds_by_anchor,
                require_primary_transition_token=bool(require_primary_transition_token),
                fallback_pre_ms=predictions.get("preutterance"),
                slot_count=slot_count_for_islands,
                slot_index=slot_idx,
            )
            dependent_offset_ms: float | None = None
            dependent_predictions: dict[str, float] = {}
            if dependent_pre_ms is not None:
                duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
                dependent_predictions["preutterance"] = float(dependent_pre_ms)
                if "offset" in anchors or "overlap" in anchors:
                    dependent_offset_ms = _dependent_offset_ms(
                        row,
                        tracks,
                        preutterance_ms=float(dependent_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                    if dependent_offset_ms is not None:
                        dependent_predictions["offset"] = float(dependent_offset_ms)
                if "offset" in anchors and dependent_offset_ms is not None:
                    target = _target_ms(row, "offset")
                    if target is not None:
                        err = abs(float(dependent_offset_ms) - float(target))
                        offset_gap_model_errors_by_anchor["offset"].append(err)
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_offset_gap_from_preutterance",
                            "offset",
                            row,
                            pred_ms=float(dependent_offset_ms),
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        _record_failure_breakdown(
                            failure_breakdown_records,
                            series="model_offset_gap_from_preutterance",
                            anchor="offset",
                            row=row,
                            error_ms=err,
                            pred_ms=float(dependent_offset_ms),
                            duration_ms=duration_ms,
                            dependent_preutterance_source=dependent_pre_source,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                if "overlap" in anchors and dependent_offset_ms is not None:
                    target = _target_ms(row, "overlap")
                    if target is not None:
                        dependent_overlap_ms = _dependent_overlap_ms(
                            row,
                            offset_ms=float(dependent_offset_ms),
                            preutterance_ms=float(dependent_pre_ms),
                            relative_anchor_priors=relative_anchor_priors,
                        )
                        dependent_predictions["overlap"] = float(dependent_overlap_ms)
                        err = abs(float(dependent_overlap_ms) - float(target))
                        overlap_relative_gap_errors_by_anchor["overlap"].append(err)
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_overlap_relative_from_preutterance",
                            "overlap",
                            row,
                            pred_ms=float(dependent_overlap_ms),
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        _record_failure_breakdown(
                            failure_breakdown_records,
                            series="model_overlap_relative_from_preutterance",
                            anchor="overlap",
                            row=row,
                            error_ms=err,
                            pred_ms=float(dependent_overlap_ms),
                            duration_ms=duration_ms,
                            dependent_preutterance_source=dependent_pre_source,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                if "fixed_end" in anchors:
                    target = _target_ms(row, "fixed_end")
                    if target is not None:
                        dependent_fixed_ms = _dependent_fixed_end_ms(
                            row,
                            tracks,
                            preutterance_ms=float(dependent_pre_ms),
                            relative_anchor_priors=relative_anchor_priors,
                        )
                        dependent_predictions["fixed_end"] = float(dependent_fixed_ms)
                        err = abs(float(dependent_fixed_ms) - float(target))
                        fixed_end_relative_gap_errors_by_anchor["fixed_end"].append(err)
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_fixed_end_relative_from_preutterance",
                            "fixed_end",
                            row,
                            pred_ms=float(dependent_fixed_ms),
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        _record_failure_breakdown(
                            failure_breakdown_records,
                            series="model_fixed_end_relative_from_preutterance",
                            anchor="fixed_end",
                            row=row,
                            error_ms=err,
                            pred_ms=float(dependent_fixed_ms),
                            duration_ms=duration_ms,
                            dependent_preutterance_source=dependent_pre_source,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
            if bool(collect_local_failure_training) and dependent_predictions:
                broad_safe_context: dict[str, object] = {}
                if dependent_predictions.get("offset") is not None and dependent_predictions.get("preutterance") is not None:
                    _broad_overlap, broad_safe_context = _safe_overlap_ms(
                        row,
                        offset_ms=float(dependent_predictions["offset"]),
                        preutterance_ms=float(dependent_predictions["preutterance"]),
                        preferred_overlap_ms=dependent_predictions.get("overlap"),
                    )
                    if "overlap" in dependent_predictions:
                        dependent_predictions["overlap"] = float(_broad_overlap)
                broad_slot_exact = True
                broad_boundary_exact = True
                for gate_anchor, gate_pred in dependent_predictions.items():
                    gate_delta = _slot_shift_delta(
                        row,
                        pred_ms=float(gate_pred),
                        duration_ms=duration_ms,
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if gate_delta is None or int(gate_delta) != 0:
                        broad_slot_exact = False
                        if gate_anchor in {"preutterance", "overlap", "fixed_end"}:
                            broad_boundary_exact = False
                for train_anchor, train_pred in dependent_predictions.items():
                    if train_anchor not in anchors:
                        continue
                    train_target = _target_ms(row, train_anchor)
                    if train_target is None:
                        continue
                    local_failure_feature_samples[train_anchor].append(
                        _local_failure_features(
                            row,
                            tracks,
                            train_anchor,
                            pred_ms=float(train_pred),
                            predictions=dependent_predictions,
                            safe_overlap_context=broad_safe_context,
                            pre_safe_prob=dependent_pre_safe_prob,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                            all_slot_exact=bool(broad_slot_exact),
                            boundary_slot_exact=bool(broad_boundary_exact),
                        )
                    )
                    local_failure_feature_labels[train_anchor].append(
                        1 if abs(float(train_pred) - float(train_target)) <= 80.0 else 0
                    )
            routed_pre_ms, routed_pre_safe_prob, routed_reasons = _routed_preutterance_decision(
                row,
                constrained_predictions=constrained_predictions,
                filename_slot_predictions=filename_slot_predictions,
                joint_predictions=joint_predictions,
                supervised_gate_models=supervised_gate_models,
                supervised_gate_threshold=float(supervised_gate_threshold),
                supervised_gate_thresholds_by_anchor=supervised_gate_thresholds_by_anchor,
                require_primary_transition_token=bool(require_primary_transition_token),
                use_heuristic_gate=bool(use_heuristic_gate),
                gate_slot_disagreement=float(gate_slot_disagreement),
                gate_pred_disagreement_ms=float(gate_pred_disagreement_ms),
                gate_joint_disagreement_ms=float(gate_joint_disagreement_ms),
                gate_order_disagreement=float(gate_order_disagreement),
                gate_joint_score_min=float(gate_joint_score_min),
                gate_review_mojibake=bool(gate_review_mojibake),
                gate_review_duplicate_tokens=bool(gate_review_duplicate_tokens),
            )
            routed_predictions: dict[str, float] = {}
            routed_source = "routed_gated_constrained_preutterance"
            if routed_pre_ms is not None:
                duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
                routed_predictions["preutterance"] = float(routed_pre_ms)
                routed_offset_ms: float | None = None
                if "offset" in anchors or "overlap" in anchors:
                    routed_offset_ms = _dependent_offset_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                    routed_predictions["offset"] = float(routed_offset_ms)
                if "overlap" in anchors and routed_offset_ms is not None:
                    routed_predictions["overlap"] = _dependent_overlap_ms(
                        row,
                        offset_ms=float(routed_offset_ms),
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                if "fixed_end" in anchors:
                    routed_predictions["fixed_end"] = _dependent_fixed_end_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
            if "overlap" in anchors:
                _record_overlap_safety(
                    overlap_safety_records,
                    series="model_routed_dependent_params",
                    row=row,
                    offset_ms=routed_predictions.get("offset"),
                    overlap_ms=routed_predictions.get("overlap"),
                    preutterance_ms=routed_predictions.get("preutterance"),
                    fixed_end_ms=routed_predictions.get("fixed_end"),
                    accepted=routed_predictions.get("overlap") is not None,
                    policy="manual_gap_dependent",
                )
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                routed_dependent_total_by_anchor[anchor] += 1
                routed_pred = routed_predictions.get(anchor)
                if routed_pred is None:
                    routed_dependent_review_by_anchor[anchor] += 1
                    reasons = routed_reasons or ("missing_routed_prediction",)
                    for reason in reasons:
                        routed_dependent_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                routed_dependent_accept_by_anchor[anchor] += 1
                routed_err = abs(float(routed_pred) - float(target))
                routed_dependent_errors_by_anchor[anchor].append(routed_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_routed_dependent_params",
                    anchor,
                    row,
                    pred_ms=float(routed_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                _record_failure_breakdown(
                    failure_breakdown_records,
                    series="model_routed_dependent_params",
                    anchor=anchor,
                    row=row,
                    error_ms=routed_err,
                    pred_ms=float(routed_pred),
                    duration_ms=duration_ms,
                    dependent_preutterance_source=routed_source,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if routed_err > 60.0 and len(worst_routed_dependent) < 200:
                    worst_routed_dependent.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(routed_pred),
                                error_ms=routed_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_routed_dependent_params",
                            "dependent_preutterance_source": routed_source,
                            "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                            "route_decision": "accept",
                        }
                    )
            calibrated_predictions: dict[str, float] = {}
            calibrated_reasons: list[str] = []
            calibrated_source = "routed_calibrated_preutterance"
            if routed_pre_ms is None:
                calibrated_reasons.extend(routed_reasons or ("missing_routed_preutterance",))
            else:
                duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
                calibrated_offset_ms: float | None = None
                if "offset" in anchors or "overlap" in anchors:
                    calibrated_offset_ms = _dependent_offset_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                        require_stable_prior=True,
                    )
                    if calibrated_offset_ms is None:
                        calibrated_reasons.append("unstable_offset_gap_prior")
                    else:
                        calibrated_predictions["offset"] = float(calibrated_offset_ms)
                if "overlap" in anchors:
                    if calibrated_offset_ms is None:
                        calibrated_reasons.append("missing_calibrated_offset_for_overlap")
                    else:
                        calibrated_overlap_ms = _dependent_overlap_ms(
                            row,
                            offset_ms=float(calibrated_offset_ms),
                            preutterance_ms=float(routed_pre_ms),
                            relative_anchor_priors=relative_anchor_priors,
                            require_stable_prior=True,
                        )
                        if calibrated_overlap_ms is None:
                            calibrated_reasons.append("unstable_overlap_gap_prior")
                        else:
                            calibrated_predictions["overlap"] = float(calibrated_overlap_ms)
                if "fixed_end" in anchors:
                    calibrated_fixed_ms = _dependent_fixed_end_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                        require_stable_prior=True,
                    )
                    if calibrated_fixed_ms is None:
                        calibrated_reasons.append("unstable_fixed_end_gap_prior")
                    else:
                        calibrated_predictions["fixed_end"] = float(calibrated_fixed_ms)
                if not calibrated_reasons:
                    calibrated_predictions["preutterance"] = float(routed_pre_ms)
            calibrated_reasons = list(dict.fromkeys(calibrated_reasons))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                routed_calibrated_total_by_anchor[anchor] += 1
                calibrated_pred = calibrated_predictions.get(anchor)
                if calibrated_pred is None:
                    routed_calibrated_review_by_anchor[anchor] += 1
                    reasons = calibrated_reasons or ("missing_calibrated_prediction",)
                    for reason in reasons:
                        routed_calibrated_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                routed_calibrated_accept_by_anchor[anchor] += 1
                calibrated_err = abs(float(calibrated_pred) - float(target))
                routed_calibrated_errors_by_anchor[anchor].append(calibrated_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_routed_calibrated_params",
                    anchor,
                    row,
                    pred_ms=float(calibrated_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                _record_failure_breakdown(
                    failure_breakdown_records,
                    series="model_routed_calibrated_params",
                    anchor=anchor,
                    row=row,
                    error_ms=calibrated_err,
                    pred_ms=float(calibrated_pred),
                    duration_ms=duration_ms,
                    dependent_preutterance_source=calibrated_source,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if calibrated_err > 60.0 and len(worst_routed_calibrated) < 200:
                    worst_routed_calibrated.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(calibrated_pred),
                                error_ms=calibrated_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_routed_calibrated_params",
                            "dependent_preutterance_source": calibrated_source,
                            "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                            "route_decision": "accept",
                        }
                    )
            span_predictions: dict[str, float] = {}
            span_ratio_source = "missing_overlap_span_ratio"
            span_ratio_stats: dict[str, object] = {}
            if routed_pre_ms is None:
                span_reasons = tuple(routed_reasons or ("missing_routed_preutterance",))
            else:
                duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
                span_reasons = ()
                span_predictions["preutterance"] = float(routed_pre_ms)
                span_offset_ms = routed_predictions.get("offset")
                if span_offset_ms is None:
                    span_offset_ms = _dependent_offset_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                if span_offset_ms is None:
                    span_reasons = ("missing_span_offset",)
                else:
                    span_predictions["offset"] = float(span_offset_ms)
                    span_overlap_ms, span_ratio_source, span_ratio_stats = _overlap_span_ratio_ms(
                        row,
                        offset_ms=float(span_offset_ms),
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                    span_predictions["overlap"] = float(span_overlap_ms)
                if "fixed_end" in anchors:
                    span_fixed_ms = routed_predictions.get("fixed_end")
                    if span_fixed_ms is None:
                        span_fixed_ms = _dependent_fixed_end_ms(
                            row,
                            tracks,
                            preutterance_ms=float(routed_pre_ms),
                            relative_anchor_priors=relative_anchor_priors,
                        )
                    if span_fixed_ms is not None:
                        span_predictions["fixed_end"] = float(span_fixed_ms)
            if "overlap" in anchors:
                _record_overlap_safety(
                    overlap_safety_records,
                    series="model_routed_overlap_span_params",
                    row=row,
                    offset_ms=span_predictions.get("offset"),
                    overlap_ms=span_predictions.get("overlap"),
                    preutterance_ms=span_predictions.get("preutterance"),
                    fixed_end_ms=span_predictions.get("fixed_end"),
                    accepted=span_predictions.get("overlap") is not None,
                    policy=f"span_ratio:{span_ratio_source}",
                )
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                routed_overlap_span_total_by_anchor[anchor] += 1
                span_pred = span_predictions.get(anchor)
                if span_pred is None:
                    routed_overlap_span_review_by_anchor[anchor] += 1
                    reasons = span_reasons or ("missing_overlap_span_prediction",)
                    for reason in reasons:
                        routed_overlap_span_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                routed_overlap_span_accept_by_anchor[anchor] += 1
                span_err = abs(float(span_pred) - float(target))
                routed_overlap_span_errors_by_anchor[anchor].append(span_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_routed_overlap_span_params",
                    anchor,
                    row,
                    pred_ms=float(span_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                _record_failure_breakdown(
                    failure_breakdown_records,
                    series="model_routed_overlap_span_params",
                    anchor=anchor,
                    row=row,
                    error_ms=span_err,
                    pred_ms=float(span_pred),
                    duration_ms=duration_ms,
                    dependent_preutterance_source=f"overlap_span_ratio:{span_ratio_source}",
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if span_err > 60.0 and len(worst_routed_overlap_span) < 200:
                    worst_routed_overlap_span.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(span_pred),
                                error_ms=span_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_routed_overlap_span_params",
                            "dependent_preutterance_source": f"overlap_span_ratio:{span_ratio_source}",
                            "overlap_ratio_prior_stats": span_ratio_stats,
                            "route_decision": "accept",
                        }
                    )
            safe_predictions: dict[str, float] = {}
            safe_reasons: list[str] = []
            safe_overlap_context: dict[str, object] = {}
            safe_source = "safe_overlap_policy"
            if routed_pre_ms is None:
                safe_reasons.extend(routed_reasons or ("missing_routed_preutterance",))
            else:
                safe_predictions["preutterance"] = float(routed_pre_ms)
                safe_offset_ms = routed_predictions.get("offset")
                if safe_offset_ms is None:
                    safe_offset_ms = _dependent_offset_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                if safe_offset_ms is None:
                    safe_reasons.append("missing_safe_overlap_offset")
                else:
                    safe_predictions["offset"] = float(safe_offset_ms)
                    if "overlap" in anchors:
                        safe_overlap_ms, safe_overlap_context = _safe_overlap_ms(
                            row,
                            offset_ms=float(safe_offset_ms),
                            preutterance_ms=float(routed_pre_ms),
                            preferred_overlap_ms=span_predictions.get("overlap", routed_predictions.get("overlap")),
                        )
                        safe_predictions["overlap"] = float(safe_overlap_ms)
                        safe_source = f"safe_overlap:{safe_overlap_context.get('policy', 'unknown')}"
                safe_fixed_ms = routed_predictions.get("fixed_end")
                if safe_fixed_ms is None:
                    safe_fixed_ms = _dependent_fixed_end_ms(
                        row,
                        tracks,
                        preutterance_ms=float(routed_pre_ms),
                        relative_anchor_priors=relative_anchor_priors,
                    )
                if safe_fixed_ms is None:
                    safe_reasons.append("missing_safe_overlap_fixed_end")
                else:
                    safe_predictions["fixed_end"] = float(safe_fixed_ms)
            safe_reasons = list(dict.fromkeys(safe_reasons))
            if "overlap" in anchors:
                _record_overlap_safety(
                    overlap_safety_records,
                    series="model_routed_safe_overlap_params",
                    row=row,
                    offset_ms=safe_predictions.get("offset"),
                    overlap_ms=safe_predictions.get("overlap"),
                    preutterance_ms=safe_predictions.get("preutterance"),
                    fixed_end_ms=safe_predictions.get("fixed_end"),
                    accepted=safe_predictions.get("overlap") is not None and safe_predictions.get("fixed_end") is not None,
                    policy=str(safe_overlap_context.get("policy", "safe_overlap")),
                )
            slot_exact_reasons: list[str] = list(safe_reasons)
            for gate_anchor in anchors:
                gate_pred = safe_predictions.get(gate_anchor)
                if gate_pred is None:
                    slot_exact_reasons.append(f"missing_slot_exact_prediction:{gate_anchor}")
                    continue
                gate_delta = _slot_shift_delta(
                    row,
                    pred_ms=float(gate_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if gate_delta is None:
                    slot_exact_reasons.append(f"unknown_slot_delta:{gate_anchor}")
                elif int(gate_delta) != 0:
                    slot_exact_reasons.append(f"slot_not_exact:{gate_anchor}:{_slot_shift_bucket(gate_delta)}")
            slot_exact_reasons = list(dict.fromkeys(slot_exact_reasons))
            slot_exact_accept = not slot_exact_reasons
            boundary_slot_exact_reasons: list[str] = list(safe_reasons)
            boundary_gate_anchors = [anchor for anchor in anchors if anchor in {"preutterance", "overlap", "fixed_end"}]
            if not boundary_gate_anchors:
                boundary_gate_anchors = list(anchors)
            for gate_anchor in boundary_gate_anchors:
                gate_pred = safe_predictions.get(gate_anchor)
                if gate_pred is None:
                    boundary_slot_exact_reasons.append(f"missing_boundary_slot_prediction:{gate_anchor}")
                    continue
                gate_delta = _slot_shift_delta(
                    row,
                    pred_ms=float(gate_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if gate_delta is None:
                    boundary_slot_exact_reasons.append(f"unknown_boundary_slot_delta:{gate_anchor}")
                elif int(gate_delta) != 0:
                    boundary_slot_exact_reasons.append(
                        f"boundary_slot_not_exact:{gate_anchor}:{_slot_shift_bucket(gate_delta)}"
                    )
            boundary_slot_exact_reasons = list(dict.fromkeys(boundary_slot_exact_reasons))
            boundary_slot_exact_accept = not boundary_slot_exact_reasons
            if "overlap" in anchors:
                _record_overlap_safety(
                    overlap_safety_records,
                    series="model_routed_slot_exact_safe_params",
                    row=row,
                    offset_ms=safe_predictions.get("offset"),
                    overlap_ms=safe_predictions.get("overlap"),
                    preutterance_ms=safe_predictions.get("preutterance"),
                    fixed_end_ms=safe_predictions.get("fixed_end"),
                    accepted=bool(slot_exact_accept),
                    policy=str(safe_overlap_context.get("policy", "safe_overlap")),
                )
                _record_overlap_safety(
                    overlap_safety_records,
                    series="model_routed_boundary_slot_exact_params",
                    row=row,
                    offset_ms=safe_predictions.get("offset"),
                    overlap_ms=safe_predictions.get("overlap"),
                    preutterance_ms=safe_predictions.get("preutterance"),
                    fixed_end_ms=safe_predictions.get("fixed_end"),
                    accepted=bool(boundary_slot_exact_accept),
                    policy=str(safe_overlap_context.get("policy", "safe_overlap")),
                )
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                routed_safe_overlap_total_by_anchor[anchor] += 1
                routed_slot_exact_total_by_anchor[anchor] += 1
                routed_boundary_slot_exact_total_by_anchor[anchor] += 1
                routed_local_gate_total_by_anchor[anchor] += 1
                routed_parameter_policy_total_by_anchor[anchor] += 1
                safe_pred = safe_predictions.get(anchor)
                if safe_pred is None:
                    routed_safe_overlap_review_by_anchor[anchor] += 1
                    reasons = safe_reasons or ("missing_safe_overlap_prediction",)
                    for reason in reasons:
                        routed_safe_overlap_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    routed_slot_exact_review_by_anchor[anchor] += 1
                    slot_reasons = slot_exact_reasons or ("missing_safe_overlap_prediction",)
                    for reason in slot_reasons:
                        routed_slot_exact_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    routed_boundary_slot_exact_review_by_anchor[anchor] += 1
                    boundary_reasons = boundary_slot_exact_reasons or ("missing_safe_overlap_prediction",)
                    for reason in boundary_reasons:
                        routed_boundary_slot_exact_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    routed_local_gate_review_by_anchor[anchor] += 1
                    local_reasons = boundary_slot_exact_reasons or safe_reasons or ("missing_safe_overlap_prediction",)
                    for reason in local_reasons:
                        routed_local_gate_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    policy_missing_accepted = False
                    if anchor == "preutterance":
                        primary_pre = constrained_predictions.get((id(row), "preutterance"))
                        if primary_pre is not None:
                            fallback_pre = float(primary_pre.get("pred_ms", 0.0) or 0.0)
                            vcv_pred, vcv_source, vcv_reason = _vcv_initial_cv_preutterance_policy(
                                row,
                                tracks,
                                island=island,
                                fallback_pred_ms=fallback_pre,
                                fallback_safe_prob=routed_pre_safe_prob,
                                duration_ms=duration_ms,
                                slot_count=slot_count_for_islands,
                                slot_index=slot_idx,
                            )
                            if vcv_pred is not None:
                                policy_err = abs(float(vcv_pred) - float(target))
                                routed_parameter_policy_accept_by_anchor[anchor] += 1
                                routed_parameter_policy_errors_by_anchor[anchor].append(policy_err)
                                _record_slot_shift(
                                    slot_shift_by_series_anchor,
                                    "model_routed_parameter_policy_params",
                                    anchor,
                                    row,
                                    pred_ms=float(vcv_pred),
                                    duration_ms=duration_ms,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                _record_failure_breakdown(
                                    failure_breakdown_records,
                                    series="model_routed_parameter_policy_params",
                                    anchor=anchor,
                                    row=row,
                                    error_ms=policy_err,
                                    pred_ms=float(vcv_pred),
                                    duration_ms=duration_ms,
                                    dependent_preutterance_source=vcv_source,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                if policy_err > 60.0 and len(worst_routed_parameter_policy) < 200:
                                    worst_routed_parameter_policy.append(
                                        {
                                            **_failure_example_base(
                                                row,
                                                wav_path=wav_path,
                                                anchor=anchor,
                                                target_ms=float(target),
                                                pred_ms=float(vcv_pred),
                                                error_ms=policy_err,
                                                slot_pos_norm=float(slot_idx + 0.5)
                                                / max(1.0, float(slot_count_for_islands)),
                                            ),
                                            "source_series": "model_routed_parameter_policy_params",
                                            "dependent_preutterance_source": vcv_source,
                                            "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                            "safe_overlap_context": safe_overlap_context,
                                            "route_decision": "accept",
                                        }
                                    )
                                policy_missing_accepted = True
                            elif vcv_reason != "not_vcv_initial_cv":
                                local_reasons = tuple(dict.fromkeys((*local_reasons, vcv_reason)))
                            if not policy_missing_accepted:
                                vcv_pred, vcv_source, vcv_reason = _vcv_sonorant_preutterance_policy(
                                    row,
                                    tracks,
                                    island=island,
                                    fallback_pred_ms=fallback_pre,
                                    fallback_safe_prob=routed_pre_safe_prob,
                                    duration_ms=duration_ms,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                if vcv_pred is not None:
                                    policy_err = abs(float(vcv_pred) - float(target))
                                    routed_parameter_policy_accept_by_anchor[anchor] += 1
                                    routed_parameter_policy_errors_by_anchor[anchor].append(policy_err)
                                    _record_slot_shift(
                                        slot_shift_by_series_anchor,
                                        "model_routed_parameter_policy_params",
                                        anchor,
                                        row,
                                        pred_ms=float(vcv_pred),
                                        duration_ms=duration_ms,
                                        slot_count=slot_count_for_islands,
                                        slot_index=slot_idx,
                                    )
                                    _record_failure_breakdown(
                                        failure_breakdown_records,
                                        series="model_routed_parameter_policy_params",
                                        anchor=anchor,
                                        row=row,
                                        error_ms=policy_err,
                                        pred_ms=float(vcv_pred),
                                        duration_ms=duration_ms,
                                        dependent_preutterance_source=vcv_source,
                                        slot_count=slot_count_for_islands,
                                        slot_index=slot_idx,
                                    )
                                    if policy_err > 60.0 and len(worst_routed_parameter_policy) < 200:
                                        worst_routed_parameter_policy.append(
                                            {
                                                **_failure_example_base(
                                                    row,
                                                    wav_path=wav_path,
                                                    anchor=anchor,
                                                    target_ms=float(target),
                                                    pred_ms=float(vcv_pred),
                                                    error_ms=policy_err,
                                                    slot_pos_norm=float(slot_idx + 0.5)
                                                    / max(1.0, float(slot_count_for_islands)),
                                                ),
                                                "source_series": "model_routed_parameter_policy_params",
                                                "dependent_preutterance_source": vcv_source,
                                                "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                                "safe_overlap_context": safe_overlap_context,
                                                "route_decision": "accept",
                                            }
                                        )
                                    policy_missing_accepted = True
                                elif vcv_reason != "not_vcv_sonorant":
                                    local_reasons = tuple(dict.fromkeys((*local_reasons, vcv_reason)))
                            if not policy_missing_accepted:
                                vcv_pred, vcv_source, vcv_reason = _slot_exact_island_start_preutterance_policy(
                                    row,
                                    tracks,
                                    island=island,
                                    fallback_pred_ms=fallback_pre,
                                    fallback_safe_prob=routed_pre_safe_prob,
                                    duration_ms=duration_ms,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                if vcv_pred is not None:
                                    policy_err = abs(float(vcv_pred) - float(target))
                                    routed_parameter_policy_accept_by_anchor[anchor] += 1
                                    routed_parameter_policy_errors_by_anchor[anchor].append(policy_err)
                                    _record_slot_shift(
                                        slot_shift_by_series_anchor,
                                        "model_routed_parameter_policy_params",
                                        anchor,
                                        row,
                                        pred_ms=float(vcv_pred),
                                        duration_ms=duration_ms,
                                        slot_count=slot_count_for_islands,
                                        slot_index=slot_idx,
                                    )
                                    _record_failure_breakdown(
                                        failure_breakdown_records,
                                        series="model_routed_parameter_policy_params",
                                        anchor=anchor,
                                        row=row,
                                        error_ms=policy_err,
                                        pred_ms=float(vcv_pred),
                                        duration_ms=duration_ms,
                                        dependent_preutterance_source=vcv_source,
                                        slot_count=slot_count_for_islands,
                                        slot_index=slot_idx,
                                    )
                                    if policy_err > 60.0 and len(worst_routed_parameter_policy) < 200:
                                        worst_routed_parameter_policy.append(
                                            {
                                                **_failure_example_base(
                                                    row,
                                                    wav_path=wav_path,
                                                    anchor=anchor,
                                                    target_ms=float(target),
                                                    pred_ms=float(vcv_pred),
                                                    error_ms=policy_err,
                                                    slot_pos_norm=float(slot_idx + 0.5)
                                                    / max(1.0, float(slot_count_for_islands)),
                                                ),
                                                "source_series": "model_routed_parameter_policy_params",
                                                "dependent_preutterance_source": vcv_source,
                                                "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                                "safe_overlap_context": safe_overlap_context,
                                                "route_decision": "accept",
                                            }
                                        )
                                    policy_missing_accepted = True
                                elif vcv_reason != "not_slot_exact_island_start_preutterance":
                                    local_reasons = tuple(dict.fromkeys((*local_reasons, vcv_reason)))
                    if not policy_missing_accepted:
                        routed_parameter_policy_review_by_anchor[anchor] += 1
                        for reason in local_reasons:
                            routed_parameter_policy_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                routed_safe_overlap_accept_by_anchor[anchor] += 1
                safe_err = abs(float(safe_pred) - float(target))
                routed_safe_overlap_errors_by_anchor[anchor].append(safe_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_routed_safe_overlap_params",
                    anchor,
                    row,
                    pred_ms=float(safe_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                _record_failure_breakdown(
                    failure_breakdown_records,
                    series="model_routed_safe_overlap_params",
                    anchor=anchor,
                    row=row,
                    error_ms=safe_err,
                    pred_ms=float(safe_pred),
                    duration_ms=duration_ms,
                    dependent_preutterance_source=safe_source if anchor == "overlap" else "routed_safe_overlap",
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if safe_err > 60.0 and len(worst_routed_safe_overlap) < 200:
                    worst_routed_safe_overlap.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(safe_pred),
                                error_ms=safe_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_routed_safe_overlap_params",
                            "dependent_preutterance_source": safe_source,
                            "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                            "safe_overlap_context": safe_overlap_context,
                            "route_decision": "accept",
                        }
                    )
                local_features = _local_failure_features(
                    row,
                    tracks,
                    anchor,
                    pred_ms=float(safe_pred),
                    predictions=safe_predictions,
                    safe_overlap_context=safe_overlap_context,
                    pre_safe_prob=routed_pre_safe_prob,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                    all_slot_exact=bool(slot_exact_accept),
                    boundary_slot_exact=bool(boundary_slot_exact_accept),
                )
                local_model = (local_failure_models or {}).get(anchor)
                local_safe_prob = _local_failure_safe_probability(local_model, local_features)
                if not boundary_slot_exact_accept:
                    routed_local_gate_review_by_anchor[anchor] += 1
                    reasons = boundary_slot_exact_reasons or ("boundary_slot_exact_rejected",)
                    for reason in reasons:
                        routed_local_gate_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                elif local_model is None:
                    routed_local_gate_review_by_anchor[anchor] += 1
                    routed_local_gate_review_reason_by_anchor[f"{anchor}|missing_local_failure_model"] += 1
                elif local_safe_prob is None or local_safe_prob < float(local_gate_threshold):
                    routed_local_gate_review_by_anchor[anchor] += 1
                    routed_local_gate_review_reason_by_anchor[f"{anchor}|local_hardfail_risk"] += 1
                else:
                    routed_local_gate_accept_by_anchor[anchor] += 1
                    routed_local_gate_errors_by_anchor[anchor].append(safe_err)
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_routed_local_hardfail_gate_params",
                        anchor,
                        row,
                        pred_ms=float(safe_pred),
                        duration_ms=duration_ms,
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    _record_failure_breakdown(
                        failure_breakdown_records,
                        series="model_routed_local_hardfail_gate_params",
                        anchor=anchor,
                        row=row,
                        error_ms=safe_err,
                        pred_ms=float(safe_pred),
                        duration_ms=duration_ms,
                        dependent_preutterance_source=f"local_hardfail_gate:{local_safe_prob:.3f}",
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if safe_err > 60.0 and len(worst_routed_local_gate) < 200:
                        worst_routed_local_gate.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=float(safe_pred),
                                    error_ms=safe_err,
                                    slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                ),
                                "source_series": "model_routed_local_hardfail_gate_params",
                                "dependent_preutterance_source": "local_hardfail_gate",
                                "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                "local_failure_safe_prob": local_safe_prob,
                                "safe_overlap_context": safe_overlap_context,
                                "route_decision": "accept",
                            }
                        )
                if not boundary_slot_exact_accept:
                    routed_boundary_slot_exact_review_by_anchor[anchor] += 1
                    reasons = boundary_slot_exact_reasons or ("boundary_slot_exact_rejected",)
                    for reason in reasons:
                        routed_boundary_slot_exact_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                else:
                    routed_boundary_slot_exact_accept_by_anchor[anchor] += 1
                    boundary_pred = safe_predictions.get(anchor)
                    if boundary_pred is None:
                        routed_boundary_slot_exact_review_by_anchor[anchor] += 1
                        routed_boundary_slot_exact_review_reason_by_anchor[
                            f"{anchor}|missing_boundary_slot_exact_prediction"
                        ] += 1
                    else:
                        boundary_err = abs(float(boundary_pred) - float(target))
                        routed_boundary_slot_exact_errors_by_anchor[anchor].append(boundary_err)
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_routed_boundary_slot_exact_params",
                            anchor,
                            row,
                            pred_ms=float(boundary_pred),
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        _record_failure_breakdown(
                            failure_breakdown_records,
                            series="model_routed_boundary_slot_exact_params",
                            anchor=anchor,
                            row=row,
                            error_ms=boundary_err,
                            pred_ms=float(boundary_pred),
                            duration_ms=duration_ms,
                            dependent_preutterance_source=safe_source if anchor == "overlap" else "boundary_slot_exact",
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        if boundary_err > 60.0 and len(worst_routed_boundary_slot_exact) < 200:
                            worst_routed_boundary_slot_exact.append(
                                {
                                    **_failure_example_base(
                                        row,
                                        wav_path=wav_path,
                                        anchor=anchor,
                                        target_ms=float(target),
                                        pred_ms=float(boundary_pred),
                                        error_ms=boundary_err,
                                        slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                    ),
                                    "source_series": "model_routed_boundary_slot_exact_params",
                                    "dependent_preutterance_source": safe_source,
                                    "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                    "safe_overlap_context": safe_overlap_context,
                                    "route_decision": "accept",
                                }
                            )
                policy_pred: float | None = None
                policy_source = ""
                policy_reasons: list[str] = []
                policy_local_threshold = max(0.99, float(local_gate_threshold))
                if anchor == "preutterance":
                    if boundary_slot_exact_accept:
                        format_type = str(row.get("format_type", "") or "").strip().lower()
                        role = str(row.get("alias_role", "") or "").strip().lower()
                        if format_type == "vcv" and role == "cv" and float(safe_pred) > 150.0:
                            vcv_pred, vcv_source, vcv_reason = _vcv_initial_cv_preutterance_policy(
                                row,
                                tracks,
                                island=island,
                                fallback_pred_ms=float(safe_pred),
                                fallback_safe_prob=routed_pre_safe_prob,
                                duration_ms=duration_ms,
                                slot_count=slot_count_for_islands,
                                slot_index=slot_idx,
                            )
                            if vcv_pred is not None:
                                policy_pred = float(vcv_pred)
                                policy_source = vcv_source
                            else:
                                policy_reasons.append(vcv_reason)
                        elif format_type == "vcv" and role != "cv":
                            vcv_pred, vcv_source, vcv_reason = _vcv_sonorant_preutterance_policy(
                                row,
                                tracks,
                                island=island,
                                fallback_pred_ms=float(safe_pred),
                                fallback_safe_prob=routed_pre_safe_prob,
                                duration_ms=duration_ms,
                                slot_count=slot_count_for_islands,
                                slot_index=slot_idx,
                            )
                            if vcv_pred is not None:
                                policy_pred = float(vcv_pred)
                                policy_source = vcv_source
                            elif vcv_reason != "not_vcv_sonorant":
                                policy_reasons.append(vcv_reason)
                            else:
                                vcv_pred, vcv_source, vcv_reason = _slot_exact_island_start_preutterance_policy(
                                    row,
                                    tracks,
                                    island=island,
                                    fallback_pred_ms=float(safe_pred),
                                    fallback_safe_prob=routed_pre_safe_prob,
                                    duration_ms=duration_ms,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                if vcv_pred is not None:
                                    policy_pred = float(vcv_pred)
                                    policy_source = vcv_source
                                elif routed_pre_safe_prob is None or routed_pre_safe_prob < 0.98:
                                    if vcv_reason != "not_slot_exact_island_start_preutterance":
                                        policy_reasons.append(vcv_reason)
                                    policy_reasons.append("vcv_transition_preutterance_low_margin")
                                else:
                                    policy_pred = float(safe_pred)
                                    policy_source = "boundary_exact_preutterance"
                        else:
                            policy_pred = float(safe_pred)
                            policy_source = "boundary_exact_preutterance"
                    else:
                        vcv_pred, vcv_source, vcv_reason = _vcv_initial_cv_preutterance_policy(
                            row,
                            tracks,
                            island=island,
                            fallback_pred_ms=float(safe_pred),
                            fallback_safe_prob=routed_pre_safe_prob,
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        if vcv_pred is not None:
                            policy_pred = float(vcv_pred)
                            policy_source = vcv_source
                        else:
                            if vcv_reason != "not_vcv_initial_cv":
                                policy_reasons.append(vcv_reason)
                            vcv_pred, vcv_source, sonorant_reason = _vcv_sonorant_preutterance_policy(
                                row,
                                tracks,
                                island=island,
                                fallback_pred_ms=float(safe_pred),
                                fallback_safe_prob=routed_pre_safe_prob,
                                duration_ms=duration_ms,
                                slot_count=slot_count_for_islands,
                                slot_index=slot_idx,
                            )
                            if vcv_pred is not None:
                                policy_pred = float(vcv_pred)
                                policy_source = vcv_source
                            else:
                                vcv_pred, vcv_source, generic_reason = _slot_exact_island_start_preutterance_policy(
                                    row,
                                    tracks,
                                    island=island,
                                    fallback_pred_ms=float(safe_pred),
                                    fallback_safe_prob=routed_pre_safe_prob,
                                    duration_ms=duration_ms,
                                    slot_count=slot_count_for_islands,
                                    slot_index=slot_idx,
                                )
                                if vcv_pred is not None:
                                    policy_pred = float(vcv_pred)
                                    policy_source = vcv_source
                                else:
                                    policy_reasons.extend(boundary_slot_exact_reasons or ("boundary_slot_exact_rejected",))
                                    if generic_reason != "not_slot_exact_island_start_preutterance":
                                        policy_reasons.append(generic_reason)
                                if sonorant_reason != "not_vcv_sonorant":
                                    policy_reasons.append(sonorant_reason)
                elif anchor in {"offset", "overlap"}:
                    if not boundary_slot_exact_accept:
                        policy_reasons.extend(boundary_slot_exact_reasons or ("boundary_slot_exact_rejected",))
                    elif local_model is None:
                        policy_reasons.append("missing_local_failure_model")
                    elif local_safe_prob is None or local_safe_prob < float(policy_local_threshold):
                        policy_reasons.append("policy_local_hardfail_risk")
                    else:
                        policy_pred = float(safe_pred)
                        policy_source = f"local_gate_ge_{policy_local_threshold:.2f}"
                elif anchor == "fixed_end":
                    primary_fixed = constrained_predictions.get((id(row), "fixed_end"))
                    fixed_prob = _supervised_safe_probability(
                        row,
                        "fixed_end",
                        primary_fixed,
                        gate_model=(supervised_gate_models or {}).get("fixed_end"),
                        filename_pred=filename_slot_predictions.get((id(row), "fixed_end")),
                        joint_pred=joint_predictions.get((id(row), "fixed_end")),
                        require_primary_transition_token=bool(require_primary_transition_token),
                    )
                    if primary_fixed is None:
                        policy_reasons.append("missing_fixed_end_constrained_prediction")
                    elif fixed_prob is None or fixed_prob < _threshold_for_anchor(
                        "fixed_end",
                        float(supervised_gate_threshold),
                        supervised_gate_thresholds_by_anchor,
                    ):
                        policy_reasons.append("fixed_end_supervised_failure_risk")
                    else:
                        fixed_pred = float(primary_fixed.get("pred_ms", 0.0) or 0.0)
                        fixed_delta = _slot_shift_delta(
                            row,
                            pred_ms=fixed_pred,
                            duration_ms=duration_ms,
                            slot_count=slot_count_for_islands,
                            slot_index=slot_idx,
                        )
                        if fixed_delta is None or int(fixed_delta) != 0:
                            policy_reasons.append(f"fixed_end_slot_not_exact:{_slot_shift_bucket(fixed_delta)}")
                        else:
                            policy_pred = fixed_pred
                            policy_source = f"fixed_end_gated_constrained:{fixed_prob:.3f}"
                else:
                    policy_reasons.append("unsupported_policy_anchor")

                if policy_pred is None:
                    routed_parameter_policy_review_by_anchor[anchor] += 1
                    for reason in dict.fromkeys(policy_reasons or ("missing_policy_prediction",)):
                        routed_parameter_policy_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                else:
                    policy_err = abs(float(policy_pred) - float(target))
                    routed_parameter_policy_accept_by_anchor[anchor] += 1
                    routed_parameter_policy_errors_by_anchor[anchor].append(policy_err)
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_routed_parameter_policy_params",
                        anchor,
                        row,
                        pred_ms=float(policy_pred),
                        duration_ms=duration_ms,
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    _record_failure_breakdown(
                        failure_breakdown_records,
                        series="model_routed_parameter_policy_params",
                        anchor=anchor,
                        row=row,
                        error_ms=policy_err,
                        pred_ms=float(policy_pred),
                        duration_ms=duration_ms,
                        dependent_preutterance_source=policy_source,
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if policy_err > 60.0 and len(worst_routed_parameter_policy) < 200:
                        worst_routed_parameter_policy.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=float(policy_pred),
                                    error_ms=policy_err,
                                    slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                ),
                                "source_series": "model_routed_parameter_policy_params",
                                "dependent_preutterance_source": policy_source,
                                "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                                "local_failure_safe_prob": local_safe_prob,
                                "safe_overlap_context": safe_overlap_context,
                                "route_decision": "accept",
                            }
                        )
                if not slot_exact_accept:
                    routed_slot_exact_review_by_anchor[anchor] += 1
                    reasons = slot_exact_reasons or ("slot_exact_rejected",)
                    for reason in reasons:
                        routed_slot_exact_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                routed_slot_exact_accept_by_anchor[anchor] += 1
                slot_exact_pred = safe_predictions.get(anchor)
                if slot_exact_pred is None:
                    routed_slot_exact_review_by_anchor[anchor] += 1
                    routed_slot_exact_review_reason_by_anchor[f"{anchor}|missing_slot_exact_prediction"] += 1
                    continue
                slot_exact_err = abs(float(slot_exact_pred) - float(target))
                routed_slot_exact_errors_by_anchor[anchor].append(slot_exact_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_routed_slot_exact_safe_params",
                    anchor,
                    row,
                    pred_ms=float(slot_exact_pred),
                    duration_ms=duration_ms,
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                _record_failure_breakdown(
                    failure_breakdown_records,
                    series="model_routed_slot_exact_safe_params",
                    anchor=anchor,
                    row=row,
                    error_ms=slot_exact_err,
                    pred_ms=float(slot_exact_pred),
                    duration_ms=duration_ms,
                    dependent_preutterance_source=safe_source if anchor == "overlap" else "slot_exact_safe_overlap",
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if slot_exact_err > 60.0 and len(worst_routed_slot_exact) < 200:
                    worst_routed_slot_exact.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(slot_exact_pred),
                                error_ms=slot_exact_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_routed_slot_exact_safe_params",
                            "dependent_preutterance_source": safe_source,
                            "dependent_preutterance_safe_prob": routed_pre_safe_prob,
                            "safe_overlap_context": safe_overlap_context,
                            "route_decision": "accept",
                        }
                    )
            row_safe, row_safety_warnings = _island_row_safe(assignment, island, predictions)
            island_context = _island_prediction_context(
                island_decode,
                assignment,
                island,
                slot_index=slot_idx,
                slot_count=slot_count_for_islands,
                safety_warnings=row_safety_warnings,
            )
            island_context["dependent_preutterance_source"] = dependent_pre_source
            island_context["dependent_preutterance_safe_prob"] = dependent_pre_safe_prob
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                pred_ms = predictions.get(anchor)
                if anchor in {"preutterance", "fixed_end"} and pred_ms is not None:
                    err = abs(float(pred_ms) - float(target))
                    vowel_island_lattice_errors_by_anchor[anchor].append(err)
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_vowel_island_lattice",
                        anchor,
                        row,
                        pred_ms=float(pred_ms),
                        duration_ms=float(tracks.duration_ms),
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if err > 60.0 and len(worst_vowel_island) < 200:
                        worst_vowel_island.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=float(pred_ms),
                                    error_ms=err,
                                    slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                ),
                                "source_series": "model_vowel_island_lattice",
                                "island_failure_reason": _classify_island_failure(anchor, err, island_context),
                                "island_context": island_context,
                                "island_overlay_events": island_events,
                            }
                        )
                if anchor == "offset" and pred_ms is not None:
                    local_err = abs(float(pred_ms) - float(target))
                    local_offset_from_island_errors_by_anchor[anchor].append(local_err)
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_local_offset_from_island_pre",
                        anchor,
                        row,
                        pred_ms=float(pred_ms),
                        duration_ms=float(tracks.duration_ms),
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if local_err > 60.0 and len(worst_vowel_island) < 200:
                        worst_vowel_island.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=float(pred_ms),
                                    error_ms=local_err,
                                    slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                ),
                                "source_series": "model_local_offset_from_island_pre",
                                "island_failure_reason": _classify_island_failure(anchor, local_err, island_context),
                                "island_context": island_context,
                                "island_overlay_events": island_events,
                            }
                        )
                if anchor == "overlap" and pred_ms is not None:
                    overlap_err = abs(float(pred_ms) - float(target))
                    overlap_relative_on_island_errors_by_anchor[anchor].append(overlap_err)
                    _record_slot_shift(
                        slot_shift_by_series_anchor,
                        "model_overlap_relative_on_island",
                        anchor,
                        row,
                        pred_ms=float(pred_ms),
                        duration_ms=float(tracks.duration_ms),
                        slot_count=slot_count_for_islands,
                        slot_index=slot_idx,
                    )
                    if overlap_err > 60.0 and len(worst_vowel_island) < 200:
                        worst_vowel_island.append(
                            {
                                **_failure_example_base(
                                    row,
                                    wav_path=wav_path,
                                    anchor=anchor,
                                    target_ms=float(target),
                                    pred_ms=float(pred_ms),
                                    error_ms=overlap_err,
                                    slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                                ),
                                "source_series": "model_overlap_relative_on_island",
                                "island_failure_reason": _classify_island_failure(anchor, overlap_err, island_context),
                                "island_context": island_context,
                                "island_overlay_events": island_events,
                            }
                        )

                vowel_island_total_by_anchor[anchor] += 1
                if pred_ms is None or not row_safe:
                    vowel_island_review_by_anchor[anchor] += 1
                    reasons = row_safety_warnings or ("missing_island_prediction",)
                    if pred_ms is None:
                        reasons = (*reasons, "missing_island_prediction")
                    for reason in reasons:
                        vowel_island_review_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                vowel_island_accept_by_anchor[anchor] += 1
                gated_err = abs(float(pred_ms) - float(target))
                vowel_island_gated_errors_by_anchor[anchor].append(gated_err)
                _record_slot_shift(
                    slot_shift_by_series_anchor,
                    "model_vowel_island_gated_accept",
                    anchor,
                    row,
                    pred_ms=float(pred_ms),
                    duration_ms=float(tracks.duration_ms),
                    slot_count=slot_count_for_islands,
                    slot_index=slot_idx,
                )
                if gated_err > 60.0 and len(worst_vowel_island_gated) < 200:
                    worst_vowel_island_gated.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=float(pred_ms),
                                error_ms=gated_err,
                                slot_pos_norm=float(slot_idx + 0.5) / max(1.0, float(slot_count_for_islands)),
                            ),
                            "source_series": "model_vowel_island_gated_accept",
                            "island_failure_reason": _classify_island_failure(anchor, gated_err, island_context),
                            "island_context": island_context,
                            "island_overlay_events": island_events,
                        }
                    )

        for row in wav_rows:
            filename_tokens = _filename_tokens_for_row(row)
            filename_slot_pos = _infer_filename_slot_pos(
                row,
                filename_tokens,
                require_primary_transition_token=bool(require_primary_transition_token),
            )
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                gate_total_by_anchor[anchor] += 1
                primary = constrained_predictions.get((id(row), anchor))
                gate_features = _gate_features(
                    row,
                    anchor,
                    primary,
                    filename_pred=filename_slot_predictions.get((id(row), anchor)),
                    joint_pred=joint_predictions.get((id(row), anchor)),
                    filename_slot_pos=filename_slot_pos,
                    filename_tokens=filename_tokens,
                )
                if bool(use_heuristic_gate):
                    reasons = _gate_constrained_prediction(
                        row,
                        anchor,
                        primary,
                        filename_pred=filename_slot_predictions.get((id(row), anchor)),
                        joint_pred=joint_predictions.get((id(row), anchor)),
                        filename_slot_pos=filename_slot_pos,
                        filename_tokens=filename_tokens,
                        slot_disagreement_threshold=float(gate_slot_disagreement),
                        pred_disagreement_ms=float(gate_pred_disagreement_ms),
                        joint_disagreement_ms=float(gate_joint_disagreement_ms),
                        order_disagreement_threshold=float(gate_order_disagreement),
                        joint_score_min=float(gate_joint_score_min),
                        review_mojibake=bool(gate_review_mojibake),
                        review_duplicate_tokens=bool(gate_review_duplicate_tokens),
                    )
                else:
                    reasons = []
                if bool(collect_gate_training):
                    if primary is None:
                        gate_feature_labels[anchor].append(0)
                    else:
                        pred_ms_for_label = float(primary.get("pred_ms", 0.0) or 0.0)
                        gate_feature_labels[anchor].append(1 if abs(pred_ms_for_label - float(target)) <= 100.0 else 0)
                    gate_feature_samples[anchor].append(gate_features)
                gate_model = (supervised_gate_models or {}).get(anchor)
                safe_prob: float | None = None
                if gate_model is not None and primary is not None:
                    try:
                        safe_prob = float(gate_model.predict_proba(gate_features.reshape(1, -1))[0, 1])
                    except Exception:
                        safe_prob = float(gate_model.predict(gate_features.reshape(1, -1))[0])
                    gate_sweep_records_by_anchor[anchor].append(
                        {
                            "safe_prob": float(safe_prob),
                            "error_ms": abs(float(primary.get("pred_ms", 0.0) or 0.0) - float(target)),
                            "primary_present": True,
                            "heuristic_rejected": bool(reasons),
                        }
                    )
                    anchor_gate_threshold = _threshold_for_anchor(
                        anchor,
                        float(supervised_gate_threshold),
                        supervised_gate_thresholds_by_anchor,
                    )
                    if safe_prob < float(anchor_gate_threshold):
                        reasons.append("supervised_failure_risk")
                elif supervised_gate_models:
                    gate_sweep_records_by_anchor[anchor].append(
                        {
                            "safe_prob": None,
                            "error_ms": None,
                            "primary_present": False,
                            "heuristic_rejected": bool(reasons),
                        }
                    )
                if reasons:
                    gate_review_by_anchor[anchor] += 1
                    for reason in reasons:
                        gate_reason_by_anchor[f"{anchor}|{reason}"] += 1
                    continue
                gate_accept_by_anchor[anchor] += 1
                assert primary is not None
                pred_ms = float(primary.get("pred_ms", 0.0) or 0.0)
                err = abs(pred_ms - float(target))
                gated_constrained_errors_by_anchor[anchor].append(err)
                if err > 60.0 and len(worst_gated_constrained) < 200:
                    worst_gated_constrained.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=pred_ms,
                                error_ms=err,
                                slot_pos_norm=float(primary.get("slot_pos_norm", _centered_slot_pos(row)) or 0.0),
                            ),
                            "joint_score": float(primary.get("joint_score", 0.0) or 0.0),
                            "time_order_norm": float(primary.get("time_order_norm", 0.0) or 0.0),
                            "gate_decision": "accept",
                        }
                    )

        for anchor in anchors:
            model = models.get(anchor)
            if model is None:
                continue
            options_by_row: list[list[CandidateOption]] = []
            rows_with_target: list[dict[str, object] | None] = []
            for row, filename_slot_pos in ordered_wav_rows:
                feature_row = _row_with_slot_pos(row, filename_slot_pos)
                target = _target_ms(row, anchor)
                if target is None:
                    rows_with_target.append(None)
                    options_by_row.append([])
                    continue
                indices = list(tracks.candidate_indices.get(anchor, ()))
                if not indices:
                    rows_with_target.append(row)
                    options_by_row.append([])
                    continue
                features = np.stack(
                    [manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx) for idx in indices],
                    axis=0,
                ).astype(np.float32)
                scores = _score_candidates(model, features)
                order_lookup = _time_norms(tracks, indices)
                slot_pos = float(filename_slot_pos)
                adjusted_scores = [
                    float(scores[pos])
                    + _family_anchor_prior_score(
                        row,
                        anchor,
                        candidate_time_norm=float(order_lookup.get(idx, 0.0)),
                        slot_pos_norm=slot_pos,
                        weight=float(joint_family_prior_weight),
                    )
                    for pos, idx in enumerate(indices)
                ]
                options_by_row.append(
                    [
                        CandidateOption(
                            candidate_index=int(idx),
                            time_ms=float(tracks.times_ms[idx]),
                            score=float(adjusted_scores[pos]),
                            order_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=slot_pos,
                        )
                        for pos, idx in enumerate(indices)
                    ]
                )
                rows_with_target.append(row)
            decoded = decode_monotonic_candidate_indices(options_by_row)
            for row, selected in zip(rows_with_target, decoded, strict=False):
                if row is None or selected is None:
                    continue
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                monotonic_errors_by_anchor[anchor].append(float(_candidate_error(tracks, int(selected), target)))

        for row in wav_rows:
            role = str(row.get("alias_role", "") or "")
            alias_family = manual_oto_alias_family(row.get("alias", ""))
            for anchor in anchors:
                target = _target_ms(row, anchor)
                if target is None:
                    continue
                total_by_anchor[anchor] += 1
                role_key = f"{role}|{anchor}"
                by_role_anchor_total[role_key] += 1
                family_key = f"{alias_family}|{anchor}"
                by_alias_family_anchor_total[family_key] += 1
                model = models.get(anchor)
                indices = list(tracks.candidate_indices.get(anchor, ()))
                if model is None or not indices:
                    missing_by_anchor[anchor] += 1
                    continue
                order_lookup = _time_norms(tracks, indices)
                slot_pos = _infer_filename_slot_pos(
                    row,
                    _filename_tokens_for_row(row),
                    require_primary_transition_token=bool(require_primary_transition_token),
                )
                if slot_pos is None:
                    slot_pos = _centered_slot_pos(row)
                feature_row = _row_with_slot_pos(row, slot_pos)
                features = np.stack(
                    [manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx) for idx in indices],
                    axis=0,
                ).astype(np.float32)
                scores = _score_candidates(model, features)
                best_idx = indices[int(np.argmax(scores))]
                err = _candidate_error(tracks, best_idx, target)
                errors_by_anchor[anchor].append(float(err))
                if err <= 30.0:
                    by_role_anchor_hit30[role_key] += 1
                    by_alias_family_anchor_hit30[family_key] += 1

                anchor_scores = tracks.anchor_scores.get(anchor, np.zeros_like(tracks.times_ms))
                raw_idx = max(indices, key=lambda idx: float(anchor_scores[idx]))
                raw_errors_by_anchor[anchor].append(float(_candidate_error(tracks, raw_idx, target)))
                adjusted_scores = np.asarray(
                    [
                        float(scores[pos])
                        + _family_anchor_prior_score(
                            row,
                            anchor,
                            candidate_time_norm=float(order_lookup.get(idx, 0.0)),
                            slot_pos_norm=float(slot_pos),
                            weight=float(family_prior_weight),
                        )
                        for pos, idx in enumerate(indices)
                    ],
                    dtype=np.float32,
                )
                family_prior_idx = indices[int(np.argmax(adjusted_scores))]
                family_prior_errors_by_anchor[anchor].append(
                    float(_candidate_error(tracks, family_prior_idx, target))
                )
                order_idx = min(indices, key=lambda idx: abs(order_lookup.get(idx, 0.0) - slot_pos))
                order_prior_errors_by_anchor[anchor].append(float(_candidate_error(tracks, order_idx, target)))
                if float(order_window) > 0.0:
                    windowed = [idx for idx in indices if abs(order_lookup.get(idx, 0.0) - slot_pos) <= float(order_window)]
                    if not windowed:
                        windowed = sorted(indices, key=lambda idx: abs(order_lookup.get(idx, 0.0) - slot_pos))[: max(1, min(5, len(indices)))]
                    window_features = np.stack(
                        [
                            manual_oto_candidate_features(feature_row, tracks, anchor=anchor, candidate_index=idx)
                            for idx in windowed
                        ],
                        axis=0,
                    ).astype(np.float32)
                    window_scores = _score_candidates(model, window_features)
                    window_best = windowed[int(np.argmax(window_scores))]
                    model_order_window_errors_by_anchor[anchor].append(float(_candidate_error(tracks, window_best, target)))
                oracle_errors_by_anchor[anchor].append(float(min(_candidate_error(tracks, idx, target) for idx in indices)))

                if err > 60.0 and len(worst) < 200:
                    pred_ms = float(tracks.times_ms[best_idx])
                    worst.append(
                        {
                            **_failure_example_base(
                                row,
                                wav_path=wav_path,
                                anchor=anchor,
                                target_ms=float(target),
                                pred_ms=pred_ms,
                                error_ms=float(err),
                                slot_pos_norm=float(slot_pos),
                            ),
                            "raw_score_best_error_ms": float(_candidate_error(tracks, raw_idx, target)),
                            "oracle_candidate_error_ms": float(min(_candidate_error(tracks, idx, target) for idx in indices)),
                        }
                    )

    def summarize(values: list[float], total: int, missing: int = 0) -> dict[str, object]:
        arr = np.asarray(values, dtype=np.float32)
        return {
            "total": int(total),
            "scored": int(len(values)),
            "missing_or_unscored": int(missing),
            "hit25": int(np.sum(arr <= 25.0)) if arr.size else 0,
            "hit30": int(np.sum(arr <= 30.0)) if arr.size else 0,
            "hit60": int(np.sum(arr <= 60.0)) if arr.size else 0,
            "recall25": float(np.mean(arr <= 25.0)) if arr.size else 0.0,
            "recall30": float(np.mean(arr <= 30.0)) if arr.size else 0.0,
            "recall60": float(np.mean(arr <= 60.0)) if arr.size else 0.0,
            "pass_le10_rate": float(np.mean(arr <= 10.0)) if arr.size else 0.0,
            "ordinary_30_60_rate": float(np.mean((arr > 30.0) & (arr <= 60.0))) if arr.size else 0.0,
            "review_gt60_rate": float(np.mean(arr > 60.0)) if arr.size else 0.0,
            "warning_60_80_rate": float(np.mean((arr > 60.0) & (arr <= 80.0))) if arr.size else 0.0,
            "reject_gt80_rate": float(np.mean(arr > 80.0)) if arr.size else 0.0,
            "median_error_ms": float(median(values)) if values else None,
            "p90_error_ms": float(np.percentile(arr, 90.0)) if arr.size else None,
            "hard_fail_gt100_rate": float(np.mean(arr > 100.0)) if arr.size else 0.0,
        }

    by_anchor: dict[str, dict[str, object]] = {}
    for anchor in anchors:
        total = int(total_by_anchor[anchor])
        gated_summary = summarize(
            gated_constrained_errors_by_anchor[anchor],
            int(gate_total_by_anchor[anchor]),
            int(gate_review_by_anchor[anchor]),
        )
        gated_summary.update(
            {
                "accepted": int(gate_accept_by_anchor[anchor]),
                "reviewed": int(gate_review_by_anchor[anchor]),
                "accept_rate": float(gate_accept_by_anchor[anchor] / gate_total_by_anchor[anchor])
                if gate_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(gate_review_by_anchor[anchor] / gate_total_by_anchor[anchor])
                if gate_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(gate_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        island_gated_summary = summarize(
            vowel_island_gated_errors_by_anchor[anchor],
            int(vowel_island_total_by_anchor[anchor]),
            int(vowel_island_review_by_anchor[anchor]),
        )
        island_gated_summary.update(
            {
                "accepted": int(vowel_island_accept_by_anchor[anchor]),
                "reviewed": int(vowel_island_review_by_anchor[anchor]),
                "accept_rate": float(vowel_island_accept_by_anchor[anchor] / vowel_island_total_by_anchor[anchor])
                if vowel_island_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(vowel_island_review_by_anchor[anchor] / vowel_island_total_by_anchor[anchor])
                if vowel_island_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(vowel_island_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_dependent_summary = summarize(
            routed_dependent_errors_by_anchor[anchor],
            int(routed_dependent_total_by_anchor[anchor]),
            int(routed_dependent_review_by_anchor[anchor]),
        )
        routed_dependent_summary.update(
            {
                "accepted": int(routed_dependent_accept_by_anchor[anchor]),
                "reviewed": int(routed_dependent_review_by_anchor[anchor]),
                "accept_rate": float(routed_dependent_accept_by_anchor[anchor] / routed_dependent_total_by_anchor[anchor])
                if routed_dependent_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_dependent_review_by_anchor[anchor] / routed_dependent_total_by_anchor[anchor])
                if routed_dependent_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_dependent_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_calibrated_summary = summarize(
            routed_calibrated_errors_by_anchor[anchor],
            int(routed_calibrated_total_by_anchor[anchor]),
            int(routed_calibrated_review_by_anchor[anchor]),
        )
        routed_calibrated_summary.update(
            {
                "accepted": int(routed_calibrated_accept_by_anchor[anchor]),
                "reviewed": int(routed_calibrated_review_by_anchor[anchor]),
                "accept_rate": float(routed_calibrated_accept_by_anchor[anchor] / routed_calibrated_total_by_anchor[anchor])
                if routed_calibrated_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_calibrated_review_by_anchor[anchor] / routed_calibrated_total_by_anchor[anchor])
                if routed_calibrated_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_calibrated_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_overlap_span_summary = summarize(
            routed_overlap_span_errors_by_anchor[anchor],
            int(routed_overlap_span_total_by_anchor[anchor]),
            int(routed_overlap_span_review_by_anchor[anchor]),
        )
        routed_overlap_span_summary.update(
            {
                "accepted": int(routed_overlap_span_accept_by_anchor[anchor]),
                "reviewed": int(routed_overlap_span_review_by_anchor[anchor]),
                "accept_rate": float(routed_overlap_span_accept_by_anchor[anchor] / routed_overlap_span_total_by_anchor[anchor])
                if routed_overlap_span_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_overlap_span_review_by_anchor[anchor] / routed_overlap_span_total_by_anchor[anchor])
                if routed_overlap_span_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_overlap_span_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_safe_overlap_summary = summarize(
            routed_safe_overlap_errors_by_anchor[anchor],
            int(routed_safe_overlap_total_by_anchor[anchor]),
            int(routed_safe_overlap_review_by_anchor[anchor]),
        )
        routed_safe_overlap_summary.update(
            {
                "accepted": int(routed_safe_overlap_accept_by_anchor[anchor]),
                "reviewed": int(routed_safe_overlap_review_by_anchor[anchor]),
                "accept_rate": float(routed_safe_overlap_accept_by_anchor[anchor] / routed_safe_overlap_total_by_anchor[anchor])
                if routed_safe_overlap_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_safe_overlap_review_by_anchor[anchor] / routed_safe_overlap_total_by_anchor[anchor])
                if routed_safe_overlap_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_safe_overlap_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_slot_exact_summary = summarize(
            routed_slot_exact_errors_by_anchor[anchor],
            int(routed_slot_exact_total_by_anchor[anchor]),
            int(routed_slot_exact_review_by_anchor[anchor]),
        )
        routed_slot_exact_summary.update(
            {
                "accepted": int(routed_slot_exact_accept_by_anchor[anchor]),
                "reviewed": int(routed_slot_exact_review_by_anchor[anchor]),
                "accept_rate": float(routed_slot_exact_accept_by_anchor[anchor] / routed_slot_exact_total_by_anchor[anchor])
                if routed_slot_exact_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_slot_exact_review_by_anchor[anchor] / routed_slot_exact_total_by_anchor[anchor])
                if routed_slot_exact_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_slot_exact_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_boundary_slot_exact_summary = summarize(
            routed_boundary_slot_exact_errors_by_anchor[anchor],
            int(routed_boundary_slot_exact_total_by_anchor[anchor]),
            int(routed_boundary_slot_exact_review_by_anchor[anchor]),
        )
        routed_boundary_slot_exact_summary.update(
            {
                "accepted": int(routed_boundary_slot_exact_accept_by_anchor[anchor]),
                "reviewed": int(routed_boundary_slot_exact_review_by_anchor[anchor]),
                "accept_rate": float(
                    routed_boundary_slot_exact_accept_by_anchor[anchor]
                    / routed_boundary_slot_exact_total_by_anchor[anchor]
                )
                if routed_boundary_slot_exact_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(
                    routed_boundary_slot_exact_review_by_anchor[anchor]
                    / routed_boundary_slot_exact_total_by_anchor[anchor]
                )
                if routed_boundary_slot_exact_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_boundary_slot_exact_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_local_gate_summary = summarize(
            routed_local_gate_errors_by_anchor[anchor],
            int(routed_local_gate_total_by_anchor[anchor]),
            int(routed_local_gate_review_by_anchor[anchor]),
        )
        routed_local_gate_summary.update(
            {
                "accepted": int(routed_local_gate_accept_by_anchor[anchor]),
                "reviewed": int(routed_local_gate_review_by_anchor[anchor]),
                "accept_rate": float(routed_local_gate_accept_by_anchor[anchor] / routed_local_gate_total_by_anchor[anchor])
                if routed_local_gate_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(routed_local_gate_review_by_anchor[anchor] / routed_local_gate_total_by_anchor[anchor])
                if routed_local_gate_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_local_gate_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        routed_parameter_policy_summary = summarize(
            routed_parameter_policy_errors_by_anchor[anchor],
            int(routed_parameter_policy_total_by_anchor[anchor]),
            int(routed_parameter_policy_review_by_anchor[anchor]),
        )
        routed_parameter_policy_summary.update(
            {
                "accepted": int(routed_parameter_policy_accept_by_anchor[anchor]),
                "reviewed": int(routed_parameter_policy_review_by_anchor[anchor]),
                "accept_rate": float(
                    routed_parameter_policy_accept_by_anchor[anchor]
                    / routed_parameter_policy_total_by_anchor[anchor]
                )
                if routed_parameter_policy_total_by_anchor[anchor]
                else 0.0,
                "review_rate": float(
                    routed_parameter_policy_review_by_anchor[anchor]
                    / routed_parameter_policy_total_by_anchor[anchor]
                )
                if routed_parameter_policy_total_by_anchor[anchor]
                else 0.0,
                "review_reasons": {
                    key.split("|", 1)[1]: int(value)
                    for key, value in sorted(routed_parameter_policy_review_reason_by_anchor.items())
                    if key.startswith(f"{anchor}|")
                },
            }
        )
        by_anchor[anchor] = {
            "model": summarize(errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_family_prior": summarize(family_prior_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_monotonic": summarize(monotonic_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_joint_lattice": summarize(joint_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_filename_slot_lattice": summarize(filename_slot_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_constrained_slot_lattice": summarize(constrained_slot_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_gated_constrained_accept": gated_summary,
            "model_routed_dependent_params": routed_dependent_summary,
            "model_routed_calibrated_params": routed_calibrated_summary,
            "model_routed_overlap_span_params": routed_overlap_span_summary,
            "model_routed_safe_overlap_params": routed_safe_overlap_summary,
            "model_routed_slot_exact_safe_params": routed_slot_exact_summary,
            "model_routed_boundary_slot_exact_params": routed_boundary_slot_exact_summary,
            "model_routed_local_hardfail_gate_params": routed_local_gate_summary,
            "model_routed_parameter_policy_params": routed_parameter_policy_summary,
            "model_offset_gap_from_preutterance": summarize(
                offset_gap_model_errors_by_anchor[anchor],
                total,
                max(0, total - len(offset_gap_model_errors_by_anchor[anchor])),
            ),
            "model_overlap_relative_from_preutterance": summarize(
                overlap_relative_gap_errors_by_anchor[anchor],
                total,
                max(0, total - len(overlap_relative_gap_errors_by_anchor[anchor])),
            ),
            "model_fixed_end_relative_from_preutterance": summarize(
                fixed_end_relative_gap_errors_by_anchor[anchor],
                total,
                max(0, total - len(fixed_end_relative_gap_errors_by_anchor[anchor])),
            ),
            "model_vowel_island_lattice": summarize(
                vowel_island_lattice_errors_by_anchor[anchor],
                total,
                max(0, total - len(vowel_island_lattice_errors_by_anchor[anchor])),
            ),
            "model_local_offset_from_island_pre": summarize(
                local_offset_from_island_errors_by_anchor[anchor],
                total,
                max(0, total - len(local_offset_from_island_errors_by_anchor[anchor])),
            ),
            "model_overlap_relative_on_island": summarize(
                overlap_relative_on_island_errors_by_anchor[anchor],
                total,
                max(0, total - len(overlap_relative_on_island_errors_by_anchor[anchor])),
            ),
            "model_vowel_island_gated_accept": island_gated_summary,
            "model_order_window": summarize(model_order_window_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "slot_order_prior": summarize(order_prior_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "raw_anchor_score": summarize(raw_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "oracle_candidates": summarize(oracle_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
        }

    worst_model_examples = sorted(worst, key=lambda item: float(item.get("error_ms", 0.0)), reverse=True)[:200]
    worst_joint_examples = sorted(worst_joint, key=lambda item: float(item.get("error_ms", 0.0)), reverse=True)[:200]
    worst_filename_slot_examples = sorted(
        worst_filename_slot,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_constrained_examples = sorted(
        worst_constrained,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_gated_constrained_examples = sorted(
        worst_gated_constrained,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_dependent_examples = sorted(
        worst_routed_dependent,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_calibrated_examples = sorted(
        worst_routed_calibrated,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_overlap_span_examples = sorted(
        worst_routed_overlap_span,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_safe_overlap_examples = sorted(
        worst_routed_safe_overlap,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_slot_exact_examples = sorted(
        worst_routed_slot_exact,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_boundary_slot_exact_examples = sorted(
        worst_routed_boundary_slot_exact,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_local_gate_examples = sorted(
        worst_routed_local_gate,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_routed_parameter_policy_examples = sorted(
        worst_routed_parameter_policy,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_vowel_island_examples = sorted(
        worst_vowel_island,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    worst_vowel_island_gated_examples = sorted(
        worst_vowel_island_gated,
        key=lambda item: float(item.get("error_ms", 0.0)),
        reverse=True,
    )[:200]
    gate_threshold_sweep_summary = _summarize_gate_threshold_sweep(
        gate_sweep_records_by_anchor,
        anchors=anchors,
        thresholds=tuple(gate_threshold_sweep),
    )

    return {
        "rows": len(rows),
        "wavs": len(_group_rows_by_wav(rows)),
        "wavs_extracted": len(cache),
        "order_window": float(order_window),
        "joint_top_per_anchor": int(joint_top_per_anchor),
        "joint_max_options_per_row": int(joint_max_options_per_row),
        "family_prior_weight": float(family_prior_weight),
        "joint_family_prior_weight": float(joint_family_prior_weight),
        "require_primary_transition_token": bool(require_primary_transition_token),
        "relative_anchor_priors_enabled": bool(relative_anchor_priors),
        "island_anchor_priors_enabled": bool(island_anchor_priors),
        "gate_policy": {
            "primary": "model_constrained_slot_lattice",
            "slot_disagreement_threshold": float(gate_slot_disagreement),
            "pred_disagreement_ms": float(gate_pred_disagreement_ms),
            "joint_disagreement_ms": float(gate_joint_disagreement_ms),
            "order_disagreement_threshold": float(gate_order_disagreement),
            "joint_score_min": float(gate_joint_score_min),
            "review_mojibake": bool(gate_review_mojibake),
            "review_duplicate_tokens": bool(gate_review_duplicate_tokens),
            "heuristic_gate_enabled": bool(use_heuristic_gate),
            "supervised_gate_threshold": float(supervised_gate_threshold),
            "supervised_gate_thresholds_by_anchor": {
                str(anchor): float(value)
                for anchor, value in sorted((supervised_gate_thresholds_by_anchor or {}).items())
            },
            "supervised_gate_enabled": bool(supervised_gate_models),
            "local_failure_gate_enabled": bool(local_failure_models),
            "local_gate_threshold": float(local_gate_threshold),
            "threshold_sweep": [float(value) for value in sorted(set(gate_threshold_sweep))],
        },
        "gate_threshold_sweep": gate_threshold_sweep_summary,
        "gate_threshold_recommendations": _recommend_gate_thresholds(gate_threshold_sweep_summary),
        "gate_training": {
            "feature_names": _gate_feature_names(),
            "sample_counts": {anchor: len(gate_feature_labels[anchor]) for anchor in anchors},
            "safe_counts": {anchor: int(sum(gate_feature_labels[anchor])) for anchor in anchors},
            "features_by_anchor": {
                anchor: np.stack(gate_feature_samples[anchor], axis=0).astype(np.float32).tolist()
                if bool(collect_gate_training) and gate_feature_samples[anchor]
                else []
                for anchor in anchors
            },
            "labels_by_anchor": {
                anchor: [int(value) for value in gate_feature_labels[anchor]]
                if bool(collect_gate_training)
                else []
                for anchor in anchors
            },
        }
        if bool(collect_gate_training)
        else {
            "feature_names": _gate_feature_names(),
            "sample_counts": {anchor: 0 for anchor in anchors},
            "safe_counts": {anchor: 0 for anchor in anchors},
        },
        "local_failure_training": {
            "feature_names": _local_failure_feature_names(),
            "sample_counts": {anchor: len(local_failure_feature_labels[anchor]) for anchor in anchors},
            "safe_counts": {anchor: int(sum(local_failure_feature_labels[anchor])) for anchor in anchors},
            "features_by_anchor": {
                anchor: np.stack(local_failure_feature_samples[anchor], axis=0).astype(np.float32).tolist()
                if bool(collect_local_failure_training) and local_failure_feature_samples[anchor]
                else []
                for anchor in anchors
            },
            "labels_by_anchor": {
                anchor: [int(value) for value in local_failure_feature_labels[anchor]]
                if bool(collect_local_failure_training)
                else []
                for anchor in anchors
            },
        }
        if bool(collect_local_failure_training)
        else {
            "feature_names": _local_failure_feature_names(),
            "sample_counts": {anchor: 0 for anchor in anchors},
            "safe_counts": {anchor: 0 for anchor in anchors},
        },
        "wav_failures": wav_failures,
        "by_anchor": by_anchor,
        "by_role_anchor_recall30": {
            key: float(by_role_anchor_hit30[key] / total) if total else 0.0
            for key, total in sorted(by_role_anchor_total.items())
        },
        "by_alias_family_anchor_recall30": {
            key: float(by_alias_family_anchor_hit30[key] / total) if total else 0.0
            for key, total in sorted(by_alias_family_anchor_total.items())
        },
        "slot_shift_summary": _summarize_slot_shifts(slot_shift_by_series_anchor),
        "worst_model_examples": worst_model_examples,
        "worst_joint_lattice_examples": worst_joint_examples,
        "worst_filename_slot_lattice_examples": worst_filename_slot_examples,
        "worst_constrained_slot_lattice_examples": worst_constrained_examples,
        "worst_gated_constrained_accept_examples": worst_gated_constrained_examples,
        "worst_routed_dependent_params_examples": worst_routed_dependent_examples,
        "worst_routed_calibrated_params_examples": worst_routed_calibrated_examples,
        "worst_routed_overlap_span_params_examples": worst_routed_overlap_span_examples,
        "worst_routed_safe_overlap_params_examples": worst_routed_safe_overlap_examples,
        "worst_routed_slot_exact_safe_params_examples": worst_routed_slot_exact_examples,
        "worst_routed_boundary_slot_exact_params_examples": worst_routed_boundary_slot_exact_examples,
        "worst_routed_local_hardfail_gate_params_examples": worst_routed_local_gate_examples,
        "worst_routed_parameter_policy_params_examples": worst_routed_parameter_policy_examples,
        "worst_vowel_island_lattice_examples": worst_vowel_island_examples,
        "worst_vowel_island_gated_accept_examples": worst_vowel_island_gated_examples,
        "failure_breakdown": _summarize_failure_breakdown(failure_breakdown_records),
        "overlap_safety_summary": _summarize_overlap_safety(overlap_safety_records),
        "worst_example_profile": {
            "model": _profile_worst_examples(worst_model_examples),
            "joint_lattice": _profile_worst_examples(worst_joint_examples),
            "filename_slot_lattice": _profile_worst_examples(worst_filename_slot_examples),
            "constrained_slot_lattice": _profile_worst_examples(worst_constrained_examples),
            "gated_constrained_accept": _profile_worst_examples(worst_gated_constrained_examples),
            "routed_dependent_params": _profile_worst_examples(worst_routed_dependent_examples),
            "routed_calibrated_params": _profile_worst_examples(worst_routed_calibrated_examples),
            "routed_overlap_span_params": _profile_worst_examples(worst_routed_overlap_span_examples),
            "routed_safe_overlap_params": _profile_worst_examples(worst_routed_safe_overlap_examples),
            "routed_slot_exact_safe_params": _profile_worst_examples(worst_routed_slot_exact_examples),
            "routed_boundary_slot_exact_params": _profile_worst_examples(worst_routed_boundary_slot_exact_examples),
            "routed_local_hardfail_gate_params": _profile_worst_examples(worst_routed_local_gate_examples),
            "routed_parameter_policy_params": _profile_worst_examples(worst_routed_parameter_policy_examples),
            "vowel_island_lattice": _profile_worst_examples(worst_vowel_island_examples),
            "vowel_island_gated_accept": _profile_worst_examples(worst_vowel_island_gated_examples),
        },
    }


def train_manual_oto_anchor_scorer(
    *,
    train_manifest: Path,
    eval_manifest: Path,
    out_dir: Path,
    anchors: tuple[str, ...] = SCORABLE_ANCHORS,
    encoder: str = "acoustic",
    max_train_rows: int | None = None,
    max_train_wavs: int | None = None,
    max_eval_rows: int | None = None,
    max_eval_wavs: int | None = None,
    sample_wavs: bool = False,
    min_score: float = 0.28,
    top_k_fallback: int = 24,
    scorer_kind: str = "binary",
    positive_ms: float = 25.0,
    negative_min_ms: float = 60.0,
    max_negatives_per_positive: int = 8,
    distance_sigma_ms: float = 35.0,
    max_distance_candidates_per_target: int = 48,
    max_iter: int = 80,
    order_window: float = 0.20,
    joint_top_per_anchor: int = 6,
    joint_max_options_per_row: int = 80,
    family_prior_weight: float = 1.0,
    joint_family_prior_weight: float = 0.0,
    require_primary_transition_token: bool = False,
    gate_slot_disagreement: float = 0.24,
    gate_pred_disagreement_ms: float = 320.0,
    gate_joint_disagreement_ms: float = 420.0,
    gate_order_disagreement: float = 0.35,
    gate_joint_score_min: float = 0.0,
    gate_review_mojibake: bool = True,
    gate_review_duplicate_tokens: bool = True,
    failure_gate: str = "heuristic",
    supervised_gate_threshold: float = 0.80,
    supervised_gate_thresholds_by_anchor: dict[str, float] | None = None,
    local_gate_threshold: float = 0.80,
    gate_threshold_sweep: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    seed: int = 20260522,
) -> dict[str, object]:
    train_rows = _load_rows(
        train_manifest,
        max_rows=max_train_rows,
        max_wavs=max_train_wavs,
        sample_wavs=bool(sample_wavs),
        seed=int(seed),
    )
    eval_rows = _load_rows(
        eval_manifest,
        max_rows=max_eval_rows,
        max_wavs=max_eval_wavs,
        sample_wavs=bool(sample_wavs),
        seed=int(seed) + 17,
    )
    scorer_kind = str(scorer_kind or "binary").strip().lower()
    if scorer_kind in {"classifier", "classification"}:
        scorer_kind = "binary"
    if scorer_kind == "pairwise":
        x_by_anchor, y_by_anchor, train_build_summary = _build_pairwise_training_arrays(
            train_rows,
            anchors=anchors,
            encoder=encoder,
            min_score=float(min_score),
            top_k_fallback=int(top_k_fallback),
            positive_ms=float(positive_ms),
            negative_min_ms=float(negative_min_ms),
            max_negatives_per_positive=int(max_negatives_per_positive),
            seed=int(seed),
        )
        models = _fit_pairwise_rankers(x_by_anchor, y_by_anchor, seed=int(seed), max_iter=int(max_iter))
    elif scorer_kind in {"distance", "graded", "regression"}:
        scorer_kind = "distance"
        x_by_anchor, y_by_anchor, train_build_summary = _build_distance_training_arrays(
            train_rows,
            anchors=anchors,
            encoder=encoder,
            min_score=float(min_score),
            top_k_fallback=int(top_k_fallback),
            distance_sigma_ms=float(distance_sigma_ms),
            max_candidates_per_target=int(max_distance_candidates_per_target),
        )
        models = _fit_distance_rankers(x_by_anchor, y_by_anchor, seed=int(seed), max_iter=int(max_iter))
    elif scorer_kind == "binary":
        x_by_anchor, y_by_anchor, train_build_summary = _build_training_arrays(
            train_rows,
            anchors=anchors,
            encoder=encoder,
            min_score=float(min_score),
            top_k_fallback=int(top_k_fallback),
            positive_ms=float(positive_ms),
            negative_min_ms=float(negative_min_ms),
            max_negatives_per_positive=int(max_negatives_per_positive),
            seed=int(seed),
        )
        models = _fit_models(x_by_anchor, y_by_anchor, seed=int(seed), max_iter=int(max_iter))
    else:
        raise ValueError(f"Unsupported scorer_kind={scorer_kind!r}; use 'binary', 'pairwise', or 'distance'.")
    relative_anchor_priors = _fit_relative_anchor_priors(train_rows)
    island_anchor_priors = _fit_island_anchor_position_priors(
        train_rows,
        anchors=anchors,
        encoder=encoder,
        min_score=float(min_score),
        top_k_fallback=int(top_k_fallback),
    )
    failure_gate = str(failure_gate or "heuristic").strip().lower()
    if failure_gate not in {"off", "heuristic", "supervised", "combined"}:
        raise ValueError("Unsupported failure_gate; use off, heuristic, supervised, or combined.")
    supervised_gate_models: dict[str, object] = {}
    gate_train_summary: dict[str, object] | None = None
    local_failure_models: dict[str, object] = {}
    local_failure_train_summary: dict[str, object] | None = None
    if failure_gate in {"supervised", "combined"}:
        gate_train_summary = evaluate_models(
            train_rows,
            models=models,
            anchors=anchors,
            encoder=encoder,
            min_score=float(min_score),
            top_k_fallback=int(top_k_fallback),
            order_window=float(order_window),
            joint_top_per_anchor=int(joint_top_per_anchor),
            joint_max_options_per_row=int(joint_max_options_per_row),
            family_prior_weight=float(family_prior_weight),
            joint_family_prior_weight=float(joint_family_prior_weight),
            require_primary_transition_token=bool(require_primary_transition_token),
            gate_slot_disagreement=float(gate_slot_disagreement),
            gate_pred_disagreement_ms=float(gate_pred_disagreement_ms),
            gate_joint_disagreement_ms=float(gate_joint_disagreement_ms),
            gate_order_disagreement=float(gate_order_disagreement),
            gate_joint_score_min=float(gate_joint_score_min),
            gate_review_mojibake=bool(gate_review_mojibake),
            gate_review_duplicate_tokens=bool(gate_review_duplicate_tokens),
            use_heuristic_gate=False,
            gate_threshold_sweep=tuple(gate_threshold_sweep),
            relative_anchor_priors=relative_anchor_priors,
            island_anchor_priors=island_anchor_priors,
            collect_gate_training=True,
        )
        supervised_gate_models = _fit_failure_gate_models(
            gate_train_summary.get("gate_training", {}) if isinstance(gate_train_summary, dict) else {},
            seed=int(seed) + 31,
        )
        local_failure_train_summary = evaluate_models(
            train_rows,
            models=models,
            anchors=anchors,
            encoder=encoder,
            min_score=float(min_score),
            top_k_fallback=int(top_k_fallback),
            order_window=float(order_window),
            joint_top_per_anchor=int(joint_top_per_anchor),
            joint_max_options_per_row=int(joint_max_options_per_row),
            family_prior_weight=float(family_prior_weight),
            joint_family_prior_weight=float(joint_family_prior_weight),
            require_primary_transition_token=bool(require_primary_transition_token),
            gate_slot_disagreement=float(gate_slot_disagreement),
            gate_pred_disagreement_ms=float(gate_pred_disagreement_ms),
            gate_joint_disagreement_ms=float(gate_joint_disagreement_ms),
            gate_order_disagreement=float(gate_order_disagreement),
            gate_joint_score_min=float(gate_joint_score_min),
            gate_review_mojibake=bool(gate_review_mojibake),
            gate_review_duplicate_tokens=bool(gate_review_duplicate_tokens),
            use_heuristic_gate=failure_gate == "combined",
            supervised_gate_models=supervised_gate_models,
            supervised_gate_threshold=float(supervised_gate_threshold),
            supervised_gate_thresholds_by_anchor=supervised_gate_thresholds_by_anchor,
            local_gate_threshold=float(local_gate_threshold),
            gate_threshold_sweep=tuple(gate_threshold_sweep),
            relative_anchor_priors=relative_anchor_priors,
            island_anchor_priors=island_anchor_priors,
            collect_local_failure_training=True,
        )
        local_failure_models = _fit_local_failure_models(
            local_failure_train_summary.get("local_failure_training", {})
            if isinstance(local_failure_train_summary, dict)
            else {},
            seed=int(seed) + 47,
        )
    eval_summary = evaluate_models(
        eval_rows,
        models=models,
        anchors=anchors,
        encoder=encoder,
        min_score=float(min_score),
        top_k_fallback=int(top_k_fallback),
        order_window=float(order_window),
        joint_top_per_anchor=int(joint_top_per_anchor),
        joint_max_options_per_row=int(joint_max_options_per_row),
        family_prior_weight=float(family_prior_weight),
        joint_family_prior_weight=float(joint_family_prior_weight),
        require_primary_transition_token=bool(require_primary_transition_token),
        gate_slot_disagreement=float(gate_slot_disagreement),
        gate_pred_disagreement_ms=float(gate_pred_disagreement_ms),
        gate_joint_disagreement_ms=float(gate_joint_disagreement_ms),
        gate_order_disagreement=float(gate_order_disagreement),
        gate_joint_score_min=float(gate_joint_score_min),
        gate_review_mojibake=bool(gate_review_mojibake),
        gate_review_duplicate_tokens=bool(gate_review_duplicate_tokens),
        use_heuristic_gate=failure_gate in {"heuristic", "combined"},
        supervised_gate_models=supervised_gate_models,
        supervised_gate_threshold=float(supervised_gate_threshold),
        supervised_gate_thresholds_by_anchor=supervised_gate_thresholds_by_anchor,
        local_failure_models=local_failure_models,
        local_gate_threshold=float(local_gate_threshold),
        gate_threshold_sweep=tuple(gate_threshold_sweep),
        relative_anchor_priors=relative_anchor_priors,
        island_anchor_priors=island_anchor_priors,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "manual_oto_anchor_scorer.pkl"
    payload = {
        "schema_version": "manual_oto_anchor_scorer_v1",
        "anchors": anchors,
        "feature_names": manual_oto_candidate_feature_names(),
        "models": models,
        "failure_gate_models": supervised_gate_models,
        "failure_gate_feature_names": _gate_feature_names(),
        "local_failure_models": local_failure_models,
        "local_failure_feature_names": _local_failure_feature_names(),
        "relative_anchor_gap_priors": relative_anchor_priors,
        "island_anchor_position_priors": island_anchor_priors,
        "encoder": encoder,
        "scorer_kind": scorer_kind,
        "min_score": float(min_score),
        "top_k_fallback": int(top_k_fallback),
        "positive_ms": float(positive_ms),
        "negative_min_ms": float(negative_min_ms),
        "distance_sigma_ms": float(distance_sigma_ms),
        "max_distance_candidates_per_target": int(max_distance_candidates_per_target),
        "order_window": float(order_window),
        "joint_top_per_anchor": int(joint_top_per_anchor),
        "joint_max_options_per_row": int(joint_max_options_per_row),
        "family_prior_weight": float(family_prior_weight),
        "joint_family_prior_weight": float(joint_family_prior_weight),
        "require_primary_transition_token": bool(require_primary_transition_token),
        "gate_slot_disagreement": float(gate_slot_disagreement),
        "gate_pred_disagreement_ms": float(gate_pred_disagreement_ms),
        "gate_joint_disagreement_ms": float(gate_joint_disagreement_ms),
        "gate_order_disagreement": float(gate_order_disagreement),
        "gate_joint_score_min": float(gate_joint_score_min),
        "gate_review_mojibake": bool(gate_review_mojibake),
        "gate_review_duplicate_tokens": bool(gate_review_duplicate_tokens),
        "failure_gate": failure_gate,
        "supervised_gate_threshold": float(supervised_gate_threshold),
        "local_gate_threshold": float(local_gate_threshold),
        "supervised_gate_thresholds_by_anchor": {
            str(anchor): float(value)
            for anchor, value in sorted((supervised_gate_thresholds_by_anchor or {}).items())
        },
        "gate_threshold_sweep": [float(value) for value in sorted(set(gate_threshold_sweep))],
        "relative_anchor_priors": _relative_prior_bucket_summary(relative_anchor_priors),
        "island_anchor_priors": {
            "schema_version": str(island_anchor_priors.get("schema_version", "")),
            "bucket_count": len(dict(island_anchor_priors.get("buckets", {}))),
            "stats": dict(island_anchor_priors.get("stats", {}))
            if isinstance(island_anchor_priors.get("stats", {}), dict)
            else {},
        },
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    summary = {
        "schema_version": "manual_oto_anchor_scorer_train_v1",
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "anchors": list(anchors),
        "feature_count": len(manual_oto_candidate_feature_names()),
        "scorer_kind": scorer_kind,
        "models_trained": sorted(models),
        "failure_gate": failure_gate,
        "failure_gate_models_trained": sorted(supervised_gate_models),
        "local_failure_models_trained": sorted(local_failure_models),
        "supervised_gate_threshold": float(supervised_gate_threshold),
        "local_gate_threshold": float(local_gate_threshold),
        "supervised_gate_thresholds_by_anchor": {
            str(anchor): float(value)
            for anchor, value in sorted((supervised_gate_thresholds_by_anchor or {}).items())
        },
        "failure_gate_train": {
            "sample_counts": (gate_train_summary or {}).get("gate_training", {}).get("sample_counts", {})
            if isinstance((gate_train_summary or {}).get("gate_training", {}), dict)
            else {},
            "safe_counts": (gate_train_summary or {}).get("gate_training", {}).get("safe_counts", {})
            if isinstance((gate_train_summary or {}).get("gate_training", {}), dict)
            else {},
        },
        "local_failure_train": {
            "sample_counts": (local_failure_train_summary or {}).get("local_failure_training", {}).get("sample_counts", {})
            if isinstance((local_failure_train_summary or {}).get("local_failure_training", {}), dict)
            else {},
            "safe_counts": (local_failure_train_summary or {}).get("local_failure_training", {}).get("safe_counts", {})
            if isinstance((local_failure_train_summary or {}).get("local_failure_training", {}), dict)
            else {},
        },
        "sample_wavs": bool(sample_wavs),
        "relative_anchor_priors": _relative_prior_bucket_summary(relative_anchor_priors),
        "island_anchor_priors": {
            "schema_version": str(island_anchor_priors.get("schema_version", "")),
            "bucket_count": len(dict(island_anchor_priors.get("buckets", {}))),
            "stats": dict(island_anchor_priors.get("stats", {}))
            if isinstance(island_anchor_priors.get("stats", {}), dict)
            else {},
        },
        "train_build": train_build_summary,
        "eval": eval_summary,
    }
    summary_path = out_dir / "manual_oto_anchor_scorer_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight candidate scorer from manual oto.ini anchors.")
    parser.add_argument("--train-manifest", default=str(Path(DEFAULT_SPLIT_DIR) / "manual_oto_anchor_train.jsonl"))
    parser.add_argument("--eval-manifest", default=str(Path(DEFAULT_SPLIT_DIR) / "manual_oto_anchor_val.jsonl"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--anchors", default=",".join(SCORABLE_ANCHORS))
    parser.add_argument("--encoder", default="acoustic")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-train-wavs", type=int)
    parser.add_argument("--max-eval-rows", type=int)
    parser.add_argument("--max-eval-wavs", type=int)
    parser.add_argument("--sample-wavs", action="store_true", help="Randomly sample wav groups when max wav limits are set.")
    parser.add_argument("--min-score", type=float, default=0.28)
    parser.add_argument("--top-k-fallback", type=int, default=24)
    parser.add_argument("--scorer-kind", choices=("binary", "classifier", "pairwise", "distance", "graded", "regression"), default="binary")
    parser.add_argument("--positive-ms", type=float, default=25.0)
    parser.add_argument("--negative-min-ms", type=float, default=60.0)
    parser.add_argument("--max-negatives-per-positive", type=int, default=8)
    parser.add_argument("--distance-sigma-ms", type=float, default=35.0)
    parser.add_argument("--max-distance-candidates-per-target", type=int, default=48)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--order-window", type=float, default=0.20)
    parser.add_argument("--joint-top-per-anchor", type=int, default=6)
    parser.add_argument("--joint-max-options-per-row", type=int, default=80)
    parser.add_argument("--family-prior-weight", type=float, default=1.0)
    parser.add_argument("--joint-family-prior-weight", type=float, default=0.0)
    parser.add_argument("--gate-slot-disagreement", type=float, default=0.24)
    parser.add_argument("--gate-pred-disagreement-ms", type=float, default=320.0)
    parser.add_argument("--gate-joint-disagreement-ms", type=float, default=420.0)
    parser.add_argument("--gate-order-disagreement", type=float, default=0.35)
    parser.add_argument("--gate-joint-score-min", type=float, default=0.0)
    parser.add_argument("--no-gate-review-mojibake", action="store_true")
    parser.add_argument("--no-gate-review-duplicate-tokens", action="store_true")
    parser.add_argument("--failure-gate", choices=("off", "heuristic", "supervised", "combined"), default="heuristic")
    parser.add_argument("--supervised-gate-threshold", type=float, default=0.80)
    parser.add_argument(
        "--local-gate-threshold",
        type=float,
        default=0.80,
        help="Safe probability threshold for routed local hard-fail gate.",
    )
    parser.add_argument(
        "--supervised-gate-threshold-by-anchor",
        default="",
        help="Comma-separated anchor=value overrides, for example preutterance=0.80,fixed_end=0.85.",
    )
    parser.add_argument(
        "--gate-threshold-sweep",
        default="0.50,0.60,0.70,0.80,0.85,0.90,0.95",
        help="Comma-separated supervised gate thresholds to summarize without rerunning eval.",
    )
    parser.add_argument(
        "--require-primary-transition-token",
        action="store_true",
        help="Experimental: fallback to slot_index when a VC/VCV/VV alias primary token is absent from filename tokens.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()
    summary = train_manual_oto_anchor_scorer(
        train_manifest=Path(args.train_manifest),
        eval_manifest=Path(args.eval_manifest),
        out_dir=Path(args.out_dir),
        anchors=_anchors_arg(str(args.anchors or "")),
        encoder=str(args.encoder or "acoustic"),
        max_train_rows=args.max_train_rows,
        max_train_wavs=args.max_train_wavs,
        max_eval_rows=args.max_eval_rows,
        max_eval_wavs=args.max_eval_wavs,
        sample_wavs=bool(args.sample_wavs),
        min_score=float(args.min_score),
        top_k_fallback=int(args.top_k_fallback),
        scorer_kind=str(args.scorer_kind),
        positive_ms=float(args.positive_ms),
        negative_min_ms=float(args.negative_min_ms),
        max_negatives_per_positive=int(args.max_negatives_per_positive),
        distance_sigma_ms=float(args.distance_sigma_ms),
        max_distance_candidates_per_target=int(args.max_distance_candidates_per_target),
        max_iter=int(args.max_iter),
        order_window=float(args.order_window),
        joint_top_per_anchor=int(args.joint_top_per_anchor),
        joint_max_options_per_row=int(args.joint_max_options_per_row),
        family_prior_weight=float(args.family_prior_weight),
        joint_family_prior_weight=float(args.joint_family_prior_weight),
        require_primary_transition_token=bool(args.require_primary_transition_token),
        gate_slot_disagreement=float(args.gate_slot_disagreement),
        gate_pred_disagreement_ms=float(args.gate_pred_disagreement_ms),
        gate_joint_disagreement_ms=float(args.gate_joint_disagreement_ms),
        gate_order_disagreement=float(args.gate_order_disagreement),
        gate_joint_score_min=float(args.gate_joint_score_min),
        gate_review_mojibake=not bool(args.no_gate_review_mojibake),
        gate_review_duplicate_tokens=not bool(args.no_gate_review_duplicate_tokens),
        failure_gate=str(args.failure_gate),
        supervised_gate_threshold=float(args.supervised_gate_threshold),
        local_gate_threshold=float(args.local_gate_threshold),
        supervised_gate_thresholds_by_anchor=_thresholds_by_anchor_arg(
            str(args.supervised_gate_threshold_by_anchor or "")
        ),
        gate_threshold_sweep=_thresholds_arg(str(args.gate_threshold_sweep or "")),
        seed=int(args.seed),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
