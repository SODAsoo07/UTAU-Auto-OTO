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
    return [
        f"{language}|{format_type}|{role}|{family}",
        f"{language}|{format_type}|{role}|*",
        f"{format_type}|{role}|*",
        f"{role}|*",
        "*",
    ]


def _fit_relative_anchor_priors(rows: list[dict[str, object]]) -> dict[str, object]:
    kinds = ("offset_from_preutterance", "overlap_from_preutterance", "fixed_end_from_preutterance")
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

    buckets: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for kind, by_key in samples.items():
        buckets[kind] = {key: float(median(values)) for key, values in by_key.items() if values}
        counts[kind] = {key: int(len(values)) for key, values in by_key.items() if values}
    return {
        "schema_version": "relative_anchor_gap_priors_v1",
        "buckets": buckets,
        "counts": counts,
    }


def _relative_anchor_gap_ms(priors: dict[str, object] | None, kind: str, row: dict[str, object]) -> float | None:
    if not isinstance(priors, dict):
        return None
    buckets = priors.get("buckets")
    if not isinstance(buckets, dict):
        return None
    by_kind = buckets.get(kind)
    if not isinstance(by_kind, dict):
        return None
    for key in _relative_gap_bucket_keys(row):
        value = by_kind.get(key)
        if value is None:
            continue
        try:
            gap = float(value)
        except Exception:
            continue
        if np.isfinite(gap):
            return gap
    return None


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
    if manifest_count >= 2:
        return manifest_count
    tokens = _filename_tokens_for_row(wav_rows[0])
    if len(tokens) >= 2:
        return len(tokens)
    return manifest_count


def _row_vowel_island_slot_index(row: dict[str, object], slot_count: int) -> int:
    count = max(1, int(slot_count))
    try:
        row_count = max(1, int(row.get("slot_count", count) or count))
        row_index = int(row.get("slot_index", 0) or 0)
        if 0 <= row_index < row_count:
            pos = (float(row_index) + 0.5) / max(1.0, float(row_count))
            idx = int(round(pos * float(count) - 0.5))
            return max(0, min(count - 1, idx))
    except Exception:
        pass
    tokens = _filename_tokens_for_row(row)
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
    gate_threshold_sweep: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
    relative_anchor_priors: dict[str, object] | None = None,
    island_anchor_priors: dict[str, object] | None = None,
    collect_gate_training: bool = False,
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
    worst_vowel_island: list[dict[str, object]] = []
    worst_vowel_island_gated: list[dict[str, object]] = []
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

        if relative_anchor_priors:
            for row in wav_rows:
                pre_prediction = constrained_predictions.get((id(row), "preutterance"))
                if pre_prediction is None:
                    continue
                pre_pred_ms = float(pre_prediction.get("pred_ms", 0.0) or 0.0)
                duration_ms = max(1.0, float(row.get("duration_ms", tracks.duration_ms) or tracks.duration_ms or 1.0))
                if "offset" in anchors:
                    target = _target_ms(row, "offset")
                    gap = _relative_anchor_gap_ms(relative_anchor_priors, "offset_from_preutterance", row)
                    if target is not None and gap is not None:
                        pred_ms = min(duration_ms, max(0.0, pre_pred_ms - float(gap)))
                        offset_gap_model_errors_by_anchor["offset"].append(abs(pred_ms - float(target)))
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_offset_gap_from_preutterance",
                            "offset",
                            row,
                            pred_ms=pred_ms,
                            duration_ms=duration_ms,
                        )
                if "overlap" in anchors:
                    target = _target_ms(row, "overlap")
                    gap = _relative_anchor_gap_ms(relative_anchor_priors, "overlap_from_preutterance", row)
                    if target is not None and gap is not None:
                        pred_ms = min(duration_ms, max(0.0, pre_pred_ms - float(gap)))
                        overlap_relative_gap_errors_by_anchor["overlap"].append(abs(pred_ms - float(target)))
                        _record_slot_shift(
                            slot_shift_by_series_anchor,
                            "model_overlap_relative_from_preutterance",
                            "overlap",
                            row,
                            pred_ms=pred_ms,
                            duration_ms=duration_ms,
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
                                if safe_prob >= 0.90:
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
            row_safe, row_safety_warnings = _island_row_safe(assignment, island, predictions)
            island_context = _island_prediction_context(
                island_decode,
                assignment,
                island,
                slot_index=slot_idx,
                slot_count=slot_count_for_islands,
                safety_warnings=row_safety_warnings,
            )
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
                    if safe_prob < float(supervised_gate_threshold):
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
        by_anchor[anchor] = {
            "model": summarize(errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_family_prior": summarize(family_prior_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_monotonic": summarize(monotonic_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_joint_lattice": summarize(joint_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_filename_slot_lattice": summarize(filename_slot_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_constrained_slot_lattice": summarize(constrained_slot_lattice_errors_by_anchor[anchor], total, int(missing_by_anchor[anchor])),
            "model_gated_constrained_accept": gated_summary,
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
            "supervised_gate_enabled": bool(supervised_gate_models),
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
        "worst_vowel_island_lattice_examples": worst_vowel_island_examples,
        "worst_vowel_island_gated_accept_examples": worst_vowel_island_gated_examples,
        "worst_example_profile": {
            "model": _profile_worst_examples(worst_model_examples),
            "joint_lattice": _profile_worst_examples(worst_joint_examples),
            "filename_slot_lattice": _profile_worst_examples(worst_filename_slot_examples),
            "constrained_slot_lattice": _profile_worst_examples(worst_constrained_examples),
            "gated_constrained_accept": _profile_worst_examples(worst_gated_constrained_examples),
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
        "gate_threshold_sweep": [float(value) for value in sorted(set(gate_threshold_sweep))],
        "relative_anchor_priors": {
            "schema_version": str(relative_anchor_priors.get("schema_version", "")),
            "bucket_counts": {
                kind: len(values) if isinstance(values, dict) else 0
                for kind, values in dict(relative_anchor_priors.get("buckets", {})).items()
            },
        },
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
        "failure_gate_train": {
            "sample_counts": (gate_train_summary or {}).get("gate_training", {}).get("sample_counts", {})
            if isinstance((gate_train_summary or {}).get("gate_training", {}), dict)
            else {},
            "safe_counts": (gate_train_summary or {}).get("gate_training", {}).get("safe_counts", {})
            if isinstance((gate_train_summary or {}).get("gate_training", {}), dict)
            else {},
        },
        "sample_wavs": bool(sample_wavs),
        "relative_anchor_priors": {
            "schema_version": str(relative_anchor_priors.get("schema_version", "")),
            "bucket_counts": {
                kind: len(values) if isinstance(values, dict) else 0
                for kind, values in dict(relative_anchor_priors.get("buckets", {})).items()
            },
        },
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
        gate_threshold_sweep=_thresholds_arg(str(args.gate_threshold_sweep or "")),
        seed=int(args.seed),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
