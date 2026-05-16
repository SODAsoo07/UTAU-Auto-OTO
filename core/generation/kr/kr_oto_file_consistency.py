"""
Korean OTO file-level consistency postprocess.

This module performs a lightweight second pass after row-level generation:
1. clamp excessive overlap/gap between adjacent rows in each wav group
2. smooth abrupt parameter jumps within the same alias type/consonant group
3. re-validate every row to keep hard ordering constraints
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from core.kr_oto_rules import classify_alias
from core.oto_file_utils import read_text_with_fallback


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _blend(a: float, b: float, w: float) -> float:
    w2 = _clamp(w, 0.0, 1.0)
    return (1.0 - w2) * float(a) + w2 * float(b)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_oto_line(line: str) -> Optional[Dict[str, object]]:
    line = line.strip()
    if not line or "=" not in line or "," not in line:
        return None
    left, right = line.split("=", 1)
    parts = [p.strip() for p in right.split(",")]
    if len(parts) < 6:
        return None
    try:
        return {
            "wav": left.strip(),
            "alias": parts[0],
            "offset": float(parts[1]),
            "cons": float(parts[2]),
            "cutoff": float(parts[3]),
            "pre": float(parts[4]),
            "ovl": float(parts[5]),
        }
    except (ValueError, IndexError):
        return None


def _format_oto_line(row: Dict[str, object]) -> str:
    return (
        f"{row['wav']}={row['alias']},"
        f"{row['offset']:.2f},{row['cons']:.2f},{row['cutoff']:.2f},"
        f"{row['pre']:.2f},{row['ovl']:.2f}"
    )


def _classify_cached(
    alias: str,
    cache: Dict[str, str],
    custom_map: Optional[Dict[str, str]] = None,
) -> str:
    key = alias.strip()
    if key not in cache:
        cache[key] = classify_alias(key, custom_map)
    return cache[key]


_BRIDGE_TYPES = {"vc", "vv"}
_CV_TYPES = {"cv", "cv_head", "vcv", "mono"}

_OVERLAP_LIMITS = {
    # (min_overlap, max_overlap)
    # default target: gap <= 80ms and overlap <= 50ms
    # bridge transition (VC/VV -> CV) relaxes overlap upper bound to 65ms
    "vc_to_cv": (-80.0, 65.0),
    "vv_to_cv": (-80.0, 65.0),
    "cv_to_vc": (-80.0, 50.0),
    "cv_to_vv": (-80.0, 50.0),
    "default": (-80.0, 50.0),
}

def _max_offset_adj_ms() -> float:
    return _env_float("UTOA_KR_CONTINUITY_MAX_OFFSET_ADJ", 180.0)

def _cvn_correction_enabled() -> bool:
    return str(os.environ.get("UTOA_CVN_CORRECTION_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _vc_neighbor_enabled() -> bool:
    if not _cvn_correction_enabled():
        return False
    return str(os.environ.get("UTOA_KR_VC_NEIGHBOR_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

def _vc_neighbor_blend() -> float:
    return _env_float("UTOA_KR_VC_NEIGHBOR_BLEND", 0.35)

def _vc_neighbor_max_shift() -> float:
    return _env_float("UTOA_KR_VC_NEIGHBOR_MAX_SHIFT", 45.0)

def _vc_neighbor_lead_ms() -> float:
    return _env_float("UTOA_KR_VC_NEIGHBOR_LEAD_MS", 6.0)

def _vc_neighbor_tail_ms() -> float:
    return _env_float("UTOA_KR_VC_NEIGHBOR_TAIL_MS", 8.0)

def _vc_neighbor_min_len() -> float:
    return _env_float("UTOA_KR_VC_NEIGHBOR_MIN_LEN", 35.0)


def _get_overlap_key(prev_type: str, next_type: str) -> str:
    if prev_type in _BRIDGE_TYPES and next_type in _CV_TYPES:
        return f"{prev_type}_to_cv"
    if prev_type in _CV_TYPES and next_type in _BRIDGE_TYPES:
        return f"cv_to_{next_type}"
    return "default"


def enforce_adjacent_continuity(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    """
    Enforce continuity between adjacent rows in a wav group.

    Priority is fixed:
    1) adjust previous row cutoff first
    2) adjust next row offset only when cutoff-only correction is insufficient
    """
    changed = 0
    min_cut_tail = 8.0

    def _row_state(row: Dict[str, object]) -> Tuple[float, float, float, float, float]:
        return (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )

    def _validate_row(
        row: Dict[str, object],
        alias_type: str,
        before_state: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> bool:
        if before_state is None:
            before_state = _row_state(row)
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=alias_type,
        )
        after_state = _row_state(row)
        return any(abs(a - b) > 1e-6 for a, b in zip(before_state, after_state))

    for i in range(len(rows) - 1):
        max_offset_adj = _max_offset_adj_ms()
        prev_row = rows[i]
        next_row = rows[i + 1]
        # TICKET-004: Skip pair if either row is locked by TextGrid-anchored timing.
        if prev_row.get("consistency_locked") or next_row.get("consistency_locked"):
            continue
        prev_type = row_types[i]
        next_type = row_types[i + 1]

        prev_end = float(prev_row["offset"]) + abs(float(prev_row["cutoff"]))
        next_offset = float(next_row["offset"])
        overlap_ms = prev_end - next_offset

        min_overlap, max_overlap = _OVERLAP_LIMITS.get(
            _get_overlap_key(prev_type, next_type),
            _OVERLAP_LIMITS["default"],
        )
        if min_overlap <= overlap_ms <= max_overlap:
            continue

        if overlap_ms > max_overlap:
            prev_offset = float(prev_row["offset"])
            curr_cut_abs = abs(float(prev_row["cutoff"]))
            target_cut_abs = next_offset + float(max_overlap) - prev_offset
            min_cut_abs = float(prev_row["cons"]) + min_cut_tail
            new_cut_abs = max(min_cut_abs, min(curr_cut_abs, target_cut_abs))
            if new_cut_abs < curr_cut_abs - 1e-6:
                before_state = _row_state(prev_row)
                prev_row["cutoff"] = -new_cut_abs
                if _validate_row(prev_row, prev_type, before_state):
                    changed += 1

            prev_end = float(prev_row["offset"]) + abs(float(prev_row["cutoff"]))
            overlap_ms = prev_end - float(next_row["offset"])
            if overlap_ms > max_overlap + 1e-6:
                off_adj = float(overlap_ms - max_overlap)
                if off_adj > max_offset_adj:
                    continue
                if off_adj > 1e-6:
                    before_state = _row_state(next_row)
                    next_row["offset"] = max(0.0, float(next_row["offset"]) + off_adj)
                    if _validate_row(next_row, next_type, before_state):
                        changed += 1

        elif overlap_ms < min_overlap:
            # TICKET-004: Check consistency_locked before extending cutoff.
            if prev_row.get("consistency_locked"):
                continue
            prev_offset = float(prev_row["offset"])
            curr_cut_abs = abs(float(prev_row["cutoff"]))
            target_cut_abs = next_offset + float(min_overlap) - prev_offset
            # TICKET-004: max_extend now defaults to 80ms (halved from 160ms)
            # to reduce risk of overextending into intentional inter-syllable gaps.
            max_extend = _env_float("UTOA_KR_CONTINUITY_MAX_EXTEND", 80.0)
            new_cut_abs = min(max(curr_cut_abs, target_cut_abs), curr_cut_abs + max_extend)
            if new_cut_abs > curr_cut_abs + 1e-6:
                before_state = _row_state(prev_row)
                prev_row["cutoff"] = -max(new_cut_abs, float(prev_row["cons"]) + min_cut_tail)
                if _validate_row(prev_row, prev_type, before_state):
                    changed += 1

            prev_end = float(prev_row["offset"]) + abs(float(prev_row["cutoff"]))
            overlap_ms = prev_end - float(next_row["offset"])
            if overlap_ms < min_overlap - 1e-6:
                off_adj = float(min_overlap - overlap_ms)
                if off_adj > max_offset_adj:
                    continue
                if off_adj > 1e-6:
                    before_state = _row_state(next_row)
                    next_row["offset"] = max(0.0, float(next_row["offset"]) - off_adj)
                    if _validate_row(next_row, next_type, before_state):
                        changed += 1

    return changed


def adjust_vc_neighbor_alignment(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    """
    Soft-align VC rows using immediate neighbor timing.
    - offset is nudged toward previous row cutoff (end)
    - cutoff is nudged toward next row offset (start)
    """
    if not _vc_neighbor_enabled():
        return 0

    changed = 0
    blend = _vc_neighbor_blend()
    max_shift = _vc_neighbor_max_shift()
    lead_ms = _vc_neighbor_lead_ms()
    tail_ms = _vc_neighbor_tail_ms()
    min_len = _vc_neighbor_min_len()

    def _row_state(row: Dict[str, object]) -> Tuple[float, float, float, float, float]:
        return (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )

    def _validate_row(
        row: Dict[str, object],
        alias_type: str,
        before_state: Tuple[float, float, float, float, float],
    ) -> bool:
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=alias_type,
        )
        after_state = _row_state(row)
        return any(abs(a - b) > 1e-6 for a, b in zip(before_state, after_state))

    for i in range(1, len(rows) - 1):
        if row_types[i] != "vc":
            continue
        prev_type = row_types[i - 1]
        next_type = row_types[i + 1]
        if prev_type == "br" or next_type == "br":
            continue

        prev_row = rows[i - 1]
        row = rows[i]
        next_row = rows[i + 1]

        prev_end = float(prev_row["offset"]) + abs(float(prev_row["cutoff"]))
        next_offset = float(next_row["offset"])
        if next_offset <= 0.0 or prev_end <= 0.0:
            continue

        before_state = _row_state(row)

        # Offset: pull toward previous cutoff end (slightly earlier)
        target_off = max(0.0, prev_end - lead_ms)
        delta_off = target_off - float(row["offset"])
        if abs(delta_off) > 1e-6:
            delta_off = _clamp(delta_off, -max_shift, max_shift)
            row["offset"] = max(0.0, float(row["offset"]) + delta_off * blend)

        # Cutoff: pull toward next offset (slightly earlier than next onset)
        next_limit = next_offset - tail_ms
        if next_limit > float(row["offset"]) + min_len:
            curr_cut_abs = abs(float(row["cutoff"]))
            target_cut_abs = max(float(row["cons"]) + 8.0, next_limit - float(row["offset"]))
            delta_cut = target_cut_abs - curr_cut_abs
            if abs(delta_cut) > 1e-6:
                delta_cut = _clamp(delta_cut, -max_shift, max_shift)
                new_cut_abs = max(float(row["cons"]) + 8.0, curr_cut_abs + delta_cut * blend)
                row["cutoff"] = -new_cut_abs

        if _validate_row(row, "vc", before_state):
            changed += 1

    return changed

_SMOOTHING_THRESHOLD_RATIO = 0.30
_SMOOTHING_BLEND_WEIGHT = 0.25
_CV_LIKE_TYPES = {"cv", "cv_head", "vcv", "mono"}

_CONSONANT_GROUPS = {
    "plosive": {"g", "k", "kk", "gg", "d", "t", "tt", "dd", "b", "p", "bb", "pp"},
    "sibilant": {"s", "ss", "sh", "h", "j", "jj", "ch"},
    "sonorant": {"m", "n", "ng", "r", "l", "y", "w"},
}


def _get_consonant_group(alias: str) -> str:
    import re

    clean = re.sub(r"[^a-z]", "", alias.lower().split()[0] if " " in alias else alias.lower())
    if not clean:
        return "other"
    for length in range(min(3, len(clean)), 0, -1):
        prefix = clean[:length]
        for group_name, members in _CONSONANT_GROUPS.items():
            if prefix in members:
                return group_name
    return "other"


_FORTIS_PLOSIVES = {"kk", "tt", "pp", "gg", "dd", "bb"}


def _smoothing_threshold_for_row(row_type: str, alias: str = "") -> float:
    """Return the smoothing jump threshold for the given row type.

    TICKET-004: Fortis/tense plosive CV rows use a higher threshold (0.65)
    so that legitimate sharp onset transitions are not blended away.
    """
    row_norm = str(row_type or "").strip().lower()
    if row_norm in _CV_LIKE_TYPES:
        # TICKET-004: Raise threshold for fortis plosives to preserve sharp onset.
        if alias:
            import re as _re
            clean = _re.sub(r"[^a-z]", "", str(alias).lower().split()[0] if " " in alias else str(alias).lower())
            for length in range(min(3, len(clean)), 0, -1):
                if clean[:length] in _FORTIS_PLOSIVES:
                    return _env_float("UTOA_KR_SMOOTHING_THRESHOLD_FORTIS", 0.65)
        return _env_float("UTOA_KR_SMOOTHING_THRESHOLD_CVLIKE", 0.45)
    if row_norm == "vc":
        return _env_float("UTOA_KR_SMOOTHING_THRESHOLD_VC", 0.30)
    return _env_float("UTOA_KR_SMOOTHING_THRESHOLD_DEFAULT", _SMOOTHING_THRESHOLD_RATIO)


def _smoothing_blend_for_row(row_type: str) -> float:
    row_norm = str(row_type or "").strip().lower()
    if row_norm in _CV_LIKE_TYPES:
        return _env_float("UTOA_KR_SMOOTHING_BLEND_CVLIKE", 0.14)
    if row_norm == "vc":
        return _env_float("UTOA_KR_SMOOTHING_BLEND_VC", 0.22)
    return _env_float("UTOA_KR_SMOOTHING_BLEND_DEFAULT", _SMOOTHING_BLEND_WEIGHT)


def smooth_abrupt_changes(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    changed = 0
    params_to_smooth = ["pre", "cons", "ovl", "cutoff_abs"]

    for i in range(1, len(rows)):
        # TICKET-004: Skip rows that are locked by TextGrid-anchored timing.
        if rows[i].get("consistency_locked"):
            continue
        row_type = str(row_types[i] or "").strip().lower()
        if row_type != str(row_types[i - 1] or "").strip().lower():
            continue
        if _get_consonant_group(rows[i]["alias"]) != _get_consonant_group(rows[i - 1]["alias"]):
            continue

        # TICKET-004: Pass alias to get fortis-aware threshold.
        threshold_ratio = _smoothing_threshold_for_row(row_type, alias=str(rows[i].get("alias", "")))
        blend_weight = _smoothing_blend_for_row(row_type)
        if row_type in _CV_LIKE_TYPES:
            params = ("pre", "ovl")
        else:
            params = tuple(params_to_smooth)

        row_changed = False
        for param in params:
            if param == "cutoff_abs":
                prev_val = abs(float(rows[i - 1]["cutoff"]))
                curr_val = abs(float(rows[i]["cutoff"]))
            else:
                prev_val = float(rows[i - 1][param])
                curr_val = float(rows[i][param])

            if prev_val == 0 and curr_val == 0:
                continue

            ref = max(abs(prev_val), abs(curr_val), 1.0)
            change_ratio = abs(curr_val - prev_val) / ref
            if change_ratio <= threshold_ratio:
                continue

            smoothed = _blend(curr_val, prev_val, blend_weight)
            if abs(smoothed - curr_val) > 0.5:
                if param == "cutoff_abs":
                    rows[i]["cutoff"] = -smoothed
                else:
                    rows[i][param] = smoothed
                row_changed = True

        if row_changed:
            rows[i]["cutoff"] = -max(abs(float(rows[i]["cutoff"])), float(rows[i]["cons"]) + 8.0)
            rows[i]["offset"], rows[i]["cons"], rows[i]["cutoff"], rows[i]["pre"], rows[i]["ovl"] = (
                validate_fn(
                    rows[i]["offset"],
                    rows[i]["cons"],
                    rows[i]["cutoff"],
                    rows[i]["pre"],
                    rows[i]["ovl"],
                    alias_type=row_types[i],
                )
            )
            changed += 1

    return changed


def apply_file_level_validation(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    changed = 0
    for i, row in enumerate(rows):
        old = (row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"])
        new = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=row_types[i],
        )
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(old, new)):
            row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = new
            changed += 1
    return changed


def apply_file_consistency_to_oto_file(
    oto_path: str,
    *,
    custom_map: Optional[Dict[str, str]] = None,
    validate_fn: Callable = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    if validate_fn is None:
        from core.oto_generator import validate_oto_params

        validate_fn = validate_oto_params

    stats = {
        "continuity_changed": 0,
        "vc_neighbor_changed": 0,
        "smoothing_changed": 0,
        "validation_changed": 0,
        "total_changed": 0,
    }

    if not oto_path or not os.path.exists(oto_path):
        return stats

    text = read_text_with_fallback(oto_path)
    rows_by_wav: Dict[str, List[Dict[str, object]]] = {}
    line_order: List[Tuple[object, ...]] = []

    for raw in text.splitlines():
        row = _parse_oto_line(raw)
        if not row:
            line_order.append(("raw", raw.rstrip("\n")))
            continue
        wav_key = str(row["wav"])
        rows_by_wav.setdefault(wav_key, []).append(row)
        line_order.append(("row", wav_key, len(rows_by_wav[wav_key]) - 1))

    alias_cache: Dict[str, str] = {}

    def _row_anchor(row):
        try:
            return float(row.get("offset", 0.0)) + float(row.get("pre", 0.0))
        except Exception:
            try:
                return float(row.get("offset", 0.0))
            except Exception:
                return 0.0

    for _wav_name, rows in rows_by_wav.items():
        ordered = sorted(list(enumerate(rows)), key=lambda item: (_row_anchor(item[1]), item[0]))
        ordered_rows = [row for _idx, row in ordered]
        row_types = [_classify_cached(str(row["alias"]), alias_cache, custom_map) for row in ordered_rows]

        stats["continuity_changed"] += enforce_adjacent_continuity(ordered_rows, row_types, validate_fn)
        stats["vc_neighbor_changed"] += adjust_vc_neighbor_alignment(ordered_rows, row_types, validate_fn)
        stats["smoothing_changed"] += smooth_abrupt_changes(ordered_rows, row_types, validate_fn)
        stats["validation_changed"] += apply_file_level_validation(ordered_rows, row_types, validate_fn)

    stats["total_changed"] = (
        stats["continuity_changed"] + stats["vc_neighbor_changed"] + stats["smoothing_changed"] + stats["validation_changed"]
    )
    if stats["total_changed"] <= 0:
        return stats

    with open(oto_path, "w", encoding="utf-8") as f:
        for item in line_order:
            if item[0] == "raw":
                f.write(str(item[1]) + "\n")
                continue
            row = rows_by_wav[str(item[1])][int(item[2])]
            f.write(_format_oto_line(row) + "\n")

    if callable(log_fn):
        log_fn(
            f"[FileConsistency] continuity={stats['continuity_changed']}, "
            f"vc_neighbor={stats['vc_neighbor_changed']}, "
            f"smoothing={stats['smoothing_changed']}, "
            f"validation={stats['validation_changed']}"
        )

    return stats


__all__ = [
    "apply_file_consistency_to_oto_file",
    "adjust_vc_neighbor_alignment",
    "enforce_adjacent_continuity",
    "smooth_abrupt_changes",
    "apply_file_level_validation",
]
