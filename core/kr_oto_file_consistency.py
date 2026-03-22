"""
Korean OTO file-level consistency postprocess.

This module performs a lightweight second pass after row-level generation:
1. clamp excessive overlap/gap between adjacent rows in each wav group
2. smooth abrupt parameter jumps within the same alias type/consonant group
3. re-validate every row to keep hard ordering constraints
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

from core.generator_finish import write_oto_lines
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

def _continuity_enabled() -> bool:
    return str(os.environ.get("UTOA_KR_CONTINUITY_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

def _vc_neighbor_enabled() -> bool:
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


def _consistency_strength_from_mapping(mapping_context: Optional[Dict[str, object]]) -> float:
    env_override = _env_float("UTOA_KR_CONSISTENCY_STRENGTH", -1.0)
    if env_override > 0.0:
        return _clamp(env_override, 0.55, 1.90)

    payload = mapping_context if isinstance(mapping_context, dict) else {}
    nested = payload.get("mapping") if isinstance(payload.get("mapping"), dict) else None
    if isinstance(nested, dict):
        payload = nested

    try:
        mapping_conf = _clamp(float(payload.get("mapping_confidence", 0.0) or 0.0), 0.0, 1.0)
    except Exception:
        mapping_conf = 0.0
    try:
        blank_mean = _clamp(float(payload.get("blank_confidence_mean", 0.0) or 0.0), 0.0, 1.0)
    except Exception:
        blank_mean = 0.0
    tier = str(payload.get("mapping_tier", "") or "").strip().lower()

    strength = 1.0
    if tier == "low":
        strength += 0.20
    elif tier == "high":
        strength -= 0.10
    strength += max(0.0, 0.72 - mapping_conf) * 0.70
    strength += max(0.0, blank_mean - 0.45) * 0.80
    return _clamp(strength, 0.55, 1.90)


def _resolve_mapping_payload(mapping_context: Optional[Dict[str, object]]) -> Dict[str, object]:
    payload = mapping_context if isinstance(mapping_context, dict) else {}
    nested = payload.get("mapping")
    if isinstance(nested, dict):
        payload = nested
    return payload


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _estimate_row_confidences(
    rows: List[Dict[str, object]],
    row_types: List[str],
    mapping_context: Optional[Dict[str, object]],
) -> List[float]:
    payload = _resolve_mapping_payload(mapping_context)
    file_conf = _clamp(_to_float(payload.get("mapping_confidence"), 0.72), 0.20, 0.98)
    blank_mean = _clamp(_to_float(payload.get("blank_confidence_mean"), 0.0), 0.0, 1.0)

    pre_abs = [float(r["offset"]) + float(r["pre"]) for r in rows]
    confs: List[float] = []
    for i, alias_type in enumerate(row_types):
        conf = file_conf
        if alias_type in {"vc", "vv"}:
            conf -= 0.08
        elif alias_type in {"cv", "cv_head"}:
            conf += 0.02

        if i > 0:
            step = pre_abs[i] - pre_abs[i - 1]
            if step < -5.0:
                conf -= 0.14
            elif step < 2.0:
                conf -= 0.05
            elif step > 420.0:
                conf -= 0.04
        if i + 1 < len(pre_abs):
            next_step = pre_abs[i + 1] - pre_abs[i]
            if next_step < -5.0:
                conf -= 0.10

        conf -= max(0.0, blank_mean - 0.45) * 0.22
        confs.append(_clamp(conf, 0.18, 0.99))
    return confs


def _pair_adjust_scale(
    left_idx: int,
    right_idx: int,
    row_confidences: Optional[List[float]],
    strength: float,
) -> float:
    if not row_confidences:
        return _clamp(float(strength), 0.35, 1.80)
    left = row_confidences[left_idx] if left_idx < len(row_confidences) else 0.6
    right = row_confidences[right_idx] if right_idx < len(row_confidences) else 0.6
    pair_conf = _clamp((float(left) + float(right)) * 0.5, 0.0, 1.0)
    # High confidence rows move less, low confidence rows move more.
    scale = (0.40 + (1.12 - pair_conf)) * float(strength)
    return _clamp(scale, 0.22, 1.90)


def _global_min_anchor_gap_ms() -> float:
    return _env_float("UTOA_KR_GLOBAL_ALIGN_MIN_GAP_MS", 2.0)


def _global_anchor_shift_ms() -> float:
    return _env_float("UTOA_KR_GLOBAL_ALIGN_MAX_SHIFT_MS", 36.0)


def _global_align_enabled() -> bool:
    return str(os.environ.get("UTOA_KR_GLOBAL_ALIGN_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _stage2_bridge_enabled() -> bool:
    return str(os.environ.get("UTOA_KR_STAGE2_BRIDGE_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _gap_rule_enabled() -> bool:
    return str(os.environ.get("UTOA_KR_GAP_RULE_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def global_anchor_pre_alignment(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
    *,
    strength: float = 1.0,
    row_confidences: Optional[List[float]] = None,
) -> int:
    anchor_indices = [i for i, t in enumerate(row_types) if t in _CV_TYPES]
    if len(anchor_indices) < 3:
        return 0

    min_gap = max(0.0, _global_min_anchor_gap_ms())
    max_shift = max(4.0, _global_anchor_shift_ms()) * _clamp(float(strength), 0.65, 2.0)
    changed = 0

    curr = {i: float(rows[i]["offset"]) + float(rows[i]["pre"]) for i in anchor_indices}
    target = dict(curr)
    # Forward pass: monotonic order with minimum gap.
    for idx in range(1, len(anchor_indices)):
        prev_i = anchor_indices[idx - 1]
        cur_i = anchor_indices[idx]
        req = target[prev_i] + min_gap
        if target[cur_i] < req:
            target[cur_i] = req

    # Backward pass: keep as close as possible while preserving monotonicity.
    for idx in range(len(anchor_indices) - 2, -1, -1):
        cur_i = anchor_indices[idx]
        next_i = anchor_indices[idx + 1]
        max_allowed = target[next_i] - min_gap
        if target[cur_i] > max_allowed:
            target[cur_i] = max_allowed

    for i in anchor_indices:
        row = rows[i]
        before = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        delta = float(target[i] - curr[i])
        row_conf = row_confidences[i] if row_confidences and i < len(row_confidences) else 0.6
        local_max = max_shift * _clamp(0.70 + (1.0 - float(row_conf)), 0.60, 1.70)
        delta = _clamp(delta, -local_max, local_max)
        move_w = _clamp((0.24 + (1.0 - float(row_conf)) * 0.50) * float(strength), 0.18, 0.86)
        if abs(delta) <= 1e-6:
            continue
        row["offset"] = max(0.0, float(row["offset"]) + (delta * move_w))
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=row_types[i],
        )
        after = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(before, after)):
            changed += 1
    return changed


def _bridge_target_ratio(alias_type: str) -> float:
    t = str(alias_type or "").strip().lower()
    if t == "vc":
        return 0.30
    if t == "vv":
        return 0.46
    return 0.38


def align_bridge_rows_to_cv_anchors(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
    *,
    strength: float = 1.0,
    row_confidences: Optional[List[float]] = None,
) -> int:
    changed = 0
    for i, alias_type in enumerate(row_types):
        if alias_type not in {"vc", "vv"}:
            continue
        prev_cv = None
        next_cv = None
        for j in range(i - 1, -1, -1):
            if row_types[j] in _CV_TYPES:
                prev_cv = j
                break
        for j in range(i + 1, len(rows)):
            if row_types[j] in _CV_TYPES:
                next_cv = j
                break
        if prev_cv is None or next_cv is None:
            continue

        prev_end = float(rows[prev_cv]["offset"]) + abs(float(rows[prev_cv]["cutoff"]))
        next_pre = float(rows[next_cv]["offset"]) + float(rows[next_cv]["pre"])
        span_lo = prev_end + 8.0
        span_hi = next_pre - 10.0
        if span_hi <= span_lo + 4.0:
            continue

        row = rows[i]
        current_pre_abs = float(row["offset"]) + float(row["pre"])
        ratio = _bridge_target_ratio(alias_type)
        target_pre_abs = span_lo + ((span_hi - span_lo) * ratio)
        delta = target_pre_abs - current_pre_abs
        if abs(delta) <= 1e-6:
            continue

        row_conf = row_confidences[i] if row_confidences and i < len(row_confidences) else 0.56
        step_w = _clamp((0.22 + (1.0 - float(row_conf)) * 0.52) * float(strength), 0.16, 0.86)
        local_cap = _clamp(_env_float("UTOA_KR_STAGE2_BRIDGE_SHIFT_CAP_MS", 42.0), 8.0, 120.0)
        delta = _clamp(delta, -local_cap, local_cap)

        before = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        row["offset"] = max(0.0, float(row["offset"]) + (delta * step_w))
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=alias_type,
        )
        after = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(before, after)):
            changed += 1
    return changed


_GAP_RULES = {
    "cv": {
        "hard": {"cons_min": 8.0, "cons_max": 260.0, "cut_min": 10.0, "cut_max": 320.0, "ovl_ratio_max": 0.92},
        "soft": {"cons_min": 72.0, "cons_max": 190.0, "cut_min": 44.0, "cut_max": 210.0},
    },
    "cv_head": {
        "hard": {"cons_min": 8.0, "cons_max": 260.0, "cut_min": 10.0, "cut_max": 320.0, "ovl_ratio_max": 0.88},
        "soft": {"cons_min": 68.0, "cons_max": 182.0, "cut_min": 40.0, "cut_max": 200.0},
    },
    "vc": {
        "hard": {"cons_min": 6.0, "cons_max": 120.0, "cut_min": 6.0, "cut_max": 84.0, "ovl_ratio_max": 0.92},
        "soft": {"cons_min": 22.0, "cons_max": 62.0, "cut_min": 10.0, "cut_max": 42.0},
    },
    "vv": {
        "hard": {"cons_min": 6.0, "cons_max": 180.0, "cut_min": 8.0, "cut_max": 140.0, "ovl_ratio_max": 0.94},
        "soft": {"cons_min": 56.0, "cons_max": 146.0, "cut_min": 18.0, "cut_max": 96.0},
    },
    "vcv": {
        "hard": {"cons_min": 8.0, "cons_max": 240.0, "cut_min": 10.0, "cut_max": 280.0, "ovl_ratio_max": 0.90},
        "soft": {"cons_min": 70.0, "cons_max": 176.0, "cut_min": 34.0, "cut_max": 180.0},
    },
    "mono": {
        "hard": {"cons_min": 8.0, "cons_max": 240.0, "cut_min": 10.0, "cut_max": 280.0, "ovl_ratio_max": 0.92},
        "soft": {"cons_min": 64.0, "cons_max": 170.0, "cut_min": 30.0, "cut_max": 170.0},
    },
}


def _nearest_in_range(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return float(lo)
    if value > hi:
        return float(hi)
    return float(value)


def apply_alias_gap_constraints(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
    *,
    strength: float = 1.0,
    row_confidences: Optional[List[float]] = None,
) -> int:
    changed = 0
    for i, row in enumerate(rows):
        alias_type = row_types[i]
        rule = _GAP_RULES.get(alias_type) or _GAP_RULES.get("cv")
        hard = rule["hard"]
        soft = rule["soft"]

        before = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        pre = max(float(row["pre"]), 0.0)
        cons = max(float(row["cons"]), pre + float(hard["cons_min"]))
        cut_abs = max(abs(float(row["cutoff"])), cons + float(hard["cut_min"]))
        ovl = max(0.0, min(float(row["ovl"]), pre * float(hard["ovl_ratio_max"])))

        cons_gap = _clamp(cons - pre, float(hard["cons_min"]), float(hard["cons_max"]))
        cut_gap = _clamp(cut_abs - cons, float(hard["cut_min"]), float(hard["cut_max"]))
        cons = pre + cons_gap
        cut_abs = cons + cut_gap

        row_conf = row_confidences[i] if row_confidences and i < len(row_confidences) else 0.6
        soft_w = _clamp((0.18 + (1.0 - float(row_conf)) * 0.38) * float(strength), 0.0, 0.60)
        soft_cons_target = _nearest_in_range(cons_gap, float(soft["cons_min"]), float(soft["cons_max"]))
        soft_cut_target = _nearest_in_range(cut_gap, float(soft["cut_min"]), float(soft["cut_max"]))
        if soft_w > 0.0:
            cons_gap = _blend(cons_gap, soft_cons_target, soft_w)
            cut_gap = _blend(cut_gap, soft_cut_target, soft_w)
            cons_gap = _clamp(cons_gap, float(hard["cons_min"]), float(hard["cons_max"]))
            cut_gap = _clamp(cut_gap, float(hard["cut_min"]), float(hard["cut_max"]))
            cons = pre + cons_gap
            cut_abs = cons + cut_gap

        row["cons"] = cons
        row["cutoff"] = -cut_abs
        row["ovl"] = ovl
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = validate_fn(
            row["offset"],
            row["cons"],
            row["cutoff"],
            row["pre"],
            row["ovl"],
            alias_type=alias_type,
        )
        after = (
            float(row["offset"]),
            float(row["cons"]),
            float(row["cutoff"]),
            float(row["pre"]),
            float(row["ovl"]),
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(before, after)):
            changed += 1
    return changed

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
    *,
    strength: float = 1.0,
    row_confidences: Optional[List[float]] = None,
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
        pair_scale = _pair_adjust_scale(i, i + 1, row_confidences, strength)
        max_offset_adj = _max_offset_adj_ms() * _clamp(float(pair_scale), 0.55, 2.10)
        prev_row = rows[i]
        next_row = rows[i + 1]
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
            target_cut_abs = curr_cut_abs + ((target_cut_abs - curr_cut_abs) * _clamp(pair_scale, 0.25, 1.45))
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
                    next_row["offset"] = max(0.0, float(next_row["offset"]) + (off_adj * _clamp(pair_scale, 0.25, 1.5)))
                    if _validate_row(next_row, next_type, before_state):
                        changed += 1

        elif overlap_ms < min_overlap:
            prev_offset = float(prev_row["offset"])
            curr_cut_abs = abs(float(prev_row["cutoff"]))
            target_cut_abs = next_offset + float(min_overlap) - prev_offset
            max_extend = 160.0
            target_cut_abs = curr_cut_abs + ((target_cut_abs - curr_cut_abs) * _clamp(pair_scale, 0.25, 1.45))
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
                    next_row["offset"] = max(0.0, float(next_row["offset"]) - (off_adj * _clamp(pair_scale, 0.25, 1.5)))
                    if _validate_row(next_row, next_type, before_state):
                        changed += 1

    return changed


def adjust_vc_neighbor_alignment(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
    *,
    strength: float = 1.0,
    row_confidences: Optional[List[float]] = None,
) -> int:
    """
    Soft-align VC rows using immediate neighbor timing.
    - offset is nudged toward previous row cutoff (end)
    - cutoff is nudged toward next row offset (start)
    """
    if not _vc_neighbor_enabled():
        return 0

    changed = 0
    strength_v = _clamp(float(strength), 0.60, 1.90)
    base_blend = _clamp(_vc_neighbor_blend() * (0.72 + (0.40 * strength_v)), 0.12, 0.88)
    base_max_shift = _vc_neighbor_max_shift() * _clamp(strength_v, 0.65, 1.90)
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
        row_conf = row_confidences[i] if row_confidences and i < len(row_confidences) else 0.56
        row_scale = _clamp((0.38 + (1.0 - float(row_conf)) * 1.02), 0.30, 1.55)
        blend = _clamp(base_blend * row_scale, 0.08, 0.90)
        max_shift = base_max_shift * _clamp(row_scale, 0.45, 1.70)

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


def smooth_abrupt_changes(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    changed = 0
    params_to_smooth = ["pre", "cons", "ovl", "cutoff_abs"]

    for i in range(1, len(rows)):
        if row_types[i] != row_types[i - 1]:
            continue
        if _get_consonant_group(rows[i]["alias"]) != _get_consonant_group(rows[i - 1]["alias"]):
            continue

        row_changed = False
        for param in params_to_smooth:
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
            if change_ratio <= _SMOOTHING_THRESHOLD_RATIO:
                continue

            smoothed = _blend(curr_val, prev_val, _SMOOTHING_BLEND_WEIGHT)
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
    mapping_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if validate_fn is None:
        from core.oto_generator import validate_oto_params

        validate_fn = validate_oto_params

    stats = {
        "global_align_changed": 0,
        "stage2_bridge_changed": 0,
        "continuity_changed": 0,
        "vc_neighbor_changed": 0,
        "gap_rule_changed": 0,
        "smoothing_changed": 0,
        "validation_changed": 0,
        "total_changed": 0,
        "consistency_strength": 1.0,
    }

    if not oto_path or not os.path.exists(oto_path):
        return stats

    strength = _consistency_strength_from_mapping(mapping_context)
    stats["consistency_strength"] = float(strength)

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
        row_confidences = _estimate_row_confidences(ordered_rows, row_types, mapping_context)

        if _global_align_enabled():
            stats["global_align_changed"] += global_anchor_pre_alignment(
                ordered_rows,
                row_types,
                validate_fn,
                strength=strength,
                row_confidences=row_confidences,
            )
        if _stage2_bridge_enabled():
            stats["stage2_bridge_changed"] += align_bridge_rows_to_cv_anchors(
                ordered_rows,
                row_types,
                validate_fn,
                strength=strength,
                row_confidences=row_confidences,
            )

        if _continuity_enabled():
            stats["continuity_changed"] += enforce_adjacent_continuity(
                ordered_rows,
                row_types,
                validate_fn,
                strength=strength,
                row_confidences=row_confidences,
            )
        stats["vc_neighbor_changed"] += adjust_vc_neighbor_alignment(
            ordered_rows,
            row_types,
            validate_fn,
            strength=strength,
            row_confidences=row_confidences,
        )
        if _gap_rule_enabled():
            stats["gap_rule_changed"] += apply_alias_gap_constraints(
                ordered_rows,
                row_types,
                validate_fn,
                strength=strength,
                row_confidences=row_confidences,
            )
        stats["smoothing_changed"] += smooth_abrupt_changes(ordered_rows, row_types, validate_fn)
        stats["validation_changed"] += apply_file_level_validation(ordered_rows, row_types, validate_fn)

    stats["total_changed"] = (
        stats["global_align_changed"]
        + stats["stage2_bridge_changed"]
        + stats["continuity_changed"]
        + stats["vc_neighbor_changed"]
        + stats["gap_rule_changed"]
        + stats["smoothing_changed"]
        + stats["validation_changed"]
    )
    if stats["total_changed"] <= 0:
        return stats

    out_lines = []
    for item in line_order:
        if item[0] == "raw":
            out_lines.append(str(item[1]))
            continue
        row = rows_by_wav[str(item[1])][int(item[2])]
        out_lines.append(_format_oto_line(row))
    write_oto_lines(oto_path, out_lines)

    if callable(log_fn):
        log_fn(
            f"[FileConsistency] global={stats['global_align_changed']}, "
            f"stage2_bridge={stats['stage2_bridge_changed']}, "
            f"continuity={stats['continuity_changed']}, "
            f"vc_neighbor={stats['vc_neighbor_changed']}, "
            f"gap_rule={stats['gap_rule_changed']}, "
            f"smoothing={stats['smoothing_changed']}, "
            f"validation={stats['validation_changed']}, "
            f"strength={stats['consistency_strength']:.2f}"
        )

    return stats


__all__ = [
    "apply_file_consistency_to_oto_file",
    "adjust_vc_neighbor_alignment",
    "enforce_adjacent_continuity",
    "smooth_abrupt_changes",
    "apply_file_level_validation",
]
