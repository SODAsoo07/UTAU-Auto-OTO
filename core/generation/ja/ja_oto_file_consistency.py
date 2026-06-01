"""
Japanese OTO file-level consistency postprocess (VC neighbor alignment).

This module performs a lightweight pass after row-level generation to
soft-align VC rows using immediate neighbor timing.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from core.ja_oto_mapping import classify_ja_alias
from core.oto_normalization import canonicalize_alias_for_matching
from core.oto_file_utils import read_text_with_fallback

_ALIAS_ATTACHED_PITCH_SUFFIX_RE = re.compile(r"[A-Ga-g](?:#|b)?[0-8]$")


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


def _call_validate(validate_fn, offset, consonant, cutoff, pre, ovl, *, alias_type=""):
    try:
        return validate_fn(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
    except TypeError:
        return validate_fn(offset, consonant, cutoff, pre, ovl)


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
        cache[key] = classify_ja_alias(key, custom_map)
    return cache[key]

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
    return str(os.environ.get("UTOA_JA_VC_NEIGHBOR_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _vc_neighbor_blend() -> float:
    return _env_float("UTOA_JA_VC_NEIGHBOR_BLEND", 0.35)


def _vc_neighbor_max_shift() -> float:
    return _env_float("UTOA_JA_VC_NEIGHBOR_MAX_SHIFT", 45.0)


def _vc_neighbor_lead_ms() -> float:
    return _env_float("UTOA_JA_VC_NEIGHBOR_LEAD_MS", 6.0)


def _vc_neighbor_tail_ms() -> float:
    return _env_float("UTOA_JA_VC_NEIGHBOR_TAIL_MS", 8.0)


def _vc_neighbor_min_len() -> float:
    return _env_float("UTOA_JA_VC_NEIGHBOR_MIN_LEN", 35.0)


def _vc_neighbor_order() -> str:
    raw = str(os.environ.get("UTOA_JA_VC_NEIGHBOR_ORDER", "row_plan")).strip().lower()
    if raw in {"row_plan", "row-plan", "filename", "alias"}:
        return "row_plan"
    if raw in {"time", "timing", "anchor"}:
        return "time"
    return "line"


def _strip_attached_pitch_suffix(alias: str) -> str:
    text = str(alias or "").strip()
    return _ALIAS_ATTACHED_PITCH_SUFFIX_RE.sub("", text).strip()


def _alias_order_keys(alias: str) -> List[str]:
    keys: List[str] = []
    for candidate in (str(alias or "").strip(), _strip_attached_pitch_suffix(alias)):
        if not candidate:
            continue
        for value in (candidate, canonicalize_alias_for_matching("japanese", candidate)):
            value = str(value or "").strip()
            if not value or value in keys:
                continue
            keys.append(value)
            if value.startswith("- "):
                tail = value[2:].strip()
                if tail and tail not in keys:
                    keys.append(tail)
    return keys


def _vc_offset_lead_ms(row: Dict[str, object], *, lead_ms: float, tail_ms: float, min_len: float) -> float:
    pre = max(0.0, float(row.get("pre", 0.0) or 0.0))
    cons = max(0.0, float(row.get("cons", 0.0) or 0.0))
    lower = max(float(min_len) + float(tail_ms), 45.0)
    target = max(pre + float(lead_ms) + float(tail_ms), cons * 0.72, lower)
    return _clamp(target, lower, 220.0)


def adjust_vc_neighbor_alignment(
    rows: List[Dict[str, object]],
    row_types: List[str],
    validate_fn: Callable,
) -> int:
    """
    Soft-align VC rows using immediate neighbor timing.
    - offset is anchored before the following CV-like row
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
        row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = _call_validate(
            validate_fn,
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
        next_anchor = next_offset + max(0.0, float(next_row.get("pre", 0.0) or 0.0))
        if next_anchor <= 0.0 or prev_end <= 0.0:
            continue

        before_state = _row_state(row)

        target_lead = _vc_offset_lead_ms(row, lead_ms=lead_ms, tail_ms=tail_ms, min_len=min_len)
        current_gap_to_next = next_anchor - float(row["offset"])
        late_crossed_next = current_gap_to_next < (float(min_len) + float(tail_ms))
        if not late_crossed_next:
            continue
        target_off = max(0.0, next_anchor - target_lead)
        if prev_end < next_anchor:
            target_off = max(target_off, max(0.0, prev_end - lead_ms))
        delta_off = target_off - float(row["offset"])
        if abs(delta_off) > 1e-6:
            large_mismatch = abs(delta_off) > max(float(max_shift), float(target_lead) * 0.70)
            effective_blend = 1.0 if late_crossed_next else (max(float(blend), 0.70) if large_mismatch else float(blend))
            effective_max_shift = (
                abs(delta_off)
                if late_crossed_next
                else (max(float(max_shift), min(abs(delta_off), 220.0)) if large_mismatch else float(max_shift))
            )
            delta_off = _clamp(delta_off, -effective_max_shift, effective_max_shift)
            row["offset"] = max(0.0, float(row["offset"]) + delta_off * effective_blend)

        next_limit = next_anchor - tail_ms
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


def _row_anchor(row):
    try:
        return float(row.get("offset", 0.0)) + float(row.get("pre", 0.0))
    except Exception:
        try:
            return float(row.get("offset", 0.0))
        except Exception:
            return 0.0


def _row_plan_order_rows(wav_name: str, rows: List[Dict[str, object]]) -> Optional[List[Dict[str, object]]]:
    try:
        from core.mfa_free_oto.row_plan import build_filename_row_plan
    except Exception:
        return None
    try:
        plan = build_filename_row_plan(str(wav_name or ""), language="japanese", format_type="cvvc")
    except Exception:
        return None
    aliases = [str(getattr(item, "alias", "") or "").strip() for item in plan]
    aliases = [alias for alias in aliases if alias]
    if not aliases:
        return None

    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        alias = str(row.get("alias", "") or "").strip()
        for key in _alias_order_keys(alias):
            buckets.setdefault(key, []).append(row)

    ordered: List[Dict[str, object]] = []
    used: set[int] = set()
    for alias in aliases:
        candidates: List[Dict[str, object]] = []
        for key in _alias_order_keys(alias):
            candidates = buckets.get(key) or []
            if candidates:
                break
        while candidates and id(candidates[0]) in used:
            candidates.pop(0)
        if not candidates:
            continue
        row = candidates.pop(0)
        ordered.append(row)
        used.add(id(row))

    if len(ordered) < 3 or (len(ordered) / float(max(1, len(rows)))) < 0.60:
        return None
    for row in rows:
        if id(row) not in used:
            ordered.append(row)
    if _row_plan_order_has_large_backtrack(ordered):
        return None
    return ordered


def _row_plan_order_has_large_backtrack(rows: List[Dict[str, object]]) -> bool:
    if len(rows) < 4:
        return False
    large_backtracks = 0
    max_backtrack = 0.0
    previous = _row_anchor(rows[0])
    for row in rows[1:]:
        current = _row_anchor(row)
        backtrack = previous - current
        if backtrack > 250.0:
            large_backtracks += 1
            max_backtrack = max(max_backtrack, backtrack)
        previous = max(previous, current)
    return bool(large_backtracks >= 2 or max_backtrack > 1200.0)


def _ordered_rows_for_vc_neighbor(wav_name: str, rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], str]:
    order = _vc_neighbor_order()
    if order == "row_plan":
        row_plan_rows = _row_plan_order_rows(wav_name, rows)
        if row_plan_rows is not None:
            return row_plan_rows, "row_plan"
        return [row for _idx, row in sorted(list(enumerate(rows)), key=lambda item: (_row_anchor(item[1]), item[0]))], "time_fallback"
    if order == "time":
        return [row for _idx, row in sorted(list(enumerate(rows)), key=lambda item: (_row_anchor(item[1]), item[0]))], "time"
    return list(rows), "line"


def apply_ja_vc_neighbor_to_oto_file(
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
        "vc_neighbor_changed": 0,
        "total_changed": 0,
        "row_plan_ordered_wavs": 0,
        "row_plan_fallback_wavs": 0,
        "time_ordered_wavs": 0,
        "line_ordered_wavs": 0,
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

    for wav_name, rows in rows_by_wav.items():
        ordered_rows, order_source = _ordered_rows_for_vc_neighbor(wav_name, rows)
        if order_source == "row_plan":
            stats["row_plan_ordered_wavs"] += 1
        elif order_source == "time_fallback":
            stats["row_plan_fallback_wavs"] += 1
        elif order_source == "time":
            stats["time_ordered_wavs"] += 1
        else:
            stats["line_ordered_wavs"] += 1
        row_types = [_classify_cached(str(row["alias"]), alias_cache, custom_map) for row in ordered_rows]
        stats["vc_neighbor_changed"] += adjust_vc_neighbor_alignment(ordered_rows, row_types, validate_fn)

    stats["total_changed"] = stats["vc_neighbor_changed"]
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
            f"[FileConsistency] vc_neighbor={stats['vc_neighbor_changed']}, "
            f"row_plan_ordered={stats['row_plan_ordered_wavs']}, "
            f"row_plan_fallback={stats['row_plan_fallback_wavs']}"
        )

    return stats


__all__ = [
    "apply_ja_vc_neighbor_to_oto_file",
    "adjust_vc_neighbor_alignment",
]
