"""
Japanese OTO file-level consistency postprocess (VC neighbor alignment).

This module performs a lightweight pass after row-level generation to
soft-align VC rows using immediate neighbor timing.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from core.ja_oto_mapping import classify_ja_alias
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


def _vc_neighbor_enabled() -> bool:
    return str(os.environ.get("UTOA_JA_VC_NEIGHBOR_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _cvn_correction_enabled() -> bool:
    return str(os.environ.get("UTOA_CVN_CORRECTION_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _cvn_low_conf_only_enabled() -> bool:
    return str(os.environ.get("UTOA_CVN_LOW_CONF_ONLY", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_mapping_is_low_conf(runtime_report: object) -> bool:
    if not isinstance(runtime_report, dict):
        return False
    mapping = runtime_report.get("mapping")
    if not isinstance(mapping, dict):
        return False
    if bool(mapping.get("file_low_conf", False)):
        return True
    trust_tier = str(mapping.get("trust_tier", "") or "").strip().lower()
    if trust_tier == "low":
        return True
    reasons = mapping.get("low_conf_reasons")
    if isinstance(reasons, list) and len(reasons) > 0:
        return True
    try:
        trust_score = float(mapping.get("trust_score", 1.0) or 1.0)
    except Exception:
        trust_score = 1.0
    if trust_tier == "mid" and trust_score < 0.66:
        return True
    return False


def _cvn_mfa_gate_allows(runtime_report: object) -> bool:
    if not _cvn_correction_enabled():
        return False
    if not _cvn_low_conf_only_enabled():
        return True
    return _runtime_mapping_is_low_conf(runtime_report)


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
        if next_offset <= 0.0 or prev_end <= 0.0:
            continue

        before_state = _row_state(row)

        target_off = max(0.0, prev_end - lead_ms)
        delta_off = target_off - float(row["offset"])
        if abs(delta_off) > 1e-6:
            delta_off = _clamp(delta_off, -max_shift, max_shift)
            row["offset"] = max(0.0, float(row["offset"]) + delta_off * blend)

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


def apply_ja_vc_neighbor_to_oto_file(
    oto_path: str,
    *,
    custom_map: Optional[Dict[str, str]] = None,
    validate_fn: Callable = None,
    log_fn: Optional[Callable[[str], None]] = None,
    runtime_report: object = None,
) -> Dict[str, int]:
    if validate_fn is None:
        from core.oto_generator import validate_oto_params

        validate_fn = validate_oto_params

    stats = {
        "vc_neighbor_changed": 0,
        "total_changed": 0,
    }

    if not oto_path or not os.path.exists(oto_path):
        return stats

    if not _vc_neighbor_enabled():
        if callable(log_fn):
            log_fn("[JA-Consistency] VC neighbor 보정 비활성화(언어 옵션).")
        return stats
    if not _cvn_mfa_gate_allows(runtime_report):
        if callable(log_fn):
            if _cvn_low_conf_only_enabled():
                log_fn("[JA-Consistency] CVN low_conf_only gate: MFA 신뢰도 기준 미충족으로 VC neighbor 보정을 건너뜁니다.")
            else:
                log_fn("[JA-Consistency] CVN gate OFF: VC neighbor 보정을 건너뜁니다.")
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
            f"[FileConsistency] vc_neighbor={stats['vc_neighbor_changed']}"
        )

    return stats


__all__ = [
    "apply_ja_vc_neighbor_to_oto_file",
    "adjust_vc_neighbor_alignment",
]
