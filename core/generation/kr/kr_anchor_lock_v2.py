from __future__ import annotations


def resolve_kr_anchor_targets(alias_type, c_end, n_start, n_end):
    alias_kind = str(alias_type or "").strip().lower()
    anchor_abs = float(c_end)
    next_onset_abs = float(n_start)
    next_vowel_abs = float(n_end)
    if alias_kind == "vc":
        anchor_abs = float(n_start)
    elif alias_kind == "vv":
        anchor_abs = float(c_end)
    return anchor_abs, next_onset_abs, next_vowel_abs


def build_kr_anchor_lock_stats_delta(alias_type, applied_rules):
    rules = set(applied_rules or [])
    delta = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    if not rules:
        return delta
    delta["anchor_locked_count"] = 1
    if "cutoff_next_onset_clamp" in rules or "cutoff_next_vowel_clamp" in rules:
        delta["cutoff_clamped_count"] = 1
        if str(alias_type or "").strip().lower() == "vc":
            delta["vc_cutoff_leak_guard_count"] = 1
    return delta


def build_kr_anchor_lock_log_record(
    *,
    fname,
    alias_text,
    format_type,
    alias_type,
    lite,
    before,
    result,
):
    rules = tuple(sorted(set(getattr(result, "applied_rules", ()) or ())))
    if not rules:
        return None
    return {
        "event": "anchor_lock",
        "language": "korean",
        "format_type": str(format_type or "").strip().lower(),
        "alias_type": alias_type,
        "file": fname,
        "alias": alias_text,
        "lite": bool(lite),
        "before": {
            "offset": float(before[0]),
            "consonant": float(before[1]),
            "cutoff": float(before[2]),
            "pre": float(before[3]),
            "ovl": float(before[4]),
        },
        "after": {
            "offset": float(result.offset),
            "consonant": float(result.consonant),
            "cutoff": float(result.cutoff),
            "pre": float(result.pre),
            "ovl": float(result.ovl),
        },
        "anchor_shift_ms": float(getattr(result, "anchor_shift_ms", 0.0) or 0.0),
        "rules": list(rules),
    }


__all__ = [
    "build_kr_anchor_lock_log_record",
    "build_kr_anchor_lock_stats_delta",
    "resolve_kr_anchor_targets",
]
