from __future__ import annotations

from dataclasses import replace


def retune_ja_anchor_profile(profile, *, format_type, alias_key):
    fmt = str(format_type or "").strip().lower()
    alias_norm = str(alias_key or "").strip().lower()
    tuned = profile
    if fmt == "vcv" and alias_norm in {"vcv", "vcv_n_bridge", "cv", "cv_head"}:
        tuned = replace(
            tuned,
            pre_window_before_ms=max(2.0, float(tuned.pre_window_before_ms) - 1.0),
            pre_window_after_ms=float(tuned.pre_window_after_ms) + 2.0,
            pre_floor_ms=float(tuned.pre_floor_ms) + 2.0,
            blend_weight=min(0.66, float(tuned.blend_weight) + 0.04),
        )
    if fmt == "vcv" and alias_norm in {"vcv", "vcv_vv_like", "vcv_n_bridge", "cv", "cv_head"}:
        tuned = replace(
            tuned,
            cut_to_next_onset_allow_ms=None,
            cut_to_next_vowel_allow_ms=None,
        )
    return tuned


def build_ja_anchor_lock_stats_delta(alias_type, applied_rules):
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


def build_ja_anchor_lock_log_record(
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
        "language": "japanese",
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
    "build_ja_anchor_lock_log_record",
    "build_ja_anchor_lock_stats_delta",
    "retune_ja_anchor_profile",
]
