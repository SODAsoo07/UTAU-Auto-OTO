from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.oto_targets import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter and reweight OTO manifest for robust OOD training.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest_prepared.jsonl"))
    parser.add_argument("--summary", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest_prepared_summary.json"))
    parser.add_argument("--min-duration-ms", type=float, default=180.0)
    parser.add_argument("--min-sample-weight", type=float, default=0.30)
    parser.add_argument("--max-offset-ratio", type=float, default=0.90)
    parser.add_argument("--min-cutoff-ratio", type=float, default=0.08)
    parser.add_argument("--activity-window-check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activity-mid-margin-ms", type=float, default=90.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--voicebank-weight-power", type=float, default=0.25)
    parser.add_argument("--role-weight-power", type=float, default=0.20)
    args = parser.parse_args()

    rows = read_jsonl(str(args.manifest))
    if not rows:
        raise SystemExit(2)
    removed = Counter()
    keep: list[dict[str, Any]] = []
    profile_cache: dict[str, dict[str, float]] = {}
    for row in rows:
        ok, reason = _row_filter_reason(
            row,
            min_duration_ms=float(args.min_duration_ms),
            min_sample_weight=float(args.min_sample_weight),
            max_offset_ratio=float(args.max_offset_ratio),
            min_cutoff_ratio=float(args.min_cutoff_ratio),
            activity_window_check=bool(args.activity_window_check),
            activity_mid_margin_ms=float(args.activity_mid_margin_ms),
            sample_rate=int(args.sample_rate),
            profile_cache=profile_cache,
        )
        if not ok:
            removed[str(reason)] += 1
            continue
        keep.append(dict(row))

    _reweight_rows(
        keep,
        voicebank_weight_power=float(args.voicebank_weight_power),
        role_weight_power=float(args.role_weight_power),
    )
    write_jsonl(str(args.out), keep)
    summary = {
        "manifest": os.path.abspath(str(args.manifest)),
        "out": os.path.abspath(str(args.out)),
        "rows_in": len(rows),
        "rows_out": len(keep),
        "drop_rate": float(len(rows) - len(keep)) / float(max(1, len(rows))),
        "removed_reasons": dict(removed),
        "by_voicebank_out": _count_field(keep, "voicebank_id"),
        "by_role_out": _count_field(keep, "alias_role"),
        "by_format_out": _count_field(keep, "format_type"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(str(args.summary))), exist_ok=True)
    with open(str(args.summary), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if keep else 2


def _row_filter_reason(
    row: dict[str, Any],
    *,
    min_duration_ms: float,
    min_sample_weight: float,
    max_offset_ratio: float,
    min_cutoff_ratio: float,
    activity_window_check: bool,
    activity_mid_margin_ms: float,
    sample_rate: int,
    profile_cache: dict[str, dict[str, float]],
) -> tuple[bool, str]:
    duration = max(1.0, float(row.get("duration_ms", 0.0) or 0.0))
    if duration < float(min_duration_ms):
        return False, "short_duration"
    sample_weight = float(row.get("sample_weight", row.get("weight", 1.0)) or 1.0)
    if sample_weight < float(min_sample_weight):
        return False, "low_sample_weight"
    offset = float(row.get("anchor_offset_ms", row.get("target_offset_ms", 0.0)) or 0.0)
    cutoff = float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0)
    pre = float(row.get("anchor_preutterance_ms", 0.0) or 0.0)
    cons = float(row.get("anchor_consonant_ms", 0.0) or 0.0)
    if cutoff <= offset + 5.0:
        return False, "degenerate_cutoff"
    if offset >= duration * float(max_offset_ratio):
        return False, "late_offset"
    if cutoff <= duration * float(min_cutoff_ratio):
        return False, "early_cutoff"
    if activity_window_check:
        audio_path = str(row.get("audio", "") or "")
        profile = _analyze_activity_profile(audio_path, sample_rate=sample_rate, cache=profile_cache)
        start = float(profile.get("active_start_ms", 0.0) or 0.0)
        end = float(profile.get("active_end_ms", duration) or duration)
        if end > start + 30.0:
            mid = (pre + cons) * 0.5
            margin = float(activity_mid_margin_ms)
            if mid < (start - margin) or mid > (end + margin):
                return False, "outside_activity_window"
    return True, "ok"


def _reweight_rows(rows: list[dict[str, Any]], *, voicebank_weight_power: float, role_weight_power: float) -> None:
    if not rows:
        return
    vb_counter = Counter(str(row.get("voicebank_id", "") or "unknown_voicebank") for row in rows)
    role_counter = Counter(str(row.get("alias_role", "") or "other").strip().lower() or "other" for row in rows)
    for row in rows:
        base = float(row.get("sample_weight", row.get("weight", 1.0)) or 1.0)
        vb = str(row.get("voicebank_id", "") or "unknown_voicebank")
        role = str(row.get("alias_role", "") or "other").strip().lower() or "other"
        w = base
        w /= float(max(1, vb_counter.get(vb, 1))) ** float(voicebank_weight_power)
        w /= float(max(1, role_counter.get(role, 1))) ** float(role_weight_power)
        row["sample_weight"] = round(float(max(0.05, min(2.5, w))), 6)


def _analyze_activity_profile(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return dict(cache[key])
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        cache[key] = {}
        return {}
    duration_ms = max(1.0, float(duration_sec) * 1000.0)
    if samples is None or int(getattr(samples, "shape", [0])[0]) <= 0 or sr <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    abs_samples = np.abs(np.asarray(samples, dtype=np.float32))
    peak = float(np.max(abs_samples)) if abs_samples.size else 0.0
    if peak <= 1e-6:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    threshold = max(peak * 0.06, float(np.percentile(abs_samples, 65.0)) * 2.0, 1e-4)
    active_idx = np.flatnonzero(abs_samples >= threshold)
    if active_idx.size <= 0:
        threshold = max(peak * 0.03, 1e-5)
        active_idx = np.flatnonzero(abs_samples >= threshold)
    if active_idx.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    start_ms = (float(active_idx[0]) / float(sr)) * 1000.0
    end_ms = (float(active_idx[-1]) / float(sr)) * 1000.0
    profile = {
        "active_start_ms": max(0.0, start_ms - 25.0),
        "active_end_ms": min(duration_ms, end_ms + 25.0),
        "duration_ms": duration_ms,
    }
    cache[key] = profile
    return dict(profile)


def _count_field(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


if __name__ == "__main__":
    raise SystemExit(main())

