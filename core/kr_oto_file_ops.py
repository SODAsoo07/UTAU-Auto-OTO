from __future__ import annotations

import datetime
import json
import os
import wave
from statistics import median

from core.kr_oto_bridge import _apply_kr_consonant_timing_shaping
from core.kr_oto_rules import (
    _canonicalize_kr_coda,
    _extract_vc_right_token,
    classify_alias,
    should_ignore_korean_alias,
)
from core.lab_generator import load_custom_phonemes
from core.oto_file_utils import (
    build_occurrence_map as _build_occurrence_map_common,
    extract_base_timing_shape as _extract_base_timing_shape_common,
    parse_oto_line as _parse_oto_line_common,
    read_oto_rows_for_profile as _read_oto_rows_for_profile_common,
    read_text_with_fallback as _read_text_with_fallback_common,
)
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key


def _median(vals):
    if not vals:
        return 0.0
    return float(median([float(v) for v in vals]))


def _safe_ratio(num, den, fallback=0.0):
    try:
        den_v = float(den)
        if abs(den_v) < 1e-6:
            return float(fallback)
        return float(num) / den_v
    except Exception:
        return float(fallback)


def _blend(a, b, w):
    w2 = max(0.0, min(1.0, float(w)))
    return (1.0 - w2) * float(a) + w2 * float(b)


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _validate_oto_params(offset, consonant, cutoff, pre, ovl):
    offset_v = max(float(offset), 0.0)
    pre_v = max(float(pre), 0.0)
    ovl_v = max(float(ovl), 0.0)
    consonant_v = max(float(consonant), 0.0)
    if ovl_v > pre_v:
        ovl_v = pre_v * 0.75
    if consonant_v < pre_v:
        consonant_v = pre_v + 30.0
    cutoff_abs = abs(float(cutoff))
    if cutoff_abs <= consonant_v:
        cutoff_abs = consonant_v + 50.0
    return offset_v, consonant_v, -cutoff_abs, pre_v, ovl_v


def _validate_kr_bridge_params(offset, consonant, cutoff_abs, pre, ovl, *, min_cons_gap=12.0, min_cut_gap=10.0):
    offset_v = max(float(offset), 0.0)
    pre_v = max(float(pre), 0.0)
    ovl_v = max(0.0, min(float(ovl), pre_v))
    cons_v = max(float(consonant), pre_v + float(min_cons_gap))
    cutoff_abs_v = max(float(cutoff_abs), cons_v + float(min_cut_gap))
    return offset_v, cons_v, -cutoff_abs_v, pre_v, ovl_v


def _wav_duration_ms(wav_path):
    try:
        with wave.open(wav_path, "rb") as wf:
            return (wf.getnframes() * 1000.0) / float(wf.getframerate())
    except Exception:
        return 0.0


def _normalize_alias_for_profile(alias):
    return canonicalize_alias_for_matching("korean", alias)


def _read_text_with_fallback(path):
    return _read_text_with_fallback_common(path)


def _parse_oto_line_profile(line):
    return _parse_oto_line_common(line, alias_filter=should_ignore_korean_alias)


def _extract_base_timing_shape(line):
    return _extract_base_timing_shape_common(line, alias_filter=should_ignore_korean_alias)


def _read_kr_oto_rows_for_profile(path):
    return _read_oto_rows_for_profile_common(
        path,
        alias_filter=should_ignore_korean_alias,
        wav_normalizer=normalize_wav_key,
        alias_normalizer=_normalize_alias_for_profile,
    )


def _occurrence_map(rows):
    return _build_occurrence_map_common(rows)


def train_kr_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    auto_rows = _read_kr_oto_rows_for_profile(auto_oto_path)
    manual_rows = _read_kr_oto_rows_for_profile(manual_oto_path)
    if not auto_rows or not manual_rows:
        return None

    custom_map = load_custom_phonemes(custom_phonemes_path)
    alias_type_cache = {}

    def _classify_alias_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    auto_map = _occurrence_map(auto_rows)
    manual_map = _occurrence_map(manual_rows)

    fields = ["offset", "cons", "cutoff", "pre", "ovl"]
    buckets = {}
    matched = 0
    for key, auto_row in auto_map.items():
        manual_row = manual_map.get(key)
        if not manual_row:
            continue
        alias_type = _classify_alias_cached(auto_row["alias"])
        bucket = buckets.setdefault(alias_type, {field: [] for field in fields})
        for field in fields:
            bucket[field].append(float(manual_row[field]) - float(auto_row[field]))
        matched += 1

    if matched < 8:
        return None

    profile = {
        "version": 1,
        "language": "korean",
        "mode": "delta",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": matched,
        "source_auto_name": os.path.basename(auto_oto_path),
        "source_manual_name": os.path.basename(manual_oto_path),
        "buckets": {},
    }
    for alias_type, values in buckets.items():
        n = len(values["offset"])
        if n < 3:
            continue
        profile["buckets"][alias_type] = {
            "n": n,
            "offset": _median(values["offset"]),
            "cons": _median(values["cons"]),
            "cutoff": _median(values["cutoff"]),
            "pre": _median(values["pre"]),
            "ovl": _median(values["ovl"]),
        }
    if not profile["buckets"]:
        return None
    return profile


def save_kr_autotune_profile(path, profile):
    if not path or not profile:
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_kr_autotune_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "buckets" not in obj:
            return None
        return obj
    except Exception:
        return None


def apply_kr_autotune_profile_to_oto(oto_path, profile, custom_phonemes_path=""):
    if not oto_path or not os.path.exists(oto_path):
        return 0
    if isinstance(profile, str):
        profile = load_kr_autotune_profile(profile)
    if not profile:
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    custom_map = load_custom_phonemes(custom_phonemes_path)
    changed = 0
    out_lines = []
    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    for raw in _read_text_with_fallback(oto_path).splitlines():
        row = _parse_oto_line_profile(raw)
        if not row:
            out_lines.append(raw.rstrip("\n"))
            continue

        stat = buckets.get(_classify_cached(row["alias"]))
        if not stat:
            out_lines.append(raw.rstrip("\n"))
            continue

        n = float(stat.get("n", 0))
        if n < 3:
            out_lines.append(raw.rstrip("\n"))
            continue

        weight = min(1.0, max(0.25, n / 24.0))
        offset = row["offset"] + _clamp(stat.get("offset", 0.0), -140.0, 140.0) * weight
        cons = row["cons"] + _clamp(stat.get("cons", 0.0), -160.0, 160.0) * weight
        cutoff = row["cutoff"] + _clamp(stat.get("cutoff", 0.0), -180.0, 180.0) * weight
        pre = row["pre"] + _clamp(stat.get("pre", 0.0), -140.0, 140.0) * weight
        ovl = row["ovl"] + _clamp(stat.get("ovl", 0.0), -120.0, 120.0) * weight
        offset, cons, cutoff, pre, ovl = _validate_oto_params(offset, cons, cutoff, pre, ovl)

        if (
            abs(offset - row["offset"]) > 1e-6
            or abs(cons - row["cons"]) > 1e-6
            or abs(cutoff - row["cutoff"]) > 1e-6
            or abs(pre - row["pre"]) > 1e-6
            or abs(ovl - row["ovl"]) > 1e-6
        ):
            changed += 1
        out_lines.append(
            f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
        )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _resolve_kr_reference_dirs():
    raw = os.environ.get("UTOA_KR_PROFILE_DIRS", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(";") if item.strip()]


def _profile_weight(n):
    return max(0.12, min(0.38, float(n) / 220.0))


def _apply_kr_profile_to_oto_file(oto_path, wav_dir, profile, custom_map=None):
    if not profile or not oto_path or not os.path.exists(oto_path):
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    rows_by_wav = {}
    for raw in _read_text_with_fallback(oto_path).splitlines():
        row = _parse_oto_line_profile(raw)
        if not row:
            continue
        rows_by_wav.setdefault(row["wav"], []).append(row)

    out_lines = []
    changed = 0
    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    for wav_name, rows in rows_by_wav.items():
        dur_ms = _wav_duration_ms(os.path.join(wav_dir, wav_name))
        for idx, row in enumerate(rows):
            stat = buckets.get(_classify_cached(row["alias"]))
            if not stat:
                out_lines.append(
                    f"{row['wav']}={row['alias']},{row['offset']:.2f},{row['cons']:.2f},{row['cutoff']:.2f},{row['pre']:.2f},{row['ovl']:.2f}"
                )
                continue

            weight = _profile_weight(float(stat.get("n", 0)))
            offset = row["offset"]
            pre_t = _clamp(stat.get("pre", row["pre"]), 0.0, 360.0)
            pre = _blend(row["pre"], pre_t, weight)
            cons_gap_now = max(row["cons"] - row["pre"], 10.0)
            cons_gap_t = _clamp(stat.get("cons_gap", cons_gap_now), 10.0, 220.0)
            cons = pre + _blend(cons_gap_now, cons_gap_t, weight)
            cut_gap_now = max(abs(row["cutoff"]) - row["cons"], 20.0)
            cut_gap_t = _clamp(stat.get("cut_gap", cut_gap_now), 20.0, 180.0)
            cut_gap = _blend(cut_gap_now, cut_gap_t, min(0.12, weight * 0.35))
            cutoff = -(cons + cut_gap)
            ovl_ratio_now = _safe_ratio(row["ovl"], row["pre"], fallback=0.0)
            ovl_ratio_t = _clamp(stat.get("ovl_ratio", ovl_ratio_now), 0.05, 0.78)
            ovl_ratio = _blend(ovl_ratio_now, ovl_ratio_t, weight)
            ovl = max(0.0, pre * max(0.10, min(0.80, ovl_ratio)))
            pre, cons, cutoff, ovl = _apply_kr_consonant_timing_shaping(
                row.get("alias", ""), pre, cons, cutoff, ovl
            )

            if idx == 0 and dur_ms > 0 and stat.get("head_off_ratio") is not None:
                target_ratio = _clamp(stat["head_off_ratio"], 0.0, 0.35)
                target_off = max(0.0, min(dur_ms * 0.7, dur_ms * target_ratio))
                offset = _blend(offset, target_off, min(0.30, weight))

            offset, cons, cutoff, pre, ovl = _validate_oto_params(offset, cons, cutoff, pre, ovl)
            if (
                abs(offset - row["offset"]) > 1e-6
                or abs(cons - row["cons"]) > 1e-6
                or abs(cutoff - row["cutoff"]) > 1e-6
                or abs(pre - row["pre"]) > 1e-6
                or abs(ovl - row["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _retarget_kr_bridge_to_next_cv(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    alias_type,
    alias_text="",
    next_row=None,
):
    alias_kind = str(alias_type or "").strip().lower()
    if alias_kind not in {"vc", "vv"} or not next_row:
        return _validate_oto_params(offset, consonant, cutoff, pre, ovl)

    next_offset = float(next_row.get("offset", 0.0))
    next_pre = max(float(next_row.get("pre", 0.0)), 0.0)
    next_cons = max(float(next_row.get("cons", next_pre)), next_pre)
    next_pre_abs = next_offset + next_pre
    next_cons_abs = next_offset + next_cons
    if next_pre_abs <= 0.0:
        return _validate_kr_bridge_params(offset, consonant, abs(cutoff), pre, ovl)

    curr_pre = max(float(pre), 0.0)
    curr_cons = max(float(consonant), curr_pre)
    curr_cut_abs = max(abs(float(cutoff)), curr_cons + 18.0)
    curr_ovl = max(float(ovl), 0.0)

    coda = _canonicalize_kr_coda(_extract_vc_right_token(alias_text))
    is_stop = coda in {"k", "t", "p", "h"}
    is_liquid = coda in {"l", "r"}
    is_nasal = coda in {"n", "m", "ng"}

    if alias_kind == "vc":
        if is_stop:
            pre_lo, pre_hi, pre_mix = 34.0, 112.0, 0.40
            ovl_gap_lo, ovl_gap_hi, ovl_gap_t = 10.0, 22.0, 14.0
            cons_gap_lo, cons_gap_hi, cons_gap_t = 18.0, 48.0, 30.0
            cut_gap_lo, cut_gap_hi, cut_gap_t = 8.0, 20.0, 12.0
            cut_allow_pre, cut_allow_cons = 8.0, 14.0
        elif is_liquid:
            pre_lo, pre_hi, pre_mix = 58.0, 188.0, 0.34
            ovl_gap_lo, ovl_gap_hi, ovl_gap_t = 7.0, 18.0, 10.0
            cons_gap_lo, cons_gap_hi, cons_gap_t = 44.0, 108.0, 70.0
            cut_gap_lo, cut_gap_hi, cut_gap_t = 14.0, 42.0, 26.0
            cut_allow_pre, cut_allow_cons = 22.0, 38.0
        elif is_nasal:
            pre_lo, pre_hi, pre_mix = 48.0, 154.0, 0.36
            ovl_gap_lo, ovl_gap_hi, ovl_gap_t = 8.0, 20.0, 12.0
            cons_gap_lo, cons_gap_hi, cons_gap_t = 30.0, 84.0, 50.0
            cut_gap_lo, cut_gap_hi, cut_gap_t = 12.0, 32.0, 20.0
            cut_allow_pre, cut_allow_cons = 18.0, 28.0
        else:
            pre_lo, pre_hi, pre_mix = 42.0, 140.0, 0.36
            ovl_gap_lo, ovl_gap_hi, ovl_gap_t = 8.0, 20.0, 12.0
            cons_gap_lo, cons_gap_hi, cons_gap_t = 24.0, 72.0, 42.0
            cut_gap_lo, cut_gap_hi, cut_gap_t = 10.0, 28.0, 18.0
            cut_allow_pre, cut_allow_cons = 16.0, 24.0
    else:
        pre_lo, pre_hi, pre_mix = 36.0, 184.0, 0.32
        ovl_gap_lo, ovl_gap_hi, ovl_gap_t = 4.0, 12.0, 8.0
        cons_gap_lo, cons_gap_hi, cons_gap_t = 58.0, 156.0, 96.0
        cut_gap_lo, cut_gap_hi, cut_gap_t = 18.0, 98.0, 56.0
        cut_allow_pre, cut_allow_cons = 28.0, 54.0

    next_cons_gap = max(next_cons - next_pre, 10.0)
    pre_new = _clamp(_blend(curr_pre, min(next_pre, pre_hi), pre_mix), pre_lo, pre_hi)
    offset_new = max(next_pre_abs - pre_new, 0.0)

    ovl_gap_now = max(curr_pre - curr_ovl, 0.0)
    ovl_gap_new = _clamp(_blend(ovl_gap_now, ovl_gap_t, 0.34), ovl_gap_lo, ovl_gap_hi)
    ovl_new = max(0.0, pre_new - ovl_gap_new)

    cons_gap_now = max(curr_cons - curr_pre, 8.0)
    cons_gap_seed = _blend(cons_gap_now, next_cons_gap, 0.28)
    cons_gap_new = _clamp(_blend(cons_gap_seed, cons_gap_t, 0.26), cons_gap_lo, cons_gap_hi)
    cons_new = pre_new + cons_gap_new

    cut_gap_now = max(curr_cut_abs - curr_cons, 10.0)
    cut_gap_new = _clamp(_blend(cut_gap_now, cut_gap_t, 0.28), cut_gap_lo, cut_gap_hi)
    cutoff_abs_new = cons_new + cut_gap_new

    next_pre_rel = max(next_pre_abs - offset_new, pre_new + 8.0)
    next_cons_rel = max(next_cons_abs - offset_new, next_pre_rel + 10.0)
    cutoff_cap = min(next_pre_rel + cut_allow_pre, next_cons_rel + cut_allow_cons)
    cutoff_floor = max(cons_new + cut_gap_lo, pre_new + 20.0)
    cutoff_abs_new = min(cutoff_abs_new, max(cutoff_floor, cutoff_cap))
    return _validate_kr_bridge_params(
        offset_new,
        cons_new,
        cutoff_abs_new,
        pre_new,
        ovl_new,
        min_cons_gap=max(12.0, cons_gap_lo),
        min_cut_gap=max(8.0, cut_gap_lo),
    )


def _apply_kr_bridge_coherence_to_oto_file(oto_path, custom_map=None):
    if not oto_path or not os.path.exists(oto_path):
        return 0

    rows_by_wav = {}
    line_order = []
    for raw in _read_text_with_fallback(oto_path).splitlines():
        row = _parse_oto_line_profile(raw)
        if not row:
            line_order.append(("raw", raw.rstrip("\n")))
            continue
        rows_by_wav.setdefault(row["wav"], []).append(row)
        line_order.append(("row", row["wav"], len(rows_by_wav[row["wav"]]) - 1))

    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    changed = 0
    for _wav_name, rows in rows_by_wav.items():
        row_types = [_classify_cached(row["alias"]) for row in rows]
        for idx, row in enumerate(rows):
            if row_types[idx] not in {"vc", "vv"}:
                continue
            next_row = None
            for j in range(idx + 1, len(rows)):
                if row_types[j] in {"cv", "cv_head", "vcv", "mono"}:
                    next_row = rows[j]
                    break
            if not next_row:
                continue

            new_vals = _retarget_kr_bridge_to_next_cv(
                row["offset"],
                row["cons"],
                row["cutoff"],
                row["pre"],
                row["ovl"],
                alias_type=row_types[idx],
                alias_text=row["alias"],
                next_row=next_row,
            )
            if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(new_vals, (row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"]))):
                row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"] = new_vals
                changed += 1

    if changed <= 0:
        return 0

    with open(oto_path, "w", encoding="utf-8") as f:
        for item in line_order:
            if item[0] == "raw":
                f.write(item[1] + "\n")
                continue
            row = rows_by_wav[item[1]][item[2]]
            f.write(
                f"{row['wav']}={row['alias']},{row['offset']:.2f},{row['cons']:.2f},{row['cutoff']:.2f},{row['pre']:.2f},{row['ovl']:.2f}\n"
            )
    return changed


__all__ = [
    "_apply_kr_bridge_coherence_to_oto_file",
    "_apply_kr_profile_to_oto_file",
    "_extract_base_timing_shape",
    "_occurrence_map",
    "_parse_oto_line_profile",
    "_read_kr_oto_rows_for_profile",
    "_read_text_with_fallback",
    "_resolve_kr_reference_dirs",
    "_retarget_kr_bridge_to_next_cv",
    "apply_kr_autotune_profile_to_oto",
    "load_kr_autotune_profile",
    "save_kr_autotune_profile",
    "train_kr_autotune_profile",
]
