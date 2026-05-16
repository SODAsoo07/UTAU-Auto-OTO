from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from statistics import median
from typing import Callable, Optional

from core.ja_oto_mapping import classify_ja_alias
from core.lab_generator import load_custom_phonemes
from core.oto_file_utils import (
    build_occurrence_map,
    parse_oto_line,
    read_oto_rows_for_profile,
    read_text_with_fallback,
)
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key


@dataclass(frozen=True)
class JaAutotuneRefreshContext:
    out_path: str
    profile_path: str
    had_profile_on_start: bool
    custom_map: object
    log_fn: object
    validate_fn: Optional[Callable[[float, float, float, float, float], tuple[float, float, float, float, float]]] = None


def _fallback_validate_oto_params(offset, consonant, cutoff, pre, ovl):
    offset_v = max(float(offset), 0.0)
    pre_v = max(float(pre), 0.0)
    ovl_v = max(0.0, float(ovl))
    consonant_v = max(float(consonant), 0.0)
    if ovl_v > pre_v:
        ovl_v = pre_v * 0.75
    if consonant_v < pre_v:
        consonant_v = pre_v + 20.0
    cutoff_abs = abs(float(cutoff))
    if cutoff_abs <= consonant_v:
        cutoff_abs = consonant_v + 40.0
    return offset_v, consonant_v, -cutoff_abs, pre_v, ovl_v


def _median(vals):
    if not vals:
        return 0.0
    return float(median(float(v) for v in vals))


def _clamp(v, lim):
    return max(-float(lim), min(float(lim), float(v)))


def _normalize_alias_for_profile(alias):
    return canonicalize_alias_for_matching("japanese", alias)


def _profile_path_for_out(out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path or "")) or os.getcwd()
    return os.path.join(out_dir, ".ja_oto_autotune_profile.json")


def _find_reference_oto(out_path):
    env_ref = os.environ.get("UTOA_JA_REF_OTO", "").strip()
    if env_ref and os.path.exists(env_ref):
        if os.path.abspath(env_ref) != os.path.abspath(out_path):
            return env_ref

    out_dir = os.path.dirname(os.path.abspath(out_path or "")) or os.getcwd()
    candidates = [
        "base_oto.ini",
        "oto.base.ini",
        "oto.manual.ini",
        "oto.correct.ini",
        "oto.reference.ini",
        "oto.gold.ini",
        "oto.human.ini",
        "oto_old.ini",
    ]
    for name in candidates:
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(out_path):
            return path
    return ""


def _read_oto_rows_for_profile(path):
    return read_oto_rows_for_profile(
        path,
        wav_normalizer=normalize_wav_key,
        alias_normalizer=_normalize_alias_for_profile,
    )


def _train_ja_autotune_profile(auto_oto_path, ref_oto_path, custom_map=None):
    auto_rows = _read_oto_rows_for_profile(auto_oto_path)
    ref_rows = _read_oto_rows_for_profile(ref_oto_path)
    if not auto_rows or not ref_rows:
        return None

    auto_map = build_occurrence_map(auto_rows)
    ref_map = build_occurrence_map(ref_rows)

    fields = ["offset", "cons", "cutoff", "pre", "ovl"]
    buckets = {}
    matched = 0
    alias_type_cache = {}

    def _classify_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_ja_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    for key, auto_row in auto_map.items():
        ref_row = ref_map.get(key)
        if not ref_row:
            continue
        alias_type = _classify_cached(auto_row["alias"])
        bucket = buckets.setdefault(alias_type, {field: [] for field in fields})
        for field in fields:
            bucket[field].append(float(ref_row[field]) - float(auto_row[field]))
        matched += 1

    if matched < 8:
        return None

    profile = {
        "version": 1,
        "language": "japanese",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": matched,
        "source_auto_name": os.path.basename(auto_oto_path),
        "source_ref_name": os.path.basename(ref_oto_path),
        "buckets": {},
    }
    for alias_type, vals in buckets.items():
        n = len(vals["offset"])
        if n < 3:
            continue
        profile["buckets"][alias_type] = {
            "n": n,
            "offset": _median(vals["offset"]),
            "cons": _median(vals["cons"]),
            "cutoff": _median(vals["cutoff"]),
            "pre": _median(vals["pre"]),
            "ovl": _median(vals["ovl"]),
        }

    if not profile["buckets"]:
        return None
    return profile


def train_ja_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    custom_map = load_custom_phonemes(custom_phonemes_path)
    return _train_ja_autotune_profile(auto_oto_path, manual_oto_path, custom_map=custom_map)


def load_ja_autotune_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "buckets" not in data:
            return None
        return data
    except Exception:
        return None


def save_ja_autotune_profile(path, profile):
    if not path or not profile:
        return False
    try:
        data = dict(profile)
        data.pop("source_auto", None)
        data.pop("source_ref", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def apply_ja_autotune_delta(
    alias_type,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    profile,
    *,
    validate_fn: Optional[Callable[[float, float, float, float, float], tuple[float, float, float, float, float]]] = None,
):
    if not profile:
        return offset, consonant, cutoff, pre, ovl
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict):
        return offset, consonant, cutoff, pre, ovl
    stat = buckets.get(alias_type)
    if not stat:
        return offset, consonant, cutoff, pre, ovl
    n = float(stat.get("n", 0))
    if n < 3:
        return offset, consonant, cutoff, pre, ovl

    validate = validate_fn or _fallback_validate_oto_params
    weight = min(1.0, max(0.25, n / 24.0))
    offset += _clamp(stat.get("offset", 0.0), 140.0) * weight
    consonant += _clamp(stat.get("cons", 0.0), 160.0) * weight
    cutoff += _clamp(stat.get("cutoff", 0.0), 180.0) * weight
    pre += _clamp(stat.get("pre", 0.0), 140.0) * weight
    ovl += _clamp(stat.get("ovl", 0.0), 100.0) * weight
    return validate(offset, consonant, cutoff, pre, ovl)


def _apply_profile_to_oto_file(
    oto_path,
    profile,
    *,
    custom_map=None,
    validate_fn: Optional[Callable[[float, float, float, float, float], tuple[float, float, float, float, float]]] = None,
):
    if not oto_path or not os.path.exists(oto_path):
        return 0
    changed = 0
    out_lines = []
    alias_type_cache = {}
    validate = validate_fn or _fallback_validate_oto_params

    def _classify_cached(alias_text):
        alias_norm = _normalize_alias_for_profile(alias_text)
        alias_type = alias_type_cache.get(alias_norm)
        if alias_type is None:
            alias_type = classify_ja_alias(alias_norm, custom_map)
            alias_type_cache[alias_norm] = alias_type
        return alias_type

    for raw in read_text_with_fallback(oto_path).splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            out_lines.append(raw.rstrip("\n"))
            continue
        alias_type = _classify_cached(parsed["alias"])
        o2, c2, ct2, pr2, ov2 = apply_ja_autotune_delta(
            alias_type,
            parsed["offset"],
            parsed["cons"],
            parsed["cutoff"],
            parsed["pre"],
            parsed["ovl"],
            profile,
            validate_fn=validate,
        )
        if (
            abs(o2 - parsed["offset"]) > 1e-6
            or abs(c2 - parsed["cons"]) > 1e-6
            or abs(ct2 - parsed["cutoff"]) > 1e-6
            or abs(pr2 - parsed["pre"]) > 1e-6
            or abs(ov2 - parsed["ovl"]) > 1e-6
        ):
            changed += 1
        out_lines.append(f"{parsed['wav']}={parsed['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}")

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def apply_ja_autotune_profile_to_oto(
    oto_path,
    profile,
    custom_phonemes_path="",
    *,
    validate_fn: Optional[Callable[[float, float, float, float, float], tuple[float, float, float, float, float]]] = None,
):
    if isinstance(profile, str):
        profile = load_ja_autotune_profile(profile)
    if not profile:
        return 0
    custom_map = load_custom_phonemes(custom_phonemes_path)
    return _apply_profile_to_oto_file(
        oto_path,
        profile,
        custom_map=custom_map,
        validate_fn=validate_fn,
    )


def refresh_ja_autotune_profile_after_generation(context: JaAutotuneRefreshContext):
    ref_oto = _find_reference_oto(context.out_path)
    if not ref_oto:
        return None

    trained = _train_ja_autotune_profile(context.out_path, ref_oto, custom_map=context.custom_map)
    if not trained or not save_ja_autotune_profile(context.profile_path, trained):
        context.log_fn("[AutoTune] reference OTO found but matched samples were insufficient for profile update.")
        return None

    bucket_count = len((trained.get("buckets") or {}))
    pair_count = int(trained.get("matched_pairs", 0))
    context.log_fn(f"[AutoTune] profile updated: pairs={pair_count}, buckets={bucket_count}")
    context.log_fn(f"[AutoTune] profile saved: {context.profile_path}")

    if not context.had_profile_on_start:
        changed = _apply_profile_to_oto_file(
            context.out_path,
            trained,
            custom_map=context.custom_map,
            validate_fn=context.validate_fn,
        )
        if changed > 0:
            context.log_fn(f"[AutoTune] bootstrap profile applied: {changed} lines adjusted")

    return trained


__all__ = [
    "JaAutotuneRefreshContext",
    "_apply_profile_to_oto_file",
    "_find_reference_oto",
    "_profile_path_for_out",
    "_train_ja_autotune_profile",
    "apply_ja_autotune_delta",
    "apply_ja_autotune_profile_to_oto",
    "load_ja_autotune_profile",
    "refresh_ja_autotune_profile_after_generation",
    "save_ja_autotune_profile",
    "train_ja_autotune_profile",
]
