from __future__ import annotations

import audioop
import contextlib
import math
import os
import wave
from statistics import median
from typing import Callable

from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.oto_normalization import normalize_wav_key

_CLASSIFY_ALIAS_TYPE_FN = None


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


def _normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _apply_suffix_to_oto_line(line: str, suffix: str) -> str:
    normalized_suffix = _normalize_alias_suffix(suffix)
    if not normalized_suffix or "=" not in str(line or ""):
        return line
    left, right = str(line).split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        alias = alias.strip()
        if alias:
            alias = f"{alias}_{normalized_suffix}"
        return f"{left}={alias},{rest}"
    alias = right.strip()
    if alias:
        alias = f"{alias}_{normalized_suffix}"
    return f"{left}={alias}"


def _pick_existing_oto(path: str) -> str:
    base = str(path or "").strip()
    if not base:
        return ""
    if os.path.isfile(base):
        return os.path.abspath(base)
    if os.path.isdir(base):
        for name in ("baseoto.ini", "base_oto.ini", "oto.ini"):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return ""


def _pick_existing_reference_oto(path: str) -> str:
    base = str(path or "").strip()
    if not base:
        return ""
    if os.path.isfile(base):
        return os.path.abspath(base)
    if os.path.isdir(base):
        for name in (
            "oto.correct.ini",
            "oto.reference.ini",
            "oto.manual.ini",
            "oto.gold.ini",
            "oto.human.ini",
            "oto.ini",
            "base_oto.ini",
            "baseoto.ini",
        ):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return ""


def resolve_no_mfa_source_oto(*, wav_dir: str, source_hint: str = "") -> str:
    candidates: list[str] = []
    hint = _pick_existing_oto(source_hint)
    if hint:
        candidates.append(hint)

    env_oto = _pick_existing_oto(os.environ.get("UTOA_NO_MFA_SOURCE_OTO", ""))
    if env_oto:
        candidates.append(env_oto)

    env_bank = _pick_existing_oto(os.environ.get("UTOA_NO_MFA_SOURCE_BANK", ""))
    if env_bank:
        candidates.append(env_bank)

    wav_dir_abs = os.path.abspath(str(wav_dir or "").strip())
    if wav_dir_abs and os.path.isdir(wav_dir_abs):
        for local_name in ("baseoto.ini", "base_oto.ini", "oto.ini"):
            local_oto = os.path.join(wav_dir_abs, local_name)
            if os.path.isfile(local_oto):
                candidates.append(os.path.abspath(local_oto))

    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        return candidate
    return ""


def resolve_no_mfa_stats_oto(*, stats_hint: str = "") -> str:
    candidates: list[str] = []
    hint = _pick_existing_reference_oto(stats_hint)
    if hint:
        candidates.append(hint)

    env_ref = _pick_existing_reference_oto(os.environ.get("UTOA_NO_MFA_STATS_OTO", ""))
    if env_ref:
        candidates.append(env_ref)

    env_ref2 = _pick_existing_reference_oto(os.environ.get("UTOA_NO_MFA_REFERENCE_OTO", ""))
    if env_ref2:
        candidates.append(env_ref2)

    seen: set[str] = set()
    for candidate in candidates:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        return candidate
    return ""


def _load_oto_lines(source_oto_path: str) -> list[str]:
    path = str(source_oto_path or "").strip()
    if not path:
        return []
    for encoding in ("utf-8-sig", "cp932", "cp949", "euc-kr", "utf-8"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                rows = []
                for raw in handle:
                    line = str(raw or "").strip()
                    if not line or "=" not in line:
                        continue
                    rows.append(line)
                return rows
        except Exception:
            continue
    return []


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _normalize_generation_mode(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"alias_auto", "alias-only", "alias_only", "blank_auto", "blank"}:
        return "alias_auto"
    if raw in {"remap", "base_remap", "base"}:
        return "remap"
    return "remap"


def _median_of(values: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    try:
        return float(median(float(v) for v in values))
    except Exception:
        return float(default)


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


def _timing_row_is_zeroish(parsed: dict[str, float]) -> bool:
    try:
        return (
            abs(float(parsed.get("offset", 0.0))) <= 1e-6
            and abs(float(parsed.get("cons", 0.0))) <= 1e-6
            and abs(float(parsed.get("cutoff", 0.0))) <= 1e-6
            and abs(float(parsed.get("pre", 0.0))) <= 1e-6
            and abs(float(parsed.get("ovl", 0.0))) <= 1e-6
        )
    except Exception:
        return False


def _build_builtin_timing_profile(language: str) -> dict[str, object] | None:
    lang = str(language or "").strip().lower() or "korean"
    preset_buckets: dict[str, dict[str, float]] = {}

    try:
        if lang == "japanese":
            from core.oto_profile_presets import get_ja_profile_preset

            preset = get_ja_profile_preset("cvvc")
        else:
            from core.oto_profile_presets import get_kr_profile_preset

            preset = get_kr_profile_preset("cvc")
        preset_buckets = dict((preset or {}).get("buckets") or {})
    except Exception:
        preset_buckets = {}

    if not preset_buckets:
        if lang == "japanese":
            preset_buckets = {
                "cv_head": {"pre": 92.0, "cons_gap": 102.0, "cut_gap": 142.0, "ovl_ratio": 0.39, "head_off_ratio": 0.12},
                "cv": {"pre": 56.0, "cons_gap": 108.0, "cut_gap": 142.0, "ovl_ratio": 0.28},
                "vc": {"pre": 214.0, "cons_gap": 31.0, "cut_gap": 31.0, "ovl_ratio": 0.33},
                "vv": {"pre": 248.0, "cons_gap": 131.0, "cut_gap": 176.0, "ovl_ratio": 0.33},
            }
        else:
            preset_buckets = {
                "cv_head": {"pre": 77.0, "cons_gap": 72.0, "cut_gap": 180.0, "ovl_ratio": 0.33, "head_off_ratio": 0.18},
                "cv": {"pre": 80.0, "cons_gap": 83.0, "cut_gap": 151.0, "ovl_ratio": 0.40},
                "vcv": {"pre": 127.0, "cons_gap": 74.0, "cut_gap": 49.0, "ovl_ratio": 0.49},
                "vc": {"pre": 119.0, "cons_gap": 46.0, "cut_gap": 74.0, "ovl_ratio": 0.50},
                "vv": {"pre": 167.0, "cons_gap": 102.0, "cut_gap": 180.0, "ovl_ratio": 0.64},
            }

    profile_buckets: dict[str, dict[str, float]] = {}
    for alias_type, stat in preset_buckets.items():
        if not isinstance(stat, dict):
            continue
        pre = max(float(stat.get("pre", 95.0) or 95.0), 20.0)
        cons_gap = max(float(stat.get("cons_gap", 60.0) or 60.0), 12.0)
        cut_gap = max(float(stat.get("cut_gap", 60.0) or 60.0), 24.0)
        ovl_ratio = max(0.05, min(0.82, float(stat.get("ovl_ratio", 0.35) or 0.35)))
        offset = 0.0
        if alias_type == "cv_head":
            try:
                offset = max(0.0, pre * float(stat.get("head_off_ratio", 0.12) or 0.12))
            except Exception:
                offset = 0.0
        cons = pre + cons_gap
        cutoff = -(cons + cut_gap)
        ovl = max(0.0, pre * ovl_ratio)
        o2, c2, ct2, pr2, ov2 = _fallback_validate_oto_params(offset, cons, cutoff, pre, ovl)
        profile_buckets[str(alias_type).strip().lower() or "other"] = {
            "n": 9999.0,
            "offset": float(o2),
            "cons": float(c2),
            "cutoff": float(ct2),
            "pre": float(pr2),
            "ovl": float(ov2),
        }

    if not profile_buckets:
        return None

    seed = profile_buckets.get("cv") or profile_buckets.get("cv_head") or next(iter(profile_buckets.values()))
    for alias_type in ("other", "mono"):
        if alias_type not in profile_buckets:
            profile_buckets[alias_type] = dict(seed)

    global_values = {
        "offset": [float(v["offset"]) for v in profile_buckets.values()],
        "cons": [float(v["cons"]) for v in profile_buckets.values()],
        "cutoff": [float(v["cutoff"]) for v in profile_buckets.values()],
        "pre": [float(v["pre"]) for v in profile_buckets.values()],
        "ovl": [float(v["ovl"]) for v in profile_buckets.values()],
    }
    return {
        "source": f"builtin:{lang}",
        "language": lang,
        "rows": float(len(profile_buckets)),
        "min_bucket_n": 1.0,
        "global": {
            "n": float(len(profile_buckets)),
            "offset": _median_of(global_values["offset"], 0.0),
            "cons": _median_of(global_values["cons"], 120.0),
            "cutoff": _median_of(global_values["cutoff"], -220.0),
            "pre": _median_of(global_values["pre"], 95.0),
            "ovl": _median_of(global_values["ovl"], 35.0),
        },
        "buckets": profile_buckets,
    }


def _classify_alias_type(language: str, alias: str) -> str:
    global _CLASSIFY_ALIAS_TYPE_FN
    if _CLASSIFY_ALIAS_TYPE_FN is None:
        try:
            from core.oto_ml_features import classify_alias_type as _classify_alias_type_core

            _CLASSIFY_ALIAS_TYPE_FN = _classify_alias_type_core
        except Exception:
            _CLASSIFY_ALIAS_TYPE_FN = False

    alias_text = str(alias or "").strip()
    lang = str(language or "").strip().lower() or "korean"
    if not alias_text:
        return "other"

    if _CLASSIFY_ALIAS_TYPE_FN:
        try:
            value = str(_CLASSIFY_ALIAS_TYPE_FN(lang, alias_text)).strip().lower()
            if value:
                return value
        except Exception:
            pass

    if alias_text.startswith("- "):
        return "cv_head"
    if " " in alias_text:
        left = alias_text.split(" ", 1)[0].strip()
        return "vv" if left in {"a", "i", "u", "e", "o"} else "vc"
    if alias_text in {"R", "br", "breath"}:
        return "br"
    return "cv"


def _build_no_mfa_timing_profile(
    *,
    language: str,
    reference_oto_path: str,
) -> dict[str, object] | None:
    path = str(reference_oto_path or "").strip()
    if not path or not os.path.isfile(path):
        return None

    min_bucket_n = max(2, _env_int("UTOA_NO_MFA_STATS_MIN_BUCKET", 5))
    fields = ("offset", "cons", "cutoff", "pre", "ovl")
    global_values = {field: [] for field in fields}
    buckets: dict[str, dict[str, list[float]]] = {}
    total_rows = 0

    for raw in read_text_with_fallback(path).splitlines():
        row = parse_oto_line(raw)
        if not row:
            continue
        total_rows += 1
        alias_type = _classify_alias_type(language, str(row.get("alias", "")))
        bucket = buckets.setdefault(alias_type, {field: [] for field in fields})
        for field in fields:
            value = float(row.get(field, 0.0))
            global_values[field].append(value)
            bucket[field].append(value)

    if total_rows < 8:
        return None

    profile_buckets: dict[str, dict[str, float]] = {}
    for alias_type, values in buckets.items():
        n = len(values["offset"])
        profile_buckets[alias_type] = {
            "n": float(n),
            "offset": _median_of(values["offset"], 0.0),
            "cons": _median_of(values["cons"], 120.0),
            "cutoff": _median_of(values["cutoff"], -220.0),
            "pre": _median_of(values["pre"], 95.0),
            "ovl": _median_of(values["ovl"], 35.0),
        }

    return {
        "source": os.path.abspath(path),
        "language": str(language or "").strip().lower() or "korean",
        "rows": float(total_rows),
        "min_bucket_n": float(min_bucket_n),
        "global": {
            "n": float(total_rows),
            "offset": _median_of(global_values["offset"], 0.0),
            "cons": _median_of(global_values["cons"], 120.0),
            "cutoff": _median_of(global_values["cutoff"], -220.0),
            "pre": _median_of(global_values["pre"], 95.0),
            "ovl": _median_of(global_values["ovl"], 35.0),
        },
        "buckets": profile_buckets,
    }


def _apply_timing_profile_to_oto_line(
    *,
    line: str,
    profile: dict[str, object] | None,
    language: str,
    only_when_zero: bool = False,
) -> tuple[str, bool]:
    if not profile:
        return line, False

    parsed = parse_oto_line(line)
    if not parsed:
        return line, False
    if only_when_zero and not _timing_row_is_zeroish(parsed):
        return line, False

    alias_type = _classify_alias_type(language, str(parsed.get("alias", "")))
    buckets = profile.get("buckets") if isinstance(profile, dict) else None
    selected = None
    if isinstance(buckets, dict):
        stat = buckets.get(alias_type)
        if isinstance(stat, dict):
            n = int(float(stat.get("n", 0.0) or 0.0))
            min_bucket_n = int(float(profile.get("min_bucket_n", 5.0) or 5.0))
            if n >= max(1, min_bucket_n):
                selected = stat
    if selected is None:
        selected = profile.get("global") if isinstance(profile, dict) else None
    if not isinstance(selected, dict):
        return line, False

    o2, c2, ct2, pr2, ov2 = _fallback_validate_oto_params(
        selected.get("offset", parsed.get("offset", 0.0)),
        selected.get("cons", parsed.get("cons", 0.0)),
        selected.get("cutoff", parsed.get("cutoff", -120.0)),
        selected.get("pre", parsed.get("pre", 0.0)),
        selected.get("ovl", parsed.get("ovl", 0.0)),
    )
    changed = (
        abs(float(o2) - float(parsed.get("offset", 0.0))) > 1e-6
        or abs(float(c2) - float(parsed.get("cons", 0.0))) > 1e-6
        or abs(float(ct2) - float(parsed.get("cutoff", 0.0))) > 1e-6
        or abs(float(pr2) - float(parsed.get("pre", 0.0))) > 1e-6
        or abs(float(ov2) - float(parsed.get("ovl", 0.0))) > 1e-6
    )
    rewritten = f"{parsed['wav']}={parsed['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}"
    return rewritten, changed


def _build_wav_lookup(wav_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    root = os.path.abspath(str(wav_dir or "").strip())
    if not os.path.isdir(root):
        return exact, normalized
    for current_root, _dirs, files in os.walk(root):
        for file_name in files:
            if not str(file_name).lower().endswith(".wav"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            key = str(file_name).strip().lower()
            exact.setdefault(key, rel_path)
            norm = normalize_wav_key(file_name)
            if norm:
                normalized.setdefault(norm, rel_path)
    return exact, normalized


def _read_wav_duration_ms(path: str) -> float:
    try:
        with wave.open(path, "rb") as wav_handle:
            frames = int(wav_handle.getnframes() or 0)
            sample_rate = int(wav_handle.getframerate() or 0)
        if frames <= 0 or sample_rate <= 0:
            return 0.0
        return (float(frames) * 1000.0) / float(sample_rate)
    except Exception:
        return 0.0


def _estimate_slot_timing(
    *,
    wav_duration_ms: float,
    slot_index: int,
    slot_total: int,
    alias_type: str,
) -> tuple[float, float, float, float, float]:
    total_slots = max(int(slot_total), 1)
    idx = max(0, min(int(slot_index), total_slots - 1))
    duration_ms = max(float(wav_duration_ms), 200.0)
    usable_start = min(35.0, duration_ms * 0.12)
    usable_end = max(usable_start + 120.0, duration_ms - 45.0)
    if usable_end <= usable_start:
        usable_start = 0.0
        usable_end = max(duration_ms, 220.0)
    slot_len = max((usable_end - usable_start) / float(total_slots), 42.0)

    alias_key = str(alias_type or "").strip().lower()
    ratios = {
        "cv_head": (0.56, 0.46, 0.82, 0.42, 0.22),
        "cv": (0.48, 0.40, 0.78, 0.38, 0.10),
        "vcv": (0.58, 0.36, 0.66, 0.52, 0.08),
        "vc": (0.62, 0.30, 0.56, 0.55, 0.08),
        "vv": (0.66, 0.36, 0.70, 0.58, 0.06),
        "mono": (0.50, 0.40, 0.74, 0.40, 0.10),
        "br": (0.30, 0.25, 0.45, 0.24, 0.05),
        "other": (0.50, 0.38, 0.72, 0.36, 0.10),
    }.get(alias_key, (0.50, 0.38, 0.72, 0.36, 0.10))
    pre_ratio, cons_ratio, cut_ratio, ovl_ratio, off_ratio = ratios

    slot_start = usable_start + (float(idx) * slot_len)
    offset = max(0.0, slot_start - (slot_len * off_ratio))
    pre = max(35.0, min(230.0, slot_len * pre_ratio))
    consonant = pre + max(20.0, min(220.0, slot_len * cons_ratio))
    cutoff = -(consonant + max(28.0, min(340.0, slot_len * cut_ratio)))
    overlap = max(0.0, min(pre * 0.86, pre * ovl_ratio))
    return _fallback_validate_oto_params(offset, consonant, cutoff, pre, overlap)


def _estimate_segment_boundaries_ms(wav_path: str, segment_count: int) -> list[float]:
    seg = int(segment_count or 0)
    if seg <= 1:
        duration_ms = _read_wav_duration_ms(wav_path)
        return [0.0, max(0.0, duration_ms)]
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wav_file:
            channels = int(wav_file.getnchannels() or 1)
            sampwidth = int(wav_file.getsampwidth() or 2)
            framerate = int(wav_file.getframerate() or 44100)
            nframes = int(wav_file.getnframes() or 0)
            if nframes <= 0 or framerate <= 0:
                raise ValueError("invalid wav header")
            raw = wav_file.readframes(nframes)
    except Exception:
        duration_ms = _read_wav_duration_ms(wav_path)
        if duration_ms <= 0.0:
            return [0.0] * (seg + 1)
        return [duration_ms * float(i) / float(seg) for i in range(seg + 1)]

    try:
        frame_bytes = max(1, channels * sampwidth)
        window_frames = max(1, int(framerate * 0.01))
        step_frames = window_frames
        total_windows = int(math.ceil(float(nframes) / float(step_frames)))
        if total_windows <= 0:
            raise ValueError("no windows")

        energy: list[float] = []
        for wi in range(total_windows):
            start_f = wi * step_frames
            end_f = min(nframes, start_f + window_frames)
            if end_f <= start_f:
                energy.append(0.0)
                continue
            start_b = start_f * frame_bytes
            end_b = end_f * frame_bytes
            chunk = raw[start_b:end_b]
            if channels > 1:
                chunk = audioop.tomono(chunk, sampwidth, 0.5, 0.5)
            energy.append(float(audioop.rms(chunk, sampwidth)))
        if not energy:
            raise ValueError("empty energy")

        smooth: list[float] = []
        radius = 2
        n = len(energy)
        for i in range(n):
            lo = max(0, i - radius)
            hi = min(n - 1, i + radius)
            smooth.append(sum(energy[lo : hi + 1]) / float(hi - lo + 1))

        duration_ms = float(nframes) * 1000.0 / float(framerate)
        boundary_idx = [0]
        prev = 0
        for i in range(1, seg):
            target = int(round(float(i) * float(n) / float(seg)))
            search_radius = max(3, int(round(float(n) / float(seg * 3))))
            lo = max(prev + 1, target - search_radius)
            hi = min(n - 2, target + search_radius)
            if hi < lo:
                cut = max(prev + 1, min(n - 2, target))
            else:
                local = smooth[lo : hi + 1]
                cut = lo + local.index(min(local))
            boundary_idx.append(cut)
            prev = cut
        boundary_idx.append(n - 1)

        step_ms = float(step_frames) * 1000.0 / float(framerate)
        out = [0.0]
        for i in range(1, len(boundary_idx) - 1):
            out.append(max(0.0, min(duration_ms, float(boundary_idx[i]) * step_ms)))
        out.append(duration_ms)
        if len(out) != seg + 1:
            raise ValueError("boundary count mismatch")
        for i in range(1, len(out)):
            if out[i] <= out[i - 1]:
                out[i] = min(duration_ms, out[i - 1] + 1.0)
        return out
    except Exception:
        duration_ms = _read_wav_duration_ms(wav_path)
        if duration_ms <= 0.0:
            return [0.0] * (seg + 1)
        return [duration_ms * float(i) / float(seg) for i in range(seg + 1)]


def _blend_value(base_value: float, target_value: float, weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    return (float(base_value) * (1.0 - w)) + (float(target_value) * w)


def _refine_remap_timing_for_slot(
    *,
    parsed: dict[str, float],
    boundaries_ms: list[float],
    slot_index: int,
    slot_total: int,
    language: str,
    blend_weight: float,
    format_type: str = "",
) -> tuple[str, bool]:
    if not parsed:
        return "", False
    total_slots = max(int(slot_total), 1)
    idx = max(0, min(int(slot_index), total_slots - 1))
    if len(boundaries_ms) < total_slots + 1:
        return "", False

    seg_start = float(boundaries_ms[idx])
    seg_end = float(boundaries_ms[idx + 1])
    wav_duration_ms = float(boundaries_ms[-1]) if boundaries_ms else 0.0
    if wav_duration_ms <= 0.0 or seg_end <= seg_start + 2.0:
        return "", False

    alias_type = _classify_alias_type(language, str(parsed.get("alias", "")))
    pre_ratio, cons_ratio, cut_ratio, ovl_ratio, off_ratio = {
        "cv_head": (0.56, 0.46, 0.82, 0.42, 0.22),
        "cv": (0.48, 0.40, 0.78, 0.38, 0.10),
        "vcv": (0.58, 0.36, 0.66, 0.52, 0.08),
        "vc": (0.62, 0.30, 0.56, 0.55, 0.08),
        "vv": (0.66, 0.36, 0.70, 0.58, 0.06),
        "mono": (0.50, 0.40, 0.74, 0.40, 0.10),
        "br": (0.30, 0.25, 0.45, 0.24, 0.05),
        "other": (0.50, 0.38, 0.72, 0.36, 0.10),
    }.get(str(alias_type or "").strip().lower(), (0.50, 0.38, 0.72, 0.36, 0.10))
    # CVVC 포맷에서 VC alias는 앞 CV와 바로 맞닿으므로 offset을 더 타이트하게 설정한다.
    fmt = str(format_type or "").strip().lower()
    if fmt == "cvvc" and str(alias_type or "").strip().lower() == "vc":
        off_ratio = 0.04

    seg_len = max(28.0, seg_end - seg_start)
    target_offset = max(0.0, seg_start - (seg_len * off_ratio))
    target_pre = max(24.0, min(260.0, seg_len * pre_ratio))
    target_cons = target_pre + max(16.0, min(250.0, seg_len * cons_ratio))
    right_blank_struct = target_cons + max(30.0, min(360.0, seg_len * cut_ratio))
    right_blank_boundary = max(6.0, wav_duration_ms - seg_end + (seg_len * 0.08))
    target_cutoff = -max(right_blank_struct, right_blank_boundary)
    target_ovl = max(0.0, min(target_pre * 0.86, target_pre * ovl_ratio))

    row_is_zero = _timing_row_is_zeroish(parsed)
    base_weight = 1.0 if row_is_zero else max(0.05, min(0.95, float(blend_weight)))

    base_offset = float(parsed.get("offset", 0.0))
    base_cons = float(parsed.get("cons", 0.0))
    base_cutoff = float(parsed.get("cutoff", 0.0))
    base_pre = float(parsed.get("pre", 0.0))
    base_ovl = float(parsed.get("ovl", 0.0))

    offset_weight = 1.0 if abs(base_offset) <= 1e-6 else base_weight
    if base_offset > seg_end + 2.0 or base_offset < max(0.0, seg_start - seg_len):
        offset_weight = max(offset_weight, 0.70)
    pre_weight = 1.0 if abs(base_pre) <= 1e-6 else base_weight
    cons_weight = 1.0 if abs(base_cons) <= 1e-6 else base_weight
    cutoff_weight = 1.0 if abs(base_cutoff) <= 1e-6 else base_weight
    ovl_weight = 1.0 if abs(base_ovl) <= 1e-6 else base_weight

    o2 = _blend_value(base_offset, target_offset, offset_weight)
    c2 = _blend_value(base_cons, target_cons, cons_weight)
    ct2 = _blend_value(base_cutoff, target_cutoff, cutoff_weight)
    pr2 = _blend_value(base_pre, target_pre, pre_weight)
    ov2 = _blend_value(base_ovl, target_ovl, ovl_weight)
    o2, c2, ct2, pr2, ov2 = _fallback_validate_oto_params(o2, c2, ct2, pr2, ov2)

    max_offset = max(0.0, min(wav_duration_ms - 1.0, seg_end - 2.0))
    o2 = max(0.0, min(max_offset, o2))
    rewritten = f"{parsed['wav']}={parsed['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}"
    changed = (
        abs(float(o2) - base_offset) > 1e-6
        or abs(float(c2) - base_cons) > 1e-6
        or abs(float(ct2) - base_cutoff) > 1e-6
        or abs(float(pr2) - base_pre) > 1e-6
        or abs(float(ov2) - base_ovl) > 1e-6
    )
    return rewritten, changed


def generate_no_mfa_auto_oto(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    alias_suffix: str = "",
    language: str = "",
    stats_oto_path: str = "",
    generation_mode: str = "remap",
    callback: Callable[[str], None] | None = None,
) -> tuple[int, int, list[str]]:
    mode = _normalize_generation_mode(generation_mode)
    _log(
        callback,
        "[No-MFA] generation mode: "
        + (
            "alias_auto (blank-alias template -> auto timing)"
            if mode == "alias_auto"
            else "remap (base timing remap + correction)"
        ),
    )
    source = resolve_no_mfa_source_oto(
        wav_dir=wav_dir,
        source_hint=source_oto_path,
    )
    if not source:
        return 0, 0, ["No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다."]

    source_rows = _load_oto_lines(source)
    if not source_rows:
        return 0, 0, [f"No-MFA 자동설정용 베이스 OTO를 읽지 못했습니다: {source}"]

    # 소스 OTO 에서 format_type을 자동 감지한다 (CVVC VC offset 비율 조정에 사용).
    _detected_format = ""
    try:
        lang_lower = str(language or "").strip().lower() or "korean"
        if lang_lower == "korean":
            from core.kr_oto_rules import detect_alias_format
            _src_aliases = []
            for _row in source_rows:
                if "=" in _row:
                    _right = _row.split("=", 1)[1]
                    _parts = _right.split(",", 1)
                    _src_aliases.append(str(_parts[0]).strip())
            if _src_aliases:
                _detected_format = str(detect_alias_format(_src_aliases) or "").strip().lower()
    except Exception:
        _detected_format = ""

    timing_profile = None
    timing_zero_only = mode == "alias_auto"
    remap_blend_weight = max(0.05, min(0.95, _env_float("UTOA_NO_MFA_REMAP_BLEND", 0.38)))
    stats_source = resolve_no_mfa_stats_oto(stats_hint=stats_oto_path)
    if not stats_source and str(stats_oto_path or "").strip():
        _log(callback, f"[No-MFA] stats oto not found: {stats_oto_path}")
    if stats_source:
        timing_profile = _build_no_mfa_timing_profile(
            language=language,
            reference_oto_path=stats_source,
        )
        if timing_profile:
            bucket_count = len((timing_profile.get("buckets") or {}))
            sample_count = int(float(timing_profile.get("rows", 0.0) or 0.0))
            _log(
                callback,
                f"[No-MFA] stats timing enabled: source={stats_source} rows={sample_count} buckets={bucket_count}",
            )
        else:
            _log(callback, f"[No-MFA] stats timing skipped: insufficient usable rows ({stats_source})")
    if timing_profile is None:
        timing_profile = _build_builtin_timing_profile(language=language)
        if timing_profile:
            timing_zero_only = True
            _log(
                callback,
                f"[No-MFA] zero-timing fallback enabled: source={timing_profile.get('source', 'builtin')} "
                "(0,0,0,0,0 rows -> wav slot boundary timing; profile fallback on failure)",
            )
    if mode == "remap":
        _log(
            callback,
            f"[No-MFA] remap acoustic refine enabled (blend={remap_blend_weight:.2f}; 0-values use full auto correction)",
        )
    force_boundary_timing = bool(mode == "alias_auto")
    if not force_boundary_timing:
        force_boundary_timing = bool(timing_zero_only and _env_bool("UTOA_NO_MFA_FORCE_BOUNDARY_TIMING", False))
    if force_boundary_timing:
        if mode == "alias_auto":
            _log(callback, "[No-MFA] alias-auto timing mode enabled (blank OTO base with boundary estimation).")
        else:
            _log(callback, "[No-MFA] forced boundary timing mode enabled (ignoring base OTO timing values).")

    exact_lookup, norm_lookup = _build_wav_lookup(wav_dir)
    if not exact_lookup and not norm_lookup:
        return 0, len(source_rows), [f"WAV 파일을 찾지 못했습니다: {wav_dir}"]

    out_lines: list[str] = []
    missing_wavs: set[str] = set()
    seen: set[tuple[str, str]] = set()
    exact_hits = 0
    norm_hits = 0
    timing_replaced = 0
    remap_refined = 0
    boundary_refined = 0
    profile_fallback_refined = 0
    total = 0
    prepared_entries: list[tuple[str, str]] = []

    for raw_line in source_rows:
        if "=" not in raw_line:
            continue
        total += 1
        left, right = raw_line.split("=", 1)
        source_wav_name = os.path.basename(str(left or "").replace("\\", "/").strip())
        source_wav_name = source_wav_name or str(left or "").strip()
        mapped_wav = exact_lookup.get(source_wav_name.lower(), "")
        if mapped_wav:
            exact_hits += 1
        else:
            mapped_wav = norm_lookup.get(normalize_wav_key(source_wav_name), "")
            if mapped_wav:
                norm_hits += 1
        if not mapped_wav:
            missing_wavs.add(source_wav_name)
            continue
        prepared_entries.append((mapped_wav, right.strip()))

    per_wav_total: dict[str, int] = {}
    for mapped_wav, _right in prepared_entries:
        per_wav_total[mapped_wav] = int(per_wav_total.get(mapped_wav, 0)) + 1
    per_wav_index: dict[str, int] = {}
    wav_duration_cache: dict[str, float] = {}
    wav_boundary_cache: dict[tuple[str, int], list[float]] = {}
    wav_root = os.path.abspath(str(wav_dir or "").strip())

    for mapped_wav, right in prepared_entries:
        candidate = f"{mapped_wav}={right}"
        slot_idx = int(per_wav_index.get(mapped_wav, 0))
        per_wav_index[mapped_wav] = slot_idx + 1

        parsed_candidate = parse_oto_line(candidate)
        if mode == "alias_auto" and parsed_candidate:
            candidate = f"{parsed_candidate['wav']}={parsed_candidate['alias']},0,0,0,0,0"
            parsed_candidate = parse_oto_line(candidate) or parsed_candidate

        timing_applied = False
        slot_total = int(per_wav_total.get(mapped_wav, 1) or 1)

        if mode == "remap" and parsed_candidate:
            wav_abs = os.path.join(wav_root, mapped_wav.replace("/", os.sep))
            boundary_key = (wav_abs, slot_total)
            boundaries = wav_boundary_cache.get(boundary_key)
            if boundaries is None:
                boundaries = _estimate_segment_boundaries_ms(wav_abs, slot_total)
                wav_boundary_cache[boundary_key] = boundaries
            if boundaries and len(boundaries) >= slot_total + 1:
                remapped, changed = _refine_remap_timing_for_slot(
                    parsed=parsed_candidate,
                    boundaries_ms=boundaries,
                    slot_index=slot_idx,
                    slot_total=slot_total,
                    language=language,
                    blend_weight=remap_blend_weight,
                    format_type=_detected_format,
                )
                if remapped:
                    candidate = remapped
                if changed:
                    timing_replaced += 1
                    remap_refined += 1
                    timing_applied = True

        if timing_zero_only and not timing_applied:
            should_estimate = bool(parsed_candidate and (force_boundary_timing or _timing_row_is_zeroish(parsed_candidate)))
            if should_estimate:
                wav_duration_ms = wav_duration_cache.get(mapped_wav)
                if wav_duration_ms is None:
                    wav_abs = os.path.join(wav_root, mapped_wav.replace("/", os.sep))
                    wav_duration_ms = _read_wav_duration_ms(wav_abs)
                    wav_duration_cache[mapped_wav] = wav_duration_ms
                if wav_duration_ms and wav_duration_ms > 0.0:
                    alias_type = _classify_alias_type(language, str(parsed_candidate.get("alias", "")))
                    o2, c2, ct2, pr2, ov2 = _estimate_slot_timing(
                        wav_duration_ms=wav_duration_ms,
                        slot_index=slot_idx,
                        slot_total=slot_total,
                        alias_type=alias_type,
                    )
                    candidate = (
                        f"{parsed_candidate['wav']}={parsed_candidate['alias']},"
                        f"{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}"
                    )
                    timing_replaced += 1
                    boundary_refined += 1
                    timing_applied = True

        if timing_profile and not timing_applied:
            candidate, changed = _apply_timing_profile_to_oto_line(
                line=candidate,
                profile=timing_profile,
                language=language,
                only_when_zero=(timing_zero_only or mode == "remap"),
            )
            if changed:
                timing_replaced += 1
                profile_fallback_refined += 1
        candidate = _apply_suffix_to_oto_line(candidate, alias_suffix)
        if "=" not in candidate:
            continue
        left_out, right_out = candidate.split("=", 1)
        alias = right_out.split(",", 1)[0].strip().lower()
        dedupe_key = (left_out.strip().lower(), alias)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out_lines.append(candidate)

    if not out_lines:
        msg = "No-MFA 자동설정 결과가 비었습니다. 베이스 OTO wav 이름과 현재 WAV 파일 이름 매칭을 확인해 주세요."
        if missing_wavs:
            sample = ", ".join(sorted(missing_wavs)[:5])
            msg = f"{msg} (예시: {sample})"
        return 0, total, [msg]

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out_lines).rstrip() + "\n")

    missing_count = max(total - (exact_hits + norm_hits), 0)
    _log(
        callback,
        f"[No-MFA] source={source} written={len(out_lines)} total={total} "
        f"(exact={exact_hits}, norm={norm_hits}, missing={missing_count})",
    )
    if timing_profile:
        _log(
            callback,
            f"[No-MFA] timing rows replaced: {timing_replaced} "
            f"(remap={remap_refined}, boundary={boundary_refined}, profile_fallback={profile_fallback_refined})",
        )
    if missing_wavs:
        sample = ", ".join(sorted(missing_wavs)[:5])
        suffix = "..." if len(missing_wavs) > 5 else ""
        _log(callback, f"[No-MFA] 매칭 실패 wav: {sample}{suffix}")
    return len(out_lines), total, []


__all__ = [
    "generate_no_mfa_auto_oto",
    "resolve_no_mfa_source_oto",
    "resolve_no_mfa_stats_oto",
]
