from __future__ import annotations

import audioop
import contextlib
import json
import math
import os
import shutil
import tempfile
import wave
from statistics import median
from typing import Callable

from core.oto_file_utils import apply_alias_suffix, parse_oto_line, read_text_with_fallback
from core.oto_normalization import normalize_wav_key

_CLASSIFY_ALIAS_TYPE_FN = None
_LAST_NO_MFA_RUNTIME_META: dict[str, object] = {}


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callable(callback):
        callback(message)


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


def _load_oto_lines(source_oto_path: str, callback: Callable[[str], None] | None = None) -> list[str]:
    path = str(source_oto_path or "").strip()
    if not path:
        return []
    if not os.path.isfile(path):
        _log(callback, f"[No-MFA] base OTO not found: {path}")
        return []
    last_error = ""
    for encoding in ("utf-8-sig", "cp932", "cp949", "euc-kr", "utf-8"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                rows = []
                for raw in handle:
                    line = str(raw or "").strip()
                    if not line or "=" not in line:
                        continue
                    rows.append(line)
            if encoding not in ("utf-8-sig", "utf-8"):
                _log(callback, f"[No-MFA] base OTO decoded as {encoding}: {path}")
            return rows
        except UnicodeDecodeError as exc:
            last_error = f"{encoding}: {exc}"
            continue
        except OSError as exc:
            _log(callback, f"[No-MFA] base OTO read error: {exc}")
            return []
    _log(callback, f"[No-MFA] base OTO decode failed for all encodings ({last_error})")
    return []


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if raw.endswith(("\\", "/")):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    if os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    return raw


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
    if raw in {"mfa_free", "mfa-free", "ssl_slot", "ssl-slot", "mfa_free_ssl_slot", "default"}:
        return "mfa_free_ssl_slot"
    if raw in {"alias_auto", "alias-only", "alias_only", "blank_auto", "blank"}:
        return "remap"
    if raw in {"remap", "base_remap", "base"}:
        return "remap"
    mode_default = str(os.environ.get("UTOA_NO_MFA_DEFAULT_MODE", "mfa_free_ssl_slot") or "").strip().lower()
    if mode_default in {"mfa_free_ssl_slot", "mfa_free", "mfa-free", "ssl_slot", "ssl-slot"}:
        return "mfa_free_ssl_slot"
    return "remap"


def _set_last_no_mfa_runtime_meta(**kwargs) -> None:
    _LAST_NO_MFA_RUNTIME_META.clear()
    _LAST_NO_MFA_RUNTIME_META.update(kwargs)


def get_last_no_mfa_runtime_meta() -> dict[str, object]:
    return dict(_LAST_NO_MFA_RUNTIME_META)


def _resolve_default_mfa_free_checkpoint() -> str:
    env_path = str(os.environ.get("UTOA_MFA_FREE_OTO_CHECKPOINT", "") or "").strip()
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)
    roots = [
        str(os.environ.get("UTOA_APP_DIR", "") or "").strip(),
        os.getcwd(),
    ]
    candidates: list[str] = []
    preferred_names = {
        "world_v1_light_c4_tune_d.pt",
        "world_v1_light_c4.pt",
    }
    for base in roots:
        if not base:
            continue
        root = os.path.join(base, "ml_workspace", "mfa_free_oto")
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                low = str(name).lower()
                if not low.endswith(".pt"):
                    continue
                full = os.path.join(dirpath, name)
                if low in preferred_names:
                    return os.path.abspath(full)
                if "world_v1" in low or "worldv1" in low:
                    candidates.append(full)
                elif "wavlm" in low:
                    candidates.append(full)
    if not candidates:
        return ""
    candidates = [os.path.abspath(path) for path in candidates if os.path.isfile(path)]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _resolve_default_no_mfa_boundary_checkpoint() -> str:
    for env_name in ("UTOA_NO_MFA_BOUNDARY_MODEL", "UTOA_PHONEME_BOUNDARY_MODEL"):
        env_path = str(os.environ.get(env_name, "") or "").strip()
        if env_path and os.path.isfile(env_path):
            return os.path.abspath(env_path)
    try:
        from core.phoneme_boundary.discovery import list_available_phoneme_boundary_models

        models = list_available_phoneme_boundary_models()
        for item in models:
            path = str(item.get("path", "") or "").strip()
            if path and os.path.isfile(path):
                return os.path.abspath(path)
    except Exception:
        pass
    roots = [
        str(os.environ.get("UTOA_APP_DIR", "") or "").strip(),
        os.getcwd(),
    ]
    candidates: list[str] = []
    for base in roots:
        if not base:
            continue
        root = os.path.join(base, "ml_workspace", "mfa_free_oto")
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                low = str(name).lower()
                if not low.endswith(".pt"):
                    continue
                if "phoneme_boundary" in low or "boundary" in low:
                    candidates.append(os.path.join(dirpath, name))
    candidates = [os.path.abspath(path) for path in candidates if os.path.isfile(path)]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _load_runtime_baseline_metrics() -> dict[str, float]:
    hint = str(os.environ.get("UTOA_NO_MFA_BASELINE_METRICS_JSON", "") or "").strip()
    candidates: list[str] = []
    if hint:
        candidates.append(hint)
    roots = [
        str(os.environ.get("UTOA_APP_DIR", "") or "").strip(),
        os.getcwd(),
    ]
    for base in roots:
        if not base:
            continue
        candidates.append(os.path.join(base, "ml", "eval", "no_mfa_runtime_c4_lab", "metrics.json"))
    for candidate in candidates:
        path = os.path.abspath(str(candidate or "").strip())
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            metrics = dict(payload.get("metrics") or {})
            out: dict[str, float] = {}
            for key in ("mis_mapping_rate", "hard_fail_rate", "nucleus_miss_rate", "one_step_shift_rate", "vc_pre_ge_80ms_rate"):
                if key in metrics:
                    out[key] = float(metrics.get(key))
            return out
        except Exception:
            continue
    return {}


def _build_runtime_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    if not metrics or not baseline:
        return {}
    out: dict[str, float] = {}
    for key in ("mis_mapping_rate", "hard_fail_rate", "nucleus_miss_rate", "one_step_shift_rate", "vc_pre_ge_80ms_rate"):
        if key in metrics and key in baseline:
            out[f"delta_{key}"] = float(metrics[key]) - float(baseline[key])
    return out


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


def _apply_no_mfa_file_consistency(
    *,
    language: str,
    oto_path: str,
    format_type: str,
    callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    lang = str(language or "").strip().lower() or "japanese"
    fmt = str(format_type or "").strip().lower()
    if lang != "japanese" or "cvvc" not in fmt:
        return {"enabled": False, "reason": "scope_not_japanese_cvvc"}
    if not oto_path or not os.path.isfile(oto_path):
        return {"enabled": True, "status": "skipped", "reason": "missing_oto"}
    try:
        from core.generation.common.oto_generator import validate_oto_params
        from core.generation.ja.ja_oto_file_consistency import apply_ja_vc_neighbor_to_oto_file

        stats = apply_ja_vc_neighbor_to_oto_file(
            oto_path,
            validate_fn=validate_oto_params,
            log_fn=callback,
        )
        changed = int(stats.get("total_changed", 0) or 0)
        if changed > 0:
            _log(callback, f"[No-MFA] file consistency: changed={changed}.")
        return {
            "enabled": True,
            "status": "applied" if changed > 0 else "no_change",
            **dict(stats),
        }
    except Exception as exc:
        _log(callback, f"[No-MFA] file consistency skipped: {exc}")
        return {"enabled": True, "status": "failed", "reason": str(exc)}


def _boundary_candidate_config_from_env():
    from core.generation.common.oto_boundary_candidates import CandidateConfig, parse_float_map, parse_role_labels

    preset = str(os.environ.get("UTOA_NO_MFA_BOUNDARY_CANDIDATE_PRESET", "cv_safe") or "cv_safe").strip().lower()
    if preset in {"cv_vc_p90_safe", "cv-vc-p90-safe", "cv_vc_p90", "cv-vc-p90"}:
        role_labels = {"cv": "vowel_onset", "vc": "vowel_end"}
        max_shift = 60.0
        role_max_shift = {"cv": 40.0, "vc": 50.0}
        role_margin = {"vc": 0.08}
        neighbor_guard = False
    elif preset in {"cv_vc_safe", "cv-vc-safe", "cv_vc", "cv-vc"}:
        role_labels = {"cv": "vowel_onset", "vc": "vowel_end"}
        max_shift = 60.0
        role_max_shift = {"cv": 60.0, "vc": 50.0}
        role_margin = {"vc": 0.08}
        neighbor_guard = False
    elif preset in {"cv_vc_guarded", "cv-vc-guarded", "guarded"}:
        role_labels = {"cv": "vowel_onset", "vc": "vowel_end"}
        max_shift = 60.0
        role_max_shift = {"cv": 60.0, "vc": 50.0}
        role_margin = {"vc": 0.05}
        neighbor_guard = True
    elif preset in {"vc_guarded", "vc-guarded"}:
        role_labels = {"vc": "vowel_end"}
        max_shift = 50.0
        role_max_shift = {"vc": 50.0}
        role_margin = {"vc": 0.05}
        neighbor_guard = True
    else:
        role_labels = {"cv": "vowel_onset"}
        max_shift = 60.0
        role_max_shift = {"cv": 60.0}
        role_margin = {}
        neighbor_guard = False

    role_override = str(os.environ.get("UTOA_NO_MFA_BOUNDARY_ROLE_LABELS", "") or "").strip()
    if role_override:
        role_labels = parse_role_labels(role_override)
    role_max_override = str(os.environ.get("UTOA_NO_MFA_BOUNDARY_ROLE_MAX_SHIFT_MS", "") or "").strip()
    if role_max_override:
        role_max_shift = parse_float_map(role_max_override)
    role_margin_override = str(os.environ.get("UTOA_NO_MFA_BOUNDARY_ROLE_MIN_SCORE_MARGIN", "") or "").strip()
    if role_margin_override:
        role_margin = parse_float_map(role_margin_override)

    return CandidateConfig(
        role_labels=role_labels,
        search_radius_ms=_env_float("UTOA_NO_MFA_BOUNDARY_SEARCH_RADIUS_MS", 120.0),
        max_shift_ms=_env_float("UTOA_NO_MFA_BOUNDARY_MAX_SHIFT_MS", max_shift),
        min_shift_ms=_env_float("UTOA_NO_MFA_BOUNDARY_MIN_SHIFT_MS", 8.0),
        min_score=_env_float("UTOA_NO_MFA_BOUNDARY_MIN_SCORE", 0.18),
        min_score_margin=_env_float("UTOA_NO_MFA_BOUNDARY_MIN_SCORE_MARGIN", 0.035),
        min_quality=_env_float("UTOA_NO_MFA_BOUNDARY_MIN_QUALITY", 0.0),
        min_new_offset_ms=_env_float("UTOA_NO_MFA_BOUNDARY_MIN_NEW_OFFSET_MS", 0.0),
        skip_edge_peaks=not _env_bool("UTOA_NO_MFA_BOUNDARY_KEEP_EDGE_PEAKS", False),
        center_penalty_per_ms=_env_float("UTOA_NO_MFA_BOUNDARY_CENTER_PENALTY_PER_MS", 0.0),
        role_max_shift_ms=role_max_shift,
        role_min_score_margin=role_margin,
        neighbor_guard_enabled=_env_bool("UTOA_NO_MFA_BOUNDARY_NEIGHBOR_GUARD", neighbor_guard),
        neighbor_guard_min_gap_ms=_env_float("UTOA_NO_MFA_BOUNDARY_NEIGHBOR_GUARD_MIN_GAP_MS", 10.0),
    )


def _apply_no_mfa_boundary_candidate_postprocess(
    *,
    language: str,
    oto_path: str,
    wav_dir: str,
    format_type: str,
    callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    if not _env_bool("UTOA_NO_MFA_BOUNDARY_CANDIDATES_ENABLE", False):
        return {"enabled": False, "reason": "disabled"}
    lang = str(language or "").strip().lower() or "japanese"
    fmt = str(format_type or "").strip().lower()
    if lang != "japanese" or "cvvc" not in fmt:
        return {"enabled": False, "reason": "scope_not_japanese_cvvc"}
    if not oto_path or not os.path.isfile(oto_path):
        return {"enabled": True, "status": "skipped", "reason": "missing_oto"}
    if not wav_dir or not os.path.isdir(wav_dir):
        return {"enabled": True, "status": "skipped", "reason": "missing_wav_dir"}
    model_path = _resolve_default_no_mfa_boundary_checkpoint()
    if not model_path:
        _log(callback, "[No-MFA] boundary candidate postprocess skipped: model not found.")
        return {"enabled": True, "status": "skipped", "reason": "missing_model"}
    config = _boundary_candidate_config_from_env()
    report_path = ""
    try:
        from core.generation.common.oto_boundary_candidates import (
            apply_boundary_candidates_to_oto_file,
            config_to_json,
        )

        _log(
            callback,
            "[No-MFA] boundary candidate postprocess: "
            f"model={os.path.basename(model_path)} roles={','.join(sorted(config.role_labels))}",
        )
        report = apply_boundary_candidates_to_oto_file(
            oto_path=oto_path,
            wav_dir=wav_dir,
            model_path=model_path,
            config=config,
            out_path=oto_path,
            device=str(os.environ.get("UTOA_NO_MFA_BOUNDARY_DEVICE", os.environ.get("UTOA_MFA_FREE_OTO_DEVICE", "auto")) or "auto"),
            use_acoustic_heuristics=not _env_bool("UTOA_NO_MFA_BOUNDARY_DISABLE_ACOUSTIC_HEURISTICS", False),
            heuristic_weight=_env_float("UTOA_NO_MFA_BOUNDARY_HEURISTIC_WEIGHT", 0.28),
        )
        if _env_bool("UTOA_NO_MFA_BOUNDARY_REPORT_ENABLE", True):
            report_path = f"{oto_path}.boundary_candidates.report.json"
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        summary = dict(report.get("summary") or {}) if isinstance(report, dict) else {}
        score_maps = dict(report.get("score_maps") or {}) if isinstance(report, dict) else {}
        adjusted_by_family = (
            dict(summary.get("adjusted_by_family_label") or {}) if isinstance(summary.get("adjusted_by_family_label"), dict) else {}
        )
        changed = int(summary.get("adjusted_rows", 0) or 0)
        _log(
            callback,
            "[No-MFA] boundary candidate postprocess: "
            f"changed={changed}, wav_scores={int(score_maps.get('loaded_wavs', 0) or 0)}/"
            f"{int(score_maps.get('requested_wavs', 0) or 0)}",
        )
        return {
            "enabled": True,
            "status": "applied" if changed > 0 else "no_change",
            "changed_rows": changed,
            "model_path": model_path,
            "report_path": report_path,
            "score_maps_loaded": int(score_maps.get("loaded_wavs", 0) or 0),
            "score_maps_requested": int(score_maps.get("requested_wavs", 0) or 0),
            "adjusted_by_family_label": adjusted_by_family,
            "config": config_to_json(config),
        }
    except Exception as exc:
        _log(callback, f"[No-MFA] boundary candidate postprocess skipped: {exc}")
        return {"enabled": True, "status": "failed", "reason": str(exc), "model_path": model_path}


def _apply_no_mfa_ml_correction(
    *,
    language: str,
    oto_path: str,
    wav_dir: str,
    format_type: str,
    callback: Callable[[str], None] | None = None,
) -> int:
    if not _env_bool("UTOA_NO_MFA_ML_ENABLE", True):
        _log(callback, "[No-MFA] existing OTO-ML correction disabled by UTOA_NO_MFA_ML_ENABLE.")
        return 0
    if not oto_path or not os.path.isfile(oto_path):
        return 0

    policy = str(os.environ.get("UTOA_NO_MFA_ML_POLICY", "auto") or "auto").strip().lower() or "auto"
    route = str(os.environ.get("UTOA_NO_MFA_ML_ROUTE", os.environ.get("UTOA_ML_ROUTE", "legacy")) or "legacy").strip()
    report: dict[str, object] = {}
    pre_safety_path = ""
    if _env_bool("UTOA_NO_MFA_ML_SAFETY_ENABLE", True):
        fd, pre_safety_path = tempfile.mkstemp(prefix="autooto_no_mfa_pre_ml_", suffix=".ini")
        os.close(fd)
        try:
            shutil.copyfile(oto_path, pre_safety_path)
        except Exception:
            with contextlib.suppress(Exception):
                os.remove(pre_safety_path)
            pre_safety_path = ""
    try:
        from core.oto_ml_refiner import apply_oto_ml_to_oto_file

        changed = int(
            apply_oto_ml_to_oto_file(
                str(language or "").strip().lower() or "korean",
                oto_path,
                tg_dir="",
                wav_dir=wav_dir,
                custom_phonemes_path="",
                callback=callback,
                enabled=True,
                format_override=str(format_type or "").strip().lower(),
                policy=policy,
                report=report,
                ml_route=route,
            )
            or 0
        )
        if pre_safety_path:
            try:
                from core.generation.common.oto_ml_safety import apply_no_mfa_lightgbm_safety_filter

                safety_report = apply_no_mfa_lightgbm_safety_filter(
                    pre_oto_path=pre_safety_path,
                    post_oto_path=oto_path,
                    language=str(language or "").strip().lower() or "korean",
                    format_type=str(format_type or "").strip().lower(),
                    max_overlap_increase_ms=_env_float("UTOA_NO_MFA_ML_SAFETY_MAX_OVERLAP_INCREASE_MS", 5.0),
                )
                if bool(safety_report.get("enabled")) and int(safety_report.get("changed_rows", 0) or 0) > 0:
                    _log(
                        callback,
                        "[No-MFA] existing OTO-ML safety filter: "
                        f"restored_terminal={int(safety_report.get('restored_terminal_rows', 0) or 0)}, "
                        f"restored_breath={int(safety_report.get('restored_breath_rows', 0) or 0)}, "
                        f"restored_pure_vowel={int(safety_report.get('restored_pure_vowel_sequence_rows', 0) or 0)}, "
                        f"restored_pure_vowel_onset={int(safety_report.get('restored_pure_vowel_onset_sequence_rows', 0) or 0)}, "
                        f"restored_middle_dot_transition={int(safety_report.get('restored_middle_dot_transition_rows', 0) or 0)}, "
                        f"restored_headless_vc_onset={int(safety_report.get('restored_headless_vc_onset_sequence_rows', 0) or 0)}, "
                        f"restored_regular_pair_onset={int(safety_report.get('restored_regular_pair_onset_sequence_rows', 0) or 0)}, "
                        f"restored_terminal_release_r={int(safety_report.get('restored_terminal_release_r_rows', 0) or 0)}, "
                        f"restored_terminal_release_r_terminal_copy={int(safety_report.get('restored_terminal_release_r_terminal_copy_rows', 0) or 0)}, "
                        f"restored_standalone_release_r_offset={int(safety_report.get('restored_standalone_release_r_offset_rows', 0) or 0)}, "
                        f"capped_standalone_release_r_cutoff={int(safety_report.get('capped_standalone_release_r_cutoff_rows', 0) or 0)}, "
                        f"capped_pitch_regular_pair_cv_pre_overlap={int(safety_report.get('capped_pitch_regular_pair_cv_pre_overlap_rows', 0) or 0)}, "
                        f"clamped_pitch_regular_pair_cv_cutoff={int(safety_report.get('clamped_pitch_regular_pair_cv_cutoff_rows', 0) or 0)}, "
                        f"restored_terminal_v={int(safety_report.get('restored_terminal_standalone_vowel_rows', 0) or 0)}, "
                        f"retimed_terminal_v={int(safety_report.get('retimed_terminal_standalone_vowel_companion_rows', 0) or 0)}, "
                        f"restored_initial_v={int(safety_report.get('restored_initial_standalone_vowel_rows', 0) or 0)}, "
                        f"restored_v_offset_cutoff={int(safety_report.get('restored_standalone_vowel_offset_cutoff_rows', 0) or 0)}, "
                        f"restored_cv_head_cutoff={int(safety_report.get('restored_cv_head_cutoff_rows', 0) or 0)}, "
                        f"restored_spaced_consonant={int(safety_report.get('restored_spaced_consonant_rows', 0) or 0)}, "
                        f"restored_vv_overlap={int(safety_report.get('restored_vv_overlap_rows', 0) or 0)}, "
                        f"restored_terminal_vc_overlap={int(safety_report.get('restored_terminal_vc_overlap_rows', 0) or 0)}, "
                        f"restored_cv_fixed_cutoff={int(safety_report.get('restored_cv_fixed_cutoff_rows', 0) or 0)}, "
                        f"restored_underscore_vc_positive_offset_shift={int(safety_report.get('restored_underscore_vc_positive_offset_shift_rows', 0) or 0)}, "
                        f"regularized_following_cv_block_offset={int(safety_report.get('regularized_following_cv_block_offset_rows', 0) or 0)}, "
                        f"retimed_romaji_vc_following_cv={int(safety_report.get('retimed_romaji_vc_following_cv_rows', 0) or 0)}, "
                        f"shifted_pitch_missing_cv_slot={int(safety_report.get('shifted_pitch_missing_cv_slot_rows', 0) or 0)}, "
                        f"shifted_pitch_headless_initial_slot={int(safety_report.get('shifted_pitch_headless_initial_slot_rows', 0) or 0)}, "
                        f"shifted_pitch_katakana_n_grid={int(safety_report.get('shifted_pitch_katakana_n_grid_rows', 0) or 0)}, "
                        f"shifted_underscore_kana_slot1={int(safety_report.get('shifted_underscore_kana_slot1_rows', 0) or 0)}, "
                        f"shifted_underscore_kana_delayed_slot={int(safety_report.get('shifted_underscore_kana_delayed_slot_rows', 0) or 0)}, "
                        f"capped_cv_head_vowel_pre_overlap={int(safety_report.get('capped_cv_head_vowel_pre_overlap_rows', 0) or 0)}, "
                        f"capped_cv_pre_overlap={int(safety_report.get('capped_cv_pre_overlap_rows', 0) or 0)}, "
                        f"capped_overlap={int(safety_report.get('capped_overlap_rows', 0) or 0)}",
                    )
            except Exception as safety_exc:
                _log(callback, f"[No-MFA] existing OTO-ML safety filter skipped: {safety_exc}")
    except Exception as exc:
        _log(callback, f"[No-MFA] existing OTO-ML correction skipped: {exc}")
        return 0
    finally:
        if pre_safety_path:
            with contextlib.suppress(Exception):
                os.remove(pre_safety_path)

    status = str(report.get("status", "") or "").strip() or "unknown"
    code = str(report.get("code", "") or "").strip()
    confidence = float(report.get("model_confidence", 0.0) or 0.0)
    _log(
        callback,
        f"[No-MFA] existing OTO-ML correction: status={status}, changed={changed}, "
        f"confidence={confidence:.3f}, code={code or 'n/a'}",
    )
    return int(changed)


def _no_mfa_sequence_residual_default_model_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    candidates = [
        os.path.join(root, "models", "korean", "no_mfa_sequence_residual_hybrid.json"),
        os.path.join(
            root,
            "eval_runs",
            "sequence_residual_benchmark_v2",
            "hybrid_baseline_model",
            "sequence_residual_baseline_model.json",
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _load_no_mfa_sequence_residual_model(callback: Callable[[str], None] | None = None) -> dict[str, object] | None:
    if not _env_bool("UTOA_NO_MFA_SEQUENCE_RESIDUAL_ENABLE", False):
        return None
    model_path = str(os.environ.get("UTOA_NO_MFA_SEQUENCE_RESIDUAL_MODEL", "") or "").strip()
    if not model_path:
        model_path = _no_mfa_sequence_residual_default_model_path()
    if not model_path or not os.path.isfile(model_path):
        _log(callback, "[No-MFA] sequence residual correction skipped: model file not found.")
        return None
    try:
        with open(model_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)
    except Exception as exc:
        _log(callback, f"[No-MFA] sequence residual correction skipped: failed to load model ({exc}).")
        return None
    if not isinstance(model, dict) or not isinstance(model.get("targets"), dict):
        _log(callback, "[No-MFA] sequence residual correction skipped: invalid model schema.")
        return None
    model["_runtime_model_path"] = os.path.abspath(model_path)
    return model


def _no_mfa_safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _no_mfa_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _no_mfa_stats_for_window(
    profile: dict[str, object] | None,
    key: str,
    start_ms: float,
    end_ms: float,
) -> tuple[float, float, float]:
    if not profile:
        return 0.0, 0.0, 0.0
    values = profile.get(key)
    if not isinstance(values, list) or not values:
        return 0.0, 0.0, 0.0
    lo = _time_to_frame(profile, start_ms)
    hi = _time_to_frame(profile, end_ms)
    if hi < lo:
        lo, hi = hi, lo
    window = []
    for value in values[lo : hi + 1]:
        try:
            window.append(float(value))
        except Exception:
            pass
    if not window:
        return 0.0, 0.0, 0.0
    mean_value = _no_mfa_mean(window)
    variance = _no_mfa_mean([(value - mean_value) ** 2.0 for value in window])
    return mean_value, max(window), math.sqrt(max(0.0, variance))


def _no_mfa_min_for_window(
    profile: dict[str, object] | None,
    key: str,
    start_ms: float,
    end_ms: float,
) -> float:
    if not profile:
        return 0.0
    values = profile.get(key)
    if not isinstance(values, list) or not values:
        return 0.0
    lo = _time_to_frame(profile, start_ms)
    hi = _time_to_frame(profile, end_ms)
    if hi < lo:
        lo, hi = hi, lo
    window = []
    for value in values[lo : hi + 1]:
        try:
            window.append(float(value))
        except Exception:
            pass
    return min(window) if window else 0.0


def _no_mfa_alias_group(alias_type: str, alias: str) -> str:
    alias_key = str(alias_type or "").strip().lower()
    alias_text = str(alias or "").strip()
    if alias_key == "endbreath":
        return "endbreath"
    if alias_key in {"vc", "vcv", "vv"} or " " in alias_text:
        return "bridge"
    return "cv"


def _no_mfa_is_endbreath_alias(alias: str) -> bool:
    text = str(alias or "").strip()
    compact = text.replace(" ", "")
    return bool(
        text.endswith("'R")
        or text.endswith(" R")
        or compact.endswith("*R")
        or text in {"R", "br", "breath", "Breath"}
    )


def _no_mfa_sequence_row_features(
    *,
    parsed: dict[str, object],
    boundaries_ms: list[float],
    activity_profile: dict[str, object] | None,
    slot_index: int,
    slot_total: int,
    language: str,
    format_type: str,
    quality: dict[str, object] | None,
) -> dict[str, object]:
    total = max(1, int(slot_total or 1))
    idx = max(0, min(int(slot_index or 0), total - 1))
    duration_ms = _no_mfa_safe_float(
        (activity_profile or {}).get("duration_ms", 0.0),
        _no_mfa_safe_float(boundaries_ms[-1] if boundaries_ms else 0.0, 0.0),
    )
    if duration_ms <= 0.0 and boundaries_ms:
        duration_ms = _no_mfa_safe_float(boundaries_ms[-1], 0.0)
    if len(boundaries_ms) >= idx + 2:
        anchor_start = _no_mfa_safe_float(boundaries_ms[idx], 0.0)
        anchor_end = _no_mfa_safe_float(boundaries_ms[idx + 1], anchor_start)
    else:
        anchor_start = 0.0
        anchor_end = max(0.0, duration_ms)
    anchor_len = max(1.0, anchor_end - anchor_start)
    next_anchor_start = _no_mfa_safe_float(boundaries_ms[idx + 2], anchor_end) if len(boundaries_ms) >= idx + 3 else anchor_end
    active_start = _no_mfa_safe_float((activity_profile or {}).get("active_start_ms", 0.0), 0.0)
    active_end = _no_mfa_safe_float((activity_profile or {}).get("active_end_ms", duration_ms), duration_ms)
    active_len = max(1.0, active_end - active_start)
    alias = str(parsed.get("alias", "") or "")
    alias_type = str((quality or {}).get("alias_type", "") or "").strip().lower()
    if _no_mfa_is_endbreath_alias(alias):
        alias_type = "endbreath"
    if not alias_type:
        alias_type = _classify_alias_type(language, alias)
    rms_local_mean, rms_local_max, rms_local_std = _no_mfa_stats_for_window(
        activity_profile,
        "smooth",
        anchor_start,
        anchor_end,
    )
    rms_onset_mean, rms_onset_max, _rms_onset_std = _no_mfa_stats_for_window(
        activity_profile,
        "smooth",
        max(active_start, anchor_start - 35.0),
        min(active_end, anchor_start + 55.0),
    )
    rms_tail_mean, _rms_tail_max, _rms_tail_std = _no_mfa_stats_for_window(
        activity_profile,
        "smooth",
        max(active_start, anchor_end - 65.0),
        min(active_end, anchor_end + 45.0),
    )
    flux_onset_mean, flux_onset_max, _flux_onset_std = _no_mfa_stats_for_window(
        activity_profile,
        "flux",
        max(active_start, anchor_start - 35.0),
        min(active_end, anchor_start + 55.0),
    )
    return {
        "language": str(language or "").strip().lower() or "korean",
        "format_type": str(format_type or "").strip().lower(),
        "alias_type": alias_type,
        "alias_group": _no_mfa_alias_group(alias_type, alias),
        "sequence_source": "filename",
        "sequence_soft_lock": "True",
        "sequence_cvn_backend": "runtime/no_mfa",
        "anchor_match_strategy": "file_order_ratio",
        "row_index_in_wav": float(idx),
        "row_ratio_in_wav": float(idx) / float(max(1, total - 1)),
        "file_row_count": float(total),
        "sequence_word_count": float(total),
        "anchor_index": float(idx),
        "anchor_start_ms": anchor_start,
        "anchor_end_ms": anchor_end,
        "anchor_mid_ms": (anchor_start + anchor_end) * 0.5,
        "anchor_len_ms": anchor_len,
        "anchor_confidence": _no_mfa_safe_float((quality or {}).get("confidence", 0.0), 0.0),
        "active_start_ms": active_start,
        "active_end_ms": active_end,
        "active_len_ms": active_len,
        "next_anchor_start_ms": next_anchor_start,
        "next_anchor_gap_ms": max(0.0, next_anchor_start - anchor_end),
        "wav_duration_ms": duration_ms,
        "base_offset": _no_mfa_safe_float(parsed.get("offset", 0.0), 0.0),
        "base_pre": _no_mfa_safe_float(parsed.get("pre", 0.0), 0.0),
        "base_cons": _no_mfa_safe_float(parsed.get("cons", 0.0), 0.0),
        "base_cutoff_abs": abs(_no_mfa_safe_float(parsed.get("cutoff", 0.0), 0.0)),
        "base_ovl": _no_mfa_safe_float(parsed.get("ovl", 0.0), 0.0),
        "rms_local_mean": rms_local_mean,
        "rms_local_max": rms_local_max,
        "rms_local_std": rms_local_std,
        "rms_onset_mean": rms_onset_mean,
        "rms_onset_max": rms_onset_max,
        "rms_tail_mean": rms_tail_mean,
        "rms_tail_min": _no_mfa_min_for_window(activity_profile, "smooth", max(active_start, anchor_end - 65.0), min(active_end, anchor_end + 45.0)),
        "zcr_onset_mean": 0.0,
        "zcr_onset_max": 0.0,
        "hf_onset_mean": 0.0,
        "hf_onset_max": 0.0,
        "flux_onset_mean": flux_onset_mean,
        "flux_onset_max": flux_onset_max,
        "label_silence_ratio": 0.0,
        "label_vowel_ratio": 0.0,
        "label_onset_ratio": 0.0,
        "label_breath_ratio": 1.0 if alias_type == "endbreath" else 0.0,
    }


def _no_mfa_model_feature_value(name: str, row: dict[str, object], scalers: dict[str, object]) -> float:
    if "=" in name:
        base, expected = name.split("=", 1)
        return 1.0 if str(row.get(base, "") or "") == expected else 0.0
    scaler = scalers.get(name, [0.0, 1.0]) if isinstance(scalers, dict) else [0.0, 1.0]
    try:
        mean = float(scaler[0])
        std = max(1e-8, float(scaler[1]))
    except Exception:
        mean = 0.0
        std = 1.0
    return (_no_mfa_safe_float(row.get(name, 0.0), 0.0) - mean) / std


def _no_mfa_prediction_scale(model: dict[str, object], row: dict[str, object], target: str) -> float:
    mode = str(model.get("target_mode", "raw") or "raw").strip().lower()
    target_mode = "normalized" if mode == "normalized" else "raw"
    if mode == "hybrid":
        target_mode = "normalized" if target in {"delta_offset", "delta_cutoff"} else "raw"
    if target_mode != "normalized":
        return 1.0
    if target in {"delta_offset", "delta_cutoff"}:
        return max(1.0, _no_mfa_safe_float(row.get("active_len_ms", 1.0), 1.0))
    return max(1.0, _no_mfa_safe_float(row.get("anchor_len_ms", 1.0), 1.0))


def _no_mfa_sequence_scope_allows(target: str, format_type: str) -> bool:
    scope = str(os.environ.get("UTOA_NO_MFA_SEQUENCE_RESIDUAL_SCOPE", "vcv_pco") or "vcv_pco").strip().lower()
    if scope in {"all", "*"}:
        return True
    if scope in {"pre_cons_ovl", "pco"}:
        return target in {"delta_pre", "delta_cons", "delta_ovl"}
    if scope in {"vcv", "vcv_pco"}:
        return str(format_type or "").strip().lower() == "vcv" and target in {"delta_pre", "delta_cons", "delta_ovl"}
    if scope in {"offset_cutoff", "oc"}:
        return target in {"delta_offset", "delta_cutoff"}
    if target in {"delta_pre", "delta_cons", "delta_ovl"}:
        return True
    return str(format_type or "").strip().lower() in {"cv", "cvc"}


def _apply_no_mfa_sequence_residual_to_line(
    *,
    line: str,
    model: dict[str, object] | None,
    boundaries_ms: list[float],
    activity_profile: dict[str, object] | None,
    slot_index: int,
    slot_total: int,
    language: str,
    format_type: str,
    quality: dict[str, object] | None,
) -> tuple[str, bool]:
    if not model:
        return line, False
    parsed = parse_oto_line(line)
    if not parsed:
        return line, False
    row = _no_mfa_sequence_row_features(
        parsed=parsed,
        boundaries_ms=boundaries_ms,
        activity_profile=activity_profile,
        slot_index=slot_index,
        slot_total=slot_total,
        language=language,
        format_type=format_type,
        quality=quality,
    )
    min_conf = max(0.0, min(1.0, _env_float("UTOA_NO_MFA_SEQUENCE_RESIDUAL_MIN_CONF", 0.0)))
    if min_conf > 0.0 and _no_mfa_safe_float(row.get("anchor_confidence", 0.0), 0.0) < min_conf:
        return line, False
    feature_names = model.get("feature_names", [])
    scalers = model.get("feature_scalers", {})
    targets = model.get("targets", {})
    if not isinstance(feature_names, list) or not isinstance(targets, dict):
        return line, False
    features = [_no_mfa_model_feature_value(str(name), row, scalers if isinstance(scalers, dict) else {}) for name in feature_names]
    blend = max(0.0, min(1.0, _env_float("UTOA_NO_MFA_SEQUENCE_RESIDUAL_BLEND", 0.35)))

    offset = _no_mfa_safe_float(parsed.get("offset", 0.0), 0.0)
    pre = _no_mfa_safe_float(parsed.get("pre", 0.0), 0.0)
    cons = _no_mfa_safe_float(parsed.get("cons", 0.0), 0.0)
    cutoff_abs = abs(_no_mfa_safe_float(parsed.get("cutoff", 0.0), 0.0))
    ovl = _no_mfa_safe_float(parsed.get("ovl", 0.0), 0.0)
    values = {
        "delta_offset": 0.0,
        "delta_pre": 0.0,
        "delta_cons": 0.0,
        "delta_cutoff": 0.0,
        "delta_ovl": 0.0,
    }
    for target, payload in targets.items():
        target_name = str(target)
        if target_name not in values or not _no_mfa_sequence_scope_allows(target_name, format_type):
            continue
        if not isinstance(payload, dict):
            continue
        weights = payload.get("weights", [])
        if not isinstance(weights, list) or len(weights) != len(features) + 1:
            continue
        pred = _no_mfa_safe_float(weights[0], 0.0)
        for feature, weight in zip(features, weights[1:]):
            pred += float(feature) * _no_mfa_safe_float(weight, 0.0)
        pred_ms = pred * _no_mfa_prediction_scale(model, row, target_name)
        values[target_name] = max(-260.0, min(260.0, pred_ms)) * blend

    if all(abs(value) <= 1e-6 for value in values.values()):
        return line, False
    o2 = offset + values["delta_offset"]
    pr2 = pre + values["delta_pre"]
    c2 = cons + values["delta_cons"]
    ct2 = -(cutoff_abs + values["delta_cutoff"])
    ov2 = ovl + values["delta_ovl"]
    o2, c2, ct2, pr2, ov2 = _fallback_validate_oto_params(o2, c2, ct2, pr2, ov2)

    active_start = _no_mfa_safe_float(row.get("active_start_ms", 0.0), 0.0)
    active_end = _no_mfa_safe_float(row.get("active_end_ms", row.get("wav_duration_ms", 0.0)), 0.0)
    anchor_end = _no_mfa_safe_float(row.get("anchor_end_ms", active_end), active_end)
    duration = max(active_end, _no_mfa_safe_float(row.get("wav_duration_ms", active_end), active_end))
    o2 = max(max(0.0, active_start - 20.0), min(max(0.0, duration - 1.0), min(anchor_end, active_end + 15.0), o2))
    rewritten = f"{parsed['wav']}={parsed['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}"
    return rewritten, rewritten != line


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


def _analyze_wav_activity(wav_path: str) -> dict[str, object]:
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
        return {
            "duration_ms": duration_ms,
            "step_ms": 10.0,
            "smooth": [],
            "flux": [],
            "threshold": 0.0,
            "active_start_ms": 0.0,
            "active_end_ms": max(0.0, duration_ms),
        }

    frame_bytes = max(1, channels * sampwidth)
    window_frames = max(1, int(framerate * 0.01))
    step_frames = window_frames
    total_windows = int(math.ceil(float(nframes) / float(step_frames)))
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

    duration_ms = float(nframes) * 1000.0 / float(framerate)
    step_ms = float(step_frames) * 1000.0 / float(framerate)
    if not energy:
        return {
            "duration_ms": duration_ms,
            "step_ms": step_ms,
            "smooth": [],
            "flux": [],
            "threshold": 0.0,
            "active_start_ms": 0.0,
            "active_end_ms": duration_ms,
        }

    smooth: list[float] = []
    radius = 2
    n = len(energy)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n - 1, i + radius)
        smooth.append(sum(energy[lo : hi + 1]) / float(hi - lo + 1))

    sorted_smooth = sorted(smooth)
    floor_idx = max(0, min(len(sorted_smooth) - 1, int(round((len(sorted_smooth) - 1) * 0.18))))
    ceil_idx = max(0, min(len(sorted_smooth) - 1, int(round((len(sorted_smooth) - 1) * 0.92))))
    floor = float(sorted_smooth[floor_idx])
    ceiling = float(sorted_smooth[ceil_idx])
    threshold = max(floor + 18.0, floor + ((ceiling - floor) * 0.18), ceiling * 0.08)

    flux = [0.0]
    for i in range(1, len(smooth)):
        flux.append(max(0.0, smooth[i] - smooth[i - 1]))

    active = [i for i, value in enumerate(smooth) if value >= threshold]
    if active:
        active_start_idx = max(0, active[0] - 1)
        active_end_idx = min(len(smooth) - 1, active[-1] + 2)
        active_start_ms = max(0.0, float(active_start_idx) * step_ms)
        active_end_ms = min(duration_ms, float(active_end_idx + 1) * step_ms)
    else:
        active_start_ms = 0.0
        active_end_ms = duration_ms

    min_active_ms = min(duration_ms, max(80.0, duration_ms * 0.18))
    if active_end_ms - active_start_ms < min_active_ms:
        active_start_ms = 0.0
        active_end_ms = duration_ms

    return {
        "duration_ms": duration_ms,
        "step_ms": step_ms,
        "smooth": smooth,
        "flux": flux,
        "threshold": threshold,
        "active_start_ms": active_start_ms,
        "active_end_ms": active_end_ms,
    }


def _frame_value(profile: dict[str, object] | None, key: str, time_ms: float) -> float:
    if not profile:
        return 0.0
    values = profile.get(key)
    if not isinstance(values, list) or not values:
        return 0.0
    step_ms = max(1e-6, float(profile.get("step_ms", 10.0) or 10.0))
    idx = max(0, min(len(values) - 1, int(round(float(time_ms) / step_ms))))
    try:
        return float(values[idx])
    except Exception:
        return 0.0


def _time_to_frame(profile: dict[str, object], time_ms: float) -> int:
    values = profile.get("smooth")
    n = len(values) if isinstance(values, list) else 0
    if n <= 0:
        return 0
    step_ms = max(1e-6, float(profile.get("step_ms", 10.0) or 10.0))
    return max(0, min(n - 1, int(round(float(time_ms) / step_ms))))


def _best_local_time(
    profile: dict[str, object] | None,
    *,
    start_ms: float,
    end_ms: float,
    key: str,
    prefer_max: bool,
    default_ms: float,
) -> float:
    if not profile:
        return float(default_ms)
    values = profile.get(key)
    if not isinstance(values, list) or not values:
        return float(default_ms)
    lo = _time_to_frame(profile, start_ms)
    hi = _time_to_frame(profile, end_ms)
    if hi < lo:
        lo, hi = hi, lo
    if hi <= lo:
        return float(default_ms)
    window = values[lo : hi + 1]
    if not window:
        return float(default_ms)
    target_value = max(window) if prefer_max else min(window)
    local_idx = window.index(target_value)
    step_ms = max(1e-6, float(profile.get("step_ms", 10.0) or 10.0))
    return float(lo + local_idx) * step_ms


def _alias_is_vowel_only(alias: str) -> bool:
    text = str(alias or "").strip().lower()
    if not text or " " in text or text.startswith("- "):
        return False
    return text in {
        "a", "i", "u", "e", "o", "eu", "eo", "ui", "wi", "wa", "we", "wo", "ya", "ye", "yo", "yu",
        "아", "이", "우", "에", "오", "어", "으", "애", "외", "위", "와", "왜", "워", "여", "야", "요", "유",
        "あ", "い", "う", "え", "お", "ん",
    }


def _candidate_penalty_for_activity(
    profile: dict[str, object] | None,
    *,
    candidate_ms: float,
    target_ms: float,
    active_lead_ms: float,
    active_tail_ms: float,
) -> float:
    score = abs(float(candidate_ms) - float(target_ms))
    if not profile:
        return score
    threshold = float(profile.get("threshold", 0.0) or 0.0)
    rms = _frame_value(profile, "smooth", candidate_ms)
    active_start = float(profile.get("active_start_ms", 0.0) or 0.0)
    active_end = float(profile.get("active_end_ms", profile.get("duration_ms", 0.0)) or 0.0)
    if candidate_ms < active_start - active_lead_ms:
        score += 200.0 + ((active_start - active_lead_ms) - candidate_ms) * 1.8
    if active_end > 0.0 and candidate_ms > active_end + active_tail_ms:
        score += 220.0 + (candidate_ms - (active_end + active_tail_ms)) * 1.8
    if threshold > 0.0 and rms < threshold * 0.55:
        score += 90.0
    elif threshold > 0.0 and rms < threshold * 0.85:
        score += 34.0
    return score


def _pick_scored_offset(
    *,
    profile: dict[str, object] | None,
    alias: str,
    alias_type: str,
    seg_start: float,
    seg_end: float,
    expected_offset: float,
    base_offset: float,
    seg_len: float,
) -> tuple[float, float, bool, str]:
    alias_key = str(alias_type or "").strip().lower()
    lead_ms = 18.0 if alias_key in {"cv_head", "cv", "mono"} else 10.0
    if _alias_is_vowel_only(alias):
        lead_ms = 6.0
    search_start = max(0.0, seg_start - max(50.0, seg_len * 0.18))
    search_end = max(search_start + 5.0, min(seg_end, seg_start + max(60.0, seg_len * 0.62)))
    onset_ms = _best_local_time(
        profile,
        start_ms=search_start,
        end_ms=search_end,
        key="flux",
        prefer_max=True,
        default_ms=seg_start,
    )
    rms_peak_ms = _best_local_time(
        profile,
        start_ms=seg_start,
        end_ms=seg_end,
        key="smooth",
        prefer_max=True,
        default_ms=seg_start + (seg_len * 0.45),
    )
    candidates: list[tuple[str, float, float]] = [
        ("expected", max(0.0, expected_offset), 0.0),
        ("onset", max(0.0, onset_ms - lead_ms), 8.0),
    ]
    if _alias_is_vowel_only(alias):
        candidates.append(("vowel_peak", max(0.0, rms_peak_ms - min(24.0, seg_len * 0.14)), 0.0))
    if base_offset > 0.0:
        candidates.append(("base", float(base_offset), 18.0))

    active_lead = max(8.0, min(28.0, lead_ms + 6.0))
    best = ("expected", max(0.0, expected_offset), float("inf"))
    second = float("inf")
    for name, cand, bias in candidates:
        score = bias + _candidate_penalty_for_activity(
            profile,
            candidate_ms=cand,
            target_ms=expected_offset,
            active_lead_ms=active_lead,
            active_tail_ms=20.0,
        )
        if score < best[2]:
            second = best[2]
            best = (name, cand, score)
        elif score < second:
            second = score
    margin = max(0.0, second - best[2]) if math.isfinite(second) else 999.0
    confidence = max(0.0, min(1.0, (margin / 55.0) + (1.0 - min(best[2], 120.0) / 150.0) * 0.55))
    low_conf = bool(confidence < 0.48 or best[2] > 82.0)
    return float(best[1]), float(confidence), low_conf, str(best[0])


def _estimate_segment_boundaries_ms(wav_path: str, segment_count: int, profile: dict[str, object] | None = None) -> list[float]:
    seg = int(segment_count or 0)
    if seg <= 1:
        duration_ms = _read_wav_duration_ms(wav_path)
        return [0.0, max(0.0, duration_ms)]
    analysis = profile if profile else _analyze_wav_activity(wav_path)
    try:
        smooth = list(analysis.get("smooth") or [])
        if not smooth:
            raise ValueError("empty energy")
        n = len(smooth)
        duration_ms = float(analysis.get("duration_ms", 0.0) or 0.0)
        step_ms = max(1e-6, float(analysis.get("step_ms", 10.0) or 10.0))
        active_start_ms = max(0.0, float(analysis.get("active_start_ms", 0.0) or 0.0))
        active_end_ms = min(duration_ms, float(analysis.get("active_end_ms", duration_ms) or duration_ms))
        active_lo = _time_to_frame(analysis, active_start_ms)
        active_hi = _time_to_frame(analysis, active_end_ms)
        if active_hi <= active_lo + seg:
            active_lo = 0
            active_hi = n - 1
        boundary_idx = [0]
        prev = active_lo
        for i in range(1, seg):
            target = int(round(active_lo + (float(i) * float(active_hi - active_lo) / float(seg))))
            search_radius = max(3, int(round(float(max(1, active_hi - active_lo)) / float(seg * 3))))
            lo = max(prev + 1, target - search_radius)
            hi = min(active_hi - 1, target + search_radius)
            if hi < lo:
                cut = max(prev + 1, min(max(prev + 1, active_hi - 1), target))
            else:
                local = smooth[lo : hi + 1]
                cut = lo + local.index(min(local))
            boundary_idx.append(cut)
            prev = cut
        boundary_idx.append(n - 1)

        out = [active_start_ms]
        for i in range(1, len(boundary_idx) - 1):
            out.append(max(0.0, min(duration_ms, float(boundary_idx[i]) * step_ms)))
        out.append(active_end_ms if active_end_ms > active_start_ms else duration_ms)
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
    activity_profile: dict[str, object] | None = None,
    slot_index: int,
    slot_total: int,
    language: str,
    blend_weight: float,
    format_type: str = "",
) -> tuple[str, bool, dict[str, object]]:
    if not parsed:
        return "", False, {}
    total_slots = max(int(slot_total), 1)
    idx = max(0, min(int(slot_index), total_slots - 1))
    if len(boundaries_ms) < total_slots + 1:
        return "", False, {}

    seg_start = float(boundaries_ms[idx])
    seg_end = float(boundaries_ms[idx + 1])
    wav_duration_ms = float(boundaries_ms[-1]) if boundaries_ms else 0.0
    if wav_duration_ms <= 0.0 or seg_end <= seg_start + 2.0:
        return "", False, {}

    alias_text = str(parsed.get("alias", ""))
    alias_type = _classify_alias_type(language, alias_text)
    pre_ratio, cons_ratio, cut_ratio, ovl_ratio, off_ratio = {
        "cv_head": (0.56, 0.46, 0.82, 0.42, 0.12),
        "cv": (0.48, 0.40, 0.78, 0.38, 0.08),
        "vcv": (0.58, 0.36, 0.66, 0.52, 0.08),
        "vc": (0.62, 0.30, 0.56, 0.55, 0.08),
        "vv": (0.66, 0.36, 0.70, 0.58, 0.04),
        "mono": (0.50, 0.40, 0.74, 0.40, 0.04),
        "br": (0.30, 0.25, 0.45, 0.24, 0.05),
        "other": (0.50, 0.38, 0.72, 0.36, 0.08),
    }.get(str(alias_type or "").strip().lower(), (0.50, 0.38, 0.72, 0.36, 0.10))
    # CVVC 포맷에서 VC alias는 앞 CV와 바로 맞닿으므로 offset을 더 타이트하게 설정한다.
    fmt = str(format_type or "").strip().lower()
    if fmt == "cvvc" and str(alias_type or "").strip().lower() == "vc":
        off_ratio = 0.04

    seg_len = max(28.0, seg_end - seg_start)
    expected_offset = max(0.0, seg_start - (seg_len * off_ratio))
    base_offset = float(parsed.get("offset", 0.0))
    target_offset, offset_conf, low_confidence, offset_source = _pick_scored_offset(
        profile=activity_profile,
        alias=alias_text,
        alias_type=alias_type,
        seg_start=seg_start,
        seg_end=seg_end,
        expected_offset=expected_offset,
        base_offset=base_offset,
        seg_len=seg_len,
    )
    target_pre = max(24.0, min(260.0, seg_len * pre_ratio))
    if activity_profile and str(alias_type or "").strip().lower() in {"cv_head", "cv", "mono", "vv"}:
        vowel_peak_ms = _best_local_time(
            activity_profile,
            start_ms=max(seg_start, target_offset),
            end_ms=seg_end,
            key="smooth",
            prefer_max=True,
            default_ms=target_offset + target_pre,
        )
        peak_pre = max(18.0, min(280.0, vowel_peak_ms - target_offset))
        peak_weight = 0.65 if (_alias_is_vowel_only(alias_text) or str(alias_type).strip().lower() == "cv_head") else 0.38
        target_pre = _blend_value(target_pre, peak_pre, peak_weight)
    target_cons = target_pre + max(16.0, min(250.0, seg_len * cons_ratio))
    right_blank_struct = target_cons + max(30.0, min(360.0, seg_len * cut_ratio))
    active_end = (
        float(activity_profile.get("active_end_ms", wav_duration_ms) or wav_duration_ms)
        if activity_profile
        else wav_duration_ms
    )
    right_blank_boundary = max(6.0, min(wav_duration_ms, active_end) - seg_end + (seg_len * 0.08))
    target_cutoff = -max(right_blank_struct, right_blank_boundary)
    target_ovl = max(0.0, min(target_pre * 0.86, target_pre * ovl_ratio))

    row_is_zero = _timing_row_is_zeroish(parsed)
    base_weight = 1.0 if row_is_zero else max(0.05, min(0.95, float(blend_weight)))
    if low_confidence:
        base_weight = max(base_weight, 0.74)

    base_cons = float(parsed.get("cons", 0.0))
    base_cutoff = float(parsed.get("cutoff", 0.0))
    base_pre = float(parsed.get("pre", 0.0))
    base_ovl = float(parsed.get("ovl", 0.0))

    offset_weight = 1.0 if abs(base_offset) <= 1e-6 else base_weight
    if base_offset > seg_end + 2.0 or base_offset < max(0.0, seg_start - seg_len):
        offset_weight = max(offset_weight, 0.70)
    if low_confidence or offset_source != "base":
        offset_weight = max(offset_weight, 0.82)
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

    active_start = (
        float(activity_profile.get("active_start_ms", 0.0) or 0.0)
        if activity_profile
        else 0.0
    )
    active_end = (
        float(activity_profile.get("active_end_ms", wav_duration_ms) or wav_duration_ms)
        if activity_profile
        else wav_duration_ms
    )
    min_offset = max(0.0, active_start - (18.0 if str(alias_type).strip().lower() in {"cv_head", "cv"} else 10.0))
    max_offset = max(0.0, min(wav_duration_ms - 1.0, active_end + 12.0, seg_end - 2.0))
    o2 = max(min_offset, min(max_offset, o2))
    rewritten = f"{parsed['wav']}={parsed['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{pr2:.2f},{ov2:.2f}"
    changed = (
        abs(float(o2) - base_offset) > 1e-6
        or abs(float(c2) - base_cons) > 1e-6
        or abs(float(ct2) - base_cutoff) > 1e-6
        or abs(float(pr2) - base_pre) > 1e-6
        or abs(float(ov2) - base_ovl) > 1e-6
    )
    quality = {
        "confidence": float(offset_conf),
        "low_confidence": bool(low_confidence),
        "offset_source": offset_source,
        "alias_type": str(alias_type or ""),
    }
    return rewritten, changed, quality


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
    normalized_out_path = _normalize_output_oto_path(out_path)
    mode = _normalize_generation_mode(generation_mode)
    _set_last_no_mfa_runtime_meta(
        requested_mode=str(generation_mode or ""),
        resolved_mode=str(mode),
        confidence=0.0,
        warnings=[],
        fallback_hint="",
        fallback_used=False,
    )
    if mode == "mfa_free_ssl_slot":
        checkpoint = _resolve_default_mfa_free_checkpoint()
        resolved_format_type = (
            str(os.environ.get("UTOA_NO_MFA_FORMAT_TYPE", "") or "").strip()
            or str(os.environ.get("UTOA_AUTO_FORMAT", "") or "").strip()
            or "CV"
        )
        try:
            from core.mfa_free_oto.workflow import (
                NoMfaWorkflowGuard,
                generate_no_mfa_oto_with_model_context,
            )

            runtime_report = generate_no_mfa_oto_with_model_context(
                wav_dir=wav_dir,
                out_path=normalized_out_path,
                source_oto_path=source_oto_path,
                alias_suffix=alias_suffix,
                checkpoint_path=checkpoint if (checkpoint and os.path.isfile(checkpoint)) else "",
                language=str(language or "japanese"),
                format_type=resolved_format_type,
                alias_type=str(os.environ.get("UTOA_NO_MFA_ALIAS_TYPE", "auto") or "auto"),
                encoder=(str(os.environ.get("UTOA_MFA_FREE_OTO_ENCODER", "") or "").strip() or "acoustic_world_v1"),
                device=(str(os.environ.get("UTOA_MFA_FREE_OTO_DEVICE", "") or "").strip() or None),
                use_slot_viterbi=not _env_bool("UTOA_MFA_FREE_OTO_DISABLE_SLOT_VITERBI", False),
                guard=NoMfaWorkflowGuard(
                    min_confidence=_env_float("UTOA_NO_MFA_RUNTIME_MIN_CONFIDENCE", 0.52),
                    min_anchor_ratio=_env_float("UTOA_NO_MFA_RUNTIME_MIN_ANCHOR_RATIO", 0.65),
                    max_slot_warnings=max(1, _env_int("UTOA_NO_MFA_RUNTIME_MAX_SLOT_WARNINGS", 6)),
                    cv_boundary_min_conf=_env_float("UTOA_NO_MFA_RUNTIME_MIN_CV_CONFIDENCE", 0.32),
                    vowel_nucleus_min_conf=_env_float("UTOA_NO_MFA_RUNTIME_MIN_NUCLEUS_CONFIDENCE", 0.34),
                    vc_pre_max_ms=_env_float("UTOA_NO_MFA_RUNTIME_VC_PRE_MAX_MS", 80.0),
                    one_step_shift_repair_enabled=_env_bool("UTOA_NO_MFA_RUNTIME_ONE_STEP_REPAIR", True),
                ),
                callback=callback,
            )
            runtime_metrics = dict(runtime_report.metrics or {})
            consistency_report = _apply_no_mfa_file_consistency(
                language=str(language or "japanese"),
                oto_path=normalized_out_path,
                format_type=resolved_format_type,
                callback=callback,
            )
            if bool(consistency_report.get("enabled")):
                runtime_metrics["file_consistency_changed"] = float(consistency_report.get("total_changed", 0) or 0)
                runtime_metrics["file_consistency_row_plan_ordered_wavs"] = float(
                    consistency_report.get("row_plan_ordered_wavs", 0) or 0
                )
                runtime_metrics["file_consistency_row_plan_fallback_wavs"] = float(
                    consistency_report.get("row_plan_fallback_wavs", 0) or 0
                )
            boundary_candidate_report = _apply_no_mfa_boundary_candidate_postprocess(
                language=str(language or "japanese"),
                oto_path=normalized_out_path,
                wav_dir=wav_dir,
                format_type=resolved_format_type,
                callback=callback,
            )
            if bool(boundary_candidate_report.get("enabled")):
                runtime_metrics["boundary_candidate_postprocess_requested"] = 1.0
                runtime_metrics["boundary_candidate_postprocess_changed"] = float(
                    boundary_candidate_report.get("changed_rows", 0) or 0
                )
                runtime_metrics["boundary_candidate_score_maps_loaded"] = float(
                    boundary_candidate_report.get("score_maps_loaded", 0) or 0
                )
                adjusted_by_family = boundary_candidate_report.get("adjusted_by_family_label")
                if isinstance(adjusted_by_family, dict):
                    for family, labels in adjusted_by_family.items():
                        if not isinstance(labels, dict):
                            continue
                        runtime_metrics[f"boundary_candidate_adjusted_{family}"] = float(
                            sum(int(value or 0) for value in labels.values())
                        )
            if _env_bool("UTOA_MFA_FREE_HSMM_LIGHTGBM_ENABLE", False):
                lightgbm_changed = _apply_no_mfa_ml_correction(
                    language=str(language or "japanese"),
                    oto_path=normalized_out_path,
                    wav_dir=wav_dir,
                    format_type=resolved_format_type,
                    callback=callback,
                )
                runtime_metrics["lightgbm_postprocess_requested"] = 1.0
                runtime_metrics["lightgbm_postprocess_changed"] = float(lightgbm_changed)
            baseline_metrics = _load_runtime_baseline_metrics()
            runtime_delta = _build_runtime_delta(runtime_metrics, baseline_metrics)
            _set_last_no_mfa_runtime_meta(
                requested_mode=str(generation_mode or ""),
                resolved_mode="mfa_free_ssl_slot",
                confidence=float(runtime_report.confidence),
                warnings=list(runtime_report.warnings),
                fallback_hint=str(runtime_report.fallback_hint or "manual_review_required"),
                fallback_used=bool(runtime_report.guard_failed),
                fallback_reason="manual_review_required" if runtime_report.guard_failed else "",
                checkpoint_path=str(checkpoint or ""),
                metrics=runtime_metrics,
                baseline_metrics=baseline_metrics,
                comparison=runtime_delta,
                boundary_candidate_postprocess=boundary_candidate_report,
            )
            _log(
                callback,
                f"[No-MFA/MFA-Free] generated={runtime_report.processed}/{runtime_report.total} "
                f"confidence={runtime_report.confidence:.2f} guard={'fail' if runtime_report.guard_failed else 'pass'}",
            )
            return int(runtime_report.processed), int(runtime_report.total), list(runtime_report.errors)
        except Exception as exc:
            _log(callback, f"[No-MFA/MFA-Free] runtime failed ({exc})")
            _set_last_no_mfa_runtime_meta(
                requested_mode=str(generation_mode or ""),
                resolved_mode="mfa_free_ssl_slot",
                confidence=0.0,
                warnings=[f"runtime_exception:{exc}"],
                fallback_hint="manual_review_required",
                fallback_used=True,
                fallback_reason="runtime_exception",
            )
            return 0, 0, [f"MFA-Free runtime failed: {exc}"]

    _log(
        callback,
        "[No-MFA] generation mode: remap (base OTO timing remap + acoustic correction)",
    )
    source = resolve_no_mfa_source_oto(
        wav_dir=wav_dir,
        source_hint=source_oto_path,
    )
    if not source:
        return 0, 0, ["No-MFA 자동설정용 베이스 OTO를 찾지 못했습니다."]

    source_rows = _load_oto_lines(source, callback)
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
    timing_zero_only = False
    remap_blend_weight = max(0.05, min(0.95, _env_float("UTOA_NO_MFA_REMAP_BLEND", 0.38)))
    if timing_profile is None:
        timing_profile = _build_builtin_timing_profile(language=language)
        if timing_profile:
            timing_zero_only = True
            _log(
                callback,
                f"[No-MFA] zero-timing fallback enabled: source={timing_profile.get('source', 'builtin')} "
                "(0,0,0,0,0 rows -> wav slot boundary timing; profile fallback on failure)",
            )
    _log(
        callback,
        f"[No-MFA] remap acoustic refine enabled (blend={remap_blend_weight:.2f}; 0-values use full auto correction)",
    )
    force_boundary_timing = bool(timing_zero_only and _env_bool("UTOA_NO_MFA_FORCE_BOUNDARY_TIMING", False))
    if force_boundary_timing:
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
    low_confidence_refined = 0
    offset_source_counts: dict[str, int] = {}
    confidence_values: list[float] = []
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
    wav_activity_cache: dict[str, dict[str, object]] = {}
    wav_root = os.path.abspath(str(wav_dir or "").strip())
    sequence_residual_model = _load_no_mfa_sequence_residual_model(callback)
    sequence_residual_changed = 0
    if sequence_residual_model:
        _log(
            callback,
            "[No-MFA] sequence residual correction enabled: "
            f"model={sequence_residual_model.get('_runtime_model_path', 'unknown')}, "
            f"scope={str(os.environ.get('UTOA_NO_MFA_SEQUENCE_RESIDUAL_SCOPE', 'vcv_pco') or 'vcv_pco')}, "
            f"blend={_env_float('UTOA_NO_MFA_SEQUENCE_RESIDUAL_BLEND', 0.35):.2f}",
        )

    for mapped_wav, right in prepared_entries:
        candidate = f"{mapped_wav}={right}"
        slot_idx = int(per_wav_index.get(mapped_wav, 0))
        per_wav_index[mapped_wav] = slot_idx + 1

        parsed_candidate = parse_oto_line(candidate)
        timing_applied = False
        slot_total = int(per_wav_total.get(mapped_wav, 1) or 1)
        activity_profile: dict[str, object] | None = None
        boundaries: list[float] = []
        quality: dict[str, object] = {}

        if mode == "remap" and parsed_candidate:
            wav_abs = os.path.join(wav_root, mapped_wav.replace("/", os.sep))
            activity_profile = wav_activity_cache.get(wav_abs)
            if activity_profile is None:
                activity_profile = _analyze_wav_activity(wav_abs)
                wav_activity_cache[wav_abs] = activity_profile
            boundary_key = (wav_abs, slot_total)
            boundaries = wav_boundary_cache.get(boundary_key)
            if boundaries is None:
                boundaries = _estimate_segment_boundaries_ms(wav_abs, slot_total, activity_profile)
                wav_boundary_cache[boundary_key] = boundaries
            if boundaries and len(boundaries) >= slot_total + 1:
                remapped, changed, quality = _refine_remap_timing_for_slot(
                    parsed=parsed_candidate,
                    boundaries_ms=boundaries,
                    activity_profile=activity_profile,
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
                    if bool((quality or {}).get("low_confidence")):
                        low_confidence_refined += 1
                    source_key = str((quality or {}).get("offset_source", "") or "unknown")
                    offset_source_counts[source_key] = int(offset_source_counts.get(source_key, 0)) + 1
                    try:
                        confidence_values.append(float((quality or {}).get("confidence", 0.0) or 0.0))
                    except Exception:
                        pass

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
        if sequence_residual_model:
            seq_line, seq_changed = _apply_no_mfa_sequence_residual_to_line(
                line=candidate,
                model=sequence_residual_model,
                boundaries_ms=boundaries,
                activity_profile=activity_profile,
                slot_index=slot_idx,
                slot_total=slot_total,
                language=language,
                format_type=_detected_format,
                quality=quality,
            )
            if seq_changed:
                candidate = seq_line
                sequence_residual_changed += 1
        candidate = apply_alias_suffix(candidate, alias_suffix)
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

    if not normalized_out_path:
        return 0, total, ["출력 OTO 경로가 비어 있습니다."]

    zero_placeholder_rows = []
    for line in out_lines:
        parsed_line = parse_oto_line(line)
        if parsed_line and _timing_row_is_zeroish(parsed_line):
            zero_placeholder_rows.append(line)
    if zero_placeholder_rows:
        sample = "; ".join(zero_placeholder_rows[:3])
        suffix = " ..." if len(zero_placeholder_rows) > 3 else ""
        msg = (
            "[ERROR] No-MFA 자동설정 placeholder OTO가 보정되지 않아 저장을 중단했습니다 "
            f"(zero_placeholder_rows={len(zero_placeholder_rows)}/{len(out_lines)}). "
            "WAV 파일 길이 읽기, 베이스 OTO 매칭, stats/reference OTO 설정을 확인하세요. "
            f"sample={sample}{suffix}"
        )
        _log(callback, msg)
        return 0, total, [msg]

    out_dir = os.path.dirname(os.path.abspath(normalized_out_path))
    try:
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(normalized_out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(out_lines).rstrip() + "\n")
    except OSError as exc:
        return 0, total, [f"출력 OTO 파일 저장 실패: {normalized_out_path} ({exc})"]

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
    if remap_refined:
        conf_mean = (sum(confidence_values) / float(len(confidence_values))) if confidence_values else 0.0
        source_desc = ", ".join(
            f"{key}:{value}" for key, value in sorted(offset_source_counts.items(), key=lambda item: item[0])
        )
        _log(
            callback,
            f"[No-MFA] acoustic candidate scoring: confidence_mean={conf_mean:.2f}, "
            f"low_confidence_rows={low_confidence_refined}, offset_sources={source_desc or 'n/a'}",
        )
    if sequence_residual_model:
        _log(callback, f"[No-MFA] sequence residual correction rows changed: {sequence_residual_changed}")
    if missing_wavs:
        sample = ", ".join(sorted(missing_wavs)[:5])
        suffix = "..." if len(missing_wavs) > 5 else ""
        _log(callback, f"[No-MFA] 매칭 실패 wav: {sample}{suffix}")
    _apply_no_mfa_ml_correction(
        language=language,
        oto_path=normalized_out_path,
        wav_dir=wav_dir,
        format_type=_detected_format,
        callback=callback,
    )
    return len(out_lines), total, []


__all__ = [
    "generate_no_mfa_auto_oto",
    "get_last_no_mfa_runtime_meta",
    "resolve_no_mfa_source_oto",
    "resolve_no_mfa_stats_oto",
]
