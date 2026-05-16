from __future__ import annotations

import json
import os
from typing import Dict, Optional

_PROFILE_CACHE: Dict[str, object] = {
    "path": "",
    "mtime_ns": -1,
    "data": {},
}


def _default_profile_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", "profiles", "mel_refine_profiles.json")


def _profile_path() -> str:
    env_path = str(os.environ.get("UTOA_MEL_REFINE_PROFILE_JSON", "") or "").strip()
    if env_path:
        return os.path.expandvars(os.path.expanduser(env_path))
    return _default_profile_path()


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_profile_data() -> Dict[str, object]:
    path = _profile_path()
    try:
        stat = os.stat(path)
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    except Exception:
        mtime_ns = -1

    cached_path = str(_PROFILE_CACHE.get("path", ""))
    cached_mtime = int(_PROFILE_CACHE.get("mtime_ns", -2))
    if path == cached_path and mtime_ns == cached_mtime:
        data = _PROFILE_CACHE.get("data")
        if isinstance(data, dict):
            return data

    data: Dict[str, object] = {}
    if mtime_ns >= 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                parsed = json.load(f) or {}
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            data = {}

    _PROFILE_CACHE["path"] = path
    _PROFILE_CACHE["mtime_ns"] = mtime_ns
    _PROFILE_CACHE["data"] = data
    return data


def _env_float(name: str) -> Optional[float]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _profile_lookup(
    profile: Dict[str, object],
    *,
    language: str,
    key: str,
    alias_type: str = "",
    onset_group: str = "",
    format_type: str = "",
) -> Optional[float]:
    if not isinstance(profile, dict):
        return None
    lang = str(language or "").strip().lower()
    alias = str(alias_type or "").strip().lower()
    onset = str(onset_group or "").strip().lower()
    fmt = str(format_type or "").strip().lower()

    global_section = profile.get("global")
    if not isinstance(global_section, dict):
        global_section = {}
    lang_section = profile.get(lang)
    if not isinstance(lang_section, dict):
        lang_section = {}

    candidates = []
    if onset:
        onset_map = lang_section.get("onset_group")
        if isinstance(onset_map, dict):
            onset_values = onset_map.get(onset)
            if isinstance(onset_values, dict):
                candidates.append(onset_values)
    if alias:
        alias_map = lang_section.get("alias_type")
        if isinstance(alias_map, dict):
            alias_values = alias_map.get(alias)
            if isinstance(alias_values, dict):
                candidates.append(alias_values)
    if fmt:
        fmt_map = lang_section.get("format_type")
        if isinstance(fmt_map, dict):
            fmt_values = fmt_map.get(fmt)
            if isinstance(fmt_values, dict):
                candidates.append(fmt_values)

    lang_global = lang_section.get("global")
    if isinstance(lang_global, dict):
        candidates.append(lang_global)
    candidates.append(global_section)

    for source in candidates:
        if not isinstance(source, dict):
            continue
        value = _coerce_float(source.get(key))
        if value is not None:
            return value
    return None


def resolve_mel_refine_float(
    language: str,
    key: str,
    default: float,
    *,
    alias_type: str = "",
    onset_group: str = "",
    format_type: str = "",
) -> float:
    env_value = _env_float(key)
    if env_value is not None:
        return float(env_value)
    profile = _load_profile_data()
    profile_value = _profile_lookup(
        profile,
        language=language,
        key=key,
        alias_type=alias_type,
        onset_group=onset_group,
        format_type=format_type,
    )
    if profile_value is not None:
        return float(profile_value)
    return float(default)


__all__ = ["resolve_mel_refine_float"]
