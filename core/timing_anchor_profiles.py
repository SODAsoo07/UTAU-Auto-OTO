from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


# Staged rollout flags
ENABLE_ANCHOR_LOCK_JA_VCV = _env_flag("UTOA_ENABLE_ANCHOR_LOCK_JA_VCV", True)
ENABLE_ANCHOR_LOCK_JA_CVVC = _env_flag("UTOA_ENABLE_ANCHOR_LOCK_JA_CVVC", False)
ENABLE_ANCHOR_LOCK_KR = _env_flag("UTOA_ENABLE_ANCHOR_LOCK_KR", True)


@dataclass(frozen=True)
class AnchorTimingProfile:
    pre_window_before_ms: float
    pre_window_after_ms: float
    pre_floor_ms: float
    ovl_gap_min_ms: float
    ovl_gap_max_ms: float
    ovl_gap_target_ms: float
    cons_gap_min_ms: float
    cons_gap_max_ms: float
    cons_gap_target_ms: float
    cut_gap_min_ms: float
    cut_gap_max_ms: float
    cut_gap_target_ms: float
    max_anchor_shift_ratio: float = 0.06
    max_anchor_shift_min_ms: float = 45.0
    max_anchor_shift_max_ms: float = 140.0
    cut_to_next_onset_allow_ms: Optional[float] = None
    cut_to_next_vowel_allow_ms: Optional[float] = None
    blend_weight: float = 0.58
    lite_blend_weight: float = 0.26
    lite_shift_scale: float = 0.40


def _ja_vcv_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=8.0,
        pre_window_after_ms=4.0,
        pre_floor_ms=46.0,
        ovl_gap_min_ms=14.0,
        ovl_gap_max_ms=28.0,
        ovl_gap_target_ms=20.0,
        cons_gap_min_ms=72.0,
        cons_gap_max_ms=150.0,
        cons_gap_target_ms=96.0,
        cut_gap_min_ms=38.0,
        cut_gap_max_ms=108.0,
        cut_gap_target_ms=62.0,
        cut_to_next_onset_allow_ms=8.0,
    )


def _ja_vcv_n_bridge_profile() -> AnchorTimingProfile:
    # n+CV 브리지: pre를 약간 늦추고 cons/cut을 늘려 다음 CV 진입 체감을 확보.
    return AnchorTimingProfile(
        pre_window_before_ms=2.0,
        pre_window_after_ms=14.0,
        pre_floor_ms=50.0,
        ovl_gap_min_ms=10.0,
        ovl_gap_max_ms=24.0,
        ovl_gap_target_ms=15.0,
        cons_gap_min_ms=96.0,
        cons_gap_max_ms=186.0,
        cons_gap_target_ms=132.0,
        cut_gap_min_ms=50.0,
        cut_gap_max_ms=136.0,
        cut_gap_target_ms=90.0,
        cut_to_next_onset_allow_ms=10.0,
    )


def _ja_vcv_vv_like_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=3.0,
        pre_window_after_ms=8.0,
        pre_floor_ms=52.0,
        ovl_gap_min_ms=4.0,
        ovl_gap_max_ms=10.0,
        ovl_gap_target_ms=7.0,
        cons_gap_min_ms=84.0,
        cons_gap_max_ms=176.0,
        cons_gap_target_ms=116.0,
        cut_gap_min_ms=42.0,
        cut_gap_max_ms=122.0,
        cut_gap_target_ms=74.0,
        cut_to_next_onset_allow_ms=6.0,
    )


def _ja_cv_head_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=10.0,
        pre_window_after_ms=10.0,
        pre_floor_ms=42.0,
        ovl_gap_min_ms=12.0,
        ovl_gap_max_ms=28.0,
        ovl_gap_target_ms=18.0,
        cons_gap_min_ms=58.0,
        cons_gap_max_ms=140.0,
        cons_gap_target_ms=92.0,
        cut_gap_min_ms=36.0,
        cut_gap_max_ms=112.0,
        cut_gap_target_ms=64.0,
        cut_to_next_onset_allow_ms=6.0,
    )


def _ja_cvvc_vc_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=10.0,
        pre_window_after_ms=6.0,
        pre_floor_ms=28.0,
        ovl_gap_min_ms=8.0,
        ovl_gap_max_ms=18.0,
        ovl_gap_target_ms=12.0,
        cons_gap_min_ms=20.0,
        cons_gap_max_ms=62.0,
        cons_gap_target_ms=36.0,
        cut_gap_min_ms=12.0,
        cut_gap_max_ms=48.0,
        cut_gap_target_ms=26.0,
        cut_to_next_onset_allow_ms=4.0,
        cut_to_next_vowel_allow_ms=-2.0,
    )


def _ja_cvvc_vv_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=8.0,
        pre_window_after_ms=8.0,
        pre_floor_ms=34.0,
        ovl_gap_min_ms=4.0,
        ovl_gap_max_ms=12.0,
        ovl_gap_target_ms=8.0,
        cons_gap_min_ms=64.0,
        cons_gap_max_ms=150.0,
        cons_gap_target_ms=100.0,
        cut_gap_min_ms=22.0,
        cut_gap_max_ms=96.0,
        cut_gap_target_ms=54.0,
        cut_to_next_onset_allow_ms=6.0,
    )


def _kr_cv_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=7.0,
        pre_window_after_ms=7.0,
        pre_floor_ms=40.0,
        ovl_gap_min_ms=14.0,
        ovl_gap_max_ms=30.0,
        ovl_gap_target_ms=21.0,
        cons_gap_min_ms=58.0,
        cons_gap_max_ms=148.0,
        cons_gap_target_ms=96.0,
        cut_gap_min_ms=34.0,
        cut_gap_max_ms=120.0,
        cut_gap_target_ms=72.0,
        cut_to_next_onset_allow_ms=7.0,
    )


def _kr_cv_head_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=7.0,
        pre_window_after_ms=7.0,
        pre_floor_ms=40.0,
        ovl_gap_min_ms=12.0,
        ovl_gap_max_ms=28.0,
        ovl_gap_target_ms=19.0,
        cons_gap_min_ms=60.0,
        cons_gap_max_ms=154.0,
        cons_gap_target_ms=98.0,
        cut_gap_min_ms=36.0,
        cut_gap_max_ms=128.0,
        cut_gap_target_ms=76.0,
        cut_to_next_onset_allow_ms=6.0,
    )


def _kr_vc_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=10.0,
        pre_window_after_ms=6.0,
        pre_floor_ms=24.0,
        ovl_gap_min_ms=8.0,
        ovl_gap_max_ms=20.0,
        ovl_gap_target_ms=13.0,
        cons_gap_min_ms=20.0,
        cons_gap_max_ms=64.0,
        cons_gap_target_ms=38.0,
        cut_gap_min_ms=10.0,
        cut_gap_max_ms=50.0,
        cut_gap_target_ms=28.0,
        cut_to_next_onset_allow_ms=4.0,
    )


def _kr_vv_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=8.0,
        pre_window_after_ms=8.0,
        pre_floor_ms=32.0,
        ovl_gap_min_ms=5.0,
        ovl_gap_max_ms=12.0,
        ovl_gap_target_ms=8.0,
        cons_gap_min_ms=58.0,
        cons_gap_max_ms=146.0,
        cons_gap_target_ms=98.0,
        cut_gap_min_ms=20.0,
        cut_gap_max_ms=92.0,
        cut_gap_target_ms=52.0,
        cut_to_next_onset_allow_ms=6.0,
    )


def _kr_vcv_profile() -> AnchorTimingProfile:
    return AnchorTimingProfile(
        pre_window_before_ms=8.0,
        pre_window_after_ms=6.0,
        pre_floor_ms=46.0,
        ovl_gap_min_ms=12.0,
        ovl_gap_max_ms=28.0,
        ovl_gap_target_ms=19.0,
        cons_gap_min_ms=72.0,
        cons_gap_max_ms=168.0,
        cons_gap_target_ms=108.0,
        cut_gap_min_ms=36.0,
        cut_gap_max_ms=132.0,
        cut_gap_target_ms=78.0,
        cut_to_next_onset_allow_ms=8.0,
        cut_to_next_vowel_allow_ms=2.0,
    )


_PROFILE_TABLE: Dict[Tuple[str, str, str], AnchorTimingProfile] = {
    ("japanese", "vcv", "vcv"): _ja_vcv_profile(),
    ("japanese", "vcv", "vcv_n_bridge"): _ja_vcv_n_bridge_profile(),
    ("japanese", "vcv", "vcv_vv_like"): _ja_vcv_vv_like_profile(),
    ("japanese", "vcv", "cv"): _ja_cv_head_profile(),
    ("japanese", "vcv", "cv_head"): _ja_cv_head_profile(),
    ("japanese", "cv", "cv"): _ja_cv_head_profile(),
    ("japanese", "cv", "cv_head"): _ja_cv_head_profile(),
    ("japanese", "cvvc", "vc"): _ja_cvvc_vc_profile(),
    ("japanese", "cvvc", "vv"): _ja_cvvc_vv_profile(),
    ("japanese", "cvvc", "cv_head"): _ja_cv_head_profile(),
    ("japanese", "cvvc", "cv"): _ja_cv_head_profile(),
    ("korean", "cv", "cv"): _kr_cv_profile(),
    ("korean", "cv", "cv_head"): _kr_cv_head_profile(),
    ("korean", "cv", "vc"): _kr_vc_profile(),
    ("korean", "cv", "vv"): _kr_vv_profile(),
    ("korean", "cvvc", "cv"): _kr_cv_profile(),
    ("korean", "cvvc", "cv_head"): _kr_cv_head_profile(),
    ("korean", "cvvc", "vc"): _kr_vc_profile(),
    ("korean", "cvvc", "vv"): _kr_vv_profile(),
    ("korean", "cvc", "cv"): _kr_cv_profile(),
    ("korean", "cvc", "cv_head"): _kr_cv_head_profile(),
    ("korean", "cvc", "vc"): _kr_vc_profile(),
    ("korean", "cvc", "vv"): _kr_vv_profile(),
    ("korean", "vcv", "cv"): _kr_cv_profile(),
    ("korean", "vcv", "cv_head"): _kr_cv_head_profile(),
    ("korean", "vcv", "vc"): _kr_vc_profile(),
    ("korean", "vcv", "vv"): _kr_vv_profile(),
    ("korean", "vcv", "vcv"): _kr_vcv_profile(),
}


def _default_kr_anchor_profile_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "ml", "configs", "kr_vcv_anchor_profile.yaml")


def _resolve_kr_anchor_profile_path() -> str:
    env_path = os.environ.get("UTOA_KR_ANCHOR_PROFILE_PATH", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    return _default_kr_anchor_profile_path()


_EXTERNAL_CACHE = {
    "path": "",
    "mtime_ns": -1,
    "table": None,
}


def _to_float(v: object, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _profile_from_dict(raw: Dict[str, object], base: AnchorTimingProfile) -> AnchorTimingProfile:
    if not isinstance(raw, dict):
        return base
    return AnchorTimingProfile(
        pre_window_before_ms=_to_float(raw.get("pre_window_before_ms"), base.pre_window_before_ms),
        pre_window_after_ms=_to_float(raw.get("pre_window_after_ms"), base.pre_window_after_ms),
        pre_floor_ms=_to_float(raw.get("pre_floor_ms"), base.pre_floor_ms),
        ovl_gap_min_ms=_to_float(raw.get("ovl_gap_min_ms"), base.ovl_gap_min_ms),
        ovl_gap_max_ms=_to_float(raw.get("ovl_gap_max_ms"), base.ovl_gap_max_ms),
        ovl_gap_target_ms=_to_float(raw.get("ovl_gap_target_ms"), base.ovl_gap_target_ms),
        cons_gap_min_ms=_to_float(raw.get("cons_gap_min_ms"), base.cons_gap_min_ms),
        cons_gap_max_ms=_to_float(raw.get("cons_gap_max_ms"), base.cons_gap_max_ms),
        cons_gap_target_ms=_to_float(raw.get("cons_gap_target_ms"), base.cons_gap_target_ms),
        cut_gap_min_ms=_to_float(raw.get("cut_gap_min_ms"), base.cut_gap_min_ms),
        cut_gap_max_ms=_to_float(raw.get("cut_gap_max_ms"), base.cut_gap_max_ms),
        cut_gap_target_ms=_to_float(raw.get("cut_gap_target_ms"), base.cut_gap_target_ms),
        max_anchor_shift_ratio=_to_float(raw.get("max_anchor_shift_ratio"), base.max_anchor_shift_ratio),
        max_anchor_shift_min_ms=_to_float(raw.get("max_anchor_shift_min_ms"), base.max_anchor_shift_min_ms),
        max_anchor_shift_max_ms=_to_float(raw.get("max_anchor_shift_max_ms"), base.max_anchor_shift_max_ms),
        cut_to_next_onset_allow_ms=(
            None
            if raw.get("cut_to_next_onset_allow_ms") is None
            else _to_float(raw.get("cut_to_next_onset_allow_ms"), 0.0)
        ),
        cut_to_next_vowel_allow_ms=(
            None
            if raw.get("cut_to_next_vowel_allow_ms") is None
            else _to_float(raw.get("cut_to_next_vowel_allow_ms"), 0.0)
        ),
        blend_weight=_to_float(raw.get("blend_weight"), base.blend_weight),
        lite_blend_weight=_to_float(raw.get("lite_blend_weight"), base.lite_blend_weight),
        lite_shift_scale=_to_float(raw.get("lite_shift_scale"), base.lite_shift_scale),
    )


def _load_external_profile_table() -> Dict[Tuple[str, str, str], AnchorTimingProfile]:
    path = _resolve_kr_anchor_profile_path()
    if not path or not os.path.isfile(path):
        return {}
    try:
        st = os.stat(path)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        return {}
    if (
        _EXTERNAL_CACHE.get("table") is not None
        and _EXTERNAL_CACHE.get("path") == path
        and int(_EXTERNAL_CACHE.get("mtime_ns", -1)) == mtime_ns
    ):
        return dict(_EXTERNAL_CACHE.get("table") or {})

    payload = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if yaml is not None:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
    except Exception:
        payload = None

    out: Dict[Tuple[str, str, str], AnchorTimingProfile] = {}
    if isinstance(payload, dict):
        raw_profiles = payload.get("korean", {}).get("vcv", {})
        if isinstance(raw_profiles, dict):
            for alias_type, raw_entry in raw_profiles.items():
                key = ("korean", "vcv", str(alias_type or "").strip().lower())
                base = _PROFILE_TABLE.get(key)
                if base is None:
                    continue
                out[key] = _profile_from_dict(raw_entry, base)

    _EXTERNAL_CACHE["path"] = path
    _EXTERNAL_CACHE["mtime_ns"] = mtime_ns
    _EXTERNAL_CACHE["table"] = dict(out)
    return out


def _merged_profile_table() -> Dict[Tuple[str, str, str], AnchorTimingProfile]:
    table = dict(_PROFILE_TABLE)
    table.update(_load_external_profile_table())
    return table


def is_anchor_lock_enabled(language: str, format_type: str) -> bool:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if lang == "japanese":
        if fmt == "vcv":
            return ENABLE_ANCHOR_LOCK_JA_VCV
        if fmt in {"cvvc", "cv"}:
            return ENABLE_ANCHOR_LOCK_JA_CVVC
        return False
    if lang == "korean":
        if fmt in {"cv", "cvvc", "cvc", "vcv"}:
            return ENABLE_ANCHOR_LOCK_KR
        return False
    return False


def get_anchor_profile(
    language: str,
    format_type: str,
    alias_type: str,
    mode: str = "rhythm_stable",
) -> Optional[AnchorTimingProfile]:
    if str(mode or "").strip().lower() != "rhythm_stable":
        return None
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    alias = str(alias_type or "").strip().lower()
    return _merged_profile_table().get((lang, fmt, alias))


__all__ = [
    "AnchorTimingProfile",
    "ENABLE_ANCHOR_LOCK_JA_VCV",
    "ENABLE_ANCHOR_LOCK_JA_CVVC",
    "ENABLE_ANCHOR_LOCK_KR",
    "get_anchor_profile",
    "is_anchor_lock_enabled",
]
