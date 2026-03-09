from __future__ import annotations

import os
from typing import Dict, List

from core.format_type_utils import normalize_format_type


_TRAINING_MIN_CONF = {
    ("korean", "cv"): 0.55,
    ("korean", "cvc"): 0.75,
    ("korean", "cvvc"): 0.78,
    ("korean", "vcv"): 0.72,
    ("japanese", "cv"): 0.40,
    ("japanese", "cvvc"): 0.72,
    ("japanese", "vcv"): 0.65,
}


_ALIAS_FAMILY_TYPES = {
    "cv": ["cv", "cv_head"],
    "vowel": ["mono"],
    "bridge": ["vc", "vv", "vcv"],
}


# Selector defaults are intentionally conservative.
# Only formats that showed stable gains after the latest retraining are enabled
# by default. Conditional candidates stay OFF until manual A/B verification.
_SELECTOR_DEFAULTS = {
    ("korean", "cv", ""): True,
    ("korean", "cv", "cv"): True,
    ("korean", "cv", "vowel"): True,
    ("korean", "cvc", ""): True,
    ("korean", "cvc", "cv"): True,
    ("korean", "cvc", "vowel"): True,
    ("korean", "cvc", "bridge"): True,
}

_DELTA_DEFAULTS = {
    ("japanese", "cvvc", "cv"): False,
}


_FAMILY_SPLIT_DEFAULTS = {
    ("korean", "cvc"): ["cv", "bridge"],
    ("korean", "cvvc"): ["cv", "bridge"],
    ("japanese", "cvvc"): ["cv", "bridge"],
}


def normalize_alias_family(alias_family: str) -> str:
    value = str(alias_family or "").strip().lower()
    if value in {"", "all", "general", "any"}:
        return ""
    if value in {"mono", "v", "vowel", "single_vowel"}:
        return "vowel"
    if value in {"cv", "head", "head_cv"}:
        return "cv"
    if value in {"bridge", "vc", "vv", "vcv"}:
        return "bridge"
    return value


def alias_family_to_alias_types(alias_family: str) -> List[str]:
    family = normalize_alias_family(alias_family)
    return list(_ALIAS_FAMILY_TYPES.get(family, []))


def infer_alias_family(language: str, row_context: Dict[str, object]) -> str:
    _ = normalize_format_type(language, row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if alias_type in {"cv", "cv_head"}:
        return "cv"
    if alias_type == "mono":
        return "vowel"
    if alias_type in {"vc", "vv", "vcv"}:
        return "bridge"
    return ""


def recommended_alias_family_splits(language: str, format_type: str) -> List[str]:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    return list(_FAMILY_SPLIT_DEFAULTS.get((lang, fmt), []))


def should_split_alias_families(language: str, format_type: str) -> bool:
    return bool(recommended_alias_family_splits(language, format_type))


def default_training_filters(language: str, format_type: str, alias_family: str = "") -> Dict[str, object]:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    min_conf = float(_TRAINING_MIN_CONF.get((lang, fmt), 0.50))
    if family == "bridge":
        min_conf = min(0.90, min_conf + 0.04)
    elif family == "vowel":
        min_conf = max(0.35, min_conf - 0.05)
    return {
        "require_train_keep": True,
        "min_mapping_confidence": float(min_conf),
        "exclude_nuclei_fallback": True,
        "use_pseudo_labels": False,
    }


def delta_enabled_by_default(language: str, format_type: str, alias_family: str = "") -> bool:
    if str(os.environ.get("UTOA_DISABLE_OTO_DELTA", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if str(os.environ.get("UTOA_FORCE_OTO_DELTA", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    if (lang, fmt, family) in _DELTA_DEFAULTS:
        return bool(_DELTA_DEFAULTS[(lang, fmt, family)])
    if family and (lang, fmt, "") in _DELTA_DEFAULTS:
        return bool(_DELTA_DEFAULTS[(lang, fmt, "")])
    return True


def selector_enabled_by_default(language: str, format_type: str, alias_family: str = "") -> bool:
    if str(os.environ.get("UTOA_DISABLE_OTO_SELECTOR", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if str(os.environ.get("UTOA_FORCE_OTO_SELECTOR", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    return bool(
        _SELECTOR_DEFAULTS.get((lang, fmt, family))
        or (_SELECTOR_DEFAULTS.get((lang, fmt, "")) if family else False)
    )


__all__ = [
    "alias_family_to_alias_types",
    "default_training_filters",
    "delta_enabled_by_default",
    "infer_alias_family",
    "normalize_alias_family",
    "recommended_alias_family_splits",
    "should_split_alias_families",
    "selector_enabled_by_default",
]
