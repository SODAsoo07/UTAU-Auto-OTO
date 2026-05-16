"""
OTO-ML policy — single source of truth for per-language / per-format defaults.

All behavioural constants (confidence floors, selector margins, error weights,
delta defaults, family splits) live here as Python dicts.  Runtime env-var
overrides are read via ``MLRuntimeConfig``; code outside this module should
never read ``UTOA_DISABLE_OTO_DELTA`` / ``UTOA_FORCE_OTO_SELECTOR`` etc.
directly — use the functions below instead.

To add a new preset/language variant, edit the ``_*`` constant dicts at the
top of this file.  No JSON files, no code duplication.
"""

from __future__ import annotations

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

# Runtime coupled min_conf defaults per language/format.
# These are used when per-format overrides are empty.
_COUPLED_MIN_CONF_DEFAULTS = {
    ("korean", "cv"): 0.55,
    ("korean", "cvc"): 0.75,
    ("korean", "cvvc"): 0.78,
    ("korean", "vcv"): 0.72,
    ("japanese", "cv"): 0.40,
    ("japanese", "cvc"): 0.50,
    ("japanese", "cvvc"): 0.72,
    ("japanese", "vcv"): 0.65,
}


_ALIAS_FAMILY_TYPES = {
    "cv": ["cv", "cv_head"],
    "vowel": ["mono"],
    "vc": ["vc", "vv"],
    "vcv": ["vcv"],
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
    ("korean", "cvc", "vc"): True,
    ("korean", "cvc", "vcv"): True,
    ("korean", "cvvc", "cv"): True,
    # Japanese CVVC stays guarded at runtime:
    # - cv family delta is disabled by default
    # - selector-only application is allowed
    # - post-guard clamps overly early timing
    ("japanese", "cvvc", ""): True,
    ("japanese", "cvvc", "cv"): True,
    ("japanese", "cvvc", "vc"): True,
    ("japanese", "cvvc", "vcv"): True,
}


_SELECTOR_MARGIN_DEFAULTS = {
    ("korean", "cv", ""): 0.06,
    ("korean", "cvc", ""): 0.06,
    ("korean", "cvvc", ""): 0.10,
    ("korean", "vcv", ""): 0.10,
    ("japanese", "cv", ""): 0.06,
    ("japanese", "cvvc", ""): 0.10,
    ("japanese", "vcv", ""): 0.10,
}


_SELECTOR_ERROR_WEIGHTS = {
    ("default", "general", ""): {
        "offset": 1.25,
        "cons": 0.85,
        "cutoff": 0.95,
        "pre": 1.10,
        "ovl": 0.45,
    },
    ("korean", "cvvc", ""): {
        "offset": 1.35,
        "cons": 0.90,
        "cutoff": 1.10,
        "pre": 1.25,
        "ovl": 0.35,
    },
    ("korean", "vcv", ""): {
        "offset": 1.30,
        "cons": 0.90,
        "cutoff": 1.05,
        "pre": 1.20,
        "ovl": 0.35,
    },
    ("japanese", "cvvc", ""): {
        "offset": 1.35,
        "cons": 0.90,
        "cutoff": 1.10,
        "pre": 1.25,
        "ovl": 0.35,
    },
    ("japanese", "vcv", ""): {
        "offset": 1.30,
        "cons": 0.90,
        "cutoff": 1.05,
        "pre": 1.20,
        "ovl": 0.35,
    },
}

_DELTA_DEFAULTS = {
    ("japanese", "cvvc", "cv"): False,
}


_FAMILY_SPLIT_DEFAULTS = {
    ("korean", "cvc"): ["cv", "vc", "vcv"],
    ("korean", "cvvc"): ["cv", "vc", "vcv"],
    ("japanese", "cvvc"): ["cv", "vc", "vcv"],
}


def normalize_alias_family(alias_family: str) -> str:
    value = str(alias_family or "").strip().lower()
    if value in {"", "all", "general", "any"}:
        return ""
    if value in {"mono", "v", "vowel", "single_vowel"}:
        return "vowel"
    if value in {"cv", "head", "head_cv"}:
        return "cv"
    if value in {"bridge"}:
        return "vc"
    if value in {"vc", "vv"}:
        return "vc"
    if value in {"vcv"}:
        return "vcv"
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
    if alias_type in {"vc", "vv"}:
        return "vc"
    if alias_type == "vcv":
        return "vcv"
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
    require_train_keep = True
    if family in {"vc", "vcv"}:
        min_conf = min(0.90, min_conf + 0.04)
        if fmt in {"cvc", "cvvc"}:
            # Bridge rows often fail the generic train_keep heuristic even when
            # the manual timing itself is valid enough for supervised learning.
            require_train_keep = False
    elif family == "vowel":
        min_conf = max(0.35, min_conf - 0.05)
    return {
        "require_train_keep": bool(require_train_keep),
        "min_mapping_confidence": float(min_conf),
        "exclude_nuclei_fallback": True,
    }


def default_coupled_min_conf(language: str, format_type: str) -> float:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    return float(_COUPLED_MIN_CONF_DEFAULTS.get((lang, fmt), 0.50))


def delta_enabled_by_default(language: str, format_type: str, alias_family: str = "") -> bool:
    from core.runtime_config import get_ml_config
    cfg = get_ml_config()
    if cfg.delta_disabled:
        return False
    if cfg.delta_forced:
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
    from core.runtime_config import get_ml_config
    cfg = get_ml_config()
    if cfg.selector_disabled:
        return False
    if cfg.selector_forced:
        return True
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    return bool(
        _SELECTOR_DEFAULTS.get((lang, fmt, family))
        or (_SELECTOR_DEFAULTS.get((lang, fmt, "")) if family else False)
    )


def selector_min_margin(language: str, format_type: str, alias_family: str = "") -> float:
    from core.runtime_config import get_ml_config
    override = get_ml_config().selector_margin_override
    if override is not None:
        return float(max(0.0, override))
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    for key in (
        (lang, fmt, family),
        (lang, fmt, ""),
        ("default", fmt, family),
        ("default", fmt, ""),
        ("default", "general", ""),
    ):
        if key in _SELECTOR_MARGIN_DEFAULTS:
            return float(_SELECTOR_MARGIN_DEFAULTS[key])
    return 0.0


def selector_error_weights(language: str, format_type: str, alias_family: str = "") -> Dict[str, float]:
    lang = str(language or "").strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    family = normalize_alias_family(alias_family)
    for key in (
        (lang, fmt, family),
        (lang, fmt, ""),
        ("default", fmt, family),
        ("default", fmt, ""),
        ("default", "general", ""),
    ):
        weights = _SELECTOR_ERROR_WEIGHTS.get(key)
        if weights:
            return dict(weights)
    return dict(_SELECTOR_ERROR_WEIGHTS[("default", "general", "")])


def get_policy_snapshot(language: str, format_type: str, alias_family: str = "") -> Dict[str, object]:
    """Return a single dict with every policy value for the given context.

    Useful for logging/debugging and for tests that want to assert on the
    complete policy without calling each function individually.
    """
    return {
        "language": language,
        "format_type": format_type,
        "alias_family": alias_family,
        "delta_enabled": delta_enabled_by_default(language, format_type, alias_family),
        "selector_enabled": selector_enabled_by_default(language, format_type, alias_family),
        "selector_min_margin": selector_min_margin(language, format_type, alias_family),
        "selector_error_weights": selector_error_weights(language, format_type, alias_family),
        "default_coupled_min_conf": default_coupled_min_conf(language, format_type),
        "training_filters": default_training_filters(language, format_type, alias_family),
        "recommended_family_splits": recommended_alias_family_splits(language, format_type),
    }


__all__ = [
    "alias_family_to_alias_types",
    "default_training_filters",
    "default_coupled_min_conf",
    "delta_enabled_by_default",
    "get_policy_snapshot",
    "infer_alias_family",
    "normalize_alias_family",
    "recommended_alias_family_splits",
    "should_split_alias_families",
    "selector_enabled_by_default",
    "selector_error_weights",
    "selector_min_margin",
]
