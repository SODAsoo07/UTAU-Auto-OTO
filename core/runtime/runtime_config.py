"""
Centralised runtime configuration for the OTO-ML pipeline.

All UTOA_* environment variables previously read in-line across oto_ml_refiner.py
(43 individual os.environ.get() calls) are consolidated here as a single typed
dataclass.  Code that needs a setting should call ``get_ml_config()`` rather than
reading the environment directly.

Design constraints
------------------
- Backward-compatible: every field has the same default that the old inline code used.
- Testable: call ``set_ml_config(MLRuntimeConfig(...))`` to inject a custom instance
  in tests or benchmarks without mutating os.environ.
- No circular imports: this module only imports from the standard library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers (private — not exported)
# ---------------------------------------------------------------------------

def _flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off", "n"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _float_bounded(name: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(float(os.environ.get(name, default)), hi))
    except Exception:
        return float(default)


def _optional_float_bounded(name: str, lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        return max(lo, min(float(raw), hi))
    except Exception:
        return None


def _str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLRuntimeConfig:
    """Typed snapshot of every UTOA_* environment variable used at runtime.

    Fields are grouped by concern.  Each field maps 1-to-1 to the env variable
    documented in its comment.  The ``from_env()`` class-method populates the
    instance from the current process environment; the no-arg constructor uses
    hard-coded defaults (useful in tests).
    """

    # ------------------------------------------------------------------
    # Global on/off
    # ------------------------------------------------------------------
    # UTOA_DISABLE_OTO_ML — set "1" to bypass ML entirely
    ml_disabled: bool = False

    # ------------------------------------------------------------------
    # Model path overrides
    # ------------------------------------------------------------------
    # UTOA_KR_OTO_ML_DIR / UTOA_JA_OTO_ML_DIR
    kr_model_dir: str = ""
    ja_model_dir: str = ""
    # UTOA_OTO_ML_WORKSPACE_ROOT
    workspace_root: str = ""
    # UTOA_OTO_ML_EXPORT_ROOT
    export_root: str = ""

    # ------------------------------------------------------------------
    # Backend / routing
    # ------------------------------------------------------------------
    # UTOA_ML_ROUTE ("legacy" | "e2e_hybrid")
    route: str = "legacy"
    # UTOA_ML_ENSEMBLE_ENABLE
    ensemble_enabled: bool = True
    # UTOA_ML_GATED_ENSEMBLE_ENABLE
    gated_ensemble_enabled: bool = False
    # UTOA_ML_LEGACY_FALLBACK_ENABLE
    legacy_fallback_enabled: bool = False
    # UTOA_ML_TWO_STAGE_MODEL_ENABLE
    two_stage_model_enabled: bool = True
    # UTOA_ML_AUTO_SELECT_VBEST
    auto_select_vbest: bool = False
    # UTOA_ML_SAME_LANGUAGE_BORROW_ONLY
    same_language_borrow_only: bool = True
    # UTOA_ML_COUPLED_DEVICE (e.g. "cpu", "cuda")
    coupled_device: str = ""

    # ------------------------------------------------------------------
    # Inference performance
    # ------------------------------------------------------------------
    # UTOA_ML_BATCH_INFERENCE_ENABLE
    batch_inference_enabled: bool = True
    # UTOA_ML_BATCH_INFERENCE_SIZE
    batch_inference_size: int = 256
    # UTOA_ML_PROGRESS_EVERY
    progress_every: int = 80
    # UTOA_ML_RUNTIME_BUNDLE_CACHE
    bundle_cache_enabled: bool = True

    # ------------------------------------------------------------------
    # Selector
    # ------------------------------------------------------------------
    # UTOA_ML_SELECTOR_ENABLE
    selector_enabled: bool = False
    # UTOA_DISABLE_SELECTOR_RUNTIME_GUARD
    selector_runtime_guard_disabled: bool = False
    # UTOA_SELECTOR_TOP1_MIN_GAIN
    selector_top1_min_gain: float = 0.0

    # ------------------------------------------------------------------
    # Delta / selector policy overrides
    # ------------------------------------------------------------------
    # UTOA_DISABLE_OTO_DELTA — force-disable delta application
    delta_disabled: bool = False
    # UTOA_FORCE_OTO_DELTA — force-enable delta application
    delta_forced: bool = False
    # UTOA_DISABLE_OTO_SELECTOR — force-disable candidate selector
    selector_disabled: bool = False
    # UTOA_FORCE_OTO_SELECTOR — force-enable candidate selector
    selector_forced: bool = False
    # UTOA_SELECTOR_MIN_MARGIN — global selector margin override (None = use table)
    selector_margin_override: Optional[float] = None

    # ------------------------------------------------------------------
    # Coupled confidence gates
    # ------------------------------------------------------------------
    # UTOA_ML_COUPLED_MIN_CONF  (global floor; None = use per-format defaults)
    coupled_min_conf: Optional[float] = None
    # UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META
    coupled_min_conf_use_model_meta: bool = True
    # UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET
    coupled_min_conf_model_offset: float = 0.0
    # UTOA_ML_COUPLED_MIN_CONF_CV_FAMILY  (optional alias-family override)
    coupled_min_conf_cv_family: Optional[float] = None
    # UTOA_ML_COUPLED_MIN_CONF_BRIDGE_FAMILY
    coupled_min_conf_bridge_family: Optional[float] = None
    # UTOA_ML_COUPLED_STRICT_CONSTRAINT
    coupled_strict_constraint: bool = True
    # UTOA_ML_ANCHOR_LOCK_MIN_CONF
    anchor_lock_min_conf: float = 0.45

    # ------------------------------------------------------------------
    # Blank / silence floor (Korean)
    # ------------------------------------------------------------------
    # UTOA_KR_CVVC_ROW_BLANK_FLOOR
    kr_cvvc_blank_floor: float = 0.64
    # UTOA_KR_CVC_ROW_BLANK_FLOOR
    kr_cvc_blank_floor: float = 0.62
    # UTOA_KR_CV_ROW_BLANK_FLOOR
    kr_cv_blank_floor: float = 0.60

    # ------------------------------------------------------------------
    # CV overlap cap ratio
    # ------------------------------------------------------------------
    # UTOA_ML_KR_CV_OVL_CAP_RATIO  (None = use formula-based default)
    kr_cv_ovl_cap_ratio: Optional[float] = None
    # UTOA_ML_JA_CV_OVL_CAP_RATIO
    ja_cv_ovl_cap_ratio: Optional[float] = None

    # ------------------------------------------------------------------
    # Korean CV placement policy
    # ------------------------------------------------------------------
    # UTOA_ML_KR_CV_KEEP_BASE_LOCATION
    kr_cv_keep_base_location: bool = True
    # UTOA_ML_HEAD_ZERO_OFFSET_GUARD
    head_zero_offset_guard: bool = True

    # ------------------------------------------------------------------
    # High-confidence row skip
    # ------------------------------------------------------------------
    # UTOA_ML_SKIP_HIGH_CONF_RULE_ROWS
    skip_high_conf_rule_rows: bool = True
    # UTOA_ML_SKIP_HIGH_CONF_MIN  (None = disabled)
    skip_high_conf_min: Optional[float] = None
    # UTOA_ML_SKIP_HIGH_CONF_MAX_BLANK
    skip_high_conf_max_blank: Optional[float] = None
    # UTOA_ML_SKIP_HIGH_CONF_MIN_MARGIN
    skip_high_conf_min_margin: float = 12.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "MLRuntimeConfig":
        """Read all UTOA_* variables from the current process environment."""
        # route normalisation
        raw_route = _str("UTOA_ML_ROUTE", "legacy").lower()
        if raw_route in {"e2e_hybrid", "e2e", "hybrid"}:
            route = "e2e_hybrid"
        else:
            route = "legacy"

        return cls(
            ml_disabled=_flag("UTOA_DISABLE_OTO_ML", False),
            # paths
            kr_model_dir=_str("UTOA_KR_OTO_ML_DIR"),
            ja_model_dir=_str("UTOA_JA_OTO_ML_DIR"),
            workspace_root=_str("UTOA_OTO_ML_WORKSPACE_ROOT"),
            export_root=_str("UTOA_OTO_ML_EXPORT_ROOT"),
            # routing
            route=route,
            ensemble_enabled=_flag("UTOA_ML_ENSEMBLE_ENABLE", True),
            gated_ensemble_enabled=_flag("UTOA_ML_GATED_ENSEMBLE_ENABLE", False),
            legacy_fallback_enabled=_flag("UTOA_ML_LEGACY_FALLBACK_ENABLE", False),
            two_stage_model_enabled=_flag("UTOA_ML_TWO_STAGE_MODEL_ENABLE", True),
            auto_select_vbest=_flag("UTOA_ML_AUTO_SELECT_VBEST", False),
            same_language_borrow_only=_flag("UTOA_ML_SAME_LANGUAGE_BORROW_ONLY", True),
            coupled_device=_str("UTOA_ML_COUPLED_DEVICE"),
            # inference perf
            batch_inference_enabled=_flag("UTOA_ML_BATCH_INFERENCE_ENABLE", True),
            batch_inference_size=max(32, _int("UTOA_ML_BATCH_INFERENCE_SIZE", 256)),
            progress_every=max(20, _int("UTOA_ML_PROGRESS_EVERY", 80)),
            bundle_cache_enabled=_flag("UTOA_ML_RUNTIME_BUNDLE_CACHE", True),
            # selector
            selector_enabled=_flag("UTOA_ML_SELECTOR_ENABLE", False),
            selector_runtime_guard_disabled=_flag("UTOA_DISABLE_SELECTOR_RUNTIME_GUARD", False),
            selector_top1_min_gain=float(os.environ.get("UTOA_SELECTOR_TOP1_MIN_GAIN", "0.0") or 0.0),
            # delta / selector policy
            delta_disabled=_flag("UTOA_DISABLE_OTO_DELTA", False),
            delta_forced=_flag("UTOA_FORCE_OTO_DELTA", False),
            selector_disabled=_flag("UTOA_DISABLE_OTO_SELECTOR", False),
            selector_forced=_flag("UTOA_FORCE_OTO_SELECTOR", False),
            selector_margin_override=_optional_float_bounded("UTOA_SELECTOR_MIN_MARGIN", lo=0.0, hi=999.0),
            # coupled confidence
            coupled_min_conf=_optional_float_bounded("UTOA_ML_COUPLED_MIN_CONF"),
            coupled_min_conf_use_model_meta=_flag("UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META", True),
            coupled_min_conf_model_offset=_float_bounded(
                "UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET", 0.0, lo=-0.30, hi=0.30
            ),
            coupled_min_conf_cv_family=_optional_float_bounded("UTOA_ML_COUPLED_MIN_CONF_CV_FAMILY"),
            coupled_min_conf_bridge_family=_optional_float_bounded("UTOA_ML_COUPLED_MIN_CONF_BRIDGE_FAMILY"),
            coupled_strict_constraint=_flag("UTOA_ML_COUPLED_STRICT_CONSTRAINT", True),
            anchor_lock_min_conf=_float_bounded("UTOA_ML_ANCHOR_LOCK_MIN_CONF", 0.45),
            # blank floors
            kr_cvvc_blank_floor=_float_bounded("UTOA_KR_CVVC_ROW_BLANK_FLOOR", 0.64),
            kr_cvc_blank_floor=_float_bounded("UTOA_KR_CVC_ROW_BLANK_FLOOR", 0.62),
            kr_cv_blank_floor=_float_bounded("UTOA_KR_CV_ROW_BLANK_FLOOR", 0.60),
            # CV overlap
            kr_cv_ovl_cap_ratio=_optional_float_bounded("UTOA_ML_KR_CV_OVL_CAP_RATIO", lo=0.40, hi=0.92),
            ja_cv_ovl_cap_ratio=_optional_float_bounded("UTOA_ML_JA_CV_OVL_CAP_RATIO", lo=0.40, hi=0.92),
            # Korean CV placement
            kr_cv_keep_base_location=_flag("UTOA_ML_KR_CV_KEEP_BASE_LOCATION", True),
            head_zero_offset_guard=_flag("UTOA_ML_HEAD_ZERO_OFFSET_GUARD", True),
            # high-conf skip
            skip_high_conf_rule_rows=_flag("UTOA_ML_SKIP_HIGH_CONF_RULE_ROWS", True),
            skip_high_conf_min=_optional_float_bounded("UTOA_ML_SKIP_HIGH_CONF_MIN"),
            skip_high_conf_max_blank=_optional_float_bounded("UTOA_ML_SKIP_HIGH_CONF_MAX_BLANK"),
            skip_high_conf_min_margin=float(
                os.environ.get("UTOA_ML_SKIP_HIGH_CONF_MIN_MARGIN", "12.0") or 12.0
            ),
        )

    def model_dir_for_language(self, language: str) -> str:
        """Return the explicit model-dir override for *language*, or ''."""
        lang = str(language or "").strip().lower()
        if lang == "japanese":
            return self.ja_model_dir
        return self.kr_model_dir

    def blank_floor_for_format(self, format_type: str) -> Optional[float]:
        """Return the Korean blank-risk floor for the given format, or None."""
        fmt = str(format_type or "").strip().lower()
        if fmt == "cvvc":
            return self.kr_cvvc_blank_floor
        if fmt == "cvc":
            return self.kr_cvc_blank_floor
        if fmt == "cv":
            return self.kr_cv_blank_floor
        return None

    def cv_ovl_cap_ratio_for_language(self, language: str) -> Optional[float]:
        """Return the explicit CV-overlap-cap override for *language*, or None."""
        lang = str(language or "").strip().lower()
        if lang == "japanese":
            return self.ja_cv_ovl_cap_ratio
        if lang == "korean":
            return self.kr_cv_ovl_cap_ratio
        return None


# ---------------------------------------------------------------------------
# Module-level singleton (lazily initialised from env on first access)
# ---------------------------------------------------------------------------

_ML_CONFIG: Optional[MLRuntimeConfig] = None


def get_ml_config() -> MLRuntimeConfig:
    """Return the process-wide ML runtime config (lazy-init from env)."""
    global _ML_CONFIG
    if _ML_CONFIG is None:
        _ML_CONFIG = MLRuntimeConfig.from_env()
    return _ML_CONFIG


def set_ml_config(config: MLRuntimeConfig) -> None:
    """Replace the process-wide ML runtime config (for tests / overrides)."""
    global _ML_CONFIG
    _ML_CONFIG = config


def reset_ml_config() -> None:
    """Force re-read from os.environ on next ``get_ml_config()`` call."""
    global _ML_CONFIG
    _ML_CONFIG = None


__all__ = [
    "MLRuntimeConfig",
    "get_ml_config",
    "set_ml_config",
    "reset_ml_config",
]
