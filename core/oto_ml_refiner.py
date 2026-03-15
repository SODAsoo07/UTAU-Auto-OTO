"""
Runtime OTO ML refiner.

Applies per-row delta corrections to generated OTO files while preserving existing
validation and mel-based safety guards.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from core.oto_ml_features import extract_feature_rows, get_delta_clip_limits, parse_oto_rows
from core.oto_ml.features.mel_patches import (
    build_wav_index,
    extract_patches_for_row_with_quality,
    find_wav_path,
    resolve_mel_patch_anchors,
)
from core.oto_ml_runtime import (
    OtoDeltaResult,
    load_oto_model_bundle,
    predict_oto_deltas,
    predict_oto_deltas_batch,
)
from core.oto_ml_lightgbm import load_lightgbm_selector_bundle, predict_lightgbm_selector_score
from core.oto_ml_policy import (
    delta_enabled_by_default,
    default_coupled_min_conf,
    infer_alias_family,
    normalize_alias_family,
    selector_min_margin,
    selector_enabled_by_default,
)
from core.oto_ml_reliability import compute_blank_risk_score, is_mel_unreliable, mel_patch_is_fallback
from core.oto_ml_selector import select_best_candidate
from core.format_type_utils import normalize_format_type
from core.pipeline_status import (
    ML_APPLIED,
    ML_BUNDLE_INVALID,
    ML_DISABLED_ENV,
    ML_INFER_FAILED,
    ML_INPUT_MISSING,
    ML_MODEL_MISSING,
    ML_NO_FEATURES,
    ML_POLICY_OFF,
    ML_ROUTE_UNAVAILABLE,
    OK,
    make_runtime_report,
    normalize_ml_policy,
)
from core.timing_anchor_profiles import get_anchor_profile, is_anchor_lock_enabled
from core.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock
from core.selector_hard_negative import log_selector_hard_negative

logger = logging.getLogger(__name__)


class CoupledLowConfidenceError(RuntimeError):
    def __init__(self, confidence: float, threshold: float):
        super().__init__(f"coupled_low_confidence(conf={confidence:.4f}, min={threshold:.4f})")
        self.confidence = float(confidence)
        self.threshold = float(threshold)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _ensemble_enabled() -> bool:
    return _env_flag("UTOA_ML_ENSEMBLE_ENABLE", True)


def _ml_route(value: Optional[str] = None) -> str:
    raw = str(value or os.environ.get("UTOA_ML_ROUTE", "legacy") or "legacy").strip().lower()
    if raw in {"autofree_v1", "autofree", "b", "route_b"}:
        return "autofree_v1"
    return "legacy"


def _gated_ensemble_enabled() -> bool:
    # Default OFF to keep runtime stable: coupled is primary, LightGBM is fallback.
    return _env_flag("UTOA_ML_GATED_ENSEMBLE_ENABLE", False)


def _selector_runtime_enabled() -> bool:
    # Runtime 기본은 델타 보정 단일 모드(속도 우선). 필요 시 env로 명시적으로 켠다.
    return _env_flag("UTOA_ML_SELECTOR_ENABLE", False)


def _legacy_fallback_enabled() -> bool:
    # Coupled/Ensemble의 내부 게이팅을 우선 사용하고, 구형 fallback 체인은 기본 비활성.
    return _env_flag("UTOA_ML_LEGACY_FALLBACK_ENABLE", False)


def _batch_inference_enabled() -> bool:
    return _env_flag("UTOA_ML_BATCH_INFERENCE_ENABLE", True)


def _batch_inference_size() -> int:
    return max(32, _env_int("UTOA_ML_BATCH_INFERENCE_SIZE", 256))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _coupled_min_confidence() -> float:
    try:
        return max(0.0, min(float(os.environ.get("UTOA_ML_COUPLED_MIN_CONF", "0.42")), 1.0))
    except Exception:
        return 0.42


def _read_bounded_conf_env(name: str) -> Optional[float]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return None


def _coupled_min_conf_use_model_meta() -> bool:
    # 운영 기본은 모델 메타(min_confidence) 기반 분리 게이트를 사용한다.
    return _env_flag("UTOA_ML_COUPLED_MIN_CONF_USE_MODEL_META", True)


def _coupled_min_conf_model_offset() -> float:
    raw = str(os.environ.get("UTOA_ML_COUPLED_MIN_CONF_MODEL_OFFSET", "")).strip()
    if not raw:
        return 0.0
    try:
        return max(-0.30, min(float(raw), 0.30))
    except Exception:
        return 0.0


def _bounded_conf(value) -> Optional[float]:
    try:
        conf = float(value)
    except Exception:
        return None
    if conf < 0.0 or conf > 1.0:
        return None
    return float(conf)


def _min_conf_from_meta(meta: object) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    candidates = [
        meta.get("runtime_min_confidence"),
        meta.get("min_confidence"),
        meta.get("min_coupled_confidence"),
    ]
    gating = meta.get("gating")
    if isinstance(gating, dict):
        candidates.append(gating.get("min_coupled_confidence"))
    for cand in candidates:
        conf = _bounded_conf(cand)
        if conf is not None and conf > 0.0:
            return float(conf)
    return None


def _route_model_min_confidence(bundle: Optional[object], model_dir: str) -> Optional[float]:
    # Explicit global override takes precedence over model-wise defaulting.
    if _read_bounded_conf_env("UTOA_ML_COUPLED_MIN_CONF") is not None:
        return None
    if not _coupled_min_conf_use_model_meta():
        return None

    conf = None
    if bundle is not None:
        conf = _min_conf_from_meta(getattr(bundle, "meta", None))
    if conf is None and model_dir:
        try:
            import json

            meta_path = os.path.join(model_dir, "model_meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                conf = _min_conf_from_meta(meta)
        except Exception:
            conf = None
    if conf is None:
        return None

    conf = float(conf) + float(_coupled_min_conf_model_offset())
    return float(max(0.0, min(conf, 1.0)))


def _ml_anchor_lock_min_conf() -> float:
    env_val = _read_bounded_conf_env("UTOA_ML_ANCHOR_LOCK_MIN_CONF")
    if env_val is None:
        return 0.45
    return float(env_val)


def _ml_anchor_blank_floor(language: str, format_type: str) -> Optional[float]:
    lang = str(language or "").strip().lower()
    if lang != "korean":
        return None
    fmt = normalize_format_type(lang, format_type) or str(format_type or "").strip().lower()
    if fmt not in {"cvvc", "cvc", "cv"}:
        return None
    env_key_by_fmt = {
        "cvvc": "UTOA_KR_CVVC_ROW_BLANK_FLOOR",
        "cvc": "UTOA_KR_CVC_ROW_BLANK_FLOOR",
        "cv": "UTOA_KR_CV_ROW_BLANK_FLOOR",
    }
    default_floor_by_fmt = {
        "cvvc": 0.64,
        "cvc": 0.62,
        "cv": 0.60,
    }
    env_val = _read_bounded_conf_env(env_key_by_fmt.get(fmt, "UTOA_KR_CVVC_ROW_BLANK_FLOOR"))
    if env_val is None:
        return float(default_floor_by_fmt.get(fmt, 0.64))
    return float(env_val)


def _coupled_min_confidence_for_alias(
    alias_type: str,
    *,
    language: str = "",
    format_type: str = "",
    base_override: Optional[float] = None,
) -> float:
    base = float(base_override) if base_override is not None else _coupled_min_confidence()
    base = max(0.0, min(base, 1.0))
    alias = str(alias_type or "").strip().lower()
    if not alias:
        return float(base)

    lang = str(language or "").strip().lower()
    if lang.startswith("ko") or lang == "korean":
        lang_token = "KR"
    elif lang.startswith("ja") or lang == "japanese":
        lang_token = "JA"
    else:
        lang_token = "GEN"

    fmt = normalize_format_type(lang, format_type) if format_type else ""
    fmt_token = str(fmt or "GENERAL").strip().upper()
    alias_token = str(alias).strip().upper()
    keys = [
        f"UTOA_ML_COUPLED_MIN_CONF_{lang_token}_{fmt_token}_{alias_token}",
        f"UTOA_ML_COUPLED_MIN_CONF_{lang_token}_{fmt_token}",
        f"UTOA_ML_COUPLED_MIN_CONF_{lang_token}_{alias_token}",
        f"UTOA_ML_COUPLED_MIN_CONF_{alias_token}",
        f"UTOA_ML_COUPLED_MIN_CONF_{fmt_token}",
    ]
    if alias in {"cv", "cv_head"}:
        keys.append("UTOA_ML_COUPLED_MIN_CONF_CV_FAMILY")
    elif alias in {"vc", "vv", "vcv"}:
        keys.append("UTOA_ML_COUPLED_MIN_CONF_BRIDGE_FAMILY")

    for key in keys:
        value = _read_bounded_conf_env(key)
        if value is not None:
            return float(value)

    # If no per-format override is set, fall back to defaults by language/format.
    if _read_bounded_conf_env("UTOA_ML_COUPLED_MIN_CONF") is None:
        default_by_fmt = default_coupled_min_conf(lang, fmt)
        base = max(base, min(float(default_by_fmt), 1.0))
    if lang_token == "KR":
        # Korean CV 계열 기본 게이트: 과도한 fallback을 줄이되, CV head는 여전히 보수적으로 유지.
        kr_floor_by_alias = {
            "cv": 0.50,
            "cv_head": 0.52,
            "vc": 0.46,
            "vv": 0.48,
            "vcv": 0.48,
        }
        floor = kr_floor_by_alias.get(alias)
        if floor is not None:
            if alias in {"cv", "cv_head"} and fmt in {"cvvc", "cvc"}:
                floor += 0.01
            return float(max(base, min(float(floor), 0.95)))
    return float(base)


def _coupled_strict_constraint() -> bool:
    return _env_flag("UTOA_ML_COUPLED_STRICT_CONSTRAINT", False)


def _kr_cv_keep_base_location_enabled() -> bool:
    return _env_flag("UTOA_ML_KR_CV_KEEP_BASE_LOCATION", True)


def _head_zero_offset_guard_enabled() -> bool:
    return _env_flag("UTOA_ML_HEAD_ZERO_OFFSET_GUARD", True)


def _read_bundle_backend(model_dir: str) -> str:
    try:
        import json

        meta_path = os.path.join(model_dir, "model_meta.json")
        if not os.path.isfile(meta_path):
            return ""
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
        return str(meta.get("backend", "") or "").strip().lower()
    except Exception:
        return ""


def _has_explicit_model_override(language: str) -> bool:
    lang = str(language).strip().lower()
    env_key = "UTOA_JA_OTO_ML_DIR" if lang == "japanese" else "UTOA_KR_OTO_ML_DIR"
    return bool(os.environ.get(env_key, "").strip())


def _model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    env_key = "UTOA_JA_OTO_ML_DIR" if lang == "japanese" else "UTOA_KR_OTO_ML_DIR"
    env_path = os.environ.get(env_key, "").strip()
    if env_path:
        return env_path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", "models", "oto_ml", lang)


def _workspace_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.environ.get("UTOA_OTO_ML_WORKSPACE_ROOT", "").strip()
    if workspace_root:
        return os.path.join(workspace_root, lang)
    # Prefer the repo-local ml_workspace if present (newer training output),
    # otherwise fall back to logs/ml_workspace for legacy runs.
    local_root = os.path.join(base_dir, "ml_workspace", "models")
    if os.path.isdir(local_root):
        return os.path.join(local_root, lang)
    return os.path.join(base_dir, "logs", "ml_workspace", "models", lang)


def _export_model_root() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configured = os.environ.get("UTOA_OTO_ML_EXPORT_ROOT", "").strip()
    if configured:
        return configured
    return os.path.join(base_dir, "ML_models")


def _structured_export_model_root_for_language(language: str) -> str:
    return os.path.join(_export_model_root(), str(language).strip().lower())


def _ml_same_language_borrow_only() -> bool:
    raw = str(os.environ.get("UTOA_ML_SAME_LANGUAGE_BORROW_ONLY", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _installed_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models_installed", "oto_ml", lang)


def _collect_model_dir_candidates(language: str, format_type: str, alias_family: str = "") -> List[str]:
    fmt = normalize_format_type(language, format_type) or "general"
    family = normalize_alias_family(alias_family)
    lang = str(language or "").strip().lower()
    candidates: List[str] = []

    def _append_version_dirs(base_path: str) -> None:
        if not base_path:
            return
        candidates.extend(
            [
                os.path.join(base_path, "v1"),
                os.path.join(base_path, "v1_ensemble"),
                os.path.join(base_path, "v1_coupled"),
            ]
        )
        if not os.path.isdir(base_path):
            return
        try:
            for name in os.listdir(base_path):
                if not str(name).lower().startswith("v"):
                    continue
                path = os.path.join(base_path, name)
                if os.path.isfile(os.path.join(path, "model_meta.json")):
                    candidates.append(path)
        except Exception:
            return

    for root in (
        _workspace_model_root_for_language(language),
        _installed_model_root_for_language(language),
        _structured_export_model_root_for_language(language),
        _model_root_for_language(language),
    ):
        if os.path.isfile(os.path.join(root, "model_meta.json")):
            candidates.append(root)
        else:
            if family:
                _append_version_dirs(os.path.join(root, fmt, "families", family))
                _append_version_dirs(os.path.join(root, f"{fmt}_{family}"))
            _append_version_dirs(os.path.join(root, fmt))
            if fmt == "cvc":
                if family:
                    _append_version_dirs(os.path.join(root, "cv", "families", family))
                    _append_version_dirs(os.path.join(root, f"cv_{family}"))
                _append_version_dirs(os.path.join(root, "cv"))
            if fmt != "general":
                if family:
                    _append_version_dirs(os.path.join(root, "general", "families", family))
                    _append_version_dirs(os.path.join(root, f"general_{family}"))
                _append_version_dirs(os.path.join(root, "general"))
    export_root = _export_model_root()
    legacy_export_candidates = []
    if family:
        legacy_export_candidates.extend(
            [
                os.path.join(export_root, f"{lang}_{fmt}_{family}_v1"),
                os.path.join(export_root, f"{lang}_{fmt}_{family}"),
            ]
        )
    legacy_export_candidates.extend(
        [
            os.path.join(export_root, f"{lang}_{fmt}_v1"),
            os.path.join(export_root, f"{lang}_{fmt}"),
            os.path.join(export_root, f"{lang}_{fmt}_profile_run_new"),
            os.path.join(export_root, f"{lang}_{fmt}_profile_run_v4"),
            os.path.join(export_root, f"{lang}_{fmt}_profile_run_v3"),
            os.path.join(export_root, f"{lang}_{fmt}_profile_run"),
        ]
    )
    if fmt == "cvc":
        if family:
            legacy_export_candidates.extend(
                [
                    os.path.join(export_root, f"{lang}_cv_{family}_v1"),
                    os.path.join(export_root, f"{lang}_cv_{family}"),
                ]
            )
        legacy_export_candidates.extend(
            [
                os.path.join(export_root, f"{lang}_cv_v1"),
                os.path.join(export_root, f"{lang}_cv"),
            ]
        )
    candidates.extend(legacy_export_candidates)
    if not _ml_same_language_borrow_only():
        # Optional cross-language fallback for local experiments only.
        for alt_lang in ("korean", "japanese"):
            if alt_lang == lang:
                continue
            alt_root = _model_root_for_language(alt_lang)
            for candidate in (
                os.path.join(alt_root, fmt, "families", family, "v1") if family else "",
                os.path.join(alt_root, f"{fmt}_{family}", "v1") if family else "",
                os.path.join(alt_root, fmt, "v1"),
                os.path.join(alt_root, "general", "v1"),
            ):
                if candidate and os.path.isfile(os.path.join(candidate, "model_meta.json")):
                    candidates.append(candidate)
    # Preserve order while removing duplicates.
    seen = set()
    ordered = []
    for path in candidates:
        if not path:
            continue
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _resolve_backend_model_dir(
    language: str,
    format_type: str,
    alias_family: str = "",
    backend: str = "",
) -> Optional[str]:
    backend_norm = str(backend or "").strip().lower()
    for candidate in _collect_model_dir_candidates(language, format_type, alias_family=alias_family):
        meta_path = os.path.join(candidate, "model_meta.json")
        if not os.path.isfile(meta_path):
            continue
        if not backend_norm:
            return candidate
        detected_backend = _read_bundle_backend(candidate)
        if detected_backend == backend_norm:
            return candidate
        if backend_norm == "lightgbm" and not detected_backend:
            return candidate
    return None


def _resolve_lightgbm_model_dir(language: str, format_type: str, alias_family: str = "") -> Optional[str]:
    return _resolve_backend_model_dir(language, format_type, alias_family=alias_family, backend="lightgbm")


def _resolve_ensemble_model_dir(language: str, format_type: str, alias_family: str = "") -> Optional[str]:
    return _resolve_backend_model_dir(language, format_type, alias_family=alias_family, backend="ensemble_v1")


def _ensure_rawmel_patches(
    feat: Dict[str, object],
    *,
    wav_dir: str,
    wav_index: Dict[str, str],
    mel_cache: Dict[str, object],
    patch_spec: Dict[str, object],
) -> None:
    if not patch_spec:
        raise RuntimeError("raw mel patch spec missing in bundle")
    wav_name = str(feat.get("wav", "") or "").strip()
    wav_path = find_wav_path(wav_name, wav_dir, wav_index)
    if not wav_path:
        raise RuntimeError(f"wav not found for raw mel patch: {wav_name}")
    onset_anchor = float(feat.get("mel_onset_anchor_ms", 0.0) or 0.0)
    tail_anchor = float(feat.get("mel_tail_anchor_ms", 0.0) or 0.0)
    if onset_anchor <= 0.0 or tail_anchor <= 0.0:
        onset_anchor, tail_anchor, _src = resolve_mel_patch_anchors(feat)
    onset_patch, tail_patch, quality = extract_patches_for_row_with_quality(
        wav_path=wav_path,
        onset_anchor_ms=onset_anchor,
        tail_anchor_ms=tail_anchor,
        patch_spec=patch_spec,
        mel_cache=mel_cache,
    )
    feat["mel_onset_patch"] = onset_patch
    feat["mel_tail_patch"] = tail_patch
    feat["mel_patch_onset_mean"] = float(quality.get("onset_mean", 0.0))
    feat["mel_patch_onset_std"] = float(quality.get("onset_std", 0.0))
    feat["mel_patch_onset_low_ratio"] = float(quality.get("onset_low_ratio", 1.0))
    feat["mel_patch_tail_mean"] = float(quality.get("tail_mean", 0.0))
    feat["mel_patch_tail_std"] = float(quality.get("tail_std", 0.0))
    feat["mel_patch_tail_low_ratio"] = float(quality.get("tail_low_ratio", 1.0))


def _resolve_model_dir(language: str, format_type: str, alias_family: str = "") -> Optional[str]:
    ensemble_dir = _resolve_ensemble_model_dir(language, format_type, alias_family=alias_family) if _ensemble_enabled() else None
    lightgbm_dir = _resolve_lightgbm_model_dir(language, format_type, alias_family=alias_family)
    if ensemble_dir:
        return ensemble_dir
    return lightgbm_dir


def _bundle_meta_exists(model_dir: str) -> bool:
    return bool(model_dir and os.path.isfile(os.path.join(model_dir, "model_meta.json")))


def _ensure_report(report: Optional[Dict[str, object]], **defaults) -> Dict[str, object]:
    if report is None:
        report = {}
    for key, value in defaults.items():
        report.setdefault(key, value)
    return report


def _selector_runtime_guard_enabled() -> bool:
    raw = str(os.environ.get("UTOA_DISABLE_SELECTOR_RUNTIME_GUARD", "")).strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def _selector_top1_min_gain() -> float:
    try:
        return float(os.environ.get("UTOA_SELECTOR_TOP1_MIN_GAIN", "0.0"))
    except Exception:
        return 0.0


def _selector_passes_runtime_guard(selector_payload: Optional[Dict[str, object]]) -> Tuple[bool, str]:
    if selector_payload is None:
        return False, "selector_missing"
    if not _selector_runtime_guard_enabled():
        return True, "selector_guard_disabled_env"

    meta = selector_payload.get("meta") or {}
    metrics = meta.get("metrics") if isinstance(meta, dict) else None
    if not isinstance(metrics, dict):
        return True, "selector_metrics_missing"

    if "top1_baseline" not in metrics or "top1_model" not in metrics:
        return True, "selector_top1_missing"
    try:
        top1_baseline = float(metrics.get("top1_baseline", 0.0))
        top1_model = float(metrics.get("top1_model", 0.0))
    except Exception:
        return True, "selector_top1_parse_failed"

    min_gain = _selector_top1_min_gain()
    if top1_model <= (top1_baseline + min_gain):
        return False, (
            f"selector_top1_guard(model={top1_model:.4f}, "
            f"baseline={top1_baseline:.4f}, min_gain={min_gain:.4f})"
        )
    return True, "selector_top1_guard_passed"


def _selector_score_margin(selected: Optional[Dict[str, object]]) -> Optional[float]:
    if not selected:
        return None
    scores = selected.get("scores")
    if not isinstance(scores, list) or len(scores) < 2:
        return None
    try:
        return float(scores[0].get("score", 0.0)) - float(scores[1].get("score", 0.0))
    except Exception:
        return None


def _selector_dynamic_min_margin(feature_row: Dict[str, object], base_margin: float) -> float:
    margin = float(base_margin)
    try:
        mapping_conf = float(feature_row.get("mapping_confidence", 0.0) or 0.0)
        if mapping_conf < 0.55:
            margin += 0.04
    except Exception:
        pass
    try:
        if int(float(feature_row.get("jump_blocked_flag", 0) or 0)) > 0:
            margin += 0.05
    except Exception:
        pass
    try:
        if int(float(feature_row.get("used_nuclei_fallback", 0) or 0)) > 0:
            margin += 0.05
    except Exception:
        pass
    try:
        silence_ratio = max(
            float(feature_row.get("db_silence_ratio", 0.0) or 0.0),
            float(feature_row.get("mel_window_silence_ratio", 0.0) or 0.0),
        )
        if silence_ratio >= 0.65:
            margin += 0.06
    except Exception:
        pass
    return max(0.0, min(margin, 0.25))


def check_oto_ml_ready(language: str, format_type: str, alias_family: str = "") -> Dict[str, object]:
    routed_format = normalize_format_type(language, format_type) or "general"
    family = normalize_alias_family(alias_family)
    model_dir = _resolve_model_dir(language, routed_format, alias_family=family)
    if not model_dir:
        return make_runtime_report(
            "ml",
            ML_MODEL_MISSING,
            "OTO ML 모델을 찾을 수 없습니다.",
            language=str(language or "").strip().lower(),
            format_type=routed_format,
            model_dir="",
            ready=False,
        )
    if not _bundle_meta_exists(model_dir):
        return make_runtime_report(
            "ml",
            ML_MODEL_MISSING,
            f"OTO ML model_meta.json이 없습니다: {model_dir}",
            language=str(language or "").strip().lower(),
            format_type=routed_format,
            model_dir=str(model_dir),
            ready=False,
        )
    bundle = load_oto_model_bundle(model_dir)
    if not bundle:
        primary_backend = _read_bundle_backend(model_dir)
        if primary_backend == "ensemble_v1":
            fallback_dir = _resolve_lightgbm_model_dir(language, routed_format, alias_family=family)
            if fallback_dir and _bundle_meta_exists(fallback_dir):
                fallback_bundle = load_oto_model_bundle(fallback_dir)
                if fallback_bundle:
                    return make_runtime_report(
                        "ml",
                        OK,
                        "OTO ML 준비 완료 (ensemble fallback available)",
                        language=str(language or "").strip().lower(),
                        format_type=routed_format,
                        alias_family=family,
                        model_dir=str(fallback_dir),
                        ready=True,
                    )
        return make_runtime_report(
            "ml",
            ML_BUNDLE_INVALID,
            f"OTO ML 번들을 읽지 못했습니다: {model_dir}",
            language=str(language or "").strip().lower(),
            format_type=routed_format,
            model_dir=str(model_dir),
            ready=False,
        )
    return make_runtime_report(
        "ml",
        OK,
            "OTO ML 준비 완료",
            language=str(language or "").strip().lower(),
            format_type=routed_format,
            alias_family=family,
            model_dir=str(model_dir),
            ready=True,
        )


def _route_format_for_feature(language: str, feature_row: Dict[str, object], format_override: Optional[str] = None) -> Optional[str]:
    lang = str(language or "").strip().lower()
    base_format = normalize_format_type(lang, (format_override or feature_row.get("format_type", "general") or "general")) or "general"
    alias_type = str(feature_row.get("alias_type", "") or "").strip().lower()
    coda_type = str(feature_row.get("coda_type", "") or "").strip().lower()
    mapping_conf = _to_float(feature_row.get("mapping_confidence"), 1.0)
    pre_expected_gap = abs(_to_float(feature_row.get("base_pre_to_expected_ms"), 0.0))

    if lang == "japanese" and base_format == "cv":
        if alias_type in {"cv", "cv_head", "mono"}:
            if mapping_conf < 0.35 or pre_expected_gap > 140.0:
                return None
            return "cv"
        return None

    if lang == "japanese" and base_format == "vcv" and not _has_explicit_model_override(lang):
        # 일본어 VCV는 현재 VV/VC에서 ML 이득이 크고, -CV도 보조 보정 가치가 있다.
        # 다만 일반 CV까지 전부 열면 과보정 위험이 있으므로 cv_head만 재개방한다.
        # 1차 매핑이 흔들리는 행은 ML을 태우지 않는다(오류 증폭 방지).
        if alias_type in {"vc", "vv"}:
            if mapping_conf < 0.62 or pre_expected_gap > 160.0:
                return None
            return "vcv"
        if alias_type in {"vcv", "cv_head"}:
            if mapping_conf < 0.78 or pre_expected_gap > 90.0:
                return None
            return "vcv"
        return None

    if lang != "korean" or base_format != "vcv":
        if lang == "korean" and base_format == "cvvc":
            if alias_type == "vv":
                return "cvvc"
            if alias_type == "vc" and coda_type in {"stop", "nasal", "liquid"}:
                return "cvvc"
            if alias_type in {"cv", "cv_head"}:
                if mapping_conf < 0.60 or pre_expected_gap > 140.0:
                    return None
                return "cvvc"
            return None
        if lang == "korean" and base_format == "cv":
            if alias_type in {"cv", "cv_head", "mono"}:
                return "cv"
            if alias_type == "vc" and coda_type in {"stop", "nasal", "liquid"}:
                return "cv"
            if alias_type == "vv":
                return "cv"
            return None
        if lang == "korean" and base_format == "cvc":
            if alias_type in {"cv", "cv_head", "mono", "vv"}:
                return "cvc"
            if alias_type == "vc" and coda_type in {"stop", "nasal", "liquid"}:
                return "cvc"
            return None
        return base_format or "general"
    if _has_explicit_model_override(lang):
        return base_format or "general"

    if alias_type == "vv":
        return "cvvc"

    # VCV 내에서도 실제 받침 성격의 VC만 CVC 편향을 재사용한다.
    if alias_type == "vc" and coda_type in {"stop", "nasal", "liquid"}:
        return "cvc"
    # KR VCV의 비브리지 alias는 vcv 전용 모델을 우선 시도하고, 없으면 general로 fallback한다.
    return "vcv"


def _get_validate_func(language: str):
    lang = str(language).strip().lower()
    if lang == "japanese":
        from core.ja_oto_generator import validate_oto_params
    else:
        from core.oto_generator import validate_oto_params
    return validate_oto_params


def _clip_delta(language: str, target: str, value: float, alias_type: str = "") -> float:
    clip = get_delta_clip_limits(language, alias_type=alias_type).get(target)
    if not clip:
        return float(value)
    lo, hi = clip
    return max(float(lo), min(float(hi), float(value)))


def _emit(callback, message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except UnicodeEncodeError:
        callback(str(message).encode("cp932", errors="replace").decode("cp932", errors="replace"))
    except Exception:
        logger.debug("OTO ML callback failed", exc_info=True)


def _scale_signed(value: float, neg_scale: float = 1.0, pos_scale: float = 1.0) -> float:
    value = float(value)
    if value < 0:
        return value * float(neg_scale)
    if value > 0:
        return value * float(pos_scale)
    return value


def _apply_japanese_delta_policy(row_context: Dict[str, object], deltas: Dict[str, float]) -> Dict[str, float]:
    format_type = normalize_format_type("japanese", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    alias_text = str(row_context.get("alias_text", "") or row_context.get("alias", "") or "").strip()

    try:
        from core.ja_oto_generator import (
            JA_FRICATIVE_ONSETS,
            JA_PLOSIVE_ONSETS,
            JA_SIBILANT_ONSETS,
            JA_VOICED_ONSETS,
            _ja_is_n_bridge_alias,
            _token_to_romaji,
        )
        from core.ja_lab_generator import split_ja_romaji_syllable
    except Exception:
        return deltas

    parts = alias_text.split()
    token = alias_text
    if len(parts) >= 2:
        if alias_type in {"vc", "vv", "vcv", "cv_head"}:
            token = parts[1]
        else:
            token = parts[0] if parts[0] != "-" else parts[1]
    elif alias_text.startswith("-"):
        token = alias_text[1:]
    token = token.strip("-_ ")
    token_romaji = _token_to_romaji(token)
    onset, _vowel = split_ja_romaji_syllable(token_romaji)
    onset = onset or token_romaji

    # n + 유성음/비음 브리지는 기본 계산이 더 안정적이다.
    # ML은 세부 정리만 맡기고 위치 축(offset/pre)은 거의 못 움직이게 제한한다.
    if format_type == "vcv" and alias_type in {"vc", "vcv"}:
        if _ja_is_n_bridge_alias(alias_text, alias_type) and onset in JA_VOICED_ONSETS:
            deltas["delta_offset"] *= 0.10
            deltas["delta_pre"] *= 0.18
            deltas["delta_cons"] *= 0.35
            deltas["delta_cutoff"] *= 0.45
            deltas["delta_ovl"] *= 0.50
            return deltas

    # 일본어 CVVC의 유성음/치찰음은 ML이 pre/cutoff를 과하게 앞으로 당기는 경향이 있다.
    # "앞당기는" 방향 delta만 더 강하게 줄인다.
    if format_type in {"cvvc", "cv"} and alias_type in {"cv", "cv_head", "vc"}:
        is_sensitive = onset in JA_VOICED_ONSETS or onset in JA_SIBILANT_ONSETS or onset in JA_FRICATIVE_ONSETS or onset in JA_PLOSIVE_ONSETS
        if is_sensitive:
            if alias_type in {"cv", "cv_head"} and (
                onset in JA_SIBILANT_ONSETS or onset in JA_FRICATIVE_ONSETS or onset in JA_PLOSIVE_ONSETS
            ):
                deltas["delta_offset"] = 0.0
            else:
                deltas["delta_offset"] = _scale_signed(deltas.get("delta_offset", 0.0), neg_scale=0.25, pos_scale=0.65)
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.18, pos_scale=0.80)
            deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.85, pos_scale=0.18)
            deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.60, pos_scale=0.85)
            deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.60, pos_scale=0.85)
            if alias_type in {"cv", "cv_head"}:
                deltas["delta_cutoff"] = min(deltas.get("delta_cutoff", 0.0), 0.0)
            return deltas
        if alias_type in {"cv", "cv_head"}:
            # CV 계열은 컷오프 연장이 다음 음절 초입 누수를 만들기 쉬우므로
            # ML이 컷오프를 늘리는 방향(+)은 사실상 차단한다.
            deltas["delta_cutoff"] = min(
                _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.90, pos_scale=0.0),
                0.0,
            )

    return deltas


def _apply_korean_delta_policy(row_context: Dict[str, object], deltas: Dict[str, float]) -> Dict[str, float]:
    format_type = normalize_format_type("korean", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    coda_type = str(row_context.get("coda_type", "") or "").strip().lower()
    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)

    if alias_type in {"cv", "cv_head"}:
        mel_unreliable = is_mel_unreliable(row_context)
        if format_type in {"cvvc", "cvc"} and _kr_cv_keep_base_location_enabled():
            deltas["delta_offset"] = 0.0
            deltas["delta_pre"] = 0.0

        blank_conf = _to_float(row_context.get("blank_span_confidence"), 0.0)
        syll_blank_conf = _to_float(row_context.get("syllable_blank_confidence"), 0.0)
        syll_sil_conf = _to_float(row_context.get("syllable_mel_silence_conf"), 0.0)
        syll_voiced_conf = _to_float(row_context.get("syllable_mel_voiced_conf"), 0.0)
        silence_ratio = max(
            _to_float(row_context.get("db_silence_ratio"), 0.0),
            _to_float(row_context.get("mel_window_silence_ratio"), 0.0),
        )
        blank_risk = max(blank_conf, syll_blank_conf, syll_sil_conf, compute_blank_risk_score(row_context))
        risky = (mapping_conf < 0.74) or (blank_risk >= 0.44) or (silence_ratio >= 0.52) or mel_unreliable
        severe = (
            (blank_risk >= 0.58)
            or (silence_ratio >= 0.62)
            or ((blank_risk - syll_voiced_conf) >= 0.18)
            or mel_unreliable
        )

        if severe:
            # 공백 가능성이 높은 CV는 offset 전진을 금지하고 pre 이동도 크게 제한한다.
            deltas["delta_offset"] = 0.0
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.06, pos_scale=0.42)
        else:
            deltas["delta_offset"] = _scale_signed(
                deltas.get("delta_offset", 0.0),
                neg_scale=0.12 if risky else 0.24,
                pos_scale=0.72 if risky else 0.84,
            )
            deltas["delta_pre"] = _scale_signed(
                deltas.get("delta_pre", 0.0),
                neg_scale=0.14 if risky else 0.26,
                pos_scale=0.62 if risky else 0.82,
            )

        deltas["delta_cons"] = _scale_signed(
            deltas.get("delta_cons", 0.0),
            neg_scale=0.58 if risky else 0.72,
            pos_scale=0.82 if risky else 0.92,
        )
        deltas["delta_cutoff"] = min(
            _scale_signed(
                deltas.get("delta_cutoff", 0.0),
                neg_scale=0.80 if risky else 0.90,
                pos_scale=0.18 if risky else 0.28,
            ),
            12.0 if risky else 20.0,
        )
        deltas["delta_ovl"] = _scale_signed(
            deltas.get("delta_ovl", 0.0),
            neg_scale=0.68 if risky else 0.82,
            pos_scale=0.78 if risky else 0.90,
        )

    elif alias_type == "vc":
        # VC는 연결 안정성을 위해 offset/pre 이동을 기본적으로 억제한다.
        deltas["delta_offset"] = _scale_signed(deltas.get("delta_offset", 0.0), neg_scale=0.35, pos_scale=0.46)
        if coda_type == "stop":
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.44, pos_scale=0.62)
            deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.58, pos_scale=0.84)
            deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.78, pos_scale=0.30)
            deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.82, pos_scale=0.82)
        elif coda_type in {"nasal", "liquid"}:
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.52, pos_scale=0.74)
            deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.66, pos_scale=0.88)
            deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.82, pos_scale=0.46)
            deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.90, pos_scale=0.90)
        else:
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.50, pos_scale=0.70)
            deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.62, pos_scale=0.86)
            deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.80, pos_scale=0.42)
            deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.86, pos_scale=0.86)
    elif alias_type == "vv":
        # VV는 pre-ovl 간격(리듬)을 보존하도록 ovl/offset을 강하게 제한한다.
        deltas["delta_offset"] = _scale_signed(deltas.get("delta_offset", 0.0), neg_scale=0.44, pos_scale=0.54)
        deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.56, pos_scale=0.78)
        deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.72, pos_scale=0.92)
        deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.80, pos_scale=0.58)
        deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.62, pos_scale=0.62)
    else:
        return deltas

    if mapping_conf < 0.72:
        for key in list(deltas.keys()):
            deltas[key] = float(deltas[key]) * 0.72
    elif mapping_conf < 0.80:
        for key in list(deltas.keys()):
            deltas[key] = float(deltas[key]) * 0.86

    if format_type in {"cvc", "cv"}:
        for key in list(deltas.keys()):
            deltas[key] = float(deltas[key]) * 0.90

    return deltas


def _apply_korean_cv_no_regression_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    format_type = normalize_format_type("korean", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if alias_type not in {"cv", "cv_head"}:
        return params
    if format_type not in {"cv", "cvvc", "cvc", "vcv"}:
        return params
    tight_cv = format_type in {"cvvc", "cvc"}

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    offset = float(offset)
    cons = float(cons)
    pre = float(pre)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))

    curr_phone_start = _to_float(row_context.get("curr_phone_start_ms"), 0.0)
    curr_phone_len = max(_to_float(row_context.get("curr_phone_len_ms"), 0.0), 0.0)
    curr_vowel_start = _to_float(row_context.get("curr_vowel_start_ms"), 0.0)
    expected_anchor = _to_float(row_context.get("expected_anchor_ms"), 0.0)
    base_offset = _to_float(row_context.get("base_offset"), offset)
    base_pre = _to_float(row_context.get("base_pre"), pre)
    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
        _to_float(row_context.get("syllable_mel_silence_conf"), 0.0),
    )
    silence_ratio = max(
        _to_float(row_context.get("db_silence_ratio"), 0.0),
        _to_float(row_context.get("mel_window_silence_ratio"), 0.0),
    )
    onset_candidate = _to_float(row_context.get("mel_offset_candidate_ms"), 0.0)
    base_pre_abs = float(base_offset) + float(base_pre)

    min_conf = _ml_anchor_lock_min_conf()
    blank_floor = _ml_anchor_blank_floor("korean", format_type)
    if mapping_conf < min_conf:
        return validate_fn(offset, cons, cutoff, pre, ovl)
    if blank_floor is not None and blank_conf >= blank_floor:
        return validate_fn(offset, cons, cutoff, pre, ovl)

    if curr_vowel_start > 0.0:
        expected_anchor = curr_vowel_start

    pre_abs = offset + pre
    pre_lead_cap = 18.0 if alias_type == "cv_head" else 24.0
    if tight_cv:
        pre_lead_cap -= 2.0
    if mapping_conf < 0.74:
        pre_lead_cap -= 4.0
    if blank_conf >= 0.58 or silence_ratio >= 0.62:
        pre_lead_cap -= 6.0
    pre_lead_cap = max(pre_lead_cap, 8.0)

    min_pre_abs = pre_abs
    if expected_anchor > 0.0:
        min_pre_abs = expected_anchor - pre_lead_cap
    if curr_phone_start > 0.0:
        phone_guard = curr_phone_start + min(14.0, max(curr_phone_len * 0.12, 7.0))
        min_pre_abs = max(min_pre_abs, phone_guard)
    if onset_candidate > 0.0:
        min_pre_abs = max(min_pre_abs, onset_candidate + (6.0 if alias_type == "cv_head" else 8.0))
    early_pre_backoff = 4.0 if (blank_conf >= 0.62 or silence_ratio >= 0.66) else 10.0
    min_pre_abs = max(min_pre_abs, base_pre_abs - early_pre_backoff)

    early_offset_guard = 6.0 if not tight_cv else 4.0
    min_offset = base_offset - early_offset_guard
    if curr_phone_start > 0.0:
        min_offset = max(min_offset, curr_phone_start - (1.5 if alias_type == "cv_head" and tight_cv else 3.0))
    if onset_candidate > 0.0:
        onset_guard = 4.0 if alias_type == "cv_head" else 3.0
        if tight_cv:
            onset_guard = max(2.0, onset_guard - 1.0)
        min_offset = max(min_offset, onset_candidate - onset_guard)
    if blank_conf >= 0.62 or silence_ratio >= 0.66:
        min_offset = max(min_offset, base_offset - 1.5)

    if offset < min_offset:
        offset = min_offset
    if pre_abs < min_pre_abs:
        pre_abs = min_pre_abs

    pre = max(pre_abs - offset, 0.0)
    cons = max(cons, pre + 8.0)
    cutoff_abs = max(cutoff_abs, cons + 10.0)
    return validate_fn(offset, cons, -cutoff_abs, pre, ovl)


def _apply_korean_cv_destination_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    format_type = normalize_format_type("korean", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if alias_type not in {"cv", "cv_head"}:
        return params
    if format_type not in {"cv", "cvvc", "cvc", "vcv"}:
        return params
    tight_cv = format_type in {"cvvc", "cvc"}

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    offset = float(offset)
    cons = float(cons)
    pre = float(pre)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))
    pre_abs = offset + pre

    base_offset = _to_float(row_context.get("base_offset"), offset)
    base_cons = _to_float(row_context.get("base_cons"), cons)
    base_cutoff_abs = _to_float(row_context.get("base_cutoff_abs"), cutoff_abs)
    base_pre = _to_float(row_context.get("base_pre"), pre)
    base_ovl = _to_float(row_context.get("base_ovl"), ovl)
    base_params = validate_fn(
        float(base_offset),
        float(base_cons),
        -(float(base_cutoff_abs) - float(base_offset)),
        float(base_pre),
        float(base_ovl),
    )
    base_pre_abs = float(base_offset) + float(base_pre)

    if format_type in {"cvvc", "cvc"} and _kr_cv_keep_base_location_enabled():
        locked_offset = float(base_offset)
        locked_pre = float(base_pre)
        locked_ovl = min(float(ovl), max(locked_pre - 1.0, 0.0))
        locked_cons = max(float(cons), locked_pre + 8.0)
        locked_cutoff_rel = max(abs(float(cutoff)), locked_cons + 10.0)
        return validate_fn(locked_offset, locked_cons, -locked_cutoff_rel, locked_pre, locked_ovl)

    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
        _to_float(row_context.get("syllable_mel_silence_conf"), 0.0),
    )
    voiced_conf = _to_float(row_context.get("syllable_mel_voiced_conf"), 0.0)
    silence_ratio = max(
        _to_float(row_context.get("db_silence_ratio"), 0.0),
        _to_float(row_context.get("mel_window_silence_ratio"), 0.0),
    )
    onset_candidate = _to_float(row_context.get("mel_offset_candidate_ms"), 0.0)
    curr_vowel_start = _to_float(row_context.get("curr_vowel_start_ms"), 0.0)
    expected_anchor = _to_float(row_context.get("expected_anchor_ms"), 0.0)
    if curr_vowel_start > 0.0:
        expected_anchor = curr_vowel_start

    severe_blank = (
        (blank_conf >= 0.58)
        or (silence_ratio >= 0.62)
        or ((blank_conf - voiced_conf) >= 0.18)
    )
    risky_blank = severe_blank or (blank_conf >= 0.48) or (silence_ratio >= 0.56) or (mapping_conf < 0.74)

    early_offset_allow = 1.0 if severe_blank else (3.0 if risky_blank else 6.0)
    early_pre_allow = 3.0 if severe_blank else (7.0 if risky_blank else 12.0)
    onset_allow = 2.0 if severe_blank else (4.0 if risky_blank else 7.0)
    anchor_lead_allow = 8.0 if severe_blank else (12.0 if risky_blank else 18.0)
    if tight_cv:
        early_offset_allow = max(0.5, early_offset_allow - 2.0)
        early_pre_allow = max(2.0, early_pre_allow - 2.0)
        onset_allow = max(1.0, onset_allow - 1.0)
        anchor_lead_allow = max(6.0, anchor_lead_allow - 3.0)

    violations = 0
    if offset < (base_offset - early_offset_allow):
        violations += 1
    if pre_abs < (base_pre_abs - early_pre_allow):
        violations += 1
    if onset_candidate > 0.0 and offset < (onset_candidate - onset_allow):
        violations += 1
    if expected_anchor > 0.0 and pre_abs < (expected_anchor - anchor_lead_allow):
        violations += 1

    head_zero_recoverable = bool(
        base_offset <= 0.5
        and onset_candidate >= 8.0
        and mapping_conf >= 0.70
        and alias_type in {"cv", "cv_head"}
        and format_type in {"cvc", "cvvc", "cv", "vcv"}
    )

    if severe_blank and violations >= 1 and not head_zero_recoverable:
        return base_params
    if risky_blank and violations >= 2 and not head_zero_recoverable:
        return base_params

    min_offset = base_offset - early_offset_allow
    if onset_candidate > 0.0:
        min_offset = max(min_offset, onset_candidate - onset_allow)
    if head_zero_recoverable:
        # Recover from template/head collapse where base offset is pinned at 0.
        recover_guard = 2.5 if tight_cv else 4.0
        min_offset = max(min_offset, onset_candidate - recover_guard)
        if curr_vowel_start > 0.0:
            min_offset = max(min_offset, curr_vowel_start - (12.0 if tight_cv else 16.0))
    if offset < min_offset:
        offset = min_offset

    min_pre_abs = base_pre_abs - early_pre_allow
    if expected_anchor > 0.0:
        min_pre_abs = max(min_pre_abs, expected_anchor - anchor_lead_allow)
    if pre_abs < min_pre_abs:
        pre_abs = min_pre_abs

    pre = max(pre_abs - offset, 0.0)
    cons = max(cons, pre + 8.0)
    cutoff_abs = max(cutoff_abs, cons + 10.0)
    return validate_fn(offset, cons, -cutoff_abs, pre, ovl)


def _apply_korean_bridge_post_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    format_type = normalize_format_type("korean", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    coda_type = str(row_context.get("coda_type", "") or "").strip().lower()
    if alias_type not in {"vc", "vv"}:
        return params
    tight_bridge = format_type in {"cvvc", "cvc"}

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    pre = float(pre)
    cons = float(cons)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))

    if alias_type == "vc":
        if coda_type == "stop":
            gap_lo, gap_hi, gap_t = 10.0, 24.0, 16.0
            cons_lo, cons_hi, cons_t = 10.0, 44.0, 24.0
            cut_lo, cut_hi, cut_t = 6.0, 18.0, 10.0
            next_allow = -1.0
        elif coda_type in {"nasal", "liquid"}:
            gap_lo, gap_hi, gap_t = 7.0, 20.0, 13.0
            cons_lo, cons_hi, cons_t = 26.0, 86.0, 48.0
            cut_lo, cut_hi, cut_t = 12.0, 36.0, 22.0
            next_allow = 22.0
        else:
            gap_lo, gap_hi, gap_t = 8.0, 22.0, 14.0
            cons_lo, cons_hi, cons_t = 20.0, 72.0, 40.0
            cut_lo, cut_hi, cut_t = 10.0, 30.0, 18.0
            next_allow = 16.0
    else:
        gap_lo, gap_hi, gap_t = 4.0, 12.0, 8.0
        cons_lo, cons_hi, cons_t = 58.0, 156.0, 98.0
        cut_lo, cut_hi, cut_t = 18.0, 98.0, 52.0
        next_allow = 34.0

    if tight_bridge:
        gap_lo = max(4.0, gap_lo - 2.0)
        gap_hi = max(gap_lo + 2.0, gap_hi - 4.0)
        gap_t = max(gap_lo, gap_t - 3.0)
        if next_allow > 0.0:
            next_allow = max(6.0, next_allow - 6.0)

    gap_now = max(pre - ovl, 0.0)
    cons_gap_now = max(cons - pre, 8.0)
    cut_gap_now = max(cutoff_abs - cons, 10.0)

    gap_new = max(gap_lo, min(gap_hi, gap_now))
    gap_new = ((gap_new * 0.72) + (gap_t * 0.28))
    cons_gap_new = max(cons_lo, min(cons_hi, cons_gap_now))
    cons_gap_new = ((cons_gap_new * 0.70) + (cons_t * 0.30))
    cut_gap_new = max(cut_lo, min(cut_hi, cut_gap_now))
    cut_gap_new = ((cut_gap_new * 0.70) + (cut_t * 0.30))

    ovl = max(0.0, pre - gap_new)
    cons = pre + cons_gap_new
    cutoff_abs = cons + cut_gap_new

    cutoff_abs = _clamp_bridge_cutoff_to_next_onset(
        row_context,
        float(offset),
        float(cons),
        float(pre),
        float(cutoff_abs),
        min_cut_gap=float(cut_lo),
        next_allow=float(next_allow),
        pre_floor=10.0,
    )

    return validate_fn(float(offset), float(cons), -float(cutoff_abs), float(pre), float(ovl))


def _clamp_bridge_cutoff_to_next_onset(
    row_context: Dict[str, object],
    offset: float,
    cons: float,
    pre: float,
    cutoff_abs: float,
    *,
    min_cut_gap: float,
    next_allow: float,
    pre_floor: float,
) -> float:
    curr_end = _to_float(row_context.get("curr_phone_end_ms"), 0.0)
    next_gap = _to_float(row_context.get("next_phone_gap_ms"), 0.0)
    if curr_end <= 0.0 or next_gap <= 0.0:
        return float(cutoff_abs)
    next_onset_rel = max((curr_end + next_gap) - float(offset), float(pre) + float(pre_floor))
    return min(float(cutoff_abs), max(float(cons) + float(min_cut_gap), next_onset_rel + float(next_allow)))


def _apply_japanese_bridge_post_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    format_type = normalize_format_type("japanese", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if format_type not in {"cvvc", "vcv"} or alias_type not in {"vc", "vv"}:
        return params

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    offset = float(offset)
    cons = float(cons)
    pre = float(pre)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))
    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)
    coda_type = str(row_context.get("coda_type", "") or "").strip().lower()

    if alias_type == "vc":
        if coda_type == "stop":
            gap_lo, gap_hi, gap_t = 8.0, 20.0, 12.0
            cons_lo, cons_hi, cons_t = 10.0, 42.0, 24.0
            cut_lo, cut_hi, cut_t = 4.0, 16.0, 8.0
            next_allow = -1.0
        elif coda_type in {"nasal", "liquid"}:
            gap_lo, gap_hi, gap_t = 8.0, 22.0, 13.0
            cons_lo, cons_hi, cons_t = 20.0, 74.0, 40.0
            cut_lo, cut_hi, cut_t = 10.0, 32.0, 18.0
            next_allow = 10.0
        else:
            gap_lo, gap_hi, gap_t = 8.0, 22.0, 13.0
            cons_lo, cons_hi, cons_t = 16.0, 62.0, 34.0
            cut_lo, cut_hi, cut_t = 8.0, 26.0, 14.0
            next_allow = 4.0
    else:
        gap_lo, gap_hi, gap_t = 4.0, 12.0, 7.0
        cons_lo, cons_hi, cons_t = 52.0, 144.0, 90.0
        cut_lo, cut_hi, cut_t = 18.0, 88.0, 44.0
        next_allow = 8.0

    if mapping_conf < 0.70:
        next_allow *= 0.85
    elif mapping_conf > 0.88:
        next_allow *= 1.08

    gap_now = max(pre - ovl, 0.0)
    cons_gap_now = max(cons - pre, 6.0)
    cut_gap_now = max(cutoff_abs - cons, 8.0)

    gap_new = max(gap_lo, min(gap_hi, gap_now))
    gap_new = (gap_new * 0.72) + (gap_t * 0.28)
    cons_gap_new = max(cons_lo, min(cons_hi, cons_gap_now))
    cons_gap_new = (cons_gap_new * 0.70) + (cons_t * 0.30)
    cut_gap_new = max(cut_lo, min(cut_hi, cut_gap_now))
    cut_gap_new = (cut_gap_new * 0.68) + (cut_t * 0.32)

    ovl = max(0.0, pre - gap_new)
    cons = pre + cons_gap_new
    cutoff_abs = cons + cut_gap_new
    cutoff_abs = _clamp_bridge_cutoff_to_next_onset(
        row_context,
        offset,
        cons,
        pre,
        cutoff_abs,
        min_cut_gap=float(cut_lo),
        next_allow=float(next_allow),
        pre_floor=8.0 if alias_type == "vc" else 6.0,
    )

    return validate_fn(offset, cons, -cutoff_abs, pre, ovl)


def _apply_japanese_cvvc_cv_post_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    format_type = normalize_format_type("japanese", row_context.get("format_type", ""))
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if format_type != "cvvc" or alias_type not in {"cv", "cv_head"}:
        return params

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    offset = float(offset)
    cons = float(cons)
    pre = float(pre)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))

    curr_phone_start = _to_float(row_context.get("curr_phone_start_ms"), 0.0)
    curr_phone_len = max(_to_float(row_context.get("curr_phone_len_ms"), 0.0), 0.0)
    curr_vowel_start = _to_float(row_context.get("curr_vowel_start_ms"), 0.0)
    expected_anchor = _to_float(row_context.get("expected_anchor_ms"), 0.0)
    base_offset = _to_float(row_context.get("base_offset"), offset)

    if curr_vowel_start > 0.0:
        expected_anchor = curr_vowel_start
    if expected_anchor <= 0.0:
        return offset, cons, -cutoff_abs, pre, ovl

    pre_abs = offset + pre
    pre_lead_cap = 18.0 if alias_type == "cv_head" else 24.0
    min_pre_abs = expected_anchor - pre_lead_cap
    if curr_phone_start > 0.0:
        phone_guard = curr_phone_start + min(12.0, max(curr_phone_len * 0.10, 6.0))
        min_pre_abs = max(min_pre_abs, phone_guard)

    min_offset = base_offset - 12.0
    if curr_phone_start > 0.0:
        min_offset = max(min_offset, curr_phone_start - 8.0)

    if offset < min_offset:
        offset = min_offset
    if pre_abs < min_pre_abs:
        pre_abs = min_pre_abs

    pre = max(pre_abs - offset, 0.0)
    cons = max(cons, pre + 6.0)
    cutoff_abs = max(cutoff_abs, cons + 8.0)
    return validate_fn(offset, cons, -cutoff_abs, pre, ovl)


def _apply_language_specific_post_guard(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    lang = str(language or "").strip().lower()
    out = params
    if lang == "korean":
        out = _apply_korean_bridge_post_guard(row_context, out, validate_fn)
        out = _apply_korean_cv_no_regression_guard(row_context, out, validate_fn)
    elif lang == "japanese":
        out = _apply_japanese_bridge_post_guard(row_context, out, validate_fn)
        out = _apply_japanese_cvvc_cv_post_guard(row_context, out, validate_fn)
    return out


def _apply_head_zero_offset_recovery_guard(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    if not _head_zero_offset_guard_enabled():
        return params

    lang = str(language or "").strip().lower()
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    if alias_type not in {"cv", "cv_head"}:
        return params

    format_type = normalize_format_type(lang, row_context.get("format_type", ""))
    if lang == "korean":
        valid_formats = {"cv", "cvc", "cvvc", "vcv"}
    elif lang == "japanese":
        valid_formats = {"cv", "cvvc", "vcv"}
    else:
        valid_formats = {"cv", "cvc", "cvvc", "vcv"}
    if format_type not in valid_formats:
        return params

    offset, cons, cutoff, pre, ovl = validate_fn(*params)
    offset = float(offset)
    cons = float(cons)
    pre = float(pre)
    ovl = float(ovl)
    cutoff_abs = abs(float(cutoff))

    base_offset = _to_float(row_context.get("base_offset"), offset)
    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)
    onset_candidate = _to_float(row_context.get("mel_offset_candidate_ms"), 0.0)
    curr_phone_start = _to_float(row_context.get("curr_phone_start_ms"), 0.0)
    curr_vowel_start = _to_float(row_context.get("curr_vowel_start_ms"), 0.0)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
        _to_float(row_context.get("syllable_mel_silence_conf"), 0.0),
    )
    silence_ratio = max(
        _to_float(row_context.get("db_silence_ratio"), 0.0),
        _to_float(row_context.get("mel_window_silence_ratio"), 0.0),
    )

    if mapping_conf < 0.64:
        return params

    anchor = max(onset_candidate, curr_phone_start, curr_vowel_start)
    if anchor < 6.0:
        return params

    collapsed = bool(offset <= 0.8 or base_offset <= 0.8 or offset < (anchor - 10.0))
    if not collapsed:
        return params

    # Avoid forcing recovery on near-all-silence leading segments.
    if blank_conf >= 0.92 and anchor < 14.0 and silence_ratio >= 0.78:
        return params

    tight_cv = format_type in {"cvvc", "cvc"}
    recover_guard = 2.0 if tight_cv else 3.5
    min_offset = max(0.0, anchor - recover_guard)
    if curr_vowel_start > 0.0:
        min_offset = max(min_offset, curr_vowel_start - (10.0 if tight_cv else 14.0))
    if offset >= min_offset:
        return params

    pre_abs = offset + pre
    offset = min_offset
    pre = max(pre_abs - offset, 0.0)
    cons = max(cons, pre + (6.0 if alias_type == "cv_head" else 8.0))
    cutoff_abs = max(cutoff_abs, cons + (8.0 if alias_type == "cv_head" else 10.0))
    ovl = min(max(ovl, 0.0), pre)
    return validate_fn(offset, cons, -cutoff_abs, pre, ovl)


def _apply_ml_destination_guard(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    lang = str(language or "").strip().lower()
    out = params
    if lang == "korean":
        out = _apply_korean_cv_destination_guard(row_context, out, validate_fn)
    out = _apply_head_zero_offset_recovery_guard(lang, row_context, out, validate_fn)
    return out


def _apply_wav_duration_hard_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
) -> Tuple[float, float, float, float, float]:
    """
    Final hard guard against invalid OTO geometry on the real wav timeline.
    This prevents cases like offset > wav_duration or cutoff before offset.
    """
    offset, consonant, cutoff, pre, ovl = [float(v) for v in params]
    dur_ms = _to_float(row_context.get("wav_duration_ms"), 0.0)
    if dur_ms <= 0.0:
        return offset, consonant, cutoff, pre, ovl

    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    min_room_after_offset = 16.0 if alias_type in {"vc", "vv", "vcv"} else 22.0
    offset = max(0.0, min(offset, max(dur_ms - min_room_after_offset, 0.0)))

    cutoff_abs = abs(cutoff)
    max_cutoff_abs = max(dur_ms - offset - 6.0, 0.0)
    cutoff_abs = min(cutoff_abs, max_cutoff_abs)

    active_len = max(dur_ms - offset - cutoff_abs, 0.0)
    if active_len < 10.0:
        target_active_len = min(max(12.0, active_len), max(dur_ms - offset - 2.0, 0.0))
        cutoff_abs = max(0.0, dur_ms - offset - target_active_len)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)

    max_cons = max(active_len - 6.0, 0.0)
    consonant = min(consonant, max_cons)
    if consonant < 8.0 and active_len >= 10.0:
        consonant = min(max(active_len * 0.72, 8.0), max_cons)

    pre = min(max(pre, 0.0), max(consonant - 6.0, 0.0))
    ovl_cap = pre * 0.82 if pre > 0.0 else 0.0
    ovl = min(max(ovl, 0.0), ovl_cap)

    if consonant <= pre + 4.0:
        if max_cons >= pre + 6.0:
            consonant = pre + 6.0
        else:
            consonant = max_cons
            pre = max(0.0, consonant - 6.0)
            ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    if dur_ms - offset - cutoff_abs <= 2.0:
        cutoff_abs = max(0.0, dur_ms - offset - 4.0)
        active_len = max(dur_ms - offset - cutoff_abs, 0.0)
        max_cons = max(active_len - 4.0, 0.0)
        consonant = min(consonant, max_cons)
        pre = min(pre, max(consonant - 4.0, 0.0))
        ovl = min(ovl, pre * 0.82 if pre > 0.0 else 0.0)

    cutoff = -max(cutoff_abs, 0.0)
    return offset, consonant, cutoff, pre, ovl


def _solve_joint_constraints(
    params: Tuple[float, float, float, float, float],
    *,
    strict: bool = False,
) -> Tuple[Tuple[float, float, float, float, float], int]:
    offset, consonant, cutoff, pre, ovl = [float(v) for v in params]
    adjust = 0

    if offset < 0.0:
        offset = 0.0
        adjust += 1
    if pre < 0.0:
        pre = 0.0
        adjust += 1
    if ovl < 0.0:
        ovl = 0.0
        adjust += 1
    if ovl > pre:
        ovl = pre
        adjust += 1

    cons_margin = 16.0 if strict else 10.0
    cut_margin = 14.0 if strict else 10.0
    min_cons = pre + cons_margin
    if consonant < min_cons:
        consonant = min_cons
        adjust += 1

    cut_abs = abs(cutoff)
    min_cut_abs = consonant + cut_margin
    if cut_abs < min_cut_abs:
        cut_abs = min_cut_abs
        adjust += 1
    cutoff = -cut_abs
    return (offset, consonant, cutoff, pre, ovl), int(adjust)


def _finalize_ml_params(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
    anchor_stats: Optional[Dict[str, int]] = None,
    *,
    strict_constraint: bool = False,
    constraint_stats: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, float, float, float]:
    validate_row = _build_alias_aware_validator(validate_fn, row_context)
    out = validate_row(*params)
    out = _apply_language_specific_post_guard(language, row_context, out, validate_row)
    out, applied_rules = _apply_anchor_lock_lite_after_ml(language, row_context, out, validate_row)
    out = _apply_ml_destination_guard(language, row_context, out, validate_row)
    out, adjust_count = _solve_joint_constraints(out, strict=bool(strict_constraint))
    out = validate_row(*out)
    out = _apply_wav_duration_hard_guard(row_context, out)
    if anchor_stats is not None and applied_rules:
        anchor_stats["anchor_locked_count"] = int(anchor_stats.get("anchor_locked_count", 0)) + 1
        if "cutoff_next_onset_clamp" in applied_rules or "cutoff_next_vowel_clamp" in applied_rules:
            anchor_stats["cutoff_clamped_count"] = int(anchor_stats.get("cutoff_clamped_count", 0)) + 1
            if str(row_context.get("alias_type", "") or "").strip().lower() == "vc":
                anchor_stats["vc_cutoff_leak_guard_count"] = int(anchor_stats.get("vc_cutoff_leak_guard_count", 0)) + 1
    if constraint_stats is not None and adjust_count > 0:
        constraint_stats["constraint_adjust_count"] = int(constraint_stats.get("constraint_adjust_count", 0)) + int(adjust_count)
    return out


def apply_oto_ml_selector_candidate(
    language: str,
    row_context: Dict[str, object],
    base_params_override: Tuple[float, float, float, float, float],
    *,
    anchor_stats: Optional[Dict[str, int]] = None,
    strict_constraint: bool = False,
    constraint_stats: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, float, float, float]:
    validate_oto_params = _get_validate_func(language)
    base_offset, base_cons, base_cutoff_abs, base_pre, base_ovl = base_params_override
    params = (
        float(base_offset),
        float(base_cons),
        -(float(base_cutoff_abs) - float(base_offset)),
        float(base_pre),
        float(base_ovl),
    )
    return _finalize_ml_params(
        language,
        row_context,
        params,
        validate_oto_params,
        anchor_stats=anchor_stats,
        strict_constraint=strict_constraint,
        constraint_stats=constraint_stats,
    )


def _apply_language_specific_delta_policy(language: str, row_context: Dict[str, object], deltas: Dict[str, float]) -> Dict[str, float]:
    lang = str(language or "").strip().lower()
    adjusted = dict(deltas)
    if lang == "japanese":
        return _apply_japanese_delta_policy(row_context, adjusted)
    if lang == "korean":
        return _apply_korean_delta_policy(row_context, adjusted)
    return adjusted


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_alias_aware_validator(validate_fn, row_context: Dict[str, object]):
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()

    def _validate(offset, consonant, cutoff, pre, ovl):
        try:
            return validate_fn(offset, consonant, cutoff, pre, ovl, alias_type=alias_type)
        except TypeError:
            return validate_fn(offset, consonant, cutoff, pre, ovl)

    return _validate


def _anchor_key_for_row(language: str, row_context: Dict[str, object]) -> str:
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    lang = str(language or "").strip().lower()
    fmt = str(row_context.get("format_type", "") or "").strip().lower()
    if lang == "japanese" and fmt == "vcv" and alias_type == "vcv":
        try:
            from core.ja_oto_generator import _extract_vcv_target_syllable
            from core.ja_lab_generator import split_ja_romaji_syllable

            token = _extract_vcv_target_syllable(str(row_context.get("alias", "") or ""))
            onset, _vowel = split_ja_romaji_syllable(token)
            if not (onset or "").strip():
                return "vcv_vv_like"
        except Exception:
            pass
    return alias_type


def _apply_anchor_lock_lite_after_ml(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
):
    lang = str(language or "").strip().lower()
    fmt = str(row_context.get("format_type", "") or "").strip().lower()
    if not is_anchor_lock_enabled(lang, fmt):
        return params, ()

    mapping_conf = _to_float(row_context.get("mapping_confidence"), 1.0)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
    )
    if lang == "korean":
        if mapping_conf < _ml_anchor_lock_min_conf():
            return params, ()
        blank_floor = _ml_anchor_blank_floor(lang, fmt)
        if blank_floor is not None and blank_conf >= blank_floor:
            return params, ()

    alias_key = _anchor_key_for_row(language, row_context)
    profile = get_anchor_profile(lang, fmt, alias_key, mode="rhythm_stable")
    if profile is None:
        return params, ()

    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    expected_anchor = _to_float(row_context.get("expected_anchor_ms"), 0.0)
    curr_start = _to_float(row_context.get("curr_phone_start_ms"), 0.0)
    curr_end = _to_float(row_context.get("curr_phone_end_ms"), 0.0)
    curr_vowel_start = _to_float(row_context.get("curr_vowel_start_ms"), 0.0)
    curr_vowel_end = _to_float(row_context.get("curr_vowel_end_ms"), 0.0)
    next_gap = _to_float(row_context.get("next_phone_gap_ms"), 0.0)

    anchor_abs = expected_anchor if expected_anchor > 0.0 else (curr_end if curr_end > 0.0 else curr_start)
    if alias_type in {"vc", "vv"} and curr_end > 0.0:
        anchor_abs = curr_end
    if alias_type in {"cv", "cv_head"} and curr_vowel_start > 0.0:
        anchor_abs = curr_vowel_start

    next_onset = None
    if curr_end > 0.0 and next_gap > 0.0:
        next_onset = curr_end + next_gap

    next_vowel = None
    if curr_vowel_end > 0.0 and next_gap > 0.0:
        next_vowel = curr_vowel_end + next_gap

    ctx = AnchorTimingContext(
        file_duration_ms=_to_float(row_context.get("wav_duration_ms"), 0.0),
        timeline_start_ms=0.0,
        timeline_end_ms=_to_float(row_context.get("wav_duration_ms"), 0.0),
        anchor_abs_ms=anchor_abs if anchor_abs > 0.0 else None,
        next_onset_abs_ms=next_onset,
        next_vowel_abs_ms=next_vowel,
        alias_type=alias_type,
        language=lang,
        format_type=fmt,
        mapping_confidence=mapping_conf,
    )
    result = apply_anchor_lock(params, ctx, profile, validate_fn=validate_fn, lite=True)
    return (result.offset, result.consonant, result.cutoff, result.pre, result.ovl), tuple(result.applied_rules or ())


def _should_skip_ml_for_high_confidence_row(row_context: Dict[str, object]) -> bool:
    if not _env_flag("UTOA_ML_SKIP_HIGH_CONF_RULE_ROWS", True):
        return False
    conf_floor = _read_bounded_conf_env("UTOA_ML_SKIP_HIGH_CONF_MIN")
    blank_max = _read_bounded_conf_env("UTOA_ML_SKIP_HIGH_CONF_MAX_BLANK")
    try:
        margin_floor = float(os.environ.get("UTOA_ML_SKIP_HIGH_CONF_MIN_MARGIN", "12.0") or 12.0)
    except Exception:
        margin_floor = 12.0
    if conf_floor is None:
        conf_floor = 0.90
    if blank_max is None:
        blank_max = 0.18
    mapping_conf = _to_float(row_context.get("mapping_confidence"), 0.0)
    mapping_margin = _to_float(row_context.get("mapping_margin"), 0.0)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
    )
    return mapping_conf >= conf_floor and mapping_margin >= margin_floor and blank_conf <= blank_max


def apply_oto_ml_delta(
    language: str,
    row_context: Dict[str, object],
    bundle,
    *,
    anchor_stats: Optional[Dict[str, int]] = None,
    base_params_override: Optional[Tuple[float, float, float, float, float]] = None,
    min_confidence: float = 0.0,
    strict_constraint: bool = False,
    constraint_stats: Optional[Dict[str, int]] = None,
    pred_result=None,
) -> Tuple[float, float, float, float, float]:
    validate_oto_params = _get_validate_func(language)
    if pred_result is None and _should_skip_ml_for_high_confidence_row(row_context):
        base_offset = float(row_context.get("base_offset", 0.0))
        base_cons = float(row_context.get("base_cons", 0.0))
        base_cutoff_abs = float(row_context.get("base_cutoff_abs", 0.0))
        base_pre = float(row_context.get("base_pre", 0.0))
        base_ovl = float(row_context.get("base_ovl", 0.0))
        base_cutoff = -(base_cutoff_abs - base_offset)
        return _finalize_ml_params(
            language,
            row_context,
            (base_offset, base_cons, base_cutoff, base_pre, base_ovl),
            validate_oto_params,
            anchor_stats=anchor_stats,
            strict_constraint=strict_constraint,
            constraint_stats=constraint_stats,
        )
    pred = pred_result if pred_result is not None else predict_oto_deltas(bundle, row_context)
    pred_backend = str(getattr(pred, "route_backend", "") or getattr(pred, "backend", bundle.backend)).strip().lower()
    if pred_backend in {"coupled_nn_v1", "coupled_nn_v2_rawmel"}:
        pred_conf = float(getattr(pred, "confidence", 0.0) or 0.0)
        min_conf = max(0.0, float(min_confidence or 0.0))
        if min_conf > 0.0 and pred_conf < min_conf:
            raise CoupledLowConfidenceError(pred_conf, min_conf)
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    deltas = {
        key: _clip_delta(language, key, val, alias_type=alias_type)
        for key, val in pred.deltas.items()
    }
    deltas = _apply_language_specific_delta_policy(language, row_context, deltas)
    if base_params_override is not None:
        base_offset, base_cons, base_cutoff_abs, base_pre, base_ovl = base_params_override
    else:
        base_offset = float(row_context.get("base_offset", 0.0))
        base_cons = float(row_context.get("base_cons", 0.0))
        base_cutoff_abs = float(row_context.get("base_cutoff_abs", 0.0))
        base_pre = float(row_context.get("base_pre", 0.0))
        base_ovl = float(row_context.get("base_ovl", 0.0))
    offset = float(base_offset) + deltas.get("delta_offset", 0.0)
    cons = float(base_cons) + deltas.get("delta_cons", 0.0)
    cutoff_abs = float(base_cutoff_abs) + deltas.get("delta_cutoff", 0.0)
    cutoff = -(cutoff_abs - offset)
    pre = float(base_pre) + deltas.get("delta_pre", 0.0)
    ovl = float(base_ovl) + deltas.get("delta_ovl", 0.0)
    return _finalize_ml_params(
        language,
        row_context,
        (offset, cons, cutoff, pre, ovl),
        validate_oto_params,
        anchor_stats=anchor_stats,
        strict_constraint=strict_constraint,
        constraint_stats=constraint_stats,
    )


def apply_oto_ml_to_oto_file(
    language: str,
    oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    callback=None,
    enabled: bool = True,
    format_override: Optional[str] = None,
    policy: Optional[str] = None,
    report: Optional[Dict[str, object]] = None,
    ml_route: Optional[str] = None,
) -> int:
    policy_name = normalize_ml_policy(policy, enabled_default=enabled)
    required_policy = policy_name == "on"
    route_name = _ml_route(ml_route)
    ml_report = _ensure_report(
        report,
        stage="ml",
        policy=policy_name,
        ml_route=route_name,
        status="skipped",
        code=OK,
        message="",
        fallback_used=False,
        attempted_routes=[],
        missing_routes=[],
        invalid_routes=[],
        loaded_models=[],
        changed_lines=0,
        infer_failures=[],
        anchor_stats={},
        selector_rows=0,
        selector_model_routes=[],
        selector_mode_counts={},
        selector_guarded_routes=[],
        selector_abstain_rows=0,
        selector_abstain_reasons=[],
        selector_hard_negative_rows=0,
        model_conf_thresholds={},
        route="",
        model_confidence=0.0,
        fallback_reason="",
        blank_confidence=0.0,
        constraint_adjust_count=0,
    )
    ml_report["policy"] = policy_name

    if policy_name == "off":
        ml_report.update(make_runtime_report("ml", ML_POLICY_OFF, "ML 정책이 off로 설정되었습니다."))
        ml_report["status"] = "skipped"
        return 0

    if os.environ.get("UTOA_DISABLE_OTO_ML", "").strip().lower() in {"1", "true", "yes", "on"}:
        ml_report.update(make_runtime_report("ml", ML_DISABLED_ENV, "환경변수로 OTO ML이 비활성화되었습니다."))
        ml_report["status"] = "skipped"
        return 0

    if route_name == "autofree_v1":
        # Route orchestration is handled in post_file_pipeline:
        # coupled/lightgbm primary + optional autofree residual auxiliary.
        route_name = "legacy"

    if not oto_path or not os.path.exists(oto_path):
        ml_report.update(make_runtime_report("ml", ML_INPUT_MISSING, f"OTO 파일이 없습니다: {oto_path}"))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    rows = parse_oto_rows(oto_path)
    if not rows:
        ml_report.update(make_runtime_report("ml", ML_INPUT_MISSING, "OTO 행을 읽지 못했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    feature_rows = extract_feature_rows(
        language,
        oto_path,
        tg_dir=tg_dir,
        wav_dir=wav_dir,
        custom_phonemes_path=custom_phonemes_path,
    )
    if not feature_rows:
        ml_report.update(make_runtime_report("ml", ML_NO_FEATURES, "ML feature를 추출하지 못했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    bundle_cache: Dict[str, Optional[object]] = {}
    fallback_bundle_cache: Dict[str, Optional[object]] = {}
    selector_cache: Dict[str, Optional[object]] = {}
    route_model_min_conf: Dict[str, Optional[float]] = {}
    batch_pred_cache: Dict[str, Dict[int, OtoDeltaResult]] = {}
    route_index_cache: Dict[str, List[int]] = {}
    route_status: Dict[str, str] = {}
    route_model_dir: Dict[str, str] = {}
    route_fallback_model_dir: Dict[str, str] = {}
    route_primary_backend: Dict[str, str] = {}
    selector_enabled_runtime = _selector_runtime_enabled()
    legacy_fallback_enabled = _legacy_fallback_enabled()
    batch_enabled = _batch_inference_enabled()
    batch_size = _batch_inference_size()
    coupled_conf_mode = "model_meta" if (_coupled_min_conf_use_model_meta() and _read_bounded_conf_env("UTOA_ML_COUPLED_MIN_CONF") is None) else "global"
    model_notice = set()
    changed = 0
    route_counts: Dict[str, int] = {}
    confidence_values: List[float] = []
    fallback_reasons: List[str] = []
    blank_conf_values: List[float] = []
    strict_constraint = _coupled_strict_constraint()
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    constraint_stats = {"constraint_adjust_count": 0}
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    rows_by_index = {int(row["line_index"]): row for row in rows}
    routed_features = 0
    wav_index = build_wav_index(wav_dir) if wav_dir else {}
    rawmel_cache: Dict[str, object] = {}
    total_features = int(len(feature_rows))
    progress_every = max(20, _env_int("UTOA_ML_PROGRESS_EVERY", 80))
    loop_started_at = time.perf_counter()
    if total_features > progress_every:
        _emit(
            callback,
            f"[OTO-ML] ML refine start: rows={total_features}, route={route_name}, "
            f"gated_ensemble={'ON' if _gated_ensemble_enabled() else 'OFF'}, "
            f"selector={'ON' if selector_enabled_runtime else 'OFF'}, "
            f"legacy_fallback={'ON' if legacy_fallback_enabled else 'OFF'}, "
            f"batch={'ON' if batch_enabled else 'OFF'}({batch_size}), "
            f"min_conf_mode={coupled_conf_mode}",
        )

    for feat_idx, feat in enumerate(feature_rows, start=1):
        if total_features > progress_every and (feat_idx % progress_every == 0 or feat_idx == total_features):
            elapsed = max(0.0, time.perf_counter() - loop_started_at)
            rows_per_sec = float(feat_idx) / elapsed if elapsed > 1e-6 else 0.0
            _emit(
                callback,
                f"[OTO-ML] ML refine progress: {feat_idx}/{total_features} ({(100.0 * feat_idx / total_features):.1f}%), {rows_per_sec:.1f} rows/s",
            )
        format_type = _route_format_for_feature(language, feat, format_override=format_override)
        if not format_type:
            continue
        alias_type_for_row = str(feat.get("alias_type", "") or "").strip().lower()
        alias_family = infer_alias_family(language, feat)
        mel_patch_fallback = mel_patch_is_fallback(feat)
        routed_features += 1
        route_label = format_type if not alias_family else f"{format_type}:{alias_family}"
        if route_label not in ml_report["attempted_routes"]:
            ml_report["attempted_routes"].append(route_label)
        cache_key = route_label
        if cache_key not in bundle_cache:
            ensemble_dir = _resolve_ensemble_model_dir(language, format_type, alias_family=alias_family) if _ensemble_enabled() else None
            lightgbm_dir = _resolve_lightgbm_model_dir(language, format_type, alias_family=alias_family)
            primary_dir = ensemble_dir or lightgbm_dir
            fallback_dir = ""
            primary_backend = _read_bundle_backend(primary_dir) if primary_dir else ""
            route_primary_backend[cache_key] = str(primary_backend or "lightgbm")
            if primary_dir and primary_backend == "ensemble_v1":
                if legacy_fallback_enabled and lightgbm_dir and lightgbm_dir != primary_dir:
                    fallback_dir = lightgbm_dir

            if not primary_dir:
                route_status[cache_key] = ML_MODEL_MISSING
                route_model_dir[cache_key] = ""
                route_fallback_model_dir[cache_key] = ""
                route_primary_backend[cache_key] = ""
                ml_report["missing_routes"].append(
                    {"format_type": format_type, "alias_family": alias_family, "code": ML_MODEL_MISSING, "model_dir": ""}
                )
                route_model_min_conf[cache_key] = None
                bundle_cache[cache_key] = None
                fallback_bundle_cache[cache_key] = None
                selector_cache[cache_key] = None
            else:
                route_status[cache_key] = OK
                route_model_dir[cache_key] = str(primary_dir)
                route_fallback_model_dir[cache_key] = str(fallback_dir or "")
                delta_allowed = bool(delta_enabled_by_default(language, format_type, alias_family=alias_family))
                selector_allowed = bool(selector_enabled_runtime and selector_enabled_by_default(language, format_type, alias_family=alias_family))

                primary_bundle = None
                fallback_bundle = None
                try:
                    primary_bundle = load_oto_model_bundle(primary_dir) if delta_allowed else None
                except Exception:
                    primary_bundle = None
                if delta_allowed and fallback_dir:
                    try:
                        fallback_bundle = load_oto_model_bundle(fallback_dir)
                    except Exception:
                        fallback_bundle = None

                if delta_allowed and primary_bundle is None and fallback_bundle is None:
                    route_status[cache_key] = ML_BUNDLE_INVALID
                    ml_report["invalid_routes"].append(
                        {
                            "format_type": format_type,
                            "alias_family": alias_family,
                            "code": ML_BUNDLE_INVALID,
                            "model_dir": route_model_dir[cache_key],
                        }
                    )

                bundle_cache[cache_key] = primary_bundle
                fallback_bundle_cache[cache_key] = fallback_bundle
                route_model_min_conf[cache_key] = _route_model_min_confidence(primary_bundle, route_model_dir[cache_key])
                if route_model_min_conf[cache_key] is not None:
                    model_conf_report = ml_report.get("model_conf_thresholds", {})
                    if isinstance(model_conf_report, dict):
                        model_conf_report[cache_key] = float(route_model_min_conf[cache_key])
                        ml_report["model_conf_thresholds"] = model_conf_report

                selector_dir = lightgbm_dir or (primary_dir if _read_bundle_backend(primary_dir) == "lightgbm" else "")
                selector_payload = load_lightgbm_selector_bundle(selector_dir) if (selector_allowed and selector_dir) else None
                if selector_payload is not None:
                    selector_ok, selector_reason = _selector_passes_runtime_guard(selector_payload)
                    if not selector_ok:
                        selector_payload = None
                        ml_report["selector_guarded_routes"].append(
                            {
                                "format_type": format_type,
                                "alias_family": alias_family,
                                "model_dir": str(selector_dir),
                                "reason": selector_reason,
                            }
                        )
                selector_cache[cache_key] = selector_payload
                bundle = bundle_cache[cache_key]
                if bundle and route_model_dir[cache_key] and route_model_dir[cache_key] not in model_notice:
                    _emit(callback, f"[OTO-ML] 모델 로드 ({bundle.backend}): {route_model_dir[cache_key]}")
                    model_min_conf = route_model_min_conf.get(cache_key)
                    if model_min_conf is not None:
                        _emit(
                            callback,
                            f"[OTO-ML] min_conf(model-aware): {float(model_min_conf):.3f} [{cache_key}]",
                        )
                    ml_report["loaded_models"].append(route_model_dir[cache_key])
                    model_notice.add(route_model_dir[cache_key])
                fb = fallback_bundle_cache.get(cache_key)
                if fb and route_fallback_model_dir.get(cache_key) and route_fallback_model_dir.get(cache_key) not in model_notice:
                    _emit(callback, f"[OTO-ML] fallback 모델 로드 ({fb.backend}): {route_fallback_model_dir[cache_key]}")
                    ml_report["loaded_models"].append(route_fallback_model_dir[cache_key])
                    model_notice.add(route_fallback_model_dir[cache_key])
                if selector_cache.get(cache_key) is not None and selector_dir and selector_dir not in ml_report["selector_model_routes"]:
                    ml_report["selector_model_routes"].append(selector_dir)

                if (
                    batch_enabled
                    and primary_bundle is not None
                    and str(primary_bundle.backend).strip().lower() == "ensemble_v1"
                ):
                    try:
                        key_indices = route_index_cache.get(cache_key)
                        if key_indices is None:
                            key_indices = []
                            for i2, feat2 in enumerate(feature_rows):
                                fmt2 = _route_format_for_feature(language, feat2, format_override=format_override)
                                if not fmt2:
                                    continue
                                fam2 = infer_alias_family(language, feat2)
                                key2 = fmt2 if not fam2 else f"{fmt2}:{fam2}"
                                if key2 == cache_key:
                                    key_indices.append(i2)
                            route_index_cache[cache_key] = key_indices

                        pred_map: Dict[int, OtoDeltaResult] = {}
                        if key_indices:
                            for st in range(0, len(key_indices), batch_size):
                                chunk_indices = key_indices[st : st + batch_size]
                                chunk_feats: List[Dict[str, object]] = []
                                chunk_keys: List[int] = []
                                for idx2 in chunk_indices:
                                    feat2 = feature_rows[idx2]
                                    if mel_patch_is_fallback(feat2):
                                        continue
                                    try:
                                        if bool((primary_bundle.payload or {}).get("coupled_rawmel_enabled", False)):
                                            _ensure_rawmel_patches(
                                                feat2,
                                                wav_dir=wav_dir,
                                                wav_index=wav_index,
                                                mel_cache=rawmel_cache,
                                                patch_spec=dict((primary_bundle.payload or {}).get("coupled_patch_spec") or {}),
                                            )
                                        chunk_feats.append(feat2)
                                        chunk_keys.append(idx2)
                                    except Exception:
                                        continue
                                if not chunk_feats:
                                    continue
                                batch_results = predict_oto_deltas_batch(primary_bundle, chunk_feats)
                                for local_i, pred in enumerate(batch_results):
                                    if local_i < len(chunk_keys):
                                        pred_map[chunk_keys[local_i]] = pred
                        batch_pred_cache[cache_key] = pred_map
                    except Exception:
                        batch_pred_cache[cache_key] = {}
        bundle = bundle_cache.get(cache_key)
        fallback_bundle = fallback_bundle_cache.get(cache_key)
        selector_bundle = selector_cache.get(cache_key)
        coupled_min_conf = _coupled_min_confidence_for_alias(
            alias_type_for_row,
            language=language,
            format_type=format_type,
            base_override=route_model_min_conf.get(cache_key),
        )
        if not bundle and not fallback_bundle and selector_bundle is None:
            continue

        selected_base_override = None
        if selector_bundle is not None:
            selected = select_best_candidate(
                language,
                feat,
                lambda candidate_row, payload=selector_bundle: predict_lightgbm_selector_score(payload, candidate_row),
            )
            if selected and selected.get("candidate"):
                margin = _selector_score_margin(selected)
                base_margin = selector_min_margin(language, format_type, alias_family=alias_family)
                min_margin = _selector_dynamic_min_margin(feat, base_margin)
                if margin is not None and margin < min_margin:
                    ml_report["selector_abstain_rows"] = int(ml_report.get("selector_abstain_rows", 0)) + 1
                    ml_report["selector_abstain_reasons"].append(
                        {
                            "line_index": int(feat.get("line_index", -1)),
                            "reason": "score_margin",
                            "margin": float(margin),
                            "min_margin": float(min_margin),
                            "format_type": str(format_type),
                            "alias_family": str(alias_family),
                        }
                    )
                    if log_selector_hard_negative(
                        feat,
                        selected,
                        reason="score_margin",
                        margin=margin,
                    ):
                        ml_report["selector_hard_negative_rows"] = int(ml_report.get("selector_hard_negative_rows", 0)) + 1
                    selected = None
            if selected and selected.get("candidate"):
                candidate = selected["candidate"]
                selected_base_override = (
                    float(candidate.get("offset", 0.0)),
                    float(candidate.get("cons", 0.0)),
                    float(candidate.get("cutoff_abs", 0.0)),
                    float(candidate.get("pre", 0.0)),
                    float(candidate.get("ovl", 0.0)),
                )
                ml_report["selector_rows"] = int(ml_report.get("selector_rows", 0)) + 1
                mode = str(candidate.get("candidate_mode", "base") or "base")
                selector_mode_counts = ml_report.get("selector_mode_counts", {})
                selector_mode_counts[mode] = int(selector_mode_counts.get(mode, 0)) + 1
                ml_report["selector_mode_counts"] = selector_mode_counts
                if log_selector_hard_negative(
                    feat,
                    selected,
                    reason="risky_candidate",
                    margin=_selector_score_margin(selected),
                ):
                    ml_report["selector_hard_negative_rows"] = int(ml_report.get("selector_hard_negative_rows", 0)) + 1

        blank_conf_values.append(_to_float(feat.get("blank_span_confidence"), 0.0))
        line_index = int(feat.get("line_index", -1))
        row = rows_by_index.get(line_index)
        if row is None:
            continue
        try:
            applied_bundle = bundle
            primary_backend = str(route_primary_backend.get(cache_key, "") or "")
            if applied_bundle is None and fallback_bundle is not None:
                applied_bundle = fallback_bundle
                if primary_backend == "ensemble_v1":
                    fallback_reasons.append("ensemble_load_failed")
                elif primary_backend in {"coupled_nn_v1", "coupled_nn_v2_rawmel"}:
                    fallback_reasons.append("coupled_load_failed")
                else:
                    fallback_reasons.append("primary_load_failed")
            pred_preview = (batch_pred_cache.get(cache_key) or {}).get(feat_idx - 1)
            applied_conf = None
            route_name = ""
            if applied_bundle is not None:
                bundle_backend_norm = str(applied_bundle.backend).strip().lower()
                needs_rawmel = False
                patch_spec = {}
                if bundle_backend_norm == "coupled_nn_v2_rawmel":
                    needs_rawmel = True
                    patch_spec = dict((applied_bundle.payload or {}).get("mel_patch_spec") or {})
                elif bundle_backend_norm == "ensemble_v1":
                    needs_rawmel = bool((applied_bundle.payload or {}).get("coupled_rawmel_enabled", False))
                    patch_spec = dict((applied_bundle.payload or {}).get("coupled_patch_spec") or {})
                if pred_preview is None:
                    if needs_rawmel and mel_patch_fallback:
                        raise CoupledLowConfidenceError(0.0, coupled_min_conf)
                    if needs_rawmel:
                        _ensure_rawmel_patches(
                            feat,
                            wav_dir=wav_dir,
                            wav_index=wav_index,
                            mel_cache=rawmel_cache,
                            patch_spec=patch_spec,
                        )
                    if (
                        bundle_backend_norm in {"coupled_nn_v1", "coupled_nn_v2_rawmel"}
                        and fallback_bundle is not None
                        and str(fallback_bundle.backend).strip().lower() == "lightgbm"
                        and _gated_ensemble_enabled()
                    ):
                        from core.oto_ml_ensemble import predict_gated_ensemble_deltas

                        pred_dict = predict_gated_ensemble_deltas(
                            lightgbm_bundle={
                                "backend": str(fallback_bundle.backend),
                                "model_dir": str(fallback_bundle.model_dir),
                                "payload": fallback_bundle.payload,
                                "meta": fallback_bundle.meta,
                                "schema": fallback_bundle.feature_schema,
                            },
                            coupled_bundle={
                                "backend": str(applied_bundle.backend),
                                "model_dir": str(applied_bundle.model_dir),
                                "payload": applied_bundle.payload,
                                "meta": applied_bundle.meta,
                                "schema": applied_bundle.feature_schema,
                            },
                            feature_row=feat,
                            min_coupled_confidence=coupled_min_conf,
                        )
                        pred_preview = OtoDeltaResult(
                            deltas=dict(pred_dict.get("deltas") or {}),
                            backend=str(pred_dict.get("route_backend", "ensemble_v1") or "ensemble_v1"),
                            applied_model=str(applied_bundle.model_dir),
                            confidence=pred_dict.get("confidence"),
                            route=str(pred_dict.get("route", "")),
                            route_backend=str(pred_dict.get("route_backend", "")),
                        )
                    elif bundle_backend_norm in {"coupled_nn_v1", "coupled_nn_v2_rawmel", "ensemble_v1"}:
                        pred_preview = predict_oto_deltas(applied_bundle, feat)
                if pred_preview is not None:
                    applied_conf = float(getattr(pred_preview, "confidence", 0.0) or 0.0)
                    pred_backend = str(
                        getattr(pred_preview, "route_backend", "") or getattr(pred_preview, "backend", bundle_backend_norm)
                    )
                    pred_backend = pred_backend.strip().lower()
                    if (
                        bundle_backend_norm in {"coupled_nn_v1", "coupled_nn_v2_rawmel"}
                        and pred_backend in {"coupled_nn_v1", "coupled_nn_v2_rawmel"}
                        and applied_conf < coupled_min_conf
                    ):
                        raise CoupledLowConfidenceError(applied_conf, coupled_min_conf)

            if applied_bundle is not None:
                apply_bundle_backend = str(applied_bundle.backend).strip().lower()
                o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                    language,
                    feat,
                    applied_bundle,
                    anchor_stats=anchor_stats,
                    base_params_override=selected_base_override,
                    min_confidence=coupled_min_conf if apply_bundle_backend in {"coupled_nn_v1", "coupled_nn_v2_rawmel"} else 0.0,
                    strict_constraint=strict_constraint,
                    constraint_stats=constraint_stats,
                    pred_result=pred_preview,
                )
                route_name = str(getattr(pred_preview, "route", "") or applied_bundle.backend)
                if bundle is None and fallback_bundle is not None and primary_backend in {"coupled_nn_v1", "coupled_nn_v2_rawmel", "ensemble_v1"}:
                    route_name = f"{primary_backend}->{str(fallback_bundle.backend).strip().lower()}_load_fallback"
                if applied_conf is not None:
                    confidence_values.append(float(applied_conf))
            elif selected_base_override is not None:
                o2, c2, ct2, p2, ov2 = apply_oto_ml_selector_candidate(
                    language,
                    feat,
                    selected_base_override,
                    anchor_stats=anchor_stats,
                    strict_constraint=strict_constraint,
                    constraint_stats=constraint_stats,
                )
                route_name = "selector_only"
            else:
                continue
            route_counts[route_name] = int(route_counts.get(route_name, 0)) + 1
        except CoupledLowConfidenceError as e:
            if fallback_bundle is not None:
                fallback_reasons.append(str(e))
                try:
                    o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                        language,
                        feat,
                        fallback_bundle,
                        anchor_stats=anchor_stats,
                        base_params_override=selected_base_override,
                        strict_constraint=strict_constraint,
                        constraint_stats=constraint_stats,
                    )
                    fallback_backend = str(fallback_bundle.backend).strip().lower() or "fallback"
                    route_counts[f"{primary_backend}->{fallback_backend}"] = int(
                        route_counts.get(f"{primary_backend}->{fallback_backend}", 0)
                    ) + 1
                except Exception as fallback_exc:
                    logger.warning("OTO ML fallback inference skipped for line %s: %s", line_index, fallback_exc)
                    ml_report["infer_failures"].append({"line_index": line_index, "message": str(fallback_exc)})
                    continue
            else:
                logger.warning("OTO ML coupled confidence rejected line %s: %s", line_index, e)
                ml_report["infer_failures"].append({"line_index": line_index, "message": str(e)})
                fallback_reasons.append(str(e))
                continue
        except Exception as e:
            logger.warning("OTO ML inference skipped for line %s: %s", line_index, e)
            ml_report["infer_failures"].append({"line_index": line_index, "message": str(e)})
            if bundle is not None and str(bundle.backend).strip().lower() in {"coupled_nn_v1", "coupled_nn_v2_rawmel", "ensemble_v1"} and fallback_bundle is not None:
                try:
                    o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                        language,
                        feat,
                        fallback_bundle,
                        anchor_stats=anchor_stats,
                        base_params_override=selected_base_override,
                        strict_constraint=strict_constraint,
                        constraint_stats=constraint_stats,
                    )
                    route_counts[f"{primary_backend}->{str(fallback_bundle.backend).strip().lower()}_exception"] = int(
                        route_counts.get(f"{primary_backend}->{str(fallback_bundle.backend).strip().lower()}_exception", 0)
                    ) + 1
                    if primary_backend == "ensemble_v1":
                        fallback_reasons.append(f"ensemble_infer_exception:{e}")
                    else:
                        fallback_reasons.append(f"coupled_infer_exception:{e}")
                except Exception:
                    continue
            else:
                continue
        if (
            abs(o2 - float(row["offset"])) > 1e-6
            or abs(c2 - float(row["cons"])) > 1e-6
            or abs(ct2 - float(row["cutoff"])) > 1e-6
            or abs(p2 - float(row["pre"])) > 1e-6
            or abs(ov2 - float(row["ovl"])) > 1e-6
        ):
            changed += 1
        raw_lines[line_index] = f"{row['wav']}={row['alias']},{o2:.2f},{c2:.2f},{ct2:.2f},{p2:.2f},{ov2:.2f}"

    if changed > 0:
        with open(oto_path, "w", encoding="utf-8") as f:
            for line in raw_lines:
                f.write(line + "\n")
        _emit(callback, f"[OTO-ML] 수치 보정 적용: {changed} lines")

    if int(anchor_stats.get("anchor_locked_count", 0)) > 0:
        _emit(
            callback,
            "[AnchorLock-Lite] 요약: "
            f"anchor_locked_count={int(anchor_stats.get('anchor_locked_count', 0))}, "
            f"cutoff_clamped_count={int(anchor_stats.get('cutoff_clamped_count', 0))}, "
            f"vc_cutoff_leak_guard_count={int(anchor_stats.get('vc_cutoff_leak_guard_count', 0))}",
        )

    ml_report["anchor_stats"] = dict(anchor_stats)
    ml_report["changed_lines"] = int(changed)
    ml_report["attempted_routes"] = list(dict.fromkeys(ml_report["attempted_routes"]))
    ml_report["missing_routes"] = list(ml_report["missing_routes"])
    ml_report["invalid_routes"] = list(ml_report["invalid_routes"])
    ml_report["loaded_models"] = list(dict.fromkeys(ml_report["loaded_models"]))
    ml_report["constraint_adjust_count"] = int(constraint_stats.get("constraint_adjust_count", 0))
    if blank_conf_values:
        ml_report["blank_confidence"] = float(sum(blank_conf_values) / float(len(blank_conf_values)))
    if confidence_values:
        ml_report["model_confidence"] = float(sum(confidence_values) / float(len(confidence_values)))
    if fallback_reasons:
        uniq_reasons = list(dict.fromkeys(str(r) for r in fallback_reasons if str(r).strip()))
        ml_report["fallback_reason"] = "; ".join(uniq_reasons[:5])
    if route_counts:
        route_sorted = sorted(route_counts.items(), key=lambda item: item[1], reverse=True)
        ml_report["route"] = str(route_sorted[0][0])
    fallback_used_runtime = bool(
        fallback_reasons
        or any(
            str(name).startswith("coupled_nn_v1->")
            or str(name).startswith("coupled_nn_v2_rawmel->")
            or str(name).startswith("ensemble_v1->")
            for name in route_counts.keys()
        )
    )
    if fallback_used_runtime:
        ml_report["fallback_used"] = True

    if routed_features == 0:
        ml_report.update(make_runtime_report("ml", ML_ROUTE_UNAVAILABLE, "ML 대상 alias가 없습니다."))
        ml_report["status"] = "skipped"
        return 0

    if changed > 0:
        ml_report.update(make_runtime_report("ml", ML_APPLIED, f"{changed} lines adjusted"))
        ml_report["status"] = "applied"
        return changed

    if ml_report["loaded_models"]:
        if ml_report["infer_failures"]:
            ml_report.update(make_runtime_report("ml", ML_INFER_FAILED, "ML 추론 중 일부 행이 실패해 기본 계산으로 유지했습니다."))
            ml_report["status"] = "fallback"
            ml_report["fallback_used"] = True
        else:
            ml_report.update(make_runtime_report("ml", ML_APPLIED, "변경이 필요하지 않았습니다."))
            ml_report["status"] = "no_change"
        return changed

    if ml_report["invalid_routes"]:
        ml_report.update(make_runtime_report("ml", ML_BUNDLE_INVALID, "OTO ML 번들을 읽지 못해 기본 계산으로 유지했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    if ml_report["missing_routes"]:
        ml_report.update(make_runtime_report("ml", ML_MODEL_MISSING, "OTO ML 모델이 없어 기본 계산으로 유지했습니다."))
        ml_report["status"] = "fallback" if required_policy else "skipped"
        ml_report["fallback_used"] = required_policy
        return 0

    if ml_report["infer_failures"]:
        ml_report.update(make_runtime_report("ml", ML_INFER_FAILED, "ML 추론 실패로 기본 계산으로 유지했습니다."))
        ml_report["status"] = "fallback"
        ml_report["fallback_used"] = True
        return 0

    ml_report.update(make_runtime_report("ml", ML_ROUTE_UNAVAILABLE, "사용 가능한 ML route가 없습니다."))
    ml_report["status"] = "skipped"
    return 0
