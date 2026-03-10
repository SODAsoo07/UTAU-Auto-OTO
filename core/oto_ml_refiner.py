"""
Runtime OTO ML refiner.

Applies per-row delta corrections to generated OTO files while preserving existing
validation and mel-based safety guards.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

from core.oto_ml_features import extract_feature_rows, get_delta_clip_limits, parse_oto_rows
from core.oto_ml_runtime import load_oto_model_bundle, predict_oto_deltas
from core.oto_ml_lightgbm import load_lightgbm_selector_bundle, predict_lightgbm_selector_score
from core.oto_ml_policy import (
    delta_enabled_by_default,
    infer_alias_family,
    normalize_alias_family,
    selector_min_margin,
    selector_enabled_by_default,
)
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


def _resolve_lightgbm_model_dir(language: str, format_type: str, alias_family: str = "") -> Optional[str]:
    fmt = normalize_format_type(language, format_type) or "general"
    family = normalize_alias_family(alias_family)
    lang = str(language or "").strip().lower()
    candidates = []
    for root in (
        _structured_export_model_root_for_language(language),
        _workspace_model_root_for_language(language),
        _installed_model_root_for_language(language),
        _model_root_for_language(language),
    ):
        if os.path.isfile(os.path.join(root, "model_meta.json")):
            candidates.append(root)
        else:
            if family:
                candidates.append(os.path.join(root, fmt, "families", family, "v1"))
                candidates.append(os.path.join(root, f"{fmt}_{family}", "v1"))
            candidates.append(os.path.join(root, fmt, "v1"))
            if fmt == "cvc":
                if family:
                    candidates.append(os.path.join(root, "cv", "families", family, "v1"))
                    candidates.append(os.path.join(root, f"cv_{family}", "v1"))
                candidates.append(os.path.join(root, "cv", "v1"))
            if fmt != "general":
                if family:
                    candidates.append(os.path.join(root, "general", "families", family, "v1"))
                    candidates.append(os.path.join(root, f"general_{family}", "v1"))
                candidates.append(os.path.join(root, "general", "v1"))
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
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "model_meta.json")):
            return candidate
    if _ml_same_language_borrow_only():
        return None
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
                return candidate
    return None


def _resolve_model_dir(language: str, format_type: str, alias_family: str = "") -> Optional[str]:
    return _resolve_lightgbm_model_dir(language, format_type, alias_family=alias_family)


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
        return "cv"
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

    if alias_type not in {"vc", "vv"}:
        return deltas

    if alias_type == "vc":
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
    else:
        # VV는 pre-ovl 간격(리듬)을 보존하도록 ovl/offset을 강하게 제한한다.
        deltas["delta_offset"] = _scale_signed(deltas.get("delta_offset", 0.0), neg_scale=0.44, pos_scale=0.54)
        deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.56, pos_scale=0.78)
        deltas["delta_cons"] = _scale_signed(deltas.get("delta_cons", 0.0), neg_scale=0.72, pos_scale=0.92)
        deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.80, pos_scale=0.58)
        deltas["delta_ovl"] = _scale_signed(deltas.get("delta_ovl", 0.0), neg_scale=0.62, pos_scale=0.62)

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


def _apply_korean_bridge_post_guard(
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
) -> Tuple[float, float, float, float, float]:
    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    coda_type = str(row_context.get("coda_type", "") or "").strip().lower()
    if alias_type not in {"vc", "vv"}:
        return params

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
    elif lang == "japanese":
        out = _apply_japanese_bridge_post_guard(row_context, out, validate_fn)
        out = _apply_japanese_cvvc_cv_post_guard(row_context, out, validate_fn)
    return out


def _finalize_ml_params(
    language: str,
    row_context: Dict[str, object],
    params: Tuple[float, float, float, float, float],
    validate_fn,
    anchor_stats: Optional[Dict[str, int]] = None,
) -> Tuple[float, float, float, float, float]:
    validate_row = _build_alias_aware_validator(validate_fn, row_context)
    out = validate_row(*params)
    out = _apply_language_specific_post_guard(language, row_context, out, validate_row)
    out, applied_rules = _apply_anchor_lock_lite_after_ml(language, row_context, out, validate_row)
    if anchor_stats is not None and applied_rules:
        anchor_stats["anchor_locked_count"] = int(anchor_stats.get("anchor_locked_count", 0)) + 1
        if "cutoff_next_onset_clamp" in applied_rules or "cutoff_next_vowel_clamp" in applied_rules:
            anchor_stats["cutoff_clamped_count"] = int(anchor_stats.get("cutoff_clamped_count", 0)) + 1
            if str(row_context.get("alias_type", "") or "").strip().lower() == "vc":
                anchor_stats["vc_cutoff_leak_guard_count"] = int(anchor_stats.get("vc_cutoff_leak_guard_count", 0)) + 1
    return out


def apply_oto_ml_selector_candidate(
    language: str,
    row_context: Dict[str, object],
    base_params_override: Tuple[float, float, float, float, float],
    *,
    anchor_stats: Optional[Dict[str, int]] = None,
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
    return _finalize_ml_params(language, row_context, params, validate_oto_params, anchor_stats=anchor_stats)


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
        mapping_confidence=_to_float(row_context.get("mapping_confidence"), 1.0),
    )
    result = apply_anchor_lock(params, ctx, profile, validate_fn=validate_fn, lite=True)
    return (result.offset, result.consonant, result.cutoff, result.pre, result.ovl), tuple(result.applied_rules or ())


def apply_oto_ml_delta(
    language: str,
    row_context: Dict[str, object],
    bundle,
    *,
    anchor_stats: Optional[Dict[str, int]] = None,
    base_params_override: Optional[Tuple[float, float, float, float, float]] = None,
) -> Tuple[float, float, float, float, float]:
    validate_oto_params = _get_validate_func(language)
    pred = predict_oto_deltas(bundle, row_context)
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
) -> int:
    policy_name = normalize_ml_policy(policy, enabled_default=enabled)
    required_policy = policy_name == "on"
    ml_report = _ensure_report(
        report,
        stage="ml",
        policy=policy_name,
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
    selector_cache: Dict[str, Optional[object]] = {}
    route_status: Dict[str, str] = {}
    route_model_dir: Dict[str, str] = {}
    model_notice = set()
    changed = 0
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    rows_by_index = {int(row["line_index"]): row for row in rows}
    routed_features = 0

    for feat in feature_rows:
        format_type = _route_format_for_feature(language, feat, format_override=format_override)
        if not format_type:
            continue
        alias_family = infer_alias_family(language, feat)
        routed_features += 1
        route_label = format_type if not alias_family else f"{format_type}:{alias_family}"
        if route_label not in ml_report["attempted_routes"]:
            ml_report["attempted_routes"].append(route_label)
        cache_key = route_label
        if cache_key not in bundle_cache:
            ready = check_oto_ml_ready(language, format_type, alias_family=alias_family)
            route_status[cache_key] = str(ready.get("code", OK) or OK)
            route_model_dir[cache_key] = str(ready.get("model_dir", "") or "")
            if ready.get("code") == ML_MODEL_MISSING:
                ml_report["missing_routes"].append(
                    {"format_type": format_type, "alias_family": alias_family, "code": ML_MODEL_MISSING, "model_dir": route_model_dir[cache_key]}
                )
                bundle_cache[cache_key] = None
                selector_cache[cache_key] = None
            elif ready.get("code") == ML_BUNDLE_INVALID:
                ml_report["invalid_routes"].append(
                    {"format_type": format_type, "alias_family": alias_family, "code": ML_BUNDLE_INVALID, "model_dir": route_model_dir[cache_key]}
                )
                bundle_cache[cache_key] = None
                selector_cache[cache_key] = None
            else:
                delta_allowed = bool(delta_enabled_by_default(language, format_type, alias_family=alias_family))
                selector_allowed = bool(
                    route_model_dir[cache_key]
                    and selector_enabled_by_default(language, format_type, alias_family=alias_family)
                )
                bundle_cache[cache_key] = load_oto_model_bundle(route_model_dir[cache_key]) if delta_allowed else None
                selector_payload = load_lightgbm_selector_bundle(route_model_dir[cache_key]) if selector_allowed else None
                if selector_payload is not None:
                    selector_ok, selector_reason = _selector_passes_runtime_guard(selector_payload)
                    if not selector_ok:
                        selector_payload = None
                        ml_report["selector_guarded_routes"].append(
                            {
                                "format_type": format_type,
                                "alias_family": alias_family,
                                "model_dir": route_model_dir[cache_key],
                                "reason": selector_reason,
                            }
                        )
                selector_cache[cache_key] = selector_payload
                bundle = bundle_cache[cache_key]
                if bundle and route_model_dir[cache_key] and route_model_dir[cache_key] not in model_notice:
                    _emit(callback, f"[OTO-ML] 모델 로드 ({bundle.backend}): {route_model_dir[cache_key]}")
                    ml_report["loaded_models"].append(route_model_dir[cache_key])
                    model_notice.add(route_model_dir[cache_key])
                if selector_cache.get(cache_key) is not None and route_model_dir[cache_key] not in ml_report["selector_model_routes"]:
                    ml_report["selector_model_routes"].append(route_model_dir[cache_key])
        bundle = bundle_cache.get(cache_key)
        selector_bundle = selector_cache.get(cache_key)
        if not bundle and selector_bundle is None:
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

        line_index = int(feat.get("line_index", -1))
        row = rows_by_index.get(line_index)
        if row is None:
            continue
        try:
            if bundle is not None:
                o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                    language,
                    feat,
                    bundle,
                    anchor_stats=anchor_stats,
                    base_params_override=selected_base_override,
                )
            elif selected_base_override is not None:
                o2, c2, ct2, p2, ov2 = apply_oto_ml_selector_candidate(
                    language,
                    feat,
                    selected_base_override,
                    anchor_stats=anchor_stats,
                )
            else:
                continue
        except Exception as e:
            logger.warning("OTO ML inference skipped for line %s: %s", line_index, e)
            ml_report["infer_failures"].append({"line_index": line_index, "message": str(e)})
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
