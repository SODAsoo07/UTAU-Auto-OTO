from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _emit(callback: Optional[Callable[[str], None]], message: str) -> None:
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        return


def _normalize_bool_str(raw: str, current: bool) -> bool:
    val = str(raw or "").strip().lower()
    if val in {"0", "false", "off", "no"}:
        return False
    if val in {"1", "true", "on", "yes"}:
        return True
    return bool(current)


@dataclass
class KoreanNormalizedOptions:
    params: Dict[str, Any]
    fallback_format: str
    auto_gen_format: str
    use_sinsy_labels: bool
    sinsy_label_path: str
    words_fallback_enabled: bool
    spn_ratio_threshold: float
    min_vowel_phone_ratio: float
    debug_reason_logging: bool
    anchor_profile_path: str
    confidence_threshold: Optional[float]
    max_index_jump_default: int
    max_index_jump_high_conf: int
    cleanup_timing_jsonl: bool
    disable_cvvc_order_lock: bool


@dataclass
class JapaneseNormalizedOptions:
    params: Dict[str, Any]
    fallback_format: str
    auto_gen_format: str
    forced_format: Optional[str]
    use_sinsy_labels: bool
    sinsy_label_path: str
    words_fallback_enabled: bool
    spn_ratio_threshold: float
    min_vowel_phone_ratio: float
    debug_reason_logging: bool
    trace_logging: bool
    confidence_threshold: Optional[float]
    cleanup_timing_jsonl: bool


def _normalize_auto_format(
    *,
    language: str,
    fallback_format: str,
    auto_format: Optional[str],
) -> str:
    if auto_format:
        try:
            from core.format_type_utils import normalize_auto_format_value

            normalized = normalize_auto_format_value(language, auto_format)
            if normalized:
                fallback_format = normalized
        except Exception:
            pass
    return str(fallback_format or "").strip().lower()


def normalize_korean_generation_options(
    *,
    params: Optional[Dict[str, Any]],
    default_params: Dict[str, Any],
    fallback_format: str,
    auto_format: Optional[str],
    kr_mapping_words_fallback_enabled: bool,
    kr_mapping_spn_ratio_threshold: float,
    kr_mapping_min_vowel_phone_ratio: float,
    kr_mapping_debug_reason_logging: bool,
    kr_anchor_profile_path: str,
    kr_mapping_confidence_threshold: Optional[float],
    kr_mapping_max_index_jump_default: int,
    kr_mapping_max_index_jump_high_conf: int,
    cleanup_timing_jsonl: bool,
    callback: Optional[Callable[[str], None]] = None,
) -> KoreanNormalizedOptions:
    normalized_params = (params or default_params.copy())
    use_sinsy_labels = bool(normalized_params.get("USE_SINSY_LABELS", False)) if normalized_params else False
    sinsy_label_path = str(normalized_params.get("SINSY_LABEL_PATH", "") or "").strip() if normalized_params else ""

    normalized_fallback = _normalize_auto_format(
        language="korean",
        fallback_format=fallback_format or "cvvc",
        auto_format=auto_format,
    )
    try:
        from core.format_type_utils import normalize_format_type

        auto_gen_format = normalize_format_type("korean", normalized_fallback or "cvvc") or "cvvc"
    except Exception:
        auto_gen_format = normalized_fallback or "cvvc"
    if auto_gen_format not in {"cv", "cvc", "cvvc", "vcv"}:
        _emit(
            callback,
            (
                "⚠ 자동 에일리어스 생성은 현재 CV/연단음, CVC, CVVC, VCV만 지원합니다. "
                f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
            ),
        )
        auto_gen_format = "cvvc"

    words_fallback_enabled = _normalize_bool_str(
        os.environ.get("UTOA_KR_MAPPING_WORDS_FALLBACK", ""),
        kr_mapping_words_fallback_enabled,
    )
    spn_ratio_threshold = _env_float("UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD", kr_mapping_spn_ratio_threshold)
    min_vowel_phone_ratio = _env_float("UTOA_KR_MAPPING_MIN_VOWEL_PHONE_RATIO", kr_mapping_min_vowel_phone_ratio)
    debug_reason_logging = _normalize_bool_str(
        os.environ.get("UTOA_KR_MAPPING_DEBUG_REASON", ""),
        kr_mapping_debug_reason_logging,
    )
    use_sinsy_labels = _normalize_bool_str(os.environ.get("UTOA_USE_SINSY_LABELS", ""), use_sinsy_labels)
    sinsy_label_path = str(os.environ.get("UTOA_SINSY_LABEL_PATH", "") or "").strip() or sinsy_label_path
    anchor_profile_path = str(os.environ.get("UTOA_KR_ANCHOR_PROFILE_PATH", "") or "").strip() or str(
        kr_anchor_profile_path or ""
    ).strip()

    confidence_threshold = kr_mapping_confidence_threshold
    env_conf = str(os.environ.get("UTOA_KR_MAPPING_CONF_THRESHOLD", "")).strip()
    if env_conf:
        try:
            confidence_threshold = float(env_conf)
        except Exception:
            pass

    max_index_jump_default = _env_int("UTOA_KR_MAPPING_MAX_INDEX_JUMP_DEFAULT", kr_mapping_max_index_jump_default)
    max_index_jump_high_conf = _env_int(
        "UTOA_KR_MAPPING_MAX_INDEX_JUMP_HIGH_CONF",
        kr_mapping_max_index_jump_high_conf,
    )
    cleanup_timing = _normalize_bool_str(os.environ.get("UTOA_CLEANUP_TIMING_JSONL", ""), cleanup_timing_jsonl)
    disable_cvvc_order_lock = _env_bool("UTOA_KR_DISABLE_CVVC_ORDER_LOCK", False)

    return KoreanNormalizedOptions(
        params=normalized_params,
        fallback_format=normalized_fallback or "cvvc",
        auto_gen_format=auto_gen_format,
        use_sinsy_labels=use_sinsy_labels,
        sinsy_label_path=sinsy_label_path,
        words_fallback_enabled=words_fallback_enabled,
        spn_ratio_threshold=float(spn_ratio_threshold),
        min_vowel_phone_ratio=float(min_vowel_phone_ratio),
        debug_reason_logging=debug_reason_logging,
        anchor_profile_path=anchor_profile_path,
        confidence_threshold=confidence_threshold,
        max_index_jump_default=int(max_index_jump_default),
        max_index_jump_high_conf=int(max_index_jump_high_conf),
        cleanup_timing_jsonl=bool(cleanup_timing),
        disable_cvvc_order_lock=bool(disable_cvvc_order_lock),
    )


def normalize_japanese_generation_options(
    *,
    params: Optional[Dict[str, Any]],
    default_params: Dict[str, Any],
    fallback_format: str,
    auto_format: Optional[str],
    ja_mapping_words_fallback_enabled: bool,
    ja_mapping_spn_ratio_threshold: float,
    ja_mapping_min_vowel_phone_ratio: float,
    ja_mapping_debug_reason_logging: bool,
    ja_mapping_confidence_threshold: Optional[float],
    cleanup_timing_jsonl: bool,
    callback: Optional[Callable[[str], None]] = None,
) -> JapaneseNormalizedOptions:
    normalized_params = (params or default_params.copy())
    use_sinsy_labels = bool(normalized_params.get("USE_SINSY_LABELS", False)) if normalized_params else False
    sinsy_label_path = str(normalized_params.get("SINSY_LABEL_PATH", "") or "").strip() if normalized_params else ""

    forced_format = None
    normalized_fallback = str(fallback_format or "cvvc").strip().lower()
    if auto_format:
        try:
            from core.format_type_utils import normalize_auto_format_value

            normalized_auto_format = normalize_auto_format_value("japanese", auto_format)
            if normalized_auto_format:
                forced_format = normalized_auto_format
                normalized_fallback = normalized_auto_format
        except Exception:
            pass

    auto_gen_format = normalized_fallback or "cvvc"
    if auto_gen_format not in {"cv", "cvvc", "vcv"}:
        _emit(
            callback,
            (
                "⚠ 자동 에일리어스 생성은 현재 CV/CVVC/VCV만 지원합니다. "
                f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
            ),
        )
        auto_gen_format = "cvvc"

    words_fallback_enabled = _normalize_bool_str(
        os.environ.get("UTOA_JA_MAPPING_WORDS_FALLBACK", ""),
        ja_mapping_words_fallback_enabled,
    )
    use_sinsy_labels = _normalize_bool_str(os.environ.get("UTOA_USE_SINSY_LABELS", ""), use_sinsy_labels)
    sinsy_label_path = str(os.environ.get("UTOA_SINSY_LABEL_PATH", "") or "").strip() or sinsy_label_path
    spn_ratio_threshold = _env_float("UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD", ja_mapping_spn_ratio_threshold)
    min_vowel_phone_ratio = _env_float("UTOA_JA_MAPPING_MIN_VOWEL_PHONE_RATIO", ja_mapping_min_vowel_phone_ratio)
    debug_reason_logging = _normalize_bool_str(
        os.environ.get("UTOA_JA_MAPPING_DEBUG_REASON", ""),
        ja_mapping_debug_reason_logging,
    )
    trace_logging = _normalize_bool_str(os.environ.get("UTOA_JA_MAPPING_TRACE", ""), True)

    confidence_threshold = ja_mapping_confidence_threshold
    env_conf = str(os.environ.get("UTOA_JA_MAPPING_CONF_THRESHOLD", "")).strip()
    if env_conf:
        try:
            confidence_threshold = float(env_conf)
        except Exception:
            pass

    cleanup_timing = _normalize_bool_str(os.environ.get("UTOA_CLEANUP_TIMING_JSONL", ""), cleanup_timing_jsonl)

    return JapaneseNormalizedOptions(
        params=normalized_params,
        fallback_format=normalized_fallback,
        auto_gen_format=auto_gen_format,
        forced_format=forced_format,
        use_sinsy_labels=use_sinsy_labels,
        sinsy_label_path=sinsy_label_path,
        words_fallback_enabled=words_fallback_enabled,
        spn_ratio_threshold=float(spn_ratio_threshold),
        min_vowel_phone_ratio=float(min_vowel_phone_ratio),
        debug_reason_logging=debug_reason_logging,
        trace_logging=trace_logging,
        confidence_threshold=confidence_threshold,
        cleanup_timing_jsonl=bool(cleanup_timing),
    )
