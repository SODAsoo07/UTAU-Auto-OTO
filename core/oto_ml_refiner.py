"""
Runtime OTO ML refiner.

Applies per-row delta corrections to generated OTO files while preserving existing
validation and mel-based safety guards.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from core.oto_ml_features import extract_feature_rows, get_delta_clip_limits, parse_oto_rows
from core.oto_ml_runtime import load_oto_model_bundle, predict_oto_deltas
from core.timing_anchor_profiles import get_anchor_profile, is_anchor_lock_enabled
from core.timing_anchor_runtime import AnchorTimingContext, apply_anchor_lock

logger = logging.getLogger(__name__)

PYTORCH_REASSIGN_MAX_MOVE_RATIO_DEFAULT = 0.08
PYTORCH_REASSIGN_MAX_MOVE_RATIO_BY_LANGUAGE = {
    "korean": 0.07,
    "japanese": 0.08,
}
PYTORCH_REASSIGN_MAX_LINE_HOPS = 2
PYTORCH_REASSIGN_MIN_CONF_GAIN = 0.12


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


def _installed_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models_installed", "oto_ml", lang)


def _pytorch_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", "models", "oto_ml_pytorch", lang)


def _installed_pytorch_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models_installed", "oto_ml_pytorch", lang)


def _resolve_lightgbm_model_dir(language: str, format_type: str) -> Optional[str]:
    fmt = str(format_type or "").strip().lower() or "general"
    candidates = []
    for root in (_installed_model_root_for_language(language), _model_root_for_language(language)):
        if os.path.isfile(os.path.join(root, "model_meta.json")):
            candidates.append(root)
        else:
            candidates.append(os.path.join(root, fmt, "v1"))
            if fmt != "general":
                candidates.append(os.path.join(root, "general", "v1"))
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "model_meta.json")):
            return candidate
    return None


def _torch_task_name(language: str, format_type: str) -> Optional[str]:
    lang = str(language or "").strip().lower()
    fmt = str(format_type or "").strip().lower()
    if lang == "japanese":
        if fmt == "vcv":
            return "bridge_vcv"
        if fmt == "cvvc":
            return "bridge_cvvc"
    if lang == "korean":
        if fmt == "cvvc":
            return "bridge_cvvc"
        if fmt == "cvc":
            return "bridge_cvc"
    return None


def _resolve_pytorch_model_dir(language: str, format_type: str) -> Optional[str]:
    task_name = _torch_task_name(language, format_type)
    if not task_name:
        return None
    candidate_roots = (
        _installed_pytorch_model_root_for_language(language),
        _pytorch_model_root_for_language(language),
    )
    suffixes = ("v1", "smoke_v1")
    for root in candidate_roots:
        for suffix in suffixes:
            candidate = os.path.join(root, task_name, suffix)
            if os.path.isfile(os.path.join(candidate, "model_meta.json")):
                return candidate
    return None


def _resolve_model_dir(language: str, format_type: str, backend_preference: str = "") -> Optional[str]:
    backend = str(backend_preference or "").strip().lower()
    if backend == "pytorch":
        return _resolve_pytorch_model_dir(language, format_type) or _resolve_lightgbm_model_dir(language, format_type)
    return _resolve_lightgbm_model_dir(language, format_type)


def _route_format_for_feature(language: str, feature_row: Dict[str, object], format_override: Optional[str] = None) -> Optional[str]:
    lang = str(language or "").strip().lower()
    base_format = str(format_override or feature_row.get("format_type", "general") or "general").strip().lower()
    alias_type = str(feature_row.get("alias_type", "") or "").strip().lower()
    coda_type = str(feature_row.get("coda_type", "") or "").strip().lower()
    mapping_conf = _to_float(feature_row.get("mapping_confidence"), 1.0)
    pre_expected_gap = abs(_to_float(feature_row.get("base_pre_to_expected_ms"), 0.0))

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
            # Korean CVVC는 현재 CV/-CV에 ML을 열면 timing이 쉽게 무너진다.
            # 연결 성격이 분명한 VV/VC만 제한적으로 허용한다.
            if alias_type == "vv":
                return "cvvc"
            if alias_type == "vc" and coda_type in {"stop", "nasal", "liquid"}:
                return "cvvc"
            return None
        if lang == "korean" and base_format == "cvc":
            # Korean CVC는 현재 ML이 CV/-CV를 과도하게 망가뜨리는 사례가 있어,
            # 종성 성격의 VC에만 제한적으로 적용한다.
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

    return "general"


def _get_validate_func(language: str):
    lang = str(language).strip().lower()
    if lang == "japanese":
        from core.ja_oto_generator import validate_oto_params
    else:
        from core.oto_generator import validate_oto_params
    return validate_oto_params


def _clip_delta(language: str, target: str, value: float) -> float:
    clip = get_delta_clip_limits(language).get(target)
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
    format_type = str(row_context.get("format_type", "") or "").strip().lower()
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
    if format_type == "cvvc" and alias_type in {"cv", "cv_head", "vc"}:
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
            return deltas

    return deltas


def _apply_language_specific_delta_policy(language: str, row_context: Dict[str, object], deltas: Dict[str, float]) -> Dict[str, float]:
    lang = str(language or "").strip().lower()
    adjusted = dict(deltas)
    if lang == "japanese":
        return _apply_japanese_delta_policy(row_context, adjusted)
    return adjusted


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _is_truthy_num(value: object) -> bool:
    return _to_float(value, 0.0) >= 0.5


def _base_pre_abs(row: Dict[str, object]) -> float:
    return _to_float(row.get("base_offset"), 0.0) + _to_float(row.get("base_pre"), 0.0)


def _relative_move_limit_ms(row: Dict[str, object]) -> float:
    # 이동 제한은 WAV 전체 길이에 대한 상대값으로 계산한다.
    lang = str(row.get("language", "") or "").strip().lower()
    ratio = float(PYTORCH_REASSIGN_MAX_MOVE_RATIO_BY_LANGUAGE.get(lang, PYTORCH_REASSIGN_MAX_MOVE_RATIO_DEFAULT))
    wav_ms = max(1.0, _to_float(row.get("wav_duration_ms"), 0.0))
    return wav_ms * ratio


def _can_try_pytorch_reassign(row: Dict[str, object]) -> bool:
    alias_type = str(row.get("alias_type", "") or "").strip().lower()
    if alias_type not in {"cv", "cv_head", "vc", "vv", "vcv"}:
        return False
    conf = _to_float(row.get("mapping_confidence"), 1.0)
    margin = _to_float(row.get("words_vs_alias_score_margin"), 0.0)
    return (
        conf < 0.72
        or _is_truthy_num(row.get("used_nuclei_fallback"))
        or margin < -0.05
    )


def _candidate_structurally_compatible(src: Dict[str, object], cand: Dict[str, object]) -> bool:
    if str(src.get("alias_type", "") or "").strip().lower() != str(cand.get("alias_type", "") or "").strip().lower():
        return False
    src_group = str(src.get("alias_group", "") or "").strip().lower()
    cand_group = str(cand.get("alias_group", "") or "").strip().lower()
    if src_group and cand_group and src_group != cand_group:
        return False
    src_onset = str(src.get("onset_class", "") or "").strip().lower()
    cand_onset = str(cand.get("onset_class", "") or "").strip().lower()
    if src_onset and cand_onset and src_onset != cand_onset:
        return False
    src_coda = str(src.get("coda_type", "") or "").strip().lower()
    cand_coda = str(cand.get("coda_type", "") or "").strip().lower()
    if src_coda and cand_coda and src_coda != cand_coda:
        return False
    return True


def _select_pytorch_reassign_candidate(row: Dict[str, object], same_wav_rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not _can_try_pytorch_reassign(row):
        return None
    if not same_wav_rows:
        return None

    src_line = _to_int(row.get("line_index"), -1)
    src_pre_abs = _base_pre_abs(row)
    src_conf = _to_float(row.get("mapping_confidence"), 0.0)
    src_margin = _to_float(row.get("words_vs_alias_score_margin"), 0.0)
    max_move_ms = _relative_move_limit_ms(row)

    best = None
    best_score = float("-inf")

    for cand in same_wav_rows:
        cand_line = _to_int(cand.get("line_index"), -1)
        if cand_line < 0 or cand_line == src_line:
            continue
        if abs(cand_line - src_line) > int(PYTORCH_REASSIGN_MAX_LINE_HOPS):
            continue
        if not _candidate_structurally_compatible(row, cand):
            continue

        cand_pre_abs = _base_pre_abs(cand)
        move_ms = abs(cand_pre_abs - src_pre_abs)
        if move_ms > max_move_ms:
            continue

        cand_conf = _to_float(cand.get("mapping_confidence"), 0.0)
        conf_gain = cand_conf - src_conf
        if conf_gain < float(PYTORCH_REASSIGN_MIN_CONF_GAIN):
            continue

        cand_margin = _to_float(cand.get("words_vs_alias_score_margin"), 0.0)
        score = (conf_gain * 2.0) + ((cand_margin - src_margin) * 0.45) - (move_ms / max(1e-6, max_move_ms))
        if _is_truthy_num(cand.get("used_nuclei_fallback")):
            score -= 0.4
        if _is_truthy_num(cand.get("used_alias_based_syllables")):
            score -= 0.2

        if score > best_score:
            best_score = score
            best = cand

    if best_score < 0.15:
        return None
    return best


def _passes_reassign_post_guard(src: Dict[str, object], offset: float, pre: float, cutoff: float) -> bool:
    max_move_ms = _relative_move_limit_ms(src)
    src_pre_abs = _base_pre_abs(src)
    new_pre_abs = float(offset) + float(pre)
    if abs(new_pre_abs - src_pre_abs) > max_move_ms:
        return False
    wav_ms = _to_float(src.get("wav_duration_ms"), 0.0)
    if wav_ms > 0:
        cutoff_abs = float(offset) + abs(float(cutoff))
        if cutoff_abs > (wav_ms + 1.0):
            return False
    return True


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
) -> Tuple[float, float, float, float, float]:
    validate_oto_params = _get_validate_func(language)
    pred = predict_oto_deltas(bundle, row_context)
    deltas = {
        key: _clip_delta(language, key, val)
        for key, val in pred.deltas.items()
    }
    deltas = _apply_language_specific_delta_policy(language, row_context, deltas)
    offset = float(row_context.get("base_offset", 0.0)) + deltas.get("delta_offset", 0.0)
    cons = float(row_context.get("base_cons", 0.0)) + deltas.get("delta_cons", 0.0)
    cutoff_abs = float(row_context.get("base_cutoff_abs", 0.0)) + deltas.get("delta_cutoff", 0.0)
    cutoff = -(cutoff_abs - offset)
    pre = float(row_context.get("base_pre", 0.0)) + deltas.get("delta_pre", 0.0)
    ovl = float(row_context.get("base_ovl", 0.0)) + deltas.get("delta_ovl", 0.0)
    out = validate_oto_params(offset, cons, cutoff, pre, ovl)
    out, applied_rules = _apply_anchor_lock_lite_after_ml(language, row_context, out, validate_oto_params)
    if anchor_stats is not None and applied_rules:
        anchor_stats["anchor_locked_count"] = int(anchor_stats.get("anchor_locked_count", 0)) + 1
        if "cutoff_next_onset_clamp" in applied_rules or "cutoff_next_vowel_clamp" in applied_rules:
            anchor_stats["cutoff_clamped_count"] = int(anchor_stats.get("cutoff_clamped_count", 0)) + 1
            if str(row_context.get("alias_type", "") or "").strip().lower() == "vc":
                anchor_stats["vc_cutoff_leak_guard_count"] = int(anchor_stats.get("vc_cutoff_leak_guard_count", 0)) + 1
    return out


def apply_oto_ml_to_oto_file(
    language: str,
    oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    callback=None,
    enabled: bool = True,
    backend_preference: str = "",
    format_override: Optional[str] = None,
    enable_pytorch_reassign: bool = True,
) -> int:
    if not enabled:
        return 0
    if os.environ.get("UTOA_DISABLE_OTO_ML", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if not oto_path or not os.path.exists(oto_path):
        return 0

    rows = parse_oto_rows(oto_path)
    if not rows:
        return 0
    feature_rows = extract_feature_rows(language, oto_path, tg_dir=tg_dir, wav_dir=wav_dir, custom_phonemes_path=custom_phonemes_path)
    if not feature_rows:
        return 0

    bundle_cache: Dict[str, Optional[object]] = {}
    model_notice = set()
    changed = 0
    anchor_stats = {
        "anchor_locked_count": 0,
        "cutoff_clamped_count": 0,
        "vc_cutoff_leak_guard_count": 0,
    }
    remapped = 0
    remap_examples: List[str] = []
    raw_lines = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    rows_by_index = {int(row["line_index"]): row for row in rows}
    features_by_wav: Dict[str, List[Dict[str, object]]] = {}
    for feat in feature_rows:
        wav_key = str(feat.get("wav_norm", "") or feat.get("wav", "") or "").strip().lower()
        features_by_wav.setdefault(wav_key, []).append(feat)

    for feat in feature_rows:
        format_type = _route_format_for_feature(language, feat, format_override=format_override)
        if not format_type:
            continue
        cache_key = f"{backend_preference or 'default'}::{format_type}"
        if cache_key not in bundle_cache:
            model_dir = _resolve_model_dir(language, format_type, backend_preference=backend_preference)
            bundle_cache[cache_key] = load_oto_model_bundle(model_dir) if model_dir else None
            bundle = bundle_cache[cache_key]
            if bundle and model_dir and model_dir not in model_notice and callback:
                _emit(callback, f"[OTO-ML] 모델 로드 ({bundle.backend}): {model_dir}")
                model_notice.add(model_dir)
        bundle = bundle_cache.get(cache_key)
        if not bundle:
            continue
        line_index = int(feat.get("line_index", -1))
        row = rows_by_index.get(line_index)
        if row is None:
            continue
        infer_feat = feat
        if enable_pytorch_reassign and getattr(bundle, "backend", "") == "pytorch":
            wav_key = str(feat.get("wav_norm", "") or feat.get("wav", "") or "").strip().lower()
            candidate = _select_pytorch_reassign_candidate(feat, features_by_wav.get(wav_key, []))
            if candidate is not None:
                infer_feat = candidate
        try:
            o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                language,
                infer_feat,
                bundle,
                anchor_stats=anchor_stats,
            )
            if infer_feat is not feat and not _passes_reassign_post_guard(feat, o2, p2, ct2):
                infer_feat = feat
                o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(
                    language,
                    infer_feat,
                    bundle,
                    anchor_stats=anchor_stats,
                )
        except Exception as e:
            logger.warning("OTO ML inference skipped for line %s: %s", line_index, e)
            continue
        if infer_feat is not feat:
            remapped += 1
            if len(remap_examples) < 3:
                remap_examples.append(
                    f"line {line_index}: {feat.get('alias','')} -> line {int(infer_feat.get('line_index', -1))}"
                )
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
        if remapped > 0:
            _emit(callback, f"[OTO-ML] 제한 재매핑 적용: {remapped} lines (WAV 상대 이동 제한)")
            if remap_examples:
                _emit(callback, f"[OTO-ML] 재매핑 예시: {' | '.join(remap_examples)}")
    if int(anchor_stats.get("anchor_locked_count", 0)) > 0:
        _emit(
            callback,
            "[AnchorLock-Lite] 요약: "
            f"anchor_locked_count={int(anchor_stats.get('anchor_locked_count', 0))}, "
            f"cutoff_clamped_count={int(anchor_stats.get('cutoff_clamped_count', 0))}, "
            f"vc_cutoff_leak_guard_count={int(anchor_stats.get('vc_cutoff_leak_guard_count', 0))}",
        )
    return changed
