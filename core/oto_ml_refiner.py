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


def _installed_model_root_for_language(language: str) -> str:
    lang = str(language).strip().lower()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models_installed", "oto_ml", lang)


def _resolve_model_dir(language: str, format_type: str) -> Optional[str]:
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


def _route_format_for_feature(language: str, feature_row: Dict[str, object], format_override: Optional[str] = None) -> Optional[str]:
    lang = str(language or "").strip().lower()
    base_format = str(format_override or feature_row.get("format_type", "general") or "general").strip().lower()
    alias_type = str(feature_row.get("alias_type", "") or "").strip().lower()
    coda_type = str(feature_row.get("coda_type", "") or "").strip().lower()

    if lang == "japanese" and base_format == "vcv" and not _has_explicit_model_override(lang):
        # 일본어 VCV는 현재 VV/VC에서 ML 이득이 크고, -CV도 보조 보정 가치가 있다.
        # 다만 일반 CV까지 전부 열면 과보정 위험이 있으므로 cv_head만 재개방한다.
        if alias_type in {"vc", "vv", "vcv", "cv_head"}:
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
            deltas["delta_offset"] *= 0.15
            deltas["delta_pre"] *= 0.20
            deltas["delta_cons"] *= 0.35
            deltas["delta_cutoff"] *= 0.50
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
            deltas["delta_pre"] = _scale_signed(deltas.get("delta_pre", 0.0), neg_scale=0.20, pos_scale=0.80)
            deltas["delta_cutoff"] = _scale_signed(deltas.get("delta_cutoff", 0.0), neg_scale=0.85, pos_scale=0.20)
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


def apply_oto_ml_delta(language: str, row_context: Dict[str, object], bundle) -> Tuple[float, float, float, float, float]:
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
    return validate_oto_params(offset, cons, cutoff, pre, ovl)


def apply_oto_ml_to_oto_file(
    language: str,
    oto_path: str,
    tg_dir: str,
    wav_dir: str,
    custom_phonemes_path: str = "",
    callback=None,
    enabled: bool = True,
    format_override: Optional[str] = None,
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
    raw_lines = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    rows_by_index = {int(row["line_index"]): row for row in rows}

    for feat in feature_rows:
        format_type = _route_format_for_feature(language, feat, format_override=format_override)
        if not format_type:
            continue
        if format_type not in bundle_cache:
            model_dir = _resolve_model_dir(language, format_type)
            bundle_cache[format_type] = load_oto_model_bundle(model_dir) if model_dir else None
            if model_dir and model_dir not in model_notice and callback:
                _emit(callback, f"[OTO-ML] 모델 로드: {model_dir}")
                model_notice.add(model_dir)
        bundle = bundle_cache.get(format_type)
        if not bundle:
            continue
        line_index = int(feat.get("line_index", -1))
        row = rows_by_index.get(line_index)
        if row is None:
            continue
        try:
            o2, c2, ct2, p2, ov2 = apply_oto_ml_delta(language, feat, bundle)
        except Exception as e:
            logger.warning("OTO ML inference skipped for line %s: %s", line_index, e)
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
    return changed
