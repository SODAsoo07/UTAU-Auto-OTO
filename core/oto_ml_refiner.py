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


def apply_oto_ml_delta(language: str, row_context: Dict[str, object], bundle) -> Tuple[float, float, float, float, float]:
    validate_oto_params = _get_validate_func(language)
    pred = predict_oto_deltas(bundle, row_context)
    deltas = {
        key: _clip_delta(language, key, val)
        for key, val in pred.deltas.items()
    }
    offset = float(row_context.get("base_offset", 0.0)) + deltas.get("delta_offset", 0.0)
    cons = float(row_context.get("base_cons", 0.0)) + deltas.get("delta_cons", 0.0)
    cutoff_abs = float(row_context.get("base_cutoff_abs", 0.0)) + deltas.get("delta_cutoff", 0.0)
    cutoff = -(cutoff_abs - offset)
    pre = float(row_context.get("base_pre", 0.0)) + deltas.get("delta_pre", 0.0)
    ovl = float(row_context.get("base_ovl", 0.0)) + deltas.get("delta_ovl", 0.0)
    return validate_oto_params(offset, cons, cutoff, pre, ovl)


def apply_oto_ml_to_oto_file(language: str, oto_path: str, tg_dir: str, wav_dir: str, custom_phonemes_path: str = "", callback=None) -> int:
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
        format_type = str(feat.get("format_type", "general")).lower() or "general"
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
