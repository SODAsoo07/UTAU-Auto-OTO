from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

logger = logging.getLogger(__name__)

MAPPING_SUPERVISED_FEATURE_VERSION = "v1"
MAPPING_SUPERVISED_FEATURE_NAMES = [
    "target_pos",
    "cand_pos",
    "pos_delta",
    "abs_pos_delta",
    "target_has_onset",
    "target_has_coda",
    "cand_has_onset",
    "cand_has_coda",
    "onset_initial_match",
    "vowel_match",
    "coda_match",
    "target_is_glide",
    "cand_is_glide",
    "special_class_match",
    "text_match_score",
    "active_ms",
    "vowel_ms",
    "blank_conf",
    "mel_voiced",
    "mel_silence",
    "mel_unvoiced",
    "mel_breath",
    "cand_duration_ms",
]

_BUNDLE_CACHE: Dict[str, Optional["MappingSupervisedBundle"]] = {}
_GLOBAL_BUNDLE_CACHE: Dict[str, Optional["MappingSupervisedBundle"]] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _meta_float(meta: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default) or default)
    except Exception:
        return float(default)


def _safe_token(value: str, fallback: str = "general") -> str:
    t = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    t = "".join(ch for ch in t if ch.isalnum() or ch == "_")
    return t or fallback


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_dir_candidates(language: str, format_type: str) -> List[str]:
    lang = _safe_token(language, "korean")
    fmt = _safe_token(format_type or "cv", "cv")
    cands: List[str] = []

    for key in (
        f"UTOA_MAPPING_SUPERVISED_MODEL_DIR_{lang.upper()}_{fmt.upper()}",
        f"UTOA_MAPPING_SUPERVISED_MODEL_DIR_{lang.upper()}",
        "UTOA_MAPPING_SUPERVISED_MODEL_DIR",
    ):
        path = str(os.environ.get(key, "") or "").strip()
        if path:
            cands.append(path)

    model_root = str(os.environ.get("UTOA_MAPPING_SUPERVISED_MODEL_ROOT", "") or "").strip()
    if not model_root:
        model_root = os.path.join(_project_root(), "ML_models")
    version = str(os.environ.get("UTOA_MAPPING_SUPERVISED_MODEL_VERSION", "v1_mapping") or "v1_mapping").strip()
    cands.append(os.path.join(model_root, lang, fmt, version))
    if fmt != "cv":
        cands.append(os.path.join(model_root, lang, "cv", version))
    return cands


def _global_model_dir_candidates(language: str) -> List[str]:
    lang = _safe_token(language, "korean")
    cands: List[str] = []
    for key in (
        f"UTOA_MAPPING_SUPERVISED_GLOBAL_MODEL_DIR_{lang.upper()}",
        "UTOA_MAPPING_SUPERVISED_GLOBAL_MODEL_DIR",
    ):
        path = str(os.environ.get(key, "") or "").strip()
        if path:
            cands.append(path)
    model_root = str(os.environ.get("UTOA_MAPPING_SUPERVISED_MODEL_ROOT", "") or "").strip()
    if not model_root:
        model_root = os.path.join(_project_root(), "ML_models")
    version = str(os.environ.get("UTOA_MAPPING_SUPERVISED_MODEL_VERSION", "v1_mapping") or "v1_mapping").strip()
    cands.append(os.path.join(model_root, lang, "cv_global", version))
    cands.append(os.path.join(model_root, lang, "cv_global"))
    cands.append(os.path.join(model_root, lang, "cv", "v1_mapping_global"))
    return cands


@dataclass
class MappingSupervisedBundle:
    model_dir: str
    booster: object
    feature_names: List[str]
    meta: Dict[str, object]


def _canonicalize_feature_row(row: Dict[str, object], feature_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for name in feature_names:
        try:
            out.append(float(row.get(name, 0.0) or 0.0))
        except Exception:
            out.append(0.0)
    return out


def _sigmoid(x: float) -> float:
    x = float(max(-24.0, min(24.0, x)))
    return 1.0 / (1.0 + math.exp(-x))


def load_mapping_supervised_bundle(language: str, format_type: str) -> Optional[MappingSupervisedBundle]:
    if lgb is None:
        return None
    lang = _safe_token(language, "korean")
    fmt = _safe_token(format_type or "cv", "cv")
    cache_key = f"{lang}:{fmt}"
    if cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    bundle: Optional[MappingSupervisedBundle] = None
    for model_dir in _model_dir_candidates(lang, fmt):
        model_txt = os.path.join(model_dir, "model.txt")
        if not os.path.isfile(model_txt):
            continue
        schema_path = os.path.join(model_dir, "feature_schema.json")
        meta_path = os.path.join(model_dir, "meta.json")
        feature_names = list(MAPPING_SUPERVISED_FEATURE_NAMES)
        meta: Dict[str, object] = {}
        if os.path.isfile(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                names = schema.get("feature_names")
                if isinstance(names, list) and names:
                    feature_names = [str(v) for v in names if str(v).strip()]
            except Exception:
                pass
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                if isinstance(loaded_meta, dict):
                    meta = loaded_meta
            except Exception:
                pass
        try:
            booster = lgb.Booster(model_file=model_txt)
            bundle = MappingSupervisedBundle(
                model_dir=model_dir,
                booster=booster,
                feature_names=feature_names,
                meta=meta,
            )
            break
        except Exception as e:
            logger.warning("Failed to load mapping supervised model (%s): %s", model_txt, e)
            continue

    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def load_mapping_supervised_global_bundle(language: str) -> Optional[MappingSupervisedBundle]:
    if lgb is None:
        return None
    lang = _safe_token(language, "korean")
    if lang in _GLOBAL_BUNDLE_CACHE:
        return _GLOBAL_BUNDLE_CACHE[lang]

    bundle: Optional[MappingSupervisedBundle] = None
    for model_dir in _global_model_dir_candidates(lang):
        model_txt = os.path.join(model_dir, "model.txt")
        if not os.path.isfile(model_txt):
            continue
        schema_path = os.path.join(model_dir, "feature_schema.json")
        meta_path = os.path.join(model_dir, "meta.json")
        feature_names = list(MAPPING_SUPERVISED_FEATURE_NAMES)
        meta: Dict[str, object] = {}
        if os.path.isfile(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                names = schema.get("feature_names")
                if isinstance(names, list) and names:
                    feature_names = [str(v) for v in names if str(v).strip()]
            except Exception:
                pass
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                if isinstance(loaded_meta, dict):
                    meta = loaded_meta
            except Exception:
                pass
        try:
            booster = lgb.Booster(model_file=model_txt)
            bundle = MappingSupervisedBundle(
                model_dir=model_dir,
                booster=booster,
                feature_names=feature_names,
                meta=meta,
            )
            break
        except Exception as e:
            logger.warning("Failed to load global mapping supervised model (%s): %s", model_txt, e)
            continue
    _GLOBAL_BUNDLE_CACHE[lang] = bundle
    return bundle


def clear_mapping_supervised_cache() -> None:
    _BUNDLE_CACHE.clear()
    _GLOBAL_BUNDLE_CACHE.clear()


def _predict_probs(bundle: MappingSupervisedBundle, feature_rows: List[List[Dict[str, object]]]) -> Optional[List[float]]:
    flat_features: List[List[float]] = []
    for feats_i in feature_rows:
        for feat in (feats_i or []):
            flat_features.append(_canonicalize_feature_row(feat or {}, bundle.feature_names))
    if not flat_features:
        return None
    try:
        raw_pred = bundle.booster.predict(flat_features)
        preds = [float(v) for v in raw_pred]
    except Exception:
        return None
    if len(preds) != len(flat_features):
        return None
    pred_min = min(preds)
    pred_max = max(preds)
    as_prob = pred_min >= 0.0 and pred_max <= 1.0
    probs = [p if as_prob else _sigmoid(p) for p in preds]
    return [max(0.0, min(1.0, float(p))) for p in probs]


def _resolve_mapping_mode() -> str:
    raw = str(os.environ.get("UTOA_MAPPING_SUPERVISED_MODE", "auto") or "").strip().lower()
    if raw in {"global_first", "global", "global-first"}:
        return "global_first"
    if raw in {"auto", "adaptive"}:
        return "auto"
    return "format_first"


def _model_score_for_auto(bundle: Optional[MappingSupervisedBundle]) -> float:
    if bundle is None:
        return -1.0
    meta = bundle.meta or {}
    top1 = _meta_float(meta, "valid_top1_model", 0.0)
    heur = _meta_float(meta, "valid_top1_heuristic", top1)
    gain = top1 - heur
    # Reward models that clearly beat heuristic, penalize models that regress.
    return float(top1 + (0.6 * gain))


def _model_weight_scale(bundle: Optional[MappingSupervisedBundle]) -> float:
    if bundle is None:
        return 0.0
    meta = bundle.meta or {}
    top1 = _meta_float(meta, "valid_top1_model", 0.0)
    heur = _meta_float(meta, "valid_top1_heuristic", top1)
    # Base scale from absolute quality
    base = max(0.35, min(1.0, 0.35 + (top1 * 0.85)))
    # Extra penalty if model fails to beat heuristic
    if top1 + 1e-9 < heur:
        base *= 0.6
    return float(max(0.1, min(1.0, base)))


def rescore_mapping_score_rows(
    *,
    language: str,
    format_type: str,
    score_rows: List[List[float]],
    feature_rows: List[List[Dict[str, object]]],
) -> Tuple[List[List[float]], Dict[str, object]]:
    meta: Dict[str, object] = {"applied": False, "reason": ""}
    if not score_rows or not feature_rows:
        meta["reason"] = "empty"
        return score_rows, meta
    if not _env_bool("UTOA_MAPPING_SUPERVISED_ENABLE", True):
        meta["reason"] = "disabled"
        return score_rows, meta

    bundle = load_mapping_supervised_bundle(language, format_type)
    global_bundle = load_mapping_supervised_global_bundle(language)
    if bundle is None and global_bundle is None:
        meta["reason"] = "model_unavailable"
        return score_rows, meta

    row_count = len(score_rows)
    if row_count != len(feature_rows):
        meta["reason"] = "shape_mismatch"
        return score_rows, meta

    row_shapes: List[int] = []
    for i in range(row_count):
        scores_i = list(score_rows[i] or [])
        feats_i = list(feature_rows[i] or [])
        if len(scores_i) != len(feats_i):
            meta["reason"] = "shape_mismatch"
            return score_rows, meta
        row_shapes.append(len(scores_i))
    if not any(row_shapes):
        meta["reason"] = "empty"
        return score_rows, meta

    mode = _resolve_mapping_mode()
    auto_margin = max(0.0, min(0.08, _env_float("UTOA_MAPPING_SUPERVISED_AUTO_MARGIN", 0.012)))
    min_top1_for_use = max(0.45, min(0.95, _env_float("UTOA_MAPPING_SUPERVISED_MIN_VALID_TOP1", 0.55)))
    if mode == "global_first":
        primary = global_bundle
        secondary = bundle
    elif mode == "auto":
        fmt_score = _model_score_for_auto(bundle)
        glb_score = _model_score_for_auto(global_bundle)
        if bundle is not None and _meta_float(bundle.meta or {}, "valid_top1_model", 0.0) < min_top1_for_use:
            bundle = None
            fmt_score = -1.0
        if global_bundle is not None and _meta_float(global_bundle.meta or {}, "valid_top1_model", 0.0) < min_top1_for_use:
            global_bundle = None
            glb_score = -1.0
        if global_bundle is not None and (bundle is None or glb_score > (fmt_score + auto_margin)):
            primary = global_bundle
            secondary = bundle
        else:
            primary = bundle
            secondary = global_bundle
    else:
        primary = bundle
        secondary = global_bundle

    primary_probs = _predict_probs(primary, feature_rows) if primary is not None else None
    secondary_probs = _predict_probs(secondary, feature_rows) if secondary is not None else None
    if primary_probs is None and secondary_probs is None:
        meta["reason"] = "predict_failed"
        return score_rows, meta

    lang_token = _safe_token(language, "korean")
    if lang_token == "japanese":
        default_primary = 28.0
        default_secondary = 12.0
    else:
        default_primary = 34.0
        default_secondary = 16.0
    primary_weight = max(0.0, min(120.0, _env_float("UTOA_MAPPING_SUPERVISED_WEIGHT_PRIMARY", default_primary)))
    secondary_weight = max(0.0, min(120.0, _env_float("UTOA_MAPPING_SUPERVISED_WEIGHT_SECONDARY", default_secondary)))
    legacy_weight = max(0.0, min(120.0, _env_float("UTOA_MAPPING_SUPERVISED_WEIGHT", 36.0)))
    if primary is not None and secondary is None:
        primary_weight = legacy_weight
    primary_weight *= _model_weight_scale(primary)
    secondary_weight *= _model_weight_scale(secondary)

    boosted_rows: List[List[float]] = []
    offset = 0
    for width, base_row in zip(row_shapes, score_rows):
        out_row: List[float] = []
        for j in range(width):
            boost = 0.0
            if primary_probs is not None:
                p1 = float(primary_probs[offset + j])
                c1 = max(0.0, min(1.0, abs(p1 - 0.5) * 2.0))
                boost += (p1 - 0.5) * 2.0 * primary_weight * c1
            if secondary_probs is not None:
                p2 = float(secondary_probs[offset + j])
                c2 = max(0.0, min(1.0, abs(p2 - 0.5) * 2.0))
                boost += (p2 - 0.5) * 2.0 * secondary_weight * c2
            out_row.append(float(base_row[j]) + float(boost))
        offset += width
        boosted_rows.append(out_row)

    meta["applied"] = True
    meta["reason"] = "ok"
    meta["mode"] = mode
    meta["primary_weight"] = float(primary_weight)
    meta["secondary_weight"] = float(secondary_weight)
    meta["primary_model_dir"] = str(primary.model_dir if primary else "")
    meta["secondary_model_dir"] = str(secondary.model_dir if secondary else "")
    if primary_probs is not None:
        meta["primary_pred_min"] = float(min(primary_probs))
        meta["primary_pred_max"] = float(max(primary_probs))
    if secondary_probs is not None:
        meta["secondary_pred_min"] = float(min(secondary_probs))
        meta["secondary_pred_max"] = float(max(secondary_probs))
    return boosted_rows, meta


__all__ = [
    "MAPPING_SUPERVISED_FEATURE_NAMES",
    "MAPPING_SUPERVISED_FEATURE_VERSION",
    "MappingSupervisedBundle",
    "clear_mapping_supervised_cache",
    "load_mapping_supervised_bundle",
    "rescore_mapping_score_rows",
]
