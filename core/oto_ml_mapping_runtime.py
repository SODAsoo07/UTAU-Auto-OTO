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

from core.mapping_supervised_runtime import MAPPING_SUPERVISED_FEATURE_NAMES

logger = logging.getLogger(__name__)

_BUNDLE_CACHE: Dict[str, Optional["MappingHintBundle"]] = {}


def _safe_token(value: str, fallback: str = "general") -> str:
    token = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    token = "".join(ch for ch in token if ch.isalnum() or ch == "_")
    return token or fallback


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model_dir_candidates(language: str, format_type: str) -> List[str]:
    lang = _safe_token(language, "korean")
    fmt = _safe_token(format_type or "cv", "cv")
    out: List[str] = []
    for key in (
        f"UTOA_MAPPING_HINT_MODEL_DIR_{lang.upper()}_{fmt.upper()}",
        f"UTOA_MAPPING_HINT_MODEL_DIR_{lang.upper()}",
        "UTOA_MAPPING_HINT_MODEL_DIR",
    ):
        path = str(os.environ.get(key, "") or "").strip()
        if path:
            out.append(path)

    model_root = str(os.environ.get("UTOA_MAPPING_HINT_MODEL_ROOT", "") or "").strip()
    if not model_root:
        model_root = os.path.join(_project_root(), "ML_models")
    version = str(os.environ.get("UTOA_MAPPING_HINT_MODEL_VERSION", "v1_mapping_hint") or "v1_mapping_hint").strip()
    out.append(os.path.join(model_root, lang, fmt, version))
    if fmt != "cv":
        out.append(os.path.join(model_root, lang, "cv", version))
    return out


def _canonicalize_feature_row(row: Dict[str, object], feature_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for name in feature_names:
        try:
            out.append(float(row.get(name, 0.0) or 0.0))
        except Exception:
            out.append(0.0)
    return out


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    x = float(max(-24.0, min(24.0, x)))
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class MappingHintBundle:
    model_dir: str
    map_booster: object
    offset_booster: Optional[object]
    cutoff_booster: Optional[object]
    feature_names: List[str]
    meta: Dict[str, object]


def load_mapping_hint_bundle(language: str, format_type: str) -> Optional[MappingHintBundle]:
    if lgb is None:
        return None
    lang = _safe_token(language, "korean")
    fmt = _safe_token(format_type or "cv", "cv")
    key = f"{lang}:{fmt}"
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    bundle: Optional[MappingHintBundle] = None
    for model_dir in _model_dir_candidates(lang, fmt):
        map_txt = os.path.join(model_dir, "model_map_ok.txt")
        if not os.path.isfile(map_txt):
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
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception:
                pass
        try:
            map_booster = lgb.Booster(model_file=map_txt)
            offset_txt = os.path.join(model_dir, "model_offset_hint.txt")
            cutoff_txt = os.path.join(model_dir, "model_cutoff_hint.txt")
            offset_booster = lgb.Booster(model_file=offset_txt) if os.path.isfile(offset_txt) else None
            cutoff_booster = lgb.Booster(model_file=cutoff_txt) if os.path.isfile(cutoff_txt) else None
            bundle = MappingHintBundle(
                model_dir=model_dir,
                map_booster=map_booster,
                offset_booster=offset_booster,
                cutoff_booster=cutoff_booster,
                feature_names=feature_names,
                meta=meta,
            )
            break
        except Exception as e:
            logger.warning("Failed to load mapping hint bundle (%s): %s", model_dir, e)
            continue

    _BUNDLE_CACHE[key] = bundle
    return bundle


def clear_mapping_hint_cache() -> None:
    _BUNDLE_CACHE.clear()


def _to_prob_list(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    as_prob = vmin >= 0.0 and vmax <= 1.0
    out = [(float(v) if as_prob else _sigmoid(float(v))) for v in values]
    return [max(0.0, min(1.0, float(v))) for v in out]


def _predict_flat(
    bundle: MappingHintBundle,
    feature_rows: List[List[Dict[str, object]]],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    flat: List[List[float]] = []
    for row in feature_rows:
        for feat in row or []:
            flat.append(_canonicalize_feature_row(feat or {}, bundle.feature_names))
    if not flat:
        return None, None, None
    try:
        map_raw = bundle.map_booster.predict(flat)
        map_prob = _to_prob_list([float(v) for v in map_raw])
    except Exception:
        return None, None, None

    off_pred: Optional[List[float]] = None
    cut_pred: Optional[List[float]] = None
    try:
        if bundle.offset_booster is not None:
            off_pred = [float(v) for v in bundle.offset_booster.predict(flat)]
    except Exception:
        off_pred = None
    try:
        if bundle.cutoff_booster is not None:
            cut_pred = [float(v) for v in bundle.cutoff_booster.predict(flat)]
    except Exception:
        cut_pred = None
    return map_prob, off_pred, cut_pred


def predict_mapping_hint_topk(
    *,
    language: str,
    format_type: str,
    score_rows: List[List[float]],
    feature_rows: List[List[Dict[str, object]]],
    top_k: int = 3,
) -> Dict[str, object]:
    meta: Dict[str, object] = {"applied": False, "reason": ""}
    if not score_rows or not feature_rows:
        meta["reason"] = "empty"
        return {"topk_rows": [], "top1_indices": [], "meta": meta}
    if len(score_rows) != len(feature_rows):
        meta["reason"] = "shape_mismatch"
        return {"topk_rows": [], "top1_indices": [], "meta": meta}

    bundle = load_mapping_hint_bundle(language, format_type)
    if bundle is None:
        meta["reason"] = "model_unavailable"
        return {"topk_rows": [], "top1_indices": [], "meta": meta}

    map_prob, off_pred, cut_pred = _predict_flat(bundle, feature_rows)
    if map_prob is None:
        meta["reason"] = "predict_failed"
        return {"topk_rows": [], "top1_indices": [], "meta": meta}

    weight = max(0.0, min(120.0, _safe_float(os.environ.get("UTOA_MAPPING_HINT_RUNTIME_WEIGHT", "34.0"), 34.0)))
    k_n = max(1, int(top_k))
    topk_rows: List[List[Dict[str, object]]] = []
    top1_indices: List[int] = []

    offset = 0
    for score_row, feat_row in zip(score_rows, feature_rows):
        width = min(len(score_row), len(feat_row))
        if width <= 0:
            topk_rows.append([])
            top1_indices.append(-1)
            continue
        cand_rows: List[Dict[str, object]] = []
        for cand_idx in range(width):
            base_score = float(score_row[cand_idx])
            p_map_ok = float(map_prob[offset + cand_idx])
            rank_score = base_score + (((p_map_ok - 0.5) * 2.0) * weight)
            m_offset_hint = float(off_pred[offset + cand_idx]) if off_pred is not None else 0.0
            m_cutoff_hint = float(cut_pred[offset + cand_idx]) if cut_pred is not None else 0.0
            cand_rows.append(
                {
                    "cand_idx": int(cand_idx),
                    "p_map_ok": max(0.0, min(1.0, p_map_ok)),
                    "m_offset_hint": m_offset_hint,
                    "m_cutoff_hint": m_cutoff_hint,
                    "base_score": base_score,
                    "score": float(rank_score),
                }
            )
        offset += width
        cand_rows = sorted(cand_rows, key=lambda r: (-float(r.get("score", 0.0)), int(r.get("cand_idx", 0))))
        trimmed: List[Dict[str, object]] = []
        for rank, cand in enumerate(cand_rows[:k_n], start=1):
            row = dict(cand)
            row["m_rank"] = int(rank)
            trimmed.append(row)
        topk_rows.append(trimmed)
        top1_indices.append(int(trimmed[0]["cand_idx"]) if trimmed else -1)

    meta["applied"] = True
    meta["reason"] = "ok"
    meta["model_dir"] = str(bundle.model_dir)
    meta["feature_names"] = list(bundle.feature_names)
    meta["weight"] = float(weight)
    meta["top_k"] = int(k_n)
    meta["rows"] = int(len(topk_rows))
    return {"topk_rows": topk_rows, "top1_indices": top1_indices, "meta": meta}


__all__ = [
    "MappingHintBundle",
    "clear_mapping_hint_cache",
    "load_mapping_hint_bundle",
    "predict_mapping_hint_topk",
]

