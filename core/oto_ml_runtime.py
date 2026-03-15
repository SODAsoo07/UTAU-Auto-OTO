"""
Runtime abstraction for OTO ML correction backends.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OtoModelBundle:
    backend: str
    model_dir: str
    meta: Dict[str, Any]
    feature_schema: Dict[str, Any]
    payload: Any


@dataclass
class OtoDeltaResult:
    deltas: Dict[str, float]
    backend: str
    applied_model: str
    confidence: Optional[float] = None
    route: str = ""
    route_backend: str = ""


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_bundle_compat(bundle: OtoModelBundle, feature_schema: Dict[str, Any]) -> bool:
    if not bundle or not feature_schema:
        return False
    backend = str(bundle.meta.get("backend", "") or bundle.backend or "").strip().lower()
    if backend == "autofree_v1":
        meta_ver = str(bundle.meta.get("feature_schema_version", ""))
        schema_ver = str(feature_schema.get("feature_schema_version", ""))
    else:
        meta_ver = str(bundle.meta.get("feature_version", ""))
        schema_ver = str(feature_schema.get("feature_version", ""))
    if meta_ver and schema_ver and meta_ver != schema_ver:
        return False
    expected_names = list(feature_schema.get("feature_names") or [])
    bundle_names = list(bundle.feature_schema.get("feature_names") or [])
    return expected_names == bundle_names


def load_oto_model_bundle(model_dir: str) -> Optional[OtoModelBundle]:
    if not model_dir or not os.path.isdir(model_dir):
        return None
    meta = _load_json(os.path.join(model_dir, "model_meta.json"))
    if not meta:
        return None
    backend = str(meta.get("backend", "")).strip().lower() or "lightgbm"
    schema = _load_json(os.path.join(model_dir, "feature_schema.json"))
    if not schema:
        if backend == "autofree_v1":
            from core.oto_ml.autofree.schema import get_feature_schema
        else:
            from core.oto_ml_features import get_feature_schema
        schema = get_feature_schema()
    coupled_device = str(os.environ.get("UTOA_ML_COUPLED_DEVICE", "auto") or "auto").strip()
    try:
        if backend == "lightgbm":
            from core.oto_ml_lightgbm import load_lightgbm_bundle

            payload = load_lightgbm_bundle(model_dir, meta=meta, schema=schema)
        elif backend == "ensemble_v1":
            from core.oto_ml_ensemble import load_ensemble_bundle

            payload = load_ensemble_bundle(
                model_dir,
                meta=meta,
                schema=schema,
                device=coupled_device,
            )
        elif backend == "autofree_v1":
            from core.oto_ml_autofree import load_autofree_bundle

            payload = load_autofree_bundle(model_dir, meta=meta, schema=schema)
        else:
            logger.warning("Unsupported OTO ML backend: %s", backend)
            return None
    except Exception as e:
        logger.warning("Failed to load OTO ML bundle (%s): %s", model_dir, e)
        return None
    if payload is None:
        return None
    return OtoModelBundle(
        backend=backend,
        model_dir=os.path.abspath(model_dir),
        meta=meta,
        feature_schema=schema,
        payload=payload,
    )


def predict_oto_deltas(bundle: OtoModelBundle, feature_row: Dict[str, Any]) -> OtoDeltaResult:
    confidence = None
    route = str(bundle.backend)
    route_backend = str(bundle.backend)
    if bundle.backend == "lightgbm":
        from core.oto_ml_lightgbm import predict_lightgbm_deltas

        deltas = predict_lightgbm_deltas(bundle.payload, feature_row, meta=bundle.meta, schema=bundle.feature_schema)
    elif bundle.backend == "ensemble_v1":
        from core.oto_ml_ensemble import predict_ensemble_deltas

        result = predict_ensemble_deltas(
            bundle.payload,
            feature_row,
            meta=bundle.meta,
            schema=bundle.feature_schema,
        )
        deltas = dict(result.get("deltas") or {})
        confidence = result.get("confidence")
        route = str(result.get("route", route) or route)
        route_backend = str(result.get("route_backend", route_backend) or route_backend)
    elif bundle.backend == "autofree_v1":
        from core.oto_ml_autofree import predict_autofree_abs

        result = predict_autofree_abs(
            bundle.payload,
            feature_row,
            meta=bundle.meta,
            schema=bundle.feature_schema,
        )
        deltas = {
            "target_offset_ms": float(result.get("target_offset_ms", 0.0)),
            "target_cons_ms": float(result.get("target_cons_ms", 0.0)),
            "target_cutoff_abs_ms": float(result.get("target_cutoff_abs_ms", 0.0)),
            "target_pre_ms": float(result.get("target_pre_ms", 0.0)),
            "target_ovl_ms": float(result.get("target_ovl_ms", 0.0)),
        }
        confidence = float(result.get("confidence", 0.0))
        route = "autofree_v1"
        route_backend = "autofree_v1"
    else:
        raise RuntimeError(f"Unsupported OTO ML backend: {bundle.backend}")
    return OtoDeltaResult(
        deltas=deltas,
        backend=bundle.backend,
        applied_model=bundle.model_dir,
        confidence=confidence,
        route=route,
        route_backend=route_backend,
    )


def predict_oto_deltas_batch(bundle: OtoModelBundle, feature_rows: List[Dict[str, Any]]) -> List[OtoDeltaResult]:
    if not feature_rows:
        return []
    if bundle.backend == "ensemble_v1":
        from core.oto_ml_ensemble import predict_ensemble_deltas_batch

        results = predict_ensemble_deltas_batch(
            bundle.payload,
            feature_rows,
            meta=bundle.meta,
            schema=bundle.feature_schema,
        )
        out: List[OtoDeltaResult] = []
        for result in results:
            out.append(
                OtoDeltaResult(
                    deltas=dict(result.get("deltas") or {}),
                    backend=bundle.backend,
                    applied_model=bundle.model_dir,
                    confidence=result.get("confidence"),
                    route=str(result.get("route", bundle.backend) or bundle.backend),
                    route_backend=str(result.get("route_backend", bundle.backend) or bundle.backend),
                )
            )
        return out
    return [predict_oto_deltas(bundle, feature_row) for feature_row in feature_rows]
