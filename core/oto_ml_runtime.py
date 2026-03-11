"""
Runtime abstraction for OTO ML correction backends.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    schema = _load_json(os.path.join(model_dir, "feature_schema.json"))
    if not schema:
        from core.oto_ml_features import get_feature_schema

        schema = get_feature_schema()
    backend = str(meta.get("backend", "")).strip().lower() or "lightgbm"
    coupled_device = str(os.environ.get("UTOA_ML_COUPLED_DEVICE", "auto") or "auto").strip()
    try:
        if backend == "lightgbm":
            from core.oto_ml_lightgbm import load_lightgbm_bundle

            payload = load_lightgbm_bundle(model_dir, meta=meta, schema=schema)
        elif backend == "coupled_nn_v1":
            from core.oto_ml_coupled import load_coupled_bundle

            payload = load_coupled_bundle(
                model_dir,
                meta=meta,
                schema=schema,
                device=coupled_device,
            )
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
    if bundle.backend == "lightgbm":
        from core.oto_ml_lightgbm import predict_lightgbm_deltas

        deltas = predict_lightgbm_deltas(bundle.payload, feature_row, meta=bundle.meta, schema=bundle.feature_schema)
    elif bundle.backend == "coupled_nn_v1":
        from core.oto_ml_coupled import predict_coupled_deltas

        deltas, confidence = predict_coupled_deltas(
            bundle.payload,
            feature_row,
            meta=bundle.meta,
            schema=bundle.feature_schema,
        )
    else:
        raise RuntimeError(f"Unsupported OTO ML backend: {bundle.backend}")
    return OtoDeltaResult(
        deltas=deltas,
        backend=bundle.backend,
        applied_model=bundle.model_dir,
        confidence=confidence,
    )
