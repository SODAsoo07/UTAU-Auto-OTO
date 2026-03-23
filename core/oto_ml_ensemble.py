"""
Ensemble helpers for OTO ML correction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import lightgbm as lgb
except Exception as _lgb_exc:  # pragma: no cover
    lgb = None
    LIGHTGBM_IMPORT_ERROR = _lgb_exc
else:
    LIGHTGBM_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as _pd_exc:  # pragma: no cover
    pd = None
    PANDAS_IMPORT_ERROR = _pd_exc
else:
    PANDAS_IMPORT_ERROR = None

try:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
except Exception as _sk_exc:  # pragma: no cover
    mean_absolute_error = None
    GroupKFold = None
    GroupShuffleSplit = None
    SKLEARN_IMPORT_ERROR = _sk_exc
else:
    SKLEARN_IMPORT_ERROR = None

from core.format_type_utils import normalize_format_type
from core.oto_ml.coupled.inference import (
    load_coupled_bundle,
    predict_coupled_deltas,
    predict_coupled_deltas_batch,
)
from core.oto_ml.coupled.training import _prepare_training_frame, train_coupled_bundle_rawmel
from core.oto_ml.features.mel_patches import MelPatchCacheIndex, make_mel_patch_key
from core.oto_ml_features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    TARGET_NAMES,
    canonicalize_feature_row,
    get_feature_schema,
    write_feature_schema,
)
from core.oto_ml_lightgbm import (
    _clip_target_series,
    _normalize_model_file_for_runtime,
    _prepare_frame,
    _split_train_valid,
    load_lightgbm_bundle,
    predict_lightgbm_deltas,
    train_lightgbm_bundle,
)

logger = logging.getLogger(__name__)

ENSEMBLE_BACKEND = "ensemble_v1"
ENSEMBLE_META_FEATURE_VERSION = "ensemble_v1_meta"
ENSEMBLE_META_CATEGORICAL_FEATURES = [
    "format_type",
    "alias_type",
    "alias_group",
    "coda_type",
    "vowel_class",
    "bridge_type",
    "prev_alias_type",
    "next_alias_type",
    "mapping_reason_code",
]
ENSEMBLE_META_BASE_FEATURES = [
    "format_type",
    "alias_type",
    "alias_group",
    "coda_type",
    "vowel_class",
    "bridge_type",
    "prev_alias_type",
    "next_alias_type",
    "mapping_reason_code",
    "row_ratio_in_wav",
    "mapping_confidence",
    "words_vs_alias_score_margin",
    "jump_blocked_flag",
    "used_exact_vowel_fix",
    "used_nuclei_fallback",
    "used_alias_occurrence_mapping",
    "blank_span_confidence",
    "db_silence_ratio",
    "mel_window_silence_ratio",
    "syllable_blank_confidence",
    "syllable_mel_voiced_conf",
    "syllable_mel_silence_conf",
    "syllable_mel_unvoiced_conf",
    "syllable_mel_breath_conf",
    "onset_patch_energy_mean",
    "onset_patch_voiced_ratio",
    "onset_patch_unvoiced_ratio",
    "tail_patch_energy_mean",
    "tail_patch_silence_ratio",
]
ENSEMBLE_META_FEATURE_NAMES = list(ENSEMBLE_META_BASE_FEATURES) + [
    "coupled_confidence",
    "gate_coupled_weight_mean",
]
for _target in TARGET_NAMES:
    _suffix = _target.replace("delta_", "")
    ENSEMBLE_META_FEATURE_NAMES.extend(
        [
            f"lightgbm_{_suffix}",
            f"coupled_{_suffix}",
            f"gated_{_suffix}",
            f"gap_{_suffix}",
            f"gap_abs_{_suffix}",
        ]
    )


def _require_training_stack() -> None:
    if lgb is None:
        raise RuntimeError(f"lightgbm is required: {LIGHTGBM_IMPORT_ERROR}")
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    if mean_absolute_error is None or GroupShuffleSplit is None:
        raise RuntimeError(f"scikit-learn is required: {SKLEARN_IMPORT_ERROR}")


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _meta_feature_schema() -> Dict[str, object]:
    return {
        "feature_version": ENSEMBLE_META_FEATURE_VERSION,
        "feature_names": list(ENSEMBLE_META_FEATURE_NAMES),
        "categorical_features": list(ENSEMBLE_META_CATEGORICAL_FEATURES),
    }


def _write_meta_feature_schema(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_meta_feature_schema(), f, ensure_ascii=False, indent=2)


def _bundle_dir(model_dir: str, name: str) -> str:
    return os.path.join(model_dir, name)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _subbundle_info(model_dir: str, backend: str, payload: Any, meta: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backend": str(backend or "").strip().lower(),
        "model_dir": os.path.abspath(model_dir),
        "payload": payload,
        "meta": meta or {},
        "schema": schema or {},
    }


def _ensure_mel_patch_keys(df, language: str):
    if "mel_patch_key" in df.columns and bool(df["mel_patch_key"].astype(str).str.strip().all()):
        return df
    out = df.copy()
    required = [
        "voicebank_id",
        "wav_norm",
        "alias_norm",
        "occurrence_index",
        "row_index_in_wav",
    ]
    missing = [name for name in required if name not in out.columns]
    if missing:
        raise RuntimeError("Missing mel patch key columns: " + ", ".join(missing))
    fmt_default = normalize_format_type(language, out["format_type"].iloc[0] if len(out) and "format_type" in out.columns else "") or "general"

    def _build(row):
        row_lang = str(row.get("language", language) or language).strip().lower()
        row_fmt = normalize_format_type(row_lang, row.get("format_type", fmt_default)) or fmt_default
        return make_mel_patch_key(
            language=row_lang,
            format_type=row_fmt,
            voicebank_id=str(row.get("voicebank_id", "") or ""),
            wav_norm=str(row.get("wav_norm", "") or ""),
            alias_norm=str(row.get("alias_norm", "") or ""),
            occurrence_index=int(float(row.get("occurrence_index", 0) or 0)),
            row_index_in_wav=int(float(row.get("row_index_in_wav", 0) or 0)),
        )

    out["mel_patch_key"] = out.apply(_build, axis=1)
    return out


def _predict_lightgbm_frame(bundle: Dict[str, Any], df):
    feature_names = list((bundle.get("schema") or {}).get("feature_names") or FEATURE_NAMES)
    categorical = list((bundle.get("meta") or {}).get("categorical_features") or CATEGORICAL_FEATURES)
    frame = df.copy()
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = "" if name in categorical else 0.0
    frame = _prepare_frame(frame, feature_names, categorical)
    out: Dict[str, List[float]] = {}
    for target, model in (bundle.get("payload") or {}).get("models", {}).items():
        out[target] = [float(v) for v in model.predict(frame)]
    return out


def _row_to_coupled_input(row: Dict[str, Any], cache_index: Optional[MelPatchCacheIndex]):
    out = dict(row)
    key = str(out.get("mel_patch_key", "") or "").strip()
    if cache_index is not None:
        if not key:
            raise RuntimeError("mel_patch_key is required for coupled rawmel ensemble")
        onset_patch, tail_patch = cache_index.get(key)
        out["mel_onset_patch"] = onset_patch
        out["mel_tail_patch"] = tail_patch
    return out


def _predict_coupled_rows(bundle: Dict[str, Any], df, cache_index: Optional[MelPatchCacheIndex]):
    out = {target: [] for target in TARGET_NAMES}
    confs: List[float] = []
    for _, row in df.iterrows():
        row_dict = _row_to_coupled_input(row.to_dict(), cache_index)
        deltas, confidence = predict_coupled_deltas(
            bundle.get("payload"),
            row_dict,
            meta=bundle.get("meta"),
            schema=bundle.get("schema"),
        )
        for target in TARGET_NAMES:
            out[target].append(float(deltas.get(target, 0.0)))
        confs.append(float(confidence or 0.0))
    return out, confs


def _gate_prediction_weights(feature_row: Dict[str, Any], coupled_confidence: float, min_coupled_confidence: float) -> Tuple[Dict[str, float], str]:
    conf = float(coupled_confidence or 0.0)
    min_conf = max(0.0, float(min_coupled_confidence or 0.0))
    blank_conf = max(
        _to_float(feature_row.get("blank_span_confidence"), 0.0),
        _to_float(feature_row.get("syllable_blank_confidence"), 0.0),
    )
    mapping_conf = _clip01(_to_float(feature_row.get("mapping_confidence"), 1.0))
    jump_blocked = int(_to_float(feature_row.get("jump_blocked_flag"), 0.0)) > 0
    alias_type = str(feature_row.get("alias_type", "") or "").strip().lower()

    if conf < max(min_conf - 0.06, 0.20):
        return {target: 0.0 for target in TARGET_NAMES}, "lightgbm_gate"
    if jump_blocked and blank_conf >= 0.55 and conf < max(min_conf + 0.12, 0.72):
        return {target: 0.0 for target in TARGET_NAMES}, "lightgbm_gate"
    if conf >= max(min_conf + 0.18, 0.82) and blank_conf < 0.35 and mapping_conf >= 0.70 and not jump_blocked:
        return {target: 1.0 for target in TARGET_NAMES}, "coupled_gate"

    width = max(0.18, 1.0 - min_conf)
    base_weight = _clip01((conf - min_conf + 0.12) / width)
    base_weight -= 0.50 * blank_conf
    if mapping_conf < 0.60:
        base_weight -= (0.60 - mapping_conf) * 0.35
    if jump_blocked:
        base_weight -= 0.16
    if alias_type in {"cv", "cv_head"} and blank_conf >= 0.50:
        base_weight -= 0.10
    base_weight = _clip01(base_weight)

    weights = {
        "delta_offset": _clip01(base_weight - (0.08 if blank_conf >= 0.45 else 0.0)),
        "delta_cutoff": _clip01(base_weight - (0.06 if blank_conf >= 0.45 else 0.0)),
        "delta_cons": _clip01(base_weight + 0.04),
        "delta_pre": _clip01(base_weight + 0.02),
        "delta_ovl": _clip01(base_weight),
    }
    if max(weights.values()) <= 0.01:
        return {target: 0.0 for target in TARGET_NAMES}, "lightgbm_gate"
    if min(weights.values()) >= 0.99:
        return {target: 1.0 for target in TARGET_NAMES}, "coupled_gate"
    return weights, "gated_blend"


def _blend_deltas(
    lightgbm_deltas: Dict[str, float],
    coupled_deltas: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target in TARGET_NAMES:
        cw = _clip01(weights.get(target, 0.0))
        lw = 1.0 - cw
        out[target] = (cw * float(coupled_deltas.get(target, 0.0))) + (lw * float(lightgbm_deltas.get(target, 0.0)))
    return out


def build_meta_feature_row(
    feature_row: Dict[str, Any],
    *,
    lightgbm_deltas: Dict[str, float],
    coupled_deltas: Dict[str, float],
    gated_deltas: Dict[str, float],
    coupled_confidence: float,
    gate_weights: Dict[str, float],
) -> Dict[str, Any]:
    canon = canonicalize_feature_row(feature_row, feature_names=list(FEATURE_NAMES))
    out: Dict[str, Any] = {}
    for name in ENSEMBLE_META_BASE_FEATURES:
        if name in ENSEMBLE_META_CATEGORICAL_FEATURES:
            out[name] = str(canon.get(name, "") or "")
        else:
            out[name] = _to_float(canon.get(name, 0.0), 0.0)
    out["coupled_confidence"] = float(coupled_confidence or 0.0)
    out["gate_coupled_weight_mean"] = float(sum(gate_weights.get(t, 0.0) for t in TARGET_NAMES) / max(1, len(TARGET_NAMES)))
    for target in TARGET_NAMES:
        suffix = target.replace("delta_", "")
        l_val = float(lightgbm_deltas.get(target, 0.0))
        c_val = float(coupled_deltas.get(target, 0.0))
        g_val = float(gated_deltas.get(target, 0.0))
        out[f"lightgbm_{suffix}"] = l_val
        out[f"coupled_{suffix}"] = c_val
        out[f"gated_{suffix}"] = g_val
        out[f"gap_{suffix}"] = c_val - l_val
        out[f"gap_abs_{suffix}"] = abs(c_val - l_val)
    return out


def predict_gated_ensemble_deltas(
    *,
    lightgbm_bundle: Dict[str, Any],
    coupled_bundle: Dict[str, Any],
    feature_row: Dict[str, Any],
    min_coupled_confidence: float = 0.0,
) -> Dict[str, Any]:
    lightgbm_deltas = predict_lightgbm_deltas(
        lightgbm_bundle.get("payload"),
        feature_row,
        meta=lightgbm_bundle.get("meta"),
        schema=lightgbm_bundle.get("schema"),
    )
    coupled_deltas, coupled_confidence = predict_coupled_deltas(
        coupled_bundle.get("payload"),
        feature_row,
        meta=coupled_bundle.get("meta"),
        schema=coupled_bundle.get("schema"),
    )
    weights, gate_name = _gate_prediction_weights(
        feature_row,
        float(coupled_confidence or 0.0),
        float(min_coupled_confidence or 0.0),
    )
    if gate_name == "coupled_gate":
        deltas = {target: float(coupled_deltas.get(target, 0.0)) for target in TARGET_NAMES}
        route_backend = str(coupled_bundle.get("backend", "") or "")
        route = route_backend or "coupled_gate"
        confidence = float(coupled_confidence or 0.0)
    elif gate_name == "lightgbm_gate":
        deltas = {target: float(lightgbm_deltas.get(target, 0.0)) for target in TARGET_NAMES}
        route_backend = "lightgbm"
        route = f"{str(coupled_bundle.get('backend', '') or 'coupled')}->lightgbm_gate"
        confidence = float(min(0.49, coupled_confidence or 0.0))
    else:
        deltas = _blend_deltas(lightgbm_deltas, coupled_deltas, weights)
        route_backend = ENSEMBLE_BACKEND
        route = f"{str(coupled_bundle.get('backend', '') or 'coupled')}->gated_blend"
        confidence = float(_clip01((coupled_confidence or 0.0) * (sum(weights.values()) / max(1, len(weights)))))
    return {
        "deltas": deltas,
        "confidence": confidence,
        "route": route,
        "route_backend": route_backend,
        "lightgbm_deltas": {target: float(lightgbm_deltas.get(target, 0.0)) for target in TARGET_NAMES},
        "coupled_deltas": {target: float(coupled_deltas.get(target, 0.0)) for target in TARGET_NAMES},
        "coupled_confidence": float(coupled_confidence or 0.0),
        "gate_weights": weights,
        "gate_name": gate_name,
    }


def predict_gated_ensemble_deltas_batch(
    *,
    lightgbm_bundle: Dict[str, Any],
    coupled_bundle: Dict[str, Any],
    feature_rows: List[Dict[str, Any]],
    min_coupled_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    if not feature_rows:
        return []
    if pd is None:
        return [
            predict_gated_ensemble_deltas(
                lightgbm_bundle=lightgbm_bundle,
                coupled_bundle=coupled_bundle,
                feature_row=row,
                min_coupled_confidence=min_coupled_confidence,
            )
            for row in feature_rows
        ]

    frame = pd.DataFrame(feature_rows)
    lightgbm_preds = _predict_lightgbm_frame(lightgbm_bundle, frame)
    coupled_pairs = predict_coupled_deltas_batch(
        coupled_bundle.get("payload"),
        feature_rows,
        meta=coupled_bundle.get("meta"),
        schema=coupled_bundle.get("schema"),
    )

    out: List[Dict[str, Any]] = []
    for row_idx, feature_row in enumerate(feature_rows):
        lightgbm_deltas = {
            target: float((lightgbm_preds.get(target) or [0.0])[row_idx])
            for target in TARGET_NAMES
        }
        coupled_deltas, coupled_confidence = coupled_pairs[row_idx] if row_idx < len(coupled_pairs) else ({}, 0.0)
        coupled_deltas = {target: float(coupled_deltas.get(target, 0.0)) for target in TARGET_NAMES}
        coupled_confidence = float(coupled_confidence or 0.0)
        weights, gate_name = _gate_prediction_weights(
            feature_row,
            coupled_confidence,
            float(min_coupled_confidence or 0.0),
        )
        if gate_name == "coupled_gate":
            deltas = dict(coupled_deltas)
            route_backend = str(coupled_bundle.get("backend", "") or "")
            route = route_backend or "coupled_gate"
            confidence = coupled_confidence
        elif gate_name == "lightgbm_gate":
            deltas = dict(lightgbm_deltas)
            route_backend = "lightgbm"
            route = f"{str(coupled_bundle.get('backend', '') or 'coupled')}->lightgbm_gate"
            confidence = float(min(0.49, coupled_confidence))
        else:
            deltas = _blend_deltas(lightgbm_deltas, coupled_deltas, weights)
            route_backend = ENSEMBLE_BACKEND
            route = f"{str(coupled_bundle.get('backend', '') or 'coupled')}->gated_blend"
            confidence = float(_clip01(coupled_confidence * (sum(weights.values()) / max(1, len(weights)))))
        out.append(
            {
                "deltas": deltas,
                "confidence": confidence,
                "route": route,
                "route_backend": route_backend,
                "lightgbm_deltas": dict(lightgbm_deltas),
                "coupled_deltas": dict(coupled_deltas),
                "coupled_confidence": coupled_confidence,
                "gate_weights": weights,
                "gate_name": gate_name,
            }
        )
    return out


def load_ensemble_bundle(model_dir: str, *, meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None, device: str = "auto"):
    meta = meta or _load_json(os.path.join(model_dir, "model_meta.json"))
    schema = schema or _load_json(os.path.join(model_dir, "feature_schema.json")) or get_feature_schema()
    lightgbm_dir = _bundle_dir(model_dir, "lightgbm")
    coupled_dir = _bundle_dir(model_dir, "coupled")
    meta_dir = _bundle_dir(model_dir, "meta")
    if not os.path.isdir(lightgbm_dir):
        raise FileNotFoundError(lightgbm_dir)
    if not os.path.isdir(coupled_dir):
        raise FileNotFoundError(coupled_dir)

    lightgbm_meta = _load_json(os.path.join(lightgbm_dir, "model_meta.json"))
    lightgbm_schema = _load_json(os.path.join(lightgbm_dir, "feature_schema.json")) or get_feature_schema()
    lightgbm_payload = load_lightgbm_bundle(lightgbm_dir, meta=lightgbm_meta, schema=lightgbm_schema)

    coupled_meta = _load_json(os.path.join(coupled_dir, "model_meta.json"))
    coupled_schema = _load_json(os.path.join(coupled_dir, "feature_schema.json")) or get_feature_schema()
    coupled_payload = load_coupled_bundle(coupled_dir, meta=coupled_meta, schema=coupled_schema, device=device)
    coupled_backend = str((coupled_meta or {}).get("backend", "") or "")

    meta_bundle = None
    meta_meta = {}
    meta_schema = {}
    if os.path.isdir(meta_dir) and os.path.isfile(os.path.join(meta_dir, "model_meta.json")):
        meta_meta = _load_json(os.path.join(meta_dir, "model_meta.json"))
        meta_schema = _load_json(os.path.join(meta_dir, "feature_schema.json")) or _meta_feature_schema()
        meta_bundle = load_lightgbm_bundle(meta_dir, meta=meta_meta, schema=meta_schema)

    return {
        "backend": ENSEMBLE_BACKEND,
        "lightgbm": _subbundle_info(lightgbm_dir, "lightgbm", lightgbm_payload, lightgbm_meta, lightgbm_schema),
        "coupled": _subbundle_info(coupled_dir, coupled_backend, coupled_payload, coupled_meta, coupled_schema),
        "meta": _subbundle_info(meta_dir, "lightgbm", meta_bundle, meta_meta, meta_schema) if meta_bundle is not None else None,
        "gating": dict((meta or {}).get("gating") or {}),
        "coupled_backend": coupled_backend,
        "coupled_rawmel_enabled": bool((coupled_payload or {}).get("rawmel_enabled", False)),
        "coupled_patch_spec": dict((coupled_payload or {}).get("mel_patch_spec") or {}),
        "schema": schema,
    }


def predict_ensemble_deltas(
    payload: Dict[str, Any],
    feature_row: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root_meta = meta or {}
    gating_meta = dict((payload or {}).get("gating") or {})
    min_conf = float(gating_meta.get("min_coupled_confidence", root_meta.get("min_coupled_confidence", 0.0)) or 0.0)
    gated = predict_gated_ensemble_deltas(
        lightgbm_bundle=(payload or {}).get("lightgbm") or {},
        coupled_bundle=(payload or {}).get("coupled") or {},
        feature_row=feature_row,
        min_coupled_confidence=min_conf,
    )

    meta_bundle = (payload or {}).get("meta")
    if not meta_bundle or not meta_bundle.get("payload"):
        gated["route"] = f"{ENSEMBLE_BACKEND}:{gated.get('route', 'gated')}"
        gated["route_backend"] = gated.get("route_backend") or ENSEMBLE_BACKEND
        return gated

    meta_row = build_meta_feature_row(
        feature_row,
        lightgbm_deltas=gated.get("lightgbm_deltas") or {},
        coupled_deltas=gated.get("coupled_deltas") or {},
        gated_deltas=gated.get("deltas") or {},
        coupled_confidence=float(gated.get("coupled_confidence", 0.0) or 0.0),
        gate_weights=gated.get("gate_weights") or {},
    )
    feature_names = list(((meta_bundle.get("schema") or {}).get("feature_names") or ENSEMBLE_META_FEATURE_NAMES))
    categorical = list(((meta_bundle.get("meta") or {}).get("categorical_features") or ENSEMBLE_META_CATEGORICAL_FEATURES))
    frame = pd.DataFrame([meta_row])
    frame = _prepare_frame(frame, feature_names, categorical)
    meta_deltas: Dict[str, float] = {}
    for target, model in ((meta_bundle.get("payload") or {}).get("models") or {}).items():
        meta_deltas[target] = float(model.predict(frame)[0])
    return {
        "deltas": meta_deltas,
        "confidence": float(max(gated.get("confidence", 0.0) or 0.0, gated.get("coupled_confidence", 0.0) or 0.0)),
        "route": f"{ENSEMBLE_BACKEND}:meta",
        "route_backend": ENSEMBLE_BACKEND,
        "lightgbm_deltas": gated.get("lightgbm_deltas") or {},
        "coupled_deltas": gated.get("coupled_deltas") or {},
        "coupled_confidence": float(gated.get("coupled_confidence", 0.0) or 0.0),
        "gate_weights": gated.get("gate_weights") or {},
        "gated_deltas": gated.get("deltas") or {},
    }


def predict_ensemble_deltas_batch(
    payload: Dict[str, Any],
    feature_rows: List[Dict[str, Any]],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not feature_rows:
        return []

    root_meta = meta or {}
    gating_meta = dict((payload or {}).get("gating") or {})
    min_conf = float(gating_meta.get("min_coupled_confidence", root_meta.get("min_coupled_confidence", 0.0)) or 0.0)
    gated_rows = predict_gated_ensemble_deltas_batch(
        lightgbm_bundle=(payload or {}).get("lightgbm") or {},
        coupled_bundle=(payload or {}).get("coupled") or {},
        feature_rows=feature_rows,
        min_coupled_confidence=min_conf,
    )

    meta_bundle = (payload or {}).get("meta")
    if not meta_bundle or not meta_bundle.get("payload"):
        out: List[Dict[str, Any]] = []
        for gated in gated_rows:
            row = dict(gated)
            row["route"] = f"{ENSEMBLE_BACKEND}:{gated.get('route', 'gated')}"
            row["route_backend"] = gated.get("route_backend") or ENSEMBLE_BACKEND
            out.append(row)
        return out

    if pd is None:
        return [
            predict_ensemble_deltas(payload, row, meta=meta, schema=schema)
            for row in feature_rows
        ]

    meta_feature_rows: List[Dict[str, Any]] = []
    for row, gated in zip(feature_rows, gated_rows):
        meta_feature_rows.append(
            build_meta_feature_row(
                row,
                lightgbm_deltas=gated.get("lightgbm_deltas") or {},
                coupled_deltas=gated.get("coupled_deltas") or {},
                gated_deltas=gated.get("deltas") or {},
                coupled_confidence=float(gated.get("coupled_confidence", 0.0) or 0.0),
                gate_weights=gated.get("gate_weights") or {},
            )
        )
    feature_names = list(((meta_bundle.get("schema") or {}).get("feature_names") or ENSEMBLE_META_FEATURE_NAMES))
    categorical = list(((meta_bundle.get("meta") or {}).get("categorical_features") or ENSEMBLE_META_CATEGORICAL_FEATURES))
    frame = pd.DataFrame(meta_feature_rows)
    frame = _prepare_frame(frame, feature_names, categorical)

    meta_preds: Dict[str, List[float]] = {}
    for target, model in ((meta_bundle.get("payload") or {}).get("models") or {}).items():
        meta_preds[target] = [float(v) for v in model.predict(frame)]

    out: List[Dict[str, Any]] = []
    for row_idx, gated in enumerate(gated_rows):
        meta_deltas = {target: float((meta_preds.get(target) or [0.0])[row_idx]) for target in TARGET_NAMES}
        out.append(
            {
                "deltas": meta_deltas,
                "confidence": float(max(gated.get("confidence", 0.0) or 0.0, gated.get("coupled_confidence", 0.0) or 0.0)),
                "route": f"{ENSEMBLE_BACKEND}:meta",
                "route_backend": ENSEMBLE_BACKEND,
                "lightgbm_deltas": gated.get("lightgbm_deltas") or {},
                "coupled_deltas": gated.get("coupled_deltas") or {},
                "coupled_confidence": float(gated.get("coupled_confidence", 0.0) or 0.0),
                "gate_weights": gated.get("gate_weights") or {},
                "gated_deltas": gated.get("deltas") or {},
            }
        )
    return out


def _build_group_folds(df, group_column: str, num_folds: int):
    n_rows = int(len(df))
    if n_rows < 8:
        raise RuntimeError("Not enough rows to train ensemble bundle.")
    if GroupKFold is None:
        train_idx, valid_idx = _split_train_valid(df, group_column)
        return [(list(train_idx), list(valid_idx))]
    group_count = int(df[group_column].nunique()) if group_column in df.columns else 0
    if group_count >= 3:
        fold_count = max(2, min(int(num_folds), group_count))
        splitter = GroupKFold(n_splits=fold_count)
        groups = df[group_column]
        return [(train.tolist(), valid.tolist()) for train, valid in splitter.split(df, groups=groups)]
    fold_count = max(2, min(int(num_folds), n_rows))
    indices = list(range(n_rows))
    out = []
    fold_size = max(1, int(round(float(n_rows) / float(fold_count))))
    for fold_idx in range(fold_count):
        st = int(fold_idx * fold_size)
        ed = n_rows if fold_idx == fold_count - 1 else min(n_rows, int((fold_idx + 1) * fold_size))
        valid_idx = indices[st:ed]
        train_idx = indices[:st] + indices[ed:]
        if valid_idx and train_idx:
            out.append((train_idx, valid_idx))
    return out


def _train_meta_bundle(meta_df, out_dir: str, language: str, format_type: str, group_column: str) -> Dict[str, Any]:
    _require_training_stack()
    os.makedirs(out_dir, exist_ok=True)
    feature_names = list(ENSEMBLE_META_FEATURE_NAMES)
    categorical = list(ENSEMBLE_META_CATEGORICAL_FEATURES)
    frame = _prepare_frame(meta_df.copy(), feature_names, categorical)
    train_idx, valid_idx = _split_train_valid(meta_df, group_column)
    X_train = frame.iloc[train_idx]
    X_valid = frame.iloc[valid_idx]
    if "_meta_sample_weight" in meta_df.columns:
        w_train = pd.to_numeric(meta_df.iloc[train_idx]["_meta_sample_weight"], errors="coerce").fillna(1.0)
        w_valid = pd.to_numeric(meta_df.iloc[valid_idx]["_meta_sample_weight"], errors="coerce").fillna(1.0)
    else:
        w_train = pd.Series([1.0] * len(train_idx), index=X_train.index, dtype=float)
        w_valid = pd.Series([1.0] * len(valid_idx), index=X_valid.index, dtype=float)

    params = {
        "objective": "regression_l1",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    metrics: Dict[str, Any] = {}
    best_rounds: List[int] = []
    for target in TARGET_NAMES:
        y_train = pd.to_numeric(meta_df.iloc[train_idx][target], errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(meta_df.iloc[valid_idx][target], errors="coerce").fillna(0.0)
        y_train = _clip_target_series(language, target, y_train)
        y_valid = _clip_target_series(language, target, y_valid)
        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=categorical, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, categorical_feature=categorical, free_raw_data=False)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred_meta = booster.predict(X_valid)
        suffix = target.replace("delta_", "")
        gated_truth = pd.to_numeric(meta_df.iloc[valid_idx][f"gated_{suffix}"], errors="coerce").fillna(0.0)
        lgb_truth = pd.to_numeric(meta_df.iloc[valid_idx][f"lightgbm_{suffix}"], errors="coerce").fillna(0.0)
        coupled_truth = pd.to_numeric(meta_df.iloc[valid_idx][f"coupled_{suffix}"], errors="coerce").fillna(0.0)
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(y_valid, [0.0] * len(y_valid))),
            "lightgbm_mae": float(mean_absolute_error(y_valid, lgb_truth)),
            "coupled_mae": float(mean_absolute_error(y_valid, coupled_truth)),
            "gated_mae": float(mean_absolute_error(y_valid, gated_truth)),
            "meta_mae": float(mean_absolute_error(y_valid, pred_meta)),
        }
        best_rounds.append(int(getattr(booster, "best_iteration", 0) or 0))

    valid_rounds = [round_i for round_i in best_rounds if round_i > 0]
    final_rounds = max(30, int(sum(valid_rounds) / max(1, len(valid_rounds))))
    full_frame = _prepare_frame(meta_df.copy(), feature_names, categorical)
    for target in TARGET_NAMES:
        y_full = pd.to_numeric(meta_df[target], errors="coerce").fillna(0.0)
        y_full = _clip_target_series(language, target, y_full)
        w_full = pd.to_numeric(meta_df["_meta_sample_weight"], errors="coerce").fillna(1.0) if "_meta_sample_weight" in meta_df.columns else None
        dtrain = lgb.Dataset(
            full_frame,
            label=y_full,
            weight=w_full,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        booster = lgb.train(params, dtrain, num_boost_round=final_rounds)
        booster.save_model(os.path.join(out_dir, f"model_{target.replace('delta_', '')}.txt"))

    _write_meta_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    meta = {
        "backend": "lightgbm",
        "language": str(language or "").strip().lower(),
        "format_type": normalize_format_type(language, format_type) or "general",
        "model_version": "v1",
        "feature_version": ENSEMBLE_META_FEATURE_VERSION,
        "feature_names": feature_names,
        "categorical_features": categorical,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(meta_df)),
        "voicebank_count": int(meta_df[group_column].nunique()) if group_column in meta_df.columns else 1,
        "holdout_metrics": metrics,
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, ensure_ascii=False, indent=2)
    return meta


def train_ensemble_bundle(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    *,
    rawmel_cache_dir: str,
    group_column: str = "voicebank_id",
    alias_types: Optional[List[str]] = None,
    min_mapping_confidence: float = 0.0,
    num_folds: int = 5,
    lightgbm_num_boost_round: int = 500,
    lightgbm_early_stopping_rounds: int = 50,
    coupled_epochs: int = 70,
    coupled_batch_size: int = 192,
    coupled_learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)
    if not rawmel_cache_dir or not os.path.isdir(rawmel_cache_dir):
        raise FileNotFoundError(rawmel_cache_dir)

    df = pd.read_csv(dataset_csv, low_memory=False)
    language = str(language or "").strip().lower()
    format_type = normalize_format_type(language, format_type) or "general"
    df = _prepare_training_frame(
        df,
        language=language,
        format_type=format_type,
        alias_types=alias_types,
        min_mapping_confidence=min_mapping_confidence,
    )
    df = df.reset_index(drop=True)
    df = _ensure_mel_patch_keys(df, language)
    if len(df) < 24:
        raise RuntimeError("Ensemble dataset is too small (need >= 24 rows).")

    folds = _build_group_folds(df, group_column, num_folds=max(2, int(num_folds)))
    cache_index = MelPatchCacheIndex.load(rawmel_cache_dir)
    oof_rows: List[Dict[str, Any]] = []
    print(
        f"[ENSEMBLE] start format={format_type} rows={int(len(df))} folds={int(len(folds))} "
        f"group={group_column} rawmel_cache={rawmel_cache_dir}"
    )

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        valid_df = df.iloc[valid_idx].copy().reset_index(drop=True)
        if len(train_df) < 16 or len(valid_df) <= 0:
            continue
        print(
            f"[ENSEMBLE] fold {fold_idx + 1}/{len(folds)} train_rows={int(len(train_df))} "
            f"valid_rows={int(len(valid_df))}"
        )
        with tempfile.TemporaryDirectory(prefix=f"utoa_ens_fold_{fold_idx}_") as td:
            train_csv = os.path.join(td, "train.csv")
            train_df.to_csv(train_csv, index=False, encoding="utf-8")
            lightgbm_dir = os.path.join(td, "lightgbm")
            coupled_dir = os.path.join(td, "coupled")
            print(f"[ENSEMBLE] fold {fold_idx + 1}/{len(folds)} train lightgbm")
            train_lightgbm_bundle(
                language=language,
                format_type=format_type,
                dataset_csv=train_csv,
                out_dir=lightgbm_dir,
                group_column=group_column,
                num_boost_round=int(lightgbm_num_boost_round),
                early_stopping_rounds=int(lightgbm_early_stopping_rounds),
                alias_types=alias_types,
                min_mapping_confidence=float(min_mapping_confidence),
            )
            print(f"[ENSEMBLE] fold {fold_idx + 1}/{len(folds)} train coupled_rawmel")
            train_coupled_bundle_rawmel(
                language=language,
                format_type=format_type,
                dataset_csv=train_csv,
                out_dir=coupled_dir,
                rawmel_cache_dir=rawmel_cache_dir,
                group_column=group_column,
                alias_types=alias_types,
                min_mapping_confidence=float(min_mapping_confidence),
                epochs=int(coupled_epochs),
                batch_size=int(coupled_batch_size),
                learning_rate=float(coupled_learning_rate),
            )
            lgb_meta = _load_json(os.path.join(lightgbm_dir, "model_meta.json"))
            lgb_schema = _load_json(os.path.join(lightgbm_dir, "feature_schema.json"))
            lgb_payload = load_lightgbm_bundle(lightgbm_dir, meta=lgb_meta, schema=lgb_schema)
            lgb_bundle = _subbundle_info(lightgbm_dir, "lightgbm", lgb_payload, lgb_meta, lgb_schema)

            coupled_meta = _load_json(os.path.join(coupled_dir, "model_meta.json"))
            coupled_schema = _load_json(os.path.join(coupled_dir, "feature_schema.json"))
            coupled_payload = load_coupled_bundle(coupled_dir, meta=coupled_meta, schema=coupled_schema, device="auto")
            coupled_bundle = _subbundle_info(coupled_dir, str(coupled_meta.get("backend", "") or ""), coupled_payload, coupled_meta, coupled_schema)

            lgb_preds = _predict_lightgbm_frame(lgb_bundle, valid_df)
            coupled_preds, coupled_confs = _predict_coupled_rows(coupled_bundle, valid_df, cache_index)

            for row_i, (_, row) in enumerate(valid_df.iterrows()):
                feature_row = row.to_dict()
                lgb_deltas = {target: float(lgb_preds.get(target, [0.0])[row_i]) for target in TARGET_NAMES}
                coupled_deltas = {target: float(coupled_preds.get(target, [0.0])[row_i]) for target in TARGET_NAMES}
                weights, _gate_name = _gate_prediction_weights(
                    feature_row,
                    float(coupled_confs[row_i] if row_i < len(coupled_confs) else 0.0),
                    0.0,
                )
                gated_deltas = _blend_deltas(lgb_deltas, coupled_deltas, weights)
                meta_row = build_meta_feature_row(
                    feature_row,
                    lightgbm_deltas=lgb_deltas,
                    coupled_deltas=coupled_deltas,
                    gated_deltas=gated_deltas,
                    coupled_confidence=float(coupled_confs[row_i] if row_i < len(coupled_confs) else 0.0),
                    gate_weights=weights,
                )
                for target in TARGET_NAMES:
                    meta_row[target] = _to_float(feature_row.get(target), 0.0)
                meta_row[group_column] = feature_row.get(group_column, "")
                meta_row["_meta_sample_weight"] = _to_float(feature_row.get("sample_weight"), 1.0)
                oof_rows.append(meta_row)
        print(f"[ENSEMBLE] fold {fold_idx + 1}/{len(folds)} done oof_rows={int(len(oof_rows))}")

    if len(oof_rows) < 24:
        raise RuntimeError("OOF ensemble training rows are too small.")

    meta_df = pd.DataFrame(oof_rows)
    root_out = os.path.abspath(out_dir)
    lightgbm_out = _bundle_dir(root_out, "lightgbm")
    coupled_out = _bundle_dir(root_out, "coupled")
    meta_out = _bundle_dir(root_out, "meta")
    os.makedirs(root_out, exist_ok=True)

    print("[ENSEMBLE] final train lightgbm")
    train_lightgbm_bundle(
        language=language,
        format_type=format_type,
        dataset_csv=dataset_csv,
        out_dir=lightgbm_out,
        group_column=group_column,
        num_boost_round=int(lightgbm_num_boost_round),
        early_stopping_rounds=int(lightgbm_early_stopping_rounds),
        alias_types=alias_types,
        min_mapping_confidence=float(min_mapping_confidence),
    )
    print("[ENSEMBLE] final train coupled_rawmel")
    train_coupled_bundle_rawmel(
        language=language,
        format_type=format_type,
        dataset_csv=dataset_csv,
        out_dir=coupled_out,
        rawmel_cache_dir=rawmel_cache_dir,
        group_column=group_column,
        alias_types=alias_types,
        min_mapping_confidence=float(min_mapping_confidence),
        epochs=int(coupled_epochs),
        batch_size=int(coupled_batch_size),
        learning_rate=float(coupled_learning_rate),
    )
    print("[ENSEMBLE] final train meta")
    meta_bundle_meta = _train_meta_bundle(meta_df, meta_out, language, format_type, group_column)

    write_feature_schema(os.path.join(root_out, "feature_schema.json"))
    lightgbm_eval = _load_json(os.path.join(lightgbm_out, "eval_summary.json"))
    coupled_eval = _load_json(os.path.join(coupled_out, "eval_summary.json"))
    root_meta = {
        "backend": ENSEMBLE_BACKEND,
        "language": language,
        "format_type": format_type,
        "model_version": "v1",
        "feature_version": get_feature_schema().get("feature_version", ""),
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "meta_enabled": True,
        "submodels": {
            "lightgbm": "lightgbm",
            "coupled": "coupled",
            "meta": "meta",
        },
        "coupled_backend": "coupled_nn_v2_rawmel",
        "gating": {
            "min_coupled_confidence": 0.0,
            "mode": "confidence_blank_mapping_gate",
        },
        "oof_folds": int(len(folds)),
        "holdout_metrics": {
            "lightgbm": dict((lightgbm_eval or {}).get("metrics") or {}),
            "coupled": dict((coupled_eval or {}).get("metrics") or {}),
            "meta": dict((meta_bundle_meta or {}).get("holdout_metrics") or {}),
        },
    }
    with open(os.path.join(root_out, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(root_meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(root_out, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "oof_rows": int(len(meta_df)),
                "lightgbm": lightgbm_eval,
                "coupled": coupled_eval,
                "meta": {"metrics": dict((meta_bundle_meta or {}).get("holdout_metrics") or {})},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[ENSEMBLE] done out_dir={root_out}")
    return root_meta
