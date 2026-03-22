from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

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
    from sklearn.model_selection import GroupShuffleSplit
except Exception as _sk_exc:  # pragma: no cover
    GroupShuffleSplit = None
    mean_absolute_error = None
    SKLEARN_IMPORT_ERROR = _sk_exc
else:
    SKLEARN_IMPORT_ERROR = None

from core.format_type_utils import normalize_format_type
from core.oto_ml.autofree.schema import (
    AUTOFREE_BACKEND,
    CATEGORICAL_FEATURES,
    CONFIDENCE_TARGET_NAME,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    TARGET_NAMES,
    TARGET_SCHEMA_VERSION,
    canonicalize_feature_row,
    get_feature_schema,
    get_target_schema,
    write_feature_schema,
    write_target_schema,
)

DEFAULT_LGB_PARAMS = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_data_in_leaf": 24,
    "verbosity": -1,
}

CONFIDENCE_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_data_in_leaf": 24,
    "verbosity": -1,
}

TARGET_MODEL_FILES = {
    "target_offset_ms": "model_target_offset.txt",
    "target_cons_ms": "model_target_cons.txt",
    "target_cutoff_abs_ms": "model_target_cutoff_abs.txt",
    "target_pre_ms": "model_target_pre.txt",
    "target_ovl_ms": "model_target_ovl.txt",
}
CONFIDENCE_MODEL_FILE = "model_confidence.txt"


def _require_training_stack() -> None:
    if lgb is None:
        raise RuntimeError(f"lightgbm is required: {LIGHTGBM_IMPORT_ERROR}")
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    if GroupShuffleSplit is None or mean_absolute_error is None:
        raise RuntimeError(f"scikit-learn is required: {SKLEARN_IMPORT_ERROR}")


def _prepare_frame(df):
    frame = df.copy()
    for col in FEATURE_NAMES:
        if col not in frame.columns:
            frame[col] = "" if col in CATEGORICAL_FEATURES else 0.0
    frame = frame[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        if col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].fillna("").astype("string").astype("category")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _split_train_valid(df, group_column: str):
    if len(df) < 8:
        raise RuntimeError("Not enough rows to train autofree bundle (need >= 8 rows).")
    if group_column in df.columns and df[group_column].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(splitter.split(df, groups=df[group_column]))
        return train_idx, valid_idx
    split_at = max(1, int(len(df) * 0.8))
    train_idx = list(range(split_at))
    valid_idx = list(range(split_at, len(df))) or list(range(max(0, split_at - 1), len(df)))
    return train_idx, valid_idx


def _ensure_schema_files(out_dir: str) -> None:
    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    write_target_schema(os.path.join(out_dir, "target_schema.json"))
    # explicit af-named files for downstream tooling
    write_feature_schema(os.path.join(out_dir, "feature_schema_af.json"))
    write_target_schema(os.path.join(out_dir, "target_schema_af.json"))


def train_autofree_bundle(
    *,
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    group_column: str = "voicebank_id",
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.isfile(dataset_csv):
        raise FileNotFoundError(dataset_csv)
    os.makedirs(out_dir, exist_ok=True)

    lang = str(language).strip().lower()
    fmt = normalize_format_type(lang, format_type) or "general"
    df = pd.read_csv(dataset_csv)
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == lang]
    if "format_type" in df.columns and fmt != "general":
        df = df[df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(lang, v)) == fmt]
    if len(df) < 8:
        raise RuntimeError("Filtered dataset is too small for autofree training.")

    feature_frame = _prepare_frame(df)
    train_idx, valid_idx = _split_train_valid(df, group_column)
    x_train = feature_frame.iloc[train_idx]
    x_valid = feature_frame.iloc[valid_idx]
    if "sample_weight" in df.columns:
        w_train = pd.to_numeric(df.iloc[train_idx]["sample_weight"], errors="coerce").fillna(1.0)
        w_valid = pd.to_numeric(df.iloc[valid_idx]["sample_weight"], errors="coerce").fillna(1.0)
    else:
        w_train = pd.Series([1.0] * len(train_idx))
        w_valid = pd.Series([1.0] * len(valid_idx))

    metrics: Dict[str, Dict[str, float]] = {}
    for target in TARGET_NAMES:
        if target not in df.columns:
            raise RuntimeError(f"missing target column: {target}")
        y_train = pd.to_numeric(df.iloc[train_idx][target], errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(df.iloc[valid_idx][target], errors="coerce").fillna(0.0)
        dtrain = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
        dvalid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
        booster = lgb.train(
            dict(DEFAULT_LGB_PARAMS),
            dtrain,
            num_boost_round=int(num_boost_round),
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
        )
        model_path = os.path.join(out_dir, TARGET_MODEL_FILES[target])
        booster.save_model(model_path)
        pred = booster.predict(x_valid)
        metrics[target] = {
            "baseline_mae": float(mean_absolute_error(y_valid, [0.0] * len(y_valid))),
            "model_mae": float(mean_absolute_error(y_valid, pred)),
        }

    conf_source = (
        pd.to_numeric(df.get(CONFIDENCE_TARGET_NAME), errors="coerce")
        if CONFIDENCE_TARGET_NAME in df.columns
        else pd.to_numeric(df.get("quality_score", 0.5), errors="coerce")
    )
    conf_source = conf_source.fillna(0.5).clip(lower=0.0, upper=1.0)
    conf_train = (conf_source.iloc[train_idx] >= 0.6).astype(int)
    conf_valid = (conf_source.iloc[valid_idx] >= 0.6).astype(int)
    dtrain_conf = lgb.Dataset(x_train, label=conf_train, weight=w_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    dvalid_conf = lgb.Dataset(x_valid, label=conf_valid, weight=w_valid, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    conf_booster = lgb.train(
        dict(CONFIDENCE_LGB_PARAMS),
        dtrain_conf,
        num_boost_round=max(100, int(num_boost_round // 2)),
        valid_sets=[dvalid_conf],
        callbacks=[lgb.early_stopping(max(20, int(early_stopping_rounds // 2)), verbose=False)],
    )
    conf_booster.save_model(os.path.join(out_dir, CONFIDENCE_MODEL_FILE))

    _ensure_schema_files(out_dir)
    meta = {
        "backend": AUTOFREE_BACKEND,
        "incompatible_with_legacy": True,
        "language": lang,
        "format_type": fmt,
        "model_version": "v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "targets": list(TARGET_NAMES),
        "confidence_target_name": CONFIDENCE_TARGET_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": metrics,
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, ensure_ascii=False, indent=2)
    return meta


def load_autofree_bundle(model_dir: str, *, meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if lgb is None or pd is None:
        return None
    if not model_dir or not os.path.isdir(model_dir):
        return None
    meta = dict(meta or {})
    if not meta:
        meta_path = os.path.join(model_dir, "model_meta.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    if str(meta.get("backend", "")).strip().lower() != AUTOFREE_BACKEND:
        return None
    if schema is None:
        schema_path = os.path.join(model_dir, "feature_schema.json")
        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f) or {}
        else:
            schema = get_feature_schema()
    models: Dict[str, Any] = {}
    try:
        for target in TARGET_NAMES:
            models[target] = lgb.Booster(model_file=os.path.join(model_dir, TARGET_MODEL_FILES[target]))
        conf_model_path = os.path.join(model_dir, CONFIDENCE_MODEL_FILE)
        conf_model = lgb.Booster(model_file=conf_model_path) if os.path.isfile(conf_model_path) else None
    except Exception:
        return None
    return {"models": models, "confidence_model": conf_model, "schema": schema, "meta": meta}


def predict_autofree_abs(
    bundle: Dict[str, Any],
    feature_row: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if pd is None:
        raise RuntimeError("pandas is required for autofree inference.")
    payload = dict(bundle or {})
    models = payload.get("models") or {}
    if not models:
        raise RuntimeError("autofree models are missing.")
    row = canonicalize_feature_row(feature_row, feature_names=FEATURE_NAMES)
    frame = _prepare_frame(pd.DataFrame([row]))
    out: Dict[str, float] = {}
    for target in TARGET_NAMES:
        model = models.get(target)
        if model is None:
            raise RuntimeError(f"autofree target model is missing: {target}")
        pred = model.predict(frame)
        out[target] = float(pred[0]) if len(pred) else 0.0
    conf_model = payload.get("confidence_model")
    if conf_model is not None:
        conf_pred = conf_model.predict(frame)
        confidence = float(conf_pred[0]) if len(conf_pred) else 0.5
    else:
        confidence = float(_clamp(_to_float(feature_row.get("source_confidence"), 0.5), 0.0, 1.0))
    out["confidence"] = float(_clamp(confidence, 0.0, 1.0))
    return out


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "CONFIDENCE_MODEL_FILE",
    "TARGET_MODEL_FILES",
    "load_autofree_bundle",
    "predict_autofree_abs",
    "train_autofree_bundle",
]
