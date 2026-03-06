"""
LightGBM backend for OTO ML correction.
"""

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
    mean_absolute_error = None
    GroupShuffleSplit = None
    SKLEARN_IMPORT_ERROR = _sk_exc
else:
    SKLEARN_IMPORT_ERROR = None

from core.oto_ml_features import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET_NAMES, canonicalize_feature_row, get_delta_clip_limits, get_feature_schema, write_feature_schema

DEFAULT_LGB_PARAMS = {
    "objective": "regression_l1",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_data_in_leaf": 40,
    "verbosity": -1,
}


def _require_training_stack():
    if lgb is None:
        raise RuntimeError(f"lightgbm is required: {LIGHTGBM_IMPORT_ERROR}")
    if pd is None:
        raise RuntimeError(f"pandas is required: {PANDAS_IMPORT_ERROR}")
    if GroupShuffleSplit is None or mean_absolute_error is None:
        raise RuntimeError(f"scikit-learn is required: {SKLEARN_IMPORT_ERROR}")


def _prepare_frame(df, feature_names, categorical_features):
    frame = df.copy()
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = "" if col in categorical_features else 0.0
    frame = frame[feature_names].copy()
    for col in feature_names:
        if col in categorical_features:
            frame[col] = frame[col].fillna("").astype("string").astype("category")
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _split_train_valid(df, group_column: str):
    if len(df) < 8:
        raise RuntimeError("Not enough rows to train OTO ML bundle (need >= 8 rows).")
    if group_column in df.columns and df[group_column].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(splitter.split(df, groups=df[group_column]))
    else:
        split_at = max(1, int(len(df) * 0.8))
        train_idx = list(range(split_at))
        valid_idx = list(range(split_at, len(df))) or list(range(max(0, split_at - 1), len(df)))
    return train_idx, valid_idx


def _clip_target_series(language: str, target: str, series):
    clip = get_delta_clip_limits(language).get(target)
    if not clip:
        return series
    lo, hi = float(clip[0]), float(clip[1])
    return series.clip(lower=lo, upper=hi)


def train_lightgbm_bundle(
    language: str,
    format_type: str,
    dataset_csv: str,
    out_dir: str,
    group_column: str = "voicebank_id",
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    alias_types: Optional[list[str]] = None,
    alias_groups: Optional[list[str]] = None,
    require_train_keep: bool = False,
    min_mapping_confidence: float = 0.0,
    exclude_nuclei_fallback: bool = False,
    use_pseudo_labels: bool = True,
    pseudo_weight_high: float = 0.7,
    pseudo_weight_mid: float = 0.4,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    df = pd.read_csv(dataset_csv)
    language = str(language).strip().lower()
    format_type = str(format_type).strip().lower()
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == language]
    if format_type and format_type != "general" and "format_type" in df.columns:
        df = df[df["format_type"].astype(str).str.lower() == format_type]
    if alias_types and "alias_type" in df.columns:
        alias_types = [str(v).strip().lower() for v in alias_types if str(v).strip()]
        if alias_types:
            df = df[df["alias_type"].astype(str).str.lower().isin(alias_types)]
    if alias_groups and "alias_group" in df.columns:
        alias_groups = [str(v).strip().lower() for v in alias_groups if str(v).strip()]
        if alias_groups:
            df = df[df["alias_group"].astype(str).str.lower().isin(alias_groups)]
    if require_train_keep and "train_keep_default" in df.columns:
        df = df[pd.to_numeric(df["train_keep_default"], errors="coerce").fillna(0).astype(int) > 0]
    if float(min_mapping_confidence) > 0.0 and "mapping_confidence" in df.columns:
        df = df[pd.to_numeric(df["mapping_confidence"], errors="coerce").fillna(0.0) >= float(min_mapping_confidence)]
    if exclude_nuclei_fallback and "used_nuclei_fallback" in df.columns:
        df = df[pd.to_numeric(df["used_nuclei_fallback"], errors="coerce").fillna(0).astype(int) <= 0]

    # Pseudo label handling: keep compatibility when columns are absent.
    if "label_source" in df.columns:
        label_series = df["label_source"].astype(str).str.strip().str.lower()
    else:
        label_series = pd.Series(["manual"] * len(df), index=df.index, dtype="string")

    if "sample_weight" in df.columns:
        sample_weight = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float)
    else:
        sample_weight = pd.Series([1.0] * len(df), index=df.index, dtype=float)

    if use_pseudo_labels:
        sample_weight.loc[label_series == "pseudo_high"] = float(pseudo_weight_high)
        sample_weight.loc[label_series == "pseudo_mid"] = float(pseudo_weight_mid)
        sample_weight.loc[label_series == "pseudo_low"] = 0.0
    else:
        sample_weight.loc[label_series.str.startswith("pseudo")] = 0.0

    df = df.copy()
    df["_train_sample_weight"] = sample_weight
    df = df[pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0) > 0.0]

    if len(df) < 8:
        raise RuntimeError("Filtered dataset is too small for training.")

    feature_schema = get_feature_schema()
    feature_names = list(feature_schema["feature_names"])
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    frame = _prepare_frame(df, feature_names, categorical_features)
    train_idx, valid_idx = _split_train_valid(df, group_column)
    X_train = frame.iloc[train_idx]
    X_valid = frame.iloc[valid_idx]
    w_train = pd.to_numeric(df.iloc[train_idx]["_train_sample_weight"], errors="coerce").fillna(1.0)
    w_valid = pd.to_numeric(df.iloc[valid_idx]["_train_sample_weight"], errors="coerce").fillna(1.0)

    out_metrics = {}
    os.makedirs(out_dir, exist_ok=True)
    targets = {}
    for target in TARGET_NAMES:
        y_train = pd.to_numeric(df.iloc[train_idx][target], errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(df.iloc[valid_idx][target], errors="coerce").fillna(0.0)
        y_train = _clip_target_series(language, target, y_train)
        y_valid = _clip_target_series(language, target, y_valid)
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            weight=w_valid,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        booster = lgb.train(
            dict(DEFAULT_LGB_PARAMS),
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        model_path = os.path.join(out_dir, f"model_{target.replace('delta_', '')}.txt")
        booster.save_model(model_path)
        pred = booster.predict(X_valid)
        out_metrics[target] = {
            "baseline_mae": float(mean_absolute_error(y_valid, [0.0] * len(y_valid))),
            "model_mae": float(mean_absolute_error(y_valid, pred)),
        }
        targets[target] = model_path

    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    meta = {
        "backend": "lightgbm",
        "language": language,
        "format_type": format_type,
        "model_version": "v1",
        "feature_version": feature_schema["feature_version"],
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "targets": list(TARGET_NAMES),
        "delta_clip_limits": get_delta_clip_limits(language),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "holdout_metrics": out_metrics,
        "filters": {
            "alias_types": list(alias_types or []),
            "alias_groups": list(alias_groups or []),
            "require_train_keep": bool(require_train_keep),
            "min_mapping_confidence": float(min_mapping_confidence),
            "exclude_nuclei_fallback": bool(exclude_nuclei_fallback),
            "use_pseudo_labels": bool(use_pseudo_labels),
            "pseudo_weight_high": float(pseudo_weight_high),
            "pseudo_weight_mid": float(pseudo_weight_mid),
        },
        "weight_summary": {
            "min": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).min()) if len(df) else 0.0,
            "max": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).max()) if len(df) else 0.0,
            "mean": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).mean()) if len(df) else 0.0,
            "pseudo_rows": int(label_series.loc[df.index].str.startswith("pseudo").sum()) if len(df) else 0,
        },
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": targets, "metrics": out_metrics}, f, ensure_ascii=False, indent=2)
    return meta


def load_lightgbm_bundle(model_dir: str, meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None):
    if lgb is None:
        raise RuntimeError(f"lightgbm is required for runtime inference: {LIGHTGBM_IMPORT_ERROR}")
    models = {}
    for target in TARGET_NAMES:
        model_name = f"model_{target.replace('delta_', '')}.txt"
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        models[target] = lgb.Booster(model_file=model_path)
    return {"models": models, "meta": meta or {}, "schema": schema or get_feature_schema()}


def predict_lightgbm_deltas(payload, feature_row: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    if pd is None:
        raise RuntimeError(f"pandas is required for runtime inference: {PANDAS_IMPORT_ERROR}")
    meta = meta or payload.get("meta") or {}
    schema = schema or payload.get("schema") or get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    frame = pd.DataFrame([canonicalize_feature_row(feature_row, feature_names=feature_names)])
    frame = _prepare_frame(frame, feature_names, list(meta.get("categorical_features") or CATEGORICAL_FEATURES))
    deltas = {}
    for target, model in payload["models"].items():
        deltas[target] = float(model.predict(frame)[0])
    return deltas


def evaluate_lightgbm_bundle(model_dir: str, dataset_csv: str, language: str = "", format_type: str = "") -> Dict[str, Any]:
    _require_training_stack()
    bundle = load_lightgbm_bundle(model_dir)
    meta = bundle["meta"]
    schema = bundle["schema"]
    df = pd.read_csv(dataset_csv)
    if language:
        df = df[df["language"].astype(str).str.lower() == str(language).strip().lower()]
    if format_type and format_type != "general":
        df = df[df["format_type"].astype(str).str.lower() == str(format_type).strip().lower()]
    frame = _prepare_frame(df, list(schema.get("feature_names") or FEATURE_NAMES), list(meta.get("categorical_features") or CATEGORICAL_FEATURES))
    summary = {"rows": int(len(df)), "targets": {}}
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
        lang_for_clip = language or meta.get("language", "")
        truth = _clip_target_series(lang_for_clip, target, truth)
        pred = bundle["models"][target].predict(frame)
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, [0.0] * len(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    return summary
