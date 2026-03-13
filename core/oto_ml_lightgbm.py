"""
LightGBM backend for OTO ML correction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
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

from core.format_type_utils import normalize_format_type
from core.oto_ml_policy import (
    alias_family_to_alias_types,
    default_training_filters,
    normalize_alias_family,
    selector_enabled_by_default,
)
from core.oto_ml_features import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET_NAMES, canonicalize_feature_row, get_delta_clip_limits, get_feature_schema, write_feature_schema
from core.oto_ml_selector import (
    SELECTOR_CATEGORICAL_FEATURES,
    SELECTOR_FEATURE_NAMES,
    canonicalize_selector_feature_row,
    get_selector_feature_schema,
    write_selector_feature_schema,
)

logger = logging.getLogger(__name__)

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

SELECTOR_RANKING_MIN_GROUPS_DEFAULT = 120


def _resolve_selector_objective(objective: str, df) -> str:
    obj = str(objective or "auto").strip().lower()
    if obj == "auto":
        try:
            min_groups = int(
                os.environ.get("UTOA_SELECTOR_RANKING_MIN_GROUPS", SELECTOR_RANKING_MIN_GROUPS_DEFAULT)
            )
        except Exception:
            min_groups = SELECTOR_RANKING_MIN_GROUPS_DEFAULT
        group_count = int(df["selector_group_id"].nunique()) if "selector_group_id" in df.columns else 0
        if group_count >= max(2, min_groups) and "selector_rank_label" in df.columns:
            return "ranking"
        return "pointwise"
    if obj == "ranking":
        if "selector_group_id" not in df.columns or "selector_rank_label" not in df.columns:
            return "pointwise"
    return obj


def _make_label_gain(max_label: int) -> list[int]:
    max_value = max(0, int(max_label))
    gains = [0]
    for idx in range(1, max_value + 1):
        gains.append((1 << idx) - 1)
    return gains


def _runtime_model_cache_dir() -> str:
    configured = str(os.environ.get("UTOA_LGB_MODEL_CACHE_DIR", "") or "").strip()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    path = os.path.join(tempfile.gettempdir(), "utoa_lgb_models")
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_model_file_for_runtime(model_path: str) -> str:
    """
    일부 LightGBM 런타임에서 CRLF 모델 텍스트 파싱 중 치명 오류가 발생할 수 있어
    로딩 전에 LF 전용 임시 사본으로 정규화한다.
    """
    try:
        with open(model_path, "rb") as f:
            raw = f.read()
    except OSError:
        return model_path
    if b"\r" not in raw:
        return model_path

    try:
        st = os.stat(model_path)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        sig = f"{os.path.abspath(model_path)}|{int(st.st_size)}|{mtime_ns}"
        digest = hashlib.sha1(sig.encode("utf-8", errors="ignore")).hexdigest()[:16]
        base = os.path.splitext(os.path.basename(model_path))[0]
        normalized_path = os.path.join(_runtime_model_cache_dir(), f"{base}_{digest}.txt")
        if os.path.exists(normalized_path):
            return normalized_path
        normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        with open(normalized_path, "wb") as f:
            f.write(normalized)
        return normalized_path
    except Exception as e:
        logger.warning("Failed to normalize LightGBM model file (%s): %s", model_path, e)
        return model_path


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


def _group_sizes_from_series(series):
    groups = []
    current = None
    count = 0
    for value in series.astype(str).tolist():
        if current is None:
            current = value
            count = 1
            continue
        if value == current:
            count += 1
            continue
        groups.append(count)
        current = value
        count = 1
    if count > 0:
        groups.append(count)
    return groups


def _selector_top1_hit_rate(df, pred):
    if "selector_group_id" not in df.columns or "selector_is_best" not in df.columns or len(df) == 0:
        return 0.0
    tmp = df.copy()
    tmp["_pred"] = pred
    picked = tmp.sort_values(["selector_group_id", "_pred"], ascending=[True, False]).groupby("selector_group_id", as_index=False).head(1)
    if len(picked) == 0:
        return 0.0
    return float(pd.to_numeric(picked["selector_is_best"], errors="coerce").fillna(0).mean())


def _selector_baseline_top1_hit_rate(df):
    if "selector_group_id" not in df.columns or "selector_is_best" not in df.columns or len(df) == 0:
        return 0.0
    tmp = df.copy()
    if "candidate_mode" in tmp.columns:
        tmp["_base_pref"] = (tmp["candidate_mode"].astype(str).str.lower() == "base").astype(int)
    else:
        tmp["_base_pref"] = 0
    picked = tmp.sort_values(["selector_group_id", "_base_pref"], ascending=[True, False]).groupby("selector_group_id", as_index=False).head(1)
    if len(picked) == 0:
        return 0.0
    return float(pd.to_numeric(picked["selector_is_best"], errors="coerce").fillna(0).mean())


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


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
    alias_family: str = "",
    require_train_keep: bool = False,
    min_mapping_confidence: float = 0.0,
    exclude_nuclei_fallback: bool = False,
) -> Dict[str, Any]:
    _require_training_stack()
    if not dataset_csv or not os.path.exists(dataset_csv):
        raise FileNotFoundError(dataset_csv)

    df = pd.read_csv(dataset_csv)
    language = str(language).strip().lower()
    format_type = normalize_format_type(language, format_type)
    alias_family = normalize_alias_family(alias_family)
    default_policy = default_training_filters(language, format_type, alias_family=alias_family)
    if alias_family and not alias_types:
        alias_types = alias_family_to_alias_types(alias_family)
    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == language]
    if format_type and format_type != "general" and "format_type" in df.columns:
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(language, v)) == format_type
        ]
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

    if "sample_weight" in df.columns:
        sample_weight = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).astype(float)
    else:
        sample_weight = pd.Series([1.0] * len(df), index=df.index, dtype=float)
    if "blank_risk_score" in df.columns and "alias_type" in df.columns:
        blank_weight = max(0.0, min(0.90, _env_float("UTOA_ML_BLANK_RISK_WEIGHT", 0.45)))
        blank_score = pd.to_numeric(df["blank_risk_score"], errors="coerce").fillna(0.0).astype(float)
        alias_type = df["alias_type"].astype(str).str.strip().str.lower()
        cv_mask = alias_type.isin(["cv", "cv_head"])
        blank_factor = (1.0 - (blank_weight * blank_score)).clip(lower=0.25, upper=1.0)
        sample_weight = sample_weight * blank_factor.where(cv_mask, 1.0)

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
            "alias_family": alias_family,
            "require_train_keep": bool(require_train_keep),
            "min_mapping_confidence": float(min_mapping_confidence),
            "exclude_nuclei_fallback": bool(exclude_nuclei_fallback),
            "blank_risk_weight": float(_env_float("UTOA_ML_BLANK_RISK_WEIGHT", 0.45)),
        },
        "default_policy": default_policy,
        "selector_default_enabled": bool(selector_enabled_by_default(language, format_type, alias_family=alias_family)),
        "weight_summary": {
            "min": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).min()) if len(df) else 0.0,
            "max": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).max()) if len(df) else 0.0,
            "mean": float(pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0).mean()) if len(df) else 0.0,
            "blank_risk_rows": int(df["blank_risk_score"].ge(0.55).sum()) if "blank_risk_score" in df.columns else 0,
        },
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"targets": targets, "metrics": out_metrics}, f, ensure_ascii=False, indent=2)
    return meta


def train_lightgbm_selector_bundle(
    language: str,
    format_type: str,
    selector_dataset_csv: str,
    out_dir: str,
    group_column: str = "voicebank_id",
    objective: str = "auto",
    num_boost_round: int = 400,
    early_stopping_rounds: int = 40,
    alias_family: str = "",
) -> Dict[str, Any]:
    _require_training_stack()
    if not selector_dataset_csv or not os.path.exists(selector_dataset_csv):
        raise FileNotFoundError(selector_dataset_csv)

    df = pd.read_csv(selector_dataset_csv, low_memory=False)
    language = str(language).strip().lower()
    format_type = normalize_format_type(language, format_type)
    alias_family = normalize_alias_family(alias_family)
    objective = _resolve_selector_objective(objective, df)
    if objective not in {"pointwise", "ranking"}:
        raise ValueError(f"Unsupported selector objective: {objective}")

    if "language" in df.columns:
        df = df[df["language"].astype(str).str.lower() == language]
    if format_type and format_type != "general" and "format_type" in df.columns:
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(language, v)) == format_type
        ]
    if len(df) < 16:
        raise RuntimeError("Selector dataset is too small for training.")

    feature_schema = get_selector_feature_schema()
    feature_names = list(feature_schema["feature_names"])
    categorical_features = [c for c in SELECTOR_CATEGORICAL_FEATURES if c in feature_names]
    frame = _prepare_frame(df, feature_names, categorical_features)
    train_idx, valid_idx = _split_train_valid(df, group_column)
    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    X_train = frame.iloc[train_idx].copy()
    X_valid = frame.iloc[valid_idx].copy()

    params = dict(DEFAULT_LGB_PARAMS)
    if objective == "ranking":
        y_train = pd.to_numeric(train_df["selector_rank_label"], errors="coerce").fillna(0).astype(int)
        y_valid = pd.to_numeric(valid_df["selector_rank_label"], errors="coerce").fillna(0).astype(int)
        max_label = int(max(y_train.max() if len(y_train) else 0, y_valid.max() if len(y_valid) else 0, 0))
        params.update({
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": _make_label_gain(max_label),
        })
        train_order = train_df.sort_values(["selector_group_id", "candidate_index"]).index
        valid_order = valid_df.sort_values(["selector_group_id", "candidate_index"]).index
        train_df = train_df.loc[train_order]
        valid_df = valid_df.loc[valid_order]
        X_train = X_train.loc[train_order]
        X_valid = X_valid.loc[valid_order]
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            group=_group_sizes_from_series(train_df["selector_group_id"]),
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=_group_sizes_from_series(valid_df["selector_group_id"]),
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
    else:
        params.update({"objective": "regression_l2", "metric": "l2"})
        y_train = pd.to_numeric(train_df["selector_quality_score"], errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(valid_df["selector_quality_score"], errors="coerce").fillna(0.0)
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "selector_model.txt")
    booster.save_model(model_path)
    write_selector_feature_schema(os.path.join(out_dir, "selector_feature_schema.json"))
    pred = booster.predict(X_valid)
    selector_summary = {
        "objective": objective,
        "rows": int(len(df)),
        "groups": int(df["selector_group_id"].nunique()) if "selector_group_id" in df.columns else 0,
        "top1_baseline": float(_selector_baseline_top1_hit_rate(valid_df)),
        "top1_model": float(_selector_top1_hit_rate(valid_df, pred)),
    }
    if objective == "pointwise":
        selector_summary["score_mae"] = float(
            mean_absolute_error(
                pd.to_numeric(valid_df["selector_quality_score"], errors="coerce").fillna(0.0),
                pred,
            )
        )

    meta = {
        "backend": "lightgbm",
        "language": language,
        "format_type": format_type,
        "model_version": "v1",
        "feature_version": feature_schema["feature_version"],
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "selector_objective": objective,
        "alias_family": alias_family,
        "selector_default_enabled": bool(selector_enabled_by_default(language, format_type, alias_family=alias_family)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": int(len(df)),
        "voicebank_count": int(df[group_column].nunique()) if group_column in df.columns else 1,
        "metrics": selector_summary,
    }
    with open(os.path.join(out_dir, "selector_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_lightgbm_selector_bundle(model_dir: str):
    if lgb is None:
        raise RuntimeError(f"lightgbm is required for runtime inference: {LIGHTGBM_IMPORT_ERROR}")
    meta_path = os.path.join(model_dir, "selector_meta.json")
    schema_path = os.path.join(model_dir, "selector_feature_schema.json")
    model_path = os.path.join(model_dir, "selector_model.txt")
    if not (os.path.exists(meta_path) and os.path.exists(schema_path) and os.path.exists(model_path)):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    runtime_model_path = _normalize_model_file_for_runtime(model_path)
    model = lgb.Booster(model_file=runtime_model_path)
    return {"model": model, "meta": meta, "schema": schema}


def predict_lightgbm_selector_score(payload, feature_row: Dict[str, Any]) -> float:
    if pd is None:
        raise RuntimeError(f"pandas is required for runtime inference: {PANDAS_IMPORT_ERROR}")
    meta = payload.get("meta") or {}
    schema = payload.get("schema") or get_selector_feature_schema()
    feature_names = list(schema.get("feature_names") or SELECTOR_FEATURE_NAMES)
    frame = pd.DataFrame([canonicalize_selector_feature_row(feature_row, feature_names=feature_names)])
    frame = _prepare_frame(frame, feature_names, list(meta.get("categorical_features") or SELECTOR_CATEGORICAL_FEATURES))
    return float(payload["model"].predict(frame)[0])


def evaluate_lightgbm_selector_bundle(model_dir: str, selector_dataset_csv: str, language: str = "", format_type: str = "") -> Dict[str, Any]:
    _require_training_stack()
    payload = load_lightgbm_selector_bundle(model_dir)
    if not payload:
        raise FileNotFoundError(f"Selector bundle not found: {model_dir}")
    meta = payload.get("meta") or {}
    schema = payload.get("schema") or get_selector_feature_schema()
    df = pd.read_csv(selector_dataset_csv)
    if language:
        df = df[df["language"].astype(str).str.lower() == str(language).strip().lower()]
    format_type = normalize_format_type(language or meta.get("language", ""), format_type)
    if format_type and format_type != "general":
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(language or meta.get("language", ""), v)) == format_type
        ]
    feature_names = list(schema.get("feature_names") or SELECTOR_FEATURE_NAMES)
    frame = _prepare_frame(df, feature_names, list(meta.get("categorical_features") or SELECTOR_CATEGORICAL_FEATURES))
    pred = payload["model"].predict(frame)
    summary = {
        "rows": int(len(df)),
        "groups": int(df["selector_group_id"].nunique()) if "selector_group_id" in df.columns else 0,
        "objective": str(meta.get("selector_objective", "") or ""),
        "top1_baseline": float(_selector_baseline_top1_hit_rate(df)),
        "top1_model": float(_selector_top1_hit_rate(df, pred)),
    }
    if "selector_quality_score" in df.columns:
        truth = pd.to_numeric(df["selector_quality_score"], errors="coerce").fillna(0.0)
        summary["score_mae"] = float(mean_absolute_error(truth, pred))
    if "candidate_mode" in df.columns and len(df) > 0:
        summary["candidate_mode_counts"] = {
            str(k): int(v)
            for k, v in df["candidate_mode"].astype(str).value_counts().to_dict().items()
        }
    return summary


def load_lightgbm_bundle(model_dir: str, meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None):
    if lgb is None:
        raise RuntimeError(f"lightgbm is required for runtime inference: {LIGHTGBM_IMPORT_ERROR}")
    models = {}
    for target in TARGET_NAMES:
        model_name = f"model_{target.replace('delta_', '')}.txt"
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        runtime_model_path = _normalize_model_file_for_runtime(model_path)
        models[target] = lgb.Booster(model_file=runtime_model_path)
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
    format_type = normalize_format_type(language or meta.get("language", ""), format_type)
    if format_type and format_type != "general":
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(language or meta.get("language", ""), v)) == format_type
        ]
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
