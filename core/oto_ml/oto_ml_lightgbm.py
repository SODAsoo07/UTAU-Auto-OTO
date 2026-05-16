"""
LightGBM backend for OTO ML correction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

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
SELECTOR_OBJECTIVE_CHOICES = {"pointwise", "ranking", "pairwise"}
SELECTOR_RANKING_BACKEND_CHOICES = {"lambdarank", "pairwise"}


def _selector_ranking_backend_default() -> str:
    raw = str(os.environ.get("UTOA_SELECTOR_RANKING_BACKEND", "lambdarank") or "").strip().lower()
    if raw in {"pairwise", "rank_xendcg", "xendcg"}:
        raw = "pairwise"
    if raw not in SELECTOR_RANKING_BACKEND_CHOICES:
        raw = "lambdarank"
    return raw


def _resolve_selector_objective(objective: str, df) -> str:
    obj = str(objective or "auto").strip().lower()
    if obj in {"rank", "lambdarank", "ranking_lambdarank"}:
        obj = "ranking"
    elif obj in {"rank_pairwise", "pair", "rank_xendcg", "xendcg"}:
        obj = "pairwise"
    if obj == "auto":
        try:
            min_groups = int(
                os.environ.get("UTOA_SELECTOR_RANKING_MIN_GROUPS", SELECTOR_RANKING_MIN_GROUPS_DEFAULT)
            )
        except Exception:
            min_groups = SELECTOR_RANKING_MIN_GROUPS_DEFAULT
        group_count = int(df["selector_group_id"].nunique()) if "selector_group_id" in df.columns else 0
        if group_count >= max(2, min_groups) and "selector_rank_label" in df.columns:
            return "pairwise" if _selector_ranking_backend_default() == "pairwise" else "ranking"
        return "pointwise"
    if obj in {"ranking", "pairwise"}:
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


def _selector_pick_top1_rows(df, pred=None):
    if "selector_group_id" not in df.columns or len(df) == 0:
        return df.iloc[0:0].copy()
    tmp = df.copy()
    if pred is None:
        if "candidate_mode" in tmp.columns:
            tmp["_rank_score"] = (tmp["candidate_mode"].astype(str).str.lower() == "base").astype(float)
        else:
            tmp["_rank_score"] = 0.0
    else:
        tmp["_rank_score"] = pred
    return (
        tmp.sort_values(["selector_group_id", "_rank_score"], ascending=[True, False])
        .groupby("selector_group_id", as_index=False)
        .head(1)
    )


def _selector_top1_hit_rate(df, pred):
    if "selector_is_best" not in df.columns:
        return 0.0
    picked = _selector_pick_top1_rows(df, pred=pred)
    if len(picked) == 0:
        return 0.0
    return float(pd.to_numeric(picked["selector_is_best"], errors="coerce").fillna(0).mean())


def _selector_baseline_top1_hit_rate(df):
    if "selector_is_best" not in df.columns:
        return 0.0
    picked = _selector_pick_top1_rows(df, pred=None)
    if len(picked) == 0:
        return 0.0
    return float(pd.to_numeric(picked["selector_is_best"], errors="coerce").fillna(0).mean())


def _selector_top1_blank_attach_rate(df, pred=None) -> float:
    if len(df) == 0:
        return 0.0
    picked = _selector_pick_top1_rows(df, pred=pred)
    if len(picked) == 0:
        return 0.0
    if "cand_blank_attach_risk" in picked.columns:
        risk = pd.to_numeric(picked["cand_blank_attach_risk"], errors="coerce").fillna(0.0)
    else:
        risk = pd.Series([0.0] * len(picked), index=picked.index, dtype=float)
    alias = picked.get("alias_type")
    if alias is None:
        head_mask = pd.Series([True] * len(picked), index=picked.index)
    else:
        alias_norm = alias.astype(str).str.strip().str.lower()
        head_mask = alias_norm.isin({"cv", "cv_head", "v", "vv", "mono"})
    risk_th = float(_env_float("UTOA_SELECTOR_TOP1_BLANK_ATTACH_TH", 0.55))
    flagged = (risk >= risk_th) & head_mask
    return float(pd.to_numeric(flagged, errors="coerce").fillna(0.0).mean())


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


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _safe_format_token(value: str) -> str:
    token = str(value or "").strip().lower().replace(" ", "_").replace("-", "_").replace("+", "plus")
    token = "".join(ch for ch in token if ch.isalnum() or ch == "_")
    return token or "general"


def _target_mode() -> str:
    raw = str(os.environ.get("UTOA_ML_TARGET_MODE", "direct") or "").strip().lower()
    if raw in {"direct", "absolute", "abs"}:
        return "direct"
    return "delta"


def _target_transform_mode() -> str:
    raw = str(os.environ.get("UTOA_ML_TARGET_TRANSFORM", "asinh") or "").strip().lower()
    if raw in {"none", "identity", "off", "0"}:
        return "none"
    return "asinh"


def _target_scale(target: str, default: float) -> float:
    token = str(target or "").strip().upper().replace("DELTA_", "")
    key = f"UTOA_ML_TARGET_SCALE_{token}"
    return max(1e-3, _env_float(key, default))


def _target_transform_forward(value: float, *, mode: str, scale: float) -> float:
    if mode == "asinh":
        return float(math.asinh(float(value) / max(1e-6, float(scale))))
    return float(value)


def _target_transform_inverse(value: float, *, mode: str, scale: float) -> float:
    if mode == "asinh":
        return float(math.sinh(float(value)) * max(1e-6, float(scale)))
    return float(value)


def _resolve_target_defaults(language: str, target: str) -> tuple[str, float]:
    lang = str(language or "").strip().lower()
    if target == "delta_cutoff":
        return ("asinh", 90.0 if lang == "korean" else 120.0)
    if target in {"delta_cons", "delta_offset"}:
        return ("asinh", 80.0 if lang == "korean" else 110.0)
    if target == "delta_pre":
        return ("asinh", 70.0 if lang == "korean" else 95.0)
    if target == "delta_ovl":
        return ("asinh", 55.0 if lang == "korean" else 80.0)
    return ("asinh", 80.0)


def _absolute_target_name(target: str) -> str:
    mapping = {
        "delta_offset": "_abs_offset",
        "delta_cons": "_abs_cons",
        "delta_cutoff": "_abs_cutoff",
        "delta_pre": "_abs_pre",
        "delta_ovl": "_abs_ovl",
    }
    return mapping.get(target, target)


def _make_absolute_targets(df):
    out = df.copy()
    out["_abs_offset"] = pd.to_numeric(out.get("base_offset"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("delta_offset"), errors="coerce").fillna(0.0)
    out["_abs_cons"] = pd.to_numeric(out.get("base_cons"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("delta_cons"), errors="coerce").fillna(0.0)
    out["_abs_cutoff"] = pd.to_numeric(out.get("base_cutoff_abs"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("delta_cutoff"), errors="coerce").fillna(0.0)
    out["_abs_pre"] = pd.to_numeric(out.get("base_pre"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("delta_pre"), errors="coerce").fillna(0.0)
    out["_abs_ovl"] = pd.to_numeric(out.get("base_ovl"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("delta_ovl"), errors="coerce").fillna(0.0)
    out["_abs_offset"] = out["_abs_offset"].clip(lower=0.0)
    out["_abs_cons"] = out[["_abs_offset", "_abs_cons"]].max(axis=1)
    out["_abs_pre"] = out["_abs_pre"].clip(lower=0.0)
    out["_abs_ovl"] = out["_abs_ovl"].clip(lower=0.0)
    out["_abs_ovl"] = out[["_abs_ovl", "_abs_pre"]].min(axis=1)
    out["_abs_cutoff"] = out[["_abs_cutoff", "_abs_cons"]].max(axis=1)
    return out


def _apply_balanced_sample_weight(df, sample_weight):
    weights = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0).astype(float)
    alias_strength = max(0.0, min(1.0, _env_float("UTOA_ML_ALIAS_BALANCE_STRENGTH", 0.45)))
    format_strength = max(0.0, min(1.0, _env_float("UTOA_ML_FORMAT_BALANCE_STRENGTH", 0.55)))
    if alias_strength > 0.0 and "alias_type" in df.columns:
        alias_counts = df["alias_type"].astype(str).str.strip().str.lower().value_counts()
        denom = df["alias_type"].astype(str).str.strip().str.lower().map(lambda k: max(1.0, float(alias_counts.get(k, 1.0))))
        alias_factor = (1.0 / denom.pow(0.5)).clip(lower=0.20, upper=3.0)
        weights = weights * (1.0 + alias_strength * ((alias_factor / max(alias_factor.mean(), 1e-6)) - 1.0))
    if format_strength > 0.0 and "format_type" in df.columns:
        fmt_norm = df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type("", v) or str(v).strip().lower())
        fmt_counts = fmt_norm.value_counts()
        denom = fmt_norm.map(lambda k: max(1.0, float(fmt_counts.get(k, 1.0))))
        fmt_factor = (1.0 / denom.pow(0.5)).clip(lower=0.20, upper=3.0)
        weights = weights * (1.0 + format_strength * ((fmt_factor / max(fmt_factor.mean(), 1e-6)) - 1.0))
    return weights.clip(lower=0.10, upper=4.50)


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
    sample_weight = _apply_balanced_sample_weight(df, sample_weight)

    target_mode = _target_mode()
    transform_mode = _target_transform_mode()
    target_scale_map: Dict[str, float] = {}
    target_transform_map: Dict[str, str] = {}
    for target in TARGET_NAMES:
        default_transform, default_scale = _resolve_target_defaults(language, target)
        target_transform_map[target] = transform_mode if transform_mode != "none" else "none"
        target_scale_map[target] = float(_target_scale(target, default_scale if default_transform == "asinh" else 1.0))

    df = df.copy()
    if target_mode == "direct":
        df = _make_absolute_targets(df)
    df["_train_sample_weight"] = sample_weight
    df = df[pd.to_numeric(df["_train_sample_weight"], errors="coerce").fillna(0.0) > 0.0]
    if "format_type" in df.columns:
        df["_format_head"] = (
            df["format_type"]
            .astype(str)
            .str.lower()
            .map(lambda v: normalize_format_type(language, v) or "general")
        )
    else:
        df["_format_head"] = (format_type or "general") if format_type else "general"

    if len(df) < 8:
        raise RuntimeError("Filtered dataset is too small for training.")

    feature_schema = get_feature_schema()
    feature_names = list(feature_schema["feature_names"])
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in feature_names]
    os.makedirs(out_dir, exist_ok=True)

    def _target_column(target_name: str) -> str:
        if target_mode == "direct":
            return _absolute_target_name(target_name)
        return target_name

    def _fit_head_models(head_df, head_out_dir: str) -> tuple[Dict[str, Any], Dict[str, str]]:
        if len(head_df) < 8:
            raise RuntimeError("head dataset is too small for training.")
        os.makedirs(head_out_dir, exist_ok=True)
        frame = _prepare_frame(head_df, feature_names, categorical_features)
        train_idx, valid_idx = _split_train_valid(head_df, group_column)
        X_train = frame.iloc[train_idx]
        X_valid = frame.iloc[valid_idx]
        w_train = pd.to_numeric(head_df.iloc[train_idx]["_train_sample_weight"], errors="coerce").fillna(1.0)
        w_valid = pd.to_numeric(head_df.iloc[valid_idx]["_train_sample_weight"], errors="coerce").fillna(1.0)

        head_metrics: Dict[str, Any] = {}
        head_targets: Dict[str, str] = {}
        for target in TARGET_NAMES:
            col_name = _target_column(target)
            y_train_raw = pd.to_numeric(head_df.iloc[train_idx][col_name], errors="coerce").fillna(0.0)
            y_valid_raw = pd.to_numeric(head_df.iloc[valid_idx][col_name], errors="coerce").fillna(0.0)
            if target_mode == "delta":
                y_train_raw = _clip_target_series(language, target, y_train_raw)
                y_valid_raw = _clip_target_series(language, target, y_valid_raw)

            t_mode = target_transform_map.get(target, "none")
            t_scale = float(target_scale_map.get(target, 1.0))
            y_train = y_train_raw.map(lambda v: _target_transform_forward(v, mode=t_mode, scale=t_scale))
            y_valid = y_valid_raw.map(lambda v: _target_transform_forward(v, mode=t_mode, scale=t_scale))
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
            params = dict(DEFAULT_LGB_PARAMS)
            if str(params.get("objective", "")).strip().lower() == "regression_l1":
                params["objective"] = "huber"
                params["alpha"] = 0.85
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            model_path = os.path.join(head_out_dir, f"model_{target.replace('delta_', '')}.txt")
            booster.save_model(model_path)
            pred_trans = booster.predict(X_valid)
            pred = [_target_transform_inverse(v, mode=t_mode, scale=t_scale) for v in pred_trans]
            head_metrics[target] = {
                "baseline_mae": float(mean_absolute_error(y_valid_raw, [0.0] * len(y_valid_raw))),
                "model_mae": float(mean_absolute_error(y_valid_raw, pred)),
            }
            head_targets[target] = model_path
        return head_metrics, head_targets

    # Root head (always trained)
    out_metrics, targets = _fit_head_models(df, out_dir)

    format_heads: Dict[str, Any] = {}
    if _env_flag("UTOA_ML_FORMAT_HEADS", True):
        for head_fmt in sorted({str(v or "general").strip().lower() for v in df["_format_head"].tolist()}):
            if not head_fmt:
                continue
            head_df = df[df["_format_head"] == head_fmt].copy()
            if len(head_df) < 16:
                continue
            head_dir = os.path.join(out_dir, "heads", _safe_format_token(head_fmt))
            head_metrics, head_targets = _fit_head_models(head_df, head_dir)
            format_heads[head_fmt] = {
                "model_dir": head_dir,
                "rows": int(len(head_df)),
                "targets": dict(head_targets),
                "holdout_metrics": dict(head_metrics),
            }

    write_feature_schema(os.path.join(out_dir, "feature_schema.json"))
    meta = {
        "backend": "lightgbm",
        "language": language,
        "format_type": format_type,
        "model_version": "v2",
        "feature_version": feature_schema["feature_version"],
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "targets": list(TARGET_NAMES),
        "target_mode": target_mode,
        "target_transform": {
            "mode": transform_mode,
            "scales": {k: float(v) for k, v in target_scale_map.items()},
        },
        "format_heads_enabled": bool(_env_flag("UTOA_ML_FORMAT_HEADS", True)),
        "format_heads": dict(format_heads),
        "default_head": "__default__",
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
            "alias_balance_strength": float(_env_float("UTOA_ML_ALIAS_BALANCE_STRENGTH", 0.45)),
            "format_balance_strength": float(_env_float("UTOA_ML_FORMAT_BALANCE_STRENGTH", 0.55)),
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


def _selector_sample_weight(df):
    if len(df) == 0:
        return pd.Series([], dtype=float)
    weights = pd.Series([1.0] * len(df), index=df.index, dtype=float)

    if "cand_blank_attach_risk" in df.columns:
        risk = pd.to_numeric(df["cand_blank_attach_risk"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    else:
        risk = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    strength = max(1.0, float(_env_float("UTOA_SELECTOR_BLANK_ATTACH_WEIGHT", 1.35)))
    floor = max(0.0, min(1.0, float(_env_float("UTOA_SELECTOR_BLANK_ATTACH_WEIGHT_FLOOR", 0.25))))
    if strength > 1.0:
        scale = 1.0 + (strength - 1.0) * (floor + (1.0 - floor) * risk)
        weights = weights * scale

    if "selector_is_best" in df.columns:
        pos_mask = pd.to_numeric(df["selector_is_best"], errors="coerce").fillna(0.0) > 0.5
        pos_weight = max(1.0, float(_env_float("UTOA_SELECTOR_POS_WEIGHT", 1.15)))
        if pos_weight > 1.0:
            weights = weights * (1.0 + (pos_weight - 1.0) * pos_mask.astype(float))
    return weights.clip(lower=0.10, upper=8.0)


def _selector_ranking_params(objective: str, max_label: int) -> Dict[str, Any]:
    if objective == "pairwise":
        return {
            "objective": "rank_xendcg",
            "metric": "ndcg",
        }
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": _make_label_gain(max_label),
    }


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
    if objective not in SELECTOR_OBJECTIVE_CHOICES:
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
    w_train = _selector_sample_weight(train_df)
    w_valid = _selector_sample_weight(valid_df)

    params = dict(DEFAULT_LGB_PARAMS)
    ranking_backend = ""
    if objective in {"ranking", "pairwise"}:
        ranking_backend = objective
        y_train = pd.to_numeric(train_df["selector_rank_label"], errors="coerce").fillna(0).astype(int)
        y_valid = pd.to_numeric(valid_df["selector_rank_label"], errors="coerce").fillna(0).astype(int)
        max_label = int(max(y_train.max() if len(y_train) else 0, y_valid.max() if len(y_valid) else 0, 0))
        params.update(_selector_ranking_params(objective, max_label))
        train_order = train_df.sort_values(["selector_group_id", "candidate_index"]).index
        valid_order = valid_df.sort_values(["selector_group_id", "candidate_index"]).index
        train_df = train_df.loc[train_order]
        valid_df = valid_df.loc[valid_order]
        X_train = X_train.loc[train_order]
        X_valid = X_valid.loc[valid_order]
        y_train = y_train.loc[train_order]
        y_valid = y_valid.loc[valid_order]
        w_train = w_train.loc[train_order]
        w_valid = w_valid.loc[valid_order]
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            group=_group_sizes_from_series(train_df["selector_group_id"]),
            weight=w_train,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=_group_sizes_from_series(valid_df["selector_group_id"]),
            weight=w_valid,
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
        "ranking_backend": ranking_backend,
        "rows": int(len(df)),
        "groups": int(df["selector_group_id"].nunique()) if "selector_group_id" in df.columns else 0,
        "top1_baseline": float(_selector_baseline_top1_hit_rate(valid_df)),
        "top1_model": float(_selector_top1_hit_rate(valid_df, pred)),
        "top1_blank_attach_baseline": float(_selector_top1_blank_attach_rate(valid_df, pred=None)),
        "top1_blank_attach_model": float(_selector_top1_blank_attach_rate(valid_df, pred=pred)),
    }
    selector_summary["top1_blank_attach_delta"] = float(
        selector_summary["top1_blank_attach_model"] - selector_summary["top1_blank_attach_baseline"]
    )
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
        "selector_ranking_backend": ranking_backend,
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
        "ranking_backend": str(meta.get("selector_ranking_backend", "") or ""),
        "top1_baseline": float(_selector_baseline_top1_hit_rate(df)),
        "top1_model": float(_selector_top1_hit_rate(df, pred)),
        "top1_blank_attach_baseline": float(_selector_top1_blank_attach_rate(df, pred=None)),
        "top1_blank_attach_model": float(_selector_top1_blank_attach_rate(df, pred=pred)),
    }
    summary["top1_blank_attach_delta"] = float(
        summary["top1_blank_attach_model"] - summary["top1_blank_attach_baseline"]
    )
    if "selector_quality_score" in df.columns:
        truth = pd.to_numeric(df["selector_quality_score"], errors="coerce").fillna(0.0)
        summary["score_mae"] = float(mean_absolute_error(truth, pred))
    if "candidate_mode" in df.columns and len(df) > 0:
        summary["candidate_mode_counts"] = {
            str(k): int(v)
            for k, v in df["candidate_mode"].astype(str).value_counts().to_dict().items()
        }
    return summary


def _clip_delta_scalar(language: str, target: str, value: float) -> float:
    clip = get_delta_clip_limits(str(language or "").strip().lower()).get(target)
    if not clip:
        return float(value)
    lo, hi = float(clip[0]), float(clip[1])
    return float(min(max(float(value), lo), hi))


def _target_transform_for_runtime(meta: Dict[str, Any], target: str) -> tuple[str, float]:
    info = meta.get("target_transform")
    if not isinstance(info, dict):
        return "none", 1.0
    raw_mode = str(info.get("mode", "none") or "none").strip().lower()
    mode = "asinh" if raw_mode == "asinh" else "none"
    modes = info.get("modes")
    if isinstance(modes, dict):
        per_target_mode = str(modes.get(target, mode) or mode).strip().lower()
        mode = "asinh" if per_target_mode == "asinh" else "none"
    scales = info.get("scales")
    scale = 1.0
    if isinstance(scales, dict):
        try:
            scale = float(scales.get(target, 1.0))
        except Exception:
            scale = 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return mode, scale


def _base_feature_name_for_target(target: str) -> str:
    mapping = {
        "delta_offset": "base_offset",
        "delta_cons": "base_cons",
        "delta_cutoff": "base_cutoff_abs",
        "delta_pre": "base_pre",
        "delta_ovl": "base_ovl",
    }
    return mapping.get(target, "")


def _target_mode_from_meta(meta: Dict[str, Any]) -> str:
    raw = str(meta.get("target_mode", "delta") or "delta").strip().lower()
    if raw in {"direct", "absolute", "abs"}:
        return "direct"
    return "delta"


def _normalized_row_format(language: str, feature_row: Dict[str, Any]) -> str:
    for key in ("format_type", "_format_head", "format"):
        raw = str(feature_row.get(key, "") or "").strip()
        if not raw:
            continue
        norm = normalize_format_type(language, raw)
        if norm:
            return norm
        return raw.lower()
    return "general"


def _resolve_head_name(meta: Dict[str, Any], feature_row: Dict[str, Any]) -> str:
    format_heads = meta.get("format_heads")
    if not isinstance(format_heads, dict) or not format_heads:
        return "__default__"
    language = str(meta.get("language", "") or feature_row.get("language", "") or "").strip().lower()
    row_fmt = _normalized_row_format(language, feature_row)
    if row_fmt in format_heads:
        return row_fmt
    return "__default__"


def _load_head_models(model_dir: str, head_name: str, head_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    models: Dict[str, Any] = {}
    targets_map = head_meta.get("targets")
    if not isinstance(targets_map, dict):
        targets_map = {}
    for target in TARGET_NAMES:
        model_path = str(targets_map.get(target, "") or "").strip()
        if not model_path:
            model_path = os.path.join(
                model_dir,
                "heads",
                _safe_format_token(head_name),
                f"model_{target.replace('delta_', '')}.txt",
            )
        elif not os.path.isabs(model_path):
            model_path = os.path.join(model_dir, model_path)
        if not os.path.exists(model_path):
            return None
        runtime_model_path = _normalize_model_file_for_runtime(model_path)
        models[target] = lgb.Booster(model_file=runtime_model_path)
    return models


def _resolve_runtime_models(payload: Dict[str, Any], head_name: str) -> Dict[str, Any]:
    default_models = dict((payload or {}).get("models") or {})
    head_models = dict(((payload or {}).get("head_models") or {}).get(head_name) or {})
    resolved: Dict[str, Any] = {}
    for target in TARGET_NAMES:
        model = head_models.get(target) or default_models.get(target)
        if model is None:
            raise RuntimeError(f"Missing LightGBM target model: {target} (head={head_name})")
        resolved[target] = model
    return resolved


def _decode_prediction_to_delta(
    *,
    raw_pred: float,
    target: str,
    feature_row: Dict[str, Any],
    meta: Dict[str, Any],
) -> float:
    mode, scale = _target_transform_for_runtime(meta, target)
    value = _target_transform_inverse(raw_pred, mode=mode, scale=scale)
    target_mode = _target_mode_from_meta(meta)
    if target_mode == "direct":
        base_name = _base_feature_name_for_target(target)
        try:
            base_value = float(feature_row.get(base_name, 0.0) or 0.0)
        except Exception:
            base_value = 0.0
        value = float(value) - float(base_value)
    language = str(meta.get("language", "") or feature_row.get("language", "") or "").strip().lower()
    return _clip_delta_scalar(language, target, value)


def predict_lightgbm_deltas_batch(
    payload,
    feature_rows: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    if pd is None:
        raise RuntimeError(f"pandas is required for runtime inference: {PANDAS_IMPORT_ERROR}")
    if not feature_rows:
        return []
    meta = meta or payload.get("meta") or {}
    schema = schema or payload.get("schema") or get_feature_schema()
    feature_names = list(schema.get("feature_names") or FEATURE_NAMES)
    categorical = list(meta.get("categorical_features") or CATEGORICAL_FEATURES)
    canonical_rows = [
        canonicalize_feature_row(feature_row, feature_names=feature_names)
        for feature_row in feature_rows
    ]
    frame = pd.DataFrame(canonical_rows)
    frame = _prepare_frame(frame, feature_names, categorical)

    head_names = [_resolve_head_name(meta, row) for row in canonical_rows]
    by_head: Dict[str, List[int]] = {}
    for idx, head_name in enumerate(head_names):
        by_head.setdefault(head_name, []).append(idx)

    out: List[Dict[str, float]] = [{target: 0.0 for target in TARGET_NAMES} for _ in range(len(canonical_rows))]
    for head_name, indices in by_head.items():
        models = _resolve_runtime_models(payload, head_name)
        sub_frame = frame.iloc[indices]
        for target in TARGET_NAMES:
            pred_values = models[target].predict(sub_frame)
            for local_idx, row_idx in enumerate(indices):
                out[row_idx][target] = float(
                    _decode_prediction_to_delta(
                        raw_pred=float(pred_values[local_idx]),
                        target=target,
                        feature_row=canonical_rows[row_idx],
                        meta=meta,
                    )
                )
    return out


def load_lightgbm_bundle(model_dir: str, meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None):
    if lgb is None:
        raise RuntimeError(f"lightgbm is required for runtime inference: {LIGHTGBM_IMPORT_ERROR}")
    model_dir = os.path.abspath(model_dir)
    if not isinstance(meta, dict) or not meta:
        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        else:
            meta = {}
    if not isinstance(schema, dict) or not schema:
        schema_path = os.path.join(model_dir, "feature_schema.json")
        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f) or {}
        else:
            schema = get_feature_schema()

    models: Dict[str, Any] = {}
    for target in TARGET_NAMES:
        model_name = f"model_{target.replace('delta_', '')}.txt"
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        runtime_model_path = _normalize_model_file_for_runtime(model_path)
        models[target] = lgb.Booster(model_file=runtime_model_path)

    head_models: Dict[str, Dict[str, Any]] = {}
    format_heads = meta.get("format_heads")
    if isinstance(format_heads, dict) and bool(meta.get("format_heads_enabled", False)):
        for head_name, head_meta in format_heads.items():
            if not isinstance(head_meta, dict):
                continue
            loaded = _load_head_models(model_dir, str(head_name), head_meta)
            if loaded:
                head_models[str(head_name)] = loaded
            else:
                logger.warning("Skipped invalid LightGBM format head: %s", head_name)

    return {
        "models": models,
        "head_models": head_models,
        "meta": meta or {},
        "schema": schema or get_feature_schema(),
    }


def predict_lightgbm_deltas(payload, feature_row: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, schema: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    preds = predict_lightgbm_deltas_batch(
        payload,
        [feature_row],
        meta=meta,
        schema=schema,
    )
    if not preds:
        return {target: 0.0 for target in TARGET_NAMES}
    return dict(preds[0])


def evaluate_lightgbm_bundle(model_dir: str, dataset_csv: str, language: str = "", format_type: str = "") -> Dict[str, Any]:
    _require_training_stack()
    meta_path = os.path.join(model_dir, "model_meta.json")
    schema_path = os.path.join(model_dir, "feature_schema.json")
    meta = {}
    schema = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    if os.path.isfile(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f) or {}
    bundle = load_lightgbm_bundle(model_dir, meta=meta, schema=schema)
    meta = bundle.get("meta") or {}
    schema = bundle.get("schema") or get_feature_schema()
    df = pd.read_csv(dataset_csv)
    if language:
        df = df[df["language"].astype(str).str.lower() == str(language).strip().lower()]
    format_type = normalize_format_type(language or meta.get("language", ""), format_type)
    if format_type and format_type != "general":
        df = df[
            df["format_type"].astype(str).str.lower().map(lambda v: normalize_format_type(language or meta.get("language", ""), v)) == format_type
        ]
    rows = df.to_dict("records")
    pred_rows = predict_lightgbm_deltas_batch(bundle, rows, meta=meta, schema=schema)
    summary = {"rows": int(len(df)), "targets": {}}
    for target in TARGET_NAMES:
        truth = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
        lang_for_clip = language or meta.get("language", "")
        truth = _clip_target_series(lang_for_clip, target, truth)
        pred = [float(row.get(target, 0.0)) for row in pred_rows]
        summary["targets"][target] = {
            "baseline_mae": float(mean_absolute_error(truth, [0.0] * len(truth))),
            "model_mae": float(mean_absolute_error(truth, pred)),
        }
    if len(pred_rows) > 0:
        summary["format_heads_used"] = {
            "enabled": bool(meta.get("format_heads_enabled", False)),
            "available": sorted(list(((bundle.get("head_models") or {}).keys()))),
        }
    return summary
