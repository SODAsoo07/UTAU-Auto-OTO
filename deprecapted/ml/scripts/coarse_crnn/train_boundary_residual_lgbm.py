from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.coarse_crnn.boundary_residual import (
    DEFAULT_DELTA_CLIPS,
    RESIDUAL_CATEGORICAL_FEATURES,
    RESIDUAL_FEATURE_NAMES,
    RESIDUAL_TARGET_NAMES,
)
from core.coarse_crnn.alias_role import normalize_role
from core.runtime.runtime_encoding import bootstrap_utf8_runtime

try:
    import lightgbm as lgb
except Exception as _lgb_exc:  # pragma: no cover
    lgb = None
    LGB_IMPORT_ERROR = _lgb_exc
else:
    LGB_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as _pd_exc:  # pragma: no cover
    pd = None
    PD_IMPORT_ERROR = _pd_exc
else:
    PD_IMPORT_ERROR = None

try:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit
except Exception as _sk_exc:  # pragma: no cover
    mean_absolute_error = None
    GroupShuffleSplit = None
    SK_IMPORT_ERROR = _sk_exc
else:
    SK_IMPORT_ERROR = None


def _require_stack() -> None:
    if lgb is None:
        raise RuntimeError(f"lightgbm is required: {LGB_IMPORT_ERROR}")
    if pd is None:
        raise RuntimeError(f"pandas is required: {PD_IMPORT_ERROR}")
    if mean_absolute_error is None or GroupShuffleSplit is None:
        raise RuntimeError(f"scikit-learn is required: {SK_IMPORT_ERROR}")


def _prepare_frame(df):
    frame = df.copy()
    for name in RESIDUAL_FEATURE_NAMES:
        if name not in frame.columns:
            frame[name] = "" if name in RESIDUAL_CATEGORICAL_FEATURES else 0.0
    frame = frame[list(RESIDUAL_FEATURE_NAMES)].copy()
    for name in RESIDUAL_FEATURE_NAMES:
        if name in RESIDUAL_CATEGORICAL_FEATURES:
            frame[name] = frame[name].fillna("").astype("string").astype("category")
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
    return frame


def _split_indices(df, group_column: str):
    if str(group_column or "").strip() and group_column in df.columns and df[group_column].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(splitter.split(df, groups=df[group_column]))
        return list(train_idx), list(valid_idx)
    split_at = max(1, int(len(df) * 0.8))
    train_idx = list(range(split_at))
    valid_idx = list(range(split_at, len(df)))
    if not valid_idx:
        valid_idx = list(range(max(0, split_at - 1), len(df)))
    return train_idx, valid_idx


def main() -> int:
    bootstrap_utf8_runtime()
    _require_stack()
    ap = argparse.ArgumentParser(description="Train LightGBM residual model for boundary decoder.")
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-column", default="voicebank_id")
    ap.add_argument("--num-boost-round", type=int, default=700)
    ap.add_argument("--early-stopping-rounds", type=int, default=70)
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--fit-stage1-role-bias", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stage1-role-column", default="role")
    ap.add_argument("--stage1-min-role-rows", type=int, default=40)
    ap.add_argument("--stage1-shrink", type=float, default=80.0)
    # Held-out validation against a residual dataset built from a *separate*
    # manifest (e.g. the val split). When given, this CSV is used for LightGBM
    # early stopping instead of an internal voicebank GroupShuffleSplit — this
    # is the honest generalization check for unseen voicebanks.
    ap.add_argument("--valid-csv", default="", help="Optional separate residual CSV used as the LightGBM validation set.")
    # Regularization knobs (defaults reproduce the legacy high-capacity model).
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=40)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--lambda-l1", type=float, default=0.0)
    ap.add_argument("--lambda-l2", type=float, default=0.0)
    # Shrinks DEFAULT_DELTA_CLIPS so per-target corrections cannot push params
    # as far — limits constraint-breaking over-correction (stretch/monotonic).
    ap.add_argument("--delta-clip-scale", type=float, default=1.0)
    args = ap.parse_args()

    csv_path = os.path.abspath(str(args.dataset_csv))
    df = pd.read_csv(csv_path, encoding="utf-8")
    if len(df) < int(args.min_rows):
        raise SystemExit(f"Not enough rows: {len(df)} < min_rows({int(args.min_rows)})")

    frame = _prepare_frame(df)
    valid_csv = str(args.valid_csv or "").strip()
    if valid_csv:
        valid_csv_path = os.path.abspath(valid_csv)
        df_valid = pd.read_csv(valid_csv_path, encoding="utf-8").reset_index(drop=True)
        df_train = df.reset_index(drop=True)
        X_train = _prepare_frame(df_train)
        X_valid = _prepare_frame(df_valid)
        split_mode = f"external_valid_csv:{valid_csv_path}"
    else:
        valid_csv_path = ""
        train_idx, valid_idx = _split_indices(df, str(args.group_column or "").strip())
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[valid_idx].reset_index(drop=True)
        X_train = frame.iloc[train_idx].reset_index(drop=True)
        X_valid = frame.iloc[valid_idx].reset_index(drop=True)
        split_mode = "internal_group_shuffle_split"
    target_stats: dict[str, Any] = {}
    metrics: dict[str, Any] = {}

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    clip_scale = max(0.05, float(args.delta_clip_scale))
    delta_clips = {
        key: (float(lo) * clip_scale, float(hi) * clip_scale)
        for key, (lo, hi) in DEFAULT_DELTA_CLIPS.items()
    }

    default_params = {
        "objective": "regression_l1",
        "boosting_type": "gbdt",
        "num_leaves": int(args.num_leaves),
        "learning_rate": float(args.learning_rate),
        "feature_fraction": float(args.feature_fraction),
        "bagging_fraction": float(args.bagging_fraction),
        "bagging_freq": 1,
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "max_depth": int(args.max_depth),
        "lambda_l1": float(args.lambda_l1),
        "lambda_l2": float(args.lambda_l2),
        "verbosity": -1,
    }

    trained_targets: list[str] = []
    for target in RESIDUAL_TARGET_NAMES:
        if target not in df_train.columns or target not in df_valid.columns:
            continue
        lo, hi = delta_clips.get(target, (-500.0, 500.0))
        y_train = pd.to_numeric(df_train[target], errors="coerce").fillna(0.0).clip(lower=float(lo), upper=float(hi))
        y_valid = pd.to_numeric(df_valid[target], errors="coerce").fillna(0.0).clip(lower=float(lo), upper=float(hi))

        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=list(RESIDUAL_CATEGORICAL_FEATURES),
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_set,
            categorical_feature=list(RESIDUAL_CATEGORICAL_FEATURES),
            free_raw_data=False,
        )

        callbacks = []
        if int(args.early_stopping_rounds) > 0 and len(X_valid) > 5:
            callbacks.append(lgb.early_stopping(int(args.early_stopping_rounds), verbose=False))

        booster = lgb.train(
            params=dict(default_params),
            train_set=train_set,
            num_boost_round=int(args.num_boost_round),
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        pred_valid = booster.predict(X_valid)
        mae_before = float(mean_absolute_error(y_valid, [0.0] * len(y_valid)))
        mae_after = float(mean_absolute_error(y_valid, pred_valid))
        improve = float(mae_before - mae_after)
        metrics[target] = {
            "mae_before": mae_before,
            "mae_after": mae_after,
            "mae_improvement": improve,
            "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
        }
        target_stats[target] = {
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "clip": [float(lo), float(hi)],
            "mean_abs_delta": float(y_train.abs().mean()),
        }
        booster.save_model(os.path.join(out_dir, f"model_{target}.txt"))
        trained_targets.append(target)
    if not trained_targets:
        raise SystemExit(f"No target columns found. expected any of: {', '.join(RESIDUAL_TARGET_NAMES)}")

    feature_schema = {
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "categorical_features": list(RESIDUAL_CATEGORICAL_FEATURES),
        "target_names": list(RESIDUAL_TARGET_NAMES),
    }
    with open(os.path.join(out_dir, "feature_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(feature_schema, handle, ensure_ascii=False, indent=2)

    meta = {
        "backend": "boundary_residual_lightgbm_v1",
        "model_version": "v1",
        "dataset_csv": csv_path,
        "valid_csv": valid_csv_path,
        "split_mode": split_mode,
        "rows": int(len(df)),
        "train_rows": int(len(df_train)),
        "valid_rows": int(len(df_valid)),
        "group_column": str(args.group_column or ""),
        "feature_version": "boundary_residual_v1",
        "lgbm_params": {k: v for k, v in default_params.items() if k != "verbosity"},
        "delta_clip_scale": float(clip_scale),
        "delta_clips": {k: [float(v[0]), float(v[1])] for k, v in delta_clips.items()},
        "metrics": metrics,
        "target_stats": target_stats,
        "trained_targets": trained_targets,
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    if bool(args.fit_stage1_role_bias):
        role_col = str(args.stage1_role_column or "role").strip()
        if role_col in df.columns:
            stage1_targets: dict[str, Any] = {}
            min_role_rows = max(1, int(args.stage1_min_role_rows))
            shrink = max(0.0, float(args.stage1_shrink))
            role_series = df[role_col].fillna("other").astype(str).map(lambda x: normalize_role(x))
            for target in RESIDUAL_TARGET_NAMES:
                if target not in df.columns:
                    continue
                target_raw = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
                lo, hi = delta_clips.get(target, (-500.0, 500.0))
                target_vals = target_raw.clip(lower=float(lo), upper=float(hi))
                global_median = float(target_vals.median())
                by_role = {}
                role_counts = {}
                for role_name in sorted(set(role_series.tolist())):
                    role_norm = normalize_role(role_name)
                    mask = role_series == role_name
                    count = int(mask.sum())
                    if count < min_role_rows:
                        continue
                    role_median = float(target_vals[mask].median())
                    role_val = (role_median * float(count) + global_median * float(shrink)) / max(1.0, float(count) + float(shrink))
                    by_role[role_norm] = float(role_val)
                    role_counts[role_norm] = int(count)
                stage1_targets[target] = {
                    "default": float(global_median),
                    "by_role": by_role,
                    "role_counts": role_counts,
                    "clip": [float(lo), float(hi)],
                }
            stage1_payload = {
                "backend": "boundary_residual_stage1_role_bias_v1",
                "version": "v1",
                "targets": stage1_targets,
                "summary": {
                    "rows": int(len(df)),
                    "role_column": role_col,
                    "min_role_rows": int(min_role_rows),
                    "shrink": float(shrink),
                },
            }
            with open(os.path.join(out_dir, "stage1_role_bias.json"), "w", encoding="utf-8") as handle:
                json.dump(stage1_payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
