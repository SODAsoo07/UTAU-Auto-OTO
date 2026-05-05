from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, List, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


TARGETS = ["delta_offset", "delta_pre", "delta_cons", "delta_cutoff", "delta_ovl"]
PRE_CONS_OVL_TARGETS = ["delta_pre", "delta_cons", "delta_ovl"]

CATEGORICAL_FEATURES = [
    "language",
    "format_type",
    "alias_type",
    "alias_group",
    "sequence_source",
    "sequence_soft_lock",
    "sequence_cvn_backend",
    "anchor_match_strategy",
]

NUMERIC_FEATURES = [
    "row_index_in_wav",
    "row_ratio_in_wav",
    "file_row_count",
    "sequence_word_count",
    "anchor_index",
    "anchor_start_ms",
    "anchor_end_ms",
    "anchor_mid_ms",
    "anchor_len_ms",
    "anchor_confidence",
    "active_start_ms",
    "active_end_ms",
    "active_len_ms",
    "next_anchor_start_ms",
    "next_anchor_gap_ms",
    "wav_duration_ms",
    "base_offset",
    "base_pre",
    "base_cons",
    "base_cutoff_abs",
    "base_ovl",
    "rms_local_mean",
    "rms_local_max",
    "rms_local_std",
    "rms_onset_mean",
    "rms_onset_max",
    "rms_tail_mean",
    "rms_tail_min",
    "zcr_onset_mean",
    "zcr_onset_max",
    "hf_onset_mean",
    "hf_onset_max",
    "flux_onset_mean",
    "flux_onset_max",
    "label_silence_ratio",
    "label_vowel_ratio",
    "label_onset_ratio",
    "label_breath_ratio",
]


def _stable_hash(text: str) -> int:
    digest = hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _load_rows(paths: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if bool(row.get("skip_training", False)):
                    continue
                rows.append(row)
    return rows


def _resolve_targets(scope: str) -> List[str]:
    scope_norm = str(scope or "all").strip().lower()
    if scope_norm in {"pre_cons_ovl", "pco", "relative"}:
        return list(PRE_CONS_OVL_TARGETS)
    return list(TARGETS)


def _normalized_target_name(target: str) -> str:
    if target in {"delta_offset", "delta_cutoff"}:
        return f"{target}_active_ratio"
    return f"{target}_anchor_ratio"


def _target_mode_for_target(mode: str, target: str) -> str:
    mode_norm = str(mode or "raw").strip().lower()
    if mode_norm == "hybrid":
        return "normalized" if target in {"delta_offset", "delta_cutoff"} else "raw"
    return mode_norm


def _target_scale(row: Dict[str, object], target: str, mode: str) -> float:
    if _target_mode_for_target(mode, target) != "normalized":
        return 1.0
    if target in {"delta_offset", "delta_cutoff"}:
        return max(1.0, _safe_float(row.get("active_len_ms", 1.0), 1.0))
    return max(1.0, _safe_float(row.get("anchor_len_ms", 1.0), 1.0))


def _target_values(rows: Sequence[Dict[str, object]], target: str, mode: str) -> np.ndarray:
    if _target_mode_for_target(mode, target) == "normalized":
        name = _normalized_target_name(target)
        return np.asarray([_safe_float(row.get(name, 0.0)) for row in rows], dtype=np.float64)
    return np.asarray([_safe_float(row.get(target, 0.0)) for row in rows], dtype=np.float64)


def _target_values_ms(rows: Sequence[Dict[str, object]], target: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(target, 0.0)) for row in rows], dtype=np.float64)


def _prediction_to_ms(rows: Sequence[Dict[str, object]], target: str, mode: str, pred: np.ndarray) -> np.ndarray:
    values = np.asarray(pred, dtype=np.float64)
    if _target_mode_for_target(mode, target) != "normalized":
        return values
    scales = np.asarray([_target_scale(row, target, mode) for row in rows], dtype=np.float64)
    return values * scales


def _split_rows(rows: Sequence[Dict[str, object]], test_ratio: float) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_voicebank = defaultdict(list)
    for row in rows:
        by_voicebank[str(row.get("voicebank_id", "") or "")].append(row)
    voicebanks = sorted(by_voicebank)
    if len(voicebanks) >= 2:
        test_count = max(1, int(round(len(voicebanks) * float(test_ratio))))
        test_vbs = set(sorted(voicebanks, key=_stable_hash)[-test_count:])
        train = [row for row in rows if str(row.get("voicebank_id", "") or "") not in test_vbs]
        test = [row for row in rows if str(row.get("voicebank_id", "") or "") in test_vbs]
        return train, test

    wavs = sorted({str(row.get("wav_norm", "") or row.get("wav", "") or "") for row in rows})
    test_count = max(1, int(round(len(wavs) * float(test_ratio)))) if len(wavs) >= 2 else 0
    test_wavs = set(sorted(wavs, key=_stable_hash)[-test_count:])
    train = [row for row in rows if str(row.get("wav_norm", "") or row.get("wav", "") or "") not in test_wavs]
    test = [row for row in rows if str(row.get("wav_norm", "") or row.get("wav", "") or "") in test_wavs]
    if not train or not test:
        return list(rows), []
    return train, test


def _build_categories(rows: Sequence[Dict[str, object]], max_categories: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in CATEGORICAL_FEATURES:
        counter = Counter(str(row.get(name, "") or "") for row in rows)
        values = [value for value, _count in counter.most_common(max(1, int(max_categories))) if value]
        out[name] = values
    return out


def _matrix(rows: Sequence[Dict[str, object]], categories: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[float, float]]]:
    raw_cols: List[List[float]] = []
    names: List[str] = []
    for name in NUMERIC_FEATURES:
        raw_cols.append([_safe_float(row.get(name, 0.0)) for row in rows])
        names.append(name)
    for name in CATEGORICAL_FEATURES:
        values = categories.get(name, [])
        for value in values:
            raw_cols.append([1.0 if str(row.get(name, "") or "") == value else 0.0 for row in rows])
            names.append(f"{name}={value}")
    if not raw_cols:
        return np.zeros((len(rows), 0), dtype=np.float64), [], {}

    x = np.asarray(raw_cols, dtype=np.float64).T
    scalers: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(names):
        col = x[:, idx]
        if "=" in name:
            scalers[name] = (0.0, 1.0)
            continue
        mean = float(np.mean(col)) if col.size else 0.0
        std = float(np.std(col)) if col.size else 1.0
        if std < 1e-8:
            std = 1.0
        x[:, idx] = (col - mean) / std
        scalers[name] = (mean, std)
    return x, names, scalers


def _apply_matrix(rows: Sequence[Dict[str, object]], names: Sequence[str], scalers: Dict[str, Tuple[float, float]]) -> np.ndarray:
    cols: List[List[float]] = []
    for name in names:
        if "=" in name:
            base, expected = name.split("=", 1)
            cols.append([1.0 if str(row.get(base, "") or "") == expected else 0.0 for row in rows])
            continue
        mean, std = scalers.get(name, (0.0, 1.0))
        cols.append([(_safe_float(row.get(name, 0.0)) - mean) / max(1e-8, std) for row in rows])
    if not cols:
        return np.zeros((len(rows), 0), dtype=np.float64)
    return np.asarray(cols, dtype=np.float64).T


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge)
    reg[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return x_aug @ weights


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size <= 0:
        return {"mae": 0.0, "median_abs": 0.0, "p90_abs": 0.0, "max_abs": 0.0, "count": 0}
    err = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    sorted_err = np.sort(err)
    p90_idx = min(sorted_err.size - 1, int(round((sorted_err.size - 1) * 0.90)))
    return {
        "mae": float(np.mean(err)),
        "median_abs": float(median([float(x) for x in err])),
        "p90_abs": float(sorted_err[p90_idx]),
        "max_abs": float(np.max(err)),
        "count": int(y_true.size),
    }


def train_baseline(
    *,
    dataset_paths: Sequence[str],
    out_dir: str,
    ridge: float = 10.0,
    test_ratio: float = 0.20,
    max_categories: int = 32,
    target_mode: str = "raw",
    target_scope: str = "all",
) -> Dict[str, object]:
    rows = _load_rows(dataset_paths)
    if not rows:
        raise ValueError("no usable training rows")
    train_rows, test_rows = _split_rows(rows, test_ratio)
    categories = _build_categories(train_rows, max_categories)
    x_train, feature_names, scalers = _matrix(train_rows, categories)
    x_test = _apply_matrix(test_rows, feature_names, scalers) if test_rows else np.zeros((0, x_train.shape[1]), dtype=np.float64)

    target_names = _resolve_targets(target_scope)
    targets: Dict[str, object] = {}
    for target in target_names:
        y_train = _target_values(train_rows, target, target_mode)
        weights = _fit_ridge(x_train, y_train, ridge=float(ridge))
        train_pred = _predict(x_train, weights)
        train_pred_ms = _prediction_to_ms(train_rows, target, target_mode, train_pred)
        y_train_ms = _target_values_ms(train_rows, target)
        target_meta: Dict[str, object] = {
            "weights": [float(x) for x in weights],
            "train": _metrics(y_train_ms, train_pred_ms),
        }
        if test_rows:
            y_test_ms = _target_values_ms(test_rows, target)
            test_pred = _predict(x_test, weights)
            test_pred_ms = _prediction_to_ms(test_rows, target, target_mode, test_pred)
            target_meta["test"] = _metrics(y_test_ms, test_pred_ms)
        targets[target] = target_meta

    alias_counts = Counter(str(row.get("alias_type", "unknown") or "unknown") for row in rows)
    model = {
        "schema": "sequence_residual_baseline_ridge_v1",
        "dataset_paths": [os.path.abspath(path) for path in dataset_paths],
        "row_count": len(rows),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "ridge": float(ridge),
        "target_mode": str(target_mode or "raw"),
        "target_scope": str(target_scope or "all"),
        "feature_names": feature_names,
        "feature_scalers": {k: [float(v[0]), float(v[1])] for k, v in scalers.items()},
        "categories": categories,
        "targets": targets,
        "alias_type_counts": dict(alias_counts),
    }

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "sequence_residual_baseline_model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2, sort_keys=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics = {
        "model_path": model_path,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "target_mode": str(target_mode or "raw"),
        "target_scope": str(target_scope or "all"),
        "targets": {
            target: {
                key: value
                for key, value in meta.items()
                if key != "weights"
            }
            for target, meta in targets.items()
        },
        "alias_type_counts": dict(alias_counts),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    return {"model_path": model_path, "metrics_path": metrics_path, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight ridge baseline for sequence residual JSONL rows.")
    parser.add_argument("--dataset", action="append", required=True, help="sequence_residual_rows.jsonl path. Can be repeated.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--max-categories", type=int, default=32)
    parser.add_argument("--target-mode", choices=["raw", "normalized", "hybrid"], default="raw")
    parser.add_argument("--target-scope", choices=["all", "pre_cons_ovl"], default="all")
    args = parser.parse_args()

    result = train_baseline(
        dataset_paths=args.dataset,
        out_dir=args.out_dir,
        ridge=float(args.ridge),
        test_ratio=float(args.test_ratio),
        max_categories=int(args.max_categories),
        target_mode=args.target_mode,
        target_scope=args.target_scope,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
