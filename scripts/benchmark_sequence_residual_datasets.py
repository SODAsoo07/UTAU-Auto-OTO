from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.train_sequence_residual_baseline import (
    TARGETS,
    _apply_matrix,
    _build_categories,
    _fit_ridge,
    _load_rows,
    _matrix,
    _metrics,
    _predict,
    _prediction_to_ms,
    _resolve_targets,
    _safe_float,
    _target_values,
    _target_values_ms,
)

import numpy as np


def _group_rows(rows: Sequence[Dict[str, object]], group_column: str) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_column, "") or "").strip()
        if key:
            grouped[key].append(row)
    return dict(grouped)


def _model_confidence(row: Dict[str, object], target: str) -> float:
    conf = max(0.0, min(1.0, _safe_float(row.get("anchor_confidence", 0.0), 0.0)))
    strategy = str(row.get("anchor_match_strategy", "") or "").strip().lower()
    if strategy in {"fallback_full_file", "file_order_ratio"}:
        conf = min(conf, 0.34)
    elif strategy == "endbreath_tail":
        conf = min(1.0, max(conf, 0.72))
    if target == "delta_cutoff" and str(row.get("cutoff_convention", "") or "").strip().lower() == "mixed_auto":
        conf *= 0.78
    if target in {"delta_offset", "delta_cutoff"} and _safe_float(row.get("anchor_mid_error_ms", 0.0)) > 850.0:
        conf *= 0.70
    return float(max(0.0, min(1.0, conf)))


def _apply_confidence_gate(
    rows: Sequence[Dict[str, object]],
    target: str,
    predictions_ms: np.ndarray,
    *,
    min_confidence: float,
    gate_targets: str,
) -> tuple[np.ndarray, float]:
    pred = np.asarray(predictions_ms, dtype=np.float64).copy()
    gate_mode = str(gate_targets or "all").strip().lower()
    if gate_mode in {"none", "off", "0"}:
        return pred, 1.0
    if gate_mode in {"offset_cutoff", "anchor", "absolute"} and target not in {"delta_offset", "delta_cutoff"}:
        return pred, 1.0
    threshold = float(max(0.0, min(1.0, min_confidence)))
    if threshold <= 0.0:
        return pred, 1.0
    applied = 0
    for idx, row in enumerate(rows):
        if _model_confidence(row, target) >= threshold:
            applied += 1
            continue
        pred[idx] = 0.0
    ratio = float(applied) / float(max(1, len(rows)))
    return pred, ratio


def _train_eval(
    train_rows: Sequence[Dict[str, object]],
    test_rows: Sequence[Dict[str, object]],
    *,
    ridge: float,
    max_categories: int,
    target_mode: str,
    target_scope: str,
    gate_min_confidence: float,
    gate_targets: str,
) -> Dict[str, object]:
    categories = _build_categories(train_rows, max_categories)
    x_train, feature_names, scalers = _matrix(train_rows, categories)
    x_test = _apply_matrix(test_rows, feature_names, scalers)
    out: Dict[str, object] = {}
    for target in _resolve_targets(target_scope):
        y_train = _target_values(train_rows, target, target_mode)
        y_test_ms = _target_values_ms(test_rows, target)
        y_train_ms = _target_values_ms(train_rows, target)
        weights = _fit_ridge(x_train, y_train, ridge=float(ridge))
        train_pred = _prediction_to_ms(train_rows, target, target_mode, _predict(x_train, weights))
        test_pred = _prediction_to_ms(test_rows, target, target_mode, _predict(x_test, weights))
        test_pred_gated, applied_ratio = _apply_confidence_gate(
            test_rows,
            target,
            test_pred,
            min_confidence=gate_min_confidence,
            gate_targets=gate_targets,
        )
        base_pred = np.zeros_like(y_test_ms, dtype=np.float64)
        out[target] = {
            "train": _metrics(y_train_ms, train_pred),
            "base": _metrics(y_test_ms, base_pred),
            "test": _metrics(y_test_ms, test_pred),
            "test_gated": _metrics(y_test_ms, test_pred_gated),
            "gated_applied_ratio": applied_ratio,
        }
    return out


def _mean_metric(folds: Sequence[Dict[str, object]], eval_key: str = "test") -> Dict[str, object]:
    out: Dict[str, object] = {}
    for target in TARGETS:
        target_folds = [
            fold["targets"][target][eval_key]
            for fold in folds
            if target in fold.get("targets", {}) and eval_key in fold["targets"][target]
        ]
        if not target_folds:
            continue
        out[target] = {
            "mae": float(sum(float(x.get("mae", 0.0)) for x in target_folds) / len(target_folds)),
            "median_abs": float(sum(float(x.get("median_abs", 0.0)) for x in target_folds) / len(target_folds)),
            "p90_abs": float(sum(float(x.get("p90_abs", 0.0)) for x in target_folds) / len(target_folds)),
            "max_abs": float(max(float(x.get("max_abs", 0.0)) for x in target_folds)),
            "folds": len(target_folds),
        }
    return out


def benchmark_datasets(
    *,
    dataset_paths: Sequence[str],
    out_dir: str,
    group_column: str = "voicebank_id",
    ridge: float = 20.0,
    max_categories: int = 32,
    target_mode: str = "raw",
    target_scope: str = "all",
    gate_min_confidence: float = 0.0,
    gate_targets: str = "all",
) -> Dict[str, object]:
    rows = _load_rows(dataset_paths)
    if not rows:
        raise ValueError("no usable rows")
    by_format: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        fmt = str(row.get("format_type", "") or "unknown").strip().lower() or "unknown"
        by_format[fmt].append(row)

    summary: Dict[str, object] = {
        "schema": "sequence_residual_dataset_benchmark_v1",
        "dataset_paths": [os.path.abspath(path) for path in dataset_paths],
        "ridge": float(ridge),
        "target_mode": str(target_mode or "raw"),
        "target_scope": str(target_scope or "all"),
        "gate_min_confidence": float(gate_min_confidence),
        "gate_targets": str(gate_targets or "all"),
        "group_column": group_column,
        "formats": {},
    }
    for fmt, fmt_rows in sorted(by_format.items()):
        grouped = _group_rows(fmt_rows, group_column)
        folds = []
        for holdout_key, test_rows in sorted(grouped.items()):
            train_rows = [row for key, vals in grouped.items() if key != holdout_key for row in vals]
            if not train_rows or not test_rows:
                continue
            folds.append(
                {
                    "holdout": holdout_key,
                    "train_count": len(train_rows),
                    "test_count": len(test_rows),
                    "targets": _train_eval(
                        train_rows,
                        test_rows,
                        ridge=float(ridge),
                        max_categories=int(max_categories),
                        target_mode=target_mode,
                        target_scope=target_scope,
                        gate_min_confidence=float(gate_min_confidence),
                        gate_targets=gate_targets,
                    ),
                }
            )
        summary["formats"][fmt] = {
            "row_count": len(fmt_rows),
            "group_count": len(grouped),
            "group_counts": {key: len(vals) for key, vals in sorted(grouped.items())},
            "folds": folds,
            "mean_test": _mean_metric(folds),
            "mean_base": _mean_metric(folds, eval_key="base"),
            "mean_test_gated": _mean_metric(folds, eval_key="test_gated"),
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sequence_residual_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return {"summary_path": out_path, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-voicebank-out benchmark for sequence residual JSONL datasets.")
    parser.add_argument("--dataset", action="append", required=True, help="sequence_residual_rows.jsonl path. Can be repeated.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group-column", default="voicebank_id")
    parser.add_argument("--ridge", type=float, default=20.0)
    parser.add_argument("--max-categories", type=int, default=32)
    parser.add_argument("--target-mode", choices=["raw", "normalized", "hybrid"], default="raw")
    parser.add_argument("--target-scope", choices=["all", "pre_cons_ovl"], default="all")
    parser.add_argument("--gate-min-confidence", type=float, default=0.0)
    parser.add_argument("--gate-targets", choices=["all", "offset_cutoff", "none"], default="all")
    args = parser.parse_args()

    result = benchmark_datasets(
        dataset_paths=args.dataset,
        out_dir=args.out_dir,
        group_column=args.group_column,
        ridge=float(args.ridge),
        max_categories=int(args.max_categories),
        target_mode=args.target_mode,
        target_scope=args.target_scope,
        gate_min_confidence=float(args.gate_min_confidence),
        gate_targets=args.gate_targets,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
