from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .types import FRAME_LABELS, SuccessCriteria


def frame_classification_metrics(
    y_true: Sequence[int | str],
    y_pred: Sequence[int | str],
    *,
    labels: Sequence[str] = FRAME_LABELS,
    ignore_label: int | str = -100,
) -> dict:
    label_list = list(labels)
    true_labels = [_normalize_label(value, label_list) for value in y_true]
    pred_labels = [_normalize_label(value, label_list) for value in y_pred]
    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in label_list:
        tp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred != label)
        support = sum(1 for truth in true_labels if truth == label)
        if label == str(ignore_label) or support == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
        }
        f1_values.append(float(f1))
    return {
        "per_label": per_label,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "frame_count": int(len(true_labels)),
    }


def boundary_error_metrics(
    reference_ms: Sequence[float],
    predicted_ms: Sequence[float],
    *,
    tolerance_ms: float = 100.0,
) -> dict:
    refs = [float(value) for value in reference_ms]
    preds = [float(value) for value in predicted_ms]
    unused = set(range(len(preds)))
    errors: list[float] = []
    missed = 0
    for ref in refs:
        if not unused:
            missed += 1
            continue
        best_idx = min(unused, key=lambda idx: abs(preds[idx] - ref))
        err = abs(preds[best_idx] - ref)
        if err <= tolerance_ms:
            errors.append(float(err))
            unused.remove(best_idx)
        else:
            missed += 1
    false_positive = len(unused)
    return {
        "count": int(len(errors)),
        "reference_count": int(len(refs)),
        "predicted_count": int(len(preds)),
        "missed": int(missed),
        "false_positive": int(false_positive),
        "median_error_ms": _percentile(errors, 50.0),
        "p90_error_ms": _percentile(errors, 90.0),
        "max_error_ms": max(errors) if errors else None,
    }


def row_failure_metrics(row_results: Sequence[dict]) -> dict:
    total = len(row_results)
    if total == 0:
        return {"rows": 0, "catastrophic_row_mismatch_rate": 0.0, "hard_failure_rate": 0.0}
    catastrophic = sum(1 for row in row_results if bool(row.get("catastrophic")))
    hard_failure = sum(1 for row in row_results if bool(row.get("hard_failure")))
    return {
        "rows": int(total),
        "catastrophic_row_mismatch_rate": catastrophic / total,
        "hard_failure_rate": hard_failure / total,
    }


def gate_report(metrics: dict, criteria: SuccessCriteria | None = None) -> dict:
    crit = criteria or SuccessCriteria()
    checks = {
        "frame_macro_f1": float(metrics.get("frame_macro_f1", 0.0)) >= crit.frame_macro_f1_min,
        "cv_boundary_median_error_ms": float(metrics.get("cv_boundary_median_error_ms", 1e9))
        <= crit.cv_boundary_median_error_ms_max,
        "cv_boundary_p90_error_ms": float(metrics.get("cv_boundary_p90_error_ms", 1e9))
        <= crit.cv_boundary_p90_error_ms_max,
        "catastrophic_row_mismatch_rate": float(metrics.get("catastrophic_row_mismatch_rate", 1.0))
        <= crit.catastrophic_row_mismatch_rate_max,
        "hard_failure_rate": float(metrics.get("hard_failure_rate", 1.0)) <= crit.hard_failure_rate_max,
    }
    return {"passed": all(checks.values()), "checks": checks, "criteria": asdict(crit)}


def _normalize_label(value: int | str, labels: Sequence[str]) -> str:
    if isinstance(value, str):
        return value
    if int(value) < 0:
        return str(value)
    return labels[int(value)]


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))
