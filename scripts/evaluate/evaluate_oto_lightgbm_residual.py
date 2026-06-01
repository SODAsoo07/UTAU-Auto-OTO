"""Evaluate an OTO LightGBM residual model on an external heldout CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.oto_ml_features import TARGET_NAMES, get_delta_clip_limits
from core.oto_ml_lightgbm import load_lightgbm_bundle, predict_lightgbm_deltas_batch

DEFAULT_GROUP_FIELDS = ("case_id", "alias_family", "vc_right_class")
RUNTIME_SAFETY_TARGET_EXCLUSIONS: dict[str, dict[str, frozenset[str]]] = {
    "alias_family": {
        "policy_breath": frozenset(TARGET_NAMES),
        "terminal_v_dash": frozenset(TARGET_NAMES),
        "cv_head": frozenset({"delta_cutoff"}),
        "spaced": frozenset({"delta_cons"}),
        "v": frozenset({"delta_offset", "delta_cutoff"}),
        "vv": frozenset({"delta_ovl"}),
    },
    "vc_right_class": {
        "terminal": frozenset({"delta_ovl"}),
    },
}


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip_delta(language: str, target: str, value: float) -> float:
    clip = get_delta_clip_limits(language).get(target)
    if not clip:
        return float(value)
    lo, hi = float(clip[0]), float(clip[1])
    return max(lo, min(hi, float(value)))


def _stats(errors: Iterable[float]) -> dict[str, float | int]:
    vals = [abs(float(value)) for value in errors]
    if not vals:
        return {"n": 0, "mae_ms": 0.0, "p90_ms": 0.0}
    ordered = sorted(vals)
    return {
        "n": len(vals),
        "mae_ms": round(sum(vals) / len(vals), 3),
        "p90_ms": round(ordered[int(0.9 * (len(ordered) - 1))], 3),
    }


def _target_summary(rows: list[dict], preds: list[dict], *, language: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for target in TARGET_NAMES:
        baseline_errors: list[float] = []
        model_errors: list[float] = []
        worse_rows = 0
        changed_rows = 0
        for row, pred in zip(rows, preds):
            truth = _clip_delta(language, target, _float(row.get(target)))
            pred_value = _float(pred.get(target))
            baseline_error = abs(truth)
            model_error = abs(pred_value - truth)
            baseline_errors.append(baseline_error)
            model_errors.append(model_error)
            if abs(pred_value) >= 1e-6:
                changed_rows += 1
            if model_error > baseline_error + 1e-6:
                worse_rows += 1
        baseline = _stats(baseline_errors)
        model = _stats(model_errors)
        baseline_mae = float(baseline.get("mae_ms", 0.0))
        model_mae = float(model.get("mae_ms", 0.0))
        improvement = baseline_mae - model_mae
        out[target] = {
            "n": int(model.get("n", 0)),
            "baseline_mae_ms": baseline_mae,
            "model_mae_ms": model_mae,
            "improvement_mae_ms": round(improvement, 3),
            "improvement_pct": round((100.0 * improvement / baseline_mae), 2) if baseline_mae > 0.0 else 0.0,
            "baseline_p90_ms": float(baseline.get("p90_ms", 0.0)),
            "model_p90_ms": float(model.get("p90_ms", 0.0)),
            "changed_rows": int(changed_rows),
            "worse_rows": int(worse_rows),
            "worse_rate": round(worse_rows / max(1, len(rows)), 4),
        }
    return out


def apply_runtime_safety_target_exclusions(
    rows: list[dict],
    preds: list[dict],
) -> tuple[list[dict], dict[str, object]]:
    gated_preds: list[dict] = [dict(pred) for pred in preds]
    excluded_row_targets: dict[str, int] = defaultdict(int)
    changed_values = 0
    for index, row in enumerate(rows):
        if index >= len(gated_preds):
            break
        pred = gated_preds[index]
        for field, bucket_targets in RUNTIME_SAFETY_TARGET_EXCLUSIONS.items():
            bucket = str(row.get(field) or "").strip()
            targets = bucket_targets.get(bucket)
            if not targets:
                continue
            for target in sorted(targets):
                if abs(_float(pred.get(target))) >= 1e-6:
                    changed_values += 1
                pred[target] = 0.0
                excluded_row_targets[f"{field}:{bucket}:{target}"] += 1
    return gated_preds, {
        "enabled": True,
        "excluded_values_changed": int(changed_values),
        "excluded_row_targets": dict(sorted(excluded_row_targets.items())),
    }


def summarize_predictions(
    rows: list[dict],
    preds: list[dict],
    *,
    language: str,
    group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS,
    min_group_n: int = 3,
) -> dict:
    overall = _target_summary(rows, preds, language=language)
    by_group: dict[str, dict[str, dict[str, object]]] = {}
    failed_groups: list[dict[str, object]] = []
    for field in group_fields:
        grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for row, pred in zip(rows, preds):
            key = str(row.get(field) or "")
            if key:
                grouped[key].append((row, pred))
        field_payload: dict[str, dict[str, object]] = {}
        for key, pairs in sorted(grouped.items()):
            if len(pairs) < int(min_group_n):
                continue
            group_rows = [row for row, _pred in pairs]
            group_preds = [pred for _row, pred in pairs]
            group_targets = _target_summary(group_rows, group_preds, language=language)
            field_payload[key] = {
                "n": len(pairs),
                "targets": group_targets,
            }
            for target, stats in group_targets.items():
                if float(stats.get("model_mae_ms", 0.0)) >= float(stats.get("baseline_mae_ms", 0.0)):
                    failed_groups.append(
                        {
                            "field": field,
                            "bucket": key,
                            "target": target,
                            "n": len(pairs),
                            "baseline_mae_ms": float(stats.get("baseline_mae_ms", 0.0)),
                            "model_mae_ms": float(stats.get("model_mae_ms", 0.0)),
                        }
                    )
        by_group[field] = field_payload
    failed_targets = [
        target
        for target, stats in overall.items()
        if float(stats.get("model_mae_ms", 0.0)) >= float(stats.get("baseline_mae_ms", 0.0))
    ]
    if failed_targets:
        recommendation = "do_not_enable"
    elif failed_groups:
        recommendation = "candidate_for_gated_postprocess_with_role_exclusions"
    else:
        recommendation = "candidate_for_gated_postprocess"
    report = {
        "rows": len(rows),
        "language": language,
        "targets": overall,
        "by_group": by_group,
        "failed_targets": failed_targets,
        "failed_groups": sorted(
            failed_groups,
            key=lambda item: (
                str(item["field"]),
                str(item["bucket"]),
                str(item["target"]),
            ),
        ),
        "recommended_use": recommendation,
    }
    gated_preds, safety_report = apply_runtime_safety_target_exclusions(rows, preds)
    gated_targets = _target_summary(rows, gated_preds, language=language)
    safety_report["targets"] = gated_targets
    safety_report["strictly_worse_targets"] = [
        target
        for target, stats in gated_targets.items()
        if float(stats.get("model_mae_ms", 0.0)) > float(stats.get("baseline_mae_ms", 0.0)) + 1e-6
    ]
    report["runtime_safety_target_exclusions"] = safety_report
    return report


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", required=True, help="Heldout residual CSV")
    parser.add_argument("--language", default="japanese")
    parser.add_argument("--group-fields", default=",".join(DEFAULT_GROUP_FIELDS))
    parser.add_argument("--min-group-n", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = _read_csv(args.dataset)
    bundle = load_lightgbm_bundle(args.model_dir)
    preds = predict_lightgbm_deltas_batch(bundle, rows)
    group_fields = [item.strip() for item in str(args.group_fields).split(",") if item.strip()]
    report = {
        "model_dir": os.path.abspath(args.model_dir),
        "dataset": os.path.abspath(args.dataset),
        **summarize_predictions(
            rows,
            preds,
            language=str(args.language or "").strip().lower(),
            group_fields=group_fields,
            min_group_n=int(args.min_group_n),
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
