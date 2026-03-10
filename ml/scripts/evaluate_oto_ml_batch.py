from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_lightgbm import (
    evaluate_lightgbm_bundle,
    evaluate_lightgbm_selector_bundle,
)
from core.oto_ml_policy import default_training_filters
from core.oto_ml_selector import build_selector_dataset_csv_from_delta_dataset
from core.runtime_encoding import bootstrap_utf8_runtime


SUPPORTED_TARGETS = [
    ("korean", "cv"),
    ("korean", "cvc"),
    ("korean", "cvvc"),
    ("korean", "vcv"),
    ("japanese", "cv"),
    ("japanese", "cvvc"),
    ("japanese", "vcv"),
]


def _dataset_path(workspace_root: str, language: str, format_type: str) -> str:
    return os.path.join(workspace_root, "datasets", language, f"dataset_{language}_{format_type}.csv")


def _selector_dataset_path(workspace_root: str, language: str, format_type: str) -> str:
    return os.path.join(workspace_root, "datasets", language, f"selector_dataset_{language}_{format_type}.csv")


def _model_dir_path(workspace_root: str, language: str, format_type: str) -> str:
    return os.path.join(workspace_root, "models", f"{language}_{format_type}_bundle")


def _report_path(reports_root: str, kind: str, language: str, format_type: str) -> str:
    return os.path.join(reports_root, f"{kind}_{language}_{format_type}.json")


def _has_selector_bundle(model_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(model_dir, "selector_model.txt"))
        and os.path.exists(os.path.join(model_dir, "selector_meta.json"))
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _discover_targets(workspace_root: str) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for language, format_type in SUPPORTED_TARGETS:
        dataset = _dataset_path(workspace_root, language, format_type)
        model_dir = _model_dir_path(workspace_root, language, format_type)
        if os.path.exists(dataset) and os.path.isdir(model_dir):
            jobs.append(
                {
                    "language": language,
                    "format_type": format_type,
                    "dataset": dataset,
                    "model_dir": model_dir,
                    "selector_dataset": _selector_dataset_path(workspace_root, language, format_type),
                }
            )

    # Fallback: if custom bundle names exist, try to infer from dataset names.
    if jobs:
        return jobs
    for dataset in glob.glob(os.path.join(workspace_root, "datasets", "*", "dataset_*.csv")):
        stem = os.path.splitext(os.path.basename(dataset))[0]
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        language = parts[1].strip().lower()
        format_type = parts[2].strip().lower()
        model_dir = _model_dir_path(workspace_root, language, format_type)
        if os.path.isdir(model_dir):
            jobs.append(
                {
                    "language": language,
                    "format_type": format_type,
                    "dataset": dataset,
                    "model_dir": model_dir,
                    "selector_dataset": _selector_dataset_path(workspace_root, language, format_type),
                }
            )
    return jobs


def _render_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "(no rows)"
    widths: Dict[str, int] = {}
    for col in columns:
        widths[col] = max(len(col), *(len(str(row.get(col, ""))) for row in rows))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    body = [" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, sep, *body])


def _delta_summary_row(language: str, format_type: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    targets = summary.get("targets") or {}
    offset = targets.get("delta_offset") or {}
    cons = targets.get("delta_cons") or {}
    cutoff = targets.get("delta_cutoff") or {}
    pre = targets.get("delta_pre") or {}
    ovl = targets.get("delta_ovl") or {}
    return {
        "lang": language,
        "format": format_type,
        "rows": int(summary.get("rows", 0) or 0),
        "off": f"{_safe_float(offset.get('baseline_mae')):.2f}->{_safe_float(offset.get('model_mae')):.2f}",
        "cons": f"{_safe_float(cons.get('baseline_mae')):.2f}->{_safe_float(cons.get('model_mae')):.2f}",
        "cut": f"{_safe_float(cutoff.get('baseline_mae')):.2f}->{_safe_float(cutoff.get('model_mae')):.2f}",
        "pre": f"{_safe_float(pre.get('baseline_mae')):.2f}->{_safe_float(pre.get('model_mae')):.2f}",
        "ovl": f"{_safe_float(ovl.get('baseline_mae')):.2f}->{_safe_float(ovl.get('model_mae')):.2f}",
    }


def _selector_summary_row(language: str, format_type: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lang": language,
        "format": format_type,
        "rows": int(summary.get("rows", 0) or 0),
        "groups": int(summary.get("groups", 0) or 0),
        "top1": f"{_safe_float(summary.get('top1_baseline')):.4f}->{_safe_float(summary.get('top1_model')):.4f}",
        "score_mae": f"{_safe_float(summary.get('score_mae')):.4f}" if "score_mae" in summary else "",
        "objective": str(summary.get("objective", "") or ""),
    }


def run_batch_evaluation(
    workspace_root: str,
    reports_root: str,
    *,
    build_selector_if_missing: bool = True,
    evaluate_selector: bool = True,
) -> Dict[str, Any]:
    jobs = _discover_targets(workspace_root)
    results: Dict[str, Any] = {
        "workspace_root": os.path.abspath(workspace_root),
        "reports_root": os.path.abspath(reports_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": [],
        "delta_table": [],
        "selector_table": [],
    }
    os.makedirs(reports_root, exist_ok=True)

    for job in jobs:
        language = job["language"]
        format_type = job["format_type"]
        dataset = job["dataset"]
        model_dir = job["model_dir"]
        selector_dataset = job["selector_dataset"]
        item: Dict[str, Any] = {
            "language": language,
            "format_type": format_type,
            "dataset": os.path.abspath(dataset),
            "model_dir": os.path.abspath(model_dir),
        }

        delta_summary = evaluate_lightgbm_bundle(
            model_dir=model_dir,
            dataset_csv=dataset,
            language=language,
            format_type=format_type,
        )
        delta_report = _report_path(reports_root, "delta_eval", language, format_type)
        with open(delta_report, "w", encoding="utf-8") as f:
            json.dump(delta_summary, f, ensure_ascii=False, indent=2)
        item["delta_report"] = os.path.abspath(delta_report)
        item["delta_summary"] = delta_summary
        results["delta_table"].append(_delta_summary_row(language, format_type, delta_summary))

        if evaluate_selector and _has_selector_bundle(model_dir):
            if build_selector_if_missing and not os.path.exists(selector_dataset):
                defaults = default_training_filters(language, format_type)
                build_selector_dataset_csv_from_delta_dataset(
                    delta_dataset_csv=dataset,
                    out_csv=selector_dataset,
                    language=language,
                    format_type=format_type,
                    require_train_keep=bool(defaults["require_train_keep"]),
                    min_mapping_confidence=float(defaults["min_mapping_confidence"]),
                    exclude_nuclei_fallback=bool(defaults["exclude_nuclei_fallback"]),
                    use_pseudo_labels=bool(defaults["use_pseudo_labels"]),
                )
            if os.path.exists(selector_dataset):
                selector_summary = evaluate_lightgbm_selector_bundle(
                    model_dir=model_dir,
                    selector_dataset_csv=selector_dataset,
                    language=language,
                    format_type=format_type,
                )
                selector_report = _report_path(reports_root, "selector_eval", language, format_type)
                with open(selector_report, "w", encoding="utf-8") as f:
                    json.dump(selector_summary, f, ensure_ascii=False, indent=2)
                item["selector_dataset"] = os.path.abspath(selector_dataset)
                item["selector_report"] = os.path.abspath(selector_report)
                item["selector_summary"] = selector_summary
                results["selector_table"].append(_selector_summary_row(language, format_type, selector_summary))

        results["jobs"].append(item)

    return results


def _write_csv_summary(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Batch-evaluate OTO ML bundles and print summary tables.")
    ap.add_argument("--workspace-root", required=True, help="Workspace root containing datasets/, models/, reports/")
    ap.add_argument("--reports-root", default="", help="Output directory for per-model evaluation reports")
    ap.add_argument("--json-out", default="", help="Optional combined JSON summary path")
    ap.add_argument("--delta-csv-out", default="", help="Optional delta summary CSV path")
    ap.add_argument("--selector-csv-out", default="", help="Optional selector summary CSV path")
    ap.add_argument("--no-selector", action="store_true", help="Skip selector evaluation")
    ap.add_argument("--no-build-selector", action="store_true", help="Do not auto-build selector dataset when missing")
    args = ap.parse_args()

    reports_root = args.reports_root or os.path.join(
        args.workspace_root,
        "reports",
        f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    results = run_batch_evaluation(
        args.workspace_root,
        reports_root,
        build_selector_if_missing=not args.no_build_selector,
        evaluate_selector=not args.no_selector,
    )

    print("[Delta]")
    delta_cols = ["lang", "format", "rows", "off", "cons", "cut", "pre", "ovl"]
    print(_render_table(results["delta_table"], delta_cols))
    print()
    print("[Selector]")
    selector_cols = ["lang", "format", "rows", "groups", "top1", "score_mae", "objective"]
    print(_render_table(results["selector_table"], selector_cols))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    if args.delta_csv_out:
        _write_csv_summary(args.delta_csv_out, results["delta_table"], delta_cols)
    if args.selector_csv_out:
        _write_csv_summary(args.selector_csv_out, results["selector_table"], selector_cols)


if __name__ == "__main__":
    main()
