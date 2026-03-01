from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_batch_prepare import (
    _discover_work_items,
    _generate_auto_oto,
    _has_textgrid_files,
    _prepare_lab_and_dict,
    _has_usable_oto_lines,
)
from core.oto_ml_collection import (
    build_datasets_from_candidates,
    discover_training_candidates,
    discover_training_candidates_from_dataset_root,
)
from core.oto_ml_lightgbm import evaluate_lightgbm_bundle, train_lightgbm_bundle
from core.mfa_runner import find_mfa_executable, run_mfa_align


def _safe_print_json(obj: Dict[str, Any]) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _timed(label: str, fn, *args, **kwargs):
    started = time.perf_counter()
    value = fn(*args, **kwargs)
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "seconds": round(elapsed, 4),
        "value": value,
    }


def _summarize_candidate_status(candidates) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for candidate in candidates:
        out[candidate.status] = out.get(candidate.status, 0) + 1
    return out


def _pick_first_unprepared_item(dataset_root: str):
    for item in _discover_work_items(dataset_root):
        tg_dir = os.path.join(item.work_dir, "textgrids_auto")
        auto_oto = os.path.join(item.work_dir, "oto_auto_ml.ini")
        if _has_textgrid_files(tg_dir) and _has_usable_oto_lines(auto_oto):
            continue
        item.tg_dir = tg_dir
        item.auto_oto = auto_oto
        item.dict_path = os.path.join(item.work_dir, "dictionary_auto.txt")
        return item
    return None


def _profile_prepare_one(dataset_root: str) -> Dict[str, Any]:
    item = _pick_first_unprepared_item(dataset_root)
    if item is None:
        return {
            "status": "no_unprepared_items",
            "work_dir": "",
            "language": "",
            "format_type": "",
            "steps": [],
        }
    logs: List[str] = []
    mfa_path = find_mfa_executable() or ""
    if not mfa_path:
        return {
            "status": "missing_mfa",
            "work_dir": item.work_dir,
            "language": item.language,
            "format_type": item.format_type,
            "steps": [],
        }

    steps: List[Dict[str, Any]] = []
    steps.append(_timed("prepare_lab_and_dict", _prepare_lab_and_dict, item, logs))
    os.makedirs(item.tg_dir, exist_ok=True)
    align_timed = _timed(
        "mfa_align",
        run_mfa_align,
        mfa_path=mfa_path,
        wav_folder=item.work_dir,
        dict_path=item.dict_path,
        output_folder=item.tg_dir,
        language=item.language,
        callback=logs.append,
    )
    steps.append(align_timed)
    ok, err = align_timed["value"]
    if ok:
        steps.append(_timed("generate_auto_oto", _generate_auto_oto, item, logs))
        status = "prepared"
    else:
        status = f"align_failed:{err}"
    return {
        "status": status,
        "work_dir": item.work_dir,
        "language": item.language,
        "format_type": item.format_type,
        "steps": [{"label": s["label"], "seconds": s["seconds"]} for s in steps],
        "logs_tail": logs[-20:],
    }


def _profile_training(dataset_csv: str, out_dir: str, language: str, format_type: str) -> Dict[str, Any]:
    train = _timed(
        "train_lightgbm_bundle",
        train_lightgbm_bundle,
        language=language,
        format_type=format_type,
        dataset_csv=dataset_csv,
        out_dir=out_dir,
    )
    evaluate = _timed(
        "evaluate_lightgbm_bundle",
        evaluate_lightgbm_bundle,
        model_dir=out_dir,
        dataset_csv=dataset_csv,
        language=language,
        format_type=format_type,
    )
    return {
        "train_seconds": train["seconds"],
        "eval_seconds": evaluate["seconds"],
        "rows": int(evaluate["value"].get("rows", 0)),
        "targets": evaluate["value"].get("targets", {}),
    }


def main():
    ap = argparse.ArgumentParser(description="Profile OTO ML pipeline bottlenecks.")
    ap.add_argument(
        "--config",
        default=os.path.join(ROOT, "ml", "configs", "training_data_roots.yaml"),
        help="Training roots YAML config",
    )
    ap.add_argument(
        "--dataset-root",
        default=os.path.join(ROOT, "dataset"),
        help="Staged dataset root",
    )
    ap.add_argument(
        "--workspace-root",
        default=os.path.abspath(os.path.join(ROOT, "..", "ml_workspace")),
        help="Workspace root",
    )
    ap.add_argument(
        "--train-dataset",
        default="",
        help="Optional dataset csv to train/evaluate during profiling",
    )
    ap.add_argument(
        "--train-language",
        default="japanese",
        help="Language for optional training benchmark",
    )
    ap.add_argument(
        "--train-format",
        default="cvvc",
        help="Format for optional training benchmark",
    )
    ap.add_argument(
        "--report",
        default="",
        help="Optional report json output path",
    )
    args = ap.parse_args()

    workspace_root = os.path.abspath(args.workspace_root)
    dataset_root = os.path.abspath(args.dataset_root)
    report_path = args.report or os.path.join(workspace_root, "reports", "pipeline_profile.json")

    config_discovery = _timed("discover_training_candidates", discover_training_candidates, args.config)
    staged_discovery = _timed("discover_training_candidates_from_dataset_root", discover_training_candidates_from_dataset_root, dataset_root)
    dry_run_prepare = _timed("discover_work_items", _discover_work_items, dataset_root)
    staged_build = _timed("build_datasets_from_candidates", build_datasets_from_candidates, staged_discovery["value"], workspace_root)
    prepare_one = _profile_prepare_one(dataset_root)

    report: Dict[str, Any] = {
        "config_discovery_seconds": config_discovery["seconds"],
        "config_candidate_status": _summarize_candidate_status(config_discovery["value"]),
        "staged_discovery_seconds": staged_discovery["seconds"],
        "staged_candidate_status": _summarize_candidate_status(staged_discovery["value"]),
        "discover_work_items_seconds": dry_run_prepare["seconds"],
        "work_item_count": len(dry_run_prepare["value"]),
        "build_datasets_seconds": staged_build["seconds"],
        "build_datasets_summary": staged_build["value"].get("summary", {}),
        "prepare_one": prepare_one,
    }

    train_dataset = args.train_dataset.strip()
    if train_dataset:
        out_dir = os.path.join(
            ROOT,
            "assets",
            "models",
            "oto_ml",
            args.train_language.strip().lower(),
            args.train_format.strip().lower(),
            "profile_run",
        )
        report["training_profile"] = _profile_training(
            dataset_csv=os.path.abspath(train_dataset),
            out_dir=out_dir,
            language=args.train_language,
            format_type=args.train_format,
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _safe_print_json(report)


if __name__ == "__main__":
    main()
