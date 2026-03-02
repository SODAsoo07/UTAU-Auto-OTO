from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_dataset import TORCH_TASKS, build_torch_dataset
from core.oto_torch_trainer import train_torch_bundle


DEFAULT_TASK_ORDER = [
    "korean_bridge_cvvc_vv",
    "korean_bridge_cvvc_vc_sonorant",
    "korean_bridge_cvvc_vc_stop",
    "korean_bridge_cvc_vc_sonorant",
    "korean_bridge_cvc_vc_stop",
]

TASK_DEFAULTS: Dict[str, Dict[str, object]] = {
    "korean_bridge_cvvc_vv": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50},
    "korean_bridge_cvvc_vc": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50},
    "korean_bridge_cvvc_vc_sonorant": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50},
    "korean_bridge_cvvc_vc_stop": {"epochs": 24, "batch_size": 24, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 50},
    "korean_bridge_cvc_vc": {"epochs": 28, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 20},
    "korean_bridge_cvc_vc_sonorant": {"epochs": 28, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 20},
    "korean_bridge_cvc_vc_stop": {"epochs": 32, "batch_size": 12, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 20},
    "japanese_bridge_vcv_vv": {"epochs": 16, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_vcv_vc": {"epochs": 16, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_vcv_vcv": {"epochs": 16, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_vcv_cv_head": {"epochs": 16, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_cvvc_vv": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_cvvc_vc": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
    "japanese_bridge_cvvc_cv_head": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 100},
}


def _repo_root() -> str:
    return ROOT


def _workspace_root() -> str:
    return os.path.dirname(_repo_root())


def _dataset_csv_for_task(task_name: str) -> str:
    language = str(TORCH_TASKS[task_name]["language"]).strip().lower()
    formats = set(TORCH_TASKS[task_name].get("formats") or [])
    if language == "korean":
        if "cvvc" in formats:
            return os.path.join(_workspace_root(), "ml_workspace", "datasets", "korean", "dataset_korean_cvvc.csv")
        if "cvc" in formats:
            return os.path.join(_workspace_root(), "ml_workspace", "datasets", "korean", "dataset_korean_cvc.csv")
    if language == "japanese":
        if "vcv" in formats:
            return os.path.join(_workspace_root(), "ml_workspace", "datasets", "japanese", "dataset_japanese_vcv.csv")
        if "cvvc" in formats:
            return os.path.join(_workspace_root(), "ml_workspace", "datasets", "japanese", "dataset_japanese_cvvc.csv")
    raise KeyError(f"No dataset CSV mapping for task: {task_name}")


def _torch_dataset_dir(task_name: str) -> str:
    language = str(TORCH_TASKS[task_name]["language"]).strip().lower()
    return os.path.join(_workspace_root(), "ml_workspace", "torch_datasets", language, task_name)


def _model_out_dir(task_name: str) -> str:
    language = str(TORCH_TASKS[task_name]["language"]).strip().lower()
    return os.path.join(_repo_root(), "assets", "models", "oto_ml_pytorch", language, task_name, "v1")


def _build_manifest_path() -> str:
    return os.path.join(_workspace_root(), "ml_workspace", "reports", "training_build_results.json")


def _dataset_root() -> str:
    return os.path.join(_repo_root(), "dataset")


def _resolve_tasks(args_tasks: List[str]) -> List[str]:
    if args_tasks:
        return args_tasks
    return list(DEFAULT_TASK_ORDER)


def main():
    ap = argparse.ArgumentParser(description="Build and train multiple PyTorch OTO bridge tasks sequentially.")
    ap.add_argument("--tasks", nargs="*", choices=sorted(TORCH_TASKS.keys()), default=[])
    ap.add_argument("--dataset-root", default=_dataset_root())
    ap.add_argument("--build-manifest", default=_build_manifest_path())
    ap.add_argument("--window-frames", type=int, default=160)
    ap.add_argument("--mel-bins", type=int, default=80)
    ap.add_argument("--device", default="", help="Optional device override, e.g. cpu or cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--epochs", type=int, default=0, help="Override all task epochs when > 0")
    ap.add_argument("--batch-size", type=int, default=0, help="Override all task batch sizes when > 0")
    ap.add_argument("--learning-rate", type=float, default=0.0, help="Override all task learning rates when > 0")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Override all task weight decay when > 0")
    ap.add_argument("--log-every-steps", type=int, default=0, help="Override all task log step interval when > 0")
    args = ap.parse_args()

    tasks = _resolve_tasks(args.tasks)
    results = []
    print(f"[TorchBatch] tasks={tasks}")

    for task_name in tasks:
        language = str(TORCH_TASKS[task_name]["language"]).strip().lower()
        dataset_csv = _dataset_csv_for_task(task_name)
        dataset_dir = _torch_dataset_dir(task_name)
        out_dir = _model_out_dir(task_name)
        defaults = dict(TASK_DEFAULTS.get(task_name, {}))
        epochs = int(args.epochs) if int(args.epochs) > 0 else int(defaults.get("epochs", 12))
        batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(defaults.get("batch_size", 32))
        learning_rate = float(args.learning_rate) if float(args.learning_rate) > 0 else float(defaults.get("learning_rate", 1e-3))
        weight_decay = float(args.weight_decay) if float(args.weight_decay) > 0 else float(defaults.get("weight_decay", 1e-4))
        log_every_steps = int(args.log_every_steps) if int(args.log_every_steps) > 0 else int(defaults.get("log_every_steps", 100))

        print(f"[TorchBatch] build task={task_name} dataset_csv={dataset_csv}")
        build_summary = build_torch_dataset(
            task_name=task_name,
            dataset_csv=dataset_csv,
            out_dir=dataset_dir,
            dataset_root=args.dataset_root,
            build_manifest_path=args.build_manifest,
            window_frames=args.window_frames,
            mel_bins=args.mel_bins,
        )
        print(
            f"[TorchBatch] build_done task={task_name} "
            f"rows={build_summary.row_count} shards={build_summary.shard_count} skipped={build_summary.skipped_rows}"
        )

        print(
            f"[TorchBatch] train task={task_name} lang={language} "
            f"epochs={epochs} batch={batch_size} lr={learning_rate} wd={weight_decay}"
        )
        meta = train_torch_bundle(
            dataset_dir=dataset_dir,
            out_dir=out_dir,
            language=language,
            task_name=task_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_amp=not args.no_amp,
            device_override=args.device,
            log_every_steps=log_every_steps,
        )
        results.append(
            {
                "task": task_name,
                "language": language,
                "dataset_rows": build_summary.row_count,
                "best_epoch": meta.get("best_epoch"),
                "train_rows": meta.get("train_rows"),
                "out_dir": out_dir,
            }
        )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
