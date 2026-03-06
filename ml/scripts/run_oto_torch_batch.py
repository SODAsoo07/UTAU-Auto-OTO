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
    "korean_bridge_cvvc",
    "korean_bridge_cvc",
    "japanese_bridge_vcv",
    "japanese_bridge_cvvc",
]

# Small-sample tasks are unstable for PyTorch and should be skipped by default
# unless explicitly overridden.
TASK_MIN_ROWS: Dict[str, int] = {
    "korean_bridge_cvvc": 600,
    "korean_bridge_cvc": 200,
    "japanese_bridge_vcv": 1500,
    "japanese_bridge_cvvc": 300,
    "korean_bridge_cvvc_vv": 300,
    "korean_bridge_cvvc_vc": 120,
    "korean_bridge_cvc_vc": 80,
    "korean_bridge_cvvc_vc_sonorant": 120,
    "korean_bridge_cvvc_vc_stop": 120,
    "korean_bridge_cvc_vc_sonorant": 80,
    "korean_bridge_cvc_vc_stop": 80,
    "japanese_bridge_vcv_vv": 600,
    "japanese_bridge_vcv_vc": 600,
    "japanese_bridge_vcv_vcv": 1500,
    "japanese_bridge_vcv_cv_head": 300,
    "japanese_bridge_cvvc_vv": 120,
    "japanese_bridge_cvvc_vc": 120,
    "japanese_bridge_cvvc_cv_head": 60,
}

TASK_DEFAULTS: Dict[str, Dict[str, object]] = {
    "korean_bridge_cvvc": {"epochs": 24, "batch_size": 32, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 50, "early_stopping_patience": 3, "early_stopping_min_delta": 0.3, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.3},
    "korean_bridge_cvc": {"epochs": 24, "batch_size": 24, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 50, "early_stopping_patience": 3, "early_stopping_min_delta": 0.3, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.3},
    "japanese_bridge_vcv": {"epochs": 12, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 3, "early_stopping_min_delta": 0.3, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.3},
    "japanese_bridge_cvvc": {"epochs": 14, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 3, "early_stopping_min_delta": 0.3, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.3},
    "korean_bridge_cvvc_vv": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvvc_vc": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvvc_vc_sonorant": {"epochs": 20, "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 50, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvvc_vc_stop": {"epochs": 24, "batch_size": 24, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 50, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvc_vc": {"epochs": 28, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 20, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvc_vc_sonorant": {"epochs": 28, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "log_every_steps": 20, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "korean_bridge_cvc_vc_stop": {"epochs": 32, "batch_size": 12, "learning_rate": 7.5e-4, "weight_decay": 1.5e-4, "log_every_steps": 20, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_vcv_vv": {"epochs": 8, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_vcv_vc": {"epochs": 10, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_vcv_vcv": {"epochs": 10, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_vcv_cv_head": {"epochs": 10, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_cvvc_vv": {"epochs": 12, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_cvvc_vc": {"epochs": 12, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
    "japanese_bridge_cvvc_cv_head": {"epochs": 12, "batch_size": 32, "learning_rate": 5e-4, "weight_decay": 2e-4, "log_every_steps": 100, "early_stopping_patience": 2, "early_stopping_min_delta": 0.5, "lr_reduce_factor": 0.5, "lr_reduce_patience": 1, "lr_reduce_min_lr": 1e-5, "lr_reduce_threshold": 0.5},
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


def _lightgbm_root(language: str) -> str:
    return os.path.join(_repo_root(), "assets", "models", "oto_ml", language)


def _installed_lightgbm_root(language: str) -> str:
    return os.path.join(_repo_root(), "models_installed", "oto_ml", language)


def _discover_lightgbm_model_dirs(root: str, format_type: str) -> List[str]:
    format_dir = os.path.join(root, format_type)
    found: List[str] = []
    if os.path.isfile(os.path.join(format_dir, "model_meta.json")):
        found.append(format_dir)
    if os.path.isdir(format_dir):
        for child in os.listdir(format_dir):
            child_dir = os.path.join(format_dir, child)
            if os.path.isfile(os.path.join(child_dir, "model_meta.json")):
                found.append(child_dir)
    v1 = os.path.join(format_dir, "v1")
    if os.path.isfile(os.path.join(v1, "model_meta.json")):
        found.append(v1)
    # Deduplicate while preserving order.
    unique: List[str] = []
    seen = set()
    for path in found:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_lightgbm_base_for_task(task_name: str) -> str:
    task_spec = TORCH_TASKS[task_name]
    language = str(task_spec["language"]).strip().lower()
    formats = sorted(set(task_spec.get("formats") or []))
    preferred = formats[0] if formats else "general"
    candidates: List[tuple[int, str]] = []
    for root in (_lightgbm_root(language), _installed_lightgbm_root(language)):
        for fmt_priority, fmt in enumerate((preferred, "general")):
            for path in _discover_lightgbm_model_dirs(root, fmt):
                candidates.append((fmt_priority, path))
    if not candidates:
        return ""
    # Prefer the latest trained bundle.
    candidates = sorted(
        candidates,
        key=lambda item: (item[0], -os.path.getmtime(os.path.join(item[1], "model_meta.json"))),
    )
    return os.path.abspath(candidates[0][1])


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
    ap.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA GPU for training (default: enabled). Use --no-require-cuda to allow CPU fallback.",
    )
    ap.add_argument("--epochs", type=int, default=0, help="Override all task epochs when > 0")
    ap.add_argument("--batch-size", type=int, default=0, help="Override all task batch sizes when > 0")
    ap.add_argument("--learning-rate", type=float, default=0.0, help="Override all task learning rates when > 0")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="Override all task weight decay when > 0")
    ap.add_argument("--log-every-steps", type=int, default=0, help="Override all task log step interval when > 0")
    ap.add_argument("--early-stopping-patience", type=int, default=-1, help="Override all task early-stopping patience when >= 0")
    ap.add_argument("--early-stopping-min-delta", type=float, default=-1.0, help="Override all task early-stopping min delta when >= 0")
    ap.add_argument("--lr-reduce-factor", type=float, default=-1.0, help="Override all task LR reduce factor when > 0")
    ap.add_argument("--lr-reduce-patience", type=int, default=-1, help="Override all task LR reduce patience when >= 0")
    ap.add_argument("--lr-reduce-min-lr", type=float, default=-1.0, help="Override all task LR reduce min_lr when >= 0")
    ap.add_argument("--lr-reduce-threshold", type=float, default=-1.0, help="Override all task LR reduce threshold when >= 0")
    ap.add_argument(
        "--residual-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train PyTorch in residual mode over LightGBM base predictions (default: enabled).",
    )
    ap.add_argument(
        "--residual-base-model-dir",
        default="",
        help="Optional fixed LightGBM model dir for residual mode. If empty, task-specific latest bundle is auto-selected.",
    )
    ap.add_argument("--min-rows", type=int, default=0, help="Global minimum rows to start training (0 = task defaults)")
    ap.add_argument("--allow-tiny", action="store_true", help="Allow training even if dataset rows are below minimum")
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
        early_stopping_patience = int(args.early_stopping_patience) if int(args.early_stopping_patience) >= 0 else int(defaults.get("early_stopping_patience", 2))
        early_stopping_min_delta = float(args.early_stopping_min_delta) if float(args.early_stopping_min_delta) >= 0 else float(defaults.get("early_stopping_min_delta", 0.5))
        lr_reduce_factor = float(args.lr_reduce_factor) if float(args.lr_reduce_factor) > 0 else float(defaults.get("lr_reduce_factor", 0.5))
        lr_reduce_patience = int(args.lr_reduce_patience) if int(args.lr_reduce_patience) >= 0 else int(defaults.get("lr_reduce_patience", 1))
        lr_reduce_min_lr = float(args.lr_reduce_min_lr) if float(args.lr_reduce_min_lr) >= 0 else float(defaults.get("lr_reduce_min_lr", 1e-5))
        lr_reduce_threshold = float(args.lr_reduce_threshold) if float(args.lr_reduce_threshold) >= 0 else float(defaults.get("lr_reduce_threshold", 0.5))

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

        min_rows = int(args.min_rows) if int(args.min_rows) > 0 else int(TASK_MIN_ROWS.get(task_name, 0))
        if not args.allow_tiny and min_rows > 0 and int(build_summary.row_count) < min_rows:
            print(
                f"[TorchBatch] skip_train task={task_name} "
                f"rows={build_summary.row_count} < min_rows={min_rows}"
            )
            results.append(
                {
                    "task": task_name,
                    "language": language,
                    "dataset_rows": build_summary.row_count,
                    "min_rows": min_rows,
                    "skipped": True,
                    "reason": "too_few_rows",
                    "out_dir": out_dir,
                }
            )
            continue

        print(
            f"[TorchBatch] train task={task_name} lang={language} "
            f"epochs={epochs} batch={batch_size} lr={learning_rate} wd={weight_decay}"
        )
        residual_base_model_dir = str(args.residual_base_model_dir or "").strip()
        task_residual_mode = bool(args.residual_mode)
        if task_residual_mode and not residual_base_model_dir:
            residual_base_model_dir = _resolve_lightgbm_base_for_task(task_name)
        if task_residual_mode:
            if residual_base_model_dir:
                print(f"[TorchBatch] residual task={task_name} base={residual_base_model_dir}")
            else:
                print(f"[TorchBatch] residual_disabled task={task_name} reason=base_model_not_found")
                task_residual_mode = False
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
            require_cuda=args.require_cuda,
            log_every_steps=log_every_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr_reduce_factor=lr_reduce_factor,
            lr_reduce_patience=lr_reduce_patience,
            lr_reduce_min_lr=lr_reduce_min_lr,
            lr_reduce_threshold=lr_reduce_threshold,
            residual_mode=task_residual_mode,
            residual_base_model_dir=residual_base_model_dir,
        )
        results.append(
            {
                "task": task_name,
                "language": language,
                "dataset_rows": build_summary.row_count,
                "min_rows": min_rows,
                "best_epoch": meta.get("best_epoch"),
                "train_rows": meta.get("train_rows"),
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "lr_reduce_factor": lr_reduce_factor,
                "lr_reduce_patience": lr_reduce_patience,
                "lr_reduce_min_lr": lr_reduce_min_lr,
                "lr_reduce_threshold": lr_reduce_threshold,
                "target_mode": meta.get("target_mode", "absolute_delta"),
                "residual_base_model_dir": meta.get("residual_base_model_dir", ""),
                "skipped": False,
                "out_dir": out_dir,
            }
        )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
