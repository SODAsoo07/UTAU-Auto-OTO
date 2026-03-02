from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_trainer import train_torch_bundle


def main():
    ap = argparse.ArgumentParser(description="Train PyTorch bridge OTO bundle.")
    ap.add_argument("--lang", required=True, choices=["korean", "japanese"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--log-every-steps", type=int, default=100)
    ap.add_argument("--early-stopping-patience", type=int, default=2)
    ap.add_argument("--early-stopping-min-delta", type=float, default=0.5)
    ap.add_argument("--lr-reduce-factor", type=float, default=0.5)
    ap.add_argument("--lr-reduce-patience", type=int, default=1)
    ap.add_argument("--lr-reduce-min-lr", type=float, default=1e-5)
    ap.add_argument("--lr-reduce-threshold", type=float, default=0.5)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="", help="Optional device override, e.g. cpu or cuda")
    args = ap.parse_args()

    meta = train_torch_bundle(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        language=args.lang,
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        device_override=args.device,
        log_every_steps=args.log_every_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_reduce_factor=args.lr_reduce_factor,
        lr_reduce_patience=args.lr_reduce_patience,
        lr_reduce_min_lr=args.lr_reduce_min_lr,
        lr_reduce_threshold=args.lr_reduce_threshold,
    )
    print(json.dumps({
        "backend": meta.get("backend"),
        "language": meta.get("language"),
        "task_name": meta.get("task_name"),
        "train_rows": meta.get("train_rows"),
        "best_epoch": meta.get("best_epoch"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
