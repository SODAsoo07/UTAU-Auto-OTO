from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.data_discovery import read_manifest
from core.coarse_crnn.model import CoarseCrnnConfig
from core.coarse_crnn.training import TrainConfig, train_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the coarse_crnn Korean/Japanese aligner.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "manifest.jsonl"))
    parser.add_argument("--out", default=os.path.join("ml_workspace", "models", "coarse_crnn", "coarse_crnn.pt"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--conv-channels", type=int, default=96)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    model_cfg = CoarseCrnnConfig(n_mels=int(args.n_mels), hidden=int(args.hidden), conv_channels=int(args.conv_channels))
    train_cfg = TrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        device=str(args.device),
    )
    result = train_from_manifest(rows, args.out, train_config=train_cfg, model_config=model_cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
