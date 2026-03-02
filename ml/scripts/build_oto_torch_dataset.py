from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_dataset import TORCH_TASKS, build_torch_dataset


def main():
    ap = argparse.ArgumentParser(description="Build PyTorch bridge OTO dataset shards.")
    ap.add_argument("--task", required=True, choices=sorted(TORCH_TASKS.keys()))
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-root", default="")
    ap.add_argument("--build-manifest", default="")
    ap.add_argument("--window-frames", type=int, default=160)
    ap.add_argument("--mel-bins", type=int, default=80)
    args = ap.parse_args()

    summary = build_torch_dataset(
        task_name=args.task,
        dataset_csv=args.dataset_csv,
        out_dir=args.out_dir,
        dataset_root=args.dataset_root,
        build_manifest_path=args.build_manifest,
        window_frames=args.window_frames,
        mel_bins=args.mel_bins,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
