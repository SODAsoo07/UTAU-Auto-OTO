from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_torch_dataset import load_torch_dataset_index
from core.oto_torch_trainer import _evaluate, _group_split, _scan_train_stats, TorchShardDataset, _collate
from core.oto_ml_pytorch import load_pytorch_bundle
from torch.utils.data import DataLoader
import torch


def main():
    ap = argparse.ArgumentParser(description="Evaluate PyTorch bridge OTO bundle.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    index = load_torch_dataset_index(os.path.join(args.dataset_dir, "index.json"))
    shards = list(index.get("shards") or [])
    train_shards, valid_shards = _group_split(shards)
    if not valid_shards:
        valid_shards = train_shards[-1:]
    categorical_names = list(index.get("categorical_feature_names") or [])
    numeric_names = list(index.get("numeric_feature_names") or [])
    target_names = list(index.get("target_names") or [])
    normalizer, vocab = _scan_train_stats(train_shards, categorical_names)
    valid_ds = TorchShardDataset(valid_shards, numeric_names, categorical_names, target_names, vocab, normalizer)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate)
    bundle = load_pytorch_bundle(args.model_dir)
    model = bundle["model"]
    device = torch.device("cpu")
    model.to(device)
    summary = _evaluate(model, valid_loader, device, numeric_names)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
