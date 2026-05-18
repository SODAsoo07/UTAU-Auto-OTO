"""Split a phoneme-boundary manifest JSONL into train / heldout with a fixed
seed. Stratifies by source bucket (heuristic: 'PLZ' if wav_path contains
PLZ_UTAU, otherwise the parent dir name) so both splits represent every
voicebank present."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


def bucket(row: dict) -> str:
    p = str(row.get("wav_path") or "")
    if "PLZ_UTAU" in p:
        return "PLZ"
    parent = os.path.basename(os.path.dirname(p)) or "?"
    return parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--heldout-out", required=True)
    ap.add_argument("--heldout-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260518)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[bucket(r)].append(r)

    rng = random.Random(int(args.seed))
    train: list[dict] = []
    held: list[dict] = []
    for b, items in sorted(by_bucket.items()):
        order = list(items)
        rng.shuffle(order)
        n_held = max(1, int(round(len(order) * float(args.heldout_ratio))))
        held.extend(order[:n_held])
        train.extend(order[n_held:])

    for path, data in [(args.train_out, train), (args.heldout_out, held)]:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "manifest": os.path.abspath(args.manifest),
        "train_out": os.path.abspath(args.train_out),
        "heldout_out": os.path.abspath(args.heldout_out),
        "seed": int(args.seed),
        "heldout_ratio": float(args.heldout_ratio),
        "total": len(rows),
        "train": len(train),
        "heldout": len(held),
        "per_bucket": {b: {"train": sum(1 for r in train if bucket(r) == b),
                            "heldout": sum(1 for r in held if bucket(r) == b)}
                        for b in sorted(by_bucket)},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
