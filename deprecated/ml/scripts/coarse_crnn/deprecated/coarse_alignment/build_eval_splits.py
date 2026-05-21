from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.coarse_crnn.deprecated.coarse_alignment.data_discovery import read_manifest, write_manifest
from core.coarse_crnn.lang import short_language


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic coarse_crnn evaluation splits.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "manifest_full.jsonl"))
    parser.add_argument("--out-dir", default=os.path.join("ml_workspace", "coarse_crnn", "eval_splits"))
    parser.add_argument("--max-per-bucket", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bucket_name(row)].append(row)

    rng = random.Random(int(args.seed))
    selected_all: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "manifest": os.path.abspath(args.manifest),
        "seed": int(args.seed),
        "max_per_bucket": int(args.max_per_bucket),
        "buckets": {},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for bucket in sorted(buckets):
        items = list(buckets[bucket])
        rng.shuffle(items)
        picked = items[: max(0, int(args.max_per_bucket))]
        selected_all.extend(picked)
        out_path = os.path.join(args.out_dir, f"{bucket}.jsonl")
        write_manifest(out_path, picked)
        summary["buckets"][bucket] = {
            "available": len(items),
            "selected": len(picked),
            "path": os.path.abspath(out_path),
        }

    all_path = os.path.join(args.out_dir, "eval_all.jsonl")
    write_manifest(all_path, selected_all)
    summary["total_selected"] = len(selected_all)
    summary["all_path"] = os.path.abspath(all_path)
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _bucket_name(row: dict[str, Any]) -> str:
    source = str(row.get("source", "") or "unknown")
    lang = short_language(row.get("language", "")) or "unknown"
    if source == "public_sinsy":
        return f"{lang}_public_sinsy_song"
    text = str(Path(str(row.get("audio", "") or "")).as_posix()).lower()
    for token in ("cvvc", "vcv", "cvc", "cv"):
        if f"/{token}/" in text or f"_{token}_" in text or f"-{token}-" in text:
            return f"{lang}_{token}"
    return f"{lang}_other"


if __name__ == "__main__":
    raise SystemExit(main())

