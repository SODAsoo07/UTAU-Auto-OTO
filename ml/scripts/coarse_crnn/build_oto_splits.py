from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

from core.coarse_crnn.oto_targets import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build voicebank-held-out OTO train/val/test manifests.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--out-dir", default=os.path.join("ml_workspace", "coarse_crnn", "oto_splits"))
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    splits, group_summary = split_by_voicebank(
        rows,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}
    for name in ("train", "val", "test"):
        path = os.path.join(args.out_dir, f"oto_{name}.jsonl")
        write_jsonl(path, splits[name])
        paths[name] = os.path.abspath(path)
    summary = {
        "manifest": os.path.abspath(args.manifest),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "paths": paths,
        "rows": {name: len(splits[name]) for name in ("train", "val", "test")},
        "voicebanks": group_summary,
        "by_language": {name: _count_field(splits[name], "language") for name in ("train", "val", "test")},
        "by_format": {name: _count_field(splits[name], "format_type") for name in ("train", "val", "test")},
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if splits["train"] and splits["val"] and splits["test"] else 2


def split_by_voicebank(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _group_key(row)
        groups[key].append(row)
    rng = random.Random(int(seed))
    strata: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for key, items in groups.items():
        strata[_stratum_key(items[0])].append((key, items))

    split_groups = {"train": [], "val": [], "test": []}
    for _stratum, group_items in sorted(strata.items()):
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(items) for _key, items in group_items)
        if len(group_items) < 3 or total < 20:
            for key, _items in group_items:
                split_groups["train"].append(key)
            continue
        targets = {
            "test": max(1, int(round(total * max(0.0, float(test_ratio))))),
            "val": max(1, int(round(total * max(0.0, float(val_ratio))))),
        }
        local_counts = {"val": 0, "test": 0}
        local_group_counts = {"val": 0, "test": 0}
        for key, items in group_items:
            candidates = []
            for split_name in ("test", "val"):
                deficit = targets[split_name] - local_counts[split_name]
                needs_group = 1 if local_group_counts[split_name] <= 0 else 0
                candidates.append((needs_group, deficit, split_name))
            candidates.sort(reverse=True)
            chosen = candidates[0][2] if candidates[0][0] > 0 or candidates[0][1] > 0 else "train"
            split_groups[chosen].append(key)
            if chosen in local_counts:
                local_counts[chosen] += len(items)
                local_group_counts[chosen] += 1

    out = {"train": [], "val": [], "test": []}
    for split_name, keys in split_groups.items():
        for key in keys:
            out[split_name].extend(groups[key])
    summary = {
        "total": len(groups),
        "train": len(split_groups["train"]),
        "val": len(split_groups["val"]),
        "test": len(split_groups["test"]),
        "strata": len(strata),
        "largest": [
            {"voicebank": key, "rows": len(items)}
            for key, items in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:12]
        ],
    }
    return out, summary


def _group_key(row: dict[str, Any]) -> str:
    language = str(row.get("language", "") or "unknown")
    fmt = str(row.get("format_type", "") or "unknown")
    voicebank = str(row.get("voicebank_id", "") or "unknown_voicebank")
    return f"{language}/{fmt}/{voicebank}"


def _stratum_key(row: dict[str, Any]) -> str:
    language = str(row.get("language", "") or "unknown")
    fmt = str(row.get("format_type", "") or "unknown")
    return f"{language}/{fmt}"


def _count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row.get(field, "") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


if __name__ == "__main__":
    raise SystemExit(main())
