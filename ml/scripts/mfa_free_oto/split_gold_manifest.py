from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from core.mfa_free_oto.manifest import load_manifest_jsonl, write_manifest_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic train/eval split for MFA-free OTO manual-gold manifests."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()

    rows = load_manifest_jsonl(args.manifest, require_manual=True, require_labels=True)
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    write_manifest_jsonl(args.train_out, train_rows)
    write_manifest_jsonl(args.eval_out, eval_rows)
    summary = {
        "manifest": args.manifest,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_out": args.train_out,
        "eval_out": args.eval_out,
    }
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    target_eval = max(1, round(len(rows) * max(0.0, min(0.9, eval_ratio))))
    eval_keys: set[str] = set()
    eval_count = 0
    for key, group_rows in group_items:
        if eval_count >= target_eval:
            break
        eval_keys.add(key)
        eval_count += len(group_rows)
    train_rows = [row for key, group in groups.items() for row in group if key not in eval_keys]
    eval_rows = [row for key, group in groups.items() for row in group if key in eval_keys]
    return train_rows, eval_rows


def _group_key(row: dict) -> str:
    for key in ("reclist_line_id", "source_line_id", "row_group", "row_id", "wav_name"):
        value = row.get(key)
        if value:
            return str(value)
    return str(row.get("wav_path") or id(row))


if __name__ == "__main__":
    raise SystemExit(main())

