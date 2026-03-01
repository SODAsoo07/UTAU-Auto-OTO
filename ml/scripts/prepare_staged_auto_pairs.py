from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_batch_prepare import prepare_staged_auto_pairs, write_prepare_report


def main():
    ap = argparse.ArgumentParser(description="Generate lab/dict/TextGrid/auto-oto for staged dataset copies.")
    ap.add_argument(
        "--dataset-root",
        default=os.path.join(ROOT, "dataset"),
        help="Staged dataset root",
    )
    ap.add_argument("--dry-run", action="store_true", help="Discover planned jobs without running MFA/OTO generation")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of work items")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    result = prepare_staged_auto_pairs(dataset_root, dry_run=args.dry_run, limit=args.limit)
    report_path = os.path.join(dataset_root, "_manifest", "prepared_auto_pairs.json")
    write_prepare_report(report_path, result)
    print(json.dumps({
        "summary": result.get("summary", {}),
        "report_path": os.path.abspath(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
