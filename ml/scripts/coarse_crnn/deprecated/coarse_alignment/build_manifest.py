from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.deprecated.coarse_alignment.data_discovery import discover_dataset_staged, discover_public_sinsy, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Korean/Japanese coarse_crnn training manifest.")
    parser.add_argument("--dataset-staged", default="dataset_staged")
    parser.add_argument("--public-root", default=r"C:\Users\oyh57\SODAsoo1\VocalSynth\Data\PublicData")
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "manifest.jsonl"))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--scan-all-public", action="store_true", help="Scan the entire public root instead of known ko/ja Sinsy lab roots.")
    args = parser.parse_args()

    rows = []
    rows.extend(discover_dataset_staged(args.dataset_staged))
    rows.extend(discover_public_sinsy(args.public_root, scan_all=bool(args.scan_all_public)))
    rows.sort(key=lambda row: (str(row.get("language", "")), str(row.get("source", "")), str(row.get("audio", ""))))
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    count = write_manifest(args.out, rows)
    summary = {
        "manifest": os.path.abspath(args.out),
        "rows": count,
        "by_language": _count_by(rows, "language"),
        "by_source": _count_by(rows, "source"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if count > 0 else 2


def _count_by(rows, key):
    out = {}
    for row in rows:
        value = str(row.get(key, "") or "unknown")
        out[value] = int(out.get(value, 0)) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())

