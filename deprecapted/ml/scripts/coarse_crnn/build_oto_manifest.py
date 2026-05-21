from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from core.coarse_crnn.oto_targets import iter_oto_manifest_rows, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an oto.ini-supervised manifest for the OTO anchor CRNN.")
    parser.add_argument("--dataset-staged", default="dataset_staged")
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--summary", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest_summary.json"))
    args = parser.parse_args()

    rows = iter_oto_manifest_rows(args.dataset_staged)
    count = write_jsonl(args.out, rows)
    by_lang = Counter(str(row.get("language", "") or "unknown") for row in rows)
    by_format = Counter(str(row.get("format_type", "") or "unknown") for row in rows)
    summary = {
        "dataset_staged": os.path.abspath(args.dataset_staged),
        "out": os.path.abspath(args.out),
        "rows": count,
        "by_language": dict(sorted(by_lang.items())),
        "by_format": dict(sorted(by_format.items())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
