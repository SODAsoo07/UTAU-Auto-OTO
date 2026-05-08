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
    parser.add_argument(
        "--exclude-blank-alias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop rows whose blank_alias_kind is non-empty (blank/silence) before writing. "
             "Plan B 1-A from CRNN-정확도-개선-방안-v2: removes SONG_CMYK_CVVC's 23.4%% blank-alias rows "
             "that otherwise pollute the cvvc|cv|other slice. Default OFF — the column is always "
             "populated regardless, so downstream evaluation can still group by blank_alias_kind.",
    )
    args = parser.parse_args()

    raw_rows = iter_oto_manifest_rows(args.dataset_staged)
    pre_total = len(raw_rows)
    if bool(args.exclude_blank_alias):
        rows = [row for row in raw_rows if not str(row.get("blank_alias_kind", "") or "").strip()]
    else:
        rows = raw_rows
    dropped = pre_total - len(rows)

    count = write_jsonl(args.out, rows)
    by_lang = Counter(str(row.get("language", "") or "unknown") for row in rows)
    by_format = Counter(str(row.get("format_type", "") or "unknown") for row in rows)
    by_blank_kind = Counter(str(row.get("blank_alias_kind", "") or "") for row in raw_rows)
    summary = {
        "dataset_staged": os.path.abspath(args.dataset_staged),
        "out": os.path.abspath(args.out),
        "rows": count,
        "raw_rows": pre_total,
        "dropped_blank_rows": dropped,
        "exclude_blank_alias": bool(args.exclude_blank_alias),
        "by_language": dict(sorted(by_lang.items())),
        "by_format": dict(sorted(by_format.items())),
        "by_blank_alias_kind_pre_filter": dict(sorted(by_blank_kind.items())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
