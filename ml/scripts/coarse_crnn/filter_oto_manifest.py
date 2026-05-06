from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter OTO manifest rows for diagnostics or controlled experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--voicebank-id", default="")
    parser.add_argument("--format-type", default="")
    parser.add_argument("--drop-blank-alias", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(str(args.manifest))
    kept, removed = filter_rows(
        rows,
        voicebank_id=str(args.voicebank_id or ""),
        format_type=str(args.format_type or ""),
        drop_blank_alias=bool(args.drop_blank_alias),
    )
    write_jsonl(str(args.out), kept)
    summary = summarize_filter(
        rows,
        kept,
        removed,
        manifest=str(args.manifest),
        out=str(args.out),
    )
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(str(args.summary))), exist_ok=True)
        with open(str(args.summary), "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if kept else 2


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    voicebank_id: str = "",
    format_type: str = "",
    drop_blank_alias: bool = False,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for row in rows:
        if voicebank_id and str(row.get("voicebank_id", "") or "") != voicebank_id:
            removed["voicebank_id"] += 1
            continue
        if format_type and str(row.get("format_type", "") or "").lower() != format_type.lower():
            removed["format_type"] += 1
            continue
        if drop_blank_alias and not str(row.get("alias", "") or "").strip():
            removed["blank_alias"] += 1
            continue
        kept.append(dict(row))
    return kept, removed


def summarize_filter(
    rows: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    removed: Counter[str],
    *,
    manifest: str,
    out: str,
) -> dict[str, Any]:
    return {
        "manifest": os.path.abspath(str(manifest)),
        "out": os.path.abspath(str(out)),
        "rows_in": len(rows),
        "rows_out": len(kept),
        "rows_removed": len(rows) - len(kept),
        "drop_rate": float((len(rows) - len(kept)) / max(1, len(rows))),
        "removed_reasons": dict(sorted(removed.items())),
        "by_voicebank_out": _count_field(kept, "voicebank_id"),
        "by_format_out": _count_field(kept, "format_type"),
        "by_context_out": _count_context(kept),
    }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _count_field(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "") or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def _count_context(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(
        "|".join(
            [
                str(row.get("format_type", "") or "other"),
                str(row.get("alias_type", "") or "other"),
                str(row.get("transition_type", "") or "other"),
            ]
        )
        for row in rows
    )
    return dict(sorted(counter.items()))


if __name__ == "__main__":
    raise SystemExit(main())
