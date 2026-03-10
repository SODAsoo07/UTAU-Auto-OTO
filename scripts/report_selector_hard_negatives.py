import argparse
import csv
import json
import os
from collections import Counter


def _parse_args():
    parser = argparse.ArgumentParser(description="Summarize selector hard-negative logs.")
    parser.add_argument("--log-path", default="", dest="log_path")
    parser.add_argument("--out-dir", default="", dest="out_dir")
    return parser.parse_args()


def _default_log_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "logs", "selector_hard_negatives.jsonl")


def _read_rows(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = _parse_args()
    log_path = os.path.abspath(args.log_path) if args.log_path else _default_log_path()
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(log_path)
    rows = _read_rows(log_path)

    summary = {
        "total_rows": len(rows),
        "by_reason": Counter(),
        "by_format": Counter(),
        "by_candidate_mode": Counter(),
        "top_aliases": Counter(),
    }
    for row in rows:
        summary["by_reason"][str(row.get("reason", "") or "")] += 1
        fmt = str(row.get("format_type", "") or "")
        lang = str(row.get("language", "") or "")
        summary["by_format"][f"{lang}:{fmt}"] += 1
        summary["by_candidate_mode"][str(row.get("candidate_mode", "") or "")] += 1
        key = f"{row.get('wav', '')}|{row.get('alias', '')}"
        summary["top_aliases"][key] += 1

    summary_out = {
        "total_rows": summary["total_rows"],
        "by_reason": dict(summary["by_reason"]),
        "by_format": dict(summary["by_format"]),
        "by_candidate_mode": dict(summary["by_candidate_mode"]),
        "top_aliases": dict(summary["top_aliases"].most_common(30)),
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "selector_hard_negative_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    csv_rows = [(k, v) for k, v in summary["top_aliases"].most_common(200)]
    _write_csv(os.path.join(out_dir, "selector_hard_negative_top_aliases.csv"), ["wav|alias", "count"], csv_rows)

    print("summary:", json_path)


if __name__ == "__main__":
    main()
