"""Diff two oto.ini files row-by-row (matched by wav+alias).
Reports per-(wav,alias) offset/preutterance/overlap/consonant/cutoff deltas,
plus per-parameter aggregates (mean/median/MAE). Treats `--gold` as truth."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def load(path):
    text = read_text_with_fallback(path)
    rows = {}
    for line in text.splitlines():
        parsed = parse_oto_line(line)
        if not parsed: continue
        key = (str(parsed["wav"]).strip().lower(), str(parsed["alias"]).strip())
        rows[key] = {
            "offset": float(parsed["offset"]),
            "consonant": float(parsed["cons"]),
            "cutoff": float(parsed["cutoff"]),
            "preutterance": float(parsed["pre"]),
            "overlap": float(parsed["ovl"]),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True, help="Reference oto.ini")
    ap.add_argument("--pred", required=True, help="Predicted oto.ini to compare against gold")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gold = load(args.gold); pred = load(args.pred)
    gk = set(gold.keys()); pk = set(pred.keys())
    common = sorted(gk & pk)
    only_gold = sorted(gk - pk); only_pred = sorted(pk - gk)

    params = ["offset", "preutterance", "overlap", "consonant", "cutoff"]
    diffs = defaultdict(list)
    per_row = []
    for key in common:
        g, p = gold[key], pred[key]
        d = {param: p[param] - g[param] for param in params}
        for param, v in d.items():
            diffs[param].append(v)
        per_row.append({"wav": key[0], "alias": key[1], "gold": g, "pred": p, "delta": d})

    def stats(vals):
        if not vals: return {}
        absv = [abs(v) for v in vals]
        s = sorted(absv)
        return {
            "n": len(vals),
            "mean_delta_ms": round(sum(vals)/len(vals), 1),
            "abs_median_ms": round(s[len(s)//2], 1),
            "abs_p90_ms": round(s[int(0.9*(len(s)-1))], 1),
            "MAE_ms": round(sum(absv)/len(absv), 1),
        }

    report = {
        "gold": os.path.abspath(args.gold),
        "pred": os.path.abspath(args.pred),
        "rows_gold": len(gold), "rows_pred": len(pred),
        "rows_matched": len(common),
        "only_in_gold": len(only_gold), "only_in_pred": len(only_pred),
        "by_param": {param: stats(diffs[param]) for param in params},
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n# per_row (truncated to first 60):\n")
            fh.write(json.dumps(per_row[:60], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
