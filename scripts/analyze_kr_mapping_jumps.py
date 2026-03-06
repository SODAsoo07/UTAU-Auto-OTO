from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.oto_ml_features import parse_oto_rows


def _occurrence_map(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str, int], Dict[str, object]]:
    out = {}
    for row in rows:
        key = (
            str(row.get("wav_norm", "")),
            str(row.get("alias_norm", "")),
            int(row.get("occurrence_index", 0)),
        )
        out[key] = row
    return out


def analyze(base_oto: str, auto_oto: str, jump_ms: float = 120.0, topn: int = 30) -> Dict[str, object]:
    base_rows = parse_oto_rows(base_oto, language="korean")
    auto_rows = parse_oto_rows(auto_oto, language="korean")

    base_map = _occurrence_map(base_rows)
    auto_map = _occurrence_map(auto_rows)
    keys = sorted(set(base_map.keys()) & set(auto_map.keys()))

    by_wav = defaultdict(lambda: {"count": 0, "jump_count": 0, "examples": []})
    jumps = []
    for key in keys:
        b = base_map[key]
        a = auto_map[key]
        b_pre_abs = float(b["offset"]) + float(b["pre"])
        a_pre_abs = float(a["offset"]) + float(a["pre"])
        b_cons_abs = float(b["offset"]) + float(b["cons"])
        a_cons_abs = float(a["offset"]) + float(a["cons"])
        b_cut_abs = float(b["offset"]) + abs(float(b["cutoff"]))
        a_cut_abs = float(a["offset"]) + abs(float(a["cutoff"]))

        diff_pre_abs = abs(a_pre_abs - b_pre_abs)
        diff_cons_abs = abs(a_cons_abs - b_cons_abs)
        diff_cut_abs = abs(a_cut_abs - b_cut_abs)
        jump_score = max(diff_pre_abs, diff_cons_abs, diff_cut_abs)
        wav = str(a.get("wav", ""))
        by_wav[wav]["count"] += 1
        if jump_score >= float(jump_ms):
            by_wav[wav]["jump_count"] += 1
            ex = {
                "wav": wav,
                "alias": str(a.get("alias", "")),
                "line_index": int(a.get("line_index", -1)),
                "jump_score": jump_score,
                "diff_pre_abs": diff_pre_abs,
                "diff_cons_abs": diff_cons_abs,
                "diff_cut_abs": diff_cut_abs,
            }
            jumps.append(ex)
            if len(by_wav[wav]["examples"]) < 3:
                by_wav[wav]["examples"].append(ex)

    jumps.sort(key=lambda x: x["jump_score"], reverse=True)
    wav_summary = []
    for wav, item in by_wav.items():
        count = int(item["count"])
        jump_count = int(item["jump_count"])
        wav_summary.append(
            {
                "wav": wav,
                "rows": count,
                "jump_rows": jump_count,
                "jump_ratio": (float(jump_count) / float(max(1, count))),
                "examples": item["examples"],
            }
        )
    wav_summary.sort(key=lambda x: x["jump_ratio"], reverse=True)

    return {
        "base_oto": os.path.abspath(base_oto),
        "auto_oto": os.path.abspath(auto_oto),
        "keys_compared": len(keys),
        "jump_threshold_ms": float(jump_ms),
        "jump_rows": len(jumps),
        "jump_ratio": (float(len(jumps)) / float(max(1, len(keys)))),
        "top_jumps": jumps[: int(topn)],
        "wav_summary": wav_summary[: int(topn)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze KR mapping jump outliers between reference and auto OTO.")
    ap.add_argument("--base-oto", required=True)
    ap.add_argument("--auto-oto", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--jump-ms", type=float, default=120.0)
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    result = analyze(args.base_oto, args.auto_oto, jump_ms=args.jump_ms, topn=args.topn)
    out = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", out)
    print(json.dumps({k: result[k] for k in ("keys_compared", "jump_rows", "jump_ratio", "jump_threshold_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
