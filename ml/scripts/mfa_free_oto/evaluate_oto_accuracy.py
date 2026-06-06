"""Evaluate auto-generated OTO timing against a hand-corrected gold OTO.

The gold set is a small, hand-corrected ``oto.ini`` (the user is an OTO expert,
so correcting OTO directly is the natural gold format). This compares any
auto-generated OTO against it and reports the metrics that actually matter:

- **mismapping rate**: fraction of rows whose preutterance lands far enough from
  gold that the played region is on the wrong phoneme (``--mismap-ms``).
- **boundary error**: |auto.preutter_abs - gold.preutter_abs| where
  ``preutter_abs = offset + preutterance`` (the C->V / vowel-onset the offset is
  built around), plus the offset error.
- coverage: rows present in gold but missing from auto (dropped rows).

Match is by (wav, alias, occurrence-index), so duplicate aliases are aligned in
order. Run the same selected files through the generator, then point this at the
two files.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from core.oto_file_utils import read_text_with_fallback, parse_oto_line


def _load_rows(path: str) -> dict[tuple[str, str, int], dict]:
    out: dict[tuple[str, str, int], dict] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for line in read_text_with_fallback(path).splitlines():
        if "=" not in line:
            continue
        row = parse_oto_line(line)
        if not row:
            continue
        wav = str(row["wav"]).strip().lower()
        alias = str(row["alias"]).strip()
        key2 = (wav, alias)
        idx = counts[key2]
        counts[key2] += 1
        offset = float(row["offset"])
        pre = float(row["pre"])
        out[(wav, alias, idx)] = {
            "offset": offset,
            "pre": pre,
            "preutter_abs": offset + pre,
            "alias": alias,
            "wav": wav,
        }
    return out


def evaluate(gold_path: str, auto_path: str, *, mismap_ms: float = 100.0, wavs: set[str] | None = None) -> dict:
    gold = _load_rows(gold_path)
    auto = _load_rows(auto_path)
    if wavs:
        keep = {w.strip().lower() for w in wavs if str(w).strip()}
        gold = {k: v for k, v in gold.items() if k[0] in keep}
        auto = {k: v for k, v in auto.items() if k[0] in keep}
    pre_errs: list[float] = []
    off_errs: list[float] = []
    mismapped: list[tuple[str, str, float]] = []
    missing: list[tuple[str, str]] = []
    matched = 0
    for key, g in gold.items():
        a = auto.get(key)
        if a is None:
            missing.append((key[0], key[1]))
            continue
        matched += 1
        pe = abs(a["preutter_abs"] - g["preutter_abs"])
        oe = abs(a["offset"] - g["offset"])
        pre_errs.append(pe)
        off_errs.append(oe)
        if pe > float(mismap_ms):
            mismapped.append((key[0], key[1], pe))

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "median": None, "mean": None, "p90": None}
        s = sorted(vals)
        return {
            "n": len(vals),
            "median": round(statistics.median(s), 1),
            "mean": round(statistics.fmean(s), 1),
            "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
            "within_30ms_pct": round(100.0 * sum(v <= 30 for v in s) / len(s), 1),
            "within_60ms_pct": round(100.0 * sum(v <= 60 for v in s) / len(s), 1),
        }

    return {
        "gold_rows": len(gold),
        "auto_rows": len(auto),
        "matched": matched,
        "missing_from_auto": len(missing),
        "missing_examples": missing[:10],
        "mismapped_count": len(mismapped),
        "mismapping_rate_pct": round(100.0 * len(mismapped) / matched, 1) if matched else None,
        "mismap_threshold_ms": mismap_ms,
        "preutter_abs_error_ms": _stats(pre_errs),
        "offset_error_ms": _stats(off_errs),
        "worst_mismapped": sorted(mismapped, key=lambda x: -x[2])[:10],
    }


def _format_report(r: dict) -> str:
    lines = [
        f"gold_rows={r['gold_rows']} auto_rows={r['auto_rows']} matched={r['matched']} "
        f"missing_from_auto={r['missing_from_auto']}",
        f"MISMAPPING: {r['mismapped_count']}/{r['matched']} = {r['mismapping_rate_pct']}% "
        f"(threshold {r['mismap_threshold_ms']}ms preutterance error)",
        f"preutterance-abs error: {r['preutter_abs_error_ms']}",
        f"offset error:           {r['offset_error_ms']}",
    ]
    if r["missing_examples"]:
        lines.append(f"  missing rows (dropped by auto): {r['missing_examples']}")
    if r["worst_mismapped"]:
        lines.append("  worst mismapped (wav, alias, err_ms):")
        for wav, alias, err in r["worst_mismapped"]:
            lines.append(f"    {err:7.0f}ms  {alias!r} in {wav}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate auto OTO vs hand-corrected gold OTO.")
    ap.add_argument("--gold", required=True, help="Hand-corrected gold oto.ini")
    ap.add_argument("--auto", required=True, help="Auto-generated oto.ini to score")
    ap.add_argument("--mismap-ms", type=float, default=100.0,
                    help="Preutterance error above which a row counts as mismapped.")
    ap.add_argument("--wavs-file", help="Optional: restrict evaluation to wav names listed in this file (one per line).")
    ap.add_argument("--json", action="store_true", help="Emit raw JSON instead of a report.")
    args = ap.parse_args()
    wavs = None
    if args.wavs_file:
        wavs = set(open(args.wavs_file, encoding="utf-8").read().splitlines())
    result = evaluate(args.gold, args.auto, mismap_ms=args.mismap_ms, wavs=wavs)
    if args.json:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
