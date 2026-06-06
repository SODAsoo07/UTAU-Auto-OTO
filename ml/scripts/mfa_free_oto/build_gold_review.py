"""Select a small, diverse set of files and export a gold-review OTO to correct.

Workflow for building a hand-corrected gold set (the user is an OTO expert, so
correcting OTO directly is the natural format):

1. Generate the auto OTO for a bank as usual.
2. Run this on that auto OTO to pick ~N structurally-diverse wavs and write a
   ``gold_review.ini`` containing only those rows (seeded with the auto values).
3. Open ``gold_review.ini`` in setParam/OpenUtau and FIX the wrong rows.
4. Score any auto OTO against it with ``evaluate_oto_accuracy.py`` (restrict the
   auto OTO to the same wavs, or the evaluator ignores unmatched rows anyway).

Selection buckets by filename structure (single sustained vowel, coda/R, and
short/medium/long syllable sequences) so the gold covers the cases that actually
break (VV chains) as well as the easy ones.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

from core.oto_file_utils import read_text_with_fallback, parse_oto_line


def _structure_bucket(wav: str) -> str:
    stem = os.path.splitext(os.path.basename(wav))[0]
    low = stem.lower()
    if low.endswith("long") or re.fullmatch(r"[aeiou]{1,3}long", low):
        return "single_vowel"
    if "r" in re.split(r"[_'\- ]", low)[-1:][0] and ("r_" in low or low.endswith("r")):
        return "coda_R"
    # syllable count by separators
    parts = [p for p in re.split(r"[_'\- ]+", stem) if p]
    n = len(parts)
    if n <= 1:
        return "single_token"
    if n <= 3:
        return "seq_short"
    if n <= 5:
        return "seq_medium"
    return "seq_long"


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a diverse gold-review OTO subset to hand-correct.")
    ap.add_argument("--auto-oto", required=True, help="Auto-generated oto.ini for the bank.")
    ap.add_argument("--out", required=True, help="Output gold_review.ini (subset to correct).")
    ap.add_argument("--count", type=int, default=16, help="Target number of wavs to include.")
    ap.add_argument("--per-bucket", type=int, default=4, help="Max wavs sampled per structure bucket.")
    args = ap.parse_args()

    rows_by_wav: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for line in read_text_with_fallback(args.auto_oto).splitlines():
        if "=" not in line:
            continue
        parsed = parse_oto_line(line)
        if not parsed:
            continue
        wav = str(parsed["wav"])
        if wav not in rows_by_wav:
            order.append(wav)
        rows_by_wav[wav].append(line.rstrip("\r"))

    buckets: dict[str, list[str]] = defaultdict(list)
    for wav in order:
        buckets[_structure_bucket(wav)].append(wav)

    selected: list[str] = []
    # round-robin across buckets until we hit count
    bucket_names = sorted(buckets)
    cursors = {b: 0 for b in bucket_names}
    per_bucket_taken = {b: 0 for b in bucket_names}
    while len(selected) < args.count:
        progressed = False
        for b in bucket_names:
            if len(selected) >= args.count:
                break
            if per_bucket_taken[b] >= args.per_bucket:
                continue
            items = buckets[b]
            if cursors[b] < len(items):
                selected.append(items[cursors[b]])
                cursors[b] += 1
                per_bucket_taken[b] += 1
                progressed = True
        if not progressed:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    total_rows = 0
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        for wav in selected:
            for line in rows_by_wav[wav]:
                f.write(line + "\r\n")
                total_rows += 1

    print(f"selected {len(selected)} wavs / {total_rows} rows -> {args.out}")
    print("bucket distribution:")
    sel_set = set(selected)
    for b in bucket_names:
        n = sum(1 for w in buckets[b] if w in sel_set)
        print(f"  {b:14s}: {n} selected (of {len(buckets[b])})")
    print("\nNext: open the file in setParam/OpenUtau, fix wrong rows, save, then:")
    print(f"  python -m ml.scripts.mfa_free_oto.evaluate_oto_accuracy --gold {args.out} --auto <new_auto_oto.ini>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
