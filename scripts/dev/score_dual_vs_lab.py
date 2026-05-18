"""Head-to-head accuracy comparison: BoundaryScorer vs PhonemeBoundary on the
same wavs, scored against HTK lab gold. For each equivalent label pair, detect
peaks in each model's score curve (NMS), find nearest gold boundary, and
aggregate hit-rates / MAE."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

HTK = 1.0 / 10000.0
SIL = {"AP", "SP", "sil", "pau", "br", "vf", "axh", "exh", "cl"}
SIL_LOW = {x.lower() for x in SIL}
DROP = {"vf"}
DROP_LOW = {x.lower() for x in DROP}
BREATH = {"br", "exh", "axh"}
BREATH_LOW = {x.lower() for x in BREATH}
VOWELS = set("aiueo")

# (pb_label, bs_label, gold_event_kind)
# gold_event_kind: how to derive gold times from lab intervals.
PAIRS = [
    ("vowel_onset",      "vowel_start",      "vowel_start"),
    ("vowel_end",        "vowel_end",        "vowel_end"),
    ("consonant_onset",  "consonant_onset",  "consonant_start"),
    ("phone_start",      "syllable_onset",   "any_start"),
    ("silence_boundary", "silence_boundary", "sil_or_breath_edge"),
]


def parse_lab(path):
    out = []
    for raw in open(path, encoding="utf-8-sig", errors="replace"):
        p = raw.strip().split()
        if len(p) < 3: continue
        try: s = float(p[0]) * HTK; e = float(p[1]) * HTK
        except: continue
        if e <= s: continue
        ph = p[2]; low = ph.lower()
        if ph in DROP or low in DROP_LOW: continue
        is_sil = ph in SIL or low in SIL_LOW
        is_breath = ph in BREATH or low in BREATH_LOW
        out.append({"start": s, "end": e, "phone": ph, "low": low,
                    "is_sil": is_sil or is_breath, "is_breath": is_breath})
    return out


def gold_times(lab, kind):
    res = []
    for r in lab:
        if kind == "vowel_start" and r["low"] in VOWELS:
            res.append(r["start"])
        elif kind == "vowel_end" and r["low"] in VOWELS:
            res.append(r["end"])
        elif kind == "consonant_start":
            if not r["is_sil"] and r["low"] not in VOWELS and r["low"] not in {"n"}:
                res.append(r["start"])
        elif kind == "any_start":
            if not r["is_sil"]:
                res.append(r["start"])
        elif kind == "sil_or_breath_edge":
            if r["is_sil"]:
                res.append(r["start"]); res.append(r["end"])
    return sorted(res)


def detect_peaks(times, curve, threshold=0.45, min_gap_ms=30.0):
    out = []
    last_t = -1e9
    for i, v in enumerate(curve):
        if v < threshold: continue
        if i > 0 and curve[i-1] > v: continue
        if i+1 < len(curve) and curve[i+1] > v: continue
        t = float(times[i]) if i < len(times) else 0.0
        if t - last_t < min_gap_ms: continue
        last_t = t
        out.append((t, float(v)))
    return out


def score_one(payload, lab, *, threshold=0.45, min_gap_ms=30.0, tol_ms=50.0):
    bs = payload["bs"]; pb = payload["pb"]
    bs_t = bs["times_ms"]; pb_t = pb["times_ms"]
    per_pair = {}
    for pb_lbl, bs_lbl, kind in PAIRS:
        gold = gold_times(lab, kind)
        pb_peaks = detect_peaks(pb_t, pb["scores"].get(pb_lbl, []), threshold, min_gap_ms)
        bs_peaks = detect_peaks(bs_t, bs["scores"].get(bs_lbl, []), threshold, min_gap_ms)
        def eval_side(peaks):
            errs = []; matched_gold = set()
            for t, _s in peaks:
                if not gold:
                    continue
                err = min(abs(t - g) for g in gold)
                errs.append(err)
                if err <= tol_ms:
                    idx = min(range(len(gold)), key=lambda i: abs(gold[i] - t))
                    matched_gold.add(idx)
            return {
                "peaks": len(peaks),
                "errs": errs,
                "matched_gold": len(matched_gold),
                "recall": (len(matched_gold) / len(gold)) if gold else None,
            }
        per_pair[pb_lbl] = {
            "gold_count": len(gold),
            "pb": eval_side(pb_peaks),
            "bs": eval_side(bs_peaks),
        }
    return per_pair


def aggregate(per_wav):
    agg = defaultdict(lambda: {"gold": 0, "pb_errs": [], "bs_errs": [],
                                "pb_peaks": 0, "bs_peaks": 0,
                                "pb_matched_gold": 0, "bs_matched_gold": 0})
    for w in per_wav:
        for pb_lbl, d in w.items():
            a = agg[pb_lbl]
            a["gold"] += d["gold_count"]
            a["pb_errs"].extend(d["pb"]["errs"])
            a["bs_errs"].extend(d["bs"]["errs"])
            a["pb_peaks"] += d["pb"]["peaks"]
            a["bs_peaks"] += d["bs"]["peaks"]
            a["pb_matched_gold"] += d["pb"]["matched_gold"]
            a["bs_matched_gold"] += d["bs"]["matched_gold"]
    return agg


def stats(errs, gold_total, matched, peaks):
    if not errs:
        return {"n_errs": 0, "matched_gold": matched, "gold_total": gold_total,
                "peaks": peaks, "recall@tol": None, "median_err_ms": None,
                "p90_err_ms": None, "mae_ms": None}
    s = sorted(errs)
    return {
        "n_errs": len(s), "matched_gold": matched, "gold_total": gold_total,
        "peaks": peaks,
        "recall@tol": round(matched / gold_total, 3) if gold_total else None,
        "median_err_ms": round(s[len(s)//2], 1),
        "p90_err_ms": round(s[int(0.9*(len(s)-1))], 1),
        "mae_ms": round(sum(s)/len(s), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dual-dir", required=True)
    ap.add_argument("--lab-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-gap-ms", type=float, default=30.0)
    ap.add_argument("--tol-ms", type=float, default=50.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    per_wav = []
    skipped = 0
    for jp in sorted(glob.glob(os.path.join(args.dual_dir, "*.dual_pred.json"))):
        stem = os.path.basename(jp)[:-len(".dual_pred.json")]
        lab = os.path.join(args.lab_dir, stem + ".lab")
        if not os.path.isfile(lab):
            skipped += 1; continue
        payload = json.load(open(jp, encoding="utf-8"))
        lab_rows = parse_lab(lab)
        per_wav.append(score_one(payload, lab_rows,
                                  threshold=args.threshold,
                                  min_gap_ms=args.min_gap_ms,
                                  tol_ms=args.tol_ms))

    agg = aggregate(per_wav)
    report = {
        "dual_dir": os.path.abspath(args.dual_dir),
        "lab_dir": os.path.abspath(args.lab_dir),
        "tol_ms": args.tol_ms, "threshold": args.threshold, "min_gap_ms": args.min_gap_ms,
        "wavs_scored": len(per_wav), "wavs_skipped_no_lab": skipped,
        "by_label": {},
    }
    for pb_lbl in [p[0] for p in PAIRS]:
        a = agg[pb_lbl]
        report["by_label"][pb_lbl] = {
            "PB": stats(a["pb_errs"], a["gold"], a["pb_matched_gold"], a["pb_peaks"]),
            "BS": stats(a["bs_errs"], a["gold"], a["bs_matched_gold"], a["bs_peaks"]),
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh: fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
