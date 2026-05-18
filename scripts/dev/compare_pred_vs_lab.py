"""Compare PhonemeBoundary prediction JSONs against HTK lab ground truth.
For each (wav, lab, pred.json) triple: align each predicted event to the nearest
gold boundary of compatible type and report per-event errors, then aggregate."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HTK_TICK_MS = 1.0 / 10_000.0
# Match the converter's policy: only true-silence markers count as sil; vocal
# fry is excluded; breath noise (br/exh/axh) is its own bucket but still maps
# to silence_boundary at evaluation time because the trained model has no
# dedicated breath head.
SIL_SET = {"AP", "SP", "sil", "pau", "cl"}
BREATH_SET = {"br", "exh", "axh"}
DROP_SET = {"vf"}

VOWELS = set("aiueo")  # JP coarse vowel set; "N" is moraic nasal


def parse_lab(path: str) -> list[dict]:
    sil_low = {x.lower() for x in SIL_SET}
    breath_low = {x.lower() for x in BREATH_SET}
    drop_low = {x.lower() for x in DROP_SET}
    rows = []
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for raw in fh:
            p = raw.strip().split()
            if len(p) < 3: continue
            try:
                s = float(p[0]) * HTK_TICK_MS
                e = float(p[1]) * HTK_TICK_MS
            except ValueError: continue
            if e <= s: continue
            ph = p[2]
            lower = ph.lower()
            if ph in DROP_SET or lower in drop_low:
                continue
            if ph in SIL_SET or lower in sil_low:
                ph_norm = "sil"
            elif ph in BREATH_SET or lower in breath_low:
                ph_norm = "br"
            else:
                ph_norm = ph
            rows.append({"phone": ph_norm, "raw": ph, "start_ms": s, "end_ms": e})
    return rows


def is_vowel(ph: str) -> bool:
    ph = (ph or "").lower()
    return ph in VOWELS


def is_consonant(ph: str) -> bool:
    ph = (ph or "").lower()
    return ph not in {"sil"} and not is_vowel(ph) and ph != "n"  # treat N moraic separately


def gold_events_from_lab(lab_rows: list[dict]) -> dict[str, list[float]]:
    """Produce per-label gold time lists matching the model's labels."""
    out = defaultdict(list)
    for i, row in enumerate(lab_rows):
        s, e, ph = row["start_ms"], row["end_ms"], row["phone"]
        out["phone_start"].append(s)
        out["phone_end"].append(e)
        if ph in {"sil", "br"}:
            # Both true silence and breath fricative get scored against the
            # silence_boundary head — the trained model has no separate breath
            # output. We keep the distinction in the parsed rows so downstream
            # error breakdowns can still split by raw phone.
            out["silence_boundary"].append(s)
            out["silence_boundary"].append(e)
        elif is_vowel(ph):
            out["vowel_onset"].append(s)
            out["vowel_end"].append(e)
            out["vowel_nucleus"].append(0.5 * (s + e))
        elif is_consonant(ph):
            out["consonant_onset"].append(s)
    return out


def nearest_err(pred_t: float, gold_times: list[float]) -> float:
    if not gold_times: return float("inf")
    return min(abs(pred_t - g) for g in gold_times)


def analyze(pred_json: str, lab_path: str, *, tol_ms: float = 50.0) -> dict:
    payload = json.load(open(pred_json, encoding="utf-8"))
    events = payload.get("events") or []
    lab_rows = parse_lab(lab_path)
    gold = gold_events_from_lab(lab_rows)
    by_label_errs = defaultdict(list)
    extras_by_label = defaultdict(int)
    matched_gold_by_label = defaultdict(set)
    for ev in events:
        lbl = ev["label"]; t = float(ev["time_ms"]); sc = float(ev["score"])
        gold_times = gold.get(lbl, [])
        if not gold_times:
            extras_by_label[lbl] += 1
            continue
        err = nearest_err(t, gold_times)
        by_label_errs[lbl].append({"t": t, "err": err, "score": sc})
        if err <= tol_ms:
            idx = min(range(len(gold_times)), key=lambda i: abs(gold_times[i] - t))
            matched_gold_by_label[lbl].add(idx)
    # missed gold (no predicted event within tol)
    missed_by_label = {}
    for lbl, gtimes in gold.items():
        matched = matched_gold_by_label.get(lbl, set())
        missed = [gt for i, gt in enumerate(gtimes) if i not in matched]
        if missed:
            missed_by_label[lbl] = len(missed)
    return {
        "wav_name": payload.get("wav_name"),
        "lab": os.path.basename(lab_path),
        "n_lab_phones": len(lab_rows),
        "n_pred_events": len(events),
        "by_label_pred_errs": dict(by_label_errs),
        "extras_by_label": dict(extras_by_label),
        "missed_gold_by_label": missed_by_label,
        "gold_counts": {k: len(v) for k, v in gold.items()},
    }


def aggregate(per_wav: list[dict]) -> dict:
    agg_errs = defaultdict(list)
    agg_extras = defaultdict(int)
    agg_missed = defaultdict(int)
    agg_gold = defaultdict(int)
    agg_pred = defaultdict(int)
    for w in per_wav:
        for lbl, lst in w["by_label_pred_errs"].items():
            agg_errs[lbl].extend(d["err"] for d in lst)
            agg_pred[lbl] += len(lst)
        for lbl, n in w["extras_by_label"].items():
            agg_extras[lbl] += n
            agg_pred[lbl] += n
        for lbl, n in w["missed_gold_by_label"].items():
            agg_missed[lbl] += n
        for lbl, n in w["gold_counts"].items():
            agg_gold[lbl] += n
    out = {"by_label": {}, "overall": {}}
    all_errs = []
    for lbl in sorted(set(list(agg_pred.keys()) + list(agg_gold.keys()))):
        errs = sorted(agg_errs[lbl])
        all_errs.extend(errs)
        recall_50 = (1.0 - agg_missed[lbl] / max(1, agg_gold[lbl]))
        n = len(errs)
        out["by_label"][lbl] = {
            "gold_count": agg_gold[lbl],
            "pred_count": agg_pred[lbl],
            "extra_pred": agg_extras[lbl],
            "missed_gold": agg_missed[lbl],
            "recall@50ms": round(recall_50, 3),
            "matched_pred_within_label": n,
            "median_err_ms": round(errs[n // 2], 1) if errs else None,
            "p90_err_ms": round(errs[int(0.9 * (n - 1))], 1) if errs else None,
            "mae_ms": round(sum(errs)/n, 1) if errs else None,
        }
    if all_errs:
        all_errs.sort()
        out["overall"] = {
            "mae_ms": round(sum(all_errs)/len(all_errs), 1),
            "median_err_ms": round(all_errs[len(all_errs)//2], 1),
            "p90_err_ms": round(all_errs[int(0.9*(len(all_errs)-1))], 1),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--viz-dir", required=True, help="Dir with *.boundary_pred.json")
    ap.add_argument("--lab-dir", required=True, help="Dir with HTK *.lab files (paired by stem)")
    ap.add_argument("--out", default="", help="Write JSON report here")
    ap.add_argument("--tol-ms", type=float, default=50.0)
    args = ap.parse_args()

    per_wav = []
    for f in sorted(os.listdir(args.viz_dir)):
        if not f.endswith(".boundary_pred.json"): continue
        stem = f[: -len(".boundary_pred.json")]
        lab = os.path.join(args.lab_dir, stem + ".lab")
        if not os.path.isfile(lab):
            continue
        per_wav.append(analyze(os.path.join(args.viz_dir, f), lab, tol_ms=args.tol_ms))
    summary = aggregate(per_wav)
    report = {
        "viz_dir": os.path.abspath(args.viz_dir),
        "lab_dir": os.path.abspath(args.lab_dir),
        "tol_ms": args.tol_ms,
        "wavs": len(per_wav),
        "summary": summary,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
