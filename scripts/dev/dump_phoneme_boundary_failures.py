"""Dump per-row / per-event boundary prediction failures for a checkpoint.
Writes:
  <out>.events.jsonl  - every event with (wav, phone, label, gold_ms, pred_ms, err_ms)
  <out>.rows.json     - per-row summary (mean/max err, miss count)
  <out>.worst.json    - top-N worst events and worst rows"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint
from core.phoneme_boundary.targets import events_from_manifest_row, read_boundary_manifest, resolve_wav_path


def best_time(times, scores, target_ms, radius_ms):
    lo = float(target_ms) - float(radius_ms)
    hi = float(target_ms) + float(radius_ms)
    best_t = None
    best_s = -1.0
    for i, s in enumerate(scores):
        if i >= len(times):
            break
        t = float(times[i])
        if t < lo or t > hi:
            continue
        s = float(s)
        if s > best_s:
            best_s = s
            best_t = t
    return best_t, best_s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--search-radius-ms", type=float, default=120.0)
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    torch = __import__("torch")
    dev = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    model, config, _meta = load_phoneme_boundary_checkpoint(args.model, map_location=str(dev))
    model = model.to(dev).eval()

    rows = read_boundary_manifest(args.manifest)
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    events_path = args.out_prefix + ".events.jsonl"
    rows_path = args.out_prefix + ".rows.json"
    worst_path = args.out_prefix + ".worst.json"

    per_row = []
    all_events = []
    with open(events_path, "w", encoding="utf-8") as ev_fh:
        for row in rows:
            wav_path = resolve_wav_path(row, manifest_dir=manifest_dir)
            events = events_from_manifest_row(row)
            if not wav_path or not events:
                continue
            score_map = infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(dev))
            errs = []
            misses = 0
            label_errs = defaultdict(list)
            for ev in events:
                scores = score_map.scores.get(ev.label, [])
                pred_t, pred_s = best_time(score_map.times_ms, scores, ev.time_ms, args.search_radius_ms)
                record = {
                    "wav_name": row.get("wav_name") or os.path.basename(wav_path),
                    "phone": ev.phone,
                    "phone_index": ev.phone_index,
                    "label": ev.label,
                    "gold_ms": round(float(ev.time_ms), 2),
                    "pred_ms": None if pred_t is None else round(float(pred_t), 2),
                    "pred_score": None if pred_t is None else round(float(pred_s), 4),
                    "err_ms": None if pred_t is None else round(abs(float(pred_t) - float(ev.time_ms)), 2),
                    "matched": pred_t is not None,
                }
                ev_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_events.append(record)
                if pred_t is None:
                    misses += 1
                else:
                    errs.append(abs(float(pred_t) - float(ev.time_ms)))
                    label_errs[ev.label].append(abs(float(pred_t) - float(ev.time_ms)))
            per_row.append({
                "wav_name": row.get("wav_name") or os.path.basename(wav_path),
                "events": len(events),
                "matched": len(errs),
                "missed": misses,
                "mean_err_ms": round(sum(errs) / len(errs), 2) if errs else None,
                "max_err_ms": round(max(errs), 2) if errs else None,
                "p90_err_ms": round(sorted(errs)[int(0.9 * (len(errs) - 1))], 2) if errs else None,
                "by_label_mean": {l: round(sum(v)/len(v), 2) for l, v in label_errs.items()},
            })

    with open(rows_path, "w", encoding="utf-8") as fh:
        json.dump(per_row, fh, ensure_ascii=False, indent=2)

    matched_events = [e for e in all_events if e["matched"]]
    worst_events = sorted(matched_events, key=lambda e: -e["err_ms"])[: args.top_n]
    worst_rows = sorted([r for r in per_row if r["mean_err_ms"] is not None],
                        key=lambda r: -r["mean_err_ms"])[: args.top_n]
    most_missed = sorted(per_row, key=lambda r: -r["missed"])[: args.top_n]
    summary = {
        "model": os.path.abspath(args.model),
        "manifest": os.path.abspath(args.manifest),
        "rows": len(per_row),
        "events": len(all_events),
        "matched_events": len(matched_events),
        "missed_events": len(all_events) - len(matched_events),
        "worst_events": worst_events,
        "worst_rows_by_mean_err": worst_rows,
        "rows_with_most_misses": [r for r in most_missed if r["missed"] > 0],
    }
    with open(worst_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps({
        "events_jsonl": events_path,
        "rows_json": rows_path,
        "worst_json": worst_path,
        "rows": len(per_row),
        "events": len(all_events),
        "matched": len(matched_events),
        "missed": len(all_events) - len(matched_events),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
