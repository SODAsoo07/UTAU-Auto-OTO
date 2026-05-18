"""Evaluate a phoneme-boundary checkpoint against a manifest.
Wraps core.phoneme_boundary.evaluation.evaluate_phoneme_boundary_manifest and
also reports tolerance-based hit rates (10/20/50ms) and per-label miss counts."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from core.phoneme_boundary.evaluation import evaluate_phoneme_boundary_manifest
from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint
from core.phoneme_boundary.targets import events_from_manifest_row, read_boundary_manifest, resolve_wav_path


def hit_rates(rows, model_path, manifest_dir, device, tolerances_ms, search_radius_ms):
    torch = __import__("torch")
    dev = torch.device("cuda" if device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    model, config, _meta = load_phoneme_boundary_checkpoint(model_path, map_location=str(dev))
    model = model.to(dev).eval()
    per_label_errors: dict[str, list[float]] = defaultdict(list)
    per_label_misses: dict[str, int] = defaultdict(int)
    per_label_total: dict[str, int] = defaultdict(int)
    for row in rows:
        wav_path = resolve_wav_path(row, manifest_dir=manifest_dir)
        events = events_from_manifest_row(row)
        if not wav_path or not events:
            continue
        score_map = infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(dev))
        for event in events:
            per_label_total[event.label] += 1
            scores = score_map.scores.get(event.label, [])
            times = score_map.times_ms
            lo = float(event.time_ms) - float(search_radius_ms)
            hi = float(event.time_ms) + float(search_radius_ms)
            best_time = None
            best_score = -1.0
            for idx, sc in enumerate(scores):
                if idx >= len(times):
                    break
                t = float(times[idx])
                if t < lo or t > hi:
                    continue
                if float(sc) > best_score:
                    best_score = float(sc)
                    best_time = t
            if best_time is None:
                per_label_misses[event.label] += 1
                continue
            per_label_errors[event.label].append(abs(best_time - float(event.time_ms)))
    out: dict[str, Any] = {"by_label": {}}
    all_errs: list[float] = []
    for label, total in sorted(per_label_total.items()):
        errs = per_label_errors[label]
        all_errs.extend(errs)
        row = {"total": total, "matched": len(errs), "missed": per_label_misses[label]}
        for tol in tolerances_ms:
            hits = sum(1 for e in errs if e <= tol)
            row[f"hit@{int(tol)}ms"] = round(hits / total, 4) if total else 0.0
        out["by_label"][label] = row
    overall = {"total": sum(per_label_total.values()), "matched": len(all_errs)}
    for tol in tolerances_ms:
        hits = sum(1 for e in all_errs if e <= tol)
        overall[f"hit@{int(tol)}ms"] = round(hits / overall["total"], 4) if overall["total"] else 0.0
    out["overall"] = overall
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--search-radius-ms", type=float, default=120.0)
    ap.add_argument("--tolerances-ms", default="10,20,50")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = read_boundary_manifest(args.manifest)
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))
    mae = evaluate_phoneme_boundary_manifest(
        rows,
        model_path=args.model,
        manifest_dir=manifest_dir,
        device=args.device,
        search_radius_ms=float(args.search_radius_ms),
    )
    tols = [float(t.strip()) for t in str(args.tolerances_ms).split(",") if t.strip()]
    hits = hit_rates(rows, args.model, manifest_dir, args.device, tols, float(args.search_radius_ms))
    report = {
        "model": os.path.abspath(args.model),
        "manifest": os.path.abspath(args.manifest),
        "manifest_rows": len(rows),
        "mae": mae,
        "hits": hits,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
