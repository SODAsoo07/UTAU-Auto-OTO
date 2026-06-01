"""Evaluate a phoneme-boundary checkpoint against a manifest.
Wraps core.phoneme_boundary.evaluation.evaluate_phoneme_boundary_manifest and
also reports tolerance-based hit rates (10/20/50ms) and per-label miss counts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.phoneme_boundary.evaluation import evaluate_phoneme_boundary_manifest
from core.phoneme_boundary.features import FeatureConfig, load_boundary_features
from core.phoneme_boundary.heuristics import compute_acoustic_boundary_scores
from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint
from core.phoneme_boundary.targets import events_from_manifest_row, read_boundary_manifest, resolve_wav_path


def summarize_boundary_score_errors(rows, score_for_row, manifest_dir, search_radius_ms):
    per_label_errors: dict[str, list[float]] = defaultdict(list)
    per_label_misses: dict[str, int] = defaultdict(int)
    per_label_total: dict[str, int] = defaultdict(int)
    used_rows = 0
    for row in rows:
        wav_path = resolve_wav_path(row, manifest_dir=manifest_dir)
        events = events_from_manifest_row(row)
        if not wav_path or not events:
            continue
        score_map = score_for_row(row, wav_path)
        used_rows += 1
        for event in events:
            per_label_total[event.label] += 1
            scores = score_map.get("scores", {}).get(event.label, [])
            times = score_map.get("times_ms", [])
            best_time = _best_time_within_radius(
                times,
                scores,
                target_ms=float(event.time_ms),
                search_radius_ms=float(search_radius_ms),
            )
            if best_time is None:
                per_label_misses[event.label] += 1
                continue
            per_label_errors[event.label].append(abs(float(best_time) - float(event.time_ms)))
    by_label = {
        label: {
            "count": len(values),
            "mae_ms": _mean(values),
            "p50_ms": _percentile(values, 50.0),
            "p90_ms": _percentile(values, 90.0),
            "missed": int(per_label_misses.get(label, 0)),
        }
        for label, values in sorted(per_label_errors.items())
    }
    all_values = [value for values in per_label_errors.values() for value in values]
    return {
        "rows": int(used_rows),
        "events": int(sum(per_label_total.values())),
        "matched_events": int(len(all_values)),
        "mae_ms": _mean(all_values),
        "p50_ms": _percentile(all_values, 50.0),
        "p90_ms": _percentile(all_values, 90.0),
        "by_label": by_label,
        "missed_by_label": dict(sorted(per_label_misses.items())),
    }


def heuristic_baseline(rows, manifest_dir, search_radius_ms):
    def score_for_row(_row, wav_path):
        feats = load_boundary_features(wav_path, FeatureConfig())
        frame_count = int(feats.features.shape[0])
        scores = compute_acoustic_boundary_scores(
            feats.samples,
            feats.sample_rate,
            frame_ms=25.0,
            hop_ms=10.0,
            frame_count=frame_count,
        )
        return {
            "times_ms": [float(idx * 10.0) for idx in range(frame_count)],
            "scores": scores,
        }

    return summarize_boundary_score_errors(rows, score_for_row, manifest_dir, search_radius_ms)


def delta_vs_baseline(model_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mae_ms_delta": float(model_report.get("mae_ms", 0.0)) - float(baseline_report.get("mae_ms", 0.0)),
        "p50_ms_delta": float(model_report.get("p50_ms", 0.0)) - float(baseline_report.get("p50_ms", 0.0)),
        "p90_ms_delta": float(model_report.get("p90_ms", 0.0)) - float(baseline_report.get("p90_ms", 0.0)),
        "by_label": {},
    }
    labels = sorted(set((model_report.get("by_label") or {})) | set((baseline_report.get("by_label") or {})))
    for label in labels:
        model = (model_report.get("by_label") or {}).get(label, {})
        base = (baseline_report.get("by_label") or {}).get(label, {})
        out["by_label"][label] = {
            "mae_ms_delta": float(model.get("mae_ms", 0.0)) - float(base.get("mae_ms", 0.0)),
            "p50_ms_delta": float(model.get("p50_ms", 0.0)) - float(base.get("p50_ms", 0.0)),
            "p90_ms_delta": float(model.get("p90_ms", 0.0)) - float(base.get("p90_ms", 0.0)),
        }
    return out


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


def _best_time_within_radius(times, scores, *, target_ms: float, search_radius_ms: float):
    lo = float(target_ms) - float(search_radius_ms)
    hi = float(target_ms) + float(search_radius_ms)
    best_time = None
    best_score = -1.0
    for idx, score in enumerate(scores):
        if idx >= len(times):
            break
        time_ms = float(times[idx])
        if time_ms < lo or time_ms > hi:
            continue
        value = float(score)
        if value > best_score:
            best_score = value
            best_time = time_ms
    return best_time


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _percentile(values, pct: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    idx = int(round((len(vals) - 1) * max(0.0, min(100.0, float(pct))) / 100.0))
    return float(vals[idx])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--search-radius-ms", type=float, default=120.0)
    ap.add_argument("--tolerances-ms", default="10,20,50")
    ap.add_argument(
        "--include-heuristic-baseline",
        action="store_true",
        help="Also evaluate raw acoustic heuristic scores and report model-vs-baseline deltas.",
    )
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
    if bool(args.include_heuristic_baseline):
        baseline = heuristic_baseline(rows, manifest_dir, float(args.search_radius_ms))
        report["heuristic_baseline"] = baseline
        report["delta_vs_heuristic"] = delta_vs_baseline(mae, baseline)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
