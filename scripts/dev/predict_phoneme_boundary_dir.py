"""Run a phoneme-boundary checkpoint over a wav directory and dump per-wav
prediction JSON files that the companion visualizer consumes. Inference-only:
no labels required, no metrics computed."""
from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.inference import infer_boundary_scores_with_model
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint


def find_wavs(wav_dir: str, *, recursive: bool) -> list[str]:
    out: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(wav_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(wav_dir)):
            if f.lower().endswith(".wav"):
                out.append(os.path.join(wav_dir, f))
    return sorted(out)


def detect_events(times_ms, scores: dict[str, list], threshold: float, min_gap_ms: float) -> list[dict]:
    events: list[dict] = []
    for label, curve in scores.items():
        last_t = -1e9
        for i, s in enumerate(curve):
            if s < threshold:
                continue
            # local max within ±1 frame
            if i > 0 and curve[i - 1] > s:
                continue
            if i + 1 < len(curve) and curve[i + 1] > s:
                continue
            t = float(times_ms[i]) if i < len(times_ms) else 0.0
            if t - last_t < min_gap_ms:
                continue
            last_t = t
            events.append({"label": label, "time_ms": round(t, 2), "score": round(float(s), 4)})
    events.sort(key=lambda e: (e["time_ms"], e["label"]))
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="Destination for per-wav JSON files.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max-wavs", type=int, default=0, help="0 = no limit.")
    ap.add_argument("--threshold", type=float, default=0.45, help="Event detection threshold.")
    ap.add_argument("--min-gap-ms", type=float, default=20.0, help="Min spacing between events of the same label.")
    ap.add_argument("--no-heuristic-blend", action="store_true",
                    help="Disable acoustic-heuristic score blending (pure model output).")
    args = ap.parse_args()

    torch = __import__("torch")
    dev = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    model, config, meta = load_phoneme_boundary_checkpoint(args.model, map_location=str(dev))
    model = model.to(dev).eval()

    wavs = find_wavs(os.path.abspath(args.wav_dir), recursive=args.recursive)
    if args.max_wavs > 0:
        wavs = wavs[: args.max_wavs]
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)

    summary = {
        "model": os.path.abspath(args.model),
        "wav_dir": os.path.abspath(args.wav_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "device": str(dev),
        "threshold": float(args.threshold),
        "wav_count": 0,
        "outputs": [],
    }
    for wav in wavs:
        try:
            sm = infer_boundary_scores_with_model(
                model=model, config=config, wav_path=wav, device=str(dev),
                use_acoustic_heuristics=not args.no_heuristic_blend,
            )
        except Exception as exc:
            summary["outputs"].append({"wav": wav, "error": str(exc)})
            continue
        events = detect_events(sm.times_ms, sm.scores, args.threshold, args.min_gap_ms)
        stem = os.path.splitext(os.path.basename(wav))[0]
        out_json = os.path.join(os.path.abspath(args.out_dir), stem + ".boundary_pred.json")
        payload = {
            "wav_path": os.path.abspath(wav),
            "wav_name": os.path.basename(wav),
            "model_path": os.path.abspath(args.model),
            "model_config": {
                "n_mels": int(config.n_mels),
                "hidden": int(config.hidden),
                "arch_type": str(config.arch_type),
                "sample_rate": int(config.sample_rate),
                "frame_ms": float(config.frame_ms),
                "hop_ms": float(config.hop_ms),
                "labels": list(config.labels),
            },
            "hop_ms": float(config.hop_ms),
            "times_ms": [round(float(t), 2) for t in sm.times_ms],
            "scores": {k: [round(float(v), 4) for v in vs] for k, vs in sm.scores.items()},
            "quality_scores": [round(float(v), 4) for v in (sm.quality_scores or [])],
            "events": events,
            "event_count": len(events),
            "threshold": float(args.threshold),
        }
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        summary["outputs"].append({"wav": wav, "json": out_json, "events": len(events)})
        summary["wav_count"] += 1

    sum_path = os.path.join(os.path.abspath(args.out_dir), "_predict_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({
        "out_dir": summary["out_dir"],
        "wav_count": summary["wav_count"],
        "errors": sum(1 for o in summary["outputs"] if o.get("error")),
        "summary": sum_path,
    }, ensure_ascii=False, indent=2))
    return 0 if summary["wav_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
