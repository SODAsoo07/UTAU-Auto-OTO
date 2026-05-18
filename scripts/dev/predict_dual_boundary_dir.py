"""Run BoundaryScorer and PhonemeBoundary against the same wav folder and
write a unified JSON per wav. Phase 0 of the PhonemeBoundary OTO integration:
no production change, just gather paired predictions for A/B comparison."""
from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model as bs_infer
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint as load_bs
from core.phoneme_boundary.inference import infer_boundary_scores_with_model as pb_infer
from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint as load_pb


def find_wavs(d, recursive):
    out = []
    if recursive:
        for r, _ds, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(".wav"):
                    out.append(os.path.join(r, f))
    else:
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".wav"):
                out.append(os.path.join(d, f))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bs-model", required=True, help="BoundaryScorer checkpoint")
    ap.add_argument("--pb-model", required=True, help="PhonemeBoundary checkpoint")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-wavs", type=int, default=0)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    torch = __import__("torch")
    dev = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    bs_model, bs_cfg, _ = load_bs(args.bs_model, map_location=str(dev))
    bs_model = bs_model.to(dev).eval()
    pb_model, pb_cfg, _ = load_pb(args.pb_model, map_location=str(dev))
    pb_model = pb_model.to(dev).eval()

    wavs = find_wavs(os.path.abspath(args.wav_dir), recursive=args.recursive)
    if args.max_wavs > 0:
        wavs = wavs[: args.max_wavs]
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)

    written = 0
    for wav in wavs:
        try:
            bs_sm = bs_infer(model=bs_model, config=bs_cfg, wav_path=wav, device=str(dev))
            pb_sm = pb_infer(model=pb_model, config=pb_cfg, wav_path=wav, device=str(dev),
                             use_acoustic_heuristics=True)
        except Exception as exc:
            print(f"# error {wav}: {exc}")
            continue
        payload = {
            "wav_path": os.path.abspath(wav),
            "wav_name": os.path.basename(wav),
            "bs": {
                "model": os.path.abspath(args.bs_model),
                "hop_ms": float(bs_cfg.hop_ms),
                "labels": list(bs_cfg.labels),
                "times_ms": [round(float(t), 1) for t in bs_sm.times_ms],
                "scores": {k: [round(float(v), 4) for v in vs] for k, vs in bs_sm.scores.items()},
                "quality": [round(float(v), 4) for v in (bs_sm.quality_scores or [])],
            },
            "pb": {
                "model": os.path.abspath(args.pb_model),
                "hop_ms": float(pb_cfg.hop_ms),
                "labels": list(pb_cfg.labels),
                "times_ms": [round(float(t), 1) for t in pb_sm.times_ms],
                "scores": {k: [round(float(v), 4) for v in vs] for k, vs in pb_sm.scores.items()},
                "quality": [round(float(v), 4) for v in (pb_sm.quality_scores or [])],
            },
        }
        stem = os.path.splitext(os.path.basename(wav))[0]
        out = os.path.join(os.path.abspath(args.out_dir), stem + ".dual_pred.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        written += 1
    print(json.dumps({"out_dir": os.path.abspath(args.out_dir), "wavs": written}, ensure_ascii=False, indent=2))
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
