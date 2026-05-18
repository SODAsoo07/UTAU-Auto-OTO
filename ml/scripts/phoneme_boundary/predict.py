from __future__ import annotations

import argparse
import json
import os

from core.phoneme_boundary.decoder import decode_expected_boundaries
from core.phoneme_boundary.inference import peak_events_from_scores, predict_boundary_scores
from core.phoneme_boundary.plan import build_phoneme_plan, load_aliases_text
from core.phoneme_boundary.types import BOUNDARY_LABELS


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict phoneme-boundary frame scores for one wav.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--aliases", default="", help="Optional text file with aliases in recording order.")
    ap.add_argument("--language", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--min-gap-ms", type=float, default=35.0)
    ap.add_argument("--acoustic-heuristics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--heuristic-weight", type=float, default=0.28)
    ap.add_argument("--decode", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    score_map = predict_boundary_scores(
        model_path=args.model,
        wav_path=args.wav,
        device=args.device,
        use_acoustic_heuristics=bool(args.acoustic_heuristics),
        heuristic_weight=float(args.heuristic_weight),
    )
    peaks = peak_events_from_scores(score_map, min_score=float(args.min_score), min_gap_ms=float(args.min_gap_ms))
    payload: dict[str, object] = {
        "wav_path": os.path.abspath(args.wav),
        "labels": list(BOUNDARY_LABELS),
        "peaks": [event.__dict__ for event in peaks],
        "phone_state_scores": score_map.phone_state_scores or {},
        "consonant_scores": score_map.consonant_scores or {},
        "vowel_scores": score_map.vowel_scores or {},
        "heuristic_scores": score_map.heuristic_scores or {},
    }
    if bool(args.decode) and str(args.aliases or "").strip():
        aliases = load_aliases_text(args.aliases)
        duration_ms = float(score_map.times_ms[-1]) if score_map.times_ms else 0.0
        plan = build_phoneme_plan(wav_path=args.wav, aliases=aliases, language=args.language, duration_ms=duration_ms)
        decoded = decode_expected_boundaries(score_map, plan)
        payload["decoded"] = [event.__dict__ for event in decoded]
    if str(args.out or "").strip():
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
