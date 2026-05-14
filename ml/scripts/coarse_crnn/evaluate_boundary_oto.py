from __future__ import annotations

import argparse
import json

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_eval import evaluate_decoded_rows
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import load_row_specs_from_source_oto
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate boundary scorer decode metrics on source oto rows.")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--source-oto", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="korean")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=args.source_oto,
        wav_dir=args.wav_dir,
        language=args.language,
        format_type=args.format_type,
    )
    if not rows_by_wav:
        raise SystemExit("No evaluable rows from source oto.")
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, args.device)
    model, cfg, _meta = load_boundary_checkpoint(args.model, map_location=str(torch_device))
    model = model.to(torch_device).eval()

    decoded_rows = []
    for wav_path, specs in sorted(rows_by_wav.items()):
        score_map = infer_boundary_scores_with_model(
            model=model,
            config=cfg,
            wav_path=wav_path,
            device=str(torch_device),
        )
        audio = compute_audio_candidates(wav_path)
        merged = merge_candidates(
            model_candidates=peak_candidates_from_scores(score_map),
            audio_candidates=audio_candidates_to_boundary_candidates(audio),
        )
        decoded = decode_wav_rows(
            wav_path=wav_path,
            duration_ms=float(audio.duration_ms),
            row_specs=specs,
            candidates=merged,
            active_start_ms=float(audio.active_start_ms),
            active_end_ms=float(audio.active_end_ms),
        )
        decoded_rows.extend(decoded.rows)

    report = evaluate_decoded_rows(decoded_rows)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

