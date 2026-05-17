from __future__ import annotations

import argparse
import json
import os

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_eval import evaluate_decoded_rows
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import load_row_specs_from_source_oto, oto_row_to_absolute_anchors
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def _build_reference_rows(
    *,
    reference_oto_path: str,
    rows_by_wav: dict[str, list],
) -> dict[tuple[str, str, int], dict[str, float]]:
    reference = os.path.abspath(str(reference_oto_path or ""))
    if not reference or not os.path.isfile(reference):
        return {}
    spec_by_key = {
        (spec.wav_name, spec.alias, int(spec.line_index)): spec
        for specs in rows_by_wav.values()
        for spec in specs
    }
    out: dict[tuple[str, str, int], dict[str, float]] = {}
    for line_idx, raw in enumerate(read_text_with_fallback(reference).splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        key = (
            str(parsed.get("wav", "") or ""),
            str(parsed.get("alias", "") or ""),
            int(line_idx),
        )
        spec = spec_by_key.get(key)
        if spec is None:
            continue
        anchors = oto_row_to_absolute_anchors(
            offset=float(parsed.get("offset", 0.0) or 0.0),
            consonant=float(parsed.get("cons", 0.0) or 0.0),
            cutoff=float(parsed.get("cutoff", 0.0) or 0.0),
            preutterance=float(parsed.get("pre", 0.0) or 0.0),
            overlap=float(parsed.get("ovl", 0.0) or 0.0),
            duration_ms=float(spec.duration_ms),
        )
        out[key] = {
            "offset_abs": float(anchors.offset_abs),
            "overlap_abs": float(anchors.overlap_abs),
            "pre_abs": float(anchors.pre_abs),
            "consonant_abs": float(anchors.consonant_abs),
            "cutoff_abs": float(anchors.cutoff_abs),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate boundary scorer decode metrics on source oto rows.")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--source-oto", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="korean")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--reference-oto",
        default="",
        help="Optional trusted reference oto.ini. When omitted, position error metrics are skipped.",
    )
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
        quality_values = [float(v) for v in list(score_map.quality_scores or [])]
        model_quality = (sum(quality_values) / float(len(quality_values))) if quality_values else 0.50
        onset_scores = [float(p.strength) for p in list(audio.onset_peaks or [])]
        audio_reliability = (sum(onset_scores) / float(len(onset_scores))) if onset_scores else 0.50
        merged = merge_candidates(
            model_candidates=peak_candidates_from_scores(score_map),
            audio_candidates=audio_candidates_to_boundary_candidates(audio),
            model_quality=model_quality,
            audio_reliability=audio_reliability,
        )
        decoded = decode_wav_rows(
            wav_path=wav_path,
            duration_ms=float(audio.duration_ms),
            row_specs=specs,
            candidates=merged,
            active_start_ms=float(audio.active_start_ms),
            active_end_ms=float(audio.active_end_ms),
            model_quality=model_quality,
            audio_reliability=audio_reliability,
            posterior_scores=score_map.scores,
            posterior_times_ms=score_map.times_ms,
            cvs_scores=score_map.cvs_scores,
            consonant_scores=score_map.consonant_scores,
            vowel_scores=score_map.vowel_scores,
            consonant_family_scores=score_map.consonant_family_scores,
            vowel_nucleus_scores=score_map.vowel_nucleus_scores,
            vowel_glide_scores=score_map.vowel_glide_scores,
        )
        decoded_rows.extend(decoded.rows)

    reference_rows = _build_reference_rows(
        reference_oto_path=str(args.reference_oto or ""),
        rows_by_wav=rows_by_wav,
    )
    report = evaluate_decoded_rows(decoded_rows, reference_rows=reference_rows or None)
    report["reference_rows_used"] = int(len(reference_rows))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
