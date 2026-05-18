from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.coarse_crnn.boundary_candidates import audio_candidates_to_boundary_candidates, merge_candidates
from core.coarse_crnn.boundary_targets import load_row_specs_from_source_oto, oto_row_to_absolute_anchors
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.stage2_oto import apply_stage2_to_decode, load_stage2_bundle
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.oto_file_utils import parse_oto_line, read_text_with_fallback


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Stage2 OTO Assigner on a voicebank.")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--language", default="")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--max-wavs", type=int, default=128)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch = __import__("torch")
    device = _resolve_device(torch, args.device)
    bank = Path(args.bank)
    oto_path = bank / "oto.ini"
    if not oto_path.is_file():
        raise SystemExit(f"oto.ini not found: {oto_path}")
    lang = args.language or _infer_language(bank)
    fmt = args.format_type or _infer_format(bank)
    targets = _load_gold_targets(str(oto_path), str(bank))
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=str(oto_path),
        wav_dir=str(bank),
        language=lang,
        format_type=fmt,
    )
    bundle = load_stage2_bundle(args.model, map_location=device, device=device)
    total = 0
    base_abs_err = [0.0] * 5
    stage2_abs_err = [0.0] * 5
    accepted = 0
    fallback = 0
    wavs = 0
    for wav_path, specs in sorted(rows_by_wav.items()):
        if wavs >= int(args.max_wavs):
            break
        if not specs:
            continue
        audio = compute_audio_candidates(wav_path)
        merged = merge_candidates(
            model_candidates=[],
            audio_candidates=audio_candidates_to_boundary_candidates(audio),
        )
        decoded = decode_wav_rows(
            wav_path=wav_path,
            duration_ms=float(audio.duration_ms),
            row_specs=specs,
            candidates=merged,
            active_start_ms=float(audio.active_start_ms),
            active_end_ms=float(audio.active_end_ms),
            model_quality=0.5,
            audio_reliability=0.5,
        )
        stage2 = apply_stage2_to_decode(
            decoded=decoded,
            candidates=merged,
            bundle=bundle,
            active_start_ms=float(audio.active_start_ms),
            active_end_ms=float(audio.active_end_ms),
            model_quality=0.5,
            audio_reliability=0.5,
        )
        accepted += int(stage2.accepted_rows)
        fallback += int(stage2.fallback_rows)
        wavs += 1
        stage2_by_line = {int(row.spec.line_index): row for row in stage2.decoded.rows}
        for row in decoded.rows:
            target = targets.get((row.spec.wav_name, row.spec.alias, int(row.spec.line_index)))
            if target is None:
                continue
            stage2_row = stage2_by_line.get(int(row.spec.line_index), row)
            base_vals = _anchor_tuple(row.anchors)
            stage2_vals = _anchor_tuple(stage2_row.anchors)
            target_vals = _anchor_tuple(target)
            for idx in range(5):
                base_abs_err[idx] += abs(base_vals[idx] - target_vals[idx])
                stage2_abs_err[idx] += abs(stage2_vals[idx] - target_vals[idx])
            total += 1
    denom = max(1, total)
    result = {
        "bank": str(bank.resolve()),
        "model": str(Path(args.model).resolve()),
        "wavs": wavs,
        "rows": total,
        "accepted_rows": accepted,
        "fallback_rows": fallback,
        "base_mae_ms": [v / denom for v in base_abs_err],
        "stage2_mae_ms": [v / denom for v in stage2_abs_err],
        "base_mean_mae_ms": sum(base_abs_err) / (denom * 5.0),
        "stage2_mean_mae_ms": sum(stage2_abs_err) / (denom * 5.0),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if total > 0 else 1


def _load_gold_targets(oto_path: str, wav_dir: str):
    targets = {}
    text = read_text_with_fallback(oto_path)
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        wav_path = os.path.join(wav_dir, wav_name)
        duration = _wav_duration_ms(wav_path)
        anchors = oto_row_to_absolute_anchors(
            offset=float(parsed.get("offset", 0.0) or 0.0),
            consonant=float(parsed.get("consonant", 0.0) or 0.0),
            cutoff=float(parsed.get("cutoff", 0.0) or 0.0),
            preutterance=float(parsed.get("preutterance", 0.0) or 0.0),
            overlap=float(parsed.get("overlap", 0.0) or 0.0),
            duration_ms=duration,
        )
        targets[(wav_name, str(parsed.get("alias", "") or ""), int(line_idx))] = anchors
    return targets


def _anchor_tuple(anchors) -> tuple[float, float, float, float, float]:
    return (
        float(anchors.offset_abs),
        float(anchors.overlap_abs),
        float(anchors.pre_abs),
        float(anchors.consonant_abs),
        float(anchors.cutoff_abs),
    )


def _wav_duration_ms(path: str) -> float:
    import wave

    if not path or not os.path.isfile(path):
        return 1.0
    with wave.open(path, "rb") as wf:
        frames = int(wf.getnframes() or 0)
        sr = int(wf.getframerate() or 0)
    return float(frames) * 1000.0 / float(sr) if frames > 0 and sr > 0 else 1.0


def _infer_language(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "japanese" in parts:
        return "japanese"
    return "korean"


def _infer_format(path: Path) -> str:
    for item in reversed([part.lower() for part in path.parts]):
        if item in {"cv", "cvc", "cvvc", "vcv", "cmpx"}:
            return item
    return "other"


def _resolve_device(torch, value: str) -> str:
    text = str(value or "auto").strip().lower()
    if text == "cuda" and torch.cuda.is_available():
        return "cuda"
    if text == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
