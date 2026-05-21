"""Trace the boundary decoder on a single wav: gold vs predicted anchors.

Reproduces the exact decode path used by evaluate_boundary_manifest /
build_boundary_residual_dataset so mis-mapping can be inspected per row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import training_rows_to_wav_groups
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.torch_utils import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.runtime.runtime_encoding import bootstrap_utf8_runtime


def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            text = str(raw).strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def main() -> int:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Trace boundary decoder on one wav.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav-substr", required=True, help="Substring to match the wav/audio path.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    rows = _read_jsonl(str(args.manifest))
    sub = str(args.wav_substr).lower()
    rows = [
        r for r in rows
        if sub in str(r.get("audio", "") or "").lower() or sub in str(r.get("wav", "") or "").lower()
    ]
    if not rows:
        raise SystemExit(f"No manifest rows match: {args.wav_substr}")
    print(f"matched manifest rows: {len(rows)}")

    groups = training_rows_to_wav_groups(rows)
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, str(args.device))
    model, cfg, _meta = load_boundary_checkpoint(str(args.model), map_location=str(torch_device))
    model = model.to(torch_device).eval()

    for wav_path, spec_anchor_rows in sorted(groups.items()):
        specs = [spec for spec, _ in spec_anchor_rows]
        refs = {(s.wav_name, s.alias, int(s.line_index)): a for s, a in spec_anchor_rows}
        print("\n" + "=" * 78)
        print(f"WAV: {wav_path}")

        score_map = infer_boundary_scores_with_model(
            model=model, config=cfg, wav_path=wav_path, device=str(torch_device)
        )
        audio = compute_audio_candidates(wav_path)
        qv = [float(v) for v in list(score_map.quality_scores or [])]
        model_quality = (sum(qv) / len(qv)) if qv else 0.50
        onsets = list(audio.onset_peaks or [])
        os_scores = [float(getattr(p, "strength", 0.0)) for p in onsets]
        audio_reliability = (sum(os_scores) / len(os_scores)) if os_scores else 0.50
        print(f"duration_ms={audio.duration_ms:.1f}  active=[{audio.active_start_ms:.0f},{audio.active_end_ms:.0f}]  "
              f"model_quality={model_quality:.3f}  audio_reliability={audio_reliability:.3f}")

        print(f"\naudio onset peaks ({len(onsets)}):")
        for p in onsets:
            t = getattr(p, "time_ms", getattr(p, "time", None))
            print(f"  t={float(t):8.1f}ms  strength={float(getattr(p,'strength',0.0)):.3f}")

        merged = merge_candidates(
            model_candidates=peak_candidates_from_scores(score_map),
            audio_candidates=audio_candidates_to_boundary_candidates(audio),
            model_quality=model_quality,
            audio_reliability=audio_reliability,
        )
        print(f"\nmerged candidates ({len(merged)}):")
        for c in merged:
            print(f"  t={float(c.time_ms):8.1f}ms  kind={c.kind:<14} score={float(c.score):.3f}  src={c.source}")

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
        print(f"\nanchor_timeline_ms={['%.1f' % t for t in (decoded.anchor_timeline_ms or [])]}")
        print(f"fallback_count={decoded.fallback_count}")

        print("\nPER-ROW  (G=gold from oto.ini, P=predicted):")
        for row in decoded.rows:
            key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
            g = refs.get(key)
            p = row.anchors
            print(f"\n  [{row.spec.line_index}] alias='{row.spec.alias}' role={row.spec.role}")
            if g is not None:
                print(f"    G offset={g.offset_abs:9.1f}  pre={g.pre_abs:9.1f}  ovl={g.overlap_abs:9.1f}  "
                      f"cons={g.consonant_abs:9.1f}  cutoff={g.cutoff_abs:9.1f}")
            print(f"    P offset={p.offset_abs:9.1f}  pre={p.pre_abs:9.1f}  ovl={p.overlap_abs:9.1f}  "
                  f"cons={p.consonant_abs:9.1f}  cutoff={p.cutoff_abs:9.1f}")
            if g is not None:
                print(f"    Δoffset={p.offset_abs-g.offset_abs:+9.1f}  Δpre={p.pre_abs-g.pre_abs:+9.1f}  "
                      f"Δcutoff={p.cutoff_abs-g.cutoff_abs:+9.1f}")
            print(f"    selected_time_ms={float(row.selected_time_ms):.1f}  fallback={row.fallback_used}  "
                  f"quality={float(row.quality_score):.3f}  reason={row.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
