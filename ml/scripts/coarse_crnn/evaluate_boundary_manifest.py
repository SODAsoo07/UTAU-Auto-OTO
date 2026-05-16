from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_eval import evaluate_decoded_rows
from core.coarse_crnn.boundary_residual import (
    apply_residual_to_decoded_row,
    load_boundary_residual_bundle,
    should_apply_residual_for_row,
)
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params, oto_row_to_absolute_anchors
from core.coarse_crnn.boundary_targets import training_rows_to_wav_groups
from core.coarse_crnn.boundary_types import DecodedOtoRow
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
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
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate boundary scorer+decoder on JSONL manifest.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-wavs", type=int, default=0)
    ap.add_argument("--residual-model-dir", default="", help="Optional LightGBM residual model dir.")
    args = ap.parse_args()

    rows = _read_jsonl(args.manifest)
    grouped = training_rows_to_wav_groups(rows)
    wav_items = sorted(grouped.items())
    if int(args.max_wavs) > 0:
        wav_items = wav_items[: int(args.max_wavs)]

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, args.device)
    model, cfg, _meta = load_boundary_checkpoint(args.model, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    residual_bundle = None
    residual_dir = os.path.abspath(str(args.residual_model_dir or "").strip()) if str(args.residual_model_dir or "").strip() else ""
    residual_error = ""
    if residual_dir and os.path.isdir(residual_dir):
        try:
            residual_bundle = load_boundary_residual_bundle(residual_dir)
        except Exception as exc:
            residual_error = str(exc)

    decoded_rows = []
    reference_rows = {}
    role_counter = defaultdict(int)
    residual_rows_changed = 0
    residual_rows_total = 0
    for wav_path, spec_anchor_rows in wav_items:
        specs = []
        for spec, anchors in spec_anchor_rows:
            specs.append(spec)
            role_counter[str(spec.role)] += 1
            reference_rows[(spec.wav_name, spec.alias, int(spec.line_index))] = {
                "offset_abs": float(anchors.offset_abs),
                "overlap_abs": float(anchors.overlap_abs),
                "pre_abs": float(anchors.pre_abs),
                "consonant_abs": float(anchors.consonant_abs),
                "cutoff_abs": float(anchors.cutoff_abs),
            }
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
        )
        if residual_bundle is None:
            decoded_rows.extend(decoded.rows)
        else:
            for row in decoded.rows:
                base_params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
                residual_rows_total += 1
                try:
                    row_quality = _quality_at_time(score_map, row.selected_time_ms, default=model_quality)
                    apply_residual, _reason = should_apply_residual_for_row(
                        row=row,
                        params_pred=base_params,
                        quality_score=row_quality,
                        default_conditional=True,
                    )
                    if apply_residual:
                        corrected_params, audit = apply_residual_to_decoded_row(
                            row=row,
                            anchor_timeline_ms=decoded.anchor_timeline_ms,
                            params_pred=base_params,
                            bundle=residual_bundle,
                        )
                        if bool(audit.get("changed", False)):
                            residual_rows_changed += 1
                        corrected_anchors = oto_row_to_absolute_anchors(
                            offset=float(corrected_params.get("offset", 0.0) or 0.0),
                            consonant=float(corrected_params.get("consonant", 0.0) or 0.0),
                            cutoff=float(corrected_params.get("cutoff", 0.0) or 0.0),
                            preutterance=float(corrected_params.get("preutterance", 0.0) or 0.0),
                            overlap=float(corrected_params.get("overlap", 0.0) or 0.0),
                            duration_ms=float(row.spec.duration_ms),
                        )
                        decoded_rows.append(
                            DecodedOtoRow(
                                spec=row.spec,
                                anchors=corrected_anchors,
                                selected_time_ms=float(row.selected_time_ms),
                                fallback_used=bool(row.fallback_used),
                                reason=str(row.reason),
                                quality_score=float(getattr(row, "quality_score", 0.0)),
                            )
                        )
                    else:
                        decoded_rows.append(row)
                except Exception:
                    decoded_rows.append(row)

    report = evaluate_decoded_rows(decoded_rows, reference_rows=reference_rows)
    report["wavs"] = len(wav_items)
    report["role_counts"] = dict(sorted(role_counter.items(), key=lambda kv: (-kv[1], kv[0])))
    report["residual_model_dir"] = residual_dir
    report["residual_enabled"] = bool(residual_bundle is not None)
    report["residual_load_error"] = str(residual_error or "")
    report["residual_rows_changed"] = int(residual_rows_changed)
    report["residual_rows_total"] = int(residual_rows_total)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _quality_at_time(score_map, time_ms: float, *, default: float) -> float:
    qs = list(getattr(score_map, "quality_scores", []) or [])
    ts = list(getattr(score_map, "times_ms", []) or [])
    if not qs or not ts:
        return max(0.0, min(1.0, float(default)))
    idx = min(range(len(ts)), key=lambda i: abs(float(ts[i]) - float(time_ms)))
    if idx < 0 or idx >= len(qs):
        return max(0.0, min(1.0, float(default)))
    return max(0.0, min(1.0, float(qs[idx])))


if __name__ == "__main__":
    raise SystemExit(main())
