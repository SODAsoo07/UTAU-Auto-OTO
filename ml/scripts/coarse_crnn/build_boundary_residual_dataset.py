from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_residual import build_residual_feature_row
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params, training_rows_to_wav_groups
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors
from core.coarse_crnn.oto_targets import resolve_cutoff_abs_ms
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.runtime_encoding import bootstrap_utf8_runtime


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _target_params_from_manifest_row(raw_row: dict[str, Any] | None, *, duration_ms: float, fallback_anchors: AbsoluteOtoAnchors) -> tuple[dict[str, float], str]:
    duration = max(1.0, float(duration_ms))
    if isinstance(raw_row, dict):
        # Preferred path: use already-normalized absolute anchor fields from manifest.
        off_abs = _to_float(raw_row.get("anchor_offset_ms"), float("nan"))
        ovl_abs = _to_float(raw_row.get("anchor_overlap_ms"), float("nan"))
        pre_abs = _to_float(raw_row.get("anchor_preutterance_ms"), float("nan"))
        cons_abs = _to_float(raw_row.get("anchor_consonant_ms"), float("nan"))
        cut_abs = _to_float(raw_row.get("anchor_cutoff_ms"), _to_float(raw_row.get("target_cutoff_abs_ms"), float("nan")))
        vals = (off_abs, ovl_abs, pre_abs, cons_abs, cut_abs)
        if all(np.isfinite(v) for v in vals):
            anchors = AbsoluteOtoAnchors(
                offset_abs=float(off_abs),
                overlap_abs=float(ovl_abs),
                pre_abs=float(pre_abs),
                consonant_abs=float(cons_abs),
                cutoff_abs=float(cut_abs),
            )
            return absolute_anchors_to_oto_params(anchors, duration_ms=duration), "anchor_abs_fields"

        # Fallback path: reconstruct absolute cutoff from param fields deterministically.
        off = _to_float(raw_row.get("target_offset_ms", raw_row.get("offset", fallback_anchors.offset_abs)), fallback_anchors.offset_abs)
        pre = _to_float(raw_row.get("target_preutterance_ms", raw_row.get("pre", 0.0)), 0.0)
        ovl = _to_float(raw_row.get("target_overlap_ms", raw_row.get("ovl", 0.0)), 0.0)
        cons = _to_float(raw_row.get("target_consonant_ms", raw_row.get("cons", 1.0)), 1.0)
        cutoff_raw = _to_float(raw_row.get("target_cutoff", raw_row.get("cutoff", 0.0)), 0.0)
        cutoff_abs = resolve_cutoff_abs_ms(offset_ms=float(off), cutoff=float(cutoff_raw), duration_ms=duration)
        pre_abs = float(off) + max(0.0, float(pre))
        ovl_abs = float(off) + max(0.0, float(ovl))
        cons_abs = float(off) + max(1.0, float(cons))
        anchors = AbsoluteOtoAnchors(
            offset_abs=max(0.0, float(off)),
            overlap_abs=max(0.0, float(ovl_abs)),
            pre_abs=max(0.0, float(pre_abs)),
            consonant_abs=max(1.0, float(cons_abs)),
            cutoff_abs=max(2.0, float(cutoff_abs)),
        )
        return absolute_anchors_to_oto_params(anchors, duration_ms=duration), "target_param_fields"

    return absolute_anchors_to_oto_params(fallback_anchors, duration_ms=duration), "fallback_ref_anchors"


def main() -> int:
    bootstrap_utf8_runtime()
    ap = argparse.ArgumentParser(description="Build LightGBM residual dataset from boundary decoder outputs.")
    ap.add_argument("--manifest", required=True, help="JSONL manifest with target OTO fields.")
    ap.add_argument("--model", required=True, help="Boundary scorer checkpoint (.pt).")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-wavs", type=int, default=0)
    args = ap.parse_args()

    rows = _read_jsonl(str(args.manifest))
    if not rows:
        raise SystemExit("Manifest rows not found.")
    groups = training_rows_to_wav_groups(rows)
    wav_items = sorted(groups.items())
    if int(args.max_wavs) > 0:
        wav_items = wav_items[: int(args.max_wavs)]

    voicebank_by_key = {
        (
            str(row.get("wav", "") or ""),
            str(row.get("alias", "") or ""),
            int(row.get("line_index", 0) or 0),
        ): str(
            row.get("voicebank_id")
            or row.get("voicebank")
            or row.get("singer")
            or row.get("work_dir")
            or "unknown"
        )
        for row in rows
        if isinstance(row, dict)
    }
    raw_row_by_key = {
        (
            str(row.get("wav", "") or ""),
            str(row.get("alias", "") or ""),
            int(row.get("line_index", 0) or 0),
        ): row
        for row in rows
        if isinstance(row, dict)
    }

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, str(args.device))
    model, cfg, _meta = load_boundary_checkpoint(str(args.model), map_location=str(torch_device))
    model = model.to(torch_device).eval()

    dataset_rows: list[dict[str, Any]] = []
    decode_failures = 0
    skipped_ref = 0
    role_counter = Counter()
    target_param_source_counter = Counter()
    for wav_path, spec_anchor_rows in wav_items:
        specs = [spec for spec, _ in spec_anchor_rows]
        refs = {
            (spec.wav_name, spec.alias, int(spec.line_index)): anchors
            for spec, anchors in spec_anchor_rows
        }
        try:
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
        except Exception:
            decode_failures += 1
            continue

        for row in decoded.rows:
            key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
            ref_anchors = refs.get(key)
            if ref_anchors is None:
                skipped_ref += 1
                continue
            pred_params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
            raw_ref_row = raw_row_by_key.get(key)
            target_params, target_src = _target_params_from_manifest_row(
                raw_ref_row,
                duration_ms=float(row.spec.duration_ms),
                fallback_anchors=ref_anchors,
            )
            target_param_source_counter[str(target_src)] += 1
            features = build_residual_feature_row(
                row=row,
                anchor_timeline_ms=decoded.anchor_timeline_ms,
                params_pred=pred_params,
            )
            features["voicebank_id"] = voicebank_by_key.get(key, "unknown")
            features["wav_path"] = str(row.spec.wav_path)
            features["wav_name"] = str(row.spec.wav_name)
            features["alias"] = str(row.spec.alias)
            features["line_index_int"] = int(row.spec.line_index)
            features["target_preutterance"] = float(target_params.get("preutterance", 0.0) or 0.0)
            features["target_overlap"] = float(target_params.get("overlap", 0.0) or 0.0)
            features["target_cutoff"] = float(target_params.get("cutoff", 0.0) or 0.0)
            features["target_offset"] = float(target_params.get("offset", 0.0) or 0.0)
            features["delta_offset"] = float(features["target_offset"]) - float(pred_params.get("offset", 0.0) or 0.0)
            features["delta_preutterance"] = float(features["target_preutterance"]) - float(pred_params.get("preutterance", 0.0) or 0.0)
            features["delta_overlap"] = float(features["target_overlap"]) - float(pred_params.get("overlap", 0.0) or 0.0)
            features["delta_cutoff"] = float(features["target_cutoff"]) - float(pred_params.get("cutoff", 0.0) or 0.0)
            dataset_rows.append(features)
            role_counter[str(features.get("role", "other"))] += 1

    if not dataset_rows:
        raise SystemExit("No residual rows produced.")

    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(f"pandas is required: {exc}") from exc

    frame = pd.DataFrame(dataset_rows)
    out_csv = os.path.abspath(str(args.out_csv))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "manifest": os.path.abspath(str(args.manifest)),
        "model": os.path.abspath(str(args.model)),
        "out_csv": out_csv,
        "rows": int(len(frame)),
        "wavs": int(len(wav_items)),
        "decode_failures": int(decode_failures),
        "skipped_ref_rows": int(skipped_ref),
        "role_counts": dict(sorted(role_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "target_param_source_counts": dict(sorted(target_param_source_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "delta_abs_mean": {
            "offset": float(frame["delta_offset"].abs().mean()),
            "preutterance": float(frame["delta_preutterance"].abs().mean()),
            "overlap": float(frame["delta_overlap"].abs().mean()),
            "cutoff": float(frame["delta_cutoff"].abs().mean()),
        },
    }
    out_summary = str(args.out_summary or "").strip()
    if not out_summary:
        out_summary = os.path.splitext(out_csv)[0] + ".summary.json"
    out_summary = os.path.abspath(out_summary)
    with open(out_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
