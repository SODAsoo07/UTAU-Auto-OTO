from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from contextlib import contextmanager

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
    ap.add_argument("--row-dump", default="", help="Optional JSONL path for per-row Stage2 diagnostics.")
    ap.add_argument("--summary-out", default="", help="Optional JSON path for aggregate metrics.")
    ap.add_argument("--improvement-margin-ms", type=float, default=10.0)
    ap.add_argument("--phoneme-boundary-model", default="", help="Optional PhonemeBoundary checkpoint for PB assist ablation.")
    ap.add_argument(
        "--pb-assist-ablation",
        action="store_true",
        help="Also decode with PhonemeBoundary candidates added and report ON/OFF deltas.",
    )
    ap.add_argument(
        "--vc-derivation-ablation",
        action="store_true",
        help="Also decode with internal VC derivation disabled and report same-row deltas.",
    )
    ap.add_argument(
        "--anchor-timeline-ablation",
        default="",
        help=(
            "Comma-separated Boundary Decoder anchor timeline modes to compare against the current base decode. "
            "Supported: linear, filename, hybrid, cv_slot, auto. Use 'all' for the four fixed modes."
        ),
    )
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
    pb_model = None
    pb_config = None
    pb_model_path = ""
    pb_errors: list[str] = []
    pb_requested = bool(args.pb_assist_ablation or str(args.phoneme_boundary_model or "").strip())
    if pb_requested:
        pb_model_path = _resolve_pb_model_path(str(args.phoneme_boundary_model or ""))
        if not pb_model_path:
            raise SystemExit("--pb-assist-ablation requires --phoneme-boundary-model or a discoverable PB checkpoint")
        from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint

        pb_model, pb_config, _pb_meta = load_phoneme_boundary_checkpoint(pb_model_path, map_location=device)
        pb_model = pb_model.to(device).eval()
    total = 0
    base_abs_err = [0.0] * 5
    stage2_abs_err = [0.0] * 5
    accepted = 0
    fallback = 0
    accepted_with_gold = 0
    accepted_improved = 0
    accepted_worse = 0
    accepted_neutral = 0
    missed_good_apply = 0
    ungated_good_apply = 0
    role_stats: dict[str, dict[str, float]] = {}
    vc_derived_rows = 0
    vc_derived_with_ablation = 0
    vc_derived_ablation_delta_sum = 0.0
    vc_derived_ablation_improved = 0
    vc_derived_ablation_worse = 0
    vc_derived_ablation_neutral = 0
    wavs = 0
    row_dump_path = Path(args.row_dump) if str(args.row_dump or "").strip() else None
    row_dump_fh = None
    margin_ms = max(0.0, float(args.improvement_margin_ms))
    if row_dump_path is not None:
        row_dump_path.parent.mkdir(parents=True, exist_ok=True)
        row_dump_fh = row_dump_path.open("w", encoding="utf-8", newline="\n")
    vc_derivation_base_enabled = _env_bool("UTOA_BOUNDARY_DERIVE_INTERNAL_VC_FROM_PRIMARY", False)
    anchor_timeline_base_mode = _current_anchor_timeline_mode()
    anchor_timeline_modes = _parse_anchor_timeline_modes(args.anchor_timeline_ablation)
    anchor_timeline_stats = {mode: _new_anchor_timeline_stats() for mode in anchor_timeline_modes}
    pb_assist_stats = _new_pb_assist_stats()
    pb_candidate_total = 0
    for wav_path, specs in sorted(rows_by_wav.items()):
        if int(args.max_wavs) > 0 and wavs >= int(args.max_wavs):
            break
        if not specs:
            continue
        audio = compute_audio_candidates(wav_path)
        merged = merge_candidates(
            model_candidates=[],
            audio_candidates=audio_candidates_to_boundary_candidates(audio),
        )
        pb_decoded = None
        pb_stage2 = None
        pb_cand_count = 0
        if pb_model is not None and pb_config is not None and _pb_assist_format_enabled(_format_for_specs(specs)):
            try:
                pb_cands = _phoneme_boundary_candidates_for_wav(
                    model=pb_model,
                    config=pb_config,
                    wav_path=wav_path,
                    device=device,
                )
                pb_cand_count = len(pb_cands)
                pb_candidate_total += int(pb_cand_count)
                pb_merged = merge_candidates(
                    model_candidates=pb_cands,
                    audio_candidates=audio_candidates_to_boundary_candidates(audio),
                    model_quality=0.5,
                    audio_reliability=0.5,
                    source_weight_overrides=_pb_source_weight_overrides(pb_candidate_count=pb_cand_count),
                )
                pb_decoded = decode_wav_rows(
                    wav_path=wav_path,
                    duration_ms=float(audio.duration_ms),
                    row_specs=specs,
                    candidates=pb_merged,
                    active_start_ms=float(audio.active_start_ms),
                    active_end_ms=float(audio.active_end_ms),
                    model_quality=0.5,
                    audio_reliability=0.5,
                )
                pb_stage2 = apply_stage2_to_decode(
                    decoded=pb_decoded,
                    candidates=pb_merged,
                    bundle=bundle,
                    active_start_ms=float(audio.active_start_ms),
                    active_end_ms=float(audio.active_end_ms),
                    model_quality=0.5,
                    audio_reliability=0.5,
                )
            except Exception as exc:
                pb_errors.append(f"{os.path.basename(wav_path)} pb_assist_failed: {exc}")
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
        decoded_vc_counterfactual = None
        if bool(args.vc_derivation_ablation):
            counterfactual_enabled = not bool(vc_derivation_base_enabled)
            with _temporary_env(
                "UTOA_BOUNDARY_DERIVE_INTERNAL_VC_FROM_PRIMARY",
                "1" if counterfactual_enabled else "0",
            ):
                decoded_vc_counterfactual = decode_wav_rows(
                    wav_path=wav_path,
                    duration_ms=float(audio.duration_ms),
                    row_specs=specs,
                    candidates=merged,
                    active_start_ms=float(audio.active_start_ms),
                    active_end_ms=float(audio.active_end_ms),
                    model_quality=0.5,
                    audio_reliability=0.5,
                )
        anchor_timeline_decodes = {}
        for timeline_mode in anchor_timeline_modes:
            with _temporary_env("UTOA_BOUNDARY_ANCHOR_TIMELINE", timeline_mode):
                anchor_timeline_decodes[timeline_mode] = decode_wav_rows(
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
        stage2_ungated = None
        if row_dump_fh is not None:
            stage2_ungated = apply_stage2_to_decode(
                decoded=decoded,
                candidates=merged,
                bundle=bundle,
                active_start_ms=float(audio.active_start_ms),
                active_end_ms=float(audio.active_end_ms),
                model_quality=0.5,
                audio_reliability=0.5,
                min_confidence=0.0,
                max_candidate_distance_ms=1.0e9,
            )
        accepted += int(stage2.accepted_rows)
        fallback += int(stage2.fallback_rows)
        wavs += 1
        stage2_by_line = {int(row.spec.line_index): row for row in stage2.decoded.rows}
        vc_counterfactual_by_line = (
            {int(row.spec.line_index): row for row in decoded_vc_counterfactual.rows}
            if decoded_vc_counterfactual is not None
            else {}
        )
        ungated_by_line = (
            {int(row.spec.line_index): row for row in stage2_ungated.decoded.rows}
            if stage2_ungated is not None
            else {}
        )
        anchor_timeline_by_mode = {
            mode: {int(item.spec.line_index): item for item in decoded_mode.rows}
            for mode, decoded_mode in anchor_timeline_decodes.items()
        }
        pb_decoded_by_line = (
            {int(item.spec.line_index): item for item in pb_decoded.rows}
            if pb_decoded is not None
            else {}
        )
        pb_stage2_by_line = (
            {int(item.spec.line_index): item for item in pb_stage2.decoded.rows}
            if pb_stage2 is not None
            else {}
        )
        for row in decoded.rows:
            target = targets.get((row.spec.wav_name, row.spec.alias, int(row.spec.line_index)))
            if target is None:
                continue
            stage2_row = stage2_by_line.get(int(row.spec.line_index), row)
            ungated_row = ungated_by_line.get(int(row.spec.line_index), stage2_row)
            base_vals = _anchor_tuple(row.anchors)
            stage2_vals = _anchor_tuple(stage2_row.anchors)
            ungated_vals = _anchor_tuple(ungated_row.anchors)
            target_vals = _anchor_tuple(target)
            vc_counterfactual_row = vc_counterfactual_by_line.get(int(row.spec.line_index))
            vc_counterfactual_vals = (
                _anchor_tuple(vc_counterfactual_row.anchors)
                if vc_counterfactual_row is not None
                else None
            )
            for idx in range(5):
                base_abs_err[idx] += abs(base_vals[idx] - target_vals[idx])
                stage2_abs_err[idx] += abs(stage2_vals[idx] - target_vals[idx])
            base_mean_err = _mean_abs_error(base_vals, target_vals)
            stage2_mean_err = _mean_abs_error(stage2_vals, target_vals)
            ungated_mean_err = _mean_abs_error(ungated_vals, target_vals)
            vc_counterfactual_mean_err = (
                _mean_abs_error(vc_counterfactual_vals, target_vals)
                if vc_counterfactual_vals is not None
                else None
            )
            pb_decoded_row = pb_decoded_by_line.get(int(row.spec.line_index))
            pb_stage2_row = pb_stage2_by_line.get(int(row.spec.line_index))
            pb_base_vals = _anchor_tuple(pb_decoded_row.anchors) if pb_decoded_row is not None else None
            pb_stage2_vals = _anchor_tuple(pb_stage2_row.anchors) if pb_stage2_row is not None else None
            pb_base_mean_err = _mean_abs_error(pb_base_vals, target_vals) if pb_base_vals is not None else None
            pb_stage2_mean_err = _mean_abs_error(pb_stage2_vals, target_vals) if pb_stage2_vals is not None else None
            delta_ms = stage2_mean_err - base_mean_err
            ungated_delta_ms = ungated_mean_err - base_mean_err
            pb_assist_row_dump = None
            if pb_base_mean_err is not None:
                pb_base_delta_ms = float(pb_base_mean_err) - float(base_mean_err)
                pb_stage2_delta_ms = (
                    float(pb_stage2_mean_err) - float(base_mean_err)
                    if pb_stage2_mean_err is not None
                    else None
                )
                _add_pb_assist_stats(
                    pb_assist_stats,
                    role=str(row.spec.role or "other"),
                    base_error_ms=base_mean_err,
                    pb_base_error_ms=float(pb_base_mean_err),
                    stage2_error_ms=stage2_mean_err,
                    pb_stage2_error_ms=(
                        float(pb_stage2_mean_err)
                        if pb_stage2_mean_err is not None
                        else float(pb_base_mean_err)
                    ),
                    improved=pb_base_delta_ms < -margin_ms,
                    worse=pb_base_delta_ms > margin_ms,
                    neutral=abs(pb_base_delta_ms) <= margin_ms,
                )
                pb_assist_row_dump = {
                    "candidate_count": int(pb_cand_count),
                    "base_error_ms": float(pb_base_mean_err),
                    "base_delta_ms": float(pb_base_delta_ms),
                    "stage2_error_ms": (
                        float(pb_stage2_mean_err)
                        if pb_stage2_mean_err is not None
                        else None
                    ),
                    "stage2_delta_ms": pb_stage2_delta_ms,
                    "base_anchors_ms": list(pb_base_vals) if pb_base_vals is not None else None,
                    "stage2_anchors_ms": list(pb_stage2_vals) if pb_stage2_vals is not None else None,
                    "base_reason": (
                        str(pb_decoded_row.reason or pb_decoded_row.anchors.reason or "")
                        if pb_decoded_row is not None
                        else ""
                    ),
                    "stage2_reason": (
                        str(pb_stage2_row.reason or pb_stage2_row.anchors.reason or "")
                        if pb_stage2_row is not None
                        else ""
                    ),
                }
            anchor_timeline_row_dump = {}
            for timeline_mode, rows_by_line in anchor_timeline_by_mode.items():
                timeline_row = rows_by_line.get(int(row.spec.line_index))
                if timeline_row is None:
                    continue
                timeline_vals = _anchor_tuple(timeline_row.anchors)
                timeline_err = _mean_abs_error(timeline_vals, target_vals)
                timeline_delta = float(timeline_err) - float(base_mean_err)
                _add_anchor_timeline_stats(
                    anchor_timeline_stats[timeline_mode],
                    role=str(row.spec.role or "other"),
                    mode_error_ms=timeline_err,
                    base_error_ms=base_mean_err,
                    improved=timeline_delta < -margin_ms,
                    worse=timeline_delta > margin_ms,
                    neutral=abs(timeline_delta) <= margin_ms,
                )
                anchor_timeline_row_dump[timeline_mode] = {
                    "error_ms": float(timeline_err),
                    "delta_ms": float(timeline_delta),
                    "anchors_ms": list(timeline_vals),
                    "fallback_used": bool(timeline_row.fallback_used),
                    "reason": str(timeline_row.reason or timeline_row.anchors.reason or ""),
                }
            role_key = str(row.spec.role or "other")
            _add_role_stats(
                role_stats,
                role=role_key,
                base_error_ms=base_mean_err,
                stage2_error_ms=stage2_mean_err,
                accepted=not bool(stage2_row.fallback_used),
                improved=delta_ms < -margin_ms,
                worse=delta_ms > margin_ms,
                neutral=abs(delta_ms) <= margin_ms,
            )
            base_reason = str(row.reason or row.anchors.reason or "")
            base_derived_internal_vc = base_reason == "derived:internal_vc_primary_pair"
            counterfactual_reason = (
                str(vc_counterfactual_row.reason or vc_counterfactual_row.anchors.reason or "")
                if vc_counterfactual_row is not None
                else ""
            )
            counterfactual_derived_internal_vc = counterfactual_reason == "derived:internal_vc_primary_pair"
            if base_derived_internal_vc:
                vc_derived_rows += 1
            vc_derivation_delta_ms = None
            has_derived_counterpart = base_derived_internal_vc or counterfactual_derived_internal_vc
            if has_derived_counterpart and vc_counterfactual_mean_err is not None:
                vc_derived_with_ablation += 1
                if base_derived_internal_vc:
                    derived_err = base_mean_err
                    non_derived_err = vc_counterfactual_mean_err
                else:
                    derived_err = vc_counterfactual_mean_err
                    non_derived_err = base_mean_err
                vc_derivation_delta_ms = float(derived_err) - float(non_derived_err)
                vc_derived_ablation_delta_sum += float(vc_derivation_delta_ms)
                if vc_derivation_delta_ms < -margin_ms:
                    vc_derived_ablation_improved += 1
                elif vc_derivation_delta_ms > margin_ms:
                    vc_derived_ablation_worse += 1
                else:
                    vc_derived_ablation_neutral += 1
            is_accepted = not bool(stage2_row.fallback_used)
            if is_accepted:
                accepted_with_gold += 1
                if delta_ms < -margin_ms:
                    accepted_improved += 1
                elif delta_ms > margin_ms:
                    accepted_worse += 1
                else:
                    accepted_neutral += 1
            if ungated_delta_ms < -margin_ms:
                ungated_good_apply += 1
                if not is_accepted:
                    missed_good_apply += 1
            if row_dump_fh is not None:
                row_dump_fh.write(
                    json.dumps(
                        {
                            "wav": row.spec.wav_name,
                            "wav_path": wav_path,
                            "alias": row.spec.alias,
                            "line_index": int(row.spec.line_index),
                            "role": row.spec.role,
                            "format_type": row.spec.format_type,
                            "language": row.spec.language,
                            "duration_ms": float(row.spec.duration_ms or audio.duration_ms or 0.0),
                            "accepted": bool(is_accepted),
                            "base_fallback_used": bool(row.fallback_used),
                            "base_reason": base_reason,
                            "base_derived_internal_vc": bool(base_derived_internal_vc),
                            "counterfactual_vc_derivation_enabled": not bool(vc_derivation_base_enabled)
                            if bool(args.vc_derivation_ablation)
                            else None,
                            "counterfactual_reason": counterfactual_reason,
                            "counterfactual_derived_internal_vc": bool(counterfactual_derived_internal_vc),
                            "fallback_used": bool(stage2_row.fallback_used),
                            "reason": str(stage2_row.reason or stage2_row.anchors.reason or ""),
                            "confidence": float(stage2_row.anchors.confidence),
                            "ungated_reason": str(ungated_row.reason or ungated_row.anchors.reason or ""),
                            "ungated_confidence": float(ungated_row.anchors.confidence),
                            "base_error_ms": float(base_mean_err),
                            "stage2_error_ms": float(stage2_mean_err),
                            "stage2_delta_ms": float(delta_ms),
                            "ungated_error_ms": float(ungated_mean_err),
                            "ungated_delta_ms": float(ungated_delta_ms),
                            "counterfactual_error_ms": (
                                float(vc_counterfactual_mean_err)
                                if vc_counterfactual_mean_err is not None
                                else None
                            ),
                            "vc_derivation_delta_ms": (
                                float(vc_derivation_delta_ms)
                                if vc_derivation_delta_ms is not None
                                else None
                            ),
                            "base_anchors_ms": list(base_vals),
                            "stage2_anchors_ms": list(stage2_vals),
                            "ungated_anchors_ms": list(ungated_vals),
                            "counterfactual_anchors_ms": (
                                list(vc_counterfactual_vals)
                                if vc_counterfactual_vals is not None
                                else None
                            ),
                            "anchor_timeline_base_mode": str(anchor_timeline_base_mode),
                            "anchor_timeline_ablation": anchor_timeline_row_dump,
                            "pb_assist_ablation": pb_assist_row_dump,
                            "target_anchors_ms": list(target_vals),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            total += 1
    if row_dump_fh is not None:
        row_dump_fh.close()
    denom = max(1, total)
    accepted_denom = max(1, accepted_with_gold)
    ungated_good_denom = max(1, ungated_good_apply)
    vc_derived_ablation_denom = max(1, vc_derived_with_ablation)
    result = {
        "bank": str(bank.resolve()),
        "model": str(Path(args.model).resolve()),
        "wavs": wavs,
        "rows": total,
        "accepted_rows": accepted,
        "fallback_rows": fallback,
        "accepted_rows_with_gold": accepted_with_gold,
        "accepted_improved_rows": accepted_improved,
        "accepted_worse_rows": accepted_worse,
        "accepted_neutral_rows": accepted_neutral,
        "bad_apply_rate": float(accepted_worse) / float(accepted_denom),
        "accepted_improve_rate": float(accepted_improved) / float(accepted_denom),
        "ungated_good_apply_rows": ungated_good_apply,
        "missed_good_apply_rows": missed_good_apply,
        "missed_good_apply_rate": float(missed_good_apply) / float(ungated_good_denom),
        "vc_derived_internal_rows": vc_derived_rows,
        "vc_derivation_ablation_rows": vc_derived_with_ablation,
        "vc_derivation_base_enabled": bool(vc_derivation_base_enabled),
        "vc_derivation_counterfactual_enabled": (
            not bool(vc_derivation_base_enabled)
            if bool(args.vc_derivation_ablation)
            else None
        ),
        "vc_derivation_mean_delta_ms": float(vc_derived_ablation_delta_sum) / float(vc_derived_ablation_denom),
        "vc_derivation_improved_rows": vc_derived_ablation_improved,
        "vc_derivation_worse_rows": vc_derived_ablation_worse,
        "vc_derivation_neutral_rows": vc_derived_ablation_neutral,
        "anchor_timeline_base_mode": str(anchor_timeline_base_mode),
        "anchor_timeline_ablation_modes": list(anchor_timeline_modes),
        "anchor_timeline_ablation": _finalize_anchor_timeline_stats(anchor_timeline_stats),
        "pb_assist_ablation_enabled": bool(pb_model is not None),
        "pb_assist_model": str(pb_model_path),
        "pb_assist_candidate_total": int(pb_candidate_total),
        "pb_assist_errors": list(pb_errors),
        "pb_assist_ablation": _finalize_pb_assist_stats(pb_assist_stats),
        "improvement_margin_ms": margin_ms,
        "row_dump": str(row_dump_path.resolve()) if row_dump_path is not None else "",
        "base_mae_ms": [v / denom for v in base_abs_err],
        "stage2_mae_ms": [v / denom for v in stage2_abs_err],
        "base_mean_mae_ms": sum(base_abs_err) / (denom * 5.0),
        "stage2_mean_mae_ms": sum(stage2_abs_err) / (denom * 5.0),
        "by_role": _finalize_role_stats(role_stats),
    }
    if str(args.summary_out or "").strip():
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _mean_abs_error(
    values: tuple[float, float, float, float, float],
    target: tuple[float, float, float, float, float],
) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(values, target)) / 5.0


def _add_role_stats(
    role_stats: dict[str, dict[str, float]],
    *,
    role: str,
    base_error_ms: float,
    stage2_error_ms: float,
    accepted: bool,
    improved: bool,
    worse: bool,
    neutral: bool,
) -> None:
    key = str(role or "other")
    item = role_stats.setdefault(
        key,
        {
            "rows": 0.0,
            "base_error_sum": 0.0,
            "stage2_error_sum": 0.0,
            "accepted_rows": 0.0,
            "accepted_improved_rows": 0.0,
            "accepted_worse_rows": 0.0,
            "accepted_neutral_rows": 0.0,
        },
    )
    item["rows"] += 1.0
    item["base_error_sum"] += float(base_error_ms)
    item["stage2_error_sum"] += float(stage2_error_ms)
    if bool(accepted):
        item["accepted_rows"] += 1.0
        if bool(improved):
            item["accepted_improved_rows"] += 1.0
        elif bool(worse):
            item["accepted_worse_rows"] += 1.0
        elif bool(neutral):
            item["accepted_neutral_rows"] += 1.0


def _finalize_role_stats(role_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for role, item in sorted(role_stats.items()):
        rows = max(1.0, float(item.get("rows", 0.0)))
        accepted = max(1.0, float(item.get("accepted_rows", 0.0)))
        base_mean = float(item.get("base_error_sum", 0.0)) / rows
        stage2_mean = float(item.get("stage2_error_sum", 0.0)) / rows
        out[str(role)] = {
            "rows": int(item.get("rows", 0.0)),
            "base_mean_error_ms": base_mean,
            "stage2_mean_error_ms": stage2_mean,
            "stage2_delta_ms": stage2_mean - base_mean,
            "accepted_rows": int(item.get("accepted_rows", 0.0)),
            "accepted_improved_rows": int(item.get("accepted_improved_rows", 0.0)),
            "accepted_worse_rows": int(item.get("accepted_worse_rows", 0.0)),
            "accepted_neutral_rows": int(item.get("accepted_neutral_rows", 0.0)),
            "bad_apply_rate": float(item.get("accepted_worse_rows", 0.0)) / accepted,
        }
    return out


def _resolve_pb_model_path(path_hint: str) -> str:
    from core.coarse_crnn.boundary_generator import _resolve_phoneme_boundary_model_path

    return _resolve_phoneme_boundary_model_path(path_hint)


def _phoneme_boundary_candidates_for_wav(*, model, config, wav_path: str, device: str):
    from core.coarse_crnn.boundary_generator import _phoneme_boundary_candidates_for_wav as _impl

    return _impl(model=model, config=config, wav_path=wav_path, device=device)


def _pb_assist_format_enabled(format_type: str) -> bool:
    allowed = _env_set("UTOA_BOUNDARY_PB_ASSIST_FORMATS", ("cvvc", "cvc", "cmpx"))
    if not allowed:
        return True
    return str(format_type or "").strip().lower() in allowed


def _format_for_specs(specs) -> str:
    if not specs:
        return ""
    return str(getattr(specs[0], "format_type", "") or "")


def _pb_source_weight_overrides(*, pb_candidate_count: int) -> dict[str, float] | None:
    if int(pb_candidate_count or 0) <= 0:
        return None
    return {
        "model:phoneme_boundary": _env_float("UTOA_BOUNDARY_PB_EVAL_MERGE_WEIGHT", 1.10),
    }


def _new_pb_assist_stats() -> dict[str, object]:
    return {
        "rows": 0.0,
        "base_error_sum": 0.0,
        "pb_base_error_sum": 0.0,
        "stage2_error_sum": 0.0,
        "pb_stage2_error_sum": 0.0,
        "improved_rows": 0.0,
        "worse_rows": 0.0,
        "neutral_rows": 0.0,
        "by_role": {},
    }


def _add_pb_assist_stats(
    item: dict[str, object],
    *,
    role: str,
    base_error_ms: float,
    pb_base_error_ms: float,
    stage2_error_ms: float,
    pb_stage2_error_ms: float,
    improved: bool,
    worse: bool,
    neutral: bool,
) -> None:
    _add_pb_assist_values(
        item,
        base_error_ms=base_error_ms,
        pb_base_error_ms=pb_base_error_ms,
        stage2_error_ms=stage2_error_ms,
        pb_stage2_error_ms=pb_stage2_error_ms,
        improved=improved,
        worse=worse,
        neutral=neutral,
    )
    by_role = item.setdefault("by_role", {})
    role_key = str(role or "other")
    role_item = by_role.setdefault(role_key, _new_pb_assist_stats())
    role_item["by_role"] = {}
    _add_pb_assist_values(
        role_item,
        base_error_ms=base_error_ms,
        pb_base_error_ms=pb_base_error_ms,
        stage2_error_ms=stage2_error_ms,
        pb_stage2_error_ms=pb_stage2_error_ms,
        improved=improved,
        worse=worse,
        neutral=neutral,
    )


def _add_pb_assist_values(
    item: dict[str, object],
    *,
    base_error_ms: float,
    pb_base_error_ms: float,
    stage2_error_ms: float,
    pb_stage2_error_ms: float,
    improved: bool,
    worse: bool,
    neutral: bool,
) -> None:
    item["rows"] = float(item.get("rows", 0.0)) + 1.0
    item["base_error_sum"] = float(item.get("base_error_sum", 0.0)) + float(base_error_ms)
    item["pb_base_error_sum"] = float(item.get("pb_base_error_sum", 0.0)) + float(pb_base_error_ms)
    item["stage2_error_sum"] = float(item.get("stage2_error_sum", 0.0)) + float(stage2_error_ms)
    item["pb_stage2_error_sum"] = float(item.get("pb_stage2_error_sum", 0.0)) + float(pb_stage2_error_ms)
    if bool(improved):
        item["improved_rows"] = float(item.get("improved_rows", 0.0)) + 1.0
    elif bool(worse):
        item["worse_rows"] = float(item.get("worse_rows", 0.0)) + 1.0
    elif bool(neutral):
        item["neutral_rows"] = float(item.get("neutral_rows", 0.0)) + 1.0


def _finalize_pb_assist_stats(item: dict[str, object]) -> dict[str, object]:
    rows = max(1.0, float(item.get("rows", 0.0)))
    base_mean = float(item.get("base_error_sum", 0.0)) / rows
    pb_base_mean = float(item.get("pb_base_error_sum", 0.0)) / rows
    stage2_mean = float(item.get("stage2_error_sum", 0.0)) / rows
    pb_stage2_mean = float(item.get("pb_stage2_error_sum", 0.0)) / rows
    by_role_raw = dict(item.get("by_role", {}) or {})
    return {
        "rows": int(item.get("rows", 0.0)),
        "base_mean_error_ms": base_mean,
        "pb_base_mean_error_ms": pb_base_mean,
        "pb_base_delta_ms": pb_base_mean - base_mean,
        "stage2_mean_error_ms": stage2_mean,
        "pb_stage2_mean_error_ms": pb_stage2_mean,
        "pb_stage2_delta_ms": pb_stage2_mean - base_mean,
        "improved_rows": int(item.get("improved_rows", 0.0)),
        "worse_rows": int(item.get("worse_rows", 0.0)),
        "neutral_rows": int(item.get("neutral_rows", 0.0)),
        "by_role": {
            str(role): _finalize_pb_assist_stats(dict(role_item))
            for role, role_item in sorted(by_role_raw.items())
        },
    }


def _parse_anchor_timeline_modes(raw: str) -> list[str]:
    aliases = {
        "legacy": "linear",
        "token": "filename",
        "tokens": "filename",
        "cv_slots": "cv_slot",
        "cv-segment": "cv_slot",
        "cv_segment": "cv_slot",
        "slot_cv": "cv_slot",
        "valley_onset": "hybrid",
        "ab": "hybrid",
    }
    supported = ("linear", "filename", "hybrid", "cv_slot", "auto")
    all_fixed = ("linear", "filename", "hybrid", "cv_slot")
    text = str(raw or "").strip().lower()
    if not text:
        return []
    parts = []
    for item in text.replace(";", ",").split(","):
        token = str(item or "").strip().lower()
        if not token:
            continue
        if token == "all":
            parts.extend(all_fixed)
            continue
        parts.append(aliases.get(token, token))
    out: list[str] = []
    for token in parts:
        if token not in supported:
            raise SystemExit(
                f"unsupported --anchor-timeline-ablation mode: {token} "
                f"(supported: {', '.join(supported)}, all)"
            )
        if token not in out:
            out.append(token)
    return out


def _current_anchor_timeline_mode() -> str:
    raw = str(os.environ.get("UTOA_BOUNDARY_ANCHOR_TIMELINE", "") or "").strip().lower()
    if not raw:
        return "linear"
    modes = _parse_anchor_timeline_modes(raw)
    return modes[0] if modes else "linear"


def _new_anchor_timeline_stats() -> dict[str, object]:
    return {
        "rows": 0.0,
        "base_error_sum": 0.0,
        "mode_error_sum": 0.0,
        "improved_rows": 0.0,
        "worse_rows": 0.0,
        "neutral_rows": 0.0,
        "by_role": {},
    }


def _add_anchor_timeline_stats(
    item: dict[str, object],
    *,
    role: str,
    mode_error_ms: float,
    base_error_ms: float,
    improved: bool,
    worse: bool,
    neutral: bool,
) -> None:
    item["rows"] = float(item.get("rows", 0.0)) + 1.0
    item["base_error_sum"] = float(item.get("base_error_sum", 0.0)) + float(base_error_ms)
    item["mode_error_sum"] = float(item.get("mode_error_sum", 0.0)) + float(mode_error_ms)
    if bool(improved):
        item["improved_rows"] = float(item.get("improved_rows", 0.0)) + 1.0
    elif bool(worse):
        item["worse_rows"] = float(item.get("worse_rows", 0.0)) + 1.0
    elif bool(neutral):
        item["neutral_rows"] = float(item.get("neutral_rows", 0.0)) + 1.0

    by_role = item.setdefault("by_role", {})
    role_key = str(role or "other")
    role_item = by_role.setdefault(
        role_key,
        {
            "rows": 0.0,
            "base_error_sum": 0.0,
            "mode_error_sum": 0.0,
            "improved_rows": 0.0,
            "worse_rows": 0.0,
            "neutral_rows": 0.0,
        },
    )
    role_item["rows"] += 1.0
    role_item["base_error_sum"] += float(base_error_ms)
    role_item["mode_error_sum"] += float(mode_error_ms)
    if bool(improved):
        role_item["improved_rows"] += 1.0
    elif bool(worse):
        role_item["worse_rows"] += 1.0
    elif bool(neutral):
        role_item["neutral_rows"] += 1.0


def _finalize_anchor_timeline_stats(stats_by_mode: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for mode, item in sorted(stats_by_mode.items()):
        rows = max(1.0, float(item.get("rows", 0.0)))
        base_mean = float(item.get("base_error_sum", 0.0)) / rows
        mode_mean = float(item.get("mode_error_sum", 0.0)) / rows
        out[str(mode)] = {
            "rows": int(item.get("rows", 0.0)),
            "base_mean_error_ms": base_mean,
            "mode_mean_error_ms": mode_mean,
            "delta_ms": mode_mean - base_mean,
            "improved_rows": int(item.get("improved_rows", 0.0)),
            "worse_rows": int(item.get("worse_rows", 0.0)),
            "neutral_rows": int(item.get("neutral_rows", 0.0)),
            "by_role": _finalize_anchor_timeline_role_stats(dict(item.get("by_role", {}) or {})),
        }
    return out


def _finalize_anchor_timeline_role_stats(role_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for role, item in sorted(role_stats.items()):
        rows = max(1.0, float(item.get("rows", 0.0)))
        base_mean = float(item.get("base_error_sum", 0.0)) / rows
        mode_mean = float(item.get("mode_error_sum", 0.0)) / rows
        out[str(role)] = {
            "rows": int(item.get("rows", 0.0)),
            "base_mean_error_ms": base_mean,
            "mode_mean_error_ms": mode_mean,
            "delta_ms": mode_mean - base_mean,
            "improved_rows": int(item.get("improved_rows", 0.0)),
            "worse_rows": int(item.get("worse_rows", 0.0)),
            "neutral_rows": int(item.get("neutral_rows", 0.0)),
        }
    return out


@contextmanager
def _temporary_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = str(value)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_set(name: str, default: tuple[str, ...]) -> set[str]:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return {str(item).strip().lower() for item in default if str(item).strip()}
    if raw in {"*", "all"}:
        return set()
    return {item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()}


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


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
