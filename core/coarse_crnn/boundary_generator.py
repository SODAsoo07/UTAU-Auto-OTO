from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_audio_features import (
    DEFAULT_GAP_MS as LWC_DEFAULT_GAP_MS,
    DEFAULT_PEAK_MIN_GAP_MS as LWC_DEFAULT_MIN_GAP_MS,
    DEFAULT_THRESHOLD as LWC_DEFAULT_THRESHOLD,
    compute_long_window_candidates,
    lwc_enabled_by_env,
    lwc_float_from_env,
    lwc_scales_from_env,
)
from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_types import BoundaryCandidate
from core.coarse_crnn.bpm_prior import (
    bpm_min_confidence,
    bpm_prior_enabled,
    bpm_range_from_env,
    bpm_snap_tolerance_ms,
    compute_beat_grid,
    estimate_bpm,
    parse_manual_bpm_env,
    role_snap_target_slot,
    snap_offset_to_grid,
)
from core.coarse_crnn.consonant_class_shift import (
    combined_shift_ms,
    shift_oto_params,
)
from core.coarse_crnn.boundary_residual import (
    apply_residual_to_decoded_row,
    load_boundary_residual_bundle,
    residual_enabled_by_env,
    resolve_boundary_residual_model_dir,
    should_apply_residual_for_row,
)
from core.coarse_crnn.oto_param_builder import build_absolute_anchors_for_role
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params, load_row_specs_from_source_oto
from core.coarse_crnn.stage2_oto import (
    apply_stage2_to_decode,
    load_stage2_bundle,
    resolve_stage2_model_path,
    stage2_enabled_from_env,
)
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto
from core.oto_file_utils import parse_oto_line, read_text_with_fallback
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates


def generate_oto_with_boundary_decoder(
    *,
    wav_dir: str,
    out_path: str,
    source_oto_path: str,
    language: str,
    format_type: str = "",
    model_path: str = "",
    stage2_model_path: str = "",
    stage2_enable: bool | None = None,
    phoneme_boundary_model_path: str = "",
    device: str = "auto",
    alias_suffix: str = "",
    callback: Callable[[str], None] | None = None,
    special_aliases: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int, list[str]]:
    source = resolve_no_mfa_source_oto(wav_dir=wav_dir, source_hint=source_oto_path)
    if not source:
        return 0, 0, ["Boundary scorer OTO 생성용 source oto.ini를 찾지 못했습니다."]
    model_file = str(model_path or "").strip()
    if not model_file:
        return 0, 0, ["Boundary scorer 모델 경로가 비어 있습니다."]
    if not os.path.isfile(model_file):
        return 0, 0, [f"Boundary scorer 모델을 찾지 못했습니다: {model_file}"]

    output_file = _normalize_output_oto_path(out_path)
    if not output_file:
        return 0, 0, ["출력 oto.ini 경로가 유효하지 않습니다."]

    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=source,
        wav_dir=wav_dir,
        language=language,
        format_type=format_type,
        alias_suffix=alias_suffix,
        special_aliases=special_aliases,
    )
    total = sum(len(items) for items in rows_by_wav.values())
    if total <= 0:
        return 0, 0, ["source oto.ini에서 처리 가능한 행을 찾지 못했습니다."]
    # Source-OTO contract (2026-05-17): only alias identity and slot order
    # are read from the source file. Numerical fields (offset/preutterance/
    # overlap/consonant/cutoff) MUST NOT influence inference output —
    # otherwise a broken source-OTO would silently degrade final quality.
    # The row-shift guard below is therefore kept inert (empty dicts) and
    # its rollback code path is never reached at runtime. Class/role/coda
    # shifts still apply; those are model-derived, not source-derived.
    # BPM-prior config (read once per generation). Manual env wins; otherwise
    # auto-detect per wav. Disabled wavs (low confidence) fall through to the
    # decoded offset unchanged.
    bpm_prior_on = bool(bpm_prior_enabled())
    bpm_manual = parse_manual_bpm_env() if bpm_prior_on else None
    bpm_range_cfg = bpm_range_from_env()
    snap_tol_ms = bpm_snap_tolerance_ms()
    min_conf = bpm_min_confidence()
    bpm_cache: dict[str, object] = {}
    bpm_snap_totals = {"snapped": 0, "skipped_no_bpm": 0, "skipped_no_target": 0, "skipped_out_of_tol": 0}
    source_rows_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    source_pre_abs_by_wav: dict[str, list[float]] = {}
    single_slot_wavs = 0
    multi_row_single_slot_wavs = 0
    for specs in rows_by_wav.values():
        if not specs:
            continue
        slot_count = int(specs[0].slot_count)
        if slot_count <= 1:
            single_slot_wavs += 1
            if len(specs) > 1:
                multi_row_single_slot_wavs += 1

    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model, config, _meta = load_boundary_checkpoint(model_file, map_location=str(torch_device))
    model = model.to(torch_device).eval()
    stage2_bundle = None
    stage2_requested = bool(stage2_enable) if stage2_enable is not None else stage2_enabled_from_env(default=False)
    stage2_model_file = ""
    stage2_load_error = ""
    if stage2_requested or str(stage2_model_path or "").strip():
        stage2_model_file = resolve_stage2_model_path(stage2_model_path)
        if stage2_model_file:
            try:
                stage2_bundle = load_stage2_bundle(
                    stage2_model_file,
                    map_location=str(torch_device),
                    device=str(torch_device),
                )
            except Exception as exc:
                stage2_load_error = str(exc)
        elif stage2_requested or str(stage2_model_path or "").strip():
            stage2_load_error = "Stage2 OTO 모델을 찾지 못했습니다."
    pb_model = None
    pb_config = None
    pb_model_file = ""
    pb_load_error = ""
    pb_requested = _env_bool("UTOA_STAGE2_OTO_USE_PHONEME_BOUNDARY", False) or bool(
        str(phoneme_boundary_model_path or "").strip()
    )
    if pb_requested and (stage2_requested or stage2_bundle is not None):
        pb_model_file = _resolve_phoneme_boundary_model_path(phoneme_boundary_model_path)
        if pb_model_file:
            try:
                from core.phoneme_boundary.model import load_phoneme_boundary_checkpoint

                pb_model, pb_config, _pb_meta = load_phoneme_boundary_checkpoint(
                    pb_model_file,
                    map_location=str(torch_device),
                )
                pb_model = pb_model.to(torch_device).eval()
            except Exception as exc:
                pb_model = None
                pb_config = None
                pb_load_error = str(exc)
        else:
            pb_load_error = "PhonemeBoundary 모델을 찾지 못했습니다."
    _log(callback, f"[OTO-Boundary] source={source}")
    _log(callback, f"[OTO-Boundary] model={model_file}")
    _log(callback, f"[OTO-Boundary] device={torch_device} wavs={len(rows_by_wav)} rows={total}")
    if stage2_bundle is not None:
        _log(callback, f"[OTO-Stage2] enabled=1 model={stage2_model_file}")
        if pb_model is not None:
            _log(callback, f"[OTO-Stage2] phoneme_boundary=1 model={pb_model_file}")
        elif pb_requested:
            _log(callback, f"[OTO-Stage2] phoneme_boundary=0 load_error={pb_load_error or 'unknown'}")
    elif stage2_requested or str(stage2_model_path or "").strip():
        _log(callback, f"[OTO-Stage2] enabled=0 load_error={stage2_load_error or 'unknown'}")
    else:
        _log(callback, "[OTO-Stage2] enabled=0")
    _log(
        callback,
        f"[OTO-Boundary] slot_summary single_slot_wavs={single_slot_wavs}/{len(rows_by_wav)} "
        f"multi_row_single_slot_wavs={multi_row_single_slot_wavs}",
    )
    _log(
        callback,
        "[OTO-Boundary] guards "
        f"cutoff_right_guard={_env_bool('UTOA_BOUNDARY_CUTOFF_RIGHT_GUARD_ENABLE', True)} "
        f"cutoff_right_margin_ms={_env_float('UTOA_BOUNDARY_CUTOFF_RIGHT_MARGIN_MS', -1.0):.3f} "
        f"transition_soft_guard={_env_bool('UTOA_BOUNDARY_TRANSITION_SOFT_RIGHT_GUARD_ENABLE', True)} "
        "source_param_policy=alias_only"
    )
    _lwc_on = lwc_enabled_by_env(default=True)
    _log(
        callback,
        "[OTO-Boundary] lwc "
        f"enable={_lwc_on} "
        f"scales_ms={','.join(f'{v:.1f}' for v in lwc_scales_from_env())} "
        f"gap_ms={lwc_float_from_env('UTOA_BOUNDARY_LWC_GAP_MS', LWC_DEFAULT_GAP_MS):.1f} "
        f"threshold={lwc_float_from_env('UTOA_BOUNDARY_LWC_THRESHOLD', LWC_DEFAULT_THRESHOLD):.3f} "
        f"min_gap_ms={lwc_float_from_env('UTOA_BOUNDARY_LWC_MIN_GAP_MS', LWC_DEFAULT_MIN_GAP_MS):.1f}"
    )
    residual_bundle = None
    residual_rows_changed = 0
    residual_rows_total = 0
    residual_load_error = ""
    residual_model_dir = resolve_boundary_residual_model_dir(boundary_model_path=model_file)
    residual_on = residual_enabled_by_env(default=True)
    if residual_on and residual_model_dir and os.path.isdir(residual_model_dir):
        try:
            residual_bundle = load_boundary_residual_bundle(residual_model_dir)
        except Exception as exc:
            residual_load_error = str(exc)
    if residual_bundle is not None:
        _log(
            callback,
            f"[OTO-Boundary-Residual] enabled=1 model_dir={residual_model_dir} "
            f"active_targets={','.join(getattr(residual_bundle, 'active_targets', []) or [])}",
        )
    elif residual_on and residual_model_dir and residual_load_error:
        _log(callback, f"[OTO-Boundary-Residual] enabled=0 load_error={residual_load_error}")
    else:
        _log(callback, "[OTO-Boundary-Residual] enabled=0")

    decoded_params: dict[tuple[str, str, int], dict[str, float]] = {}
    spec_by_key = {
        (spec.wav_name, spec.alias, int(spec.line_index)): spec
        for specs in rows_by_wav.values()
        for spec in specs
    }
    errors: list[str] = []
    shift_guard_totals = {
        "rows_checked": 0,
        "rows_supported": 0,
        "one_step_rows": 0,
        "mis_mapping_rows": 0,
        "row_rollbacks": 0,
        "wav_rollbacks": 0,
        "source_reject_rows": 0,
    }
    for wav_idx, (wav_path, specs) in enumerate(sorted(rows_by_wav.items()), start=1):
        try:
            score_map = infer_boundary_scores_with_model(
                model=model,
                config=config,
                wav_path=wav_path,
                device=str(torch_device),
            )
            audio = compute_audio_candidates(wav_path)
            model_quality = _mean_quality(score_map.quality_scores)
            audio_reliability = _estimate_audio_reliability(audio)
            model_cands = peak_candidates_from_scores(score_map)
            audio_cands = audio_candidates_to_boundary_candidates(audio)
            pb_cand_count = 0
            if stage2_bundle is not None and pb_model is not None and pb_config is not None:
                try:
                    pb_cands = _phoneme_boundary_candidates_for_wav(
                        model=pb_model,
                        config=pb_config,
                        wav_path=wav_path,
                        device=str(torch_device),
                    )
                    pb_cand_count = len(pb_cands)
                    model_cands = model_cands + pb_cands
                except Exception as exc:
                    errors.append(f"{os.path.basename(wav_path)} phoneme_boundary_failed: {exc}")
            lwc_cand_count = 0
            if lwc_enabled_by_env(default=True):
                try:
                    lwc_threshold = lwc_float_from_env("UTOA_BOUNDARY_LWC_THRESHOLD", LWC_DEFAULT_THRESHOLD)
                    if _is_weak_audio_reliability(audio_reliability):
                        lwc_threshold = max(
                            0.10,
                            float(lwc_threshold)
                            - _env_float("UTOA_BOUNDARY_LWC_WEAK_THRESHOLD_RELAX", 0.06),
                        )
                    lwc_cands = compute_long_window_candidates(
                        wav_path=wav_path,
                        scales_ms=lwc_scales_from_env(),
                        gap_ms=lwc_float_from_env("UTOA_BOUNDARY_LWC_GAP_MS", LWC_DEFAULT_GAP_MS),
                        threshold=float(lwc_threshold),
                        min_gap_ms=lwc_float_from_env("UTOA_BOUNDARY_LWC_MIN_GAP_MS", LWC_DEFAULT_MIN_GAP_MS),
                        active_start_ms=float(audio.active_start_ms),
                        active_end_ms=float(audio.active_end_ms),
                    )
                    lwc_cand_count = len(lwc_cands)
                    audio_cands = audio_cands + lwc_cands
                except Exception as exc:
                    errors.append(f"{os.path.basename(wav_path)} lwc_failed: {exc}")
            merged = merge_candidates(
                model_candidates=model_cands,
                audio_candidates=audio_cands,
                model_quality=model_quality,
                audio_reliability=audio_reliability,
                source_weight_overrides=_stage2_source_weight_overrides(
                    pb_candidate_count=pb_cand_count,
                    boundary_scorer_quality=model_quality,
                ),
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
            if stage2_bundle is not None:
                try:
                    stage2_result = apply_stage2_to_decode(
                        decoded=decoded,
                        candidates=merged,
                        bundle=stage2_bundle,
                        active_start_ms=float(audio.active_start_ms),
                        active_end_ms=float(audio.active_end_ms),
                        model_quality=model_quality,
                        audio_reliability=audio_reliability,
                    )
                    decoded = stage2_result.decoded
                    if stage2_result.errors:
                        errors.extend(f"{os.path.basename(wav_path)} stage2_failed: {msg}" for msg in stage2_result.errors)
                    _log(
                        callback,
                        f"[OTO-Stage2] wav={wav_idx}/{len(rows_by_wav)} "
                        f"accepted={stage2_result.accepted_rows}/{len(decoded.rows)} "
                        f"fallback={stage2_result.fallback_rows}",
                    )
                except Exception as exc:
                    errors.append(f"{os.path.basename(wav_path)} stage2_failed: {exc}")
            selected_times = [round(float(item.selected_time_ms), 1) for item in decoded.rows]
            unique_selected = len(set(selected_times))
            reused_selected = max(0, len(selected_times) - unique_selected)
            _log(
                callback,
                f"[OTO-Boundary] wav={wav_idx}/{len(rows_by_wav)} peaks={len(model_cands)} "
                f"pb={pb_cand_count} lwc={lwc_cand_count} merged={len(merged)} "
                f"fallback={decoded.fallback_count}/{len(decoded.rows)} "
                f"selected_unique={unique_selected}/{len(selected_times)} reused={reused_selected} "
                f"quality={model_quality:.3f} audio_rel={audio_reliability:.3f}",
            )
            row_entries: list[dict[str, object]] = []
            for row in decoded.rows:
                params = absolute_anchors_to_oto_params(row.anchors, duration_ms=float(row.spec.duration_ms))
                if residual_bundle is not None:
                    residual_rows_total += 1
                    try:
                        row_quality = _quality_at_time(score_map, row.selected_time_ms, default=model_quality)
                        apply_residual, residual_reason = should_apply_residual_for_row(
                            row=row,
                            params_pred=params,
                            quality_score=row_quality,
                            default_conditional=True,
                        )
                        if apply_residual:
                            corrected, audit = apply_residual_to_decoded_row(
                                row=row,
                                anchor_timeline_ms=decoded.anchor_timeline_ms,
                                params_pred=params,
                                bundle=residual_bundle,
                            )
                            params = corrected
                            if bool(audit.get("changed", False)):
                                residual_rows_changed += 1
                        else:
                            _log(
                                callback,
                                f"[OTO-Boundary-Residual] skip wav={row.spec.wav_name} alias={row.spec.alias} reason={residual_reason}",
                            )
                    except Exception as exc:
                        errors.append(f"{row.spec.wav_name}:{row.spec.alias} residual_failed: {exc}")
                row_entries.append({"row": row, "params": dict(params)})
            wav_key = os.path.basename(str(wav_path or "")).strip().lower()
            guarded_rows, guard_stats = _apply_row_shift_guard_for_wav(
                row_entries=row_entries,
                source_pre_abs=source_pre_abs_by_wav.get(wav_key, []),
                source_rows_by_key=source_rows_by_key,
                callback=callback,
            )
            for stat_key in shift_guard_totals:
                shift_guard_totals[stat_key] += int(guard_stats.get(stat_key, 0) or 0)
            for item in guarded_rows:
                row_obj = item.get("row")
                params_obj = item.get("params")
                if row_obj is None or not isinstance(params_obj, dict):
                    continue
                # Per-consonant-class + per-role offset shift (model-bias
                # compensation: voiced stops detected early, sonorants late;
                # V-CV rows lose the consonant transient and need a positive
                # role shift to land on the next syllable). All shifts default
                # to 0 — only applies when env table set.
                shift_ms = combined_shift_ms(
                    alias=str(row_obj.spec.alias or ""),
                    role=str(row_obj.spec.role or ""),
                )
                if abs(shift_ms) >= 0.5:
                    params_obj = shift_oto_params(
                        params_obj,
                        shift_ms=shift_ms,
                        duration_ms=float(row_obj.spec.duration_ms),
                    )
                # BPM beat-grid snap (2026-05-17): for BGM-recorded banks,
                # snap snap-eligible roles (vv / coda-bridge vc / head CV)
                # to the nearest beat within tolerance. Disabled wavs (low
                # confidence) and ineligible roles fall through unchanged.
                if bpm_prior_on:
                    bpm_est = bpm_cache.get(str(wav_path))
                    if bpm_est is None and str(wav_path) not in bpm_cache:
                        if bpm_manual is not None:
                            bpm_est = bpm_manual
                        else:
                            bpm_est = estimate_bpm(
                                str(wav_path),
                                int(row_obj.spec.slot_count),
                                bpm_range=bpm_range_cfg,
                            )
                        bpm_cache[str(wav_path)] = bpm_est
                    if bpm_est is not None and float(getattr(bpm_est, "confidence", 0.0)) >= min_conf:
                        target_slot = role_snap_target_slot(
                            str(row_obj.spec.role or ""),
                            int(row_obj.spec.slot_index),
                            int(row_obj.spec.slot_count),
                            alias=str(row_obj.spec.alias or ""),
                        )
                        if target_slot is None:
                            bpm_snap_totals["skipped_no_target"] += 1
                        else:
                            beat_grid = compute_beat_grid(
                                bpm_est,
                                int(row_obj.spec.slot_count),
                                duration_ms=float(row_obj.spec.duration_ms),
                            )
                            if 0 <= int(target_slot) < len(beat_grid):
                                target_beat = float(beat_grid[int(target_slot)])
                                current_off = float(params_obj.get("offset", 0.0) or 0.0)
                                delta = target_beat - current_off
                                if abs(delta) <= float(snap_tol_ms):
                                    params_obj = shift_oto_params(
                                        params_obj,
                                        shift_ms=delta,
                                        duration_ms=float(row_obj.spec.duration_ms),
                                    )
                                    bpm_snap_totals["snapped"] += 1
                                else:
                                    bpm_snap_totals["skipped_out_of_tol"] += 1
                    else:
                        bpm_snap_totals["skipped_no_bpm"] += 1
                key = (row_obj.spec.wav_name, row_obj.spec.alias, int(row_obj.spec.line_index))
                decoded_params[key] = dict(params_obj)
        except Exception as exc:
            errors.append(f"{os.path.basename(wav_path)}: {exc}")

    processed = 0
    output_lines: list[str] = []
    occurrence: dict[tuple[str, str], int] = defaultdict(int)
    for line_idx, raw in enumerate(read_text_with_fallback(source).splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        alias = str(parsed.get("alias", "") or "")
        pair = (wav_name, alias)
        occ = occurrence[pair]
        occurrence[pair] += 1
        key = (wav_name, alias, line_idx)
        params = decoded_params.get(key)
        if params is None:
            spec = spec_by_key.get(key)
            params = _alias_only_fallback_params(spec=spec)
            if occ == 0:
                errors.append(f"{wav_name}:{alias} decode miss -> alias-only fallback")
        else:
            processed += 1
        alias_out = _apply_suffix(alias, alias_suffix)
        output_lines.append(
            f"{wav_name}={alias_out},{params['offset']:.3f},{params['consonant']:.3f},"
            f"{params['cutoff']:.3f},{params['preutterance']:.3f},{params['overlap']:.3f}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(output_lines) + ("\n" if output_lines else ""))
    if residual_bundle is not None:
        _log(
            callback,
            f"[OTO-Boundary-Residual] rows_changed={residual_rows_changed}/{max(1, residual_rows_total)}",
        )
    _log(
        callback,
        "[OTO-Boundary-ShiftGuard] "
        f"rows_checked={shift_guard_totals['rows_checked']} "
        f"supported={shift_guard_totals['rows_supported']} "
        f"one_step={shift_guard_totals['one_step_rows']} "
        f"mis_mapping={shift_guard_totals['mis_mapping_rows']} "
        f"row_rollbacks={shift_guard_totals['row_rollbacks']} "
        f"wav_rollbacks={shift_guard_totals['wav_rollbacks']} "
        f"source_reject={shift_guard_totals.get('source_reject_rows', 0)}",
    )
    if bpm_prior_on:
        bpm_resolved = sum(1 for v in bpm_cache.values() if v is not None)
        _log(
            callback,
            "[OTO-Boundary-BpmPrior] "
            f"wavs_resolved={bpm_resolved}/{len(bpm_cache)} "
            f"snapped={bpm_snap_totals['snapped']} "
            f"skipped_no_bpm={bpm_snap_totals['skipped_no_bpm']} "
            f"skipped_no_target={bpm_snap_totals['skipped_no_target']} "
            f"skipped_out_of_tol={bpm_snap_totals['skipped_out_of_tol']}",
        )
    _log(callback, f"[OTO-Boundary] wrote={output_file} processed={processed}/{total}")
    return int(processed), int(total), errors


def _normalize_output_oto_path(out_path: str) -> str:
    raw = str(out_path or "").strip()
    if not raw:
        return ""
    if os.path.isdir(raw):
        return os.path.join(os.path.abspath(raw), "oto.ini")
    if raw.lower().endswith(".ini"):
        return os.path.abspath(raw)
    return os.path.join(os.path.abspath(raw), "oto.ini")


_PB_TO_BOUNDARY_KIND = {
    "phone_start": "syllable_onset",
    "phone_end": "vowel_end",
    "consonant_onset": "consonant_onset",
    "vowel_onset": "vowel_start",
    "vowel_nucleus": "vowel_stable",
    "vowel_end": "vowel_end",
    "silence_boundary": "silence_boundary",
}

_PB_LABEL_SCORE_WEIGHTS = {
    "phone_start": 0.84,
    "phone_end": 0.78,
    "consonant_onset": 0.92,
    "vowel_onset": 1.00,
    "vowel_nucleus": 0.96,
    "vowel_end": 0.90,
    "silence_boundary": 0.58,
}


def _stage2_source_weight_overrides(
    *,
    pb_candidate_count: int,
    boundary_scorer_quality: float,
) -> dict[str, float] | None:
    if int(pb_candidate_count or 0) <= 0:
        return None
    pb_weight = _clamp(_env_float("UTOA_STAGE2_OTO_PB_MERGE_WEIGHT", 1.10), 0.0, 5.0)
    bs_weight = _clamp(_env_float("UTOA_STAGE2_OTO_BS_MERGE_WEIGHT", 0.56), 0.0, 5.0)
    low_quality_threshold = _env_float("UTOA_STAGE2_OTO_BS_LOW_QUALITY_THRESHOLD", 0.45)
    if float(boundary_scorer_quality or 0.0) < float(low_quality_threshold):
        scale = _clamp(_env_float("UTOA_STAGE2_OTO_BS_LOW_QUALITY_SCALE", 0.72), 0.0, 1.0)
        bs_weight *= scale
    return {
        "model:phoneme_boundary": pb_weight,
        "model": bs_weight,
    }


def _phoneme_boundary_candidates_for_wav(*, model, config, wav_path: str, device: str) -> list[BoundaryCandidate]:
    from core.phoneme_boundary.inference import infer_boundary_scores_with_model, peak_events_from_scores

    score_map = infer_boundary_scores_with_model(
        model=model,
        config=config,
        wav_path=wav_path,
        device=device,
        use_acoustic_heuristics=_env_bool("UTOA_STAGE2_OTO_PB_HEURISTICS_ENABLE", True),
        heuristic_weight=_env_float("UTOA_STAGE2_OTO_PB_HEURISTIC_WEIGHT", 0.22),
    )
    min_score = _env_float("UTOA_STAGE2_OTO_PB_MIN_SCORE", 0.38)
    min_gap = _env_float("UTOA_STAGE2_OTO_PB_MIN_GAP_MS", 28.0)
    weight = _clamp(_env_float("UTOA_STAGE2_OTO_PB_SCORE_WEIGHT", 0.92), 0.0, 1.0)
    out: list[BoundaryCandidate] = []
    for event in peak_events_from_scores(score_map, min_score=min_score, min_gap_ms=min_gap):
        label = str(event.label)
        kind = _PB_TO_BOUNDARY_KIND.get(label, "")
        if not kind:
            continue
        label_weight = _phoneme_boundary_label_score_weight(label)
        out.append(
            BoundaryCandidate(
                time_ms=float(event.time_ms),
                kind=kind,
                score=_clamp(float(event.confidence) * weight * label_weight, 0.0, 1.0),
                source=f"model:phoneme_boundary:{label}",
            )
        )
    return out


def _phoneme_boundary_label_score_weight(label: str) -> float:
    key = str(label or "").strip()
    default = float(_PB_LABEL_SCORE_WEIGHTS.get(key, 0.84))
    if key == "silence_boundary":
        default = _env_float("UTOA_STAGE2_OTO_PB_SILENCE_SCORE_WEIGHT", default)
    return _clamp(default, 0.0, 1.0)


def _resolve_phoneme_boundary_model_path(path_hint: str = "") -> str:
    candidates: list[str] = []
    for raw in (
        path_hint,
        os.environ.get("UTOA_STAGE2_OTO_PHONEME_BOUNDARY_MODEL_PATH", ""),
        os.environ.get("UTOA_PHONEME_BOUNDARY_MODEL_PATH", ""),
    ):
        text = str(raw or "").strip()
        if text:
            candidates.append(text)
    for candidate in candidates:
        expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
        if os.path.isfile(expanded):
            return expanded
    try:
        from core.phoneme_boundary.discovery import list_available_phoneme_boundary_models

        models = list_available_phoneme_boundary_models()
    except Exception:
        models = []
    for item in models:
        path = str(item.get("path", "") or "").strip() if isinstance(item, dict) else ""
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return ""


def _read_source_rows_for_shift_guard(source_path: str) -> tuple[dict[tuple[str, str, int], dict[str, float]], dict[str, list[float]]]:
    rows_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    pre_abs_by_wav: dict[str, list[float]] = defaultdict(list)
    text = read_text_with_fallback(str(source_path or ""))
    if not text:
        return rows_by_key, dict(pre_abs_by_wav)
    for line_idx, raw in enumerate(text.splitlines()):
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        alias = str(parsed.get("alias", "") or "")
        if not wav_name or not alias:
            continue
        row_params = _source_row_params(parsed)
        rows_by_key[(wav_name, alias, int(line_idx))] = row_params
        if _is_blank_or_silence_alias(alias):
            continue
        wav_key = os.path.basename(str(wav_name).replace("\\", "/")).strip().lower()
        pre_abs_by_wav[wav_key].append(float(row_params["offset"]) + float(row_params["preutterance"]))
    return rows_by_key, dict(pre_abs_by_wav)


def _source_row_params(parsed: dict[str, Any]) -> dict[str, float]:
    return {
        "offset": float(parsed.get("offset", 0.0) or 0.0),
        "consonant": float(parsed.get("cons", 0.0) or 0.0),
        "cutoff": float(parsed.get("cutoff", 0.0) or 0.0),
        "preutterance": float(parsed.get("pre", 0.0) or 0.0),
        "overlap": float(parsed.get("ovl", 0.0) or 0.0),
    }


def _apply_row_shift_guard_for_wav(
    *,
    row_entries: list[dict[str, object]],
    source_pre_abs: list[float],
    source_rows_by_key: dict[tuple[str, str, int], dict[str, float]],
    callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    stats = {
        "rows_checked": int(len(row_entries)),
        "rows_supported": 0,
        "one_step_rows": 0,
        "mis_mapping_rows": 0,
        "row_rollbacks": 0,
        "wav_rollbacks": 0,
        "source_reject_rows": 0,
    }
    if not row_entries:
        return row_entries, stats
    if not _env_bool("UTOA_BOUNDARY_ROW_SHIFT_ROLLBACK_ENABLE", False):
        return row_entries, stats
    if not source_pre_abs:
        return row_entries, stats

    match_max = max(1.0, _env_float("UTOA_BOUNDARY_ROW_SHIFT_MATCH_MAX_MS", 120.0))
    mismatch_min = max(match_max, _env_float("UTOA_BOUNDARY_ROW_SHIFT_MISMATCH_MIN_MS", 120.0))
    severe_abs = max(mismatch_min, _env_float("UTOA_BOUNDARY_ROW_SHIFT_SEVERE_ABS_MS", 260.0))
    # VV-specific tighter severe-abs threshold. On vowel-only wavs the CRNN
    # cannot find a transient and decoded VV offsets drift ±150~250 ms while
    # source-OTO holds the correct slot. The default severe_abs=260 misses
    # these (2026-05-17 KO TABI: only 53% of VV within ±50 ms, bimodal
    # ±200 ms tail). 140 ms catches the tail without dragging in tight rows.
    severe_abs_vv = max(mismatch_min, _env_float("UTOA_BOUNDARY_ROW_SHIFT_VV_SEVERE_ABS_MS", 140.0))
    rollback_only_transitions = _env_bool("UTOA_BOUNDARY_ROW_SHIFT_ONLY_TRANSITIONS", True)
    use_source_row = _env_bool("UTOA_BOUNDARY_ROW_SHIFT_USE_SOURCE_ROW", True)
    use_alias_fallback_on_invalid = _env_bool(
        "UTOA_BOUNDARY_ROW_SHIFT_ALIAS_FALLBACK_ON_INVALID_SOURCE",
        False,
    )
    wav_rate_max = _clamp(_env_float("UTOA_BOUNDARY_ROW_SHIFT_WAV_ROLLBACK_RATE", 0.34), 0.05, 1.0)

    non_sil_entries: list[dict[str, object]] = []
    for item in row_entries:
        row = item.get("row")
        if row is None:
            continue
        if _is_blank_or_silence_alias(str(row.spec.alias or "")):
            continue
        non_sil_entries.append(item)
    if not non_sil_entries:
        return row_entries, stats

    flagged_indices: set[int] = set()
    # Track transition vs non-transition flags separately. Wav-level
    # rollback only escalates from transition flags (vc/vv/v-cv) — including
    # 'other'-role flags in the rate would cascade into intact vc_onset
    # rows on KO 8-mora wavs (2026-05-17 v14 regression: vc_onset ±50%
    # 73.9% → 66.8% when 'other' counted toward wav-level rate).
    transition_supported = 0
    transition_flagged: set[int] = set()
    for idx, item in enumerate(non_sil_entries):
        row = item.get("row")
        params = item.get("params")
        if row is None or not isinstance(params, dict):
            continue
        role = normalize_role(str(row.spec.role or "other"))
        # 'other' role is where bare KO CV aliases (`geu`/`ga`/`ge`) end up
        # because phones_from_text doesn't decompose single-token KO syllables.
        # These show −50 ms mean / 101 ms |Δ| in the 2026-05-17 KO TABI
        # diagnostic; row-shift rollback to source-OTO catches the worst
        # spread cases (D ≥ severe_abs=260) and source matches manual on
        # the head CV rows where alias-type misclassification is most common.
        if rollback_only_transitions and role not in {"vc", "vv", "v-cv", "other"}:
            continue
        if idx >= len(source_pre_abs):
            continue
        stats["rows_supported"] += 1
        pred_pre = _row_pre_abs(params)
        own_delta = abs(float(pred_pre) - float(source_pre_abs[idx]))
        nearest_idx, nearest_delta = _nearest_index(source_pre_abs, float(pred_pre))
        idx_gap = abs(int(nearest_idx) - int(idx))
        nearest_ok = float(nearest_delta) <= float(match_max)
        own_bad = float(own_delta) >= float(mismatch_min)
        role_severe_abs = float(severe_abs_vv) if role == "vv" else float(severe_abs)
        one_step = bool(idx_gap == 1 and own_bad and nearest_ok)
        mis_map = bool((idx_gap >= 1 and own_bad and nearest_ok) or float(own_delta) >= role_severe_abs)
        is_transition = role in {"vc", "vv", "v-cv"}
        if is_transition:
            transition_supported += 1
        if one_step:
            stats["one_step_rows"] += 1
            flagged_indices.add(idx)
            if is_transition:
                transition_flagged.add(idx)
        if mis_map:
            stats["mis_mapping_rows"] += 1
            flagged_indices.add(idx)
            if is_transition:
                transition_flagged.add(idx)

    supported = max(1, int(stats["rows_supported"]))
    # Wav-level rollback rate ignores 'other'-only flags so a high count of
    # bare-CV mis-mappings doesn't drag intact vc_onset rows into rollback.
    transition_supported_eff = max(1, int(transition_supported))
    transition_rate = float(len(transition_flagged)) / float(transition_supported_eff)
    flagged_rate = float(len(flagged_indices)) / float(supported)
    wav_level_rollback = bool(len(transition_flagged) > 0 and transition_rate >= float(wav_rate_max))
    if wav_level_rollback:
        stats["wav_rollbacks"] = 1
        _log(
            callback,
            f"[OTO-Boundary-ShiftGuard] wav_rollback=1 flagged={len(flagged_indices)}/{supported} rate={flagged_rate:.3f}",
        )

    for idx, item in enumerate(non_sil_entries):
        needs_rollback = wav_level_rollback or idx in flagged_indices
        if not needs_rollback:
            continue
        row = item.get("row")
        if row is None:
            continue
        source_key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
        params = source_rows_by_key.get(source_key) if use_source_row else None
        replacement: dict[str, float] | None = None
        if isinstance(params, dict):
            if _is_source_row_params_usable(params=params, duration_ms=float(row.spec.duration_ms)):
                replacement = dict(params)
            else:
                stats["source_reject_rows"] += 1
                if use_alias_fallback_on_invalid:
                    replacement = _alias_only_fallback_params(spec=row.spec)
        elif use_alias_fallback_on_invalid:
            replacement = _alias_only_fallback_params(spec=row.spec)
        if replacement is None:
            # Keep decoded params when source row is missing/degenerate to avoid
            # collapsing every row to zero/full-span values.
            continue
        item["params"] = dict(replacement)
        stats["row_rollbacks"] += 1
    return row_entries, stats


def _row_pre_abs(params: dict[str, float]) -> float:
    return float(params.get("offset", 0.0) or 0.0) + max(0.0, float(params.get("preutterance", 0.0) or 0.0))


def _nearest_index(values: list[float], target: float) -> tuple[int, float]:
    best_idx = -1
    best_delta = None
    for idx, value in enumerate(values):
        delta = abs(float(target) - float(value))
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    return int(best_idx), float(best_delta if best_delta is not None else 0.0)


def _is_source_row_params_usable(*, params: dict[str, float], duration_ms: float) -> bool:
    duration = max(1.0, float(duration_ms))
    offset = float(params.get("offset", 0.0) or 0.0)
    consonant = float(params.get("consonant", 0.0) or 0.0)
    cutoff = float(params.get("cutoff", 0.0) or 0.0)
    pre = float(params.get("preutterance", 0.0) or 0.0)
    overlap = float(params.get("overlap", 0.0) or 0.0)
    if offset < -1e-6 or offset > duration + 4.0:
        return False
    if consonant < -1e-6 or pre < -1e-6 or overlap < -1e-6:
        return False
    near_zero_shape = bool(consonant <= 1.0 and pre <= 1.0 and overlap <= 1.0)
    full_span_like = bool(
        (cutoff < 0.0 and abs(cutoff) >= (0.90 * duration))
        or (cutoff > 0.0 and cutoff >= (0.90 * duration))
        or (abs(cutoff) <= 1e-6)
    )
    if near_zero_shape and full_span_like:
        return False
    return True


def _is_blank_or_silence_alias(alias: str) -> bool:
    text = str(alias or "").strip().lower()
    if not text:
        return True
    if text.startswith("-"):
        text = text[1:].strip()
    if not text:
        return True
    silence_tokens = {"r", "br", "pau", "sil", "rest"}
    parts = [token for token in text.split() if token]
    return bool(parts) and all(token in silence_tokens for token in parts)


def _apply_suffix(alias: str, suffix: str) -> str:
    base = str(alias or "").strip()
    add = str(suffix or "").strip()
    if not base or not add:
        return base
    if base.endswith(add):
        return base
    return f"{base}{add}"


def _log(callback: Callable[[str], None] | None, text: str) -> None:
    if callback is None:
        return
    try:
        callback(str(text))
    except Exception:
        pass


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _mean_quality(values: list[float] | None) -> float:
    data = [float(v) for v in list(values or []) if v is not None]
    if not data:
        return 0.50
    return max(0.0, min(1.0, sum(data) / float(len(data))))


def _quality_at_time(score_map, time_ms: float, *, default: float) -> float:
    qs = list(getattr(score_map, "quality_scores", []) or [])
    ts = list(getattr(score_map, "times_ms", []) or [])
    if not qs or not ts:
        return max(0.0, min(1.0, float(default)))
    idx = min(range(len(ts)), key=lambda i: abs(float(ts[i]) - float(time_ms)))
    if idx < 0 or idx >= len(qs):
        return max(0.0, min(1.0, float(default)))
    return max(0.0, min(1.0, float(qs[idx])))


def _estimate_audio_reliability(audio) -> float:
    onset_strength = [float(p.strength) for p in list(getattr(audio, "onset_peaks", []) or [])]
    vowel_quality = [
        (float(seg.vowel_confidence) * 0.6) + (float(seg.stability_score) * 0.4)
        for seg in list(getattr(audio, "stable_vowel_segments", []) or [])
    ]
    merged = onset_strength + vowel_quality
    if not merged:
        return 0.50
    return max(0.0, min(1.0, sum(merged) / float(len(merged))))


def _is_weak_audio_reliability(audio_reliability: float) -> bool:
    threshold = _clamp(_env_float("UTOA_BOUNDARY_WEAK_AUDIO_RELIABILITY_MAX", 0.42), 0.05, 0.95)
    return float(audio_reliability) <= float(threshold)


def _alias_only_fallback_params(*, spec) -> dict[str, float]:
    if spec is None:
        return {
            "offset": 0.0,
            "consonant": 90.0,
            "cutoff": -140.0,
            "preutterance": 70.0,
            "overlap": 30.0,
        }
    duration = max(40.0, float(spec.duration_ms or 0.0))
    slot_count = max(1, int(spec.slot_count or 1))
    slot_index = max(0, min(slot_count - 1, int(spec.slot_index or 0)))
    span = max(1, slot_count)
    active_start = max(0.0, min(duration - 4.0, 0.02 * duration))
    active_end = max(active_start + 8.0, min(duration, 0.95 * duration))
    stride = (active_end - active_start) / float(span)
    left_anchor = active_start + (stride * float(slot_index))
    right_anchor = min(duration, left_anchor + stride)
    center = 0.5 * (left_anchor + right_anchor)
    anchors = build_absolute_anchors_for_role(
        role=str(getattr(spec, "role", "other") or "other"),
        center_ms=center,
        duration_ms=duration,
        left_anchor_ms=left_anchor,
        right_anchor_ms=right_anchor,
        active_start_ms=active_start,
        active_end_ms=active_end,
        confidence=0.0,
        reason="alias_only_fallback",
    )
    return absolute_anchors_to_oto_params(anchors, duration_ms=duration)


__all__ = ["generate_oto_with_boundary_decoder"]
