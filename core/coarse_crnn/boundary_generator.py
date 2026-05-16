from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable

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
    _log(callback, f"[OTO-Boundary] source={source}")
    _log(callback, f"[OTO-Boundary] model={model_file}")
    _log(callback, f"[OTO-Boundary] device={torch_device} wavs={len(rows_by_wav)} rows={total}")
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
            lwc_cand_count = 0
            if lwc_enabled_by_env(default=True):
                try:
                    lwc_cands = compute_long_window_candidates(
                        wav_path=wav_path,
                        scales_ms=lwc_scales_from_env(),
                        gap_ms=lwc_float_from_env("UTOA_BOUNDARY_LWC_GAP_MS", LWC_DEFAULT_GAP_MS),
                        threshold=lwc_float_from_env("UTOA_BOUNDARY_LWC_THRESHOLD", LWC_DEFAULT_THRESHOLD),
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
            selected_times = [round(float(item.selected_time_ms), 1) for item in decoded.rows]
            unique_selected = len(set(selected_times))
            reused_selected = max(0, len(selected_times) - unique_selected)
            _log(
                callback,
                f"[OTO-Boundary] wav={wav_idx}/{len(rows_by_wav)} peaks={len(model_cands)} "
                f"lwc={lwc_cand_count} merged={len(merged)} "
                f"fallback={decoded.fallback_count}/{len(decoded.rows)} "
                f"selected_unique={unique_selected}/{len(selected_times)} reused={reused_selected} "
                f"quality={model_quality:.3f} audio_rel={audio_reliability:.3f}",
            )
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
                key = (row.spec.wav_name, row.spec.alias, int(row.spec.line_index))
                decoded_params[key] = params
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
