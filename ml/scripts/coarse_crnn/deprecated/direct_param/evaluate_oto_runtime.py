from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np

from core.coarse_crnn.deprecated.direct_param.oto_evaluate import _analyze_activity_profile, _is_hard_failure, _resolve_alias_role_fields
from core.coarse_crnn.deprecated.direct_param.oto_boundary_decoding import correct_boundary_params_from_candidates
from core.coarse_crnn.deprecated.direct_param.oto_inference import predict_oto_with_model
from core.coarse_crnn.deprecated.direct_param.oto_model import load_oto_checkpoint
from core.coarse_crnn.deprecated.direct_param.oto_predictor_generator import (
    _apply_audio_candidate_snap,
    _apply_vc_candidate_matcher,
    _apply_conservative_right_boundary_guard,
    _apply_low_confidence_fallback,
    _apply_activity_window_fallback,
    _get_audio_candidates,
    _env_bool,
    _env_float,
    _clamp,
)
from core.coarse_crnn.deprecated.direct_param.oto_sequence_alignment import decode_sequence_alignment_for_states, parse_sequence_role_allowlist
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES, OtoAnchors, anchors_to_oto_params, oto_params_to_anchors, read_jsonl
from core.coarse_crnn.torch_utils import resolve_torch_device


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CRNN OTO with the same lightweight runtime postprocess used by "
            "the generator, without leaking target/source OTO values into fallback."
        )
    )
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "oto_manifest.jsonl"))
    parser.add_argument("--model", default=os.path.join("ml_workspace", "models", "coarse_crnn", "oto_anchor_crnn.pt"))
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "oto_runtime_eval.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="")
    parser.add_argument("--format-type", default="")
    parser.add_argument(
        "--use-row-base-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use row OTO params as base_params for fallback (closer to real generator path).",
    )
    parser.add_argument(
        "--preserve-audio-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select whole WAV groups so candidate sequence snap is evaluated in row order.",
    )
    args = parser.parse_args()

    rows = read_jsonl(str(args.manifest))
    selected = _select_rows(
        rows,
        max_items=int(args.max_items),
        seed=int(args.seed),
        language=str(args.language or ""),
        format_type=str(args.format_type or ""),
        preserve_audio_groups=bool(args.preserve_audio_groups),
    )
    result = evaluate_runtime_manifest(
        selected,
        model_path=str(args.model),
        device=str(args.device),
        use_row_base_fallback=bool(args.use_row_base_fallback),
    )
    result["manifest"] = os.path.abspath(str(args.manifest))
    result["model_path"] = os.path.abspath(str(args.model))
    result["requested_items"] = len(selected)
    result["selection"] = {
        "max_items": int(args.max_items),
        "seed": int(args.seed),
        "language": str(args.language or ""),
        "format_type": str(args.format_type or ""),
        "preserve_audio_groups": bool(args.preserve_audio_groups),
        "use_row_base_fallback": bool(args.use_row_base_fallback),
    }
    os.makedirs(os.path.dirname(os.path.abspath(str(args.out))), exist_ok=True)
    with open(str(args.out), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    compact = {key: value for key, value in result.items() if key not in {"files", "failures", "model_meta"}}
    compact["out"] = os.path.abspath(str(args.out))
    compact["failures"] = result.get("failures", [])[:5]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if int(result.get("evaluated_items", 0) or 0) > 0 else 2


def evaluate_runtime_manifest(
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    device: str = "auto",
    use_row_base_fallback: bool = False,
) -> dict[str, Any]:
    torch = __import__("torch")
    torch_device = resolve_torch_device(torch, device)
    model, model_config, meta = load_oto_checkpoint(model_path, map_location=str(torch_device))
    model = model.to(torch_device).eval()

    activity_profile_cache: dict[str, dict[str, float]] = {}
    audio_candidate_cache: dict[str, Any] = {}
    audio_candidate_sequence_state: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in _sort_for_runtime(rows):
        try:
            item = _evaluate_one_runtime(
                row,
                model=model,
                model_config=model_config,
                device=str(torch_device),
                activity_profile_cache=activity_profile_cache,
                audio_candidate_cache=audio_candidate_cache,
                audio_candidate_sequence_state=audio_candidate_sequence_state,
                use_row_base_fallback=bool(use_row_base_fallback),
            )
        except Exception as exc:
            failures.append(f"{row.get('audio', '')}: {exc}")
            continue
        files.append(item)

    if _env_bool("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ENABLE", False):
        _apply_sequence_alignment_to_runtime_files(
            files,
            model_config=model_config,
            audio_candidate_cache=audio_candidate_cache,
        )

    counters = _runtime_counters(files)

    raw_files = [{**item, **item["raw"]} for item in files]
    runtime_files = [{**item, **item["runtime"]} for item in files]
    return {
        "model_meta": meta,
        "device": str(torch_device),
        "evaluated_items": len(files),
        "failed_items": len(failures),
        "raw": _summarize_variant(raw_files),
        "runtime": _summarize_variant(runtime_files),
        "deltas": _summary_delta(raw_files, runtime_files),
        "postprocess_counts": counters,
        "by_language_format_role": _aggregate_by_language_format_role(files),
        "failures": failures[:20],
        "files": files,
    }


def _evaluate_one_runtime(
    row: dict[str, Any],
    *,
    model,
    model_config,
    device: str,
    activity_profile_cache: dict[str, dict[str, float]],
    audio_candidate_cache: dict[str, Any],
    audio_candidate_sequence_state: dict[str, int],
    use_row_base_fallback: bool,
) -> dict[str, Any]:
    wav_path = str(row.get("audio", "") or "")
    language = str(row.get("language", "") or "")
    format_type = str(row.get("format_type", "") or "")
    alias = str(row.get("alias", "") or "")
    duration_ms = float(row.get("duration_ms", 0.0) or 0.0)
    prediction = predict_oto_with_model(
        model=model,
        config=model_config,
        wav_path=wav_path,
        language=language,
        format_type=format_type,
        alias=alias,
        prev_alias=str(row.get("prev_alias", "") or ""),
        next_alias=str(row.get("next_alias", "") or ""),
        row_index_in_wav=int(row.get("row_index_in_wav", 0) or 0),
        file_row_count=int(row.get("file_row_count", 1) or 1),
        device=device,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
        prev_is_special=bool(float(row.get("prev_is_special", 0.0) or 0.0) >= 0.5),
        next_is_special=bool(float(row.get("next_is_special", 0.0) or 0.0) >= 0.5),
    )
    duration_ms = max(1.0, duration_ms or float(prediction.duration_ms))
    target_anchors = OtoAnchors(
        offset=float(row.get("anchor_offset_ms", row.get("target_offset_ms", 0.0)) or 0.0),
        overlap=float(row.get("anchor_overlap_ms", 0.0) or 0.0),
        preutterance=float(row.get("anchor_preutterance_ms", 0.0) or 0.0),
        consonant=float(row.get("anchor_consonant_ms", 0.0) or 0.0),
        cutoff=float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0),
    )
    target_params = anchors_to_oto_params(target_anchors, duration_ms=duration_ms)
    base_params = dict(target_params) if bool(use_row_base_fallback) else None
    profile = _analyze_activity_profile(
        wav_path,
        sample_rate=int(getattr(model_config, "sample_rate", 16000) or 16000),
        cache=activity_profile_cache,
    )

    raw = _variant_result(prediction.anchors, prediction.params, target_anchors, target_params, duration_ms, profile)

    params, guard_changed = _apply_conservative_right_boundary_guard(
        prediction.params,
        language=language,
        alias=alias,
        duration_ms=duration_ms,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
    )
    params, low_conf_reason = _apply_low_confidence_fallback(
        predicted_params=params,
        predicted_confidence=float(prediction.confidence),
        predicted_error_ms=prediction.predicted_error_ms,
        predicted_low_confidence=bool(prediction.low_confidence),
        confidence_components=prediction.confidence_components,
        base_params=base_params,
        language=language,
        alias=alias,
        duration_ms=duration_ms,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
    )
    params, activity_reason = _apply_activity_window_fallback(
        predicted_anchors=prediction.anchors.to_dict(),
        predicted_params=params,
        base_params=base_params,
        wav_path=wav_path,
        sample_rate=int(getattr(model_config, "sample_rate", 16000) or 16000),
        cache=activity_profile_cache,
        row_index=int(row.get("row_index_in_wav", 0) or 0),
        row_count=int(row.get("file_row_count", 1) or 1),
        language=language,
        alias=alias,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
    )
    role_fields = _resolve_alias_role_fields(row)
    role_text = str(role_fields.get("alias_role", "") or "")
    boundary_reason = ""
    if _env_bool("UTOA_OTO_CRNN_RUNTIME_BOUNDARY_CANDIDATE_ENABLE", False):
        params, boundary_reason = correct_boundary_params_from_candidates(
            predicted_params=params,
            audio_candidates=_get_audio_candidates(wav_path, cache=audio_candidate_cache, config=model_config),
            role=role_text,
            duration_ms=duration_ms,
            max_delta_ms=260.0,
            blend=0.72,
        )
    params, vc_reason = _apply_vc_candidate_matcher(
        predicted_params=params,
        wav_path=wav_path,
        language=language,
        alias=alias,
        format_type=format_type,
        duration_ms=duration_ms,
        cache=audio_candidate_cache,
        config=model_config,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
        predicted_confidence=float(prediction.confidence),
        predicted_error_ms=prediction.predicted_error_ms,
        predicted_low_confidence=bool(prediction.low_confidence),
        confidence_components=prediction.confidence_components,
    )
    if vc_reason:
        params, _ = _apply_conservative_right_boundary_guard(
            params,
            language=language,
            alias=alias,
            duration_ms=duration_ms,
            is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
        )
    params, snap_reason = _apply_audio_candidate_snap(
        predicted_params=params,
        wav_path=wav_path,
        language=language,
        format_type=format_type,
        alias=alias,
        duration_ms=duration_ms,
        cache=audio_candidate_cache,
        sequence_state=audio_candidate_sequence_state,
        config=model_config,
        is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
    )
    if snap_reason:
        params, _ = _apply_conservative_right_boundary_guard(
            params,
            language=language,
            alias=alias,
            duration_ms=duration_ms,
            is_special=bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5),
        )
    runtime_anchors = oto_params_to_anchors(
        offset=float(params.get("offset", 0.0) or 0.0),
        consonant=float(params.get("consonant", 0.0) or 0.0),
        cutoff=float(params.get("cutoff", 0.0) or 0.0),
        preutterance=float(params.get("preutterance", 0.0) or 0.0),
        overlap=float(params.get("overlap", 0.0) or 0.0),
        duration_ms=duration_ms,
    )
    runtime = _variant_result(runtime_anchors, params, target_anchors, target_params, duration_ms, profile)
    moved = any(
        abs(float(runtime["params"].get(key, 0.0)) - float(raw["params"].get(key, 0.0))) > 1e-4
        for key in ("offset", "consonant", "cutoff", "preutterance", "overlap")
    )
    return {
        "audio": wav_path,
        "alias": alias,
        "language": language,
        "format_type": format_type,
        "voicebank_id": str(row.get("voicebank_id", "") or "unknown_voicebank"),
        "alias_type": str(row.get("alias_type", "") or "other"),
        "transition_type": str(row.get("transition_type", "") or "other"),
        **_resolve_alias_role_fields(row),
        "row_index_in_wav": int(row.get("row_index_in_wav", 0) or 0),
        "file_row_count": int(row.get("file_row_count", 1) or 1),
        "duration_ms": duration_ms,
        "target_anchors": target_anchors.to_dict(),
        "target_params": dict(target_params),
        "activity_profile": dict(profile),
        "confidence": float(prediction.confidence),
        "predicted_error_ms": (
            float(prediction.predicted_error_ms) if prediction.predicted_error_ms is not None else None
        ),
        "low_confidence": bool(prediction.low_confidence),
        "confidence_components": dict(prediction.confidence_components or {}),
        "sequence_candidate_times_ms": list(prediction.sequence_candidate_times_ms or []),
        "sequence_candidate_scores": list(prediction.sequence_candidate_scores or []),
        "sequence_candidate_classes": list(prediction.sequence_candidate_classes or []),
        "right_guard_changed": bool(guard_changed),
        "low_confidence_fallback": bool(low_conf_reason),
        "low_confidence_reason": str(low_conf_reason),
        "activity_fallback": bool(activity_reason),
        "activity_reason": str(activity_reason),
        "boundary_slot_graph": bool(boundary_reason),
        "boundary_slot_graph_reason": str(boundary_reason),
        "vc_matcher": bool(vc_reason),
        "vc_matcher_reason": str(vc_reason),
        "candidate_snap": bool(snap_reason),
        "candidate_snap_reason": str(snap_reason),
        "use_row_base_fallback": bool(use_row_base_fallback),
        "runtime_moved": bool(moved),
        "raw": raw,
        "runtime": runtime,
    }


def _apply_sequence_alignment_to_runtime_files(
    files: list[dict[str, Any]],
    *,
    model_config,
    audio_candidate_cache: dict[str, Any],
) -> None:
    states: list[dict[str, Any]] = []
    by_key = {}
    for idx, item in enumerate(files):
        row = SimpleNamespace(
            wav_abs=str(item.get("audio", "") or ""),
            row_index=int(item.get("row_index_in_wav", 0) or 0),
            alias=str(item.get("alias", "") or ""),
            alias_role=str(item.get("alias_role", "") or ""),
        )
        state = {
            "row": row,
            "params": dict((item.get("runtime") or {}).get("params") or {}),
            "duration_ms": float(item.get("duration_ms", 0.0) or 0.0),
            "sequence_candidate_times_ms": list(item.get("sequence_candidate_times_ms") or []),
            "sequence_candidate_scores": list(item.get("sequence_candidate_scores") or []),
            "sequence_candidate_classes": list(item.get("sequence_candidate_classes") or []),
            "sequence_aligned": False,
            "sequence_alignment_hold": False,
            "sequence_alignment_reason": "",
        }
        states.append(state)
        by_key[id(state)] = idx
    moved, held = decode_sequence_alignment_for_states(
        row_states=states,
        language="",
        role_resolver=lambda row: str(getattr(row, "alias_role", "") or ""),
        candidate_getter=lambda wav_abs: _get_audio_candidates(str(wav_abs), cache=audio_candidate_cache, config=model_config),
        max_shift_ms=max(0.0, _env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MAX_SHIFT_MS", 220.0)),
        anchor_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ANCHOR_BLEND", 0.60), 0.0, 1.0),
        boundary_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_BOUNDARY_BLEND", 0.45), 0.0, 1.0),
        min_move_ms=max(0.0, _env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MIN_MOVE_MS", 8.0)),
        model_score_blend=_clamp(_env_float("UTOA_OTO_CRNN_SEQUENCE_ALIGN_MODEL_SCORE_BLEND", 0.35), 0.0, 1.0),
        allowed_roles=parse_sequence_role_allowlist(os.environ.get("UTOA_OTO_CRNN_SEQUENCE_ALIGN_ROLES", "vc,vv")),
    )
    for state in states:
        idx = by_key[id(state)]
        item = files[idx]
        item["sequence_aligned"] = bool(state.get("sequence_aligned", False))
        item["sequence_alignment_hold"] = bool(state.get("sequence_alignment_hold", False))
        item["sequence_alignment_reason"] = str(state.get("sequence_alignment_reason", ""))
        item["sequence_alignment_applied"] = bool(state.get("sequence_aligned", False) or state.get("sequence_alignment_hold", False))
        item["sequence_alignment_moved_total"] = int(moved)
        item["sequence_alignment_held_total"] = int(held)
        if not bool(state.get("sequence_aligned", False)):
            continue
        params = dict(state.get("params") or {})
        duration_ms = float(item.get("duration_ms", 0.0) or 0.0)
        anchors = oto_params_to_anchors(
            offset=float(params.get("offset", 0.0) or 0.0),
            consonant=float(params.get("consonant", 0.0) or 0.0),
            cutoff=float(params.get("cutoff", 0.0) or 0.0),
            preutterance=float(params.get("preutterance", 0.0) or 0.0),
            overlap=float(params.get("overlap", 0.0) or 0.0),
            duration_ms=duration_ms,
        )
        target_anchors = OtoAnchors(**dict(item.get("target_anchors") or {}))
        target_params = dict(item.get("target_params") or {})
        profile = dict(item.get("activity_profile") or {})
        item["runtime"] = _variant_result(anchors, params, target_anchors, target_params, duration_ms, profile)
        item["runtime_moved"] = True


def _runtime_counters(files: list[dict[str, Any]]) -> dict[str, int]:
    counters = {
        "right_guard_changed": 0,
        "low_confidence_fallback": 0,
        "activity_fallback": 0,
        "boundary_slot_graph": 0,
        "candidate_snap": 0,
        "vc_matcher": 0,
        "sequence_aligned": 0,
        "sequence_alignment_hold": 0,
        "runtime_moved": 0,
        "offset_improved": 0,
        "offset_worse": 0,
        "offset_same": 0,
        "preutterance_improved": 0,
        "preutterance_worse": 0,
        "preutterance_same": 0,
    }
    for item in files:
        for key in (
            "right_guard_changed",
            "low_confidence_fallback",
            "activity_fallback",
            "boundary_slot_graph",
            "candidate_snap",
            "vc_matcher",
            "sequence_aligned",
            "sequence_alignment_hold",
            "runtime_moved",
        ):
            counters[key] += 1 if bool(item.get(key, False)) else 0
        for field, prefix in (("offset", "offset"), ("preutterance", "preutterance")):
            raw_err = float(item["raw"]["param_abs_errors_ms"][field])
            runtime_err = float(item["runtime"]["param_abs_errors_ms"][field])
            if runtime_err < raw_err - 1e-6:
                counters[f"{prefix}_improved"] += 1
            elif runtime_err > raw_err + 1e-6:
                counters[f"{prefix}_worse"] += 1
            else:
                counters[f"{prefix}_same"] += 1
    return counters


def _variant_result(
    anchors: Any,
    params: dict[str, float],
    target_anchors: OtoAnchors,
    target_params: dict[str, float],
    duration_ms: float,
    activity_profile: dict[str, float],
) -> dict[str, Any]:
    if isinstance(anchors, OtoAnchors):
        anchor_dict = anchors.to_dict()
    else:
        vals = [float(v) for v in anchors]
        anchor_dict = dict(zip(OTO_ANCHOR_NAMES, vals))
    target_anchor_dict = target_anchors.to_dict()
    anchor_errors = {
        name: abs(float(anchor_dict[name]) - float(target_anchor_dict[name]))
        for name in OTO_ANCHOR_NAMES
    }
    param_errors = {
        "offset": abs(float(params["offset"]) - float(target_params["offset"])),
        "consonant": abs(float(params["consonant"]) - float(target_params["consonant"])),
        "cutoff_abs": abs(float(anchor_dict["cutoff"]) - float(target_anchor_dict["cutoff"])),
        "preutterance": abs(float(params["preutterance"]) - float(target_params["preutterance"])),
        "overlap": abs(float(params["overlap"]) - float(target_params["overlap"])),
    }
    hard_failure, hard_reason = _is_hard_failure(
        anchor_dict,
        duration_ms=float(duration_ms),
        activity_profile=activity_profile,
    )
    return {
        "anchors": {key: float(value) for key, value in anchor_dict.items()},
        "params": {key: float(value) for key, value in params.items()},
        "anchor_abs_errors_ms": anchor_errors,
        "param_abs_errors_ms": param_errors,
        "hard_failure": bool(hard_failure),
        "hard_failure_reason": str(hard_reason),
    }


def _summarize_variant(files: list[dict[str, Any]]) -> dict[str, Any]:
    if not files:
        return {}
    param_errors = {
        key: [float(row["param_abs_errors_ms"][key]) for row in files]
        for key in ("offset", "consonant", "cutoff_abs", "preutterance", "overlap")
    }
    anchor_errors = {
        key: [float(row["anchor_abs_errors_ms"][key]) for row in files]
        for key in OTO_ANCHOR_NAMES
    }
    pre_errors = param_errors["preutterance"]
    return {
        "anchor_mae_ms": {key: _mean(values) for key, values in anchor_errors.items()},
        "param_mae_ms": {key: _mean(values) for key, values in param_errors.items()},
        "preutterance_acc_20ms": _hit_rate(pre_errors, 20.0),
        "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
        "position_bad_rate_100ms": _position_bad_rate(files, 100.0),
        "position_bad_rate_250ms": _position_bad_rate(files, 250.0),
        "hard_failure_rate": float(sum(1 for row in files if bool(row.get("hard_failure")))) / float(len(files)),
    }


def _summary_delta(raw_files: list[dict[str, Any]], runtime_files: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _summarize_variant(raw_files)
    runtime = _summarize_variant(runtime_files)
    out: dict[str, Any] = {}
    for metric in ("param_mae_ms", "anchor_mae_ms"):
        out[metric] = {}
        for key, raw_value in (raw.get(metric) or {}).items():
            runtime_value = (runtime.get(metric) or {}).get(key)
            out[metric][key] = (
                float(runtime_value) - float(raw_value)
                if raw_value is not None and runtime_value is not None
                else None
            )
    for key in (
        "preutterance_acc_20ms",
        "preutterance_acc_50ms",
        "position_bad_rate_100ms",
        "position_bad_rate_250ms",
        "hard_failure_rate",
    ):
        raw_value = raw.get(key)
        runtime_value = runtime.get(key)
        out[key] = float(runtime_value) - float(raw_value) if raw_value is not None and runtime_value is not None else None
    return out


def _aggregate_by_language_format_role(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in files:
        key = "|".join(
            [
                str(row.get("language", "") or "unknown"),
                str(row.get("format_type", "") or "unknown"),
                str(row.get("alias_role", "") or "other"),
            ]
        )
        groups[key].append(row)
    out: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        raw_files = [{**row, **row["raw"]} for row in rows]
        runtime_files = [{**row, **row["runtime"]} for row in rows]
        out[key] = {
            "files": len(rows),
            "raw": _summarize_variant(raw_files),
            "runtime": _summarize_variant(runtime_files),
            "deltas": _summary_delta(raw_files, runtime_files),
            "moved": sum(1 for row in rows if bool(row.get("runtime_moved"))),
            "candidate_snap": sum(1 for row in rows if bool(row.get("candidate_snap"))),
            "boundary_slot_graph": sum(1 for row in rows if bool(row.get("boundary_slot_graph"))),
            "vc_matcher": sum(1 for row in rows if bool(row.get("vc_matcher"))),
            "activity_fallback": sum(1 for row in rows if bool(row.get("activity_fallback"))),
        }
    return out


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    max_items: int,
    seed: int,
    language: str,
    format_type: str,
    preserve_audio_groups: bool,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("audio", "") or "")
        and (not language or str(row.get("language", "") or "") == language)
        and (not format_type or str(row.get("format_type", "") or "") == format_type)
    ]
    rng = random.Random(int(seed))
    if not preserve_audio_groups:
        rng.shuffle(candidates)
        return candidates[: int(max_items)] if int(max_items) > 0 else candidates
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[os.path.abspath(str(row.get("audio", "") or "")).lower()].append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    selected: list[dict[str, Any]] = []
    limit = int(max_items)
    for group in groups:
        group_sorted = _sort_for_runtime(group)
        if limit > 0 and selected and len(selected) + len(group_sorted) > limit:
            continue
        selected.extend(group_sorted)
        if limit > 0 and len(selected) >= limit:
            break
    if limit > 0 and len(selected) < limit:
        selected = selected[:limit]
    return selected


def _sort_for_runtime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            os.path.abspath(str(row.get("audio", "") or "")).lower(),
            int(row.get("row_index_in_wav", 0) or 0),
            str(row.get("alias", "") or ""),
        ),
    )


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def _hit_rate(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32) <= float(threshold)))


def _is_position_bad(param_errors: dict[str, float], threshold_ms: float) -> bool:
    return any(
        float(param_errors.get(key, 0.0) or 0.0) > float(threshold_ms)
        for key in ("offset", "consonant", "cutoff_abs", "preutterance")
    )


def _position_bad_rate(files: list[dict[str, Any]], threshold_ms: float) -> float | None:
    if not files:
        return None
    bad = sum(1 for row in files if _is_position_bad(dict(row.get("param_abs_errors_ms") or {}), threshold_ms))
    return float(bad) / float(len(files))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "off", "no"}


if __name__ == "__main__":
    raise SystemExit(main())


