from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.oto_inference import predict_oto_with_model
from core.coarse_crnn.oto_model import load_oto_checkpoint
from core.coarse_crnn.oto_targets import OTO_ANCHOR_NAMES, OtoAnchors, anchors_to_oto_params, extract_alias_features
from core.coarse_crnn.training import resolve_torch_device


@dataclass
class OtoEvalConfig:
    model_path: str
    device: str = "auto"
    max_items: int = 0
    seed: int = 1337
    language: str = ""
    format_type: str = ""
    gate_enabled: bool = False
    gate_min_evaluated_items: int = 200
    gate_min_preutterance_acc_50ms: float = 0.72
    gate_max_param_mae_ms: tuple[str, ...] = ("offset=90", "consonant=90", "cutoff_abs=140", "preutterance=55", "overlap=70")
    gate_alias_context_min_files: int = 25
    gate_alias_context_min_preutterance_acc_50ms: float = 0.55
    gate_max_hard_failure_rate: float = 0.08
    gate_min_worst_voicebank_preutterance_acc_50ms: float = 0.45
    gate_max_worst_voicebank_hard_failure_rate: float = 0.22
    gate_max_fp_breath_include_rate: float = 0.20
    gate_max_fn_voiced_miss_rate: float = 0.55
    gate_max_s_family_leak_rate: float = 0.25
    gate_max_onset_lag_p90_ms: float = 380.0
    gate_max_syllable_shift_rate: float = 0.45


def evaluate_oto_manifest(rows: list[dict[str, Any]], config: OtoEvalConfig) -> dict[str, Any]:
    torch = __import__("torch")
    device = resolve_torch_device(torch, config.device)
    model, model_config, meta = load_oto_checkpoint(config.model_path, map_location=str(device))
    model = model.to(device).eval()
    selected = _select_rows(rows, config)
    files: list[dict[str, Any]] = []
    failures: list[str] = []
    anchor_errors: dict[str, list[float]] = {name: [] for name in OTO_ANCHOR_NAMES}
    param_errors: dict[str, list[float]] = {
        "offset": [],
        "consonant": [],
        "cutoff_abs": [],
        "preutterance": [],
        "overlap": [],
    }
    pre_hits_20 = 0
    pre_hits_50 = 0
    pre_count = 0
    low_conf_count = 0
    predicted_error_values: list[float] = []
    hard_failure_count = 0
    activity_profile_cache: dict[str, dict[str, float]] = {}
    voicebank_activity_profile_cache: dict[str, dict[str, float]] = {}
    voicebank_row_map: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        voicebank_row_map.setdefault(str(row.get("voicebank_id", "") or "unknown_voicebank"), []).append(row)
    fp_breath_count = 0
    fn_voiced_count = 0
    s_family_leak_count = 0
    onset_lag_values: list[float] = []
    onset_shift_count = 0

    for row in selected:
        try:
            result = _evaluate_one(
                row,
                config,
                model=model,
                model_config=model_config,
                device=str(device),
                activity_profile_cache=activity_profile_cache,
                voicebank_activity_profile_cache=voicebank_activity_profile_cache,
                voicebank_row_map=voicebank_row_map,
            )
        except Exception as exc:
            failures.append(f"{row.get('audio', '')}: {exc}")
            continue
        files.append(result)
        for name, value in result["anchor_abs_errors_ms"].items():
            anchor_errors[name].append(float(value))
        for name, value in result["param_abs_errors_ms"].items():
            param_errors[name].append(float(value))
        pre_error = float(result["param_abs_errors_ms"]["preutterance"])
        pre_hits_20 += 1 if pre_error <= 20.0 else 0
        pre_hits_50 += 1 if pre_error <= 50.0 else 0
        pre_count += 1
        low_conf_count += 1 if bool(result.get("low_confidence", False)) else 0
        hard_failure_count += 1 if bool(result.get("hard_failure", False)) else 0
        pred_err = result.get("predicted_error_ms")
        if pred_err is not None:
            predicted_error_values.append(float(pred_err))
        fp_breath_count += 1 if bool(result.get("fp_breath_include", False)) else 0
        fn_voiced_count += 1 if bool(result.get("fn_voiced_miss", False)) else 0
        s_family_leak_count += 1 if bool(result.get("s_family_leak", False)) else 0
        lag = result.get("onset_lag_ms")
        if lag is not None:
            onset_lag_values.append(float(lag))
        onset_shift_count += 1 if bool(result.get("syllable_shift_flag", False)) else 0

    summary = {
        "model_path": os.path.abspath(config.model_path),
        "model_meta": meta,
        "device": str(device),
        "requested_items": len(selected),
        "evaluated_items": len(files),
        "failed_items": len(failures),
        "anchor_mae_ms": {name: _mean(values) for name, values in anchor_errors.items()},
        "param_mae_ms": {name: _mean(values) for name, values in param_errors.items()},
        "preutterance_acc_20ms": float(pre_hits_20) / float(pre_count) if pre_count else None,
        "preutterance_acc_50ms": float(pre_hits_50) / float(pre_count) if pre_count else None,
        "low_confidence_rate": (float(low_conf_count) / float(pre_count)) if pre_count else None,
        "hard_failure_rate": (float(hard_failure_count) / float(pre_count)) if pre_count else None,
        "predicted_error_ms_mean": _mean(predicted_error_values),
        "fp_breath_include_rate": (float(fp_breath_count) / float(pre_count)) if pre_count else None,
        "fn_voiced_miss_rate": (float(fn_voiced_count) / float(pre_count)) if pre_count else None,
        "s_family_leak_rate": (float(s_family_leak_count) / float(pre_count)) if pre_count else None,
        "onset_lag_mean_ms": _mean(onset_lag_values),
        "onset_lag_p90_ms": _percentile(onset_lag_values, 90.0),
        "syllable_shift_rate": (float(onset_shift_count) / float(pre_count)) if pre_count else None,
        "by_language": _aggregate_by(files, "language"),
        "by_format": _aggregate_by(files, "format_type"),
        "by_voicebank": _aggregate_by(files, "voicebank_id"),
        "by_alias_context": _aggregate_by_alias_context(files),
        "by_alias_role": _aggregate_by_alias_role(files),
        "worst_voicebank": _worst_voicebank_summary(files),
        "failures": failures[:20],
        "files": files,
    }
    summary["gate"] = _evaluate_gate(summary, config) if bool(config.gate_enabled) else {"enabled": False, "passed": None}
    return summary


def _evaluate_one(
    row: dict[str, Any],
    config: OtoEvalConfig,
    *,
    model,
    model_config,
    device: str,
    activity_profile_cache: dict[str, dict[str, float]],
    voicebank_activity_profile_cache: dict[str, dict[str, float]],
    voicebank_row_map: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    prediction = predict_oto_with_model(
        model=model,
        config=model_config,
        wav_path=str(row.get("audio", "") or ""),
        language=str(row.get("language", "") or ""),
        format_type=str(row.get("format_type", "") or ""),
        alias=str(row.get("alias", "") or ""),
        prev_alias=str(row.get("prev_alias", "") or ""),
        next_alias=str(row.get("next_alias", "") or ""),
        row_index_in_wav=int(row.get("row_index_in_wav", 0) or 0),
        file_row_count=int(row.get("file_row_count", 1) or 1),
        device=device,
    )
    target_anchors = OtoAnchors(
        offset=float(row.get("anchor_offset_ms", 0.0) or 0.0),
        overlap=float(row.get("anchor_overlap_ms", 0.0) or 0.0),
        preutterance=float(row.get("anchor_preutterance_ms", 0.0) or 0.0),
        consonant=float(row.get("anchor_consonant_ms", 0.0) or 0.0),
        cutoff=float(row.get("anchor_cutoff_ms", row.get("target_cutoff_abs_ms", 0.0)) or 0.0),
    )
    target_params = anchors_to_oto_params(target_anchors, duration_ms=float(row.get("duration_ms", prediction.duration_ms) or prediction.duration_ms))
    pred_anchor_dict = prediction.anchors.to_dict()
    target_anchor_dict = target_anchors.to_dict()
    anchor_errors = {name: abs(float(pred_anchor_dict[name]) - float(target_anchor_dict[name])) for name in OTO_ANCHOR_NAMES}
    param_errors = {
        "offset": abs(float(prediction.params["offset"]) - float(target_params["offset"])),
        "consonant": abs(float(prediction.params["consonant"]) - float(target_params["consonant"])),
        "cutoff_abs": abs(float(prediction.anchors.cutoff) - float(target_anchors.cutoff)),
        "preutterance": abs(float(prediction.params["preutterance"]) - float(target_params["preutterance"])),
        "overlap": abs(float(prediction.params["overlap"]) - float(target_params["overlap"])),
    }
    duration_ms = float(row.get("duration_ms", prediction.duration_ms) or prediction.duration_ms)
    voicebank_id = str(row.get("voicebank_id", "") or "unknown_voicebank")
    voicebank_profile = _resolve_voicebank_activity_baseline(
        voicebank_id=voicebank_id,
        rows=voicebank_row_map.get(voicebank_id, []),
        sample_rate=int(getattr(model_config, "sample_rate", 16000) or 16000),
        cache=voicebank_activity_profile_cache,
    )
    profile = _analyze_activity_profile(
        str(row.get("audio", "") or ""),
        sample_rate=int(getattr(model_config, "sample_rate", 16000) or 16000),
        cache=activity_profile_cache,
        voicebank_profile=voicebank_profile,
    )
    activity_diag = _compute_activity_diagnostics(
        predicted_anchors=pred_anchor_dict,
        activity_profile=profile,
        alias=str(row.get("alias", "") or ""),
        language=str(row.get("language", "") or ""),
        row_index_in_wav=int(row.get("row_index_in_wav", 0) or 0),
        file_row_count=int(row.get("file_row_count", 1) or 1),
        duration_ms=duration_ms,
    )
    hard_failure, hard_reason = _is_hard_failure(
        pred_anchor_dict,
        duration_ms=duration_ms,
        activity_profile=profile,
    )
    return {
        "audio": str(row.get("audio", "") or ""),
        "alias": str(row.get("alias", "") or ""),
        "language": str(row.get("language", "") or ""),
        "format_type": str(row.get("format_type", "") or ""),
        "voicebank_id": voicebank_id,
        "alias_type": str(row.get("alias_type", "") or "other"),
        "transition_type": str(row.get("transition_type", "") or "other"),
        **_resolve_alias_role_fields(row),
        "duration_ms": duration_ms,
        "confidence": float(prediction.confidence),
        "predicted_error_ms": (
            float(prediction.predicted_error_ms) if prediction.predicted_error_ms is not None else None
        ),
        "low_confidence": bool(prediction.low_confidence),
        "hard_failure": bool(hard_failure),
        "hard_failure_reason": str(hard_reason),
        "fp_breath_include": bool(activity_diag["fp_breath_include"]),
        "fn_voiced_miss": bool(activity_diag["fn_voiced_miss"]),
        "s_family_leak": bool(activity_diag["s_family_leak"]),
        "left_leak_ms": float(activity_diag["left_leak_ms"]),
        "right_leak_ms": float(activity_diag["right_leak_ms"]),
        "core_coverage": float(activity_diag["core_coverage"]),
        "onset_lag_ms": (
            float(activity_diag["onset_lag_ms"]) if activity_diag["onset_lag_ms"] is not None else None
        ),
        "syllable_shift_flag": bool(activity_diag["syllable_shift_flag"]),
        "confidence_components": dict(prediction.confidence_components or {}),
        "anchor_abs_errors_ms": anchor_errors,
        "param_abs_errors_ms": param_errors,
    }


def _select_rows(rows: list[dict[str, Any]], config: OtoEvalConfig) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (not config.language or str(row.get("language", "") or "") == config.language)
        and (not config.format_type or str(row.get("format_type", "") or "") == config.format_type)
    ]
    rng = random.Random(int(config.seed))
    rng.shuffle(candidates)
    if int(config.max_items) > 0:
        candidates = candidates[: int(config.max_items)]
    return candidates


def _aggregate_by(files: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        groups.setdefault(str(row.get(key, "") or "unknown"), []).append(row)
    out: dict[str, Any] = {}
    for group, rows in groups.items():
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
            "fp_breath_include_rate": (
                float(sum(1 for row in rows if bool(row.get("fp_breath_include", False)))) / float(len(rows))
                if rows
                else None
            ),
            "fn_voiced_miss_rate": (
                float(sum(1 for row in rows if bool(row.get("fn_voiced_miss", False)))) / float(len(rows))
                if rows
                else None
            ),
            "s_family_leak_rate": (
                float(sum(1 for row in rows if bool(row.get("s_family_leak", False)))) / float(len(rows))
                if rows
                else None
            ),
            "onset_lag_mean_ms": _mean(
                [float(row["onset_lag_ms"]) for row in rows if row.get("onset_lag_ms") is not None]
            ),
            "onset_lag_p90_ms": _percentile(
                [float(row["onset_lag_ms"]) for row in rows if row.get("onset_lag_ms") is not None],
                90.0,
            ),
            "hard_failure_rate": (
                float(sum(1 for row in rows if bool(row.get("hard_failure", False)))) / float(len(rows))
                if rows
                else None
            ),
        }
    return out


def _aggregate_by_alias_context(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        key = "|".join(
            [
                str(row.get("format_type", "") or "unknown"),
                str(row.get("alias_type", "") or "unknown"),
                str(row.get("transition_type", "") or "unknown"),
            ]
        )
        groups.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "format_type": group.split("|", 2)[0],
            "alias_type": group.split("|", 2)[1],
            "transition_type": group.split("|", 2)[2],
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "consonant_mae_ms": _mean([float(row["param_abs_errors_ms"]["consonant"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
            "overlap_mae_ms": _mean([float(row["param_abs_errors_ms"]["overlap"]) for row in rows]),
        }
    return out


def _aggregate_by_alias_role(files: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        role = str(row.get("alias_role", "") or "other")
        diphthong = "diph" if bool(row.get("is_diphthong", False)) else "mono"
        special = "special" if bool(row.get("is_special", False)) else "normal"
        key = "|".join([role, diphthong, special])
        groups.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        role, diph_flag, special_flag = group.split("|", 2)
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        out[group] = {
            "files": len(rows),
            "alias_role": role,
            "is_diphthong": diph_flag == "diph",
            "is_special": special_flag == "special",
            "preutterance_mae_ms": _mean(pre_errors),
            "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
            "offset_mae_ms": _mean([float(row["param_abs_errors_ms"]["offset"]) for row in rows]),
            "consonant_mae_ms": _mean([float(row["param_abs_errors_ms"]["consonant"]) for row in rows]),
            "cutoff_abs_mae_ms": _mean([float(row["param_abs_errors_ms"]["cutoff_abs"]) for row in rows]),
            "overlap_mae_ms": _mean([float(row["param_abs_errors_ms"]["overlap"]) for row in rows]),
        }
    return out


def _resolve_alias_role_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return alias_role, is_diphthong, is_special for an eval result row.

    Falls back to extract_alias_features() when the manifest was built before
    the alias_role feature was added (values are None rather than a string).
    """
    if row.get("alias_role") is not None:
        role_text = str(row["alias_role"] or "other")
        return {
            "alias_role": role_text,
            "is_diphthong": bool(float(row.get("is_diphthong", 0.0) or 0.0) >= 0.5),
            "is_special": bool(float(row.get("is_special", 0.0) or 0.0) >= 0.5) or role_text == "special",
        }
    feats = extract_alias_features(
        str(row.get("alias", "") or ""),
        language=str(row.get("language", "") or ""),
    )
    return {
        "alias_role": str(feats.get("alias_role", "") or "other"),
        "is_diphthong": bool(float(feats.get("is_diphthong", 0.0) or 0.0) >= 0.5),
        "is_special": bool(float(feats.get("is_special", 0.0) or 0.0) >= 0.5),
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, float(q)))


def _hit_rate(values: list[float], threshold: float) -> float | None:
    return float(sum(1 for value in values if float(value) <= float(threshold))) / float(len(values)) if values else None


def _worst_voicebank_summary(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        groups.setdefault(str(row.get("voicebank_id", "") or "unknown_voicebank"), []).append(row)
    if not groups:
        return None
    stats: list[dict[str, Any]] = []
    for voicebank, rows in groups.items():
        pre_errors = [float(row["param_abs_errors_ms"]["preutterance"]) for row in rows]
        stats.append(
            {
                "voicebank_id": str(voicebank),
                "files": len(rows),
                "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
                "hard_failure_rate": (
                    float(sum(1 for row in rows if bool(row.get("hard_failure", False)))) / float(len(rows))
                    if rows
                    else None
                ),
            }
        )
    worst_acc = min(
        stats,
        key=lambda item: float(item["preutterance_acc_50ms"]) if item.get("preutterance_acc_50ms") is not None else 1.0,
    )
    worst_fail = max(
        stats,
        key=lambda item: float(item["hard_failure_rate"]) if item.get("hard_failure_rate") is not None else 0.0,
    )
    return {
        "count": len(stats),
        "worst_preutterance_acc_50ms": dict(worst_acc),
        "worst_hard_failure_rate": dict(worst_fail),
    }


def _is_hard_failure(
    predicted_anchors: dict[str, float],
    *,
    duration_ms: float,
    activity_profile: dict[str, float],
) -> tuple[bool, str]:
    duration = max(1.0, float(duration_ms))
    offset = float(predicted_anchors.get("offset", 0.0) or 0.0)
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cons = float(predicted_anchors.get("consonant", 0.0) or 0.0)
    cutoff = float(predicted_anchors.get("cutoff", 0.0) or 0.0)
    mid = (pre + cons) * 0.5
    if cutoff <= offset + 5.0:
        return True, "degenerate_cutoff"
    if offset >= duration * 0.90:
        return True, "late_offset"
    if cutoff <= duration * 0.08:
        return True, "early_cutoff"
    start_ms = float(activity_profile.get("active_start_ms", 0.0) or 0.0)
    end_ms = float(activity_profile.get("active_end_ms", duration) or duration)
    if end_ms > start_ms + 30.0:
        if mid < (start_ms - 80.0) or mid > (end_ms + 80.0):
            return True, "outside_activity_window"
    return False, "ok"


def _analyze_activity_profile(
    wav_path: str,
    *,
    sample_rate: int,
    cache: dict[str, dict[str, float]],
    voicebank_profile: dict[str, float] | None = None,
) -> dict[str, float]:
    key = os.path.abspath(str(wav_path or "")).lower()
    if key in cache:
        return dict(cache[key])
    try:
        samples, sr, duration_sec = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
    except Exception:
        cache[key] = {}
        return {}
    duration_ms = max(1.0, float(duration_sec) * 1000.0)
    if samples is None or int(getattr(samples, "shape", [0])[0]) <= 0 or sr <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    abs_samples = np.abs(np.asarray(samples, dtype=np.float32))
    peak = float(np.max(abs_samples)) if abs_samples.size else 0.0
    if peak <= 1e-6:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    vb = dict(voicebank_profile or {})
    allow_thr = float(vb.get("allow_rms_threshold", 0.0) or 0.0)
    reject_thr = float(vb.get("reject_rms_threshold", 0.0) or 0.0)
    threshold = max(peak * 0.06, float(np.percentile(abs_samples, 65.0)) * 2.0, allow_thr, 1e-4)
    active_idx = np.flatnonzero(abs_samples >= threshold)
    if active_idx.size <= 0:
        threshold = max(peak * 0.03, reject_thr, 1e-5)
        active_idx = np.flatnonzero(abs_samples >= threshold)
    if active_idx.size <= 0:
        profile = {"active_start_ms": 0.0, "active_end_ms": duration_ms, "duration_ms": duration_ms}
        cache[key] = profile
        return dict(profile)
    start_ms = (float(active_idx[0]) / float(sr)) * 1000.0
    end_ms = (float(active_idx[-1]) / float(sr)) * 1000.0
    core_start_ms = max(0.0, start_ms - 10.0)
    core_end_ms = min(duration_ms, end_ms + 10.0)
    profile = {
        "active_start_ms": max(0.0, start_ms - 25.0),
        "active_end_ms": min(duration_ms, end_ms + 25.0),
        "active_core_start_ms": core_start_ms,
        "active_core_end_ms": core_end_ms,
        "onset_ms": core_start_ms,
        "duration_ms": duration_ms,
    }
    cache[key] = profile
    return dict(profile)


def _resolve_voicebank_activity_baseline(
    *,
    voicebank_id: str,
    rows: list[dict[str, Any]],
    sample_rate: int,
    cache: dict[str, dict[str, float]],
) -> dict[str, float]:
    key = f"__voicebank_profile__::{str(voicebank_id or 'unknown_voicebank')}"
    cached = cache.get(key)
    if isinstance(cached, dict) and cached:
        return dict(cached)
    max_wavs = 64
    wav_paths: list[str] = []
    for row in rows:
        path = str(row.get("audio", "") or "").strip()
        if not path or path in wav_paths:
            continue
        wav_paths.append(path)
        if len(wav_paths) >= max_wavs:
            break
    if not wav_paths:
        cache[key] = {}
        return {}
    rms_values: list[float] = []
    for wav_path in wav_paths:
        try:
            samples, sr, _dur = load_wav_mono(str(wav_path), target_sr=int(sample_rate))
        except Exception:
            continue
        if samples is None or sr <= 0:
            continue
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size <= 0:
            continue
        rms_values.extend(np.abs(arr).tolist())
    if not rms_values:
        cache[key] = {}
        return {}
    rms = np.asarray(rms_values, dtype=np.float32)
    noise_floor = float(np.percentile(rms, 30.0))
    speech_ref = float(np.percentile(rms, 92.0))
    allow_thr = max(noise_floor * 2.5, speech_ref * 0.24, 1e-6)
    reject_thr = max(noise_floor * 3.4, speech_ref * 0.30, allow_thr + 1e-7)
    out = {
        "noise_floor_rms": noise_floor,
        "speech_ref_rms": speech_ref,
        "allow_rms_threshold": allow_thr,
        "reject_rms_threshold": reject_thr,
        "sampled_wavs": int(len(wav_paths)),
    }
    cache[key] = out
    return dict(out)


def _compute_activity_diagnostics(
    *,
    predicted_anchors: dict[str, float],
    activity_profile: dict[str, float],
    alias: str,
    language: str,
    row_index_in_wav: int,
    file_row_count: int,
    duration_ms: float,
) -> dict[str, Any]:
    start = float(activity_profile.get("active_start_ms", 0.0) or 0.0)
    end = float(activity_profile.get("active_end_ms", start) or start)
    core_start = float(activity_profile.get("active_core_start_ms", start) or start)
    core_end = float(activity_profile.get("active_core_end_ms", end) or end)
    offset = float(predicted_anchors.get("offset", 0.0) or 0.0)
    pre = float(predicted_anchors.get("preutterance", 0.0) or 0.0)
    cutoff = float(predicted_anchors.get("cutoff", 0.0) or 0.0)

    idx = max(0, int(row_index_in_wav))
    row_count = max(1, int(file_row_count))
    full_start = min(start, core_start)
    full_end = max(end, core_end)
    full_span = max(1.0, full_end - full_start)
    center_ratio = (float(idx) + 0.5) / float(row_count)
    center_ratio = max(0.0, min(1.0, center_ratio))
    local_center = full_start + (full_span * center_ratio)
    local_span = max(80.0, min(900.0, (full_span / float(row_count)) * 1.9))
    local_start = max(full_start, local_center - (local_span * 0.5))
    local_end = min(full_end, local_center + (local_span * 0.5))
    if local_end <= local_start + 10.0:
        local_start = full_start
        local_end = full_end
    s_family = _is_s_family_alias(alias=alias, language=language)
    # For FP detection, use a wider adaptive allowance than the local row window.
    # Local window is still used for FN/shift so we don't miss row-level drift.
    allow_left = min(local_start - (local_span * 0.30), start - 24.0)
    allow_right = max(local_end + (local_span * 0.38), end + 32.0)
    allow_left = max(0.0, allow_left)
    allow_right = min(float(duration_ms), allow_right)
    if allow_right <= allow_left + 30.0:
        allow_left = max(0.0, start - 18.0)
        allow_right = min(float(duration_ms), end + 24.0)
    left_leak = max(0.0, allow_left - offset)
    right_leak = max(0.0, cutoff - allow_right)
    fp_left_thr = 22.0 if s_family else 36.0
    fp_right_thr = 30.0 if s_family else 52.0
    fp = bool(left_leak > fp_left_thr or right_leak > fp_right_thr)
    inter = max(0.0, min(cutoff, local_end) - max(offset, local_start))
    core_len = max(1.0, local_end - local_start)
    core_cov = float(inter / core_len)
    min_cov = 0.30
    mid = (offset + cutoff) * 0.5
    fn = bool(core_cov < min_cov and (mid < (local_start - 35.0) or mid > (local_end + 35.0)))

    onset_ref = float(local_center)
    onset_lag = float(pre - onset_ref)
    shift_tol = max(80.0, min(450.0, local_span * 0.60))
    shift_flag = bool(abs(onset_lag) > shift_tol)
    return {
        "left_leak_ms": float(left_leak),
        "right_leak_ms": float(right_leak),
        "core_coverage": float(core_cov),
        "fp_breath_include": bool(fp),
        "fn_voiced_miss": bool(fn),
        "onset_lag_ms": float(onset_lag),
        "syllable_shift_flag": bool(shift_flag),
        "s_family_leak": bool(fp and s_family),
        "local_window_start_ms": float(local_start),
        "local_window_end_ms": float(local_end),
        "local_window_center_ms": float(local_center),
        "local_window_span_ms": float(local_end - local_start),
        "fp_allow_start_ms": float(allow_left),
        "fp_allow_end_ms": float(allow_right),
    }


def _is_s_family_alias(*, alias: str, language: str) -> bool:
    text = str(alias or "").strip().lower()
    if not text:
        return False
    tokens = text.replace("-", " ").split()
    lang = str(language or "").strip().lower()
    if lang.startswith("ja") or lang.startswith("jp") or lang == "japanese":
        jp_set = {"s", "sh", "su", "shi", "sya", "syu", "syo", "sa", "se", "so"}
        return any(token in jp_set for token in tokens)
    if lang.startswith("ko") or lang.startswith("kr") or lang == "korean":
        ko_set = {"s", "ss", "sh", "ㅅ", "ㅆ"}
        return any(token in ko_set for token in tokens)
    return any(token in {"s", "sh", "ss"} for token in tokens)


def _evaluate_gate(summary: dict[str, Any], config: OtoEvalConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    evaluated_items = int(summary.get("evaluated_items", 0) or 0)
    checks.append(
        {
            "name": "evaluated_items",
            "passed": evaluated_items >= int(config.gate_min_evaluated_items),
            "actual": evaluated_items,
            "threshold": int(config.gate_min_evaluated_items),
            "operator": ">=",
        }
    )

    pre_acc_50 = _float_or_none(summary.get("preutterance_acc_50ms"))
    checks.append(
        {
            "name": "preutterance_acc_50ms",
            "passed": (pre_acc_50 is not None and pre_acc_50 >= float(config.gate_min_preutterance_acc_50ms)),
            "actual": pre_acc_50,
            "threshold": float(config.gate_min_preutterance_acc_50ms),
            "operator": ">=",
        }
    )

    hard_failure_rate = _float_or_none(summary.get("hard_failure_rate"))
    checks.append(
        {
            "name": "hard_failure_rate",
            "passed": (
                hard_failure_rate is not None
                and hard_failure_rate <= float(config.gate_max_hard_failure_rate)
            ),
            "actual": hard_failure_rate,
            "threshold": float(config.gate_max_hard_failure_rate),
            "operator": "<=",
        }
    )
    fp_breath_include_rate = _float_or_none(summary.get("fp_breath_include_rate"))
    checks.append(
        {
            "name": "fp_breath_include_rate",
            "passed": (
                fp_breath_include_rate is not None
                and fp_breath_include_rate <= float(config.gate_max_fp_breath_include_rate)
            ),
            "actual": fp_breath_include_rate,
            "threshold": float(config.gate_max_fp_breath_include_rate),
            "operator": "<=",
        }
    )
    fn_voiced_miss_rate = _float_or_none(summary.get("fn_voiced_miss_rate"))
    checks.append(
        {
            "name": "fn_voiced_miss_rate",
            "passed": (
                fn_voiced_miss_rate is not None
                and fn_voiced_miss_rate <= float(config.gate_max_fn_voiced_miss_rate)
            ),
            "actual": fn_voiced_miss_rate,
            "threshold": float(config.gate_max_fn_voiced_miss_rate),
            "operator": "<=",
        }
    )
    s_family_leak_rate = _float_or_none(summary.get("s_family_leak_rate"))
    checks.append(
        {
            "name": "s_family_leak_rate",
            "passed": (
                s_family_leak_rate is not None
                and s_family_leak_rate <= float(config.gate_max_s_family_leak_rate)
            ),
            "actual": s_family_leak_rate,
            "threshold": float(config.gate_max_s_family_leak_rate),
            "operator": "<=",
        }
    )
    onset_lag_p90_ms = _float_or_none(summary.get("onset_lag_p90_ms"))
    checks.append(
        {
            "name": "onset_lag_p90_ms",
            "passed": (
                onset_lag_p90_ms is not None
                and onset_lag_p90_ms <= float(config.gate_max_onset_lag_p90_ms)
            ),
            "actual": onset_lag_p90_ms,
            "threshold": float(config.gate_max_onset_lag_p90_ms),
            "operator": "<=",
        }
    )
    syllable_shift_rate = _float_or_none(summary.get("syllable_shift_rate"))
    checks.append(
        {
            "name": "syllable_shift_rate",
            "passed": (
                syllable_shift_rate is not None
                and syllable_shift_rate <= float(config.gate_max_syllable_shift_rate)
            ),
            "actual": syllable_shift_rate,
            "threshold": float(config.gate_max_syllable_shift_rate),
            "operator": "<=",
        }
    )

    param_mae = dict(summary.get("param_mae_ms") or {})
    mae_thresholds = _parse_key_value_tuple(getattr(config, "gate_max_param_mae_ms", ()))
    for key, threshold in sorted(mae_thresholds.items()):
        value = _float_or_none(param_mae.get(key))
        checks.append(
            {
                "name": f"param_mae_ms.{key}",
                "passed": (value is not None and value <= float(threshold)),
                "actual": value,
                "threshold": float(threshold),
                "operator": "<=",
            }
        )

    context_groups = summary.get("by_alias_context") or {}
    min_files = int(config.gate_alias_context_min_files)
    min_ctx_acc = float(config.gate_alias_context_min_preutterance_acc_50ms)
    eligible_contexts = []
    for key, item in context_groups.items():
        files = int((item or {}).get("files", 0) or 0)
        if files < min_files:
            continue
        acc50 = _float_or_none((item or {}).get("preutterance_acc_50ms"))
        if acc50 is None:
            continue
        eligible_contexts.append((str(key), files, acc50))
    if eligible_contexts:
        worst = min(eligible_contexts, key=lambda row: float(row[2]))
        ctx_pass = bool(worst[2] >= min_ctx_acc)
        checks.append(
            {
                "name": "alias_context_preutterance_acc_50ms",
                "passed": ctx_pass,
                "actual": float(worst[2]),
                "threshold": min_ctx_acc,
                "operator": ">=",
                "context": str(worst[0]),
                "context_files": int(worst[1]),
                "eligible_contexts": len(eligible_contexts),
            }
        )
    else:
        checks.append(
            {
                "name": "alias_context_preutterance_acc_50ms",
                "passed": True,
                "actual": None,
                "threshold": min_ctx_acc,
                "operator": ">=",
                "context": "skipped:no_context_meets_min_files",
                "context_files": 0,
                "eligible_contexts": 0,
            }
        )

    worst_voicebank = summary.get("worst_voicebank") or {}
    worst_acc = _float_or_none(((worst_voicebank.get("worst_preutterance_acc_50ms") or {}).get("preutterance_acc_50ms")))
    checks.append(
        {
            "name": "worst_voicebank_preutterance_acc_50ms",
            "passed": (
                worst_acc is not None
                and worst_acc >= float(config.gate_min_worst_voicebank_preutterance_acc_50ms)
            ),
            "actual": worst_acc,
            "threshold": float(config.gate_min_worst_voicebank_preutterance_acc_50ms),
            "operator": ">=",
        }
    )
    worst_fail = _float_or_none(((worst_voicebank.get("worst_hard_failure_rate") or {}).get("hard_failure_rate")))
    checks.append(
        {
            "name": "worst_voicebank_hard_failure_rate",
            "passed": (
                worst_fail is not None
                and worst_fail <= float(config.gate_max_worst_voicebank_hard_failure_rate)
            ),
            "actual": worst_fail,
            "threshold": float(config.gate_max_worst_voicebank_hard_failure_rate),
            "operator": "<=",
        }
    )

    passed = all(bool(item.get("passed", False)) for item in checks)
    failed_checks = [str(item.get("name")) for item in checks if not bool(item.get("passed", False))]
    return {
        "enabled": True,
        "passed": passed,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _parse_key_value_tuple(items: object) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or ():
        text = str(item or "").strip()
        if not text or "=" not in text:
            continue
        key, raw = text.split("=", 1)
        key_text = key.strip()
        if not key_text:
            continue
        try:
            out[key_text] = float(raw)
        except Exception:
            continue
    return out


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def write_oto_eval_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["OtoEvalConfig", "evaluate_oto_manifest", "write_oto_eval_json"]
