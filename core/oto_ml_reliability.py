"""
Centralized reliability + blank-risk helpers for OTO ML.

Used by runtime guards and training weighting so the criteria stay consistent.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from core.silence_profile_runtime import resolve_silence_reliability_profile


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raw = str(value or "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _resolve_profile(row_context: Dict[str, object], profile: Optional[Dict[str, object]]) -> Dict[str, object]:
    if isinstance(profile, dict):
        return profile
    return resolve_silence_reliability_profile(row_context)


def _profile_blank_cfg(profile: Dict[str, object]) -> Dict[str, object]:
    return dict((profile or {}).get("blank") or {})


def _profile_mel_cfg(profile: Dict[str, object]) -> Dict[str, object]:
    return dict((profile or {}).get("mel") or {})


def mel_patch_is_fallback(row_context: Dict[str, object]) -> bool:
    src = str(row_context.get("mel_patch_source", "") or "").strip().lower()
    if not src:
        onset = _to_float(row_context.get("mel_offset_candidate_ms"), 0.0)
        tail = _to_float(row_context.get("mel_cutoff_candidate_ms"), 0.0)
        if onset > 0.0 and tail > 0.0:
            return False
        return True
    return src not in {"mel_candidate"}


def compute_blank_risk_score(
    row_context: Dict[str, object],
    *,
    profile: Optional[Dict[str, object]] = None,
) -> float:
    resolved_profile = _resolve_profile(row_context, profile)
    blank_conf = max(
        _to_float(row_context.get("blank_span_confidence"), 0.0),
        _to_float(row_context.get("syllable_blank_confidence"), 0.0),
        _to_float(row_context.get("syllable_mel_silence_conf"), 0.0),
    )
    silence_ratio = max(
        _to_float(row_context.get("db_silence_ratio"), 0.0),
        _to_float(row_context.get("mel_window_silence_ratio"), 0.0),
    )
    voiced_hint = max(
        _to_float(row_context.get("syllable_mel_voiced_conf"), 0.0),
        _to_float(row_context.get("mel_voiced_formant_ratio"), 0.0),
    )
    blank_cfg = _profile_blank_cfg(resolved_profile)
    rms_norm = max(
        0.0,
        _to_float(row_context.get("rms_norm_wav"), 0.0),
        _to_float(row_context.get("rms_norm_vb"), 0.0),
    )
    f0_presence = _clamp01(
        max(
            _to_float(row_context.get("f0_valid_ratio"), 0.0),
            _to_float(row_context.get("f0_voicing_mean"), 0.0),
            _to_float(row_context.get("f0_voicing_near_pre"), 0.0),
            _to_float(row_context.get("f0_continuity"), 0.0),
        )
    )

    rms_floor = max(1e-4, _to_float(blank_cfg.get("rms_low_priority_floor"), 0.26))
    rms_low_factor = _clamp01((rms_floor - min(rms_norm, rms_floor)) / rms_floor)
    voiced_penalty_scale_min = _clamp01(_to_float(blank_cfg.get("rms_low_voiced_penalty_scale_min"), 0.25))
    voiced_penalty_scale = 1.0 - (1.0 - voiced_penalty_scale_min) * rms_low_factor

    score = (0.58 * blank_conf) + (0.32 * silence_ratio) - ((0.22 * voiced_penalty_scale) * voiced_hint)
    # Priority override: when RMS is low, blank risk should dominate even if F0 is present.
    f0_min_for_priority = _clamp01(_to_float(blank_cfg.get("rms_low_priority_f0_min"), 0.08))
    if rms_low_factor > 1e-6 and f0_presence >= f0_min_for_priority:
        rms_priority_boost = max(0.0, _to_float(blank_cfg.get("rms_low_priority_boost"), 0.34))
        score += rms_priority_boost * rms_low_factor * (0.65 + (0.35 * f0_presence))

    if mel_patch_is_fallback(row_context):
        score += _to_float(blank_cfg.get("fallback_boost"), 0.0)
    score = (_to_float(blank_cfg.get("score_scale"), 1.0) * score) + _to_float(blank_cfg.get("score_bias"), 0.0)
    return _clamp01(score)


def evaluate_voiced_approval(
    row_context: Dict[str, object],
    *,
    profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Stage-2 voiced approval gate.
    Even if stage-1 blank risk does not mark blank, this gate can reject
    voiced promotion unless voiced/mel/f0 evidence is jointly sufficient.
    """
    resolved_profile = _resolve_profile(row_context, profile)
    blank_cfg = _profile_blank_cfg(resolved_profile)

    alias_type = str(row_context.get("alias_type", "") or "").strip().lower()
    gate_target_alias = {"cv", "cv_head", "mono", "vcv", "vv"}
    required = alias_type in gate_target_alias
    enabled = _to_bool(blank_cfg.get("voiced_gate_enable"), True)
    if not enabled:
        return {
            "enabled": False,
            "required": bool(required),
            "approved": True,
            "reason": "gate_disabled",
            "voiced_conf": 0.0,
            "rms_norm": 0.0,
            "f0_continuity": 0.0,
        }
    if not required:
        return {
            "enabled": True,
            "required": False,
            "approved": True,
            "reason": "alias_not_target",
            "voiced_conf": 0.0,
            "rms_norm": 0.0,
            "f0_continuity": 0.0,
        }

    voiced_conf = _clamp01(
        max(
            _to_float(row_context.get("syllable_mel_voiced_conf"), 0.0),
            _to_float(row_context.get("mel_voiced_formant_ratio"), 0.0),
            _to_float(row_context.get("f0_voicing_near_pre"), 0.0),
        )
    )
    rms_norm = max(
        0.0,
        _to_float(row_context.get("rms_norm_wav"), 0.0),
        _to_float(row_context.get("rms_norm_vb"), 0.0),
    )
    f0_continuity = _clamp01(
        max(
            _to_float(row_context.get("f0_continuity"), 0.0),
            _to_float(row_context.get("f0_valid_ratio"), 0.0),
        )
    )

    voiced_conf_min = _clamp01(_to_float(blank_cfg.get("voiced_conf_min"), 0.56))
    rms_min = max(0.0, _to_float(blank_cfg.get("voiced_rms_min"), 0.18))
    f0_cont_min = _clamp01(_to_float(blank_cfg.get("voiced_f0_continuity_min"), 0.16))

    checks = {
        "voiced_conf": bool(voiced_conf >= voiced_conf_min),
        "rms_norm": bool(rms_norm >= rms_min),
        "f0_continuity": bool(f0_continuity >= f0_cont_min),
    }
    approved = bool(checks["voiced_conf"] and checks["rms_norm"] and checks["f0_continuity"])
    if approved:
        reason = "approved"
    else:
        failed = [k for k, ok in checks.items() if not ok]
        reason = "fail:" + ",".join(failed) if failed else "fail:unknown"
    return {
        "enabled": True,
        "required": True,
        "approved": bool(approved),
        "reason": str(reason),
        "voiced_conf": float(voiced_conf),
        "rms_norm": float(rms_norm),
        "f0_continuity": float(f0_continuity),
        "voiced_conf_min": float(voiced_conf_min),
        "rms_min": float(rms_min),
        "f0_continuity_min": float(f0_cont_min),
    }


def compute_mel_reliability_score(
    row_context: Dict[str, object],
    *,
    profile: Optional[Dict[str, object]] = None,
    blank_risk_score: Optional[float] = None,
) -> float:
    resolved_profile = _resolve_profile(row_context, profile)
    mel_cfg = _profile_mel_cfg(resolved_profile)
    mel_fallback_penalty = _to_float(mel_cfg.get("fallback_penalty"), 0.0)
    if mel_patch_is_fallback(row_context):
        return _clamp01(max(0.0, 0.0 - mel_fallback_penalty))
    mel_energy = _to_float(row_context.get("mel_window_energy_mean"), 0.0)
    mel_silence = _to_float(row_context.get("mel_window_silence_ratio"), 0.0)
    blank_risk = (
        _clamp01(_to_float(blank_risk_score, 0.0))
        if blank_risk_score is not None
        else compute_blank_risk_score(row_context, profile=resolved_profile)
    )
    if mel_energy <= 1e-6 and mel_silence <= 1e-6:
        return _clamp01(max(0.0, 0.0 - mel_fallback_penalty))
    onset_mean = _to_float(row_context.get("mel_patch_onset_mean"), 0.0)
    tail_mean = _to_float(row_context.get("mel_patch_tail_mean"), 0.0)
    tail_low_ratio = _to_float(row_context.get("mel_patch_tail_low_ratio"), 1.0)
    patch_energy = 0.5 * (onset_mean + tail_mean)
    patch_energy_score = 1.0 / (1.0 + math.exp(-patch_energy)) if abs(patch_energy) < 40.0 else (1.0 if patch_energy > 0 else 0.0)
    patch_silence_penalty = 0.35 * max(0.0, min(1.0, tail_low_ratio))
    score = (
        (0.48 * mel_energy)
        + (0.22 * (1.0 - mel_silence))
        + (0.18 * (1.0 - blank_risk))
        + (0.12 * patch_energy_score)
        - patch_silence_penalty
    )
    score = (_to_float(mel_cfg.get("score_scale"), 1.0) * score) + _to_float(mel_cfg.get("score_bias"), 0.0)
    return _clamp01(score)


def is_mel_unreliable(
    row_context: Dict[str, object],
    threshold: Optional[float] = 0.42,
    *,
    profile: Optional[Dict[str, object]] = None,
    blank_risk_score: Optional[float] = None,
) -> bool:
    resolved_profile = _resolve_profile(row_context, profile)
    mel_cfg = _profile_mel_cfg(resolved_profile)
    resolved_threshold = float(threshold) if threshold is not None else _to_float(mel_cfg.get("threshold"), 0.42)
    return compute_mel_reliability_score(
        row_context,
        profile=resolved_profile,
        blank_risk_score=blank_risk_score,
    ) < float(resolved_threshold)


def blank_risk_flag(
    row_context: Dict[str, object],
    threshold: Optional[float] = 0.55,
    *,
    profile: Optional[Dict[str, object]] = None,
) -> int:
    resolved_profile = _resolve_profile(row_context, profile)
    blank_cfg = _profile_blank_cfg(resolved_profile)
    resolved_threshold = float(threshold) if threshold is not None else _to_float(blank_cfg.get("threshold"), 0.55)
    return 1 if compute_blank_risk_score(row_context, profile=resolved_profile) >= float(resolved_threshold) else 0


def apply_blank_risk_weight(
    base_weight: float,
    row_context: Dict[str, object],
    *,
    weight: float = 0.45,
    profile: Optional[Dict[str, object]] = None,
) -> float:
    score = compute_blank_risk_score(row_context, profile=profile)
    factor = 1.0 - (max(0.0, min(0.95, float(weight))) * score)
    return max(0.05, float(base_weight) * factor)
