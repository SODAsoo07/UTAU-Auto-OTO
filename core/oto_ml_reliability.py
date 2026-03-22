"""
Centralized reliability + blank-risk helpers for OTO ML.

Used by runtime guards and training weighting so the criteria stay consistent.
"""

from __future__ import annotations

import math
from typing import Dict


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mel_patch_is_fallback(row_context: Dict[str, object]) -> bool:
    src = str(row_context.get("mel_patch_source", "") or "").strip().lower()
    if not src:
        onset = _to_float(row_context.get("mel_offset_candidate_ms"), 0.0)
        tail = _to_float(row_context.get("mel_cutoff_candidate_ms"), 0.0)
        if onset > 0.0 and tail > 0.0:
            return False
        return True
    return src not in {"mel_candidate"}


def compute_blank_risk_score(row_context: Dict[str, object]) -> float:
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
    score = (0.58 * blank_conf) + (0.32 * silence_ratio) - (0.22 * voiced_hint)
    return _clamp01(score)


def compute_mel_reliability_score(row_context: Dict[str, object]) -> float:
    if mel_patch_is_fallback(row_context):
        return 0.0
    mel_energy = _to_float(row_context.get("mel_window_energy_mean"), 0.0)
    mel_silence = _to_float(row_context.get("mel_window_silence_ratio"), 0.0)
    blank_risk = compute_blank_risk_score(row_context)
    if mel_energy <= 1e-6 and mel_silence <= 1e-6:
        return 0.0
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
    return _clamp01(score)


def is_mel_unreliable(row_context: Dict[str, object], threshold: float = 0.42) -> bool:
    return compute_mel_reliability_score(row_context) < float(threshold)


def blank_risk_flag(row_context: Dict[str, object], threshold: float = 0.55) -> int:
    return 1 if compute_blank_risk_score(row_context) >= float(threshold) else 0


def apply_blank_risk_weight(
    base_weight: float,
    row_context: Dict[str, object],
    *,
    weight: float = 0.45,
) -> float:
    score = compute_blank_risk_score(row_context)
    factor = 1.0 - (max(0.0, min(0.95, float(weight))) * score)
    return max(0.05, float(base_weight) * factor)
