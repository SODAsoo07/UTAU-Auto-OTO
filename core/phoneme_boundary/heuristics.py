from __future__ import annotations

import numpy as np

from core.phoneme_boundary.types import BOUNDARY_LABELS


def compute_acoustic_boundary_scores(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    frame_count: int | None = None,
) -> dict[str, list[float]]:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    sr = int(sample_rate or 16000)
    if x.size <= 0 or sr <= 0:
        frames = int(frame_count or 0)
        return {label: [0.0] * frames for label in BOUNDARY_LABELS}
    win = max(16, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n_frames = 1 + int((x.size - win) // hop)
    if frame_count is not None:
        n_frames = max(0, int(frame_count))
    if n_frames <= 0:
        return {label: [] for label in BOUNDARY_LABELS}
    padded_len = (n_frames - 1) * hop + win
    if x.size < padded_len:
        x = np.pad(x, (0, padded_len - x.size))
    shape = (n_frames, win)
    strides = (x.strides[0] * hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    window = np.hanning(win).astype(np.float32)
    framed = frames * window[None, :]
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-10)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1).astype(np.float32)
    spec = np.abs(np.fft.rfft(framed, axis=1)).astype(np.float32)
    spec_sum = np.maximum(spec.sum(axis=1), 1e-8)
    flux = np.zeros((n_frames,), dtype=np.float32)
    if n_frames > 1:
        diff = np.maximum(0.0, spec[1:] - spec[:-1])
        flux[1:] = diff.sum(axis=1) / np.maximum(spec_sum[1:], 1e-8)
    flatness = np.exp(np.mean(np.log(np.maximum(spec, 1e-8)), axis=1)) / np.maximum(np.mean(spec, axis=1), 1e-8)
    active = _normalize(rms)
    zcr_n = _normalize(zcr)
    flux_n = _normalize(flux)
    flat_n = np.clip(flatness, 0.0, 1.0)
    stability = 1.0 - _normalize(np.abs(np.gradient(active)) + 0.8 * flux_n)
    silence = np.clip(1.0 - active * 1.35, 0.0, 1.0)
    vowel_prob = np.clip((0.70 * active) + (0.25 * stability) - (0.30 * zcr_n) - (0.18 * flat_n), 0.0, 1.0)
    consonant_prob = np.clip((0.55 * active) + (0.35 * zcr_n) + (0.35 * flux_n) - (0.25 * stability), 0.0, 1.0)
    dv = np.gradient(vowel_prob)
    dc = np.gradient(consonant_prob)
    vowel_onset = _normalize(np.maximum(0.0, dv) + 0.55 * flux_n + 0.25 * np.maximum(0.0, active - np.roll(active, 1)))
    vowel_nucleus = np.clip(vowel_prob * (0.60 + 0.40 * stability) * (1.0 - 0.35 * flux_n), 0.0, 1.0)
    consonant_onset = _normalize(np.maximum(0.0, dc) + 0.55 * flux_n + 0.25 * zcr_n)
    vowel_end = _normalize(np.maximum(0.0, -dv) + 0.35 * np.maximum(0.0, np.roll(silence, -1) - silence))
    phone_start = np.maximum(consonant_onset, vowel_onset)
    phone_end = np.maximum(vowel_end, _normalize(np.maximum(0.0, silence - np.roll(silence, 1))))
    silence_boundary = _normalize(np.abs(np.gradient(silence)) + 0.30 * np.maximum(0.0, silence))
    return {
        "phone_start": phone_start.astype(np.float32).tolist(),
        "phone_end": phone_end.astype(np.float32).tolist(),
        "consonant_onset": consonant_onset.astype(np.float32).tolist(),
        "vowel_onset": vowel_onset.astype(np.float32).tolist(),
        "vowel_nucleus": vowel_nucleus.astype(np.float32).tolist(),
        "vowel_end": vowel_end.astype(np.float32).tolist(),
        "silence_boundary": silence_boundary.astype(np.float32).tolist(),
        "acoustic_consonant_prob": consonant_prob.astype(np.float32).tolist(),
        "acoustic_vowel_prob": vowel_prob.astype(np.float32).tolist(),
        "acoustic_silence_prob": silence.astype(np.float32).tolist(),
        "acoustic_flux": flux_n.astype(np.float32).tolist(),
        "acoustic_stability": stability.astype(np.float32).tolist(),
    }


def blend_boundary_scores(
    model_scores: dict[str, list[float]],
    heuristic_scores: dict[str, list[float]],
    *,
    heuristic_weight: float = 0.28,
) -> dict[str, list[float]]:
    weight = max(0.0, min(0.95, float(heuristic_weight)))
    if weight <= 0.0:
        return {k: list(v) for k, v in model_scores.items()}
    out: dict[str, list[float]] = {}
    for label, values in model_scores.items():
        h_values = list(heuristic_scores.get(label, []) or [])
        m_values = list(values or [])
        n = min(len(m_values), len(h_values))
        if n <= 0:
            out[label] = m_values
            continue
        blended = [(1.0 - weight) * float(m_values[i]) + weight * float(h_values[i]) for i in range(n)]
        if len(m_values) > n:
            blended.extend(float(v) for v in m_values[n:])
        out[label] = [max(0.0, min(1.0, float(v))) for v in blended]
    return out


def blend_boundary_scores_adaptive(
    model_scores: dict[str, list[float]],
    heuristic_scores: dict[str, list[float]],
    *,
    quality_scores: list[float] | None,
    base_weight: float = 0.10,
    low_conf_threshold: float = 0.55,
    max_weight: float = 0.28,
) -> dict[str, list[float]]:
    if quality_scores is None:
        return blend_boundary_scores(model_scores, heuristic_scores, heuristic_weight=float(max_weight))
    q = np.asarray(list(quality_scores or []), dtype=np.float32)
    if q.size <= 0:
        return blend_boundary_scores(model_scores, heuristic_scores, heuristic_weight=float(max_weight))
    base = max(0.0, min(0.95, float(base_weight)))
    max_w = max(base, min(0.95, float(max_weight)))
    threshold = max(0.05, min(0.95, float(low_conf_threshold)))
    ratio = np.clip((threshold - q) / threshold, 0.0, 1.0)
    weights = base + (max_w - base) * ratio
    if weights.size >= 3:
        smooth = weights.copy()
        smooth[1:-1] = 0.2 * weights[:-2] + 0.6 * weights[1:-1] + 0.2 * weights[2:]
        weights = smooth
    out: dict[str, list[float]] = {}
    for label, values in model_scores.items():
        h_values = np.asarray(list(heuristic_scores.get(label, []) or []), dtype=np.float32)
        m_values = np.asarray(list(values or []), dtype=np.float32)
        n = int(min(m_values.size, h_values.size, weights.size))
        if n <= 0:
            out[label] = m_values.astype(np.float32).tolist()
            continue
        blended = (1.0 - weights[:n]) * m_values[:n] + weights[:n] * h_values[:n]
        if m_values.size > n:
            blended = np.concatenate([blended, m_values[n:]], axis=0)
        out[label] = np.clip(blended, 0.0, 1.0).astype(np.float32).tolist()
    return out


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return arr
    lo = float(np.percentile(arr, 5.0))
    hi = float(np.percentile(arr, 95.0))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

__all__ = ["blend_boundary_scores", "blend_boundary_scores_adaptive", "compute_acoustic_boundary_scores"]
