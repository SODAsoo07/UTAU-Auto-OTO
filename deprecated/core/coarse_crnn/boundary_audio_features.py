"""Audio-side long-window spectral comparison for boundary candidate generation.

This module implements Phase F-1 of the [[14-5-CVVC-체감정확도-개선-로드맵]]:
multi-scale long-window comparison on CMVN-normalized log-mel features.

The CRNN's frame-level head is biased toward sharp boundaries because its
training targets (Gaussian sigma 14~30 ms) are localized in time. Slow
transitions such as Korean codas (unreleased stops, nasal codas, liquid codas)
produce weak frame-delta signals and get missed. This module compares the
mean spectrum of a window before the candidate frame against a window after it
(with a small gap in between), which integrates slow change over 30~120 ms
and surfaces boundaries that the model fails to peak on.

Output is a list of `BoundaryCandidate` with `source="audio:lwc"`, fed into
the same `merge_candidates` path as model peaks and existing audio cues.
"""

from __future__ import annotations

import os

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_types import BoundaryCandidate


DEFAULT_SCALES_MS: tuple[float, ...] = (30.0, 60.0, 120.0)
DEFAULT_GAP_MS: float = 20.0
DEFAULT_PEAK_MIN_GAP_MS: float = 25.0
DEFAULT_THRESHOLD: float = 0.38
DEFAULT_VOICING_PERCENTILE_LO: float = 0.20
DEFAULT_VOICING_PERCENTILE_HI: float = 0.80
DEFAULT_VOICING_BLEND: float = 0.20


def _prefix_sum(mel: np.ndarray) -> np.ndarray:
    """Return [T+1, F] prefix sum so that mean(mel[a:b]) = (P[b] - P[a]) / (b-a)."""
    if mel.ndim != 2:
        raise ValueError("logmel must be 2-D [T, F]")
    feat_dim = int(mel.shape[1])
    return np.concatenate(
        [np.zeros((1, feat_dim), dtype=mel.dtype), np.cumsum(mel, axis=0)],
        axis=0,
    )


def _range_mean(prefix: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    counts = np.maximum(1, b - a).astype(np.float32)
    sums = prefix[b] - prefix[a]
    return (sums / counts[:, None]).astype(np.float32)


def long_window_spectral_score(
    logmel: np.ndarray,
    *,
    hop_s: float,
    scales_ms: tuple[float, ...] = DEFAULT_SCALES_MS,
    gap_ms: float = DEFAULT_GAP_MS,
) -> np.ndarray:
    """Compute a per-frame boundary score by comparing left and right windows.

    Returns a length-T float32 array normalized to [0, 1] via per-utterance
    p95 division. Higher = stronger spectral change between the window before
    and after frame t.

    Multi-scale outputs are combined by element-wise max so the score peaks
    wherever any time scale sees a strong transition.
    """
    arr = np.asarray(logmel, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0,), dtype=np.float32)
    frame_count = int(arr.shape[0])
    if frame_count < 4 or hop_s <= 0.0:
        return np.zeros((max(0, frame_count),), dtype=np.float32)

    prefix = _prefix_sum(arr)
    hop_ms = float(hop_s) * 1000.0
    gap_frames = max(1, int(round(float(gap_ms) / max(1e-6, hop_ms))))
    t_idx = np.arange(frame_count, dtype=np.int32)

    combined = np.zeros((frame_count,), dtype=np.float32)
    for window_ms in scales_ms:
        win = max(2, int(round(float(window_ms) / max(1e-6, hop_ms))))
        left_start = np.clip(t_idx - gap_frames - win, 0, frame_count)
        left_end = np.clip(t_idx - gap_frames, 0, frame_count)
        right_start = np.clip(t_idx + gap_frames, 0, frame_count)
        right_end = np.clip(t_idx + gap_frames + win, 0, frame_count)

        left_mean = _range_mean(prefix, left_start, left_end)
        right_mean = _range_mean(prefix, right_start, right_end)

        diff = right_mean - left_mean
        l2 = np.sqrt(np.sum(diff * diff, axis=1) / float(arr.shape[1])).astype(np.float32)

        valid = (left_end > left_start) & (right_end > right_start)
        l2 = np.where(valid, l2, 0.0).astype(np.float32)
        combined = np.maximum(combined, l2)

    # Per-utterance scale: divide by p95 so threshold is utterance-agnostic.
    nonzero = combined[combined > 0.0]
    if nonzero.size > 4:
        p95 = float(np.quantile(nonzero, 0.95))
    else:
        p95 = float(np.max(combined)) if combined.size else 1.0
    if p95 <= 1e-6:
        return np.zeros_like(combined)
    return np.clip(combined / p95, 0.0, 1.0).astype(np.float32)


def voicing_mask_from_logmel(
    logmel: np.ndarray,
    *,
    low_q: float = DEFAULT_VOICING_PERCENTILE_LO,
    high_q: float = DEFAULT_VOICING_PERCENTILE_HI,
    blend: float = DEFAULT_VOICING_BLEND,
) -> np.ndarray:
    """Cheap voicing mask: total log-mel energy above an utterance-adaptive floor."""
    arr = np.asarray(logmel, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return np.zeros((0,), dtype=bool)
    energy = np.sum(arr, axis=1)
    if energy.size <= 2:
        return np.ones_like(energy, dtype=bool)
    lo = float(np.quantile(energy, float(low_q)))
    hi = float(np.quantile(energy, float(high_q)))
    threshold = lo + float(blend) * (hi - lo)
    return (energy >= threshold).astype(bool)


def pick_long_window_peaks(
    score: np.ndarray,
    times_ms: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_ms: float = DEFAULT_PEAK_MIN_GAP_MS,
    voicing_mask: np.ndarray | None = None,
    voicing_radius_frames: int = 3,
) -> list[tuple[float, float]]:
    """Return [(time_ms, score)] of local maxima passing threshold + voicing gate.

    Peaks are sorted by ascending time. Ties are broken by score descending so
    the strongest peak in a min-gap window survives.
    """
    s = np.asarray(score, dtype=np.float32).reshape(-1)
    t = np.asarray(times_ms, dtype=np.float32).reshape(-1)
    if s.size < 3 or s.size != t.size:
        return []

    is_peak = (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= float(threshold))
    candidate_idx = np.where(is_peak)[0] + 1

    if voicing_mask is not None and len(voicing_mask) == len(s) and candidate_idx.size > 0:
        radius = max(0, int(voicing_radius_frames))
        keep: list[int] = []
        for i in candidate_idx:
            lo = max(0, int(i) - radius)
            hi = min(len(voicing_mask), int(i) + radius + 1)
            if bool(np.any(voicing_mask[lo:hi])):
                keep.append(int(i))
        candidate_idx = np.array(keep, dtype=np.int32)

    if candidate_idx.size == 0:
        return []

    order = candidate_idx[np.argsort(-s[candidate_idx])]
    accepted: list[tuple[float, float]] = []
    for i in order:
        t_ms = float(t[int(i)])
        if all(abs(t_ms - pt) >= float(min_gap_ms) for pt, _ in accepted):
            accepted.append((t_ms, float(s[int(i)])))
    accepted.sort(key=lambda item: item[0])
    return accepted


def _classify_energy_direction(
    *,
    peak_idx: int,
    prefix: np.ndarray,
    window_frames: int,
    gap_frames: int,
    feat_dim: int,
) -> float:
    """Return right_energy - left_energy (sum across CMVN-normalized mel bins).

    Positive → energy rising at the boundary (onset-like).
    Negative → energy falling (tail-like, e.g. Korean coda or note end).
    Near zero → comparable energy (vowel transition).
    """
    T = prefix.shape[0] - 1
    left_start = max(0, peak_idx - gap_frames - window_frames)
    left_end = max(0, peak_idx - gap_frames)
    right_start = min(T, peak_idx + gap_frames)
    right_end = min(T, peak_idx + gap_frames + window_frames)
    if left_end <= left_start or right_end <= right_start:
        return 0.0
    left_sum = float(np.sum(prefix[left_end] - prefix[left_start])) / float(max(1, left_end - left_start))
    right_sum = float(np.sum(prefix[right_end] - prefix[right_start])) / float(max(1, right_end - right_start))
    return float(right_sum - left_sum)


def _emit_candidates_for_peak(*, t_ms: float, score: float, energy_delta: float) -> list[BoundaryCandidate]:
    """Decide which boundary labels a long-window peak should populate.

    The peak itself is label-agnostic (it only says "spectra differ here"), so
    we emit a small fan-out of labels weighted by how likely each is given the
    direction of energy change. The wav_decoder will pick at most one per row.
    """
    base = max(0.0, min(1.0, float(score)))
    out: list[BoundaryCandidate] = []
    if energy_delta > 1.5:
        out.append(BoundaryCandidate(time_ms=t_ms, kind="consonant_onset", score=base, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="next_onset", score=base * 0.92, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="syllable_onset", score=base * 0.85, source="audio:lwc"))
    elif energy_delta < -1.5:
        out.append(BoundaryCandidate(time_ms=t_ms, kind="vowel_end", score=base, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="silence_boundary", score=base * 0.70, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="transition_peak", score=base * 0.78, source="audio:lwc"))
    else:
        out.append(BoundaryCandidate(time_ms=t_ms, kind="transition_peak", score=base, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="vowel_start", score=base * 0.85, source="audio:lwc"))
        out.append(BoundaryCandidate(time_ms=t_ms, kind="next_onset", score=base * 0.75, source="audio:lwc"))
    return out


def long_window_candidates_from_logmel(
    logmel: np.ndarray,
    *,
    hop_s: float,
    scales_ms: tuple[float, ...] = DEFAULT_SCALES_MS,
    gap_ms: float = DEFAULT_GAP_MS,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_ms: float = DEFAULT_PEAK_MIN_GAP_MS,
    active_start_ms: float | None = None,
    active_end_ms: float | None = None,
    voicing_mask: np.ndarray | None = None,
) -> list[BoundaryCandidate]:
    """End-to-end pipeline from log-mel features to BoundaryCandidate list."""
    arr = np.asarray(logmel, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 4 or hop_s <= 0.0:
        return []

    score = long_window_spectral_score(arr, hop_s=hop_s, scales_ms=scales_ms, gap_ms=gap_ms)
    if score.size == 0:
        return []

    frame_count = int(arr.shape[0])
    times_ms = (np.arange(frame_count, dtype=np.float32) * float(hop_s) * 1000.0)

    if voicing_mask is None:
        voicing_mask = voicing_mask_from_logmel(arr)

    peaks = pick_long_window_peaks(
        score,
        times_ms,
        threshold=float(threshold),
        min_gap_ms=float(min_gap_ms),
        voicing_mask=voicing_mask,
    )
    if not peaks:
        return []

    prefix = _prefix_sum(arr)
    hop_ms = float(hop_s) * 1000.0
    gap_frames = max(1, int(round(float(gap_ms) / max(1e-6, hop_ms))))
    # Use the middle scale for energy direction classification.
    classify_window_ms = float(scales_ms[len(scales_ms) // 2]) if scales_ms else 60.0
    classify_frames = max(2, int(round(classify_window_ms / max(1e-6, hop_ms))))

    feat_dim = int(arr.shape[1])
    out: list[BoundaryCandidate] = []
    for t_ms, peak_score in peaks:
        if active_start_ms is not None and t_ms < float(active_start_ms) - 5.0:
            continue
        if active_end_ms is not None and t_ms > float(active_end_ms) + 5.0:
            continue
        peak_idx = int(round(t_ms / max(1e-6, hop_ms)))
        peak_idx = max(0, min(frame_count - 1, peak_idx))
        energy_delta = _classify_energy_direction(
            peak_idx=peak_idx,
            prefix=prefix,
            window_frames=classify_frames,
            gap_frames=gap_frames,
            feat_dim=feat_dim,
        )
        out.extend(_emit_candidates_for_peak(t_ms=t_ms, score=peak_score, energy_delta=energy_delta))

    out.sort(key=lambda item: (item.time_ms, item.kind))
    return out


def compute_long_window_candidates(
    *,
    wav_path: str,
    hop_ms: float = 10.0,
    frame_ms: float = 25.0,
    n_mels: int = 64,
    scales_ms: tuple[float, ...] = DEFAULT_SCALES_MS,
    gap_ms: float = DEFAULT_GAP_MS,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap_ms: float = DEFAULT_PEAK_MIN_GAP_MS,
    active_start_ms: float | None = None,
    active_end_ms: float | None = None,
    target_sr: int = 16000,
) -> list[BoundaryCandidate]:
    """Load wav, compute log-mel, return long-window candidates.

    A convenience entry point used by `boundary_generator`. The caller can also
    reuse pre-computed log-mel by going through `long_window_candidates_from_logmel`.
    """
    samples, sr, _duration = load_wav_mono(str(wav_path), target_sr=int(target_sr))
    if samples.size <= 0 or sr <= 0:
        return []
    logmel, hop_s = log_mel_spectrogram(
        samples,
        int(sr),
        n_mels=int(n_mels),
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),
    )
    return long_window_candidates_from_logmel(
        logmel,
        hop_s=float(hop_s),
        scales_ms=scales_ms,
        gap_ms=float(gap_ms),
        threshold=float(threshold),
        min_gap_ms=float(min_gap_ms),
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
    )


def lwc_enabled_by_env(default: bool = True) -> bool:
    raw = str(os.environ.get("UTOA_BOUNDARY_LWC_ENABLE", "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def lwc_scales_from_env(default: tuple[float, ...] = DEFAULT_SCALES_MS) -> tuple[float, ...]:
    raw = str(os.environ.get("UTOA_BOUNDARY_LWC_SCALES_MS", "") or "").strip()
    if not raw:
        return tuple(default)
    parts: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except Exception:
            continue
        if value > 0.0:
            parts.append(value)
    return tuple(parts) if parts else tuple(default)


def lwc_float_from_env(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


__all__ = [
    "DEFAULT_GAP_MS",
    "DEFAULT_PEAK_MIN_GAP_MS",
    "DEFAULT_SCALES_MS",
    "DEFAULT_THRESHOLD",
    "compute_long_window_candidates",
    "long_window_candidates_from_logmel",
    "long_window_spectral_score",
    "lwc_enabled_by_env",
    "lwc_float_from_env",
    "lwc_scales_from_env",
    "pick_long_window_peaks",
    "voicing_mask_from_logmel",
]
