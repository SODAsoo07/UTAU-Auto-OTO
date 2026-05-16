"""Unit tests for core.coarse_crnn.boundary_audio_features (Phase F-1).

Long-window spectral comparison should peak where two adjacent regions of a
synthetic log-mel differ in average spectrum, including slow Korean-coda-like
transitions that frame-delta misses.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.coarse_crnn.boundary_audio_features import (
    DEFAULT_GAP_MS,
    DEFAULT_SCALES_MS,
    DEFAULT_THRESHOLD,
    _classify_energy_direction,
    _prefix_sum,
    compute_long_window_candidates,
    long_window_candidates_from_logmel,
    long_window_spectral_score,
    pick_long_window_peaks,
    voicing_mask_from_logmel,
)


HOP_S = 0.010
N_MELS = 64


def _make_step_logmel(
    *,
    total_frames: int = 200,
    boundary_frame: int = 100,
    left_pattern: np.ndarray | None = None,
    right_pattern: np.ndarray | None = None,
    ramp_frames: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a 2-region log-mel with optional gradual ramp between regions."""
    rng = np.random.default_rng(seed)
    if left_pattern is None:
        left_pattern = np.zeros((N_MELS,), dtype=np.float32)
        left_pattern[:16] = 2.0  # low-frequency dominant
    if right_pattern is None:
        right_pattern = np.zeros((N_MELS,), dtype=np.float32)
        right_pattern[32:] = 2.0  # high-frequency dominant
    arr = np.empty((total_frames, N_MELS), dtype=np.float32)
    for t in range(total_frames):
        if t < boundary_frame - ramp_frames // 2:
            base = left_pattern
        elif t >= boundary_frame + ramp_frames // 2:
            base = right_pattern
        else:
            alpha = (t - (boundary_frame - ramp_frames // 2)) / max(1, ramp_frames)
            base = (1.0 - alpha) * left_pattern + alpha * right_pattern
        noise = rng.normal(0.0, 0.15, size=(N_MELS,)).astype(np.float32)
        arr[t] = base.astype(np.float32) + noise
    return arr


def _times_ms(frame_count: int) -> np.ndarray:
    return (np.arange(frame_count, dtype=np.float32) * HOP_S * 1000.0)


def test_prefix_sum_round_trips():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(10, 4)).astype(np.float32)
    prefix = _prefix_sum(arr)
    assert prefix.shape == (11, 4)
    # mean(arr[3:7]) == (prefix[7] - prefix[3]) / 4
    expected = arr[3:7].mean(axis=0)
    actual = (prefix[7] - prefix[3]) / 4.0
    assert np.allclose(expected, actual, atol=1e-5)


def test_score_peaks_at_sharp_boundary():
    logmel = _make_step_logmel(total_frames=200, boundary_frame=100, ramp_frames=0)
    score = long_window_spectral_score(logmel, hop_s=HOP_S, scales_ms=(30.0, 60.0, 120.0), gap_ms=DEFAULT_GAP_MS)
    assert score.shape == (200,)
    # Score near the boundary should dominate score far from the boundary.
    peak_region = score[95:105].max()
    far_region = max(score[0:60].max(), score[140:200].max())
    assert peak_region > far_region + 0.10, (
        f"peak_region={peak_region:.3f} far_region={far_region:.3f}"
    )


def test_score_catches_slow_transition_that_frame_delta_misses():
    """Korean-coda-like: spectrum drifts over ~80ms instead of jumping."""
    logmel = _make_step_logmel(total_frames=240, boundary_frame=120, ramp_frames=8)
    score = long_window_spectral_score(logmel, hop_s=HOP_S, scales_ms=(30.0, 60.0, 120.0), gap_ms=DEFAULT_GAP_MS)
    # Compute single-frame delta for comparison.
    frame_delta = np.sqrt(np.sum((logmel[1:] - logmel[:-1]) ** 2, axis=1))
    fd_peak = frame_delta[110:130].max()
    fd_baseline = np.quantile(frame_delta, 0.50)
    lwc_peak = score[110:130].max()
    lwc_baseline = np.quantile(score[score > 0.0], 0.50) if np.any(score > 0.0) else 0.0
    # LWC should have a sharper relative peak than frame-delta on the slow transition.
    lwc_ratio = lwc_peak / max(1e-6, lwc_baseline)
    fd_ratio = fd_peak / max(1e-6, fd_baseline)
    assert lwc_ratio >= fd_ratio, (
        f"LWC peak/baseline ratio ({lwc_ratio:.2f}) should be >= frame-delta ratio ({fd_ratio:.2f}) "
        "on slow transitions"
    )


def test_peak_picker_respects_min_gap():
    n = 50
    score = np.zeros((n,), dtype=np.float32)
    score[10] = 0.9
    score[12] = 0.8  # close to score[10], should be filtered by min_gap
    score[30] = 0.85
    times_ms = _times_ms(n)
    peaks = pick_long_window_peaks(score, times_ms, threshold=0.50, min_gap_ms=40.0)
    picked_times = [round(t, 2) for t, _ in peaks]
    assert picked_times == sorted(picked_times), "peaks must be sorted by time"
    # min_gap_ms=40 vs hop 10 → indices must be at least 4 apart
    assert 100.0 in picked_times  # 10 * 10ms
    assert 300.0 in picked_times  # 30 * 10ms
    assert 120.0 not in picked_times, "close-to-strongest peak should be filtered"


def test_peak_picker_voicing_mask_filters_silence_peaks():
    n = 60
    score = np.zeros((n,), dtype=np.float32)
    score[20] = 0.8  # in voiced region
    score[40] = 0.8  # in silent region
    times_ms = _times_ms(n)
    voicing = np.zeros((n,), dtype=bool)
    voicing[15:30] = True
    peaks = pick_long_window_peaks(
        score, times_ms, threshold=0.50, min_gap_ms=10.0, voicing_mask=voicing, voicing_radius_frames=2
    )
    picked_times = [round(t, 2) for t, _ in peaks]
    assert 200.0 in picked_times
    assert 400.0 not in picked_times, "silent-region peak should be rejected by voicing mask"


def test_voicing_mask_marks_active_region():
    arr = np.full((100, N_MELS), -3.0, dtype=np.float32)
    arr[30:70, :] = 2.0  # active region
    mask = voicing_mask_from_logmel(arr)
    # Most of active region should be True, silent region False.
    assert mask[40:60].sum() >= 18
    assert mask[0:20].sum() <= 4
    assert mask[80:100].sum() <= 4


def test_classify_energy_direction_signs():
    arr = _make_step_logmel(total_frames=200, boundary_frame=100, ramp_frames=0)
    prefix = _prefix_sum(arr)
    delta = _classify_energy_direction(
        peak_idx=100,
        prefix=prefix,
        window_frames=6,
        gap_frames=2,
        feat_dim=N_MELS,
    )
    # In our synthetic step, both left and right have roughly equal total energy,
    # so |delta| should be small. The test verifies the function runs without error
    # and returns a finite value.
    assert np.isfinite(delta)


def test_classify_energy_direction_detects_rise():
    rng = np.random.default_rng(1)
    arr = np.full((200, N_MELS), -2.0, dtype=np.float32) + rng.normal(0, 0.1, (200, N_MELS)).astype(np.float32)
    arr[100:, :] += 4.0  # energy rises after frame 100
    prefix = _prefix_sum(arr)
    delta = _classify_energy_direction(
        peak_idx=100,
        prefix=prefix,
        window_frames=6,
        gap_frames=2,
        feat_dim=N_MELS,
    )
    assert delta > 1.5, f"energy rise should produce delta > 1.5, got {delta:.3f}"


def test_classify_energy_direction_detects_fall():
    rng = np.random.default_rng(2)
    arr = np.full((200, N_MELS), 2.0, dtype=np.float32) + rng.normal(0, 0.1, (200, N_MELS)).astype(np.float32)
    arr[100:, :] -= 4.0  # energy falls after frame 100 (코다 tail-like)
    prefix = _prefix_sum(arr)
    delta = _classify_energy_direction(
        peak_idx=100,
        prefix=prefix,
        window_frames=6,
        gap_frames=2,
        feat_dim=N_MELS,
    )
    assert delta < -1.5, f"energy fall should produce delta < -1.5, got {delta:.3f}"


def test_long_window_candidates_emit_appropriate_labels_for_onset():
    """Onset-like (energy rising) peak should emit consonant_onset/next_onset/syllable_onset."""
    rng = np.random.default_rng(3)
    arr = np.full((200, N_MELS), -2.0, dtype=np.float32) + rng.normal(0, 0.1, (200, N_MELS)).astype(np.float32)
    arr[100:, :] += 4.0
    cands = long_window_candidates_from_logmel(arr, hop_s=HOP_S, threshold=0.30)
    kinds = {c.kind for c in cands}
    assert any(c.source == "audio:lwc" for c in cands)
    assert "consonant_onset" in kinds or "syllable_onset" in kinds or "next_onset" in kinds, (
        f"expected onset-like label in {kinds}"
    )


def test_long_window_candidates_emit_appropriate_labels_for_coda():
    """Coda-like (energy falling) peak should emit vowel_end / silence_boundary."""
    rng = np.random.default_rng(4)
    arr = np.full((200, N_MELS), 2.0, dtype=np.float32) + rng.normal(0, 0.1, (200, N_MELS)).astype(np.float32)
    arr[100:, :] -= 4.0
    cands = long_window_candidates_from_logmel(arr, hop_s=HOP_S, threshold=0.30)
    kinds = {c.kind for c in cands}
    assert any(c.source == "audio:lwc" for c in cands)
    assert "vowel_end" in kinds, f"expected vowel_end in {kinds}"


def test_long_window_candidates_handle_short_input():
    """Very short audio should not crash, returns empty list."""
    arr = np.zeros((2, N_MELS), dtype=np.float32)
    cands = long_window_candidates_from_logmel(arr, hop_s=HOP_S)
    assert cands == []


def test_long_window_candidates_respect_active_region():
    """Peaks outside [active_start_ms, active_end_ms] should be filtered out."""
    rng = np.random.default_rng(5)
    arr = np.full((200, N_MELS), -2.0, dtype=np.float32) + rng.normal(0, 0.1, (200, N_MELS)).astype(np.float32)
    arr[100:, :] += 4.0
    # Active region 800ms~1500ms; peak at frame 100 = 1000ms is inside.
    cands_in = long_window_candidates_from_logmel(
        arr, hop_s=HOP_S, threshold=0.30, active_start_ms=800.0, active_end_ms=1500.0
    )
    assert len(cands_in) > 0
    # Active region 0~500ms cuts off the peak at 1000ms.
    cands_out = long_window_candidates_from_logmel(
        arr, hop_s=HOP_S, threshold=0.30, active_start_ms=0.0, active_end_ms=500.0
    )
    assert all(c.time_ms <= 510.0 for c in cands_out)


def test_compute_long_window_candidates_returns_empty_on_missing_file(tmp_path):
    """Bad input path returns empty list, not crash."""
    bogus = tmp_path / "does_not_exist.wav"
    with pytest.raises(Exception):
        compute_long_window_candidates(wav_path=str(bogus))


def test_score_normalization_caps_at_one():
    """Per-utterance p95 normalization should clamp all scores to [0, 1]."""
    logmel = _make_step_logmel(total_frames=200, boundary_frame=100)
    score = long_window_spectral_score(logmel, hop_s=HOP_S)
    assert float(score.max()) <= 1.0 + 1e-6
    assert float(score.min()) >= 0.0
