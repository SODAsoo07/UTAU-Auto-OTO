"""Unit tests for core.coarse_crnn.anchor_timeline (Phase H A+B hybrid)."""
from __future__ import annotations

import math
import os
import struct
import wave

import numpy as np
import pytest

from core.coarse_crnn.anchor_timeline import (
    DEFAULT_VALLEY_MIN_DEPTH_RATIO,
    anchor_timeline_mode_from_env,
    best_onset_in_range,
    build_hybrid_anchor_timeline,
    build_segment_boundaries,
    compute_rms_envelope,
    detect_rms_valleys,
)
from core.coarse_crnn.boundary_types import BoundaryCandidate


SR = 16000


# ---------------------------------------------------------------------------
# Synthetic wav helpers
# ---------------------------------------------------------------------------

def _synth_syllables(*, syllable_centers_ms: list[float], syllable_durations_ms: list[float], valley_amp: float = 0.0, peak_amp: float = 0.4, total_ms: float = 5000.0) -> np.ndarray:
    """Build a wav whose RMS envelope has peaks at given centers, valleys in between.

    Noise is present at `valley_amp` everywhere. Each syllable is a tone with a
    short attack/release ramp added on top of the noise floor — so the
    envelope rises from noise to peak and back to noise (not to 0), matching
    how real speech sits above the recording's noise floor.
    """
    n = int(round(total_ms / 1000.0 * SR))
    rng = np.random.default_rng(0)
    noise = rng.normal(size=n).astype(np.float32) * float(valley_amp)
    out = noise.copy()
    for center_ms, dur_ms in zip(syllable_centers_ms, syllable_durations_ms):
        center = int(round(center_ms / 1000.0 * SR))
        half = max(1, int(round(dur_ms / 2.0 / 1000.0 * SR)))
        start = max(0, center - half)
        end = min(n, center + half)
        if end <= start:
            continue
        length = end - start
        # Sustained tone with 10ms attack/release ramps.
        taper = min(length // 3, max(1, int(round(0.010 * SR))))
        env = np.ones((length,), dtype=np.float32) * float(peak_amp)
        if length > 2 * taper:
            env[:taper] = float(peak_amp) * np.linspace(0.0, 1.0, taper, dtype=np.float32)
            env[-taper:] = float(peak_amp) * np.linspace(1.0, 0.0, taper, dtype=np.float32)
        t = np.arange(length, dtype=np.float32) / float(SR)
        tone = np.sin(2.0 * math.pi * 200.0 * t).astype(np.float32) * env
        out[start:end] = (out[start:end] + tone).astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _write_wav(samples: np.ndarray, path: str, sr: int = SR) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# compute_rms_envelope
# ---------------------------------------------------------------------------

def test_compute_rms_envelope_returns_nonempty():
    samples = np.random.default_rng(0).normal(size=SR).astype(np.float32) * 0.3
    rms, hop_s = compute_rms_envelope(samples, SR)
    assert rms.size > 50
    assert hop_s > 0.0
    assert rms.dtype == np.float32


def test_compute_rms_envelope_quiet_input_low_values():
    rms, _ = compute_rms_envelope(np.zeros(SR, dtype=np.float32), SR)
    assert float(rms.max()) < 1e-3


def test_compute_rms_envelope_handles_empty_input():
    rms, _ = compute_rms_envelope(np.zeros((0,), dtype=np.float32), SR)
    assert rms.size == 0


# ---------------------------------------------------------------------------
# detect_rms_valleys
# ---------------------------------------------------------------------------

def test_detect_valleys_finds_inter_syllable_dips():
    # 4 syllables centered at 500ms intervals starting 500ms, durations 200ms
    samples = _synth_syllables(
        syllable_centers_ms=[500.0, 1000.0, 1500.0, 2000.0],
        syllable_durations_ms=[200.0, 200.0, 200.0, 200.0],
        valley_amp=0.005,
        peak_amp=0.5,
        total_ms=2500.0,
    )
    rms, hop_s = compute_rms_envelope(samples, SR)
    valleys = detect_rms_valleys(
        rms,
        hop_s=hop_s,
        active_start_ms=300.0,
        active_end_ms=2200.0,
        min_depth_ratio=0.3,
        min_gap_ms=200.0,
        smooth_frames=7,
    )
    # 4 syllables → ≥2 inter-syllable valleys, well-separated by min_gap_ms.
    assert len(valleys) >= 2, f"expected at least 2 valleys, got {len(valleys)} at {[v.time_ms for v in valleys]}"
    # Valleys should fall in or near the gaps (around 750, 1250, 1750)
    valley_times = sorted(v.time_ms for v in valleys)
    expected_gaps = [750.0, 1250.0, 1750.0]
    matched = sum(1 for gap in expected_gaps if any(abs(v - gap) < 200.0 for v in valley_times))
    assert matched >= 2, f"only {matched}/3 expected gaps matched: {valley_times}"


def test_detect_valleys_empty_when_no_dips():
    # Constant amplitude — no valleys
    samples = np.ones((SR * 2,), dtype=np.float32) * 0.3
    rms, hop_s = compute_rms_envelope(samples, SR)
    valleys = detect_rms_valleys(
        rms, hop_s=hop_s, active_start_ms=100.0, active_end_ms=1900.0,
        min_depth_ratio=DEFAULT_VALLEY_MIN_DEPTH_RATIO,
    )
    # No valley should pass the depth gate
    assert valleys == []


def test_detect_valleys_respects_min_gap():
    # Multiple close dips — only the deepest within gap window should survive.
    n = SR * 2
    rms = np.full((n // 160,), 0.4, dtype=np.float32)  # ~10ms hop
    # Manually inject valleys at frames 30, 35, 60, 90 (300, 350, 600, 900 ms)
    rms[30] = 0.1
    rms[35] = 0.05  # deeper, very close to 30
    rms[60] = 0.1
    rms[90] = 0.1
    valleys = detect_rms_valleys(
        rms, hop_s=0.010,
        active_start_ms=100.0, active_end_ms=1500.0,
        min_depth_ratio=0.2,
        min_gap_ms=200.0,
    )
    times = sorted(v.time_ms for v in valleys)
    # Should keep the deeper one at frame 35, drop frame 30. Then 60, 90 each at ≥200ms.
    assert all(abs(times[i] - times[i - 1]) >= 200.0 for i in range(1, len(times)))


# ---------------------------------------------------------------------------
# build_segment_boundaries
# ---------------------------------------------------------------------------

def test_segment_boundaries_pure_linear_when_no_valleys():
    boundaries, mode = build_segment_boundaries(
        slot_count=4, valleys=[],
        active_start_ms=1000.0, active_end_ms=5000.0,
    )
    assert mode == "linear"
    assert len(boundaries) == 5
    assert boundaries[0] == 1000.0
    assert boundaries[-1] == 5000.0
    # Evenly spaced
    spacing = boundaries[1] - boundaries[0]
    for i in range(2, len(boundaries)):
        assert abs((boundaries[i] - boundaries[i - 1]) - spacing) < 1e-3


def test_segment_boundaries_valleys_only_when_enough():
    from core.coarse_crnn.anchor_timeline import RmsValley
    valleys = [
        RmsValley(time_ms=1500.0, rms_value=0.05, depth=0.8),
        RmsValley(time_ms=2700.0, rms_value=0.04, depth=0.9),
        RmsValley(time_ms=3800.0, rms_value=0.06, depth=0.7),
    ]
    boundaries, mode = build_segment_boundaries(
        slot_count=4, valleys=valleys,
        active_start_ms=1000.0, active_end_ms=5000.0,
    )
    assert mode == "valleys_only"
    assert boundaries == [1000.0, 1500.0, 2700.0, 3800.0, 5000.0]


def test_segment_boundaries_picks_deepest_when_excess():
    from core.coarse_crnn.anchor_timeline import RmsValley
    valleys = [
        RmsValley(time_ms=1200.0, rms_value=0.10, depth=0.2),  # shallow
        RmsValley(time_ms=2500.0, rms_value=0.05, depth=0.9),
        RmsValley(time_ms=3500.0, rms_value=0.04, depth=0.95),
        RmsValley(time_ms=4500.0, rms_value=0.10, depth=0.3),  # shallow
    ]
    boundaries, mode = build_segment_boundaries(
        slot_count=3, valleys=valleys,
        active_start_ms=1000.0, active_end_ms=5000.0,
    )
    # 3 slots → 2 boundaries → picks 2 deepest (2500, 3500)
    assert mode == "valleys_only"
    assert boundaries == [1000.0, 2500.0, 3500.0, 5000.0]


def test_segment_boundaries_mixed_when_some_valleys():
    from core.coarse_crnn.anchor_timeline import RmsValley
    valleys = [
        RmsValley(time_ms=2500.0, rms_value=0.05, depth=0.9),  # one valley near linear target
    ]
    boundaries, mode = build_segment_boundaries(
        slot_count=4, valleys=valleys,
        active_start_ms=1000.0, active_end_ms=5000.0,
    )
    # Need 3 boundaries; have 1 valley. Linear targets: 2000, 3000, 4000.
    # Valley at 2500 is closer to 2000 (and within segment width 1000).
    assert mode == "mixed"
    assert len(boundaries) == 5
    assert boundaries[0] == 1000.0
    assert boundaries[-1] == 5000.0


def test_segment_boundaries_slot_count_1():
    boundaries, mode = build_segment_boundaries(
        slot_count=1, valleys=[], active_start_ms=100.0, active_end_ms=900.0,
    )
    assert boundaries == [100.0, 900.0]
    assert mode == "linear"


# ---------------------------------------------------------------------------
# best_onset_in_range
# ---------------------------------------------------------------------------

def test_best_onset_in_range_filters_by_time():
    candidates = [
        BoundaryCandidate(time_ms=100.0, kind="syllable_onset", score=0.5, source="model"),
        BoundaryCandidate(time_ms=500.0, kind="syllable_onset", score=0.8, source="model"),
        BoundaryCandidate(time_ms=900.0, kind="syllable_onset", score=0.9, source="model"),
    ]
    out = best_onset_in_range(candidates, start_ms=200.0, end_ms=700.0)
    assert out is not None
    assert out.time_ms == 500.0


def test_best_onset_in_range_prefers_model_over_audio():
    candidates = [
        BoundaryCandidate(time_ms=400.0, kind="syllable_onset", score=0.95, source="audio:lwc"),
        BoundaryCandidate(time_ms=500.0, kind="syllable_onset", score=0.5, source="model"),
    ]
    out = best_onset_in_range(candidates, start_ms=300.0, end_ms=600.0)
    # Even though audio:lwc has higher score, model is preferred (tier 0 vs tier 1)
    assert out is not None
    assert out.source == "model"


def test_best_onset_in_range_returns_none_when_empty():
    assert best_onset_in_range([], start_ms=0.0, end_ms=1000.0) is None
    candidates = [BoundaryCandidate(time_ms=5000.0, kind="syllable_onset", score=0.9, source="model")]
    assert best_onset_in_range(candidates, start_ms=0.0, end_ms=100.0) is None


def test_best_onset_in_range_skips_non_onset_kinds():
    candidates = [
        BoundaryCandidate(time_ms=500.0, kind="vowel_end", score=0.95, source="model"),
        BoundaryCandidate(time_ms=600.0, kind="syllable_onset", score=0.4, source="model"),
    ]
    out = best_onset_in_range(candidates, start_ms=400.0, end_ms=700.0)
    assert out is not None
    assert out.time_ms == 600.0


# ---------------------------------------------------------------------------
# build_hybrid_anchor_timeline (end-to-end)
# ---------------------------------------------------------------------------

def test_hybrid_timeline_returns_slot_count_anchors(tmp_path):
    samples = _synth_syllables(
        syllable_centers_ms=[500.0, 1000.0, 1500.0, 2000.0],
        syllable_durations_ms=[200.0, 200.0, 200.0, 200.0],
        valley_amp=0.005,
        peak_amp=0.5,
        total_ms=2500.0,
    )
    wav_path = str(tmp_path / "s.wav")
    _write_wav(samples, wav_path)
    candidates = [
        BoundaryCandidate(time_ms=t, kind="syllable_onset", score=0.7, source="model")
        for t in [500.0, 1000.0, 1500.0, 2000.0]
    ]
    result = build_hybrid_anchor_timeline(
        wav_path=wav_path,
        slot_count=4,
        active_start_ms=300.0,
        active_end_ms=2200.0,
        candidates=candidates,
        valley_min_depth_ratio=0.3,
    )
    assert result is not None
    assert len(result.timeline) == 4
    # Monotonic
    for i in range(1, 4):
        assert result.timeline[i] >= result.timeline[i - 1]
    # Each anchor near a syllable center (within 200ms tolerance)
    expected = [500.0, 1000.0, 1500.0, 2000.0]
    for picked, want in zip(result.timeline, expected):
        assert abs(picked - want) < 200.0, f"anchor {picked} not near expected {want}"


def test_hybrid_timeline_returns_none_for_unreadable_wav(tmp_path):
    bogus = str(tmp_path / "missing.wav")
    result = build_hybrid_anchor_timeline(
        wav_path=bogus, slot_count=4,
        active_start_ms=0.0, active_end_ms=2000.0,
        candidates=[],
    )
    assert result is None


def test_hybrid_timeline_handles_flat_noise(tmp_path):
    # Pure noise — any valleys detected are spurious; verify the call still
    # returns a usable timeline rather than crashing.
    samples = (np.random.default_rng(0).normal(size=SR * 3).astype(np.float32) * 0.3).clip(-1.0, 1.0)
    wav_path = str(tmp_path / "flat.wav")
    _write_wav(samples, wav_path)
    result = build_hybrid_anchor_timeline(
        wav_path=wav_path, slot_count=4,
        active_start_ms=500.0, active_end_ms=2500.0,
        candidates=[],
    )
    assert result is not None
    assert len(result.timeline) == 4
    # Timeline must be monotonic regardless of valley/linear mode chosen.
    for i in range(1, 4):
        assert result.timeline[i] >= result.timeline[i - 1]
    # All anchors stay inside the active region.
    for t in result.timeline:
        assert 500.0 <= t <= 2500.0


def test_hybrid_timeline_onsets_used_counter(tmp_path):
    samples = _synth_syllables(
        syllable_centers_ms=[600.0, 1200.0, 1800.0],
        syllable_durations_ms=[200.0, 200.0, 200.0],
        valley_amp=0.005, peak_amp=0.5, total_ms=2500.0,
    )
    wav_path = str(tmp_path / "s.wav")
    _write_wav(samples, wav_path)
    candidates = [
        BoundaryCandidate(time_ms=605.0, kind="syllable_onset", score=0.9, source="model"),
        BoundaryCandidate(time_ms=1195.0, kind="consonant_onset", score=0.8, source="model"),
        # Third syllable has no onset candidate → segment falls back
    ]
    result = build_hybrid_anchor_timeline(
        wav_path=wav_path, slot_count=3,
        active_start_ms=400.0, active_end_ms=2100.0,
        candidates=candidates,
        valley_min_depth_ratio=0.3,
    )
    assert result is not None
    assert result.onsets_used == 2  # two segments resolved by onset, one by fallback


# ---------------------------------------------------------------------------
# anchor_timeline_mode_from_env
# ---------------------------------------------------------------------------

def test_mode_from_env_defaults_linear(monkeypatch):
    monkeypatch.delenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", raising=False)
    assert anchor_timeline_mode_from_env() == "linear"


def test_mode_from_env_recognizes_hybrid(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "hybrid")
    assert anchor_timeline_mode_from_env() == "hybrid"
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "AB")
    assert anchor_timeline_mode_from_env() == "hybrid"


def test_mode_from_env_recognizes_linear(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "linear")
    assert anchor_timeline_mode_from_env() == "linear"
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "legacy")
    assert anchor_timeline_mode_from_env() == "linear"


def test_mode_from_env_unknown_returns_default(monkeypatch):
    monkeypatch.setenv("UTOA_BOUNDARY_ANCHOR_TIMELINE", "nonsense")
    assert anchor_timeline_mode_from_env(default="hybrid") == "hybrid"
