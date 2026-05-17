"""Unit tests for BPM beat-grid prior (2026-05-17 design)."""
from __future__ import annotations

import numpy as np
import pytest

from core.coarse_crnn.bpm_prior import (
    BpmEstimate,
    bpm_min_confidence,
    bpm_prior_enabled,
    bpm_range_from_env,
    bpm_snap_tolerance_ms,
    compute_beat_grid,
    detect_lead_in_ms,
    estimate_bpm_acoustic,
    estimate_bpm_filename,
    parse_manual_bpm_env,
    role_snap_target_slot,
    snap_offset_to_grid,
)


# ---------------------------------------------------------------------------
# detect_lead_in_ms
# ---------------------------------------------------------------------------

def test_lead_in_finds_first_loud_frame():
    """200ms silence + 500ms tone @ 16kHz → lead-in ≈ 200ms."""
    sr = 16000
    silence = np.zeros(int(sr * 0.2), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.5)) / sr).astype(np.float32)
    audio = np.concatenate([silence, tone])
    lead_in = detect_lead_in_ms(audio, sr, frame_ms=20)
    assert 180.0 <= lead_in <= 240.0


def test_lead_in_zero_for_empty_audio():
    assert detect_lead_in_ms(np.zeros(0, dtype=np.float32), 16000) == 0.0


def test_lead_in_zero_for_silence_only():
    """If everything is below threshold, lead-in defaults to 0 (no anchor)."""
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1 s of pure silence
    assert detect_lead_in_ms(audio, sr) == 0.0


# ---------------------------------------------------------------------------
# estimate_bpm_filename
# ---------------------------------------------------------------------------

def test_filename_bpm_typical_8mora_120bpm():
    """8 syllables in (8 × 250) = 2000 ms of singing + 250 ms lead-in
    + 150 ms tail = 2400 ms total → 2-mora-per-beat = 120 BPM."""
    candidates = estimate_bpm_filename(
        duration_ms=2400.0,
        slot_count=8,
        lead_in_ms=250.0,
        tail_buffer_ms=150.0,
        bpm_range=(100.0, 150.0),
    )
    assert candidates, "expected at least one in-range BPM candidate"
    bpm, mora = candidates[0]
    assert 115.0 <= bpm <= 125.0
    assert mora == 2


def test_filename_bpm_skips_out_of_range():
    """If neither 1-mora nor 2-mora interpretation lands in range, return []."""
    # Very long wav (40 s, 8 syllables, mora=2 → 6 BPM; mora=1 → 12 BPM)
    candidates = estimate_bpm_filename(
        duration_ms=40000.0,
        slot_count=8,
        bpm_range=(100.0, 150.0),
    )
    assert candidates == []


def test_filename_bpm_zero_slot_count_returns_empty():
    assert estimate_bpm_filename(duration_ms=2000.0, slot_count=0) == []
    assert estimate_bpm_filename(duration_ms=2000.0, slot_count=1) == []


# ---------------------------------------------------------------------------
# estimate_bpm_acoustic
# ---------------------------------------------------------------------------

def test_acoustic_bpm_finds_periodic_pulses_at_120bpm():
    """Synthetic 120 BPM (500ms inter-onset) pulse train → detect ~120 BPM."""
    sr = 16000
    duration_s = 5.0
    period_ms = 500.0  # 120 BPM
    n_pulses = int(duration_s * 1000 / period_ms)
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    pulse = 0.5 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.05)) / sr).astype(np.float32)
    for i in range(n_pulses):
        start = int(i * period_ms / 1000 * sr)
        end = start + pulse.size
        if end <= audio.size:
            audio[start:end] = pulse
    bpm, conf = estimate_bpm_acoustic(audio, sr, bpm_range=(100.0, 150.0))
    assert bpm is not None
    assert 110.0 <= bpm <= 130.0
    assert conf > 0.3


def test_acoustic_bpm_returns_none_for_silence():
    sr = 16000
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)
    bpm, conf = estimate_bpm_acoustic(audio, sr, bpm_range=(100.0, 150.0))
    assert bpm is None or conf == 0.0


# ---------------------------------------------------------------------------
# compute_beat_grid + snap_offset_to_grid
# ---------------------------------------------------------------------------

def test_beat_grid_120bpm_2mora_per_beat():
    """120 BPM, 2 mora/beat → 250 ms per mora. Lead-in 200 ms."""
    est = BpmEstimate(bpm=120.0, mora_per_beat=2, lead_in_ms=200.0, confidence=0.95, source="manual")
    grid = compute_beat_grid(est, slot_count=4)
    assert grid == [200.0, 450.0, 700.0, 950.0]


def test_beat_grid_empty_for_zero_slots():
    est = BpmEstimate(bpm=120.0, mora_per_beat=2, lead_in_ms=0.0, confidence=0.9, source="manual")
    assert compute_beat_grid(est, slot_count=0) == []


def test_snap_within_tolerance():
    grid = [200.0, 450.0, 700.0, 950.0]
    # Within 80 ms tolerance → snap to 700
    assert snap_offset_to_grid(680.0, grid, tolerance_ms=80.0) == 700.0
    # Exactly on beat → stays
    assert snap_offset_to_grid(450.0, grid, tolerance_ms=80.0) == 450.0


def test_snap_outside_tolerance_passes_through():
    grid = [200.0, 450.0, 700.0, 950.0]
    # 100 ms from nearest beat (700), tolerance 80 → no snap
    assert snap_offset_to_grid(600.0, grid, tolerance_ms=80.0) == 600.0


def test_snap_empty_grid_passes_through():
    assert snap_offset_to_grid(500.0, [], tolerance_ms=80.0) == 500.0


# ---------------------------------------------------------------------------
# role_snap_target_slot
# ---------------------------------------------------------------------------

def test_role_snap_target_head_cv_uses_own_slot():
    assert role_snap_target_slot("-cv", slot_index=0, slot_count=8) == 0
    assert role_snap_target_slot("cv", slot_index=3, slot_count=8) == 3


def test_role_snap_target_vv_uses_next_slot():
    """VV transition lives at start of next syllable."""
    assert role_snap_target_slot("vv", slot_index=2, slot_count=8) == 3


def test_role_snap_target_v_cv_uses_next_slot():
    assert role_snap_target_slot("v-cv", slot_index=1, slot_count=8) == 2


def test_role_snap_target_vc_coda_bridge_uses_next_slot():
    """KO coda-bridge ``NG g`` left token is a consonant → snap to next syllable start."""
    assert role_snap_target_slot("vc", slot_index=0, slot_count=8, alias="NG g") == 1
    assert role_snap_target_slot("vc", slot_index=2, slot_count=8, alias="ng g") == 3
    assert role_snap_target_slot("vc", slot_index=4, slot_count=8, alias="L H") == 5


def test_role_snap_target_vc_onset_skips():
    """Onset VC ``a bw`` boundary is mid-syllable → no clean beat alignment."""
    assert role_snap_target_slot("vc", slot_index=0, slot_count=8, alias="a bw") is None
    assert role_snap_target_slot("vc", slot_index=3, slot_count=8, alias="eu g") is None


def test_role_snap_target_other_bare_ko_cv_uses_own_slot():
    """'other' role catches bare KO CV (``geu``/``ga``) due to phone
    decomposer gap. These should snap to their own slot's beat."""
    assert role_snap_target_slot("other", slot_index=3, slot_count=8, alias="geu") == 3
    assert role_snap_target_slot("other", slot_index=5, slot_count=8, alias="ga") == 5
    assert role_snap_target_slot("other", slot_index=0, slot_count=8, alias="kkya") == 0


def test_role_snap_target_other_non_cv_skips():
    """'other' with non-CV pattern still skipped (e.g., bare vowels which
    are role='vowel', plus dash markers)."""
    assert role_snap_target_slot("other", slot_index=0, slot_count=8, alias="-") is None
    assert role_snap_target_slot("other", slot_index=0, slot_count=8, alias="a -") is None
    assert role_snap_target_slot("other", slot_index=0, slot_count=8, alias="") is None


def test_role_snap_target_br_skips():
    assert role_snap_target_slot("br", slot_index=0, slot_count=8) is None


def test_role_snap_target_clamps_overflow():
    """If slot_index+1 exceeds slot_count, clamp to last slot."""
    assert role_snap_target_slot("vv", slot_index=7, slot_count=8) == 7


# ---------------------------------------------------------------------------
# env parsers
# ---------------------------------------------------------------------------

def test_parse_manual_bpm_env_with_full_overrides():
    env = {
        "UTOA_BOUNDARY_BPM_MS": "128",
        "UTOA_BOUNDARY_BPM_MORA_PER_BEAT": "2",
        "UTOA_BOUNDARY_BPM_LEAD_IN_MS": "300",
    }
    est = parse_manual_bpm_env(env)
    assert est is not None
    assert est.bpm == 128.0
    assert est.mora_per_beat == 2
    assert est.lead_in_ms == 300.0
    assert est.confidence == 1.0
    assert est.source == "manual"


def test_parse_manual_bpm_env_unset_returns_none():
    assert parse_manual_bpm_env({}) is None
    assert parse_manual_bpm_env({"UTOA_BOUNDARY_BPM_MS": ""}) is None


def test_parse_manual_bpm_env_invalid_returns_none():
    assert parse_manual_bpm_env({"UTOA_BOUNDARY_BPM_MS": "not_a_number"}) is None
    assert parse_manual_bpm_env({"UTOA_BOUNDARY_BPM_MS": "0"}) is None


def test_bpm_prior_enabled_default_on():
    assert bpm_prior_enabled({}) is True
    assert bpm_prior_enabled({"UTOA_BOUNDARY_BPM_PRIOR_ENABLE": "on"}) is True
    assert bpm_prior_enabled({"UTOA_BOUNDARY_BPM_PRIOR_ENABLE": "off"}) is False


def test_bpm_range_default_and_override():
    assert bpm_range_from_env({}) == (100.0, 150.0)
    env = {"UTOA_BOUNDARY_BPM_RANGE_MIN": "90", "UTOA_BOUNDARY_BPM_RANGE_MAX": "180"}
    assert bpm_range_from_env(env) == (90.0, 180.0)
    # Inverted range falls back to default
    bad = {"UTOA_BOUNDARY_BPM_RANGE_MIN": "200", "UTOA_BOUNDARY_BPM_RANGE_MAX": "100"}
    assert bpm_range_from_env(bad) == (100.0, 150.0)


def test_bpm_snap_tolerance_default():
    assert bpm_snap_tolerance_ms({}) == 80.0
    assert bpm_snap_tolerance_ms({"UTOA_BOUNDARY_BPM_SNAP_TOLERANCE_MS": "120"}) == 120.0


def test_bpm_min_confidence_default():
    assert bpm_min_confidence({}) == 0.5
    assert bpm_min_confidence({"UTOA_BOUNDARY_BPM_MIN_CONFIDENCE": "0.85"}) == 0.85
    # Clamps out-of-range
    assert bpm_min_confidence({"UTOA_BOUNDARY_BPM_MIN_CONFIDENCE": "1.5"}) == 1.0
    assert bpm_min_confidence({"UTOA_BOUNDARY_BPM_MIN_CONFIDENCE": "-0.2"}) == 0.0
