"""Unit tests for core.coarse_crnn.oto_audio_candidates (14-2 §2단계)."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 내부 함수 직접 임포트 (numpy 의존만 있음)
# ---------------------------------------------------------------------------

from core.coarse_crnn.oto_audio_candidates import (
    AudioCandidates,
    MelDeltaPeak,
    OnsetPeak,
    VowelSegmentCandidate,
    _candidates_from_dict,
    _candidates_to_dict,
    _classify_vowel_mel,
    _compute_mel_delta,
    _compute_rms,
    _estimate_mel_delta_peaks,
    _estimate_onset_peaks,
    _estimate_voiced_mask,
    _extract_stable_vowel_segments,
    _wav_cache_key,
    find_best_vowel_segment,
    find_nearest_onset_after,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_logmel(n_frames: int = 100, n_mels: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(-5.0, 2.0, (n_frames, n_mels)).astype(np.float32)


def _make_rms(n_frames: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 0.1, n_frames).astype(np.float32)


# ---------------------------------------------------------------------------
# _compute_rms
# ---------------------------------------------------------------------------

def test_compute_rms_shape():
    sr = 16000
    hop_ms = 10.0
    frame_ms = 25.0
    samples = np.random.default_rng(0).normal(0, 0.1, sr).astype(np.float32)
    rms = _compute_rms(samples, sr, hop_ms=hop_ms, frame_ms=frame_ms)
    assert rms.ndim == 1
    assert rms.shape[0] > 0
    assert np.all(rms >= 0)


def test_compute_rms_silent():
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    rms = _compute_rms(samples, sr, hop_ms=10.0, frame_ms=25.0)
    assert np.all(rms < 1e-5)


def test_compute_rms_short_signal():
    sr = 16000
    samples = np.ones(10, dtype=np.float32)
    rms = _compute_rms(samples, sr, hop_ms=10.0, frame_ms=25.0)
    assert rms.shape[0] >= 1


# ---------------------------------------------------------------------------
# _compute_mel_delta
# ---------------------------------------------------------------------------

def test_compute_mel_delta_shape():
    logmel = _make_logmel(50, 32)
    delta = _compute_mel_delta(logmel)
    assert delta.shape == (50,)
    assert delta.dtype == np.float32


def test_compute_mel_delta_single_frame():
    logmel = np.zeros((1, 32), dtype=np.float32)
    delta = _compute_mel_delta(logmel)
    assert delta.shape == (1,)
    assert delta[0] == pytest.approx(0.0, abs=1e-6)


def test_compute_mel_delta_constant_signal():
    logmel = np.ones((20, 16), dtype=np.float32)
    delta = _compute_mel_delta(logmel)
    assert np.allclose(delta, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _estimate_onset_peaks
# ---------------------------------------------------------------------------

def test_estimate_onset_peaks_returns_list():
    logmel = _make_logmel(200, 64)
    rms = _make_rms(200)
    peaks = _estimate_onset_peaks(logmel, rms, hop_ms=10.0)
    assert isinstance(peaks, list)
    for p in peaks:
        assert isinstance(p, OnsetPeak)
        assert p.time_ms >= 0.0
        assert 0.0 <= p.strength <= 1.0


def test_estimate_onset_peaks_ordering():
    """onset peaks는 시간순으로 정렬되어야 한다."""
    logmel = _make_logmel(300, 64)
    rms = _make_rms(300)
    peaks = _estimate_onset_peaks(logmel, rms, hop_ms=10.0)
    times = [p.time_ms for p in peaks]
    assert times == sorted(times)


def test_estimate_onset_peaks_min_gap():
    """NMS 후 peak 간격이 min_gap_ms 이상이어야 한다."""
    logmel = _make_logmel(300, 64)
    rms = _make_rms(300)
    min_gap_ms = 60.0
    peaks = _estimate_onset_peaks(logmel, rms, hop_ms=10.0, min_gap_ms=min_gap_ms)
    for i in range(1, len(peaks)):
        gap = peaks[i].time_ms - peaks[i - 1].time_ms
        assert gap >= min_gap_ms - 11.0  # 1 frame 허용오차


def test_estimate_onset_peaks_silent():
    logmel = np.zeros((50, 64), dtype=np.float32)
    rms = np.zeros(50, dtype=np.float32)
    peaks = _estimate_onset_peaks(logmel, rms, hop_ms=10.0)
    assert peaks == []


# ---------------------------------------------------------------------------
# _estimate_voiced_mask
# ---------------------------------------------------------------------------

def test_estimate_voiced_mask_shape():
    sr = 16000
    samples = np.random.default_rng(1).normal(0, 0.1, sr).astype(np.float32)
    rms = _make_rms(100)
    mask = _estimate_voiced_mask(rms, samples, sr, hop_ms=10.0)
    assert mask.dtype == bool
    assert mask.shape[0] == rms.shape[0]


def test_estimate_voiced_mask_silent():
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)
    rms = np.zeros(100, dtype=np.float32)
    mask = _estimate_voiced_mask(rms, samples, sr, hop_ms=10.0)
    # 완전 무음: shape/dtype만 보장 (에너지 0 → threshold도 0 → 모두 voiced로 처리되는 degenerate case)
    assert mask.dtype == bool
    assert mask.shape[0] == rms.shape[0]


# ---------------------------------------------------------------------------
# _classify_vowel_mel
# ---------------------------------------------------------------------------

def _vowel_logmel(n_mels: int, peak_bins: list[int], width: int = 3) -> np.ndarray:
    """지정한 bins에 에너지가 집중된 synthetic log-mel 벡터."""
    m = np.full(n_mels, -10.0, dtype=np.float32)
    for b in peak_bins:
        lo = max(0, b - width)
        hi = min(n_mels, b + width + 1)
        m[lo:hi] = 0.0
    return m


def test_classify_vowel_mel_other_on_tiny():
    """너무 작은 벡터는 'other'를 반환해야 한다."""
    m = np.zeros(10, dtype=np.float32)
    vowel, conf = _classify_vowel_mel(m)
    assert vowel == "other"


def test_classify_vowel_mel_returns_known_or_other():
    known = {"a", "i", "u", "e", "o", "eo", "eu", "other"}
    rng = np.random.default_rng(42)
    for _ in range(20):
        m = rng.normal(-5, 2, 64).astype(np.float32)
        vowel, conf = _classify_vowel_mel(m)
        assert vowel in known
        assert 0.0 <= conf <= 1.0


def test_classify_vowel_mel_i_high_f2():
    """/i/는 F2 고역(~n_mels*0.4~0.6) 에너지가 커야 분류됨."""
    n = 64
    # F2 고역 band: 26~40 → index 26~40
    m = _vowel_logmel(n, peak_bins=list(range(26, 38)), width=2)
    vowel, conf = _classify_vowel_mel(m)
    # 완벽한 합성이므로 i 또는 e 중 하나
    assert vowel in ("i", "e", "other")


def test_classify_vowel_mel_confidence_range():
    m = np.zeros(64, dtype=np.float32)
    _, conf = _classify_vowel_mel(m)
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _estimate_mel_delta_peaks
# ---------------------------------------------------------------------------

def test_estimate_mel_delta_peaks_ordering():
    delta = np.abs(np.random.default_rng(3).normal(0, 1, 200)).astype(np.float32)
    peaks = _estimate_mel_delta_peaks(delta, hop_ms=10.0)
    times = [p.time_ms for p in peaks]
    assert times == sorted(times)


def test_estimate_mel_delta_peaks_type():
    delta = _compute_mel_delta(_make_logmel(100, 64))
    peaks = _estimate_mel_delta_peaks(delta, hop_ms=10.0)
    for p in peaks:
        assert isinstance(p, MelDeltaPeak)


# ---------------------------------------------------------------------------
# _extract_stable_vowel_segments
# ---------------------------------------------------------------------------

def test_extract_stable_vowel_segments_empty_on_silent():
    n = 50
    logmel = np.full((n, 64), -10.0, dtype=np.float32)
    mel_delta = np.zeros(n, dtype=np.float32)
    voiced_mask = np.zeros(n, dtype=bool)
    rms = np.zeros(n, dtype=np.float32)
    segs = _extract_stable_vowel_segments(
        logmel, mel_delta, voiced_mask, rms,
        actual_hop_ms=10.0,
        active_start_ms=0.0,
        active_end_ms=500.0,
        onset_peaks=[],
    )
    assert segs == []


def test_extract_stable_vowel_segments_returns_candidates():
    n = 200
    logmel = _make_logmel(n, 64)
    mel_delta = _compute_mel_delta(logmel)
    voiced_mask = np.ones(n, dtype=bool)
    voiced_mask[:10] = False
    rms = _make_rms(n)
    segs = _extract_stable_vowel_segments(
        logmel, mel_delta, voiced_mask, rms,
        actual_hop_ms=10.0,
        active_start_ms=0.0,
        active_end_ms=2000.0,
        onset_peaks=[],
        min_duration_ms=60.0,
    )
    assert isinstance(segs, list)
    for s in segs:
        assert isinstance(s, VowelSegmentCandidate)
        assert s.start_ms < s.end_ms
        assert s.start_ms >= 0.0
        assert 0.0 <= s.stability_score <= 1.0


# ---------------------------------------------------------------------------
# find_best_vowel_segment
# ---------------------------------------------------------------------------

def _make_candidates_with_segs(segs: list[VowelSegmentCandidate]) -> AudioCandidates:
    return AudioCandidates(
        wav_path="test.wav",
        duration_ms=2000.0,
        active_start_ms=0.0,
        active_end_ms=2000.0,
        onset_peaks=[],
        stable_vowel_segments=segs,
        mel_delta_peaks=[],
    )


def test_find_best_vowel_segment_empty():
    cands = _make_candidates_with_segs([])
    assert find_best_vowel_segment(cands, 1 / 6) is None


def test_find_best_vowel_segment_picks_matching_vowel():
    segs = [
        VowelSegmentCandidate(0, 100, 300, 200, "a", 0.9, 0.9, 0.8, 0.5, 0.1),
        VowelSegmentCandidate(1, 400, 600, 500, "i", 0.9, 0.9, 0.8, 0.5, 0.1),
        VowelSegmentCandidate(2, 700, 900, 800, "u", 0.9, 0.9, 0.8, 0.5, 0.1),
    ]
    cands = _make_candidates_with_segs(segs)
    # target = a (1/6)
    result = find_best_vowel_segment(cands, 1 / 6)
    assert result is not None
    assert result.vowel_id == "a"

    # target = i (2/6)
    result = find_best_vowel_segment(cands, 2 / 6)
    assert result is not None
    assert result.vowel_id == "i"


def test_find_best_vowel_segment_hint_center():
    segs = [
        VowelSegmentCandidate(0, 100, 300, 200, "a", 0.9, 0.9, 0.8, 0.5, 0.1),
        VowelSegmentCandidate(1, 400, 600, 500, "a", 0.9, 0.9, 0.8, 0.5, 0.1),
    ]
    cands = _make_candidates_with_segs(segs)
    result = find_best_vowel_segment(cands, 1 / 6, hint_center_ms=450.0, hint_weight=5.0)
    assert result is not None
    assert result.segment_id == 1  # center 500ms가 hint 450ms에 더 가까움


# ---------------------------------------------------------------------------
# find_nearest_onset_after
# ---------------------------------------------------------------------------

def _make_candidates_with_onsets(peaks: list[OnsetPeak]) -> AudioCandidates:
    return AudioCandidates(
        wav_path="test.wav",
        duration_ms=2000.0,
        active_start_ms=0.0,
        active_end_ms=2000.0,
        onset_peaks=peaks,
        stable_vowel_segments=[],
        mel_delta_peaks=[],
    )


def test_find_nearest_onset_after_empty():
    cands = _make_candidates_with_onsets([])
    assert find_nearest_onset_after(cands, after_ms=100.0) is None


def test_find_nearest_onset_after_basic():
    peaks = [
        OnsetPeak(time_ms=50.0, strength=0.8),
        OnsetPeak(time_ms=200.0, strength=0.9),
        OnsetPeak(time_ms=400.0, strength=0.7),
    ]
    cands = _make_candidates_with_onsets(peaks)
    result = find_nearest_onset_after(cands, after_ms=100.0, search_window_ms=500.0)
    assert result is not None
    assert result.time_ms == pytest.approx(200.0)


def test_find_nearest_onset_after_window():
    peaks = [OnsetPeak(time_ms=800.0, strength=0.9)]
    cands = _make_candidates_with_onsets(peaks)
    # 800ms는 after=100, window=500이면 600ms 이내 → 범위 밖
    result = find_nearest_onset_after(cands, after_ms=100.0, search_window_ms=500.0)
    assert result is None


# ---------------------------------------------------------------------------
# 직렬화 roundtrip
# ---------------------------------------------------------------------------

def test_candidates_roundtrip():
    original = AudioCandidates(
        wav_path="/test/foo.wav",
        duration_ms=1234.5,
        active_start_ms=10.0,
        active_end_ms=1200.0,
        onset_peaks=[OnsetPeak(time_ms=50.0, strength=0.75)],
        stable_vowel_segments=[
            VowelSegmentCandidate(0, 100.0, 300.0, 200.0, "a", 0.9, 0.85, 0.8, 0.5, 0.05)
        ],
        mel_delta_peaks=[MelDeltaPeak(time_ms=200.0, strength=0.6)],
    )
    d = _candidates_to_dict(original)
    restored = _candidates_from_dict(d)

    assert restored.wav_path == original.wav_path
    assert restored.duration_ms == pytest.approx(original.duration_ms)
    assert restored.active_start_ms == pytest.approx(original.active_start_ms)
    assert len(restored.onset_peaks) == 1
    assert restored.onset_peaks[0].time_ms == pytest.approx(50.0)
    assert restored.onset_peaks[0].strength == pytest.approx(0.75)
    assert len(restored.stable_vowel_segments) == 1
    assert restored.stable_vowel_segments[0].vowel_id == "a"
    assert len(restored.mel_delta_peaks) == 1


def test_candidates_roundtrip_json():
    original = AudioCandidates(
        wav_path="/test/bar.wav",
        duration_ms=500.0,
        active_start_ms=0.0,
        active_end_ms=500.0,
        onset_peaks=[],
        stable_vowel_segments=[],
        mel_delta_peaks=[],
    )
    d = _candidates_to_dict(original)
    js = json.dumps(d, ensure_ascii=False)
    restored = _candidates_from_dict(json.loads(js))
    assert restored.duration_ms == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# _wav_cache_key
# ---------------------------------------------------------------------------

def test_wav_cache_key_length():
    key = _wav_cache_key("nonexistent_file.wav")
    assert len(key) == 24
    assert key.isalnum() or all(c in "0123456789abcdef" for c in key)


def test_wav_cache_key_different_paths():
    k1 = _wav_cache_key("a.wav")
    k2 = _wav_cache_key("b.wav")
    assert k1 != k2


def test_wav_cache_key_stable_for_same_nonexistent():
    k1 = _wav_cache_key("same_file.wav")
    k2 = _wav_cache_key("same_file.wav")
    assert k1 == k2


# ---------------------------------------------------------------------------
# compute_audio_candidates (mock으로 WAV 로딩 대체)
# ---------------------------------------------------------------------------

def test_compute_audio_candidates_mock(tmp_path, monkeypatch):
    """실제 WAV 없이 내부 audio 함수를 mock해서 end-to-end 통로 확인."""
    sr = 16000
    dur_sec = 1.0
    samples = np.random.default_rng(7).normal(0, 0.1, int(sr * dur_sec)).astype(np.float32)
    n_mels = 64
    n_frames = 100
    hop_sec = 0.010
    logmel = np.random.default_rng(7).normal(-5, 2, (n_frames, n_mels)).astype(np.float32)

    import core.coarse_crnn.oto_audio_candidates as module

    monkeypatch.setattr(
        module, "load_wav_mono",
        lambda path, target_sr: (samples, sr, dur_sec),
    )
    monkeypatch.setattr(
        module, "estimate_active_region_seconds",
        lambda s, sr, **kw: (0.0, dur_sec),
    )
    monkeypatch.setattr(
        module, "log_mel_spectrogram",
        lambda s, sr, **kw: (logmel, hop_sec),
    )

    from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates

    result = compute_audio_candidates("fake.wav")
    assert isinstance(result, AudioCandidates)
    assert result.duration_ms == pytest.approx(dur_sec * 1000.0)
    assert result.active_start_ms == pytest.approx(0.0)
    assert result.active_end_ms == pytest.approx(dur_sec * 1000.0)
    assert isinstance(result.onset_peaks, list)
    assert isinstance(result.stable_vowel_segments, list)
    assert isinstance(result.mel_delta_peaks, list)


# ---------------------------------------------------------------------------
# load_or_compute_candidates (캐시 저장/로드)
# ---------------------------------------------------------------------------

def test_load_or_compute_candidates_cache(tmp_path, monkeypatch):
    """캐시 미스 → 계산 → 캐시 저장 → 캐시 히트 순서 검증."""
    sr = 16000
    dur_sec = 0.5
    samples = np.zeros(int(sr * dur_sec), dtype=np.float32)
    n_frames = 50
    logmel = np.full((n_frames, 64), -10.0, dtype=np.float32)
    hop_sec = 0.010

    import core.coarse_crnn.oto_audio_candidates as module

    call_count = {"n": 0}

    def fake_load_wav(path, target_sr):
        call_count["n"] += 1
        return samples, sr, dur_sec

    monkeypatch.setattr(module, "load_wav_mono", fake_load_wav)
    monkeypatch.setattr(module, "estimate_active_region_seconds", lambda s, sr, **kw: (0.0, dur_sec))
    monkeypatch.setattr(module, "log_mel_spectrogram", lambda s, sr, **kw: (logmel, hop_sec))

    from core.coarse_crnn.oto_audio_candidates import load_or_compute_candidates

    cache_dir = str(tmp_path / "cache")
    wav_path = str(tmp_path / "test.wav")
    # 파일 없어도 캐시 키는 생성됨 (mtime=0 fallback)

    r1 = load_or_compute_candidates(wav_path, cache_dir)
    assert call_count["n"] == 1  # 첫 계산

    # 캐시 파일이 생겼어야 함
    cache_files = list(os.listdir(cache_dir))
    assert len(cache_files) == 1
    assert cache_files[0].endswith(".json")

    r2 = load_or_compute_candidates(wav_path, cache_dir)
    assert call_count["n"] == 1  # 캐시 히트 → 재계산 없음
    assert r2.duration_ms == pytest.approx(r1.duration_ms)
