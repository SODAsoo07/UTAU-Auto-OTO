import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.pipeline_status import ALIGN_NOT_READY
from core.sequence_aligner import (
    _build_frame_voicing_mask,
    _insert_internal_pause_rows,
    _prepare_analysis_signal,
    _refine_word_boundaries_with_onset_cues,
    _resolve_word_rows,
    check_sequence_aligner_ready,
    run_sequence_align,
)


class _FakeIntervalTier:
    def __init__(self, name: str, minTime: float = 0.0, maxTime: float = 0.0):
        self.name = name
        self.minTime = minTime
        self.maxTime = maxTime
        self.rows = []

    def add(self, start: float, end: float, mark: str):
        self.rows.append((float(start), float(end), str(mark)))


class _FakeTextGrid:
    def __init__(self, minTime: float = 0.0, maxTime: float = 0.0):
        self.minTime = minTime
        self.maxTime = maxTime
        self.tiers = []

    def append(self, tier):
        self.tiers.append(tier)

    def write(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('File type = "ooTextFile"\n')
            handle.write(f"tiers={len(self.tiers)}\n")


class _FakeTextGridModule:
    TextGrid = _FakeTextGrid
    IntervalTier = _FakeIntervalTier


def _write_wav(path: str, *, sr: int = 16000, duration_sec: float = 0.5):
    frames = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, num=frames, endpoint=False, dtype=np.float32)
    # Use a simple tone with silent margins so sequence segmentation has both silence and voiced parts.
    signal = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    signal[: int(0.05 * sr)] = 0.0
    signal[-int(0.05 * sr) :] = 0.0
    pcm = np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())


def test_prepare_analysis_signal_removes_dc_and_caps_gain():
    sr = 16000
    x = np.full((sr,), 0.2, dtype=np.float32)
    x += 0.01 * np.sin(2.0 * np.pi * 120.0 * np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32))
    y = _prepare_analysis_signal(
        x,
        sr,
        dc_remove=True,
        highpass_hz=50.0,
        max_gain_db=6.0,
        target_rms=0.08,
    )
    assert abs(float(np.mean(y))) < 1e-3
    gain_ratio = float(np.sqrt(np.mean(np.square(y)) + 1e-12) / np.sqrt(np.mean(np.square(x)) + 1e-12))
    assert gain_ratio <= float(np.power(10.0, 6.0 / 20.0)) + 1e-6


def test_build_frame_voicing_mask_respects_consonant_cues():
    rms = np.array([0.001, 0.001, 0.012, 0.032, 0.002, 0.001], dtype=np.float32)
    zcr = np.array([0.01, 0.01, 0.24, 0.22, 0.01, 0.01], dtype=np.float32)
    hf = np.array([0.01, 0.01, 0.30, 0.28, 0.01, 0.01], dtype=np.float32)
    flux = np.array([0.01, 0.01, 0.18, 0.16, 0.01, 0.01], dtype=np.float32)
    mask = _build_frame_voicing_mask(
        rms,
        silence_threshold=0.02,
        min_active_frames=1,
        max_gap_frames=0,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        lookback_frames=2,
    )
    assert bool(mask[3])
    assert bool(mask[2])


def test_check_sequence_ready_rejects_unsupported_language(tmp_path):
    report = check_sequence_aligner_ready(language="english", wav_folder=str(tmp_path))
    assert str(report.get("code", "")) == ALIGN_NOT_READY


def test_run_sequence_align_writes_textgrid(tmp_path, monkeypatch):
    wav_dir = tmp_path / "wav"
    out_dir = tmp_path / "textgrids"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / "ga.wav"
    _write_wav(str(wav_path))

    # No lab file to ensure filename fallback path is exercised.
    monkeypatch.setattr("core.sequence_aligner._import_textgrid_module", lambda: (_FakeTextGridModule, ""))

    ok, message = run_sequence_align(
        "",
        str(wav_dir),
        "",
        str(out_dir),
        language="korean",
        callback=None,
    )
    assert ok is True
    assert "sequence alignment complete" in str(message)
    assert os.path.isfile(str(out_dir / "ga.TextGrid"))


def test_resolve_word_rows_keeps_nonzero_spans_for_many_tokens():
    words = [f"w{i}" for i in range(12)]
    # Simulate a fully active region from frame 0 to frame 119 with 10ms hop.
    labels = ["stable_vowel"] * 120
    rows = _resolve_word_rows(
        words,
        duration_sec=1.2,
        labels=labels,
        hop_sec=0.01,
        min_span_ms=20.0,
    )
    active = [row for row in rows if str(row[2]).strip().lower() not in {"", "spn", "sil", "pau"}]
    assert len(active) == len(words)
    assert min((float(end) - float(start)) for start, end, _ in active) > 0.003


def test_insert_internal_pause_rows_splits_low_energy_gap():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "ka"),
    ]
    labels = ["stable_vowel"] * 100
    for i in range(47, 53):
        labels[i] = "sil"
    rms = np.full((100,), 0.08, dtype=np.float32)
    rms[47:53] = 0.001

    updated = _insert_internal_pause_rows(
        rows,
        duration_sec=1.0,
        labels=labels,
        rms=rms,
        hop_sec=0.01,
        silence_threshold=0.02,
        min_pause_ms=18.0,
        search_ms=45.0,
    )
    pause_rows = [row for row in updated if str(row[2]).strip().lower() in {"spn", "sil", "pau"}]
    assert pause_rows
    assert any((float(e) - float(s)) >= 0.015 for s, e, _ in pause_rows)


def test_insert_internal_pause_rows_splits_energy_gap_without_sil_labels():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "ka"),
    ]
    labels = ["stable_vowel"] * 100
    rms = np.full((100,), 0.08, dtype=np.float32)
    rms[47:54] = 0.004
    zcr = np.full((100,), 0.05, dtype=np.float32)
    hf = np.full((100,), 0.06, dtype=np.float32)
    flux = np.full((100,), 0.04, dtype=np.float32)

    updated = _insert_internal_pause_rows(
        rows,
        duration_sec=1.0,
        labels=labels,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        hop_sec=0.01,
        silence_threshold=0.02,
        min_pause_ms=18.0,
        search_ms=45.0,
    )
    pause_rows = [row for row in updated if str(row[2]).strip().lower() in {"spn", "sil", "pau"}]
    assert pause_rows
    assert any((float(e) - float(s)) >= 0.018 for s, e, _ in pause_rows)


def test_insert_internal_pause_rows_avoids_unvoiced_fricative_split():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "sa"),
    ]
    labels = ["stable_vowel"] * 100
    rms = np.full((100,), 0.07, dtype=np.float32)
    rms[47:54] = 0.011
    zcr = np.full((100,), 0.08, dtype=np.float32)
    hf = np.full((100,), 0.10, dtype=np.float32)
    flux = np.full((100,), 0.05, dtype=np.float32)
    zcr[47:54] = 0.24
    hf[47:54] = 0.28
    flux[47:54] = 0.18

    updated = _insert_internal_pause_rows(
        rows,
        duration_sec=1.0,
        labels=labels,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        hop_sec=0.01,
        silence_threshold=0.02,
        min_pause_ms=18.0,
        search_ms=45.0,
    )
    pause_rows = [row for row in updated if str(row[2]).strip().lower() in {"spn", "sil", "pau"}]
    assert not pause_rows


def test_refine_vowel_vowel_legato_keeps_boundary_near_valley():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "i"),
    ]
    n = 100
    hop = 0.01
    rms = np.full((n,), 0.12, dtype=np.float32)
    zcr = np.full((n,), 0.08, dtype=np.float32)
    hf = np.full((n,), 0.15, dtype=np.float32)
    flux = np.full((n,), 0.10, dtype=np.float32)
    # Set a gentle valley near 0.52s in legato region.
    rms[52:54] = 0.06
    flux[52:54] = 0.05

    refined = _refine_word_boundaries_with_onset_cues(
        rows,
        duration_sec=1.0,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        labels=["stable_vowel"] * n,
        language="japanese",
        hop_sec=hop,
        silence_threshold=0.04,
    )
    boundary = float(refined[0][1])
    assert 0.50 <= boundary <= 0.54


def test_refine_japanese_nasalized_g_onset_prefers_smooth_valley():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "ga"),
    ]
    n = 100
    hop = 0.01
    rms = np.full((n,), 0.10, dtype=np.float32)
    zcr = np.full((n,), 0.06, dtype=np.float32)
    hf = np.full((n,), 0.14, dtype=np.float32)
    flux = np.full((n,), 0.09, dtype=np.float32)
    # Slight valley after nominal boundary, typical of voiced/nasalized transition.
    rms[53:56] = 0.072
    flux[53:56] = 0.06

    refined = _refine_word_boundaries_with_onset_cues(
        rows,
        duration_sec=1.0,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        labels=["stable_vowel"] * n,
        language="japanese",
        hop_sec=hop,
        silence_threshold=0.035,
    )
    boundary = float(refined[0][1])
    assert 0.50 <= boundary <= 0.57


def test_refine_voiceless_onset_uses_cvn_candidates():
    rows = [
        (0.00, 0.50, "a"),
        (0.50, 1.00, "ka"),
    ]
    n = 100
    hop = 0.01
    rms = np.full((n,), 0.10, dtype=np.float32)
    zcr = np.full((n,), 0.07, dtype=np.float32)
    hf = np.full((n,), 0.10, dtype=np.float32)
    flux = np.full((n,), 0.08, dtype=np.float32)
    labels = ["stable_vowel"] * n

    cvn = np.full((n,), 0.18, dtype=np.float32)
    cvn[46:50] = 0.90
    cvn[50:54] = 0.36

    refined_base = _refine_word_boundaries_with_onset_cues(
        rows,
        duration_sec=1.0,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        labels=labels,
        language="japanese",
        hop_sec=hop,
        silence_threshold=0.04,
    )
    refined_cvn = _refine_word_boundaries_with_onset_cues(
        rows,
        duration_sec=1.0,
        rms=rms,
        zcr=zcr,
        hf_ratio=hf,
        flux=flux,
        labels=labels,
        language="japanese",
        hop_sec=hop,
        silence_threshold=0.04,
        cvn_c_prob=cvn,
        cvn_assist_mix=0.7,
    )
    base_boundary = float(refined_base[0][1])
    cvn_boundary = float(refined_cvn[0][1])
    assert base_boundary >= 0.495
    assert cvn_boundary < base_boundary
    assert cvn_boundary <= 0.495
