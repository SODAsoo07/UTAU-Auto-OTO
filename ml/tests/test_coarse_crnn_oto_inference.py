from __future__ import annotations

import numpy as np

from core.coarse_crnn.oto_inference import _trim_leading_silence_for_inference


def test_trim_leading_silence_keeps_padding_before_active_audio(monkeypatch):
    sr = 16000
    silence = np.zeros((sr,), dtype=np.float32)
    t = np.arange(int(sr * 0.4), dtype=np.float32) / float(sr)
    voiced = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    samples = np.concatenate([silence, voiced])

    monkeypatch.delenv("UTOA_OTO_CRNN_SKIP_LEADING_SILENCE_KEEP_MS", raising=False)
    trimmed, skipped_ms = _trim_leading_silence_for_inference(samples, sample_rate=sr)

    assert 820.0 <= skipped_ms <= 880.0
    assert len(trimmed) < len(samples)
    assert len(trimmed) == len(samples) - int(round(sr * skipped_ms / 1000.0))


def test_trim_leading_silence_ignores_short_preroll(monkeypatch):
    sr = 16000
    silence = np.zeros((int(sr * 0.12),), dtype=np.float32)
    voiced = np.ones((int(sr * 0.2),), dtype=np.float32) * 0.2
    samples = np.concatenate([silence, voiced])

    monkeypatch.delenv("UTOA_OTO_CRNN_SKIP_LEADING_SILENCE_MIN_SKIP_MS", raising=False)
    trimmed, skipped_ms = _trim_leading_silence_for_inference(samples, sample_rate=sr)

    assert skipped_ms == 0.0
    assert trimmed is samples


def test_trim_leading_silence_can_be_disabled():
    sr = 16000
    samples = np.concatenate(
        [
            np.zeros((sr,), dtype=np.float32),
            np.ones((int(sr * 0.2),), dtype=np.float32) * 0.2,
        ]
    )

    trimmed, skipped_ms = _trim_leading_silence_for_inference(samples, sample_rate=sr, enabled=False)

    assert skipped_ms == 0.0
    assert trimmed is samples


def test_trim_leading_silence_handles_low_volume_noisy_preroll(monkeypatch):
    sr = 16000
    rng = np.random.default_rng(1234)
    noisy_preroll = rng.normal(0.0, 0.012, int(sr * 0.7)).astype(np.float32)
    t = np.arange(int(sr * 0.45), dtype=np.float32) / float(sr)
    quiet_voiced = (0.03 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    quiet_voiced += rng.normal(0.0, 0.012, quiet_voiced.shape[0]).astype(np.float32)
    samples = np.concatenate([noisy_preroll, quiet_voiced])

    monkeypatch.delenv("UTOA_OTO_CRNN_SKIP_LEADING_SILENCE_NOISE_MULTIPLIER", raising=False)
    trimmed, skipped_ms = _trim_leading_silence_for_inference(samples, sample_rate=sr)

    assert 450.0 <= skipped_ms <= 620.0
    assert len(trimmed) < len(samples)
