import wave

import numpy as np

from core.cvn_feature_extractor import extract_frame_features, extract_frame_features_from_audio


def _write_stereo_wav_i16(path, sample_rate: int, left: np.ndarray, right: np.ndarray) -> None:
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768.0, 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def test_extract_frame_features_from_audio_detects_voiced_region():
    sample_rate = 16000
    silence = np.zeros(int(sample_rate * 0.20), dtype=np.float32)
    t_tone = np.arange(int(sample_rate * 0.25), dtype=np.float32) / float(sample_rate)
    tone = (0.35 * np.sin(2.0 * np.pi * 220.0 * t_tone)).astype(np.float32)
    noise = (
        0.12
        * np.random.RandomState(0).normal(size=int(sample_rate * 0.25)).astype(np.float32)
    )
    audio = np.concatenate([silence, tone, noise], axis=0)

    features = extract_frame_features_from_audio(audio, sample_rate, n_mels=40)
    assert features.num_frames > 10
    assert features.log_mel.shape == (features.num_frames, 40)
    assert features.f0_hz.shape[0] == features.num_frames
    assert features.rms.shape[0] == features.num_frames
    assert features.sample_rate == sample_rate

    silence_mask = features.times_ms < 180.0
    assert silence_mask.any()
    assert float(np.mean(features.rms[silence_mask])) < 1e-4

    tone_mask = (features.times_ms > 220.0) & (features.times_ms < 430.0)
    assert tone_mask.any()
    tone_voiced_ratio = float(np.mean(features.voiced_mask[tone_mask]))
    assert tone_voiced_ratio > 0.7
    tone_f0 = features.f0_hz[tone_mask & features.voiced_mask]
    assert tone_f0.size > 0
    assert 180.0 <= float(np.mean(tone_f0)) <= 280.0

    noise_mask = features.times_ms > 470.0
    assert noise_mask.any()
    assert float(np.mean(features.voiced_mask[noise_mask])) < 0.3


def test_extract_frame_features_from_audio_empty_signal_returns_single_frame():
    features = extract_frame_features_from_audio(np.zeros((0,), dtype=np.float32), 16000)
    assert features.num_frames == 1
    assert features.log_mel.shape == (1, 80)
    assert float(features.rms[0]) == 0.0
    assert bool(features.voiced_mask[0]) is False


def test_extract_frame_features_reads_wav_and_resamples(tmp_path):
    sample_rate = 22050
    t = np.arange(int(sample_rate * 0.1), dtype=np.float32) / float(sample_rate)
    left = (0.20 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    right = (0.20 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    wav_path = tmp_path / "stereo.wav"
    _write_stereo_wav_i16(wav_path, sample_rate, left, right)

    features = extract_frame_features(str(wav_path))
    assert features.sample_rate == 16000
    assert features.num_frames > 0
    assert features.log_mel.shape[1] == 80
    assert np.isfinite(features.log_mel).all()
