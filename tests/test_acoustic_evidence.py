from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.acoustic_evidence import (
    FEATURE_BACKEND_ID,
    LEGACY_FEATURE_BACKEND_ID,
    MULTIBAND_V3_FEATURE_BACKEND_ID,
    extract_acoustic_evidence,
    load_acoustic_evidence_npz,
    write_acoustic_evidence_npz,
)


def _mixed_test_wav(path: Path) -> None:
    sample_rate = 44100
    rng = np.random.default_rng(1234)
    silence = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
    noise = (0.08 * rng.standard_normal(int(0.20 * sample_rate))).astype(np.float32)
    times = np.arange(int(0.65 * sample_rate), dtype=np.float64) / sample_rate
    voiced = (0.24 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    tail = np.zeros(int(0.10 * sample_rate), dtype=np.float32)
    sf.write(path, np.concatenate((silence, noise, voiced, tail)), sample_rate)


def test_multiband_pitch_v2_exposes_finite_richer_tracks_and_f0(tmp_path: Path) -> None:
    wav = tmp_path / "mixed.wav"
    _mixed_test_wav(wav)
    evidence = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
        feature_backend=FEATURE_BACKEND_ID,
    )
    required = {
        "f0_hz",
        "hnr",
        "voicing",
        "spectral_entropy",
        "spectral_shape_delta",
        "low_band_ratio",
        "mid_band_ratio",
        "breath_likelihood",
        "closure_likelihood",
        "burst_likelihood",
        "nucleus_likelihood",
        "sonorant_boundary_likelihood",
    }
    assert required <= set(evidence.tracks)
    assert all(np.all(np.isfinite(evidence.tracks[name])) for name in evidence.tracks)
    voiced = (evidence.frame_times_ms >= 400.0) & (evidence.frame_times_ms <= 900.0)
    noise = (evidence.frame_times_ms >= 170.0) & (evidence.frame_times_ms <= 320.0)
    detected_f0 = evidence.tracks["f0_hz"][voiced]
    detected_f0 = detected_f0[detected_f0 > 0.0]
    assert len(detected_f0) >= 50
    assert 205.0 <= float(np.median(detected_f0)) <= 235.0
    assert float(np.mean(evidence.tracks["periodicity"][voiced])) > float(
        np.mean(evidence.tracks["periodicity"][noise])
    )
    assert evidence.metadata["feature_backend"] == FEATURE_BACKEND_ID
    assert evidence.metadata["pitch_method"] == "fft_autocorrelation_parabolic_v1"


def test_legacy_feature_backend_remains_explicitly_replayable(tmp_path: Path) -> None:
    wav = tmp_path / "legacy.wav"
    _mixed_test_wav(wav)
    evidence = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
        feature_backend=LEGACY_FEATURE_BACKEND_ID,
    )
    assert evidence.metadata["feature_backend"] == LEGACY_FEATURE_BACKEND_ID
    assert "f0_hz" not in evidence.tracks
    assert evidence.feature_config_sha256


def test_multiband_pitch_v3_exposes_confidence_gated_pitch_and_local_contrast(tmp_path: Path) -> None:
    wav = tmp_path / "v3_mixed.wav"
    _mixed_test_wav(wav)
    evidence = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
        feature_backend=MULTIBAND_V3_FEATURE_BACKEND_ID,
    )
    required = {
        "pitch_confidence",
        "pitch_confidence_rise_local",
        "energy_rise_local",
        "periodicity_rise_local",
        "voicing_rise_local",
        "spectral_change_local",
        "vowel_onset_likelihood",
        "boundary_local_contrast",
        "burst_local_contrast",
    }
    assert required <= set(evidence.tracks)
    assert all(np.all(np.isfinite(evidence.tracks[name])) for name in required)
    voiced = (evidence.frame_times_ms >= 400.0) & (evidence.frame_times_ms <= 900.0)
    noise = (evidence.frame_times_ms >= 170.0) & (evidence.frame_times_ms <= 320.0)
    assert float(np.mean(evidence.tracks["pitch_confidence"][voiced])) > float(
        np.mean(evidence.tracks["pitch_confidence"][noise])
    )
    assert evidence.metadata["feature_backend"] == MULTIBAND_V3_FEATURE_BACKEND_ID
    assert evidence.metadata["pitch_confidence_policy"] == "harmonicity_energy_voicing_v1"
    assert evidence.metadata["local_contrast_radius_frames"] == 4


def test_evidence_npz_roundtrip_preserves_hash_timebase_and_tracks(tmp_path: Path) -> None:
    wav = tmp_path / "roundtrip.wav"
    _mixed_test_wav(wav)
    evidence = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
    )
    artifact = write_acoustic_evidence_npz(evidence, tmp_path / "evidence.npz")
    loaded = load_acoustic_evidence_npz(artifact, expected_wav_sha256=evidence.wav_sha256)
    assert loaded.feature_config_sha256 == evidence.feature_config_sha256
    assert loaded.metadata == evidence.metadata
    assert set(loaded.tracks) == set(evidence.tracks)
    assert np.array_equal(loaded.frame_times_ms, evidence.frame_times_ms)
    assert all(np.array_equal(loaded.tracks[name], evidence.tracks[name]) for name in evidence.tracks)


def test_sequence_aware_activity_spans_quiet_early_and_loud_late_regions(tmp_path: Path) -> None:
    sample_rate = 44100
    silence = np.zeros(int(0.30 * sample_rate), dtype=np.float32)
    quiet_time = np.arange(int(0.65 * sample_rate), dtype=np.float64) / sample_rate
    loud_time = np.arange(int(0.65 * sample_rate), dtype=np.float64) / sample_rate
    quiet = (0.025 * np.sin(2.0 * np.pi * 190.0 * quiet_time)).astype(np.float32)
    loud = (0.25 * np.sin(2.0 * np.pi * 230.0 * loud_time)).astype(np.float32)
    gap = np.zeros(int(0.30 * sample_rate), dtype=np.float32)
    wav = tmp_path / "quiet_then_loud.wav"
    sf.write(wav, np.concatenate((silence, quiet, gap, loud, silence)), sample_rate)

    single_region = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
    )
    sequence = extract_acoustic_evidence(
        wav,
        internal_sample_rate=44100,
        convert_to_mono=True,
        resampler="soxr_hq",
        expected_activity_regions=8,
    )

    assert sequence.active_start_ms < single_region.active_start_ms
    assert 200.0 <= sequence.active_start_ms <= 420.0
    assert sequence.active_end_ms >= 1850.0
    assert sequence.metadata["expected_activity_regions"] == 8
    assert sequence.metadata["activity_noise_margin_db"] == 6.0


def test_dense_speech_activity_floor_keeps_the_softer_first_syllables(tmp_path: Path) -> None:
    # Sung almost end to end: the softer first part makes up more than 15% of the file, so the
    # 15th-percentile "noise" level is speech and the default threshold drops that part.
    sample_rate = 44100
    edge = np.zeros(int(0.10 * sample_rate), dtype=np.float32)
    soft_time = np.arange(int(0.50 * sample_rate), dtype=np.float64) / sample_rate
    loud_time = np.arange(int(1.70 * sample_rate), dtype=np.float64) / sample_rate
    soft = (0.12 * np.sin(2.0 * np.pi * 220.0 * soft_time)).astype(np.float32)
    loud = (0.25 * np.sin(2.0 * np.pi * 220.0 * loud_time)).astype(np.float32)
    wav = tmp_path / "dense.wav"
    sf.write(wav, np.concatenate((edge, soft, loud, edge)), sample_rate)

    def evidence(dense: bool):
        return extract_acoustic_evidence(wav, internal_sample_rate=44100, convert_to_mono=True, resampler="soxr_hq",
                                         expected_activity_regions=14, dense_speech_activity_floor=dense)

    default, dense = evidence(False), evidence(True)
    assert default.active_start_ms > 400.0
    assert dense.active_start_ms < 150.0
    assert dense.metadata["dense_speech_floor_db"] < dense.metadata["noise_db"] + 1e-9
    assert dense.feature_config_sha256 != default.feature_config_sha256
