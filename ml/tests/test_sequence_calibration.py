import numpy as np

from core.alignment.sequence_calibration import (
    VoicebankCalibration,
    WavCalibSample,
    aggregate_calibration,
    reclist_duration_bias,
    resolve_effective_silence_threshold,
)


def test_aggregate_calibration_estimates_reclist_and_alias_duration():
    samples = [
        WavCalibSample(1000.0, 0.01, 0.02, 0.09, 0.12, 0.20, 40.0, 80.0, 2, ("ka", "a"), "ka a"),
        WavCalibSample(1100.0, 0.02, 0.03, 0.10, 0.13, 0.22, 45.0, 90.0, 2, ("ki", "i"), "ki i"),
        WavCalibSample(1200.0, 0.01, 0.02, 0.11, 0.14, 0.23, 50.0, 95.0, 2, ("ku", "u"), "ku u"),
    ]

    calibration = aggregate_calibration(samples)

    assert calibration.sample_count == 3
    assert calibration.reclist_kind == "cvvc"
    assert calibration.alias_duration_median_ms["ka a"] == 1000.0
    assert calibration.leading_silence_ms_p50 == 45.0


def test_resolve_effective_silence_threshold_keeps_active_ratio_guard():
    rms = np.asarray([0.01] * 80 + [0.20] * 20, dtype=np.float32)
    calibration = VoicebankCalibration(
        sample_count=8,
        silence_floor_rms=0.005,
        voiced_floor_rms=0.20,
    )

    threshold = resolve_effective_silence_threshold(
        0.12,
        calibration=calibration,
        floor_weight=1.0,
        values=rms,
        min_active_ratio=0.05,
        max_active_ratio=0.30,
    )

    assert float(np.mean(rms >= threshold)) <= 0.30


def test_reclist_duration_bias_uses_repeated_alias_median():
    calibration = VoicebankCalibration(
        sample_count=3,
        alias_duration_ms={"ka a": [900.0, 1100.0, 1000.0]},
    )

    bias = reclist_duration_bias(
        calibration=calibration,
        alias_key="ka a",
        nominal_total_frames=80,
        hop_sec=0.01,
    )

    assert 1.20 <= bias <= 1.30
