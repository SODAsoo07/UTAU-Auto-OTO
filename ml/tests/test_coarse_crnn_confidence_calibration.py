from __future__ import annotations


def test_apply_calibration_disabled_passthrough():
    from core.coarse_crnn.oto_confidence_calibration import ConfidenceCalibration, apply_calibration

    c = ConfidenceCalibration(enabled=False)
    assert apply_calibration(0.25, c) == 0.25


def test_apply_calibration_enabled_clipped():
    from core.coarse_crnn.oto_confidence_calibration import ConfidenceCalibration, apply_calibration

    c = ConfidenceCalibration(enabled=True, scale=2.0, bias=-0.5, min_clip=0.1, max_clip=0.9)
    out = apply_calibration(0.8, c)
    assert 0.1 <= out <= 0.9
