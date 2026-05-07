"""Smoke + invariant tests for the confidence calibration helpers (0-order)."""
from __future__ import annotations

import math


def test_apply_temperature_identity_when_T_one():
    from core.coarse_crnn.oto_confidence_calibration import apply_temperature

    for p in (0.05, 0.30, 0.50, 0.70, 0.95):
        assert abs(apply_temperature(p, 1.0) - p) < 1e-5


def test_apply_temperature_high_T_pulls_toward_half():
    """T > 1 must move probabilities toward 0.5 (flatten)."""
    from core.coarse_crnn.oto_confidence_calibration import apply_temperature

    # 0.95 should drop; 0.05 should rise.
    assert apply_temperature(0.95, 5.0) < 0.95
    assert apply_temperature(0.05, 5.0) > 0.05
    assert apply_temperature(0.95, 5.0) > 0.5
    assert apply_temperature(0.05, 5.0) < 0.5


def test_apply_temperature_low_T_sharpens():
    """T < 1 sharpens away from 0.5."""
    from core.coarse_crnn.oto_confidence_calibration import apply_temperature

    assert apply_temperature(0.7, 0.5) > 0.7
    assert apply_temperature(0.3, 0.5) < 0.3


def test_apply_temperature_clipping_safe():
    """Extreme inputs must not produce NaN/inf."""
    from core.coarse_crnn.oto_confidence_calibration import apply_temperature

    for p in (0.0, 1.0, 1e-12, 1.0 - 1e-12):
        out = apply_temperature(p, 2.0)
        assert math.isfinite(out)
        assert 0.0 < out < 1.0


def test_fit_temperature_picks_higher_T_for_collapsed_distribution():
    """When good and bad rows have nearly identical confidences, the fit
    should NOT pick T < 1 (which would sharpen the same collapsed prob).
    Instead T should hover near 1 or higher because no temperature can
    separate them, and the BCE landscape is flatter at higher T."""
    from core.coarse_crnn.oto_confidence_calibration import fit_temperature_scaling

    probs = [0.37] * 20 + [0.36] * 20  # near-collapsed
    is_correct = [True] * 20 + [False] * 20
    best_t, _ = fit_temperature_scaling(probs, is_correct)
    assert best_t >= 1.0


def test_fit_temperature_scales_separable_distribution():
    """When good and bad rows ARE separable, fit should not pick a wildly
    sharpening T. Sanity: BCE is finite and best_t is in (0, 20]."""
    from core.coarse_crnn.oto_confidence_calibration import fit_temperature_scaling

    probs = [0.85] * 50 + [0.20] * 50
    is_correct = [True] * 50 + [False] * 50
    best_t, nll = fit_temperature_scaling(probs, is_correct)
    assert 0.0 < best_t <= 20.0
    assert math.isfinite(nll)


def test_fit_temperature_handles_empty_input():
    from core.coarse_crnn.oto_confidence_calibration import fit_temperature_scaling

    best_t, nll = fit_temperature_scaling([], [])
    assert best_t == 1.0
    assert math.isnan(nll)


def test_fit_isotonic_lookup_is_monotone_non_decreasing():
    from core.coarse_crnn.oto_confidence_calibration import fit_isotonic_lookup

    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10
    is_correct = [False, False, False, True, True, False, True, True, True] * 10
    iso = fit_isotonic_lookup(probs, is_correct, n_bins=10, min_bin_count=2)
    table = iso.table
    assert len(table) == 10
    for prev, cur in zip(table, table[1:]):
        assert cur >= prev - 1e-9


def test_fit_isotonic_lookup_handles_no_data_gracefully():
    from core.coarse_crnn.oto_confidence_calibration import fit_isotonic_lookup

    iso = fit_isotonic_lookup([], [], n_bins=8)
    assert len(iso.table) == 8
    for prev, cur in zip(iso.table, iso.table[1:]):
        assert cur >= prev - 1e-9


def test_isotonic_lookup_apply_clips_to_unit_interval():
    from core.coarse_crnn.oto_confidence_calibration import IsotonicLookup

    iso = IsotonicLookup(n_bins=4, table=(0.05, 0.30, 0.60, 0.95))
    assert iso.apply(-1.0) == 0.05
    assert iso.apply(2.0) == 0.95
    assert iso.apply(0.0) == 0.05
    assert iso.apply(1.0) == 0.95


def test_compute_low_confidence_threshold_meets_target_rate():
    from core.coarse_crnn.oto_confidence_calibration import (
        compute_low_confidence_threshold,
        fallback_rate,
    )

    probs = [i / 100.0 for i in range(100)]  # 0.00 .. 0.99
    threshold = compute_low_confidence_threshold(probs, target_low_rate=0.30)
    rate = fallback_rate(probs, threshold)
    # Threshold should pick a cut close to the 30th percentile, so the
    # actual fallback rate should be at most target + a small bin margin.
    assert rate <= 0.31


def test_compute_low_confidence_threshold_handles_empty():
    from core.coarse_crnn.oto_confidence_calibration import compute_low_confidence_threshold

    # Falls back to legacy default when no data.
    assert abs(compute_low_confidence_threshold([], target_low_rate=0.5) - 0.58) < 1e-6


def test_good_bad_mean_separates_correct_from_incorrect():
    from core.coarse_crnn.oto_confidence_calibration import good_bad_mean

    probs = [0.9, 0.85, 0.80, 0.10, 0.20, 0.15]
    is_correct = [True, True, True, False, False, False]
    good, bad = good_bad_mean(probs, is_correct)
    assert good > bad
    assert abs(good - 0.85) < 1e-6
    assert abs(bad - 0.15) < 1e-6


def test_good_bad_mean_empty_groups_return_nan():
    from core.coarse_crnn.oto_confidence_calibration import good_bad_mean

    good, bad = good_bad_mean([], [])
    assert math.isnan(good)
    assert math.isnan(bad)
    good_only, bad_only = good_bad_mean([0.7, 0.8], [True, True])
    assert math.isfinite(good_only)
    assert math.isnan(bad_only)
