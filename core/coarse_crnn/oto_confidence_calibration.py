"""Confidence calibration post-processing for the CRNN OTO predictor.

Implements the 0-order item from [[CRNN-정확도-개선-방안-v2]]:
"fallback saturation 해소". When the trained model's confidence head ends up
collapsed (good_mean ≈ bad_mean ≈ 0.35), simple affine `scale*p + bias`
calibration as currently implemented in OtoCrnnConfig cannot pull the
distribution apart. This module provides two stronger tools:

1. **Temperature scaling**: invert sigmoid to logits, divide by T, re-apply
   sigmoid. Fitted on a held-out set by minimising binary cross-entropy
   against a per-row "is_correct" label (e.g. preutterance error <= 50 ms).
2. **Isotonic lookup table**: a sklearn-free monotonic mapping with K bins.
   Useful when the relationship between raw confidence and true accuracy is
   non-monotonic in the affine sense.

Neither method requires retraining the model. Both consume the per-row
`confidence` + `param_abs_errors_ms.preutterance` columns from
``ml/scripts/coarse_crnn/evaluate_oto.py`` output JSON.

The threshold helper (`compute_low_confidence_threshold`) picks a cutoff
that satisfies a target low-confidence rate, e.g. "fallback rate <= 0.85".

Design notes:
- All input/output probabilities live in [0, 1]. Internal temperature work
  uses logits with safe clipping to keep `log` finite.
- No sklearn dependency. Isotonic lookup is implemented as a piecewise
  constant table keyed by the raw probability bin index.
- All functions are pure: no global state, no file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import math


_EPS = 1e-6


def _clip01(value: float) -> float:
    return max(_EPS, min(1.0 - _EPS, float(value)))


def _logit(p: float) -> float:
    p = _clip01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def apply_temperature(prob: float, temperature: float) -> float:
    """Re-scale a probability through ``sigmoid(logit(p) / T)``.

    T > 1 flattens the distribution toward 0.5. T < 1 sharpens it. T=1
    returns the input unchanged.
    """
    t = max(_EPS, float(temperature))
    if abs(t - 1.0) <= _EPS:
        return _clip01(prob)
    return _sigmoid(_logit(prob) / t)


def apply_temperature_batch(probs: Iterable[float], temperature: float) -> list[float]:
    return [apply_temperature(p, temperature) for p in probs]


def fit_temperature_scaling(
    probs: Iterable[float],
    is_correct: Iterable[bool],
    *,
    candidates: Iterable[float] = (
        0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0,
        2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 10.0, 12.5, 15.0,
    ),
) -> tuple[float, float]:
    """Pick the temperature T from a fixed candidate grid that minimises
    binary cross-entropy of `sigmoid(logit(p) / T)` against `is_correct`.

    Returns ``(best_T, best_nll_per_row)``.

    Why a grid (not LBFGS): the dependency-free design avoids torch here so
    the calibration script can run without GPU/torch import overhead. The
    grid is wide and log-spaced so a 0.25-step scan is enough for the
    convexity that BCE-on-sigmoid normally exhibits over T.
    """
    pairs: list[tuple[float, bool]] = []
    for prob, label in zip(probs, is_correct):
        try:
            p = float(prob)
            y = bool(label)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        pairs.append((p, y))
    if not pairs:
        return 1.0, float("nan")

    def _nll_for(t: float) -> float:
        if t <= 0.0:
            return float("inf")
        total = 0.0
        for p, y in pairs:
            cal = apply_temperature(p, t)
            if y:
                total -= math.log(_clip01(cal))
            else:
                total -= math.log(1.0 - _clip01(cal))
        return total / float(len(pairs))

    best_t = 1.0
    best_nll = _nll_for(1.0)
    for cand in candidates:
        nll = _nll_for(float(cand))
        if nll < best_nll - _EPS:
            best_nll = nll
            best_t = float(cand)
    return best_t, best_nll


# ---------------------------------------------------------------------------
# Isotonic lookup (piecewise constant, no sklearn)
# ---------------------------------------------------------------------------


@dataclass
class IsotonicLookup:
    """Monotonic non-decreasing mapping ``p -> p_calibrated`` stored as
    a piecewise-constant lookup table keyed by uniform bins over [0, 1].
    """

    n_bins: int = 20
    table: tuple[float, ...] = field(default_factory=tuple)

    def apply(self, prob: float) -> float:
        if not self.table:
            return _clip01(prob)
        n = len(self.table)
        idx = int(round(_clip01(prob) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return float(self.table[idx])


def fit_isotonic_lookup(
    probs: Iterable[float],
    is_correct: Iterable[bool],
    *,
    n_bins: int = 20,
    min_bin_count: int = 5,
) -> IsotonicLookup:
    """Fit a monotonic non-decreasing lookup table by:

    1. binning rows by their raw probability into ``n_bins`` uniform bins
    2. computing the empirical accuracy in each bin
    3. enforcing monotone non-decreasing values by left-to-right max sweep

    Bins with fewer than ``min_bin_count`` rows borrow from the nearest
    populated neighbour to avoid noisy spikes from tiny groups.
    """
    bins_correct: list[int] = [0] * n_bins
    bins_total: list[int] = [0] * n_bins
    for prob, label in zip(probs, is_correct):
        try:
            p = float(prob)
            y = bool(label)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        idx = int(round(_clip01(p) * (n_bins - 1)))
        idx = max(0, min(n_bins - 1, idx))
        bins_total[idx] += 1
        if y:
            bins_correct[idx] += 1

    raw: list[float] = []
    for c, t in zip(bins_correct, bins_total):
        if t < min_bin_count:
            raw.append(float("nan"))
        else:
            raw.append(c / t if t > 0 else 0.0)

    # Forward fill NaN bins by nearest populated neighbour, then backward fill.
    last = None
    forward = list(raw)
    for i, value in enumerate(forward):
        if math.isfinite(value):
            last = value
        elif last is not None:
            forward[i] = last
    last = None
    backward = list(forward)
    for i in range(len(backward) - 1, -1, -1):
        if math.isfinite(backward[i]):
            last = backward[i]
        elif last is not None:
            backward[i] = last
    if last is None:
        # No populated bins at all.
        backward = [_clip01(i / max(1, n_bins - 1)) for i in range(n_bins)]
    backward = [_clip01(v) for v in backward]

    # Left-to-right max sweep enforces non-decreasing monotonicity.
    monotone: list[float] = []
    running = backward[0]
    for value in backward:
        if value > running:
            running = value
        monotone.append(running)
    return IsotonicLookup(n_bins=n_bins, table=tuple(monotone))


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def compute_low_confidence_threshold(
    calibrated_probs: Iterable[float],
    *,
    target_low_rate: float,
) -> float:
    """Pick the largest threshold T such that the fraction of rows with
    ``calibrated_prob < T`` is at most ``target_low_rate``.

    Returns the threshold in [0, 1]. If the inputs are all NaN or empty,
    returns 0.58 (the legacy default in OtoCrnnConfig).
    """
    target = max(0.0, min(1.0, float(target_low_rate)))
    values: list[float] = []
    for prob in calibrated_probs:
        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p):
            values.append(_clip01(p))
    if not values:
        return 0.58
    values.sort()
    # Index of the highest value still allowed to fall below the threshold.
    cutoff_idx = int(math.floor(target * len(values)))
    cutoff_idx = max(0, min(len(values) - 1, cutoff_idx))
    return float(values[cutoff_idx])


def fallback_rate(probs: Iterable[float], threshold: float) -> float:
    total = 0
    low = 0
    for prob in probs:
        try:
            p = float(prob)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        total += 1
        if _clip01(p) < float(threshold):
            low += 1
    if total == 0:
        return 0.0
    return float(low) / float(total)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def good_bad_mean(probs: Iterable[float], is_correct: Iterable[bool]) -> tuple[float, float]:
    """Mean confidence of "good" (correct) vs "bad" (incorrect) rows.

    Returned as ``(good_mean, bad_mean)``. If either group is empty, that
    group's mean is NaN. The gap ``good_mean - bad_mean`` is the same
    diagnostic surfaced in 14-1 ("Confidence calibration").
    """
    good_sum = 0.0
    good_n = 0
    bad_sum = 0.0
    bad_n = 0
    for prob, label in zip(probs, is_correct):
        try:
            p = float(prob)
            y = bool(label)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p):
            continue
        p = _clip01(p)
        if y:
            good_sum += p
            good_n += 1
        else:
            bad_sum += p
            bad_n += 1
    good_mean = good_sum / good_n if good_n > 0 else float("nan")
    bad_mean = bad_sum / bad_n if bad_n > 0 else float("nan")
    return good_mean, bad_mean


__all__ = [
    "IsotonicLookup",
    "apply_temperature",
    "apply_temperature_batch",
    "fit_temperature_scaling",
    "fit_isotonic_lookup",
    "compute_low_confidence_threshold",
    "fallback_rate",
    "good_bad_mean",
]
