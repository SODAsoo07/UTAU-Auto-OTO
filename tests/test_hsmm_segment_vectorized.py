"""The vectorized segment search must equal the original per-duration loop bit for bit."""
from types import SimpleNamespace

import numpy as np
import pytest

from universal_auto_oto import hsmm_decoder as h


@pytest.mark.parametrize("block", [7, 1 << 10, 1 << 21])
def test_best_segment_ends_matches_reference(monkeypatch, block):
    monkeypatch.setattr(h, "SEGMENT_BLOCK_ELEMENTS", block)
    rng = np.random.default_rng(block)
    for trial in range(400):
        size = int(rng.integers(2, 300))
        frames = size - 1
        low = int(rng.integers(1, min(20, frames) + 1))
        high = max(low, min(frames, low + int(rng.integers(0, 300))))
        kind = trial % 3
        scores = (rng.integers(-2, 2, size=frames).astype(float) if kind == 1 else rng.normal(size=frames) * (1e6 if kind == 2 else 1))
        prefix = np.concatenate(([0.0], np.cumsum(scores)))
        best = np.round(rng.normal(size=size)) if kind == 1 else rng.normal(size=size) * 10
        best[rng.random(size) < 0.3] = -np.inf
        if trial % 40 == 0:
            best[:] = -np.inf
        transition = np.zeros(size) if kind == 1 else rng.normal(size=size)
        durations = np.zeros(high + 1) if kind == 1 else rng.normal(size=high + 1)
        got = h._best_segment_ends(best, prefix, transition, durations, low, high)
        want = h._best_segment_ends_reference(best, prefix, transition, list(durations), low, high)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1])


def test_duration_scores_match_scalar_definition():
    rng = np.random.default_rng(3)
    for _ in range(300):
        spec = SimpleNamespace(mode_duration_ms=float(rng.choice([0.0, rng.uniform(5, 800)])),
                               duration_sigma_ms=float(rng.choice([0.0, rng.uniform(0.2, 300)])),
                               over_duration_penalty_scale=float(rng.choice([1.0, 0.0, rng.uniform(0, 3)])))
        shift, frames = float(rng.choice([5.0, 4.988662131519274, rng.uniform(1, 20)])), int(rng.integers(0, 500))
        want = np.array([h._duration_score(spec, float(d) * shift) for d in range(frames + 1)])
        assert np.array_equal(h._duration_scores(spec, shift, frames), want)
