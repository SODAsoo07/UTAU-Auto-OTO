"""Plosive closure onset detection via burst-backtrack.

When the HSMM decoder reports a phone_change event for a plosive consonant,
it typically lands on the burst (energy spike). The musically correct boundary
for OTO preutterance is the *closure onset* — the point where energy drops
before the burst. This module backtracks from the burst to find that drop.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


PLOSIVE_PHONES: frozenset[str] = frozenset({
    "k", "g", "t", "d", "b", "p", "q", "c",
    "kk", "tt", "pp", "dd", "gg", "bb",
    "ky", "gy", "ty", "dy", "by", "py",
    "kw", "gw", "tw",
})

_DEFAULT_LOOKBACK_MS = 60.0
_MIN_CLOSURE_MS = 8.0
_SILENCE_RISE_THRESHOLD = 0.25
_ENERGY_DROP_RATIO = 0.40


def is_plosive_phone(phone: str) -> bool:
    return str(phone or "").strip().lower() in PLOSIVE_PHONES


def find_closure_onset_ms(
    burst_ms: float,
    times_ms: Sequence[float],
    silence_probs: Sequence[float],
    *,
    lookback_ms: float = _DEFAULT_LOOKBACK_MS,
    min_closure_ms: float = _MIN_CLOSURE_MS,
) -> float | None:
    """Find closure onset by backtracking from burst position.

    Scans backward from ``burst_ms`` looking for the frame where silence
    probability rises above a threshold (indicating the closure gap before the
    burst). Returns the time in ms of the closure onset, or None if no clear
    closure is detected.

    Parameters
    ----------
    burst_ms : float
        Time of the detected burst / phone_change event.
    times_ms : sequence of float
        Frame time axis (ascending, in ms).
    silence_probs : sequence of float
        Per-frame silence probability (0..1), aligned with times_ms.
    lookback_ms : float
        Maximum distance to search backward from burst.
    min_closure_ms : float
        Minimum gap between closure onset and burst to accept.

    Returns
    -------
    float or None
        Closure onset time in ms, or None if not found.
    """
    times = np.asarray(times_ms, dtype=np.float64)
    silence = np.asarray(silence_probs, dtype=np.float32)
    n = int(times.size)
    if n == 0 or silence.size != n:
        return None

    burst_idx = int(np.searchsorted(times, burst_ms, side="right")) - 1
    burst_idx = max(0, min(n - 1, burst_idx))

    search_start_ms = burst_ms - lookback_ms
    start_idx = int(np.searchsorted(times, search_start_ms, side="left"))
    start_idx = max(0, min(burst_idx, start_idx))

    if burst_idx - start_idx < 2:
        return None

    # Strategy: scan backward from burst. First find the closure region (high
    # silence), then continue backward to find where silence drops below
    # threshold — that transition is the closure onset.
    closure_end_frame = None
    for i in range(burst_idx - 1, start_idx - 1, -1):
        if float(silence[i]) >= _SILENCE_RISE_THRESHOLD:
            closure_end_frame = i
            break

    closure_frame = None
    if closure_end_frame is not None:
        # Walk further back to find where silence drops (= closure start)
        for i in range(closure_end_frame, start_idx - 1, -1):
            if float(silence[i]) < _SILENCE_RISE_THRESHOLD:
                closure_frame = i + 1
                break
        if closure_frame is None:
            closure_frame = start_idx

    if closure_frame is None:
        # Fallback: find steepest energy drop
        energy_proxy = 1.0 - silence[start_idx:burst_idx]
        if energy_proxy.size >= 2:
            gradient = np.diff(energy_proxy)
            min_grad_idx = int(np.argmin(gradient))
            if gradient[min_grad_idx] < -0.08:
                closure_frame = start_idx + min_grad_idx

    if closure_frame is None:
        return None

    closure_ms = float(times[closure_frame])
    if burst_ms - closure_ms < min_closure_ms:
        return None

    return closure_ms


def refine_consonant_onset_for_plosive(
    phone: str,
    burst_ms: float,
    times_ms: Sequence[float],
    silence_probs: Sequence[float],
    *,
    lookback_ms: float = _DEFAULT_LOOKBACK_MS,
) -> float:
    """Return refined consonant onset: closure onset if plosive, else burst_ms unchanged."""
    if not is_plosive_phone(phone):
        return burst_ms
    closure = find_closure_onset_ms(
        burst_ms, times_ms, silence_probs,
        lookback_ms=lookback_ms,
    )
    return closure if closure is not None else burst_ms
