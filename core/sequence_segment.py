from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Phoneme → SIL/C/V classification tables
# ---------------------------------------------------------------------------

_SILENCE_MARKS: frozenset[str] = frozenset({
    "", "sil", "pau", "sp", "spn", "br", "bre", "breath",
})

_KR_VOWELS: frozenset[str] = frozenset({
    "a", "ae", "e", "eo", "eu", "i", "o", "oe", "u",
    "wa", "wae", "we", "wi", "wo", "ui",
    "ya", "yae", "ye", "yeo", "yo", "yu",
    "oo",
})

_JA_VOWELS: frozenset[str] = frozenset({
    "a", "i", "u", "e", "o",
    "A", "I", "U", "E", "O",
})

# Column indices in the log-emission matrix
_COL_C = 0    # regular (voiceless/mixed) consonant
_COL_V = 1    # vowel
_COL_SIL = 2  # silence
_COL_VC = 3   # voiced consonant: blended 0.35·C + 0.65·V  (acoustically vowel-like)

_CVN_COL: dict[str, int] = {"C": _COL_C, "V": _COL_V, "SIL": _COL_SIL}

# Nasals and liquids: acoustically vowel-like but phonologically consonantal.
# OTO onset is treated as C, but pyin sees these frames as fully voiced →
# using a V-biased column avoids misplacing the C segment into the following vowel.
_VOICED_CONSONANT_MARKS: frozenset[str] = frozenset({
    "n", "m", "ng", "l", "r",  # Korean/Japanese nasals and liquid
    "N",                         # Japanese moraic nasal
})


def cvn_type_of_mark(mark: str) -> str:
    m = str(mark or "").strip().lower()
    if m in _SILENCE_MARKS:
        return "SIL"
    if m in _KR_VOWELS or m in _JA_VOWELS or mark in _JA_VOWELS:
        return "V"
    return "C"


def _emission_col_for_mark(mark: str) -> int:
    """Return the log-emission column index to use for this phoneme mark."""
    m = str(mark or "").strip().lower()
    if m in _SILENCE_MARKS:
        return _COL_SIL
    if m in _KR_VOWELS or m in _JA_VOWELS or mark in _JA_VOWELS:
        return _COL_V
    if m in _VOICED_CONSONANT_MARKS or mark in _VOICED_CONSONANT_MARKS:
        return _COL_VC
    return _COL_C


# ---------------------------------------------------------------------------
# Minimal interval duck-type compatible with NormalizedIntervalTier entries
# ---------------------------------------------------------------------------

@dataclass
class SegmentInterval:
    """Minimal interval with mark / minTime / maxTime in seconds."""
    mark: str
    minTime: float
    maxTime: float


# ---------------------------------------------------------------------------
# Acoustic feature → 3-class (C/V/SIL) log-emission matrix
# ---------------------------------------------------------------------------

def _sil_threshold() -> float:
    raw = str(os.environ.get("UTOA_SEQ_SEG_SIL_THRESHOLD", "0.08") or "0.08").strip()
    try:
        return float(np.clip(float(raw), 0.01, 0.50))
    except Exception:
        return 0.08


def _build_emission(features) -> np.ndarray:
    """
    Returns (T, 4) log-probs per frame:
      col 0 (C)  — regular consonant
      col 1 (V)  — vowel
      col 2 (SIL) — silence
      col 3 (VC) — voiced consonant: 0.35·p_C + 0.65·p_V  (nasal/liquid blend)
    """
    from core.cvn_classifier import predict_cvn

    posterior = predict_cvn(features, smooth_window=3, min_run_frames=1)
    p_cv = posterior.probs.astype(np.float64)  # (T, 2): [P(C), P(V)]

    rms = np.asarray(features.rms, dtype=np.float64)
    rms_ref = float(np.percentile(rms, 90)) if rms.size else 1.0
    if rms_ref < 1e-8:
        rms_ref = 1e-8

    rms_norm = np.clip(rms / rms_ref, 0.0, 2.0)
    sil_th = _sil_threshold()
    p_sil = np.clip(1.0 / (1.0 + np.exp(30.0 * (rms_norm - sil_th))), 0.0, 0.97)

    p_c = p_cv[:, 0] * (1.0 - p_sil)
    p_v = p_cv[:, 1] * (1.0 - p_sil)
    p_vc = 0.35 * p_c + 0.65 * p_v  # voiced-consonant blend

    eps = 1e-9
    return np.stack([
        np.log(np.maximum(p_c, eps)),
        np.log(np.maximum(p_v, eps)),
        np.log(np.maximum(p_sil, eps)),
        np.log(np.maximum(p_vc, eps)),
    ], axis=1)


# ---------------------------------------------------------------------------
# Monotone Viterbi segmentation
# ---------------------------------------------------------------------------

def _viterbi_monotone(log_emission: np.ndarray, state_cols: list[int]) -> list[int]:
    """
    Monotone Viterbi: assigns each frame to exactly one state from state_cols,
    states must be used in order (no skip, no revert), each state ≥ 1 frame.

    state_cols[k] is the column index in log_emission to use for state k.
    Returns the start frame index for each state.
    """
    T = int(log_emission.shape[0])
    K = len(state_cols)

    if K == 0:
        return []
    if T == 0:
        return [0] * K
    if K == 1:
        return [0]
    if K > T:
        return [min(k, T - 1) for k in range(K)]

    NEG_INF = -1e30
    dp = np.full((T, K), NEG_INF, dtype=np.float64)
    bp = np.full((T, K), -1, dtype=np.int32)

    dp[0, 0] = float(log_emission[0, state_cols[0]])
    bp[0, 0] = 0

    for t in range(1, T):
        # state k can first appear at frame k (each prior state needs ≥ 1 frame)
        k_max = min(K - 1, t)
        for k in range(k_max + 1):
            emit = float(log_emission[t, state_cols[k]])
            best_val = NEG_INF
            best_prev = k

            # Stay in k
            if dp[t - 1, k] > NEG_INF:
                val = dp[t - 1, k] + emit
                if val > best_val:
                    best_val = val
                    best_prev = k

            # Advance from k-1 (only if enough frames remain for states k+1..K-1)
            if k > 0 and dp[t - 1, k - 1] > NEG_INF:
                if (T - t - 1) >= (K - k - 1):  # remaining frames ≥ remaining states
                    val = dp[t - 1, k - 1] + emit
                    if val > best_val:
                        best_val = val
                        best_prev = k - 1

            if best_val > NEG_INF:
                dp[t, k] = best_val
                bp[t, k] = best_prev

    # Backtrack from (T-1, K-1)
    path = [0] * T
    path[T - 1] = K - 1
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(bp[t, path[t]])

    # First frame of each state
    starts = [0] * K
    first_seen = [False] * K
    for t in range(T):
        k = int(path[t])
        if not first_seen[k]:
            starts[k] = t
            first_seen[k] = True

    # Fallback for any state not visited (shouldn't happen when K ≤ T)
    for k in range(K):
        if not first_seen[k]:
            starts[k] = starts[k - 1] + 1 if k > 0 else 0

    return starts


# ---------------------------------------------------------------------------
# Frame index → seconds conversion
# ---------------------------------------------------------------------------

def _frame_boundary_to_seconds(
    frame_idx: int,
    times_ms: np.ndarray,
    hop_ms: float,
) -> float:
    """Boundary time (in seconds) between frames frame_idx-1 and frame_idx."""
    if times_ms.size == 0:
        return 0.0
    if frame_idx <= 0:
        return max(0.0, float(times_ms[0]) - hop_ms / 2.0) / 1000.0
    if frame_idx >= len(times_ms):
        return (float(times_ms[-1]) + hop_ms / 2.0) / 1000.0
    return float(times_ms[frame_idx - 1] + times_ms[frame_idx]) / 2.0 / 1000.0


# ---------------------------------------------------------------------------
# High-trust boundary clamping
# ---------------------------------------------------------------------------

# For high-trust alignments, Viterbi corrections are clamped to this range
# around the original MFA boundary to prevent overcorrection.
_HIGH_TRUST_MAX_SHIFT_S: float = 0.025  # 25 ms


def _clamp_to_original(
    corrected: list[SegmentInterval],
    original: Sequence,
    max_shift_s: float,
) -> list[SegmentInterval]:
    """
    Limit each Viterbi boundary movement to ±max_shift_s relative to the
    original MFA boundary.  Boundaries are shared between adjacent intervals,
    so clamping is done on boundary points (not on interval endpoints
    independently) to keep adjacent intervals consistent.
    """
    n = len(corrected)
    # boundary[i] = start of interval[i]; boundary[n] = end of last interval
    orig_b = [float(getattr(iv, "minTime", 0.0)) for iv in original]
    orig_b.append(float(getattr(original[-1], "maxTime", 0.0)))
    vit_b = [seg.minTime for seg in corrected]
    vit_b.append(corrected[-1].maxTime)

    clamped_b = [
        ob + max(-max_shift_s, min(max_shift_s, vb - ob))
        for ob, vb in zip(orig_b, vit_b)
    ]

    result: list[SegmentInterval] = []
    for i, seg in enumerate(corrected):
        s, e = clamped_b[i], clamped_b[i + 1]
        if e <= s:
            # Preserve original duration if clamping collapses the interval
            orig_dur = max(0.001, float(getattr(original[i], "maxTime", 0.0))
                          - float(getattr(original[i], "minTime", 0.0)))
            e = s + orig_dur
        result.append(SegmentInterval(mark=seg.mark, minTime=s, maxTime=e))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_phone_intervals_by_viterbi(
    phone_intervals: Sequence,
    wav_path: str,
    *,
    trust_tier: str = "mid",
    prefer_sequence: bool = False,
) -> list:
    """
    Correct phone interval boundaries using acoustic Viterbi segmentation.

    The known phoneme sequence (from filename) is treated as ground truth.
    Frame-level SIL/C/V posteriors are derived from audio features and used
    to find optimal time boundaries via monotone Viterbi DP.

    Activation rules:
    - "low" / "mid" trust, or prefer_sequence=True: full replacement
    - "high" trust: apply with ±25 ms cap per boundary (gentle fine-tuning)
    - Disabled entirely by UTOA_SEQ_SEGMENT_ENABLE=0
    - Forced uncapped (even for high trust) by UTOA_SEQ_SEGMENT_FORCE=1

    Returns original intervals unchanged on any error or when skipped.
    """
    env_enable = str(os.environ.get("UTOA_SEQ_SEGMENT_ENABLE", "1") or "1").strip().lower()
    if env_enable in {"0", "false", "no", "off"}:
        return list(phone_intervals)

    env_force = str(os.environ.get("UTOA_SEQ_SEGMENT_FORCE", "0") or "0").strip().lower()
    force = env_force in {"1", "true", "yes", "on"}

    tier = str(trust_tier or "").strip().lower()
    should_apply_full = force or prefer_sequence or tier in {"mid", "low"}
    should_apply_capped = not should_apply_full and tier == "high"

    if not should_apply_full and not should_apply_capped:
        return list(phone_intervals)

    if not wav_path or not phone_intervals:
        return list(phone_intervals)

    try:
        from core.cvn_feature_extractor import extract_frame_features
        features = extract_frame_features(wav_path, hop_ms=10.0, frame_ms=25.0)
    except Exception:
        return list(phone_intervals)

    # Need at least K + 2 frames (K phones + leading/trailing SIL)
    if features.num_frames < len(phone_intervals) + 2:
        return list(phone_intervals)

    marks = [str(getattr(iv, "mark", "") or "") for iv in phone_intervals]

    # Wrap with SIL buffers to absorb leading/trailing audio silence.
    # Compute per-state emission column so voiced consonants get their V-biased blend.
    full_marks = ["[sil]"] + marks + ["[sil]"]
    state_cols = [_emission_col_for_mark(m) for m in full_marks]

    try:
        log_em = _build_emission(features)
        starts = _viterbi_monotone(log_em, state_cols)
    except Exception:
        return list(phone_intervals)

    if len(starts) != len(full_marks):
        return list(phone_intervals)

    times_ms = features.times_ms
    hop_ms = float(features.hop_ms)

    corrected: list[SegmentInterval] = []
    for i, mark in enumerate(marks):
        seg_idx = i + 1  # offset past leading SIL
        start_sec = _frame_boundary_to_seconds(starts[seg_idx], times_ms, hop_ms)
        next_start = starts[seg_idx + 1] if seg_idx + 1 < len(starts) else features.num_frames
        end_sec = _frame_boundary_to_seconds(next_start, times_ms, hop_ms)
        if end_sec <= start_sec:
            end_sec = start_sec + hop_ms / 1000.0
        corrected.append(SegmentInterval(mark=mark, minTime=start_sec, maxTime=end_sec))

    if should_apply_capped:
        corrected = _clamp_to_original(corrected, phone_intervals, _HIGH_TRUST_MAX_SHIFT_S)

    return corrected


__all__ = [
    "SegmentInterval",
    "cvn_type_of_mark",
    "correct_phone_intervals_by_viterbi",
]
