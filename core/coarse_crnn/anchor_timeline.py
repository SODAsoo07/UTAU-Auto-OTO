"""Hybrid (RMS-valley + onset-density) anchor timeline for multi-syllable wavs.

Phase H of [[14-5-CVVC-체감정확도-개선-로드맵]]. The legacy
`_build_anchor_timeline` in `wav_decoder` linearly partitions the active region
into `slot_count` equal segments and snaps each slot to the nearest onset
within ±180 ms. That fails for multi-syllable wavs (e.g. an 8-mora CVVC sample
like `_jjeuNG'jjeuN'..'jja.wav`): syllable durations vary, so equal partition
puts later slots hundreds-to-thousands of ms before their true position.

This module replaces linear partitioning with a two-signal estimator:

A. **RMS valleys** in the active region give *segment boundaries* — places
   where energy dips between two utterances. Deepest `slot_count − 1` valleys
   split the active region into `slot_count` segments.
B. **Onset-density** within each segment gives the *slot anchor* — the
   strongest model/audio onset candidate that falls inside that segment.

Both signals are independently weak in some voicebanks (CVVC 連続音 has no
valleys; weak-consonant codas have no onsets), so the builder always falls
back to linear segments when a signal is missing, never producing worse
output than the legacy timeline.

Behaviour is gated by `UTOA_BOUNDARY_ANCHOR_TIMELINE`:
- `linear` / unset → legacy partitioning (default for safety).
- `hybrid` → run A+B, fall back to linear segments only inside the hybrid
  path; never drop the row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.coarse_crnn.audio import load_wav_mono
from core.coarse_crnn.boundary_types import BoundaryCandidate


DEFAULT_RMS_FRAME_MS: float = 25.0
DEFAULT_RMS_HOP_MS: float = 10.0
DEFAULT_VALLEY_MIN_DEPTH_RATIO: float = 0.40
DEFAULT_VALLEY_MIN_GAP_MS: float = 80.0
DEFAULT_VALLEY_SMOOTH_FRAMES: int = 3
DEFAULT_ONSET_KINDS: tuple[str, ...] = (
    "syllable_onset",
    "consonant_onset",
    "vowel_start",
)


# ---------------------------------------------------------------------------
# RMS envelope + valley detection (signal A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RmsValley:
    time_ms: float
    rms_value: float
    depth: float  # how far below the nearest peaks the valley sits


def compute_rms_envelope(
    samples: np.ndarray,
    sr: int,
    *,
    frame_ms: float = DEFAULT_RMS_FRAME_MS,
    hop_ms: float = DEFAULT_RMS_HOP_MS,
) -> tuple[np.ndarray, float]:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size <= 0 or sr <= 0:
        return np.zeros((0,), dtype=np.float32), 0.0
    win = max(16, int(round(float(sr) * float(frame_ms) / 1000.0)))
    hop = max(1, int(round(float(sr) * float(hop_ms) / 1000.0)))
    if x.size < win:
        return np.zeros((0,), dtype=np.float32), float(hop) / float(sr)
    frame_count = 1 + int((x.size - win) // hop)
    rms = np.zeros((frame_count,), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop
        frame = x[start : start + win]
        rms[idx] = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + 1e-12))
    return rms, float(hop) / float(sr)


def _smooth_centered(rms: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or rms.size < 3:
        return rms.astype(np.float32, copy=True)
    kernel = np.ones((int(window),), dtype=np.float32) / float(int(window))
    return np.convolve(rms, kernel, mode="same").astype(np.float32)


def detect_rms_valleys(
    rms: np.ndarray,
    *,
    hop_s: float,
    active_start_ms: float,
    active_end_ms: float,
    min_depth_ratio: float = DEFAULT_VALLEY_MIN_DEPTH_RATIO,
    min_gap_ms: float = DEFAULT_VALLEY_MIN_GAP_MS,
    smooth_frames: int = DEFAULT_VALLEY_SMOOTH_FRAMES,
) -> list[RmsValley]:
    """Local-minima detector with depth gating relative to neighbouring peaks.

    Returns valleys inside the active region, sorted by time. Depth is the
    larger of (left_peak − valley, right_peak − valley) so isolated dips
    register a small depth while clean inter-syllable troughs register a big
    one.
    """
    arr = np.asarray(rms, dtype=np.float32).reshape(-1)
    if arr.size < 3 or hop_s <= 0.0:
        return []

    smoothed = _smooth_centered(arr, max(1, int(smooth_frames)))
    active_start_idx = max(0, int(round(float(active_start_ms) / 1000.0 / hop_s)))
    active_end_idx = min(int(arr.size), int(round(float(active_end_ms) / 1000.0 / hop_s)))
    if active_end_idx - active_start_idx < 5:
        return []

    region = smoothed[active_start_idx:active_end_idx]
    if region.size < 3:
        return []
    # Reference scale for depth: span between p20 and p95 of region.
    p20 = float(np.quantile(region, 0.20))
    p95 = float(np.quantile(region, 0.95))
    scale = max(1e-6, p95 - p20)

    valleys: list[RmsValley] = []
    # Local minima: strictly less than both neighbours.
    for local_idx in range(1, region.size - 1):
        v = float(region[local_idx])
        if v > region[local_idx - 1] or v > region[local_idx + 1]:
            continue
        # Find nearest peak to the left and right (highest value within ±50 frames).
        lo_lookback = max(0, local_idx - 50)
        hi_lookahead = min(region.size, local_idx + 50)
        left_peak = float(np.max(region[lo_lookback:local_idx + 1]))
        right_peak = float(np.max(region[local_idx:hi_lookahead]))
        depth = max(left_peak - v, right_peak - v)
        depth_ratio = depth / scale
        if depth_ratio < float(min_depth_ratio):
            continue
        global_idx = active_start_idx + local_idx
        time_ms = float(global_idx) * float(hop_s) * 1000.0
        valleys.append(RmsValley(time_ms=time_ms, rms_value=v, depth=float(depth_ratio)))

    # Min gap enforcement: greedily keep deepest, drop close neighbours.
    if not valleys:
        return []
    valleys_sorted_by_depth = sorted(valleys, key=lambda v: -v.depth)
    kept: list[RmsValley] = []
    for v in valleys_sorted_by_depth:
        if all(abs(v.time_ms - k.time_ms) >= float(min_gap_ms) for k in kept):
            kept.append(v)
    kept.sort(key=lambda v: v.time_ms)
    return kept


# ---------------------------------------------------------------------------
# Segment boundary construction
# ---------------------------------------------------------------------------


def build_segment_boundaries(
    *,
    slot_count: int,
    valleys: list[RmsValley],
    active_start_ms: float,
    active_end_ms: float,
) -> tuple[list[float], str]:
    """Return (slot_count + 1) boundary times and a mode tag for telemetry.

    Modes:
      - ``valleys_only``: enough valleys to pick `slot_count − 1` directly.
      - ``mixed``: some valleys present, missing positions linearly interpolated.
      - ``linear``: no valleys → pure linear partition (legacy behaviour).
    """
    slot_count = max(1, int(slot_count))
    start = float(active_start_ms)
    end = float(active_end_ms)
    if slot_count == 1 or end - start < 1.0:
        return [start, end], "linear"

    desired = slot_count - 1
    if not valleys:
        boundaries = [start + (end - start) * float(i) / float(slot_count) for i in range(slot_count + 1)]
        boundaries[0] = start
        boundaries[-1] = end
        return boundaries, "linear"

    sorted_by_depth = sorted(valleys, key=lambda v: -v.depth)
    if len(sorted_by_depth) >= desired:
        chosen = sorted([v.time_ms for v in sorted_by_depth[:desired]])
        boundaries = [start, *chosen, end]
        return boundaries, "valleys_only"

    # Mixed: have some valleys, fill in remaining slots with linear interpolation.
    chosen_times = sorted([v.time_ms for v in sorted_by_depth])
    # Build segments greedily: place each chosen valley between linear targets.
    linear_targets = [start + (end - start) * float(i) / float(slot_count) for i in range(1, slot_count)]
    # Replace each linear target with the closest unused valley if reasonable.
    used = [False] * len(chosen_times)
    final: list[float] = []
    for target in linear_targets:
        best_idx = -1
        best_dist = float("inf")
        for j, t in enumerate(chosen_times):
            if used[j]:
                continue
            d = abs(target - t)
            if d < best_dist:
                best_dist = d
                best_idx = j
        # Only substitute if the valley is within half the segment width — otherwise stay linear.
        segment_width = (end - start) / float(slot_count)
        if best_idx >= 0 and best_dist <= segment_width:
            used[best_idx] = True
            final.append(chosen_times[best_idx])
        else:
            final.append(target)
    final.sort()
    boundaries = [start, *final, end]
    return boundaries, "mixed"


# ---------------------------------------------------------------------------
# Onset density picker (signal B)
# ---------------------------------------------------------------------------


def best_onset_in_range(
    candidates: Iterable[BoundaryCandidate],
    *,
    start_ms: float,
    end_ms: float,
    preferred_kinds: tuple[str, ...] = DEFAULT_ONSET_KINDS,
) -> BoundaryCandidate | None:
    """Return the highest-scoring candidate inside [start_ms, end_ms].

    Preference order: model-source onsets > audio-source onsets, then by score.
    """
    if start_ms > end_ms:
        start_ms, end_ms = end_ms, start_ms
    preferred = set(preferred_kinds)
    best: BoundaryCandidate | None = None
    best_key: tuple[int, float] | None = None
    for c in candidates:
        if c.kind not in preferred:
            continue
        if c.time_ms < start_ms or c.time_ms > end_ms:
            continue
        source = str(c.source or "")
        # Lower priority value = better; model peaks beat audio onsets beat fallback.
        if source.startswith("model"):
            tier = 0
        elif source.startswith("audio:lwc"):
            tier = 1
        elif source.startswith("audio"):
            tier = 2
        else:
            tier = 3
        # Maximize (negative tier, score) — i.e., minimize tier, then maximize score.
        key = (-tier, float(c.score))
        if best is None or key > best_key:
            best = c
            best_key = key
    return best


# ---------------------------------------------------------------------------
# Hybrid timeline builder (A + B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridTimelineResult:
    timeline: list[float]
    boundaries: list[float]
    valley_mode: str  # 'valleys_only' | 'mixed' | 'linear'
    onsets_used: int  # count of segments resolved by onset (vs linear fallback inside segment)
    valley_count: int


def build_hybrid_anchor_timeline(
    *,
    wav_path: str,
    slot_count: int,
    active_start_ms: float,
    active_end_ms: float,
    candidates: Iterable[BoundaryCandidate],
    valley_min_depth_ratio: float = DEFAULT_VALLEY_MIN_DEPTH_RATIO,
    valley_min_gap_ms: float = DEFAULT_VALLEY_MIN_GAP_MS,
    rms_frame_ms: float = DEFAULT_RMS_FRAME_MS,
    rms_hop_ms: float = DEFAULT_RMS_HOP_MS,
    onset_kinds: tuple[str, ...] = DEFAULT_ONSET_KINDS,
) -> HybridTimelineResult | None:
    """Return slot_count anchor positions using A+B, or None if wav unreadable.

    Always returns a fully-populated timeline when the wav is loadable, even if
    every signal is absent (falls back to linear). The caller should only
    interpret ``None`` as "use the legacy path".
    """
    slot_count = max(1, int(slot_count))
    try:
        samples, sr, _duration = load_wav_mono(str(wav_path))
    except Exception:
        return None
    if samples.size <= 0 or sr <= 0:
        return None

    rms, hop_s = compute_rms_envelope(samples, sr, frame_ms=rms_frame_ms, hop_ms=rms_hop_ms)
    valleys = detect_rms_valleys(
        rms,
        hop_s=hop_s,
        active_start_ms=float(active_start_ms),
        active_end_ms=float(active_end_ms),
        min_depth_ratio=float(valley_min_depth_ratio),
        min_gap_ms=float(valley_min_gap_ms),
    )
    boundaries, mode = build_segment_boundaries(
        slot_count=slot_count,
        valleys=valleys,
        active_start_ms=float(active_start_ms),
        active_end_ms=float(active_end_ms),
    )

    cand_list = list(candidates)
    timeline: list[float] = []
    onsets_used = 0
    for i in range(slot_count):
        seg_start = float(boundaries[i])
        seg_end = float(boundaries[i + 1])
        best = best_onset_in_range(
            cand_list,
            start_ms=seg_start,
            end_ms=seg_end,
            preferred_kinds=onset_kinds,
        )
        if best is not None:
            timeline.append(float(best.time_ms))
            onsets_used += 1
        else:
            # Bias toward the segment start (where a syllable would begin)
            # rather than dead-center; this matches typical voicebank manual
            # placement better than a midpoint guess.
            timeline.append(seg_start + 0.15 * (seg_end - seg_start))

    # Monotonic guarantee — segments are already sorted, but onset pick could
    # have grabbed a candidate near the segment boundary that crosses the next
    # segment's start. Re-sort just in case.
    timeline.sort()
    return HybridTimelineResult(
        timeline=timeline,
        boundaries=list(boundaries),
        valley_mode=mode,
        onsets_used=onsets_used,
        valley_count=len(valleys),
    )


# ---------------------------------------------------------------------------
# Environment-variable helpers (for wav_decoder integration)
# ---------------------------------------------------------------------------


def anchor_timeline_mode_from_env(default: str = "linear") -> str:
    raw = str(os.environ.get("UTOA_BOUNDARY_ANCHOR_TIMELINE", "") or "").strip().lower()
    if raw in {"filename", "tokens", "token"}:
        return "filename"
    if raw in {"hybrid", "ab", "valley_onset"}:
        return "hybrid"
    if raw in {"linear", "legacy"}:
        return "linear"
    return default


def env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


__all__ = [
    "DEFAULT_ONSET_KINDS",
    "DEFAULT_RMS_FRAME_MS",
    "DEFAULT_RMS_HOP_MS",
    "DEFAULT_VALLEY_MIN_DEPTH_RATIO",
    "DEFAULT_VALLEY_MIN_GAP_MS",
    "HybridTimelineResult",
    "RmsValley",
    "anchor_timeline_mode_from_env",
    "best_onset_in_range",
    "build_hybrid_anchor_timeline",
    "build_segment_boundaries",
    "compute_rms_envelope",
    "detect_rms_valleys",
    "env_float",
]
