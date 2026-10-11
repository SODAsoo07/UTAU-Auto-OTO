from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


HSMM_DECODER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HSMMStateSpec:
    state_id: str
    state_type: str
    min_duration_ms: float
    max_duration_ms: float
    mode_duration_ms: float = 0.0
    duration_sigma_ms: float = 40.0
    # Scales the duration penalty applied when the selected segment is *longer*
    # than ``mode_duration_ms``. Values < 1.0 make the duration prior asymmetric
    # so that sustained segments (e.g. held vowels) are not trimmed from the
    # front to match the nominal mode duration. 1.0 keeps the symmetric penalty.
    over_duration_penalty_scale: float = 1.0


@dataclass(frozen=True)
class HSMMStateFrameWindow:
    start_min_frame: int = 0
    start_max_frame: int | None = None
    end_min_frame: int = 1
    end_max_frame: int | None = None


@dataclass(frozen=True)
class HSMMDecodedState:
    state_id: str
    state_type: str
    start_ms: float
    end_ms: float
    score: float
    duration_ms: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class HSMMDecodeResult:
    ok: bool
    states: tuple[HSMMDecodedState, ...] = ()
    score: float = float("-inf")
    reason: str = ""
    timeout: bool = False
    pruned_endpoint_count: int = 0
    schema_version: int = HSMM_DECODER_SCHEMA_VERSION
    meta: Mapping[str, object] = field(default_factory=dict)


def decode_segmental_hsmm(
    states: Sequence[HSMMStateSpec],
    frame_times_ms: Sequence[float],
    frame_scores: Mapping[str, Sequence[float]],
    *,
    beam_width_per_state: int = 64,
    timeout_ms: int = 5000,
    allow_leading_gap: bool = False,
    allow_trailing_gap: bool = True,
    leading_gap_penalty_per_ms: float = 0.0,
    max_leading_gap_ms: float = 0.0,
    trailing_gap_penalty_per_ms: float = 0.0,
    trailing_gap_frame_weights: Sequence[float] | None = None,
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
    internal_gap_frame_weights: Sequence[float] | None = None,
    state_frame_windows: Mapping[str, HSMMStateFrameWindow] | None = None,
    transition_start_scores: Mapping[str, Sequence[float]] | None = None,
) -> HSMMDecodeResult:
    if not states:
        return HSMMDecodeResult(ok=False, reason="empty_states")
    times = [float(t) for t in frame_times_ms]
    if not times:
        return HSMMDecodeResult(ok=False, reason="empty_frames")
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        return HSMMDecodeResult(ok=False, reason="invalid_frame_times")
    if len({spec.state_id for spec in states}) != len(states):
        return HSMMDecodeResult(ok=False, reason="duplicate_state_id")
    for window in (state_frame_windows or {}).values():
        if ((window.start_max_frame is not None and window.start_max_frame < window.start_min_frame)
                or (window.end_max_frame is not None and window.end_max_frame < window.end_min_frame)):
            return HSMMDecodeResult(ok=False, reason="invalid_state_window")
    frame_shift = _infer_frame_shift_ms(times)
    frame_count = len(times)
    tail_weights = _validated_tail_weights(trailing_gap_frame_weights, frame_count)
    if trailing_gap_frame_weights is not None and tail_weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_trailing_gap_weights")
    gap_weights = _validated_tail_weights(internal_gap_frame_weights, frame_count)
    if internal_gap_frame_weights is not None and gap_weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_internal_gap_weights")
    # Weighted internal gaps cost penalty * frame_shift * weight per skipped frame,
    # so a pause over silence is cheap and a gap over speech is not.
    gap_prefix = None if gap_weights is None else np.concatenate(([0.0], np.cumsum(gap_weights)))
    tail_penalty = max(0.0, float(trailing_gap_penalty_per_ms))
    tail_costs = [tail_penalty * (frame_count-end) * frame_shift for end in range(frame_count+1)]
    if tail_weights is not None:
        tail_costs = [0.0] * (frame_count+1)
        for frame in range(frame_count-1, -1, -1):
            tail_costs[frame] = tail_costs[frame+1] + tail_penalty * frame_shift * tail_weights[frame]
    start_time = time.perf_counter()
    timeout_s = max(0.001, float(timeout_ms) / 1000.0)
    prefix_by_state = {
        spec.state_id: _prefix_scores(_resolve_scores(spec, frame_scores, frame_count))
        for spec in states
    }
    transition_by_state = {
        spec.state_id: _resolve_transition_scores(spec.state_id, transition_start_scores, frame_count)
        for spec in states
    }
    windows = _normalize_state_windows(state_frame_windows, frame_count)

    prev: dict[int, float]
    if allow_leading_gap:
        leading_penalty = max(0.0, float(leading_gap_penalty_per_ms))
        max_leading_frames = frame_count - 1
        if float(max_leading_gap_ms or 0.0) > 0.0:
            max_leading_frames = min(max_leading_frames, int(math.ceil(float(max_leading_gap_ms) / frame_shift)))
        prev = {
            start_frame: -(leading_penalty * float(start_frame) * frame_shift)
            for start_frame in range(max_leading_frames + 1)
        }
    else:
        prev = {0: 0.0}
    pruned = 0
    layers: list[dict[int, tuple[float, int, int]]] = []
    for state_index, spec in enumerate(states):
        curr: dict[int, tuple[float, int, int]] = {}
        min_frames = max(1, int(math.ceil(float(spec.min_duration_ms) / frame_shift)))
        max_frames = max(min_frames, int(math.ceil(float(spec.max_duration_ms) / frame_shift)))
        max_frames = min(max_frames, frame_count)
        prefix = prefix_by_state[spec.state_id]
        duration_scores = [
            _duration_score(spec, float(duration_frames) * frame_shift)
            for duration_frames in range(max_frames + 1)
        ]
        max_gap_frames = 0
        if allow_internal_gaps and state_index > 0:
            max_gap_frames = max(0, int(math.ceil(float(max_internal_gap_ms) / frame_shift)))
        gap_penalty = max(0.0, float(internal_gap_penalty_per_ms))
        window = windows.get(spec.state_id)
        for prev_end, prev_score in prev.items():
            first_start = prev_end
            last_start = min(frame_count - min_frames, prev_end + max_gap_frames)
            if window is not None:
                first_start = max(first_start, int(window.start_min_frame))
                last_start = min(last_start, int(window.start_max_frame if window.start_max_frame is not None else frame_count - min_frames))
            if last_start < first_start:
                continue
            for start_frame in range(first_start, last_start + 1):
                if gap_prefix is None:
                    skipped_ms = float(max(0, start_frame - prev_end)) * frame_shift
                else:
                    skipped_ms = float(gap_prefix[start_frame] - gap_prefix[prev_end]) * frame_shift
                base_score = prev_score - (gap_penalty * skipped_ms)
                first_end = start_frame + min_frames
                last_end = min(frame_count, start_frame + max_frames)
                if window is not None:
                    first_end = max(first_end, int(window.end_min_frame))
                    last_end = min(last_end, int(window.end_max_frame if window.end_max_frame is not None else frame_count))
                if last_end < first_end:
                    continue
                prefix_start = prefix[start_frame]
                for end in range(first_end, last_end + 1):
                    duration_frames = end - start_frame
                    seg_score = (
                        (prefix[end] - prefix_start) / duration_frames
                        + duration_scores[duration_frames]
                    )
                    score = base_score + seg_score + transition_by_state[spec.state_id][start_frame]
                    existing = curr.get(end)
                    if existing is None or score > existing[0]:
                        curr[end] = (score, prev_end, start_frame)
        if time.perf_counter() - start_time > timeout_s:
            return HSMMDecodeResult(
                ok=False,
                reason="decoder_timeout",
                timeout=True,
                pruned_endpoint_count=pruned,
                meta={"state_index": state_index, "frame_count": frame_count},
            )
        if not curr:
            return HSMMDecodeResult(
                ok=False,
                reason="timeline_decode_failed",
                pruned_endpoint_count=pruned,
                meta={"state_index": state_index, "frame_count": frame_count},
            )
        # Apply the terminal objective BEFORE pruning. Otherwise a narrow beam
        # can discard the required full-coverage endpoint or the best tail cost.
        is_terminal = state_index == len(states) - 1
        if is_terminal and not allow_trailing_gap:
            curr = {end: value for end, value in curr.items() if end == frame_count}
        if beam_width_per_state > 0 and len(curr) > beam_width_per_state:
            sorted_items = sorted(
                curr.items(),
                key=lambda item: item[1][0] - (tail_costs[item[0]] if is_terminal else 0.0),
                reverse=True,
            )
            pruned += len(sorted_items) - int(beam_width_per_state)
            curr = dict(sorted_items[: int(beam_width_per_state)])
        layers.append(curr)
        prev = {end: value[0] for end, value in curr.items()}

    if not allow_trailing_gap:
        if frame_count not in prev:
            return HSMMDecodeResult(
                ok=False,
                reason="timeline_decode_failed",
                pruned_endpoint_count=pruned,
                meta={"frame_count": frame_count, "requires_full_coverage": True},
            )
        best_end, best_score = frame_count, prev[frame_count]
    else:
        best_end, best_score = max(
            prev.items(),
            key=lambda item: item[1] - tail_costs[item[0]],
        )
        best_score = best_score - tail_costs[best_end]
    best_path: list[tuple[int, int]] = []
    path_end = best_end
    for layer in reversed(layers):
        _score, prev_end, start_frame = layer[path_end]
        best_path.append((start_frame, path_end))
        path_end = prev_end
    best_path.reverse()
    if best_end <= 0 or not best_path:
        return HSMMDecodeResult(ok=False, reason="timeline_decode_failed", pruned_endpoint_count=pruned)
    return _path_result(
        states, times, best_path, best_score,
        prefix_by_state=prefix_by_state, transition_by_state=transition_by_state, tail_costs=tail_costs,
        tail_weights=tail_weights, frame_shift=frame_shift, pruned=pruned,
        allow_leading_gap=allow_leading_gap, max_leading_gap_ms=max_leading_gap_ms,
        allow_internal_gaps=allow_internal_gaps, max_internal_gap_ms=max_internal_gap_ms,
        allow_trailing_gap=allow_trailing_gap,
    )


def decode_segmental_hsmm_exact(
    states: Sequence[HSMMStateSpec],
    frame_times_ms: Sequence[float],
    frame_scores: Mapping[str, Sequence[float]],
    *,
    allow_leading_gap: bool = False,
    allow_trailing_gap: bool = True,
    leading_gap_penalty_per_ms: float = 0.0,
    max_leading_gap_ms: float = 0.0,
    trailing_gap_penalty_per_ms: float = 0.0,
    trailing_gap_frame_weights: Sequence[float] | None = None,
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
    internal_gap_frame_weights: Sequence[float] | None = None,
    state_frame_windows: Mapping[str, HSMMStateFrameWindow] | None = None,
    transition_start_scores: Mapping[str, Sequence[float]] | None = None,
) -> HSMMDecodeResult:
    """Same objective as ``decode_segmental_hsmm`` searched without any beam or coarse pass.

    Per state the best predecessor end for every start (a sliding maximum over the allowed
    internal gap) and the best start for every end (one array operation per duration) are
    computed over all frames at once, so the full search costs states x (gap + duration) array
    operations instead of a Python loop over every (previous end, start, end) triple.
    """
    if not states:
        return HSMMDecodeResult(ok=False, reason="empty_states")
    times = [float(t) for t in frame_times_ms]
    if not times:
        return HSMMDecodeResult(ok=False, reason="empty_frames")
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        return HSMMDecodeResult(ok=False, reason="invalid_frame_times")
    if len({spec.state_id for spec in states}) != len(states):
        return HSMMDecodeResult(ok=False, reason="duplicate_state_id")
    for window in (state_frame_windows or {}).values():
        if ((window.start_max_frame is not None and window.start_max_frame < window.start_min_frame)
                or (window.end_max_frame is not None and window.end_max_frame < window.end_min_frame)):
            return HSMMDecodeResult(ok=False, reason="invalid_state_window")
    frame_shift = _infer_frame_shift_ms(times)
    frame_count = len(times)
    tail_weights = _validated_tail_weights(trailing_gap_frame_weights, frame_count)
    if trailing_gap_frame_weights is not None and tail_weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_trailing_gap_weights")
    gap_weights = _validated_tail_weights(internal_gap_frame_weights, frame_count)
    if internal_gap_frame_weights is not None and gap_weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_internal_gap_weights")
    gap_prefix = (np.arange(frame_count + 1, dtype=np.float64) if gap_weights is None
                  else np.concatenate(([0.0], np.cumsum(np.asarray(gap_weights, dtype=np.float64)))))
    tail_penalty = max(0.0, float(trailing_gap_penalty_per_ms))
    tail_costs = [tail_penalty * (frame_count - end) * frame_shift for end in range(frame_count + 1)]
    if tail_weights is not None:
        tail_costs = [0.0] * (frame_count + 1)
        for frame in range(frame_count - 1, -1, -1):
            tail_costs[frame] = tail_costs[frame + 1] + tail_penalty * frame_shift * tail_weights[frame]
    prefix_by_state = {
        spec.state_id: np.concatenate(([0.0], np.cumsum(np.asarray(_resolve_scores(spec, frame_scores, frame_count)))))
        for spec in states
    }
    transition_by_state = {
        spec.state_id: np.asarray(_resolve_transition_scores(spec.state_id, transition_start_scores, frame_count))
        for spec in states
    }
    windows = _normalize_state_windows(state_frame_windows, frame_count)

    prev = np.full(frame_count + 1, -np.inf)
    if allow_leading_gap:
        max_leading_frames = frame_count - 1
        if float(max_leading_gap_ms or 0.0) > 0.0:
            max_leading_frames = min(max_leading_frames, int(math.ceil(float(max_leading_gap_ms) / frame_shift)))
        prev[:max_leading_frames + 1] = -(max(0.0, float(leading_gap_penalty_per_ms))
                                          * np.arange(max_leading_frames + 1) * frame_shift)
    else:
        prev[0] = 0.0
    gap_cost = max(0.0, float(internal_gap_penalty_per_ms)) * frame_shift
    frames = np.arange(frame_count + 1)
    backpointers: list[tuple[np.ndarray, np.ndarray]] = []
    for state_index, spec in enumerate(states):
        min_frames = max(1, int(math.ceil(float(spec.min_duration_ms) / frame_shift)))
        max_frames = min(max(min_frames, int(math.ceil(float(spec.max_duration_ms) / frame_shift))), frame_count)
        max_gap_frames = 0
        if allow_internal_gaps and state_index > 0:
            max_gap_frames = max(0, int(math.ceil(float(max_internal_gap_ms) / frame_shift)))
        # best[s] = max over previous ends p in [s - gap, s] of prev[p] - gap cost(p, s); ties keep the earliest p
        base = prev + gap_cost * gap_prefix
        best = np.full(frame_count + 1, -np.inf)
        best_prev = frames.copy()
        for shift in range(max_gap_frames, -1, -1):
            candidate = np.full(frame_count + 1, -np.inf)
            candidate[shift:] = base[:frame_count + 1 - shift]
            better = candidate > best
            best = np.where(better, candidate, best)
            best_prev = np.where(better, frames - shift, best_prev)
        best = best - gap_cost * gap_prefix
        best[frame_count - min_frames + 1:] = -np.inf
        window = windows.get(spec.state_id)
        if window is not None:
            best[:window.start_min_frame] = -np.inf
            if window.start_max_frame is not None:
                best[window.start_max_frame + 1:] = -np.inf
        prefix = prefix_by_state[spec.state_id]
        transition = np.append(transition_by_state[spec.state_id], 0.0)
        durations = [_duration_score(spec, float(d) * frame_shift) for d in range(max_frames + 1)]
        curr = np.full(frame_count + 1, -np.inf)
        best_start = np.zeros(frame_count + 1, dtype=np.int64)
        for duration_frames in range(max_frames, min_frames - 1, -1):
            starts = frames[:frame_count + 1 - duration_frames]
            ends = starts + duration_frames
            score = (best[starts] + (prefix[ends] - prefix[starts]) / duration_frames
                     + durations[duration_frames] + transition[starts])
            better = score > curr[ends]
            curr[ends[better]] = score[better]
            best_start[ends[better]] = starts[better]
        if window is not None:
            curr[:window.end_min_frame] = -np.inf
            if window.end_max_frame is not None:
                curr[window.end_max_frame + 1:] = -np.inf
        if not np.isfinite(curr).any():
            return HSMMDecodeResult(
                ok=False,
                reason="timeline_decode_failed",
                meta={"state_index": state_index, "frame_count": frame_count, "exact_search": True},
            )
        backpointers.append((best_start, best_prev))
        prev = curr
    final = prev - np.asarray(tail_costs)
    if not allow_trailing_gap:
        if not np.isfinite(prev[frame_count]):
            return HSMMDecodeResult(
                ok=False,
                reason="timeline_decode_failed",
                meta={"frame_count": frame_count, "requires_full_coverage": True, "exact_search": True},
            )
        best_end = frame_count
    else:
        best_end = int(np.argmax(final))
    best_score = float(final[best_end])
    if best_end <= 0 or not math.isfinite(best_score):
        return HSMMDecodeResult(ok=False, reason="timeline_decode_failed", meta={"exact_search": True})
    best_path: list[tuple[int, int]] = []
    path_end = best_end
    for best_start, best_prev in reversed(backpointers):
        start_frame = int(best_start[path_end])
        best_path.append((start_frame, path_end))
        path_end = int(best_prev[start_frame])
    best_path.reverse()
    result = _path_result(
        states, times, best_path, best_score,
        prefix_by_state=prefix_by_state, transition_by_state=transition_by_state, tail_costs=tail_costs,
        tail_weights=tail_weights, frame_shift=frame_shift, pruned=0,
        allow_leading_gap=allow_leading_gap, max_leading_gap_ms=max_leading_gap_ms,
        allow_internal_gaps=allow_internal_gaps, max_internal_gap_ms=max_internal_gap_ms,
        allow_trailing_gap=allow_trailing_gap,
    )
    return _copy_result_with_meta(result, {"exact_search": True, "coarse_to_refine": False})


def _path_result(
    states: Sequence[HSMMStateSpec],
    times: Sequence[float],
    best_path: Sequence[tuple[int, int]],
    best_score: float,
    *,
    prefix_by_state: Mapping[str, Sequence[float]],
    transition_by_state: Mapping[str, Sequence[float]],
    tail_costs: Sequence[float],
    tail_weights: Sequence[float] | None,
    frame_shift: float,
    pruned: int,
    allow_leading_gap: bool,
    max_leading_gap_ms: float,
    allow_internal_gaps: bool,
    max_internal_gap_ms: float,
    allow_trailing_gap: bool,
) -> HSMMDecodeResult:
    frame_count = len(times)
    best_end = best_path[-1][1]
    decoded: list[HSMMDecodedState] = []
    for spec, (start_frame, end_frame) in zip(states, best_path):
        start_ms = times[min(start_frame, frame_count - 1)]
        # Treat end as exclusive. The boundary after the last covered frame is one shift past the last frame start.
        end_ms = times[min(end_frame - 1, frame_count - 1)] + frame_shift
        duration_ms = max(0.0, end_ms - start_ms)
        prefix = prefix_by_state[spec.state_id]
        decoded.append(
            HSMMDecodedState(
                state_id=spec.state_id,
                state_type=spec.state_type,
                start_ms=start_ms,
                end_ms=end_ms,
                score=_segment_mean(prefix, start_frame, end_frame),
                duration_ms=duration_ms,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
    internal_gaps = _internal_gap_segments(best_path, states, times, frame_shift)
    return HSMMDecodeResult(
        ok=True,
        states=tuple(decoded),
        score=float(best_score),
        reason="ok",
        pruned_endpoint_count=pruned,
        meta={
            "frame_shift_ms": frame_shift,
            "frame_count": frame_count,
            "leading_gap_ms": float(best_path[0][0]) * frame_shift,
            "internal_gap_ms": sum(float(item["duration_ms"]) for item in internal_gaps),
            "internal_gap_count": len(internal_gaps),
            "max_internal_gap_observed_ms": max(
                (float(item["duration_ms"]) for item in internal_gaps),
                default=0.0,
            ),
            "internal_gaps": internal_gaps,
            "trailing_gap_ms": float(max(0, frame_count - best_end)) * frame_shift,
            "trailing_gap_cost": tail_costs[best_end] if allow_trailing_gap else 0.0,
            "trailing_gap_weighted": tail_weights is not None,
            "allow_leading_gap": bool(allow_leading_gap),
            "max_leading_gap_ms": float(max_leading_gap_ms or 0.0),
            "allow_internal_gaps": bool(allow_internal_gaps),
            "max_internal_gap_ms": float(max_internal_gap_ms or 0.0),
            "allow_trailing_gap": bool(allow_trailing_gap),
            "transition_start_score_state_count": sum(
                int(any(value != 0.0 for value in scores))
                for scores in transition_by_state.values()
            ),
        },
    )


def decode_segmental_hsmm_coarse_to_refine(
    states: Sequence[HSMMStateSpec],
    frame_times_ms: Sequence[float],
    frame_scores: Mapping[str, Sequence[float]],
    *,
    coarse_stride: int = 4,
    refine_window_ms: float = 110.0,
    fallback_to_full: bool = True,
    beam_width_per_state: int = 64,
    timeout_ms: int = 5000,
    allow_leading_gap: bool = False,
    allow_trailing_gap: bool = True,
    leading_gap_penalty_per_ms: float = 0.0,
    max_leading_gap_ms: float = 0.0,
    trailing_gap_penalty_per_ms: float = 0.0,
    trailing_gap_frame_weights: Sequence[float] | None = None,
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
    internal_gap_frame_weights: Sequence[float] | None = None,
    transition_start_scores: Mapping[str, Sequence[float]] | None = None,
    coarse_beam_width_per_state: int | None = None,
) -> HSMMDecodeResult:
    """Run a low-resolution pass, then refine inside per-state frame windows.

    ``coarse_beam_width_per_state`` overrides the coarse pass beam; 0 searches the coarse
    lattice without pruning, so a pruned coarse pass cannot lock the refine windows onto
    a path that is off by whole syllables.

    This mirrors the practical SHIRO-style workflow: use a cheap first pass to
    keep the expensive HSMM search from exploring every plausible boundary.
    """
    times = [float(t) for t in frame_times_ms]
    weights = _validated_tail_weights(trailing_gap_frame_weights, len(times))
    if trailing_gap_frame_weights is not None and weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_trailing_gap_weights")
    gap_weights = _validated_tail_weights(internal_gap_frame_weights, len(times))
    if internal_gap_frame_weights is not None and gap_weights is None:
        return HSMMDecodeResult(ok=False, reason="invalid_internal_gap_weights")
    stride = max(1, int(coarse_stride or 1))
    if len(times) > 500 and stride < 6:
        stride = 6
    elif len(times) > 800 and stride < 8:
        stride = 8
    if stride <= 1 or len(times) < max(12, stride * 4):
        result = decode_segmental_hsmm(
            states,
            times,
            frame_scores,
            beam_width_per_state=beam_width_per_state,
            timeout_ms=timeout_ms,
            allow_leading_gap=allow_leading_gap,
            allow_trailing_gap=allow_trailing_gap,
            leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
            max_leading_gap_ms=max_leading_gap_ms,
            trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
            trailing_gap_frame_weights=weights,
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
            internal_gap_frame_weights=gap_weights,
            transition_start_scores=transition_start_scores,
        )
        return _copy_result_with_meta(result, {"coarse_to_refine": False, "coarse_to_refine_reason": "too_short_or_disabled"})
    if not states:
        return HSMMDecodeResult(ok=False, reason="empty_states")

    coarse_indices = list(range(0, len(times), stride))
    if coarse_indices[-1] != len(times) - 1:
        coarse_indices.append(len(times) - 1)
    coarse_times = [times[idx] for idx in coarse_indices]
    coarse_scores = {
        key: _sample_scores_for_indices(values, coarse_indices)
        for key, values in frame_scores.items()
    }
    coarse_transition_scores = {
        key: _sample_scores_for_indices(values, coarse_indices)
        for key, values in (transition_start_scores or {}).items()
    }
    coarse_timeout_ms = max(1, int(timeout_ms * 0.35))
    coarse = decode_segmental_hsmm(
        states,
        coarse_times,
        coarse_scores,
        beam_width_per_state=(max(8, int(beam_width_per_state)) if coarse_beam_width_per_state is None
                              else max(0, int(coarse_beam_width_per_state))),
        timeout_ms=coarse_timeout_ms,
        allow_leading_gap=allow_leading_gap,
        allow_trailing_gap=allow_trailing_gap,
        leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
        max_leading_gap_ms=max_leading_gap_ms,
        trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
        trailing_gap_frame_weights=None if weights is None else [weights[i] for i in coarse_indices],
        allow_internal_gaps=allow_internal_gaps,
        internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
        max_internal_gap_ms=max_internal_gap_ms,
        internal_gap_frame_weights=None if gap_weights is None else [gap_weights[i] for i in coarse_indices],
        transition_start_scores=coarse_transition_scores,
    )
    if not coarse.ok:
        if fallback_to_full:
            fallback = decode_segmental_hsmm(
                states,
                times,
                frame_scores,
                beam_width_per_state=beam_width_per_state,
                timeout_ms=timeout_ms,
                allow_leading_gap=allow_leading_gap,
                allow_trailing_gap=allow_trailing_gap,
                leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
                max_leading_gap_ms=max_leading_gap_ms,
                trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
                trailing_gap_frame_weights=weights,
                allow_internal_gaps=allow_internal_gaps,
                internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
                max_internal_gap_ms=max_internal_gap_ms,
                internal_gap_frame_weights=gap_weights,
                transition_start_scores=transition_start_scores,
            )
            return _copy_result_with_meta(
                fallback,
                {
                    "coarse_to_refine": False,
                    "coarse_to_refine_reason": f"coarse_failed:{coarse.reason}",
                    "coarse_timeout": bool(coarse.timeout),
                },
            )
        return _copy_result_with_meta(
            coarse,
            {"coarse_to_refine": False, "coarse_to_refine_reason": f"coarse_failed:{coarse.reason}"},
        )

    frame_shift = _infer_frame_shift_ms(times)
    window_frames = max(1, int(math.ceil(max(0.0, float(refine_window_ms)) / frame_shift)))
    windows = _windows_from_coarse_result(coarse, times, window_frames=window_frames)
    refined = decode_segmental_hsmm(
        states,
        times,
        frame_scores,
        beam_width_per_state=beam_width_per_state,
        timeout_ms=timeout_ms,
        allow_leading_gap=allow_leading_gap,
        allow_trailing_gap=allow_trailing_gap,
        leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
        max_leading_gap_ms=max_leading_gap_ms,
        trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
        trailing_gap_frame_weights=weights,
        allow_internal_gaps=allow_internal_gaps,
        internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
        max_internal_gap_ms=max_internal_gap_ms,
        internal_gap_frame_weights=gap_weights,
        state_frame_windows=windows,
        transition_start_scores=transition_start_scores,
    )
    meta = {
        "coarse_to_refine": True,
        "coarse_stride": int(stride),
        "refine_window_ms": float(refine_window_ms),
        "refine_window_frames": int(window_frames),
        "coarse_score": float(coarse.score),
        "coarse_pruned_endpoint_count": int(coarse.pruned_endpoint_count),
        "coarse_timeout": bool(coarse.timeout),
        "coarse_frame_count": len(coarse_times),
        **({"coarse_beam_width_per_state": int(coarse_beam_width_per_state)}
           if coarse_beam_width_per_state is not None else {}),
    }
    if refined.ok:
        return _copy_result_with_meta(refined, meta)
    if fallback_to_full:
        fallback = decode_segmental_hsmm(
            states,
            times,
            frame_scores,
            beam_width_per_state=beam_width_per_state,
            timeout_ms=timeout_ms,
            allow_leading_gap=allow_leading_gap,
            allow_trailing_gap=allow_trailing_gap,
            leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
            max_leading_gap_ms=max_leading_gap_ms,
            trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
            trailing_gap_frame_weights=weights,
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
            internal_gap_frame_weights=gap_weights,
            transition_start_scores=transition_start_scores,
        )
        return _copy_result_with_meta(
            fallback,
            {**meta, "coarse_to_refine": False, "coarse_to_refine_reason": f"refine_failed:{refined.reason}"},
        )
    return _copy_result_with_meta(refined, {**meta, "coarse_to_refine_reason": f"refine_failed:{refined.reason}"})


def _validated_tail_weights(values: Sequence[float] | None, count: int) -> list[float] | None:
    if values is None:
        return None
    try:
        weights = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(weights) != count or any(not math.isfinite(v) or not 0 <= v <= 1 for v in weights):
        return None
    return weights


def _resolve_scores(
    spec: HSMMStateSpec,
    frame_scores: Mapping[str, Sequence[float]],
    frame_count: int,
) -> list[float]:
    values = frame_scores.get(spec.state_id)
    if values is None:
        values = frame_scores.get(spec.state_type)
    if values is None:
        return [-10.0] * frame_count
    out = [float(v) for v in values]
    if len(out) < frame_count:
        out = out + [-10.0] * (frame_count - len(out))
    return out[:frame_count]


def _resolve_transition_scores(
    state_id: str,
    transition_start_scores: Mapping[str, Sequence[float]] | None,
    frame_count: int,
) -> list[float]:
    if not transition_start_scores:
        return [0.0] * frame_count
    values = transition_start_scores.get(state_id)
    if values is None:
        return [0.0] * frame_count
    result = [float(value) for value in values[:frame_count]]
    if len(result) < frame_count:
        result.extend([0.0] * (frame_count - len(result)))
    return [value if math.isfinite(value) else 0.0 for value in result]


def _normalize_state_windows(
    state_frame_windows: Mapping[str, HSMMStateFrameWindow] | None,
    frame_count: int,
) -> dict[str, HSMMStateFrameWindow]:
    if not state_frame_windows:
        return {}
    out: dict[str, HSMMStateFrameWindow] = {}
    for state_id, window in state_frame_windows.items():
        start_min = max(0, min(frame_count, int(window.start_min_frame)))
        start_max_raw = window.start_max_frame
        start_max = None if start_max_raw is None else max(0, min(frame_count, int(start_max_raw)))
        end_min = max(1, min(frame_count, int(window.end_min_frame)))
        end_max_raw = window.end_max_frame
        end_max = None if end_max_raw is None else max(1, min(frame_count, int(end_max_raw)))
        if start_max is not None and start_max < start_min:
            continue
        if end_max is not None and end_max < end_min:
            continue
        out[str(state_id)] = HSMMStateFrameWindow(
            start_min_frame=start_min,
            start_max_frame=start_max,
            end_min_frame=end_min,
            end_max_frame=end_max,
        )
    return out


def _prefix_scores(scores: Sequence[float]) -> list[float]:
    prefix = [0.0]
    total = 0.0
    for value in scores:
        total += float(value)
        prefix.append(total)
    return prefix


def _segment_mean(prefix: Sequence[float], start: int, end: int) -> float:
    if end <= start:
        return -1e9
    return (float(prefix[end]) - float(prefix[start])) / float(end - start)


def _duration_score(spec: HSMMStateSpec, duration_ms: float) -> float:
    mode = float(spec.mode_duration_ms or 0.0)
    if mode <= 0.0:
        return 0.0
    sigma = max(1.0, float(spec.duration_sigma_ms or 40.0))
    z = (float(duration_ms) - mode) / sigma
    penalty = -0.5 * z * z
    if float(duration_ms) > mode:
        over_scale = max(0.0, float(getattr(spec, "over_duration_penalty_scale", 1.0)))
        penalty *= over_scale
    return penalty


def _infer_frame_shift_ms(times: Sequence[float]) -> float:
    if len(times) >= 2:
        diffs = [float(b) - float(a) for a, b in zip(times[:-1], times[1:]) if float(b) > float(a)]
        if diffs:
            return max(0.001, sorted(diffs)[len(diffs) // 2])
    return 5.0


def _sample_scores_for_indices(values: Sequence[float], indices: Sequence[int]) -> list[float]:
    src = [float(value) for value in values]
    if not src:
        return [0.0 for _ in indices]
    last = len(src) - 1
    return [src[max(0, min(last, int(index)))] for index in indices]


def _windows_from_coarse_result(
    coarse: HSMMDecodeResult,
    times: Sequence[float],
    *,
    window_frames: int,
) -> dict[str, HSMMStateFrameWindow]:
    frame_count = len(times)
    if frame_count <= 0:
        return {}
    frame_shift = _infer_frame_shift_ms(times)
    out: dict[str, HSMMStateFrameWindow] = {}
    for state in coarse.states:
        start = _time_to_frame_index(float(state.start_ms), times, frame_shift_ms=frame_shift)
        end = _time_to_frame_index(float(state.end_ms), times, frame_shift_ms=frame_shift)
        end = max(start + 1, end)
        out[state.state_id] = HSMMStateFrameWindow(
            start_min_frame=max(0, start - int(window_frames)),
            start_max_frame=min(frame_count - 1, start + int(window_frames)),
            end_min_frame=max(1, end - int(window_frames)),
            end_max_frame=min(frame_count, end + int(window_frames)),
        )
    return out


def _time_to_frame_index(time_ms: float, times: Sequence[float], *, frame_shift_ms: float) -> int:
    if not times:
        return 0
    if len(times) >= 2:
        first = float(times[0])
        return max(0, min(len(times), int(round((float(time_ms) - first) / max(0.001, frame_shift_ms)))))
    return 0


def _copy_result_with_meta(result: HSMMDecodeResult, extra_meta: Mapping[str, object]) -> HSMMDecodeResult:
    return HSMMDecodeResult(
        ok=result.ok,
        states=result.states,
        score=result.score,
        reason=result.reason,
        timeout=result.timeout,
        pruned_endpoint_count=result.pruned_endpoint_count,
        schema_version=result.schema_version,
        meta={**dict(result.meta or {}), **dict(extra_meta or {})},
    )


def _internal_gap_ms(path: Sequence[tuple[int, int]], frame_shift_ms: float) -> float:
    total_frames = 0
    for previous, current in zip(path[:-1], path[1:]):
        total_frames += max(0, int(current[0]) - int(previous[1]))
    return float(total_frames) * float(frame_shift_ms)


def _internal_gap_segments(
    path: Sequence[tuple[int, int]],
    states: Sequence[HSMMStateSpec],
    frame_times_ms: Sequence[float],
    frame_shift_ms: float,
) -> list[dict[str, object]]:
    gaps: list[dict[str, object]] = []
    for index, (previous, current) in enumerate(zip(path[:-1], path[1:])):
        start_frame = int(previous[1])
        end_frame = int(current[0])
        if end_frame <= start_frame:
            continue
        start_ms = float(frame_times_ms[min(max(0, start_frame - 1), len(frame_times_ms) - 1)]) + float(frame_shift_ms)
        end_ms = float(frame_times_ms[min(max(0, end_frame), len(frame_times_ms) - 1)])
        gaps.append(
            {
                "after_state_index": index,
                "before_state_index": index + 1,
                "after_state_id": states[index].state_id,
                "before_state_id": states[index + 1].state_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": float(end_frame - start_frame) * float(frame_shift_ms),
            }
        )
    return gaps


__all__ = [
    "HSMM_DECODER_SCHEMA_VERSION",
    "HSMMDecodeResult",
    "HSMMDecodedState",
    "HSMMStateSpec",
    "HSMMStateFrameWindow",
    "decode_segmental_hsmm_coarse_to_refine",
    "decode_segmental_hsmm",
    "decode_segmental_hsmm_exact",
]
