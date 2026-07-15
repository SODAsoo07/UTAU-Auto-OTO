from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence


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
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
    state_frame_windows: Mapping[str, HSMMStateFrameWindow] | None = None,
) -> HSMMDecodeResult:
    if not states:
        return HSMMDecodeResult(ok=False, reason="empty_states")
    times = [float(t) for t in frame_times_ms]
    if not times:
        return HSMMDecodeResult(ok=False, reason="empty_frames")
    frame_shift = _infer_frame_shift_ms(times)
    frame_count = len(times)
    start_time = time.perf_counter()
    timeout_s = max(0.001, float(timeout_ms) / 1000.0)
    prefix_by_state = {
        spec.state_id: _prefix_scores(_resolve_scores(spec, frame_scores, frame_count))
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
                skipped_ms = float(max(0, start_frame - prev_end)) * frame_shift
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
                    score = base_score + seg_score
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
        if beam_width_per_state > 0 and len(curr) > beam_width_per_state:
            sorted_items = sorted(curr.items(), key=lambda item: item[1][0], reverse=True)
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
        trailing_penalty = max(0.0, float(trailing_gap_penalty_per_ms))
        best_end, best_score = max(
            prev.items(),
            key=lambda item: item[1] - (trailing_penalty * float(max(0, frame_count - item[0])) * frame_shift),
        )
        best_score = best_score - (trailing_penalty * float(max(0, frame_count - best_end)) * frame_shift)
    best_path: list[tuple[int, int]] = []
    path_end = best_end
    for layer in reversed(layers):
        _score, prev_end, start_frame = layer[path_end]
        best_path.append((start_frame, path_end))
        path_end = prev_end
    best_path.reverse()
    if best_end <= 0 or not best_path:
        return HSMMDecodeResult(ok=False, reason="timeline_decode_failed", pruned_endpoint_count=pruned)

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
            "internal_gap_ms": _internal_gap_ms(best_path, frame_shift),
            "trailing_gap_ms": float(max(0, frame_count - best_end)) * frame_shift,
            "allow_leading_gap": bool(allow_leading_gap),
            "max_leading_gap_ms": float(max_leading_gap_ms or 0.0),
            "allow_internal_gaps": bool(allow_internal_gaps),
            "max_internal_gap_ms": float(max_internal_gap_ms or 0.0),
            "allow_trailing_gap": bool(allow_trailing_gap),
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
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
) -> HSMMDecodeResult:
    """Run a low-resolution pass, then refine inside per-state frame windows.

    This mirrors the practical SHIRO-style workflow: use a cheap first pass to
    keep the expensive HSMM search from exploring every plausible boundary.
    """
    times = [float(t) for t in frame_times_ms]
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
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
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
    coarse_timeout_ms = max(1, int(timeout_ms * 0.35))
    coarse = decode_segmental_hsmm(
        states,
        coarse_times,
        coarse_scores,
        beam_width_per_state=max(8, int(beam_width_per_state)),
        timeout_ms=coarse_timeout_ms,
        allow_leading_gap=allow_leading_gap,
        allow_trailing_gap=allow_trailing_gap,
        leading_gap_penalty_per_ms=leading_gap_penalty_per_ms,
        max_leading_gap_ms=max_leading_gap_ms,
        trailing_gap_penalty_per_ms=trailing_gap_penalty_per_ms,
        allow_internal_gaps=allow_internal_gaps,
        internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
        max_internal_gap_ms=max_internal_gap_ms,
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
                allow_internal_gaps=allow_internal_gaps,
                internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
                max_internal_gap_ms=max_internal_gap_ms,
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
        allow_internal_gaps=allow_internal_gaps,
        internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
        max_internal_gap_ms=max_internal_gap_ms,
        state_frame_windows=windows,
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
            allow_internal_gaps=allow_internal_gaps,
            internal_gap_penalty_per_ms=internal_gap_penalty_per_ms,
            max_internal_gap_ms=max_internal_gap_ms,
        )
        return _copy_result_with_meta(
            fallback,
            {**meta, "coarse_to_refine": False, "coarse_to_refine_reason": f"refine_failed:{refined.reason}"},
        )
    return _copy_result_with_meta(refined, {**meta, "coarse_to_refine_reason": f"refine_failed:{refined.reason}"})


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


__all__ = [
    "HSMM_DECODER_SCHEMA_VERSION",
    "HSMMDecodeResult",
    "HSMMDecodedState",
    "HSMMStateSpec",
    "HSMMStateFrameWindow",
    "decode_segmental_hsmm_coarse_to_refine",
    "decode_segmental_hsmm",
]
