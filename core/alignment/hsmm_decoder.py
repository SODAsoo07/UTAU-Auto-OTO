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
    trailing_gap_penalty_per_ms: float = 0.0,
    allow_internal_gaps: bool = False,
    internal_gap_penalty_per_ms: float = 0.0,
    max_internal_gap_ms: float = 0.0,
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

    prev: dict[int, tuple[float, list[tuple[int, int]]]]
    if allow_leading_gap:
        leading_penalty = max(0.0, float(leading_gap_penalty_per_ms))
        prev = {
            start_frame: (-(leading_penalty * float(start_frame) * frame_shift), [])
            for start_frame in range(frame_count)
        }
    else:
        prev = {0: (0.0, [])}
    pruned = 0
    for state_index, spec in enumerate(states):
        curr: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        min_frames = max(1, int(math.ceil(float(spec.min_duration_ms) / frame_shift)))
        max_frames = max(min_frames, int(math.ceil(float(spec.max_duration_ms) / frame_shift)))
        max_frames = min(max_frames, frame_count)
        prefix = prefix_by_state[spec.state_id]
        max_gap_frames = 0
        if allow_internal_gaps and state_index > 0:
            max_gap_frames = max(0, int(math.ceil(float(max_internal_gap_ms) / frame_shift)))
        gap_penalty = max(0.0, float(internal_gap_penalty_per_ms))
        for prev_end, (prev_score, prev_path) in prev.items():
            last_start = min(frame_count - min_frames, prev_end + max_gap_frames)
            for start_frame in range(prev_end, last_start + 1):
                skipped_ms = float(max(0, start_frame - prev_end)) * frame_shift
                first_end = start_frame + min_frames
                last_end = min(frame_count, start_frame + max_frames)
                for end in range(first_end, last_end + 1):
                    if time.perf_counter() - start_time > timeout_s:
                        return HSMMDecodeResult(
                            ok=False,
                            reason="decoder_timeout",
                            timeout=True,
                            pruned_endpoint_count=pruned,
                            meta={"state_index": state_index, "frame_count": frame_count},
                        )
                    duration_ms = (end - start_frame) * frame_shift
                    seg_score = _segment_mean(prefix, start_frame, end) + _duration_score(spec, duration_ms)
                    score = prev_score + seg_score - (gap_penalty * skipped_ms)
                    existing = curr.get(end)
                    if existing is None or score > existing[0]:
                        curr[end] = (score, prev_path + [(start_frame, end)])
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
        prev = curr

    if not allow_trailing_gap:
        if frame_count not in prev:
            return HSMMDecodeResult(
                ok=False,
                reason="timeline_decode_failed",
                pruned_endpoint_count=pruned,
                meta={"frame_count": frame_count, "requires_full_coverage": True},
            )
        best_end, (best_score, best_path) = frame_count, prev[frame_count]
    else:
        trailing_penalty = max(0.0, float(trailing_gap_penalty_per_ms))
        best_end, (best_score, best_path) = max(
            prev.items(),
            key=lambda item: item[1][0] - (trailing_penalty * float(max(0, frame_count - item[0])) * frame_shift),
        )
        best_score = best_score - (trailing_penalty * float(max(0, frame_count - best_end)) * frame_shift)
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
            "allow_internal_gaps": bool(allow_internal_gaps),
            "allow_trailing_gap": bool(allow_trailing_gap),
        },
    )


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
    return -0.5 * z * z


def _infer_frame_shift_ms(times: Sequence[float]) -> float:
    if len(times) >= 2:
        diffs = [float(b) - float(a) for a, b in zip(times[:-1], times[1:]) if float(b) > float(a)]
        if diffs:
            return max(0.001, sorted(diffs)[len(diffs) // 2])
    return 5.0


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
    "decode_segmental_hsmm",
]
