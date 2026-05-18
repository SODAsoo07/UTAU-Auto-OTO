from __future__ import annotations

from dataclasses import dataclass

from core.phoneme_boundary.types import BoundaryFrameScores, DecodedBoundaryEvent, PhonemePlan, PhoneToken


@dataclass(frozen=True)
class ExpectedBoundaryEvent:
    label: str
    alias_index: int
    phone_index: int
    phone: str
    coarse: str = ""


def expected_events_from_plan(plan: PhonemePlan) -> list[ExpectedBoundaryEvent]:
    out: list[ExpectedBoundaryEvent] = []
    for alias in plan.aliases:
        phones = list(alias.phones)
        if not phones:
            continue
        for phone in phones:
            out.append(_expected("phone_start", phone))
            if str(phone.coarse).startswith("C_"):
                out.append(_expected("consonant_onset", phone))
            elif str(phone.coarse) == "V":
                out.append(_expected("vowel_onset", phone))
                out.append(_expected("vowel_nucleus", phone))
            elif str(phone.coarse) in {"SIL", "BR"}:
                out.append(_expected("silence_boundary", phone))
            out.append(_expected("phone_end", phone))
    return _collapse_adjacent_duplicates(out)


def decode_expected_boundaries(
    score_map: BoundaryFrameScores,
    plan: PhonemePlan,
    *,
    active_start_ms: float | None = None,
    active_end_ms: float | None = None,
    min_score: float = 0.15,
) -> list[DecodedBoundaryEvent]:
    expected = expected_events_from_plan(plan)
    if not expected:
        return []
    times = list(score_map.times_ms)
    if not times:
        return []
    start = 0.0 if active_start_ms is None else max(0.0, float(active_start_ms))
    end = float(plan.duration_ms or 0.0)
    if end <= start:
        end = float(active_end_ms) if active_end_ms is not None else float(times[-1])
    if end <= start:
        end = float(times[-1])
    span = max(1.0, end - start)

    decoded_viterbi = _decode_with_state_viterbi(
        score_map=score_map,
        expected=expected,
        start=start,
        end=end,
        min_score=float(min_score),
    )
    if decoded_viterbi:
        return decoded_viterbi

    # Fallback: old plan-center greedy decode (kept for compatibility).
    step = span / max(1, len(expected))
    decoded: list[DecodedBoundaryEvent] = []
    prev_time = start
    for idx, event in enumerate(expected):
        center = start + (idx + 0.5) * step
        lo = max(start, center - 0.75 * step, prev_time)
        hi = min(end, center + 0.75 * step)
        selected_time, selected_score = _select_best(score_map, event.label, lo, hi)
        fallback = False
        if selected_time is None:
            selected_time = center
            selected_score = 0.0
            fallback = True
        if selected_score < float(min_score):
            fallback = True
        selected_time = max(prev_time, float(selected_time))
        decoded.append(
            DecodedBoundaryEvent(
                expected_label=event.label,
                selected_time_ms=selected_time,
                score=float(selected_score),
                alias_index=int(event.alias_index),
                phone_index=int(event.phone_index),
                phone=str(event.phone),
                source="plan_decoder",
                fallback_used=bool(fallback),
                meta={"coarse": event.coarse},
            )
        )
        prev_time = selected_time + 1.0
    return decoded


def _decode_with_state_viterbi(
    *,
    score_map: BoundaryFrameScores,
    expected: list[ExpectedBoundaryEvent],
    start: float,
    end: float,
    min_score: float,
) -> list[DecodedBoundaryEvent]:
    if not expected:
        return []
    state_scores = score_map.phone_state_scores or {}
    if not state_scores:
        return []
    times = list(score_map.times_ms or [])
    if not times:
        return []
    span = max(1.0, float(end) - float(start))
    step = span / max(1, len(expected))
    centers = [float(start) + (i + 0.5) * step for i in range(len(expected))]
    band = max(35.0, step * 1.35)

    candidate_indices: list[list[int]] = []
    for i in range(len(expected)):
        lo = max(float(start), centers[i] - band)
        hi = min(float(end), centers[i] + band)
        indices = [idx for idx, t in enumerate(times) if lo <= float(t) <= hi]
        if not indices:
            nearest = min(range(len(times)), key=lambda idx: abs(float(times[idx]) - centers[i]))
            indices = [nearest]
        candidate_indices.append(indices)

    local_scores: list[list[float]] = []
    boundary_scores: list[list[float]] = []
    for i, event in enumerate(expected):
        row_local: list[float] = []
        row_boundary: list[float] = []
        for frame_idx in candidate_indices[i]:
            boundary = _event_boundary_value(score_map, event.label, frame_idx)
            state_bonus = _state_bonus(score_map, event, frame_idx)
            quality = _quality_value(score_map, frame_idx)
            center_penalty = abs(float(times[frame_idx]) - centers[i]) / max(step, 1.0)
            row_local.append((1.25 * boundary) + (0.45 * state_bonus) + (0.10 * quality) - (0.12 * center_penalty))
            row_boundary.append(boundary)
        local_scores.append(row_local)
        boundary_scores.append(row_boundary)

    dp = [float(value) for value in local_scores[0]]
    backtrack: list[list[int]] = [[-1 for _ in candidate_indices[0]]]
    for i in range(1, len(expected)):
        cur_scores = [float("-inf")] * len(candidate_indices[i])
        cur_prev = [-1] * len(candidate_indices[i])
        prev_event = expected[i - 1]
        cur_event = expected[i]
        for cur_pos, cur_idx in enumerate(candidate_indices[i]):
            cur_time = float(times[cur_idx])
            best_score = float("-inf")
            best_prev = -1
            for prev_pos, prev_idx in enumerate(candidate_indices[i - 1]):
                prev_time = float(times[prev_idx])
                if cur_time < prev_time:
                    continue
                gap = cur_time - prev_time
                exp_gap = max(1.0, centers[i] - centers[i - 1])
                gap_penalty = 0.08 * abs(gap - exp_gap) / exp_gap
                if gap < _min_gap_ms(prev_event, cur_event):
                    gap_penalty += 0.35
                score = float(dp[prev_pos]) + float(local_scores[i][cur_pos]) - float(gap_penalty)
                if score > best_score:
                    best_score = score
                    best_prev = prev_pos
            cur_scores[cur_pos] = best_score
            cur_prev[cur_pos] = best_prev
        if all(not _is_finite(value) for value in cur_scores):
            return []
        dp = cur_scores
        backtrack.append(cur_prev)

    last = max(range(len(dp)), key=lambda idx: dp[idx])
    if not _is_finite(dp[last]):
        return []
    chosen: list[int] = [0] * len(expected)
    chosen[-1] = last
    for i in range(len(expected) - 1, 0, -1):
        prev = backtrack[i][chosen[i]]
        if prev < 0:
            return []
        chosen[i - 1] = prev

    decoded: list[DecodedBoundaryEvent] = []
    prev_time = float(start)
    for i, event in enumerate(expected):
        candidate_pos = chosen[i]
        frame_idx = candidate_indices[i][candidate_pos]
        time_ms = max(prev_time, float(times[frame_idx]))
        boundary = float(boundary_scores[i][candidate_pos])
        decoded.append(
            DecodedBoundaryEvent(
                expected_label=event.label,
                selected_time_ms=time_ms,
                score=boundary,
                alias_index=int(event.alias_index),
                phone_index=int(event.phone_index),
                phone=str(event.phone),
                source="state_viterbi_decoder",
                fallback_used=bool(boundary < float(min_score)),
                meta={"coarse": event.coarse},
            )
        )
        prev_time = time_ms
    return decoded


def _expected(label: str, phone: PhoneToken) -> ExpectedBoundaryEvent:
    return ExpectedBoundaryEvent(
        label=label,
        alias_index=int(phone.alias_index),
        phone_index=int(phone.index),
        phone=str(phone.text),
        coarse=str(phone.coarse),
    )


def _collapse_adjacent_duplicates(events: list[ExpectedBoundaryEvent]) -> list[ExpectedBoundaryEvent]:
    out: list[ExpectedBoundaryEvent] = []
    for event in events:
        if out and out[-1].label == event.label and out[-1].phone_index == event.phone_index:
            continue
        out.append(event)
    return out


def _select_best(score_map: BoundaryFrameScores, label: str, lo_ms: float, hi_ms: float) -> tuple[float | None, float]:
    values = list(score_map.scores.get(label, []) or [])
    times = list(score_map.times_ms)
    if not values or not times:
        return None, 0.0
    best_time: float | None = None
    best_score = -1.0
    for idx, value in enumerate(values):
        time_ms = float(times[min(idx, len(times) - 1)])
        if time_ms < float(lo_ms) or time_ms > float(hi_ms):
            continue
        score = float(value)
        if score > best_score:
            best_score = score
            best_time = time_ms
    return best_time, max(0.0, best_score)


def _event_boundary_value(score_map: BoundaryFrameScores, label: str, frame_idx: int) -> float:
    values = list(score_map.scores.get(label, []) or [])
    if not values:
        return 0.0
    idx = min(max(0, int(frame_idx)), len(values) - 1)
    return max(0.0, min(1.0, float(values[idx])))


def _quality_value(score_map: BoundaryFrameScores, frame_idx: int) -> float:
    quality = list(score_map.quality_scores or [])
    if not quality:
        return 0.5
    idx = min(max(0, int(frame_idx)), len(quality) - 1)
    return max(0.0, min(1.0, float(quality[idx])))


def _state_bonus(score_map: BoundaryFrameScores, event: ExpectedBoundaryEvent, frame_idx: int) -> float:
    state = _event_state_hint(event)
    if not state:
        return 0.0
    state_scores = score_map.phone_state_scores or {}
    values = list(state_scores.get(state, []) or [])
    if not values:
        return 0.0
    idx = min(max(0, int(frame_idx)), len(values) - 1)
    return max(0.0, min(1.0, float(values[idx])))


def _event_state_hint(event: ExpectedBoundaryEvent) -> str | None:
    label = str(event.label or "")
    coarse = str(event.coarse or "")
    if label in {"vowel_onset", "vowel_nucleus", "vowel_end"}:
        return "vowel"
    if label == "consonant_onset":
        return "consonant"
    if label == "silence_boundary":
        return "silence"
    if label in {"phone_start", "phone_end"}:
        if coarse.startswith("C_"):
            return "consonant"
        if coarse == "V":
            return "vowel"
        if coarse in {"SIL", "BR"}:
            return "silence"
    return None


def _min_gap_ms(prev_event: ExpectedBoundaryEvent, cur_event: ExpectedBoundaryEvent) -> float:
    if (
        int(prev_event.alias_index) == int(cur_event.alias_index)
        and int(prev_event.phone_index) == int(cur_event.phone_index)
    ):
        return 0.0
    return 4.0


def _is_finite(value: float) -> bool:
    return not (value == float("-inf") or value == float("inf") or value != value)


__all__ = ["ExpectedBoundaryEvent", "decode_expected_boundaries", "expected_events_from_plan"]
