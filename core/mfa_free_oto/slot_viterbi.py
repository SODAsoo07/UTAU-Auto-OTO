from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .decode import refine_peak_time
from .types import EVENT_LABELS, FramePosterior, is_vowel_phone

# Minimum spacing (ms) enforced between adjacent slots, keyed by role transition.
_ROLE_GAP_CV_TO_NUCLEUS_MS = 18.0
_ROLE_GAP_NUCLEUS_TO_CV_MS = 22.0
_ROLE_GAP_NUCLEUS_TO_NUCLEUS_MS = 26.0
# A row whose mean slot period is at or below this is treated as "dense", which
# enables extra nucleus peak suppression.
_DENSE_ROW_SLOT_PERIOD_MS = 72.0


@dataclass(frozen=True)
class ExpectedSlot:
    slot_index: int
    phone_index: int
    phone: str
    role: str
    event_label: str


@dataclass(frozen=True)
class SlotAssignment:
    slot_index: int
    phone_index: int
    phone: str
    role: str
    event_label: str
    selected_time_ms: float
    score: float
    frame_index: int
    expected_time_ms: float | None = None


@dataclass(frozen=True)
class SlotViterbiResult:
    assignments: tuple[SlotAssignment, ...]
    path_score: float
    average_score: float
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.assignments) and not any(warning.startswith("hard_") for warning in self.warnings)


@dataclass(frozen=True)
class SlotCandidate:
    frame_index: int
    time_ms: float
    score: float


def expected_cv_slots_from_phones(phones: Sequence[str], *, language: str = "") -> list[ExpectedSlot]:
    slots: list[ExpectedSlot] = []
    previous_phone: str | None = None
    previous_was_consonant = False
    for phone_index, raw_phone in enumerate(phones):
        phone = str(raw_phone).strip()
        if not phone:
            continue
        is_vowel = is_vowel_phone(phone, language)
        if is_vowel and previous_phone is not None and previous_was_consonant:
            slots.append(
                ExpectedSlot(
                    slot_index=len(slots),
                    phone_index=phone_index,
                    phone=phone,
                    role="cv_boundary",
                    event_label="cv_boundary",
                )
            )
        if is_vowel:
            slots.append(
                ExpectedSlot(
                    slot_index=len(slots),
                    phone_index=phone_index,
                    phone=phone,
                    role="vowel_nucleus",
                    event_label="vowel_nucleus",
                )
            )
        previous_phone = phone
        previous_was_consonant = not is_vowel
    return slots


def assign_slots_viterbi(
    posterior: FramePosterior,
    expected_phones: Sequence[str] | None = None,
    *,
    expected_slots: Sequence[ExpectedSlot] | None = None,
    min_event_score: float = 0.03,
    top_k_per_slot: int = 32,
    min_gap_ms: float = 10.0,
    same_phone_min_gap_ms: float = 24.0,
    expected_time_weight: float = 0.20,
    transition_weight: float = 0.0003,
    low_score_warning: float = 0.20,
    long_row_segment_size: int = 10,
    segment_overlap_slots: int = 1,
    local_window_slots: float = 2.6,
    nucleus_min_peak_distance_ms: float = 26.0,
    language: str = "",
) -> SlotViterbiResult:
    slots = (
        list(expected_slots)
        if expected_slots is not None
        else expected_cv_slots_from_phones(expected_phones or [], language=language)
    )
    warnings: list[str] = []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return SlotViterbiResult(assignments=(), path_score=-1e9, average_score=0.0, warnings=("hard_empty_posterior",))
    if not slots:
        return SlotViterbiResult(assignments=(), path_score=0.0, average_score=0.0, warnings=("hard_no_expected_slots",))

    duration_ms = float(times[-1]) if times.size == 1 else float(max(times[-1], times[1] - times[0]))
    expected_times = _expected_slot_times(slots, duration_ms)
    selected_by_slot: dict[int, SlotAssignment] = {}
    path_score_total = 0.0
    segment_bounds = _segment_slot_ranges(
        len(slots),
        max(2, int(long_row_segment_size)),
        overlap=max(0, int(segment_overlap_slots)),
    )
    if len(segment_bounds) > 1:
        warnings.append(f"segmented_decode:{len(segment_bounds)} overlap={max(0, int(segment_overlap_slots))}")
    slot_period_ms = float(duration_ms) / float(max(1, len(slots) + 1))
    window_ms = max(45.0, slot_period_ms * float(max(1.6, local_window_slots)))
    dense_row = bool(slot_period_ms <= _DENSE_ROW_SLOT_PERIOD_MS)
    previous_assignment: SlotAssignment | None = None
    previous_slot: ExpectedSlot | None = None
    for seg_idx, (start, end) in enumerate(segment_bounds):
        seg_slots = slots[start:end]
        candidate_steps: list[list[SlotCandidate]] = []
        for local_idx, slot in enumerate(seg_slots):
            expected_time = expected_times.get(slot.slot_index)
            min_time_ms: float | None = None
            if local_idx == 0 and previous_assignment is not None and previous_slot is not None:
                required = _required_gap_ms(
                    previous_slot,
                    slot,
                    min_gap_ms=min_gap_ms,
                    same_phone_min_gap_ms=same_phone_min_gap_ms,
                )
                min_time_ms = float(previous_assignment.selected_time_ms + required)
            candidates = _slot_candidates(
                posterior,
                slot,
                min_event_score=min_event_score,
                top_k=top_k_per_slot,
                expected_time_ms=expected_time,
                expected_time_weight=expected_time_weight,
                min_time_ms=min_time_ms,
                max_time_ms=None,
                window_ms=window_ms,
                nucleus_min_peak_distance_ms=nucleus_min_peak_distance_ms,
                dense_row=dense_row,
            )
            candidate_steps.append(candidates)
        if any(not candidates for candidates in candidate_steps):
            missing = [str(slot.slot_index) for slot, candidates in zip(seg_slots, candidate_steps) if not candidates]
            return SlotViterbiResult(
                assignments=(),
                path_score=-1e9,
                average_score=0.0,
                warnings=(f"hard_missing_candidates:{','.join(missing)}",),
            )
        selected, seg_path_score = _solve_viterbi_path(
            seg_slots,
            candidate_steps,
            min_gap_ms=min_gap_ms,
            same_phone_min_gap_ms=same_phone_min_gap_ms,
            transition_weight=transition_weight,
        )
        if not selected:
            return SlotViterbiResult(assignments=(), path_score=-1e9, average_score=0.0, warnings=("hard_no_monotonic_path",))
        path_score_total += float(seg_path_score)
        seg_assignments: list[SlotAssignment] = []
        for slot, candidates, selected_idx in zip(seg_slots, candidate_steps, selected):
            candidate = candidates[selected_idx]
            assigned = SlotAssignment(
                slot_index=slot.slot_index,
                phone_index=slot.phone_index,
                phone=slot.phone,
                role=slot.role,
                event_label=slot.event_label,
                selected_time_ms=candidate.time_ms,
                score=candidate.score,
                frame_index=candidate.frame_index,
                expected_time_ms=expected_times.get(slot.slot_index),
            )
            seg_assignments.append(assigned)
        for assigned in seg_assignments:
            slot = slots[assigned.slot_index]
            existing = selected_by_slot.get(assigned.slot_index)
            if existing is None:
                selected_by_slot[assigned.slot_index] = assigned
                continue
            chosen = _select_overlap_assignment(
                existing,
                assigned,
                expected_time_ms=expected_times.get(assigned.slot_index),
                slot=slot,
                slots=slots,
                selected_by_slot=selected_by_slot,
                min_gap_ms=min_gap_ms,
                same_phone_min_gap_ms=same_phone_min_gap_ms,
            )
            selected_by_slot[assigned.slot_index] = chosen
        if end - 1 in selected_by_slot:
            previous_assignment = selected_by_slot[end - 1]
            previous_slot = slots[end - 1]

    assignments: list[SlotAssignment] = []
    for slot_idx in range(len(slots)):
        assigned = selected_by_slot.get(slot_idx)
        if assigned is None:
            return SlotViterbiResult(assignments=(), path_score=-1e9, average_score=0.0, warnings=("hard_incomplete_stitch",))
        assignments.append(assigned)

    raw_scores: list[float] = []
    for assigned in assignments:
        raw_scores.append(float(assigned.score))
        if float(assigned.score) < low_score_warning:
            warnings.append(f"low_score_slot:{assigned.slot_index}:{assigned.score:.3f}")
        _append_acoustic_warnings(
            slots[assigned.slot_index],
            SlotCandidate(frame_index=assigned.frame_index, time_ms=assigned.selected_time_ms, score=assigned.score),
            posterior,
            warnings,
        )
    _append_gap_warnings(assignments, warnings, min_gap_ms=min_gap_ms, same_phone_min_gap_ms=same_phone_min_gap_ms)
    return SlotViterbiResult(
        assignments=tuple(assignments),
        path_score=float(path_score_total),
        average_score=float(np.mean(np.asarray(raw_scores, dtype=np.float32))) if raw_scores else 0.0,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def slot_assignments_to_decoded_events(result: SlotViterbiResult):
    from .types import DecodedEvent

    return [
        DecodedEvent(
            label=assignment.event_label,
            selected_time_ms=assignment.selected_time_ms,
            score=assignment.score,
            expected_phone=assignment.phone,
            frame_index=assignment.frame_index,
        )
        for assignment in result.assignments
    ]


def _slot_candidates(
    posterior: FramePosterior,
    slot: ExpectedSlot,
    *,
    min_event_score: float,
    top_k: int,
    expected_time_ms: float | None,
    expected_time_weight: float,
    min_time_ms: float | None,
    max_time_ms: float | None,
    window_ms: float,
    nucleus_min_peak_distance_ms: float,
    dense_row: bool,
) -> list[SlotCandidate]:
    if slot.event_label not in EVENT_LABELS:
        return []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    event_values = np.asarray(posterior.event_scores.get(slot.event_label, []), dtype=np.float32)
    if event_values.shape[0] != times.shape[0]:
        return []
    class_prior = _slot_class_prior(posterior, slot)
    acoustic_prior = _slot_acoustic_prior(posterior, slot)
    values = np.clip(event_values, 0.0, 1.0) * 0.62 + class_prior * 0.24 + acoustic_prior * 0.14
    peak_indices = _local_peak_indices(values, min_score=min_event_score)
    if slot.role == "vowel_nucleus" and dense_row:
        peak_indices = _suppress_peak_indices(
            peak_indices,
            values,
            times,
            min_distance_ms=max(12.0, float(nucleus_min_peak_distance_ms * 0.78)),
        )
    if slot.role == "vowel_nucleus":
        peak_indices = _suppress_peak_indices(
            peak_indices,
            values,
            times,
            min_distance_ms=max(12.0, float(nucleus_min_peak_distance_ms)),
        )
    if not peak_indices:
        best_event_idx = int(np.argmax(event_values))
        if float(event_values[best_event_idx]) >= float(min_event_score):
            peak_indices = [best_event_idx]
        else:
            return []
    duration_ms = float(max(times[-1], 1.0))
    candidates: list[SlotCandidate] = []
    for idx in peak_indices:
        score = float(values[idx])
        if float(event_values[idx]) < float(min_event_score):
            continue
        if expected_time_ms is not None and duration_ms > 0.0:
            distance = abs(float(times[idx]) - expected_time_ms) / duration_ms
            score -= expected_time_weight * distance
            if abs(float(times[idx]) - expected_time_ms) > float(window_ms):
                score -= 0.22
        if min_time_ms is not None and float(times[idx]) + 1e-5 < float(min_time_ms):
            continue
        if max_time_ms is not None and float(times[idx]) - 1e-5 > float(max_time_ms):
            continue
        candidates.append(
            SlotCandidate(frame_index=idx, time_ms=refine_peak_time(times, values, idx), score=score)
        )
    candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]
    return sorted(candidates, key=lambda candidate: candidate.time_ms)


def _slot_class_prior(posterior: FramePosterior, slot: ExpectedSlot) -> np.ndarray:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    vowel = np.asarray(posterior.class_probs.get("vowel", []), dtype=np.float32)
    consonant = np.asarray(posterior.class_probs.get("consonant", []), dtype=np.float32)
    if vowel.shape[0] != times.shape[0]:
        vowel = np.zeros_like(times)
    if consonant.shape[0] != times.shape[0]:
        consonant = np.zeros_like(times)
    if slot.role == "cv_boundary":
        left_consonant = _left_context_max(consonant, frames=4)
        return np.clip(0.55 * vowel + 0.45 * left_consonant, 0.0, 1.0)
    if slot.role == "vowel_nucleus":
        return np.clip(vowel, 0.0, 1.0)
    return np.zeros_like(times)


def _slot_acoustic_prior(posterior: FramePosterior, slot: ExpectedSlot) -> np.ndarray:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    transition = _score_track(posterior, "transition_likelihood", times)
    flux = _score_track(posterior, "flux_likelihood", times)
    sonorant = _score_track(posterior, "sonorant_onset_likelihood", times)
    voicing = _score_track(posterior, "voicing", times)
    nucleus = np.maximum(_score_track(posterior, "nucleus_likelihood", times), _score_track(posterior, "world_nucleus", times))
    periodicity = _score_track(posterior, "world_periodicity", times)
    spectral_stability = _score_track(posterior, "world_spectral_stability", times)
    silence = _score_track(posterior, "silence_likelihood", times)
    non_silence = 1.0 - silence
    if slot.role == "cv_boundary":
        return np.clip(
            0.42 * transition + 0.24 * flux + 0.18 * non_silence + 0.16 * sonorant,
            0.0,
            1.0,
        )
    if slot.role == "vowel_nucleus":
        return np.clip(
            (0.34 * voicing)
            + (0.34 * nucleus)
            + (0.18 * periodicity)
            + (0.08 * spectral_stability)
            + (0.06 * non_silence),
            0.0,
            1.0,
        )
    return np.zeros_like(times)


def _score_track(posterior: FramePosterior, name: str, times: np.ndarray) -> np.ndarray:
    values = np.asarray(posterior.acoustic_scores.get(name, []), dtype=np.float32)
    if values.shape[0] != times.shape[0]:
        return np.zeros_like(times)
    return np.clip(values, 0.0, 1.0)


def _append_acoustic_warnings(
    slot: ExpectedSlot,
    candidate: SlotCandidate,
    posterior: FramePosterior,
    warnings: list[str],
) -> None:
    idx = candidate.frame_index
    silence = _track_value(posterior, "silence_likelihood", idx)
    transition = _track_value(posterior, "transition_likelihood", idx)
    voicing = _track_value(posterior, "voicing", idx)
    if silence is not None and silence > 0.75:
        warnings.append(f"low_energy_slot:{slot.slot_index}:{silence:.3f}")
    if slot.role == "cv_boundary" and transition is not None and transition < 0.20:
        warnings.append(f"weak_flux_slot:{slot.slot_index}:{transition:.3f}")
    if slot.role == "vowel_nucleus" and voicing is not None and voicing < 0.20:
        warnings.append(f"weak_voicing_slot:{slot.slot_index}:{voicing:.3f}")


def _track_value(posterior: FramePosterior, name: str, idx: int) -> float | None:
    values = posterior.acoustic_scores.get(name)
    if values is None or idx < 0 or idx >= len(values):
        return None
    return float(values[idx])


def _left_context_max(values: np.ndarray, *, frames: int) -> np.ndarray:
    out = np.zeros_like(values)
    for idx in range(values.shape[0]):
        start = max(0, idx - frames)
        out[idx] = float(np.max(values[start : idx + 1])) if idx + 1 > start else float(values[idx])
    return out


def _local_peak_indices(values: np.ndarray, *, min_score: float) -> list[int]:
    peaks: list[int] = []
    for idx, value in enumerate(values):
        if float(value) < min_score:
            continue
        left = values[idx - 1] if idx > 0 else -1.0
        right = values[idx + 1] if idx + 1 < values.shape[0] else -1.0
        if value >= left and value >= right:
            peaks.append(idx)
    return peaks


def _expected_slot_times(slots: Sequence[ExpectedSlot], duration_ms: float) -> Mapping[int, float]:
    if not slots:
        return {}
    step = float(duration_ms) / float(len(slots) + 1)
    return {slot.slot_index: step * float(idx + 1) for idx, slot in enumerate(slots)}


def _append_gap_warnings(
    assignments: Sequence[SlotAssignment],
    warnings: list[str],
    *,
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
) -> None:
    for prev, cur in zip(assignments[:-1], assignments[1:]):
        gap = cur.selected_time_ms - prev.selected_time_ms
        required = _required_gap_ms(
            ExpectedSlot(
                slot_index=prev.slot_index,
                phone_index=prev.phone_index,
                phone=prev.phone,
                role=prev.role,
                event_label=prev.event_label,
            ),
            ExpectedSlot(
                slot_index=cur.slot_index,
                phone_index=cur.phone_index,
                phone=cur.phone,
                role=cur.role,
                event_label=cur.event_label,
            ),
            min_gap_ms=min_gap_ms,
            same_phone_min_gap_ms=same_phone_min_gap_ms,
        )
        if gap < required + 1e-5:
            warnings.append(f"tight_gap:{prev.slot_index}->{cur.slot_index}:{gap:.1f}ms")


def _required_gap_ms(
    prev_slot: ExpectedSlot,
    cur_slot: ExpectedSlot,
    *,
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
) -> float:
    required = same_phone_min_gap_ms if prev_slot.phone_index == cur_slot.phone_index else min_gap_ms
    if prev_slot.role == "cv_boundary" and cur_slot.role == "vowel_nucleus":
        required = max(required, _ROLE_GAP_CV_TO_NUCLEUS_MS)
    elif prev_slot.role == "vowel_nucleus" and cur_slot.role == "cv_boundary":
        required = max(required, _ROLE_GAP_NUCLEUS_TO_CV_MS)
    elif prev_slot.role == "vowel_nucleus" and cur_slot.role == "vowel_nucleus":
        required = max(required, _ROLE_GAP_NUCLEUS_TO_NUCLEUS_MS)
    return float(required)


def _segment_slot_ranges(total: int, segment_size: int, *, overlap: int) -> list[tuple[int, int]]:
    if total <= 0:
        return []
    if total <= segment_size:
        return [(0, total)]
    out: list[tuple[int, int]] = []
    step = max(1, int(segment_size) - max(0, int(overlap)))
    start = 0
    while start < total:
        end = min(total, start + segment_size)
        out.append((start, end))
        if end >= total:
            break
        start += step
    return out


def _solve_viterbi_path(
    slots: Sequence[ExpectedSlot],
    candidate_steps: Sequence[Sequence[SlotCandidate]],
    *,
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
    transition_weight: float,
) -> tuple[list[int], float]:
    dp: list[list[tuple[float, int | None]]] = []
    for step_idx, candidates in enumerate(candidate_steps):
        row: list[tuple[float, int | None]] = []
        slot = slots[step_idx]
        for candidate in candidates:
            if step_idx == 0:
                row.append((candidate.score, None))
                continue
            best_score = -1e9
            best_prev: int | None = None
            prev_slot = slots[step_idx - 1]
            required_gap = _required_gap_ms(
                prev_slot,
                slot,
                min_gap_ms=min_gap_ms,
                same_phone_min_gap_ms=same_phone_min_gap_ms,
            )
            for prev_idx, prev_candidate in enumerate(candidate_steps[step_idx - 1]):
                prev_score, _ = dp[step_idx - 1][prev_idx]
                gap = candidate.time_ms - prev_candidate.time_ms
                if gap + 1e-5 < required_gap:
                    continue
                transition_penalty = transition_weight * max(0.0, gap)
                score = prev_score + candidate.score - transition_penalty
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            row.append((best_score, best_prev))
        dp.append(row)

    if not dp:
        return [], -1e9
    final_scores = [score for score, _ in dp[-1]]
    final_idx = int(np.argmax(np.asarray(final_scores, dtype=np.float32)))
    path_score = float(final_scores[final_idx])
    if path_score <= -1e8:
        return [], path_score
    selected: list[int] = []
    idx = final_idx
    for step_idx in range(len(candidate_steps) - 1, -1, -1):
        selected.append(idx)
        _score, prev_idx = dp[step_idx][idx]
        if prev_idx is None:
            break
        idx = prev_idx
    selected.reverse()
    if len(selected) != len(candidate_steps):
        return [], -1e9
    return selected, path_score


def _suppress_peak_indices(
    peak_indices: Sequence[int],
    values: np.ndarray,
    times: np.ndarray,
    *,
    min_distance_ms: float,
) -> list[int]:
    if not peak_indices:
        return []
    ordered = sorted(peak_indices, key=lambda idx: float(values[idx]), reverse=True)
    selected: list[int] = []
    for idx in ordered:
        t = float(times[idx])
        if any(abs(t - float(times[prev])) < min_distance_ms for prev in selected):
            continue
        selected.append(idx)
    return sorted(selected)


def _select_overlap_assignment(
    current: SlotAssignment,
    incoming: SlotAssignment,
    *,
    expected_time_ms: float | None,
    slot: ExpectedSlot,
    slots: Sequence[ExpectedSlot],
    selected_by_slot: Mapping[int, SlotAssignment],
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
) -> SlotAssignment:
    current_ok = _is_feasible_overlap_assignment(
        current,
        slot=slot,
        slots=slots,
        selected_by_slot=selected_by_slot,
        min_gap_ms=min_gap_ms,
        same_phone_min_gap_ms=same_phone_min_gap_ms,
    )
    incoming_ok = _is_feasible_overlap_assignment(
        incoming,
        slot=slot,
        slots=slots,
        selected_by_slot=selected_by_slot,
        min_gap_ms=min_gap_ms,
        same_phone_min_gap_ms=same_phone_min_gap_ms,
    )
    if incoming_ok and not current_ok:
        return incoming
    if current_ok and not incoming_ok:
        return current
    exp = float(expected_time_ms) if expected_time_ms is not None else None
    if slot.role == "cv_boundary":
        if exp is not None:
            cur_dist = abs(float(current.selected_time_ms) - exp)
            inc_dist = abs(float(incoming.selected_time_ms) - exp)
            if inc_dist > cur_dist + 6.0:
                return current
        if float(incoming.score) < float(current.score) + 0.08:
            return current
    current_cost = _assignment_cost(current, expected_time_ms=exp)
    incoming_cost = _assignment_cost(incoming, expected_time_ms=exp)
    return incoming if incoming_cost < current_cost else current


def _assignment_cost(assignment: SlotAssignment, *, expected_time_ms: float | None) -> float:
    score_term = 1.0 - float(max(0.0, min(1.0, assignment.score)))
    if expected_time_ms is None:
        return score_term
    time_term = abs(float(assignment.selected_time_ms) - float(expected_time_ms)) * 0.0022
    return score_term + time_term


def _is_feasible_overlap_assignment(
    assignment: SlotAssignment,
    *,
    slot: ExpectedSlot,
    slots: Sequence[ExpectedSlot],
    selected_by_slot: Mapping[int, SlotAssignment],
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
) -> bool:
    prev_assigned = selected_by_slot.get(slot.slot_index - 1)
    if prev_assigned is not None:
        prev_slot = slots[slot.slot_index - 1]
        required = _required_gap_ms(prev_slot, slot, min_gap_ms=min_gap_ms, same_phone_min_gap_ms=same_phone_min_gap_ms)
        if float(assignment.selected_time_ms) + 1e-5 < float(prev_assigned.selected_time_ms + required):
            return False
    next_assigned = selected_by_slot.get(slot.slot_index + 1)
    if next_assigned is not None:
        next_slot = slots[slot.slot_index + 1]
        required = _required_gap_ms(slot, next_slot, min_gap_ms=min_gap_ms, same_phone_min_gap_ms=same_phone_min_gap_ms)
        if float(next_assigned.selected_time_ms) + 1e-5 < float(assignment.selected_time_ms + required):
            return False
    return True
