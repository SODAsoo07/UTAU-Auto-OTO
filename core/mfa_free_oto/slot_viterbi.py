from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .types import EVENT_LABELS, FramePosterior, is_vowel_phone


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


def expected_cv_slots_from_phones(phones: Sequence[str]) -> list[ExpectedSlot]:
    slots: list[ExpectedSlot] = []
    previous_phone: str | None = None
    previous_was_consonant = False
    for phone_index, raw_phone in enumerate(phones):
        phone = str(raw_phone).strip()
        if not phone:
            continue
        is_vowel = is_vowel_phone(phone)
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
    min_gap_ms: float = 8.0,
    same_phone_min_gap_ms: float = 20.0,
    expected_time_weight: float = 0.20,
    transition_weight: float = 0.0003,
    low_score_warning: float = 0.20,
) -> SlotViterbiResult:
    slots = list(expected_slots) if expected_slots is not None else expected_cv_slots_from_phones(expected_phones or [])
    warnings: list[str] = []
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return SlotViterbiResult(assignments=(), path_score=-1e9, average_score=0.0, warnings=("hard_empty_posterior",))
    if not slots:
        return SlotViterbiResult(assignments=(), path_score=0.0, average_score=0.0, warnings=("hard_no_expected_slots",))

    duration_ms = float(times[-1]) if times.size == 1 else float(max(times[-1], times[1] - times[0]))
    expected_times = _expected_slot_times(slots, duration_ms)
    candidate_steps = [
        _slot_candidates(
            posterior,
            slot,
            min_event_score=min_event_score,
            top_k=top_k_per_slot,
            expected_time_ms=expected_times.get(slot.slot_index),
            expected_time_weight=expected_time_weight,
        )
        for slot in slots
    ]
    if any(not candidates for candidates in candidate_steps):
        missing = [str(slot.slot_index) for slot, candidates in zip(slots, candidate_steps) if not candidates]
        return (
            SlotViterbiResult(
                assignments=(),
                path_score=-1e9,
                average_score=0.0,
                warnings=(f"hard_missing_candidates:{','.join(missing)}",),
            )
        )

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
            required_gap = same_phone_min_gap_ms if prev_slot.phone_index == slot.phone_index else min_gap_ms
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

    final_scores = [score for score, _ in dp[-1]]
    final_idx = int(np.argmax(np.asarray(final_scores, dtype=np.float32)))
    path_score = float(final_scores[final_idx])
    if path_score <= -1e8:
        return SlotViterbiResult(assignments=(), path_score=path_score, average_score=0.0, warnings=("hard_no_monotonic_path",))

    selected: list[int] = []
    idx = final_idx
    for step_idx in range(len(candidate_steps) - 1, -1, -1):
        selected.append(idx)
        _score, prev_idx = dp[step_idx][idx]
        if prev_idx is None:
            break
        idx = prev_idx
    selected.reverse()
    if len(selected) != len(slots):
        return SlotViterbiResult(assignments=(), path_score=path_score, average_score=0.0, warnings=("hard_incomplete_backtrace",))

    assignments: list[SlotAssignment] = []
    raw_scores: list[float] = []
    for slot, candidates, selected_idx in zip(slots, candidate_steps, selected):
        candidate = candidates[selected_idx]
        raw_scores.append(candidate.score)
        if candidate.score < low_score_warning:
            warnings.append(f"low_score_slot:{slot.slot_index}:{candidate.score:.3f}")
        _append_acoustic_warnings(slot, candidate, posterior, warnings)
        assignments.append(
            SlotAssignment(
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
        )
    _append_gap_warnings(assignments, warnings, min_gap_ms=min_gap_ms, same_phone_min_gap_ms=same_phone_min_gap_ms)
    return SlotViterbiResult(
        assignments=tuple(assignments),
        path_score=path_score,
        average_score=float(np.mean(np.asarray(raw_scores, dtype=np.float32))) if raw_scores else 0.0,
        warnings=tuple(warnings),
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
    peak_indices = _local_peak_indices(event_values, min_score=min_event_score)
    if not peak_indices:
        peak_indices = [int(np.argmax(values))]
    duration_ms = float(max(times[-1], 1.0))
    candidates: list[SlotCandidate] = []
    for idx in peak_indices:
        score = float(values[idx])
        if expected_time_ms is not None and duration_ms > 0.0:
            distance = abs(float(times[idx]) - expected_time_ms) / duration_ms
            score -= expected_time_weight * distance
        candidates.append(SlotCandidate(frame_index=idx, time_ms=float(times[idx]), score=score))
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
    voicing = _score_track(posterior, "voicing", times)
    silence = _score_track(posterior, "silence_likelihood", times)
    non_silence = 1.0 - silence
    if slot.role == "cv_boundary":
        return np.clip(0.50 * transition + 0.30 * flux + 0.20 * non_silence, 0.0, 1.0)
    if slot.role == "vowel_nucleus":
        return np.clip(0.55 * voicing + 0.45 * non_silence, 0.0, 1.0)
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
        required = same_phone_min_gap_ms if prev.phone_index == cur.phone_index else min_gap_ms
        if gap < required + 1e-5:
            warnings.append(f"tight_gap:{prev.slot_index}->{cur.slot_index}:{gap:.1f}ms")
