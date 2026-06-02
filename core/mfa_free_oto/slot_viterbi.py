from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .decode import refine_peak_time
from .types import EVENT_LABELS, FramePosterior, is_vowel_phone

# Minimum spacing (ms) enforced between adjacent slots, keyed by role transition.
_ROLE_GAP_CV_TO_NUCLEUS_MS = 18.0
_ROLE_GAP_NUCLEUS_TO_CV_MS = 22.0
_ROLE_GAP_NUCLEUS_TO_NUCLEUS_MS = 26.0
_ROLE_GAP_CONSECUTIVE_CV_RATIO = 0.45
_ROLE_GAP_CONSECUTIVE_CV_MIN_MS = 80.0
_ROLE_GAP_CONSECUTIVE_CV_MAX_MS = 220.0
# A row whose mean slot period is at or below this is treated as "dense", which
# enables extra nucleus peak suppression.
_DENSE_ROW_SLOT_PERIOD_MS = 72.0
_SONORANT_PHONES = frozenset({"m", "n", "ng", "ny", "r", "l", "w", "y", "j"})
_NASAL_PHONES = frozenset({"m", "n", "ng", "ny", "my"})
_LIQUID_PHONES = frozenset({"r", "l", "ry"})
_KOREAN_OBSTRUENT_VC_EXPECTED_SHIFT_RATIO_BY_PHONE = {
    "b": 0.82,
    "d": 0.74,
    "h": 0.48,
    "p": -0.15,
    "k": -0.10,
}
_JAPANESE_SLOT_DURATION_PRIOR_DEFAULT_WEIGHT = 0.0
_JAPANESE_SLOT_DURATION_PRIOR_MIN_DISTANCE = 0.85
_JAPANESE_SLOT_DURATION_PRIOR_MAX_PENALTY = 0.28


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
    expected_time_ms: float | None = None


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
    cv_local_refine_enabled: bool = True,
    cv_local_refine_window_ms: float = 55.0,
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
    active_start_ms, active_end_ms = _expected_time_window(posterior, times, duration_ms, language=language)
    expected_times = _expected_slot_times(slots, duration_ms, start_ms=active_start_ms, end_ms=active_end_ms)
    selected_by_slot: dict[int, SlotAssignment] = {}
    path_score_total = 0.0
    segment_bounds = _segment_slot_ranges(
        len(slots),
        max(2, int(long_row_segment_size)),
        overlap=max(0, int(segment_overlap_slots)),
    )
    if len(segment_bounds) > 1:
        warnings.append(f"segmented_decode:{len(segment_bounds)} overlap={max(0, int(segment_overlap_slots))}")
    slot_period_ms = float(max(1.0, active_end_ms - active_start_ms)) / float(max(1, len(slots)))
    window_ms = max(45.0, slot_period_ms * float(max(1.6, local_window_slots)))
    dense_row = bool(slot_period_ms <= _DENSE_ROW_SLOT_PERIOD_MS)
    previous_assignment: SlotAssignment | None = None
    previous_slot: ExpectedSlot | None = None
    for seg_idx, (start, end) in enumerate(segment_bounds):
        seg_slots = slots[start:end]
        candidate_steps: list[list[SlotCandidate]] = []
        for local_idx, slot in enumerate(seg_slots):
            expected_time = _candidate_expected_time_ms(
                slot,
                expected_times.get(slot.slot_index),
                language=language,
                slot_period_ms=slot_period_ms,
                slot_count=len(slots),
            )
            min_time_ms: float | None = None
            if (
                local_idx == 0
                and previous_assignment is not None
                and previous_slot is not None
                and slot.slot_index > previous_slot.slot_index
            ):
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
                slot_period_ms=slot_period_ms,
                slot_count=len(slots),
                language=language,
                min_time_ms=min_time_ms,
                max_time_ms=None,
                window_ms=window_ms,
                expected_time_hard_window_ms=_expected_time_hard_window_ms(
                    slot,
                    language=language,
                    slot_period_ms=slot_period_ms,
                    slot_count=len(slots),
                ),
                expected_time_fallback_enabled=_expected_time_fallback_enabled(
                    slot,
                    language=language,
                    slot_count=len(slots),
                ),
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
            slot_period_ms=slot_period_ms,
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
                expected_time_ms=_candidate_expected_time_ms(
                    slot,
                    expected_times.get(slot.slot_index),
                    language=language,
                    slot_period_ms=slot_period_ms,
                    slot_count=len(slots),
                ),
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
                expected_time_ms=_candidate_expected_time_ms(
                    slot,
                    expected_times.get(assigned.slot_index),
                    language=language,
                    slot_period_ms=slot_period_ms,
                    slot_count=len(slots),
                ),
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
    if cv_local_refine_enabled and _should_run_cv_local_refine(assignments, warnings):
        refined_assignments = _refine_cv_assignments_locally(
            assignments=assignments,
            slots=slots,
            posterior=posterior,
            expected_times=expected_times,
            min_gap_ms=min_gap_ms,
            same_phone_min_gap_ms=same_phone_min_gap_ms,
            window_ms=max(20.0, float(cv_local_refine_window_ms)),
        )
        if refined_assignments != assignments:
            assignments = refined_assignments
            warnings.append("cv_local_refined")
            raw_scores = [float(item.score) for item in assignments]
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
    slot_period_ms: float,
    slot_count: int,
    language: str,
    min_time_ms: float | None,
    max_time_ms: float | None,
    window_ms: float,
    expected_time_hard_window_ms: float | None,
    expected_time_fallback_enabled: bool,
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
    sonorant_phone_change = _is_sonorant_phone_change_slot(slot)
    vowel_transition_slot = str(slot.role) == "vv"
    if sonorant_phone_change:
        values = np.clip(
            (0.25 * np.clip(event_values, 0.0, 1.0)) + (0.15 * class_prior) + (0.60 * acoustic_prior),
            0.0,
            1.0,
        )
    elif vowel_transition_slot:
        values = np.clip(
            (0.24 * np.clip(event_values, 0.0, 1.0)) + (0.16 * class_prior) + (0.60 * acoustic_prior),
            0.0,
            1.0,
        )
    else:
        values = np.clip(event_values, 0.0, 1.0) * 0.62 + class_prior * 0.24 + acoustic_prior * 0.14
    peak_indices = _local_peak_indices(values, min_score=min_event_score)
    if _is_vowel_nucleus_slot(slot) and dense_row:
        peak_indices = _suppress_peak_indices(
            peak_indices,
            values,
            times,
            min_distance_ms=max(12.0, float(nucleus_min_peak_distance_ms * 0.78)),
        )
    if _is_vowel_nucleus_slot(slot):
        peak_indices = _suppress_peak_indices(
            peak_indices,
            values,
            times,
            min_distance_ms=max(12.0, float(nucleus_min_peak_distance_ms)),
        )
    fallback_indices: set[int] = set()
    if expected_time_ms is not None and expected_time_hard_window_ms is not None:
        hard_window = max(1.0, float(expected_time_hard_window_ms))
        in_window = [
            idx
            for idx in peak_indices
            if abs(float(times[idx]) - float(expected_time_ms)) <= hard_window
        ]
        if in_window:
            peak_indices = in_window
        elif expected_time_fallback_enabled and times.size > 0:
            fallback_idx = int(np.argmin(np.abs(times - float(expected_time_ms))))
            peak_indices = [fallback_idx]
            fallback_indices.add(fallback_idx)
    if not peak_indices:
        gate_values = values if (sonorant_phone_change or vowel_transition_slot) else event_values
        best_event_idx = int(np.argmax(gate_values))
        if float(gate_values[best_event_idx]) >= float(min_event_score):
            peak_indices = [best_event_idx]
        else:
            return []
    duration_ms = float(max(times[-1], 1.0))
    duration_prior_weight = _slot_duration_prior_weight(
        slot,
        language=language,
        slot_count=slot_count,
    )
    candidates: list[SlotCandidate] = []
    if fallback_indices:
        idx = next(iter(fallback_indices))
        if min_time_ms is not None and float(times[idx]) + 1e-5 < float(min_time_ms):
            return []
        if max_time_ms is not None and float(times[idx]) - 1e-5 > float(max_time_ms):
            return []
        score = float(values[idx])
        if expected_time_ms is not None and duration_ms > 0.0:
            score -= expected_time_weight * (abs(float(times[idx]) - expected_time_ms) / duration_ms)
            score -= _slot_duration_prior_penalty(
                float(times[idx]),
                expected_time_ms=float(expected_time_ms),
                slot_period_ms=slot_period_ms,
                weight=duration_prior_weight,
            )
        return [
            SlotCandidate(
                frame_index=idx,
                time_ms=float(times[idx]),
                score=max(score, float(min_event_score)),
                expected_time_ms=float(expected_time_ms) if expected_time_ms is not None else None,
            )
        ]
    for idx in peak_indices:
        score = float(values[idx])
        gate_score = (
            float(values[idx])
            if (sonorant_phone_change or vowel_transition_slot or str(slot.role or "").strip().lower() == "v")
            else float(event_values[idx])
        )
        is_expected_fallback = idx in fallback_indices
        if is_expected_fallback:
            if expected_time_ms is not None and duration_ms > 0.0:
                distance = abs(float(times[idx]) - expected_time_ms) / duration_ms
                score -= expected_time_weight * distance
            if min_time_ms is not None and float(times[idx]) + 1e-5 < float(min_time_ms):
                continue
            if max_time_ms is not None and float(times[idx]) - 1e-5 > float(max_time_ms):
                continue
            candidates.append(
                SlotCandidate(frame_index=idx, time_ms=float(times[idx]), score=max(score, float(min_event_score)))
            )
            continue
        if gate_score < float(min_event_score) and not is_expected_fallback:
            continue
        if expected_time_ms is not None and duration_ms > 0.0:
            distance = abs(float(times[idx]) - expected_time_ms) / duration_ms
            score -= expected_time_weight * distance
            score -= _slot_duration_prior_penalty(
                float(times[idx]),
                expected_time_ms=float(expected_time_ms),
                slot_period_ms=slot_period_ms,
                weight=duration_prior_weight,
            )
            if abs(float(times[idx]) - expected_time_ms) > float(window_ms):
                score -= 0.22
        if min_time_ms is not None and float(times[idx]) + 1e-5 < float(min_time_ms):
            continue
        if max_time_ms is not None and float(times[idx]) - 1e-5 > float(max_time_ms):
            continue
        candidates.append(
            SlotCandidate(
                frame_index=idx,
                time_ms=refine_peak_time(times, values, idx),
                score=score,
                expected_time_ms=float(expected_time_ms) if expected_time_ms is not None else None,
            )
        )
    candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:top_k]
    return sorted(candidates, key=lambda candidate: candidate.time_ms)


def _slot_duration_prior_weight(
    slot: ExpectedSlot,
    *,
    language: str,
    slot_count: int,
) -> float:
    if not _is_japanese_language(language):
        return 0.0
    if int(slot_count) < 4:
        return 0.0
    configured = _env_float(
        "UTOA_NO_MFA_JA_SLOT_DURATION_PRIOR_WEIGHT",
        _JAPANESE_SLOT_DURATION_PRIOR_DEFAULT_WEIGHT,
    )
    weight = max(0.0, float(configured))
    if weight <= 0.0:
        return 0.0
    if _is_sonorant_phone_change_slot(slot):
        return weight * 1.25
    return 0.0


def _slot_duration_prior_penalty(
    time_ms: float,
    *,
    expected_time_ms: float,
    slot_period_ms: float,
    weight: float,
) -> float:
    if float(weight) <= 0.0:
        return 0.0
    period = float(slot_period_ms)
    if not np.isfinite(period) or period <= 0.0:
        return 0.0
    distance = abs(float(time_ms) - float(expected_time_ms)) / max(1.0, period)
    min_distance = _env_float(
        "UTOA_NO_MFA_JA_SLOT_DURATION_PRIOR_MIN_DISTANCE",
        _JAPANESE_SLOT_DURATION_PRIOR_MIN_DISTANCE,
    )
    if distance < max(0.0, float(min_distance)):
        return 0.0
    return float(min(_JAPANESE_SLOT_DURATION_PRIOR_MAX_PENALTY, float(weight) * min(distance, 2.5)))


def _slot_class_prior(posterior: FramePosterior, slot: ExpectedSlot) -> np.ndarray:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    silence = np.asarray(posterior.class_probs.get("silence", []), dtype=np.float32)
    vowel = np.asarray(posterior.class_probs.get("vowel", []), dtype=np.float32)
    consonant = np.asarray(posterior.class_probs.get("consonant", []), dtype=np.float32)
    if silence.shape[0] != times.shape[0]:
        silence = np.zeros_like(times)
    if vowel.shape[0] != times.shape[0]:
        vowel = np.zeros_like(times)
    if consonant.shape[0] != times.shape[0]:
        consonant = np.zeros_like(times)
    if _is_cv_boundary_slot(slot):
        left_consonant = _left_context_max(consonant, frames=4)
        return np.clip(0.55 * vowel + 0.45 * left_consonant, 0.0, 1.0)
    if _is_vowel_nucleus_slot(slot) or slot.role == "vv":
        return np.clip(vowel, 0.0, 1.0)
    if str(slot.event_label) == "phone_change":
        active = np.clip(1.0 - silence, 0.0, 1.0)
        if _is_sonorant_phone_change_slot(slot):
            return np.clip((0.40 * consonant) + (0.34 * vowel) + (0.26 * active), 0.0, 1.0)
        return np.clip((0.72 * consonant) + (0.28 * active), 0.0, 1.0)
    return np.zeros_like(times)


def _slot_acoustic_prior(posterior: FramePosterior, slot: ExpectedSlot) -> np.ndarray:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    transition = _score_track(posterior, "transition_likelihood", times)
    flux = _score_track(posterior, "flux_likelihood", times)
    vowel_boundary = _score_track(posterior, "vowel_boundary_likelihood", times)
    sonorant = _score_track(posterior, "sonorant_onset_likelihood", times)
    spectral_delta = _score_track(posterior, "spectral_shape_delta_likelihood", times)
    voicing = _score_track(posterior, "voicing", times)
    nucleus = np.maximum(_score_track(posterior, "nucleus_likelihood", times), _score_track(posterior, "world_nucleus", times))
    periodicity = _score_track(posterior, "world_periodicity", times)
    spectral_stability = _score_track(posterior, "world_spectral_stability", times)
    silence = _score_track(posterior, "silence_likelihood", times)
    non_silence = 1.0 - silence
    if str(slot.event_label) == "phone_change":
        if _is_sonorant_phone_change_slot(slot):
            sonorant_edge = np.maximum(sonorant, np.clip((0.68 * spectral_delta) + (0.32 * transition), 0.0, 1.0))
            return np.clip(
                (0.46 * sonorant_edge)
                + (0.18 * transition)
                + (0.12 * flux)
                + (0.14 * voicing)
                + (0.10 * non_silence),
                0.0,
                1.0,
            )
        unvoiced = np.clip(1.0 - voicing, 0.0, 1.0)
        return np.clip(
            (0.42 * transition) + (0.24 * flux) + (0.20 * unvoiced) + (0.14 * non_silence),
            0.0,
            1.0,
        )
    if slot.role == "vv":
        boundary = np.maximum(vowel_boundary, np.clip((0.62 * spectral_delta) + (0.26 * transition) + (0.12 * non_silence), 0.0, 1.0))
        return np.clip(
            (0.44 * boundary)
            + (0.18 * transition)
            + (0.16 * nucleus)
            + (0.14 * voicing)
            + (0.08 * non_silence),
            0.0,
            1.0,
        )
    if _is_cv_boundary_slot(slot):
        return np.clip(
            0.42 * transition + 0.24 * flux + 0.18 * non_silence + 0.16 * sonorant,
            0.0,
            1.0,
        )
    if _is_vowel_nucleus_slot(slot):
        weights = _nucleus_prior_weights()
        return np.clip(
            (weights["voicing"] * voicing)
            + (weights["nucleus"] * nucleus)
            + (weights["periodicity"] * periodicity)
            + (weights["stability"] * spectral_stability)
            + (weights["non_silence"] * non_silence),
            0.0,
            1.0,
        )
    return np.zeros_like(times)


def _is_sonorant_phone_change_slot(slot: ExpectedSlot) -> bool:
    if str(slot.event_label) != "phone_change":
        return False
    return _is_sonorant_phone(slot.phone)


def _is_nasal_phone(phone: str) -> bool:
    return str(phone or "").strip().lower() in _NASAL_PHONES


def _is_liquid_phone(phone: str) -> bool:
    return str(phone or "").strip().lower() in _LIQUID_PHONES


def _is_cv_boundary_slot(slot: ExpectedSlot) -> bool:
    if str(slot.event_label) != "cv_boundary":
        return False
    role = str(slot.role or "").strip().lower()
    return role in {"cv_boundary", "cv", "cv_head", "implicit_cv", "vcv"}


def _is_vowel_nucleus_slot(slot: ExpectedSlot) -> bool:
    if str(slot.event_label) != "vowel_nucleus":
        return False
    role = str(slot.role or "").strip().lower()
    return role in {"vowel_nucleus", "v"}


def _is_sonorant_phone(phone: str) -> bool:
    normalized = str(phone or "").strip().lower()
    if not normalized:
        return False
    if normalized in _SONORANT_PHONES:
        return True
    return normalized.startswith(("m", "n", "r", "l"))


def _is_korean_language(language: str) -> bool:
    lang = str(language or "").strip().lower()
    return bool(lang.startswith("ko") or lang.startswith("kr"))


def _is_japanese_language(language: str) -> bool:
    lang = str(language or "").strip().lower()
    return bool(lang.startswith("ja") or lang == "japanese")


def _ja_slot_active_window_enabled() -> bool:
    raw = str(os.environ.get("UTOA_NO_MFA_JA_SLOT_ACTIVE_WINDOW", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


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
    if _is_cv_boundary_slot(slot) and transition is not None and transition < 0.20:
        warnings.append(f"weak_flux_slot:{slot.slot_index}:{transition:.3f}")
    if _is_vowel_nucleus_slot(slot) and voicing is not None and voicing < 0.20:
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


def _expected_slot_times(
    slots: Sequence[ExpectedSlot],
    duration_ms: float,
    *,
    start_ms: float = 0.0,
    end_ms: float | None = None,
) -> Mapping[int, float]:
    if not slots:
        return {}
    start = max(0.0, min(float(start_ms), float(duration_ms)))
    end = float(duration_ms) if end_ms is None else max(start, min(float(end_ms), float(duration_ms)))
    if len(slots) == 1:
        return {slots[0].slot_index: start + ((end - start) * 0.5)}
    span = max(1.0, end - start)
    return {
        slot.slot_index: start + span * (float(idx) / float(len(slots) - 1))
        for idx, slot in enumerate(slots)
    }


def _expected_time_window(
    posterior: FramePosterior,
    times: np.ndarray,
    duration_ms: float,
    *,
    language: str,
) -> tuple[float, float]:
    korean = _is_korean_language(language)
    japanese = _is_japanese_language(language) and _ja_slot_active_window_enabled()
    if not (korean or japanese):
        return 0.0, float(duration_ms)
    if times.size <= 2:
        return 0.0, float(duration_ms)
    silence = _score_track(posterior, "silence_likelihood", times)
    if silence.shape[0] != times.shape[0] or not np.any(np.isfinite(silence)):
        return 0.0, float(duration_ms)
    if korean:
        silence_max = _env_float("UTOA_NO_MFA_KR_SLOT_ACTIVE_SILENCE_MAX", 0.66)
        edge_min_ms = 80.0
        head_margin_ms = _env_float("UTOA_NO_MFA_KR_SLOT_ACTIVE_HEAD_MARGIN_MS", 55.0)
        tail_margin_ms = _env_float("UTOA_NO_MFA_KR_SLOT_ACTIVE_TAIL_MARGIN_MS", 60.0)
    else:
        silence_max = _env_float("UTOA_NO_MFA_JA_SLOT_ACTIVE_SILENCE_MAX", 0.68)
        edge_min_ms = _env_float("UTOA_NO_MFA_JA_SLOT_ACTIVE_MIN_EDGE_MS", 120.0)
        head_margin_ms = _env_float("UTOA_NO_MFA_JA_SLOT_ACTIVE_HEAD_MARGIN_MS", 18.0)
        tail_margin_ms = _env_float("UTOA_NO_MFA_JA_SLOT_ACTIVE_TAIL_MARGIN_MS", 45.0)
    active = silence < float(silence_max)
    indices = np.flatnonzero(active)
    if indices.size <= 0:
        return 0.0, float(duration_ms)
    raw_start = float(times[int(indices[0])])
    raw_end = float(times[int(indices[-1])])
    if raw_end <= raw_start + 120.0:
        return 0.0, float(duration_ms)
    leading = raw_start
    trailing = float(duration_ms) - raw_end
    if leading < float(edge_min_ms) and trailing < float(edge_min_ms):
        return 0.0, float(duration_ms)
    span = max(1.0, raw_end - raw_start)
    head_margin = min(float(head_margin_ms), span * 0.04)
    tail_margin = min(float(tail_margin_ms), span * 0.04)
    start = max(0.0, raw_start + head_margin)
    end = min(float(duration_ms), raw_end - tail_margin)
    if end <= start + 120.0:
        return raw_start, raw_end
    return start, end


def _expected_time_hard_window_ms(
    slot: ExpectedSlot,
    *,
    language: str,
    slot_period_ms: float,
    slot_count: int,
) -> float | None:
    if not _is_korean_language(language):
        return None
    if int(slot_count) < 3:
        return None
    sonorant_vc = _is_korean_sonorant_vc_grid_slot(slot, slot_count=slot_count, min_slot_count=4)
    obstruent_vc = _is_korean_obstruent_vc_grid_slot(slot, slot_count=slot_count)
    if not (_is_cv_boundary_slot(slot) or _is_vowel_nucleus_slot(slot) or sonorant_vc or obstruent_vc):
        return None
    if obstruent_vc:
        configured_obstruent = _env_float("UTOA_NO_MFA_KR_OBSTRUENT_VC_EXPECTED_HARD_WINDOW_MS", 0.0)
        if configured_obstruent > 0.0:
            return configured_obstruent
        period = float(slot_period_ms)
        if not np.isfinite(period) or period <= 0.0:
            return 170.0
        return float(max(135.0, min(210.0, period * 0.70)))
    if sonorant_vc:
        configured_vc = _env_float("UTOA_NO_MFA_KR_VC_EXPECTED_HARD_WINDOW_MS", 0.0)
        if configured_vc > 0.0:
            return configured_vc
        period = float(slot_period_ms)
        if not np.isfinite(period) or period <= 0.0:
            return 180.0
        return float(max(115.0, min(220.0, period * 0.78)))
    configured = _env_float("UTOA_NO_MFA_KR_CV_EXPECTED_HARD_WINDOW_MS", 0.0)
    if configured > 0.0:
        return configured
    period = float(slot_period_ms)
    if not np.isfinite(period) or period <= 0.0:
        return 240.0
    if _is_vowel_nucleus_slot(slot):
        return float(max(220.0, min(380.0, period * 1.35)))
    return float(max(180.0, min(340.0, period * 1.35)))


def _expected_time_fallback_enabled(
    slot: ExpectedSlot,
    *,
    language: str,
    slot_count: int,
) -> bool:
    if not _is_korean_language(language):
        return False
    return bool(
        (int(slot_count) >= 3 and _is_vowel_nucleus_slot(slot))
        or _is_korean_sonorant_vc_grid_slot(slot, slot_count=slot_count, min_slot_count=4)
        or _is_korean_obstruent_vc_grid_slot(slot, slot_count=slot_count)
    )


def _candidate_expected_time_ms(
    slot: ExpectedSlot,
    expected_time_ms: float | None,
    *,
    language: str,
    slot_period_ms: float,
    slot_count: int,
) -> float | None:
    if expected_time_ms is None:
        return None
    expected = float(expected_time_ms)
    if not _is_korean_sonorant_vc_grid_slot(slot, slot_count=slot_count):
        if _is_korean_obstruent_vc_grid_slot(slot, slot_count=slot_count):
            period = float(slot_period_ms)
            if not np.isfinite(period) or period <= 0.0 or not _is_korean_language(language):
                return expected
            phone = str(slot.phone or "").strip().lower()
            ratio = _env_phone_float(
                "UTOA_NO_MFA_KR_OBSTRUENT_VC_EXPECTED_SHIFT_RATIO",
                phone,
                _KOREAN_OBSTRUENT_VC_EXPECTED_SHIFT_RATIO_BY_PHONE.get(phone, 0.35),
            )
            return max(0.0, expected + period * float(ratio))
        return expected
    period = float(slot_period_ms)
    if not np.isfinite(period) or period <= 0.0:
        return expected
    if not _is_korean_language(language):
        return expected
    phone = str(slot.phone or "").strip().lower()
    if _is_nasal_phone(phone):
        ratio = _env_float("UTOA_NO_MFA_KR_NASAL_VC_EXPECTED_SHIFT_RATIO", 0.45)
        return max(0.0, expected - period * float(ratio))
    if _is_liquid_phone(phone):
        ratio = _env_float("UTOA_NO_MFA_KR_LIQUID_VC_EXPECTED_SHIFT_RATIO", 0.55)
        return max(0.0, expected + period * float(ratio))
    return expected


def _is_korean_sonorant_vc_grid_slot(
    slot: ExpectedSlot,
    *,
    slot_count: int,
    min_slot_count: int = 8,
) -> bool:
    if int(slot_count) < int(min_slot_count):
        return False
    if str(slot.event_label) != "phone_change":
        return False
    if str(slot.role or "").strip().lower() != "vc":
        return False
    return bool(_is_nasal_phone(slot.phone) or _is_liquid_phone(slot.phone))


def _is_korean_obstruent_vc_grid_slot(slot: ExpectedSlot, *, slot_count: int) -> bool:
    if int(slot_count) < 8:
        return False
    if str(slot.event_label) != "phone_change":
        return False
    if str(slot.role or "").strip().lower() != "vc":
        return False
    phone = str(slot.phone or "").strip().lower()
    if not phone:
        return False
    return not _is_sonorant_phone(phone)


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
    slot_period_ms: float,
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
            required_gap = max(required_gap, _same_role_sequence_gap_ms(prev_slot, slot, slot_period_ms))
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


def _same_role_sequence_gap_ms(prev_slot: ExpectedSlot, cur_slot: ExpectedSlot, slot_period_ms: float) -> float:
    prev_role = str(prev_slot.role or "").strip().lower()
    cur_role = str(cur_slot.role or "").strip().lower()
    if prev_role != cur_role:
        return 0.0
    if prev_role != "cv":
        return 0.0
    period = float(slot_period_ms)
    if not np.isfinite(period) or period <= 0.0:
        return 0.0
    return float(
        min(
            _ROLE_GAP_CONSECUTIVE_CV_MAX_MS,
            max(_ROLE_GAP_CONSECUTIVE_CV_MIN_MS, period * _ROLE_GAP_CONSECUTIVE_CV_RATIO),
        )
    )


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
    if _is_cv_boundary_slot(slot):
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


def _should_run_cv_local_refine(assignments: Sequence[SlotAssignment], warnings: Sequence[str]) -> bool:
    if not assignments:
        return False
    avg_score = float(np.mean(np.asarray([float(item.score) for item in assignments], dtype=np.float32)))
    low_score_count = sum(1 for item in assignments if float(item.score) < 0.22)
    has_gap_warning = any(str(w).startswith("tight_gap:") for w in warnings)
    has_low_score_warning = any(str(w).startswith("low_score_slot:") for w in warnings)
    return bool(avg_score < 0.44 or low_score_count >= 2 or has_gap_warning or has_low_score_warning)


def _refine_cv_assignments_locally(
    *,
    assignments: Sequence[SlotAssignment],
    slots: Sequence[ExpectedSlot],
    posterior: FramePosterior,
    expected_times: Mapping[int, float],
    min_gap_ms: float,
    same_phone_min_gap_ms: float,
    window_ms: float,
) -> list[SlotAssignment]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    event_values = np.asarray(posterior.event_scores.get("cv_boundary", []), dtype=np.float32)
    if event_values.shape[0] != times.shape[0]:
        return list(assignments)
    transition = _score_track(posterior, "transition_likelihood", times)
    flux = _score_track(posterior, "flux_likelihood", times)
    sonorant = _score_track(posterior, "sonorant_onset_likelihood", times)
    silence = _score_track(posterior, "silence_likelihood", times)
    blended = np.clip(
        (0.60 * np.clip(event_values, 0.0, 1.0))
        + (0.17 * transition)
        + (0.13 * flux)
        + (0.10 * np.clip(sonorant + (1.0 - silence), 0.0, 1.0)),
        0.0,
        1.0,
    )

    out = list(assignments)
    for idx, assignment in enumerate(out):
        slot = slots[assignment.slot_index]
        if not _is_cv_boundary_slot(slot):
            continue
        expected_time = expected_times.get(assignment.slot_index)
        min_allowed: float | None = None
        max_allowed: float | None = None
        if idx - 1 >= 0:
            prev = out[idx - 1]
            prev_slot = slots[prev.slot_index]
            min_allowed = float(prev.selected_time_ms) + _required_gap_ms(
                prev_slot,
                slot,
                min_gap_ms=min_gap_ms,
                same_phone_min_gap_ms=same_phone_min_gap_ms,
            )
        if idx + 1 < len(out):
            nxt = out[idx + 1]
            nxt_slot = slots[nxt.slot_index]
            max_allowed = float(nxt.selected_time_ms) - _required_gap_ms(
                slot,
                nxt_slot,
                min_gap_ms=min_gap_ms,
                same_phone_min_gap_ms=same_phone_min_gap_ms,
            )
        center = float(assignment.selected_time_ms if expected_time is None else expected_time)
        left = center - float(window_ms)
        right = center + float(window_ms)
        if min_allowed is not None:
            left = max(left, min_allowed)
        if max_allowed is not None:
            right = min(right, max_allowed)
        if right <= left + 1e-4:
            continue
        best_idx: int | None = None
        best_score = -1.0
        for frame_idx, time_value in enumerate(times):
            t_ms = float(time_value)
            if t_ms + 1e-5 < left or t_ms - 1e-5 > right:
                continue
            score = float(blended[frame_idx])
            if expected_time is not None:
                score -= min(0.30, abs(t_ms - float(expected_time)) * 0.0025)
            if score > best_score:
                best_score = score
                best_idx = frame_idx
        if best_idx is None:
            continue
        candidate_time = float(times[best_idx])
        candidate_score = float(blended[best_idx])
        current_event_score = float(event_values[max(0, min(len(event_values) - 1, int(assignment.frame_index)))])
        candidate_event_score = float(event_values[best_idx])
        move_ms = abs(candidate_time - float(assignment.selected_time_ms))
        current_score = float(assignment.score)
        current_dist = abs(float(assignment.selected_time_ms) - float(expected_time)) if expected_time is not None else 0.0
        candidate_dist = abs(candidate_time - float(expected_time)) if expected_time is not None else 0.0
        better = (
            (candidate_score > current_score + 0.03)
            or (expected_time is not None and candidate_dist + 6.0 < current_dist)
        )
        if move_ms > 28.0:
            better = False
        if candidate_event_score + 0.01 < current_event_score:
            better = False
        if move_ms > 18.0 and candidate_event_score < current_event_score + 0.04:
            better = False
        if not better:
            continue
        out[idx] = SlotAssignment(
            slot_index=assignment.slot_index,
            phone_index=assignment.phone_index,
            phone=assignment.phone,
            role=assignment.role,
            event_label=assignment.event_label,
            selected_time_ms=candidate_time,
            score=max(current_score, candidate_score),
            frame_index=int(best_idx),
            expected_time_ms=assignment.expected_time_ms,
        )
    return out


def _nucleus_prior_weights() -> dict[str, float]:
    weights = {
        "voicing": _env_float("UTOA_NO_MFA_NUCLEUS_W_VOICING", 0.34),
        "nucleus": _env_float("UTOA_NO_MFA_NUCLEUS_W_NUCLEUS", 0.34),
        "periodicity": _env_float("UTOA_NO_MFA_NUCLEUS_W_PERIODICITY", 0.18),
        "stability": _env_float("UTOA_NO_MFA_NUCLEUS_W_STABILITY", 0.08),
        "non_silence": _env_float("UTOA_NO_MFA_NUCLEUS_W_NONSILENCE", 0.06),
    }
    total = float(sum(max(0.0, value) for value in weights.values()))
    if total <= 1e-9:
        return {"voicing": 0.34, "nucleus": 0.34, "periodicity": 0.18, "stability": 0.08, "non_silence": 0.06}
    return {key: max(0.0, value) / total for key, value in weights.items()}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError:
        return float(default)


def _env_phone_float(prefix: str, phone: str, default: float) -> float:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(phone or "").strip().upper())
    if token:
        raw = os.environ.get(f"{prefix}_{token}")
        if raw is not None:
            text = str(raw).strip()
            if text:
                try:
                    return float(text)
                except ValueError:
                    return float(default)
    return _env_float(prefix, default)
