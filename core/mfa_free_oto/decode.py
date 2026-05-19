from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import DecodedEvent, EVENT_LABELS, FramePosterior, is_vowel_phone


@dataclass(frozen=True)
class ExpectedEvent:
    label: str
    phone: str | None = None


def expected_events_from_phones(phones: Sequence[str]) -> list[ExpectedEvent]:
    events: list[ExpectedEvent] = []
    previous_phone: str | None = None
    previous_was_consonant = False
    for phone in phones:
        vowel = is_vowel_phone(str(phone))
        if previous_phone is not None:
            events.append(ExpectedEvent("phone_change", str(phone)))
        if vowel:
            if previous_was_consonant:
                events.append(ExpectedEvent("cv_boundary", str(phone)))
            events.append(ExpectedEvent("vowel_nucleus", str(phone)))
        previous_phone = str(phone)
        previous_was_consonant = not vowel
    return events


def decode_monotonic_events(
    posterior: FramePosterior,
    *,
    expected_phones: Sequence[str] | None = None,
    expected_events: Sequence[ExpectedEvent | str] | None = None,
    min_score: float = 0.05,
    min_gap_ms: float = 10.0,
    top_k_per_event: int = 24,
) -> list[DecodedEvent]:
    if expected_events is None:
        expected_events = expected_events_from_phones(expected_phones or [])
    normalized_events = [
        event if isinstance(event, ExpectedEvent) else ExpectedEvent(str(event), None)
        for event in expected_events
        if (event.label if isinstance(event, ExpectedEvent) else str(event)) in EVENT_LABELS
    ]
    if not normalized_events:
        normalized_events = [ExpectedEvent("cv_boundary"), ExpectedEvent("vowel_nucleus")]

    times = np.asarray(posterior.times_ms, dtype=np.float32)
    if times.size == 0:
        return []
    candidate_steps = [
        _event_candidates(posterior, event.label, min_score=min_score, top_k=top_k_per_event)
        for event in normalized_events
    ]
    if any(not candidates for candidates in candidate_steps):
        return []
    dp: list[list[tuple[float, int | None]]] = []
    for step_idx, candidates in enumerate(candidate_steps):
        step_scores: list[tuple[float, int | None]] = []
        for cand_idx, candidate in enumerate(candidates):
            if step_idx == 0:
                step_scores.append((candidate.score, None))
                continue
            best_score = -1e9
            best_prev: int | None = None
            for prev_idx, prev_candidate in enumerate(candidate_steps[step_idx - 1]):
                prev_score, _ = dp[step_idx - 1][prev_idx]
                if candidate.time_ms + 1e-5 < prev_candidate.time_ms + min_gap_ms:
                    continue
                transition_penalty = 0.0005 * max(0.0, candidate.time_ms - prev_candidate.time_ms)
                score = prev_score + candidate.score - transition_penalty
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            step_scores.append((best_score, best_prev))
        dp.append(step_scores)
    last_scores = [score for score, _ in dp[-1]]
    last_idx = int(np.argmax(np.asarray(last_scores, dtype=np.float32)))
    if last_scores[last_idx] <= -1e8:
        return []
    selected: list[int] = []
    for step_idx in range(len(candidate_steps) - 1, -1, -1):
        selected.append(last_idx)
        _score, prev_idx = dp[step_idx][last_idx]
        if prev_idx is None:
            break
        last_idx = prev_idx
    selected.reverse()
    if len(selected) != len(candidate_steps):
        return []
    decoded: list[DecodedEvent] = []
    for event, candidates, selected_idx in zip(normalized_events, candidate_steps, selected):
        candidate = candidates[selected_idx]
        decoded.append(
            DecodedEvent(
                label=event.label,
                selected_time_ms=candidate.time_ms,
                score=candidate.score,
                expected_phone=event.phone,
                frame_index=candidate.frame_index,
            )
        )
    return decoded


@dataclass(frozen=True)
class _Candidate:
    frame_index: int
    time_ms: float
    score: float


def _event_candidates(
    posterior: FramePosterior,
    label: str,
    *,
    min_score: float,
    top_k: int,
) -> list[_Candidate]:
    times = np.asarray(posterior.times_ms, dtype=np.float32)
    values = np.asarray(posterior.event_scores.get(label, []), dtype=np.float32)
    if values.shape[0] != times.shape[0]:
        return []
    local_maxima: list[int] = []
    for idx, value in enumerate(values):
        if float(value) < min_score:
            continue
        left = values[idx - 1] if idx > 0 else -1.0
        right = values[idx + 1] if idx + 1 < values.shape[0] else -1.0
        if value >= left and value >= right:
            local_maxima.append(idx)
    if not local_maxima:
        local_maxima = [int(np.argmax(values))]
    local_maxima = sorted(local_maxima, key=lambda idx: float(values[idx]), reverse=True)[:top_k]
    candidates = [
        _Candidate(frame_index=idx, time_ms=float(times[idx]), score=float(values[idx]))
        for idx in local_maxima
    ]
    return sorted(candidates, key=lambda candidate: candidate.time_ms)
