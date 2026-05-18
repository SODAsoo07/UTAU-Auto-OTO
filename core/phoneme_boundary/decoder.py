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


__all__ = ["ExpectedBoundaryEvent", "decode_expected_boundaries", "expected_events_from_plan"]
