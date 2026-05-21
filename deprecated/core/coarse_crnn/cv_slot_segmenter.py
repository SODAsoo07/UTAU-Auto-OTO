from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.coarse_crnn.boundary_types import BoundaryCandidate


CV_ONSET_KINDS = frozenset({"syllable_onset", "consonant_onset", "vowel_start"})
CV_VOWEL_KINDS = frozenset({"vowel_start", "vowel_stable"})


@dataclass(frozen=True)
class CvSlot:
    index: int
    start_ms: float
    vowel_start_ms: float
    nucleus_ms: float
    end_ms: float
    confidence: float
    source: str


def segment_cv_slots(
    *,
    wav_name: str = "",
    slot_count: int,
    duration_ms: float,
    candidates: Iterable[BoundaryCandidate],
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    min_slot_gap_ms: float = 24.0,
) -> list[CvSlot]:
    """Split a wav into filename-order CV slots without identifying phones.

    This intentionally answers only "where are the next CV-like chunks?".
    Alias text is not parsed beyond the caller-provided slot count, so it can
    assign slots in file/reclist order without guessing the phoneme identity.
    """
    _ = wav_name
    count = max(1, int(slot_count))
    end = float(active_end_ms) if active_end_ms is not None else float(duration_ms)
    start = max(0.0, min(float(active_start_ms), end))
    end = max(start + 1.0, min(float(duration_ms), end))
    min_gap = max(1.0, float(min_slot_gap_ms))
    cand_list = sorted(
        [c for c in candidates if start - 80.0 <= float(c.time_ms) <= end + 80.0],
        key=lambda c: float(c.time_ms),
    )

    starts: list[float] = []
    confidences: list[float] = []
    sources: list[str] = []
    for idx in range(count):
        prior = _linear_slot_start(idx=idx, count=count, start_ms=start, end_ms=end)
        picked = _pick_near(
            cand_list,
            target_ms=prior,
            kinds=CV_ONSET_KINDS,
            window_ms=_slot_pick_window_ms(count=count, start_ms=start, end_ms=end),
        )
        t = prior if picked is None else float(picked.time_ms)
        if starts:
            t = max(t, starts[-1] + min_gap)
        t = min(t, end)
        starts.append(float(t))
        confidences.append(0.35 if picked is None else _clamp01(float(picked.score)))
        sources.append("cv_slot:linear" if picked is None else f"cv_slot:{picked.kind}:{picked.source}")

    starts = _repair_monotonic(starts, active_start_ms=start, active_end_ms=end, min_gap_ms=min_gap)
    slots: list[CvSlot] = []
    for idx, slot_start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else end
        slot_end = max(float(slot_start) + 1.0, float(next_start))
        vowel = _pick_near(
            cand_list,
            target_ms=float(slot_start) + min(70.0, max(12.0, 0.25 * (slot_end - slot_start))),
            kinds=CV_VOWEL_KINDS,
            window_ms=max(40.0, min(140.0, 0.60 * (slot_end - slot_start))),
            lo_ms=float(slot_start) - 8.0,
            hi_ms=float(slot_end) + 12.0,
        )
        vowel_start = (
            _clamp(float(vowel.time_ms), float(slot_start), float(slot_end))
            if vowel is not None
            else float(slot_start)
        )
        nucleus = _clamp(
            float(vowel_start) + min(80.0, max(16.0, 0.25 * (slot_end - vowel_start))),
            float(vowel_start),
            float(slot_end),
        )
        conf = confidences[idx]
        if vowel is not None:
            conf = 0.65 * conf + 0.35 * _clamp01(float(vowel.score))
        slots.append(
            CvSlot(
                index=int(idx),
                start_ms=float(slot_start),
                vowel_start_ms=float(vowel_start),
                nucleus_ms=float(nucleus),
                end_ms=float(slot_end),
                confidence=_clamp01(conf),
                source=str(sources[idx]),
            )
        )
    return slots


def cv_slot_timeline_from_slots(slots: Iterable[CvSlot]) -> list[float]:
    return [float(slot.start_ms) for slot in sorted(slots, key=lambda s: int(s.index))]


def _linear_slot_start(*, idx: int, count: int, start_ms: float, end_ms: float) -> float:
    if count <= 1:
        return float(start_ms)
    return float(start_ms) + (float(end_ms) - float(start_ms)) * float(idx) / float(count - 1)


def _slot_pick_window_ms(*, count: int, start_ms: float, end_ms: float) -> float:
    if count <= 1:
        return 180.0
    span = max(1.0, float(end_ms) - float(start_ms))
    return max(60.0, min(220.0, 0.42 * span / float(max(1, count - 1))))


def _pick_near(
    candidates: list[BoundaryCandidate],
    *,
    target_ms: float,
    kinds: frozenset[str],
    window_ms: float,
    lo_ms: float | None = None,
    hi_ms: float | None = None,
) -> BoundaryCandidate | None:
    best: BoundaryCandidate | None = None
    best_cost: float | None = None
    lo = float("-inf") if lo_ms is None else float(lo_ms)
    hi = float("inf") if hi_ms is None else float(hi_ms)
    for cand in candidates:
        if cand.kind not in kinds:
            continue
        t = float(cand.time_ms)
        if t < lo or t > hi:
            continue
        dist = abs(t - float(target_ms))
        if dist > float(window_ms):
            continue
        source_bonus = 12.0 if str(cand.source or "").startswith("model") else 0.0
        kind_penalty = 55.0 if cand.kind == "vowel_start" and "syllable_onset" in kinds else 0.0
        cost = dist + kind_penalty - (32.0 * _clamp01(float(cand.score))) - source_bonus
        if best is None or cost < float(best_cost):
            best = cand
            best_cost = cost
    return best


def _repair_monotonic(
    starts: list[float],
    *,
    active_start_ms: float,
    active_end_ms: float,
    min_gap_ms: float,
) -> list[float]:
    if not starts:
        return []
    out = [float(starts[0])]
    for value in starts[1:]:
        out.append(max(float(value), out[-1] + float(min_gap_ms)))
    overflow = out[-1] - float(active_end_ms)
    if overflow > 0.0 and len(out) > 1:
        out = [v - overflow for v in out]
        out[0] = max(float(active_start_ms), out[0])
        for idx in range(1, len(out)):
            out[idx] = max(out[idx], out[idx - 1] + float(min_gap_ms))
    return [_clamp(v, float(active_start_ms), float(active_end_ms)) for v in out]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


__all__ = ["CV_ONSET_KINDS", "CV_VOWEL_KINDS", "CvSlot", "cv_slot_timeline_from_slots", "segment_cv_slots"]
