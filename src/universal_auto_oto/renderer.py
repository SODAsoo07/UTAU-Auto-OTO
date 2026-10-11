from __future__ import annotations

from dataclasses import replace

from .domain import PlannedAlias, RenderedOtoRow, Slot, Timeline
from .oto_io import validate_oto_line_values


def render_alias(
    wav_name: str,
    timeline: Timeline,
    plan: PlannedAlias,
    *,
    review_threshold: float,
    timing_profile: dict | None = None,
    format_hint: str | None = None,
) -> RenderedOtoRow:
    slot_index = min(max(0, plan.slot_index), len(timeline.slots) - 1)
    current = timeline.slots[slot_index]
    following = timeline.slots[min(slot_index + 1, len(timeline.slots) - 1)]
    role = plan.role
    if role in {"cv", "cv_head"}:
        anchors = _render_cvc_cv(current) if (format_hint or "").casefold() == "cvc" else _render_cv(current)
    elif role == "cvc":
        anchors = _render_cvc(current, following)
    elif role == "vc":
        anchors = _render_vc(current, following)
    elif role == "cvc_vc":
        anchors = _render_cvc_vc(current)
    elif role == "vcv":
        anchors = _render_vc(current, following)
    elif role == "vc_end":
        anchors = _render_release(current)
    elif role == "vv":
        anchors = _render_vv(current)
    elif role == "release":
        anchors = _render_release(current)
    elif role == "breath":
        anchors = _render_breath(current)
    else:
        anchors = _render_cv(current)
        role = "cv"
    if timing_profile:
        anchors = _render_profiled(current, following, timing_profile, fallback=anchors)
    offset_abs, overlap_abs, pre_abs, fixed_abs, cutoff_abs, repaired = _sanitize_anchors(
        *anchors,
        duration_ms=timeline.duration_ms,
    )
    offset = offset_abs
    overlap = overlap_abs - offset_abs
    preutterance = pre_abs - offset_abs
    consonant = fixed_abs - offset_abs
    # UTAU의 음수 cutoff는 WAV 끝 기준 right blank가 아니라
    # offset부터 사용할 구간의 길이(minus blank)다.
    cutoff = -(cutoff_abs - offset_abs)
    role_confidence = min(current.boundary_confidence, current.transition_confidence)
    if role in {"vc", "vcv", "cvc", "cvc_vc"} and following.index != current.index:
        role_confidence = min(role_confidence, following.boundary_confidence, following.transition_confidence)
    confidence = max(0.0, min(1.0, 0.58 * timeline.confidence + 0.42 * role_confidence))
    reasons: list[str] = list(timeline.warnings)
    if repaired:
        reasons.append("template_constraint_repaired")
        confidence *= 0.82
    valid, validation_reasons = validate_oto_line_values(
        duration_ms=timeline.duration_ms,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=preutterance,
        overlap=overlap,
    )
    reasons.extend(validation_reasons)
    if confidence < review_threshold:
        reasons.append("low_confidence")
    split = "clean" if valid and confidence >= review_threshold and not repaired else "review"
    return RenderedOtoRow(
        wav_name=wav_name,
        alias=plan.source.alias,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        preutterance=preutterance,
        overlap=overlap,
        role=role,
        slot_index=slot_index,
        confidence=confidence,
        split=split,
        reasons=tuple(dict.fromkeys(reasons)),
        source_index=plan.source.source_index,
        absolute={
            "offset_abs": offset_abs,
            "overlap_abs": overlap_abs,
            "pre_abs": pre_abs,
            "fixed_end_abs": fixed_abs,
            "cutoff_abs": cutoff_abs,
        },
    )


def force_review(row: RenderedOtoRow, reason: str) -> RenderedOtoRow:
    return replace(row, split="review", reasons=tuple(dict.fromkeys((*row.reasons, reason))))


def _render_cv(slot: Slot) -> tuple[float, float, float, float, float]:
    lead = min(70.0, max(20.0, (slot.transition_ms - slot.start_ms) * 0.55))
    return (
        slot.start_ms,
        max(slot.start_ms, slot.transition_ms - lead),
        slot.transition_ms,
        max(slot.stable_ms, slot.transition_ms + 30.0),
        slot.end_ms - 15.0,
    )


def _render_cvc(current: Slot, following: Slot) -> tuple[float, float, float, float, float]:
    pre = current.transition_ms
    fixed = max(pre + 45.0, min(current.stable_ms, pre + 75.0))
    return (
        max(current.start_ms, pre - 80.0),
        max(current.start_ms, pre - 35.0),
        pre,
        fixed,
        min(current.end_ms - 10.0, max(fixed + 25.0, pre + 140.0)),
    )


def _render_cvc_cv(slot: Slot) -> tuple[float, float, float, float, float]:
    pre = slot.transition_ms
    fixed = max(pre + 55.0, min(slot.stable_ms, pre + 90.0))
    return (
        max(slot.start_ms, pre - 110.0),
        max(slot.start_ms, pre - 55.0),
        pre,
        fixed,
        min(slot.end_ms - 10.0, max(fixed + 25.0, pre + 160.0)),
    )


def _render_cvc_vc(slot: Slot) -> tuple[float, float, float, float, float]:
    pre = slot.transition_ms
    fixed = max(pre + 40.0, min(slot.stable_ms, pre + 65.0))
    return (
        max(slot.start_ms, pre - 55.0),
        max(slot.start_ms, pre - 20.0),
        pre,
        fixed,
        min(slot.end_ms - 10.0, max(fixed + 20.0, pre + 125.0)),
    )


def _render_vc(current: Slot, following: Slot) -> tuple[float, float, float, float, float]:
    if following.index == current.index:
        return _render_release(current)
    offset = max(current.start_ms, min(current.stable_ms, current.end_ms - 120.0))
    pre = following.start_ms
    overlap = max(offset, pre - min(100.0, max(45.0, (current.end_ms - current.start_ms) * 0.18)))
    fixed = max(pre + 25.0, following.transition_ms)
    cutoff = max(fixed + 15.0, following.stable_ms)
    return offset, overlap, pre, fixed, cutoff


def _render_vv(slot: Slot) -> tuple[float, float, float, float, float]:
    duration = slot.end_ms - slot.start_ms
    pre = slot.start_ms + min(160.0, max(60.0, duration * 0.12))
    return (
        slot.start_ms,
        max(slot.start_ms, pre - 55.0),
        pre,
        min(slot.end_ms - 60.0, pre + max(50.0, duration * 0.18)),
        slot.end_ms - 20.0,
    )


def _render_release(slot: Slot) -> tuple[float, float, float, float, float]:
    duration = slot.end_ms - slot.start_ms
    offset = max(slot.start_ms, slot.end_ms - max(420.0, duration * 0.35))
    pre = max(offset + 30.0, slot.end_ms - 150.0)
    return (
        offset,
        max(offset, pre - 60.0),
        pre,
        max(pre + 25.0, slot.end_ms - 70.0),
        slot.end_ms - 15.0,
    )


def _render_breath(slot: Slot) -> tuple[float, float, float, float, float]:
    duration = slot.end_ms - slot.start_ms
    pre = slot.start_ms + min(90.0, max(35.0, duration * 0.12))
    return (
        slot.start_ms,
        max(slot.start_ms, pre - 35.0),
        pre,
        min(slot.end_ms - 35.0, pre + 45.0),
        slot.end_ms - 15.0,
    )


def _render_profiled(
    current: Slot,
    following: Slot,
    profile: dict,
    *,
    fallback: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    ratios = profile.get("ratios") or {}
    required = ("offset_abs", "overlap_abs", "pre_abs", "fixed_abs", "cutoff_abs")
    configured_fields = profile.get("apply_fields")
    apply_fields = set(required if configured_fields is None else configured_fields)
    if not apply_fields or not apply_fields <= set(required):
        raise ValueError("timing profile apply_fields가 비어 있거나 지원되지 않는 anchor를 포함합니다")
    if not all(field in ratios for field in apply_fields):
        raise ValueError("timing profile에 적용 대상 anchor ratio가 없습니다")
    base = current.start_ms
    unit = (following.start_ms - current.start_ms) if following.index != current.index else (current.end_ms - current.start_ms)
    unit = max(1.0, unit)
    values = dict(zip(required, fallback))
    for field in apply_fields:
        values[field] = base + float(ratios[field]) * unit
    if apply_fields == {"cutoff_abs"}:
        # A cutoff-only profile must not pull fixed/pre/overlap backward during
        # sanitization; preserve the unprofiled anchors as the hard lower bound.
        values["cutoff_abs"] = max(values["fixed_abs"], values["cutoff_abs"])
    return tuple(values[field] for field in required)


def _sanitize_anchors(
    offset_abs: float,
    overlap_abs: float,
    pre_abs: float,
    fixed_abs: float,
    cutoff_abs: float,
    *,
    duration_ms: float,
) -> tuple[float, float, float, float, float, bool]:
    original = (offset_abs, overlap_abs, pre_abs, fixed_abs, cutoff_abs)
    latest_cutoff = max(6.0, duration_ms - 1.0)
    offset_abs = min(max(0.0, offset_abs), latest_cutoff - 5.0)
    cutoff_abs = min(max(offset_abs + 5.0, cutoff_abs), latest_cutoff)
    fixed_abs = min(max(offset_abs, fixed_abs), cutoff_abs)
    pre_abs = min(max(offset_abs, pre_abs), fixed_abs)
    overlap_abs = min(max(offset_abs, overlap_abs), pre_abs)
    sanitized = (offset_abs, overlap_abs, pre_abs, fixed_abs, cutoff_abs)
    repaired = any(abs(a - b) > 0.1 for a, b in zip(original, sanitized))
    return (*sanitized, repaired)
