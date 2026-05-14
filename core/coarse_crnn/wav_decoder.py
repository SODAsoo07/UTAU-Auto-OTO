from __future__ import annotations

from collections import defaultdict

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import (
    ANCHOR_ROLES,
    TRANSITION_ROLES,
    AbsoluteOtoAnchors,
    BoundaryCandidate,
    BoundaryDecodeResult,
    DecodedOtoRow,
    OtoRowSpec,
    role_label_preferences,
)
from core.coarse_crnn.oto_param_builder import build_absolute_anchors_for_role


def decode_wav_rows(
    *,
    wav_path: str,
    duration_ms: float,
    row_specs: list[OtoRowSpec],
    candidates: list[BoundaryCandidate],
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    min_anchor_gap_ms: float = 24.0,
) -> BoundaryDecodeResult:
    if not row_specs:
        return BoundaryDecodeResult(
            wav_path=wav_path,
            duration_ms=float(duration_ms),
            rows=[],
            anchor_timeline_ms=[],
            fallback_count=0,
        )
    slot_count = max(1, max(int(spec.slot_count) for spec in row_specs))
    anchor_timeline = _build_anchor_timeline(
        slot_count=slot_count,
        duration_ms=duration_ms,
        candidates=candidates,
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        min_anchor_gap_ms=min_anchor_gap_ms,
    )
    rows: list[DecodedOtoRow] = []
    fallback_count = 0
    by_kind: dict[str, list[BoundaryCandidate]] = defaultdict(list)
    for cand in candidates:
        by_kind[cand.kind].append(cand)
    for bucket in by_kind.values():
        bucket.sort(key=lambda item: item.time_ms)

    sorted_specs = sorted(row_specs, key=lambda item: int(item.line_index))
    for spec in sorted_specs:
        role = normalize_role(spec.role)
        left_anchor = anchor_timeline[min(max(0, spec.slot_index), len(anchor_timeline) - 1)]
        right_anchor = anchor_timeline[min(max(0, spec.slot_index + 1), len(anchor_timeline) - 1)] if spec.slot_index + 1 < len(anchor_timeline) else float(active_end_ms or duration_ms)
        selected = _select_row_candidate(role=role, left_anchor=left_anchor, right_anchor=right_anchor, by_kind=by_kind)
        fallback_used = selected is None
        if selected is None:
            selected_time = 0.5 * (left_anchor + right_anchor) if role in TRANSITION_ROLES else left_anchor
            score = 0.0
            source = "fallback:anchor_pair"
        else:
            selected_time = selected.time_ms
            score = selected.score
            source = selected.source
        anchors = build_absolute_anchors_for_role(
            role=role,
            center_ms=selected_time,
            duration_ms=float(duration_ms),
            left_anchor_ms=left_anchor,
            right_anchor_ms=right_anchor if role in TRANSITION_ROLES else None,
            active_start_ms=float(active_start_ms),
            active_end_ms=float(active_end_ms) if active_end_ms is not None else float(duration_ms),
            confidence=float(score),
            reason=source,
        )
        if fallback_used:
            fallback_count += 1
        rows.append(
            DecodedOtoRow(
                spec=spec,
                anchors=anchors,
                selected_time_ms=float(selected_time),
                fallback_used=bool(fallback_used),
                reason=str(source),
            )
        )
    return BoundaryDecodeResult(
        wav_path=wav_path,
        duration_ms=float(duration_ms),
        rows=rows,
        anchor_timeline_ms=anchor_timeline,
        fallback_count=int(fallback_count),
    )


def _build_anchor_timeline(
    *,
    slot_count: int,
    duration_ms: float,
    candidates: list[BoundaryCandidate],
    active_start_ms: float,
    active_end_ms: float | None,
    min_anchor_gap_ms: float,
) -> list[float]:
    active_end = float(active_end_ms) if active_end_ms is not None else float(duration_ms)
    active_start = max(0.0, min(active_end, float(active_start_ms)))
    duration = max(active_start + 1.0, active_end)
    onset_like = [row for row in candidates if row.kind in {"syllable_onset", "vowel_start", "consonant_onset"}]
    onset_like.sort(key=lambda item: item.time_ms)
    timeline: list[float] = []
    for idx in range(max(1, int(slot_count))):
        prior = active_start if slot_count <= 1 else active_start + ((duration - active_start) * float(idx) / float(max(1, slot_count - 1)))
        picked = _pick_closest(onset_like, prior, window_ms=180.0)
        timeline.append(float(prior if picked is None else picked.time_ms))
    for idx in range(1, len(timeline)):
        min_allowed = timeline[idx - 1] + float(min_anchor_gap_ms)
        if timeline[idx] < min_allowed:
            timeline[idx] = min_allowed
    if timeline:
        timeline[-1] = min(timeline[-1], duration)
    return timeline


def _pick_closest(candidates: list[BoundaryCandidate], target_ms: float, *, window_ms: float) -> BoundaryCandidate | None:
    best = None
    best_cost = None
    for cand in candidates:
        dist = abs(float(cand.time_ms) - float(target_ms))
        if dist > float(window_ms):
            continue
        cost = dist - (float(cand.score) * 32.0)
        if best is None or cost < float(best_cost):
            best = cand
            best_cost = cost
    return best


def _select_row_candidate(
    *,
    role: str,
    left_anchor: float,
    right_anchor: float,
    by_kind: dict[str, list[BoundaryCandidate]],
) -> BoundaryCandidate | None:
    labels = role_label_preferences(role)
    role_key = normalize_role(role)
    if role_key in TRANSITION_ROLES:
        lo = float(left_anchor) - 28.0
        hi = float(right_anchor) + 42.0
    else:
        lo = float(left_anchor) - 90.0
        hi = float(right_anchor) + 90.0
    best = None
    best_cost = None
    target = 0.5 * (left_anchor + right_anchor) if role_key in TRANSITION_ROLES else left_anchor
    for label in labels:
        for cand in by_kind.get(label, []):
            t = float(cand.time_ms)
            if t < lo or t > hi:
                continue
            dist = abs(t - target)
            if role_key in TRANSITION_ROLES:
                dist = abs(t - (left_anchor + 0.62 * max(0.0, right_anchor - left_anchor)))
            cost = dist - (float(cand.score) * 38.0)
            if best is None or cost < float(best_cost):
                best = cand
                best_cost = cost
    return best


__all__ = ["decode_wav_rows"]

