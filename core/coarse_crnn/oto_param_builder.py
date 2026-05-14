from __future__ import annotations

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params


ROLE_GAP_RULES_MS: dict[str, tuple[float, float, float, float]] = {
    "cv": (20.0, 80.0, 35.0, 130.0),
    "v": (4.0, 36.0, 14.0, 92.0),
    "vc": (8.0, 48.0, 18.0, 82.0),
    "vv": (14.0, 72.0, 26.0, 116.0),
    "v-cv": (16.0, 82.0, 30.0, 118.0),
    "br": (4.0, 34.0, 12.0, 64.0),
    "other": (8.0, 64.0, 18.0, 100.0),
}


def build_absolute_anchors_for_role(
    *,
    role: str,
    center_ms: float,
    duration_ms: float,
    left_anchor_ms: float | None = None,
    right_anchor_ms: float | None = None,
    active_start_ms: float = 0.0,
    active_end_ms: float | None = None,
    confidence: float = 0.0,
    reason: str = "",
) -> AbsoluteOtoAnchors:
    duration = max(20.0, float(duration_ms))
    active_end = float(active_end_ms) if active_end_ms is not None else duration
    active_start = max(0.0, min(active_end, float(active_start_ms)))
    role_key = normalize_role(role)
    cons_min, cons_max, tail_min, tail_max = ROLE_GAP_RULES_MS.get(role_key, ROLE_GAP_RULES_MS["other"])

    center = _clamp(float(center_ms), active_start, active_end)
    left = float(left_anchor_ms) if left_anchor_ms is not None else active_start
    right = float(right_anchor_ms) if right_anchor_ms is not None else active_end
    left = _clamp(left, 0.0, duration)
    right = _clamp(right, left, duration)

    cons_gap = _clamp(0.45 * (right - left), cons_min, cons_max)
    tail_gap = _clamp(0.60 * (right - left), tail_min, tail_max)

    if role_key in {"vc", "vv", "v-cv"}:
        pre_abs = _clamp(center, left, right)
        offset_abs = _clamp(max(left - 24.0, pre_abs - 40.0), 0.0, pre_abs)
    elif role_key in {"br", "endbr"}:
        pre_abs = _clamp(center, active_start, active_end)
        offset_abs = _clamp(pre_abs - 14.0, 0.0, pre_abs)
        cons_gap = _clamp(cons_gap, 4.0, 30.0)
        tail_gap = _clamp(tail_gap, 10.0, 58.0)
    else:
        pre_abs = _clamp(center, active_start, active_end)
        offset_abs = _clamp(pre_abs - max(18.0, 0.35 * cons_gap), 0.0, pre_abs)

    overlap_abs = _clamp(pre_abs - max(4.0, 0.35 * cons_gap), offset_abs, pre_abs)
    consonant_abs = _clamp(pre_abs + cons_gap, pre_abs + 1.0, duration - 1.0)
    cutoff_abs = _clamp(consonant_abs + tail_gap, consonant_abs + 1.0, duration)
    if right_anchor_ms is not None and role_key in {"vc", "vv", "v-cv"}:
        cutoff_abs = _clamp(cutoff_abs, consonant_abs + 1.0, min(duration, right + 72.0))

    return AbsoluteOtoAnchors(
        offset_abs=offset_abs,
        overlap_abs=overlap_abs,
        pre_abs=pre_abs,
        consonant_abs=consonant_abs,
        cutoff_abs=cutoff_abs,
        confidence=max(0.0, min(1.0, float(confidence))),
        reason=str(reason or role_key),
    )


def anchors_to_runtime_oto_params(anchors: AbsoluteOtoAnchors, *, duration_ms: float) -> dict[str, float]:
    return absolute_anchors_to_oto_params(anchors, duration_ms=duration_ms)


def stretch_risk_flags(params: dict[str, float]) -> dict[str, bool]:
    pre = max(1.0, float(params.get("preutterance", 0.0) or 0.0))
    cons = max(pre, float(params.get("consonant", 0.0) or 0.0))
    cutoff_abs = cons + abs(float(params.get("cutoff", 0.0) or 0.0))
    cons_gap = max(0.0, cons - pre)
    tail_gap = max(0.0, cutoff_abs - cons)
    overlap = max(0.0, float(params.get("overlap", 0.0) or 0.0))
    overlap_ratio = overlap / max(1.0, pre)
    return {
        "cons_gap_outlier": bool(cons_gap > 120.0),
        "tail_gap_outlier": bool(tail_gap > 170.0),
        "overlap_outlier": bool(overlap_ratio > 0.92),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = [
    "ROLE_GAP_RULES_MS",
    "anchors_to_runtime_oto_params",
    "build_absolute_anchors_for_role",
    "stretch_risk_flags",
]

