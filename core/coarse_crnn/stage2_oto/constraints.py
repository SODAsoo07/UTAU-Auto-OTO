from __future__ import annotations

import os

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, BoundaryCandidate, DecodedOtoRow
from core.coarse_crnn.stage2_oto.features import candidate_distance_for_anchors
from core.coarse_crnn.stage2_oto.types import Stage2Decision


def apply_stage2_gate(
    *,
    row: DecodedOtoRow,
    predicted_anchors_ms: tuple[float, float, float, float, float],
    confidence: float,
    candidates: list[BoundaryCandidate],
    min_confidence: float,
    max_candidate_distance_ms: float,
) -> Stage2Decision:
    duration = max(1.0, float(row.spec.duration_ms or 0.0))
    repaired = repair_anchor_order(predicted_anchors_ms, duration_ms=duration)
    conf = _clamp01(float(confidence))
    if conf < float(min_confidence):
        return Stage2Decision(
            anchors_ms=_base_anchor_tuple(row),
            confidence=conf,
            accepted=False,
            reason=f"stage2_fallback:low_confidence:{conf:.3f}",
        )
    max_dist = candidate_distance_for_anchors(
        anchors_ms=repaired,
        role=normalize_role(row.spec.role),
        candidates=candidates,
        duration_ms=duration,
    )
    if candidates and max_dist > float(max_candidate_distance_ms):
        return Stage2Decision(
            anchors_ms=_base_anchor_tuple(row),
            confidence=conf,
            accepted=False,
            reason=f"stage2_fallback:candidate_distance:{max_dist:.1f}",
            max_candidate_distance_ms=float(max_dist),
        )
    if not validate_anchor_shape(repaired, duration_ms=duration):
        return Stage2Decision(
            anchors_ms=_base_anchor_tuple(row),
            confidence=conf,
            accepted=False,
            reason="stage2_fallback:invalid_shape",
            max_candidate_distance_ms=float(max_dist),
        )
    return Stage2Decision(
        anchors_ms=repaired,
        confidence=conf,
        accepted=True,
        reason="stage2:model",
        max_candidate_distance_ms=float(max_dist),
    )


def repair_anchor_order(
    anchors_ms: tuple[float, float, float, float, float],
    *,
    duration_ms: float,
) -> tuple[float, float, float, float, float]:
    duration = max(1.0, float(duration_ms))
    offset = _clamp(float(anchors_ms[0]), 0.0, duration - 1.0)
    overlap = _clamp(float(anchors_ms[1]), offset, duration - 1.0)
    pre = _clamp(float(anchors_ms[2]), overlap, duration - 1.0)
    consonant = _clamp(float(anchors_ms[3]), pre, duration - 1.0)
    cutoff = _clamp(float(anchors_ms[4]), consonant + 1.0, duration)
    return (offset, overlap, pre, consonant, cutoff)


def validate_anchor_shape(
    anchors_ms: tuple[float, float, float, float, float],
    *,
    duration_ms: float,
) -> bool:
    duration = max(1.0, float(duration_ms))
    offset, overlap, pre, consonant, cutoff = [float(v) for v in anchors_ms]
    if not (0.0 <= offset <= overlap <= pre <= consonant < cutoff <= duration):
        return False
    if cutoff - offset < 8.0:
        return False
    if consonant - offset < 1.0:
        return False
    return True


def anchors_to_dataclass(
    anchors_ms: tuple[float, float, float, float, float],
    *,
    confidence: float,
    reason: str,
) -> AbsoluteOtoAnchors:
    return AbsoluteOtoAnchors(
        offset_abs=float(anchors_ms[0]),
        overlap_abs=float(anchors_ms[1]),
        pre_abs=float(anchors_ms[2]),
        consonant_abs=float(anchors_ms[3]),
        cutoff_abs=float(anchors_ms[4]),
        confidence=_clamp01(float(confidence)),
        reason=str(reason or "stage2"),
    )


def min_confidence_from_env(default: float = 0.55) -> float:
    return _env_float("UTOA_STAGE2_OTO_MIN_CONFIDENCE", default)


def max_candidate_distance_from_env(default: float = 250.0) -> float:
    return _env_float("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_MS", default)


def _base_anchor_tuple(row: DecodedOtoRow) -> tuple[float, float, float, float, float]:
    anchors = row.anchors
    return (
        float(anchors.offset_abs),
        float(anchors.overlap_abs),
        float(anchors.pre_abs),
        float(anchors.consonant_abs),
        float(anchors.cutoff_abs),
    )


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "anchors_to_dataclass",
    "apply_stage2_gate",
    "max_candidate_distance_from_env",
    "min_confidence_from_env",
    "repair_anchor_order",
    "validate_anchor_shape",
]
