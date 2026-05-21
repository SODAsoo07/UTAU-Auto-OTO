from __future__ import annotations

import os

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors, BoundaryCandidate, DecodedOtoRow
from core.coarse_crnn.stage2_oto.features import candidate_distance_for_anchors, nearest_candidate_for_field
from core.coarse_crnn.stage2_oto.types import ANCHOR_FIELDS
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
    conf = _clamp01(float(confidence))
    if conf < float(min_confidence):
        return Stage2Decision(
            anchors_ms=_base_anchor_tuple(row),
            confidence=conf,
            accepted=False,
            reason=f"stage2_fallback:low_confidence:{conf:.3f}",
        )
    repaired = repair_anchor_order(predicted_anchors_ms, duration_ms=duration)
    projected = project_anchors_to_boundary_candidates(
        row=row,
        anchors_ms=repaired,
        candidates=candidates,
        duration_ms=duration,
    )
    repaired = repair_anchor_order(projected, duration_ms=duration)
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
        reason="stage2:boundary_adapter",
        max_candidate_distance_ms=float(max_dist),
    )


def project_anchors_to_boundary_candidates(
    *,
    row: DecodedOtoRow,
    anchors_ms: tuple[float, float, float, float, float],
    candidates: list[BoundaryCandidate],
    duration_ms: float,
) -> tuple[float, float, float, float, float]:
    if not _env_bool("UTOA_STAGE2_OTO_BOUNDARY_PROJECT_ENABLE", True):
        return tuple(float(v) for v in anchors_ms)  # type: ignore[return-value]
    if not candidates:
        return tuple(float(v) for v in anchors_ms)  # type: ignore[return-value]
    role = normalize_role(row.spec.role)
    duration = max(1.0, float(duration_ms))
    field_caps = _field_residual_caps_from_env()
    pb_scale = _clamp(_env_float("UTOA_STAGE2_OTO_PB_RESIDUAL_SCALE", 0.65), 0.05, 2.0)
    projected: list[float] = []
    for field, predicted in zip(ANCHOR_FIELDS, anchors_ms):
        cand = nearest_candidate_for_field(
            field=field,
            role=role,
            anchor_ms=float(predicted),
            candidates=candidates,
            duration_ms=duration,
        )
        if cand is None:
            projected.append(float(predicted))
            continue
        cap = float(field_caps.get(field, field_caps.get("default", 120.0)))
        if "model:phoneme_boundary" in str(cand.source or ""):
            cap *= pb_scale
        residual = _clamp(float(predicted) - float(cand.time_ms), -cap, cap)
        projected.append(float(cand.time_ms) + residual)
    return (
        float(projected[0]),
        float(projected[1]),
        float(projected[2]),
        float(projected[3]),
        float(projected[4]),
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


def min_confidence_from_env(default: float = 0.75) -> float:
    return _env_float("UTOA_STAGE2_OTO_MIN_CONFIDENCE", default)


def max_candidate_distance_from_env(default: float = 70.0) -> float:
    return _env_float("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_MS", default)


def max_candidate_distance_policy_from_env(default: float = 70.0) -> dict[str, object]:
    return {
        "default": max_candidate_distance_from_env(default),
        "by_role": _parse_named_float_map("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_ROLE"),
        "by_format": _parse_named_float_map("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_FORMAT"),
        "by_format_role": _parse_format_role_float_map("UTOA_STAGE2_OTO_MAX_CANDIDATE_DIST_BY_FORMAT_ROLE"),
    }


def resolve_max_candidate_distance_ms(*, role: object, format_type: object, policy: dict[str, object] | None) -> float:
    data = dict(policy or {})
    default = max(1.0, float(data.get("default", 250.0) or 250.0))
    by_role_raw = data.get("by_role", {})
    by_format_raw = data.get("by_format", {})
    by_format_role_raw = data.get("by_format_role", {})
    by_role = by_role_raw if isinstance(by_role_raw, dict) else {}
    by_format = by_format_raw if isinstance(by_format_raw, dict) else {}
    by_format_role = by_format_role_raw if isinstance(by_format_role_raw, dict) else {}
    role_key = normalize_role(role)
    format_key = _normalize_key(format_type, fallback="other")
    pair_key = (format_key, role_key)
    value = by_format_role.get(pair_key)
    if value is None:
        value = by_role.get(role_key)
    if value is None:
        value = by_format.get(format_key)
    try:
        return max(1.0, float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


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


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _parse_named_float_map(name: str) -> dict[str, float]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return {}
    out: dict[str, float] = {}
    for token in raw.replace(";", ",").split(","):
        text = str(token or "").strip()
        if not text or "=" not in text:
            continue
        key_raw, val_raw = text.split("=", 1)
        key = _normalize_key(key_raw, fallback="")
        if not key:
            continue
        try:
            out[key] = max(1.0, float(val_raw))
        except ValueError:
            continue
    return out


def _field_residual_caps_from_env() -> dict[str, float]:
    caps: dict[str, float] = {
        "default": max(1.0, _env_float("UTOA_STAGE2_OTO_MAX_RESIDUAL_MS", 70.0)),
        "offset_abs": 70.0,
        "overlap_abs": 60.0,
        "pre_abs": 50.0,
        "consonant_abs": 70.0,
        "cutoff_abs": 120.0,
    }
    raw = _parse_named_float_map("UTOA_STAGE2_OTO_MAX_RESIDUAL_BY_FIELD")
    for key, value in raw.items():
        if key in set(ANCHOR_FIELDS) | {"default"}:
            caps[key] = max(1.0, float(value))
    return caps


def _parse_format_role_float_map(name: str) -> dict[tuple[str, str], float]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return {}
    out: dict[tuple[str, str], float] = {}
    for token in raw.replace(";", ",").split(","):
        text = str(token or "").strip()
        if not text or "=" not in text:
            continue
        key_raw, val_raw = text.split("=", 1)
        key_txt = str(key_raw or "").strip()
        if "/" in key_txt:
            fmt_raw, role_raw = key_txt.split("/", 1)
        elif ":" in key_txt:
            fmt_raw, role_raw = key_txt.split(":", 1)
        else:
            continue
        fmt = _normalize_key(fmt_raw, fallback="")
        role = normalize_role(role_raw)
        if not fmt or not role:
            continue
        try:
            out[(fmt, role)] = max(1.0, float(val_raw))
        except ValueError:
            continue
    return out


def _normalize_key(value: object, *, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text if text else str(fallback or "")


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


__all__ = [
    "anchors_to_dataclass",
    "apply_stage2_gate",
    "max_candidate_distance_policy_from_env",
    "max_candidate_distance_from_env",
    "min_confidence_from_env",
    "project_anchors_to_boundary_candidates",
    "repair_anchor_order",
    "resolve_max_candidate_distance_ms",
    "validate_anchor_shape",
]
