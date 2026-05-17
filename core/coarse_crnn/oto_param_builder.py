from __future__ import annotations

import os

from core.coarse_crnn.alias_role import normalize_role
from core.coarse_crnn.boundary_types import AbsoluteOtoAnchors
from core.coarse_crnn.boundary_targets import absolute_anchors_to_oto_params
from core.coarse_crnn.boundary_types import TRANSITION_ROLES


ROLE_GAP_RULES_MS: dict[str, tuple[float, float, float, float]] = {
    # Keep conservative tails to avoid over-long cutoff in practical CV/CVVC.
    "cv": (14.0, 62.0, 20.0, 86.0),
    "v": (4.0, 28.0, 10.0, 62.0),
    "vc": (8.0, 38.0, 14.0, 62.0),
    "vv": (10.0, 52.0, 18.0, 76.0),
    "v-cv": (12.0, 56.0, 20.0, 82.0),
    "br": (4.0, 34.0, 12.0, 64.0),
    "other": (8.0, 48.0, 14.0, 72.0),
}

ROLE_RIGHT_GUARD_MARGIN_MS: dict[str, float] = {
    "cv": 10.0,
    "v": 8.0,
    "vc": 6.0,
    "vv": 6.0,
    "v-cv": 6.0,
    "br": 6.0,
    "other": 8.0,
}

ROLE_MAX_SPAN_MS: dict[str, float] = {
    "cv": 210.0,
    "v": 185.0,
    "vc": 170.0,
    "vv": 170.0,
    "v-cv": 180.0,
    "br": 120.0,
    "other": 190.0,
}

ROLE_MIN_PRE_CONS_GAP_MS: dict[str, float] = {
    "cv": 20.0,
    "v": 14.0,
    "vc": 18.0,
    "vv": 16.0,
    "v-cv": 18.0,
    "br": 6.0,
    "other": 16.0,
}

ROLE_MAX_PRE_CONS_GAP_MS: dict[str, float] = {
    # Upper guard to avoid VC/V-CV rows behaving like wide CV anchors.
    "cv": 180.0,
    "v": 120.0,
    "vc": 66.0,
    "vv": 84.0,
    "v-cv": 90.0,
    "br": 40.0,
    "other": 130.0,
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
    cons_min, cons_max, tail_min, tail_max = _resolve_gap_caps(role_key)

    center = _clamp(float(center_ms), active_start, active_end)
    left = float(left_anchor_ms) if left_anchor_ms is not None else active_start
    right = float(right_anchor_ms) if right_anchor_ms is not None else active_end
    left = _clamp(left, 0.0, duration)
    right = _clamp(right, left, duration)

    cons_gap = _clamp(0.45 * (right - left), cons_min, cons_max)
    tail_gap = _clamp(0.60 * (right - left), tail_min, tail_max)

    offset_lead_scale = _env_float("UTOA_BOUNDARY_OFFSET_LEAD_SCALE", 0.92)
    offset_lead_min = _env_float("UTOA_BOUNDARY_OFFSET_LEAD_MIN_MS", 30.0)
    offset_lead_max = _env_float("UTOA_BOUNDARY_OFFSET_LEAD_MAX_MS", 92.0)

    if role_key in {"vc", "vv", "v-cv"}:
        # Transition rows were often perceived late in listening tests.
        # Pull preutterance earlier than the transition center by a span-based lead.
        transition_pre_lead = _resolve_transition_pre_lead_ms(left_anchor_ms=left, right_anchor_ms=right)
        pre_abs = _clamp(center - transition_pre_lead, active_start, active_end)
        if role_key == "vc":
            # VC rows often include too much preceding vowel when offset lead is
            # too large. Keep a tighter lead for VC only.
            vc_offset_left_pad = _env_float("UTOA_BOUNDARY_VC_OFFSET_LEFT_PAD_MS", 14.0)
            vc_offset_pre_lead = _env_float("UTOA_BOUNDARY_VC_OFFSET_PRE_LEAD_MS", 24.0)
            offset_abs = _clamp(
                max(left - max(0.0, vc_offset_left_pad), pre_abs - max(1.0, vc_offset_pre_lead)),
                active_start,
                pre_abs,
            )
        else:
            offset_abs = _clamp(max(left - 24.0, pre_abs - 40.0), active_start, pre_abs)
    elif role_key in {"br", "endbr"}:
        pre_abs = _clamp(center, active_start, active_end)
        offset_abs = _clamp(pre_abs - 14.0, active_start, pre_abs)
        cons_gap = _clamp(cons_gap, 4.0, 30.0)
        tail_gap = _clamp(tail_gap, 10.0, 58.0)
    else:
        pre_abs = _clamp(center, active_start, active_end)
        # CV/anchor roles were consistently late by ~30-70ms in practice.
        # Lead offset by most of the modeled consonant span, then clamp.
        lead = _clamp((cons_gap * max(0.0, offset_lead_scale)) + 4.0, offset_lead_min, offset_lead_max)
        offset_abs = _clamp(pre_abs - lead, active_start, pre_abs)

    overlap_abs = _clamp(pre_abs - max(4.0, 0.35 * cons_gap), offset_abs, pre_abs)
    consonant_abs = _clamp(pre_abs + cons_gap, pre_abs + 1.0, duration - 1.0)
    cutoff_abs = _clamp(consonant_abs + tail_gap, consonant_abs + 1.0, duration)

    anchors = AbsoluteOtoAnchors(
        offset_abs=offset_abs,
        overlap_abs=overlap_abs,
        pre_abs=pre_abs,
        consonant_abs=consonant_abs,
        cutoff_abs=cutoff_abs,
        confidence=max(0.0, min(1.0, float(confidence))),
        reason=str(reason or role_key),
    )
    guarded = apply_right_boundary_guard(
        anchors=anchors,
        role=role_key,
        duration_ms=duration,
        right_anchor_ms=right_anchor_ms,
    )
    span_guarded = apply_span_length_guard(
        anchors=guarded,
        role=role_key,
        duration_ms=duration,
    )
    return apply_pre_cons_gap_guard(
        anchors=span_guarded,
        role=role_key,
        duration_ms=duration,
        active_start_ms=active_start,
    )


def apply_right_boundary_guard(
    *,
    anchors: AbsoluteOtoAnchors,
    role: str,
    duration_ms: float,
    right_anchor_ms: float | None,
) -> AbsoluteOtoAnchors:
    if right_anchor_ms is None or not _env_bool("UTOA_BOUNDARY_CUTOFF_RIGHT_GUARD_ENABLE", True):
        return anchors
    role_key = normalize_role(role)
    margin = _env_float(
        "UTOA_BOUNDARY_CUTOFF_RIGHT_MARGIN_MS",
        ROLE_RIGHT_GUARD_MARGIN_MS.get(role_key, ROLE_RIGHT_GUARD_MARGIN_MS["other"]),
    )
    duration = max(1.0, float(duration_ms))
    right = _clamp(float(right_anchor_ms), 0.0, duration)
    upper = min(duration, right - max(0.0, float(margin)))
    if upper <= 4.0:
        return anchors

    offset_abs = _clamp(float(anchors.offset_abs), 0.0, duration)
    overlap_abs = _clamp(float(anchors.overlap_abs), offset_abs, duration)
    pre_abs = _clamp(float(anchors.pre_abs), overlap_abs, duration)
    consonant_abs = _clamp(float(anchors.consonant_abs), pre_abs + 1.0, duration)
    cutoff_abs = _clamp(float(anchors.cutoff_abs), consonant_abs + 1.0, duration)

    # For transition roles, keep pre position stable to avoid boundary drift.
    if role_key in TRANSITION_ROLES and _env_bool("UTOA_BOUNDARY_TRANSITION_SOFT_RIGHT_GUARD_ENABLE", True):
        upper = min(duration, max(pre_abs + 2.0, right - max(0.0, float(margin))))
        cutoff_abs = min(cutoff_abs, upper)
        consonant_abs = min(consonant_abs, cutoff_abs - 1.0)
        consonant_abs = max(pre_abs + 1.0, consonant_abs)
        cutoff_abs = max(consonant_abs + 1.0, cutoff_abs)
        return AbsoluteOtoAnchors(
            offset_abs=offset_abs,
            overlap_abs=overlap_abs,
            pre_abs=pre_abs,
            consonant_abs=consonant_abs,
            cutoff_abs=cutoff_abs,
            confidence=float(anchors.confidence),
            reason=str(anchors.reason),
        )

    # Right-bound compaction: keep pre -> consonant -> cutoff inside next-anchor guard.
    cutoff_abs = min(cutoff_abs, upper)
    consonant_abs = min(consonant_abs, cutoff_abs - 1.0)
    pre_abs = min(pre_abs, consonant_abs - 1.0)
    overlap_abs = min(overlap_abs, pre_abs)
    offset_abs = min(offset_abs, pre_abs)

    # Re-apply invariants in order.
    offset_abs = _clamp(offset_abs, 0.0, duration)
    overlap_abs = _clamp(overlap_abs, offset_abs, duration)
    pre_abs = _clamp(pre_abs, overlap_abs, duration)
    consonant_abs = _clamp(consonant_abs, pre_abs + 1.0, duration)
    cutoff_abs = _clamp(cutoff_abs, consonant_abs + 1.0, duration)
    return AbsoluteOtoAnchors(
        offset_abs=offset_abs,
        overlap_abs=overlap_abs,
        pre_abs=pre_abs,
        consonant_abs=consonant_abs,
        cutoff_abs=cutoff_abs,
        confidence=float(anchors.confidence),
        reason=str(anchors.reason),
    )


def anchors_to_runtime_oto_params(anchors: AbsoluteOtoAnchors, *, duration_ms: float) -> dict[str, float]:
    return absolute_anchors_to_oto_params(anchors, duration_ms=duration_ms)


def stretch_risk_flags(params: dict[str, float], *, duration_ms: float | None = None) -> dict[str, bool]:
    pre = max(1.0, float(params.get("preutterance", 0.0) or 0.0))
    cons = max(pre, float(params.get("consonant", 0.0) or 0.0))
    cutoff = float(params.get("cutoff", 0.0) or 0.0)
    if duration_ms is not None and cutoff < 0.0:
        cutoff_abs = max(cons, float(duration_ms) + cutoff)
    else:
        cutoff_abs = cons + abs(cutoff)
    cons_gap = max(0.0, cons - pre)
    tail_gap = max(0.0, cutoff_abs - cons)
    overlap = max(0.0, float(params.get("overlap", 0.0) or 0.0))
    overlap_ratio = overlap / max(1.0, pre)
    return {
        "cons_gap_outlier": bool(cons_gap > 120.0),
        "tail_gap_outlier": bool(tail_gap > 170.0),
        "overlap_outlier": bool(overlap_ratio > 0.92),
    }


def apply_span_length_guard(
    *,
    anchors: AbsoluteOtoAnchors,
    role: str,
    duration_ms: float,
) -> AbsoluteOtoAnchors:
    role_key = normalize_role(role)
    max_span = _env_float(
        "UTOA_BOUNDARY_MAX_SPAN_MS",
        ROLE_MAX_SPAN_MS.get(role_key, ROLE_MAX_SPAN_MS["other"]),
    )
    duration = max(1.0, float(duration_ms))
    offset_abs = _clamp(float(anchors.offset_abs), 0.0, duration)
    overlap_abs = _clamp(float(anchors.overlap_abs), offset_abs, duration)
    pre_abs = _clamp(float(anchors.pre_abs), overlap_abs, duration)
    consonant_abs = _clamp(float(anchors.consonant_abs), pre_abs + 1.0, duration)
    cutoff_abs = _clamp(float(anchors.cutoff_abs), consonant_abs + 1.0, duration)
    span = cutoff_abs - offset_abs
    if span <= max(8.0, max_span):
        return AbsoluteOtoAnchors(
            offset_abs=offset_abs,
            overlap_abs=overlap_abs,
            pre_abs=pre_abs,
            consonant_abs=consonant_abs,
            cutoff_abs=cutoff_abs,
            confidence=float(anchors.confidence),
            reason=str(anchors.reason),
        )
    cutoff_abs = min(cutoff_abs, offset_abs + max_span)
    consonant_abs = min(consonant_abs, cutoff_abs - 1.0)
    pre_abs = min(pre_abs, consonant_abs - 1.0)
    overlap_abs = min(overlap_abs, pre_abs)
    return AbsoluteOtoAnchors(
        offset_abs=_clamp(offset_abs, 0.0, duration),
        overlap_abs=_clamp(overlap_abs, offset_abs, duration),
        pre_abs=_clamp(pre_abs, overlap_abs, duration),
        consonant_abs=_clamp(consonant_abs, pre_abs + 1.0, duration),
        cutoff_abs=_clamp(cutoff_abs, consonant_abs + 1.0, duration),
        confidence=float(anchors.confidence),
        reason=str(anchors.reason),
    )


def apply_pre_cons_gap_guard(
    *,
    anchors: AbsoluteOtoAnchors,
    role: str,
    duration_ms: float,
    active_start_ms: float = 0.0,
) -> AbsoluteOtoAnchors:
    """Keep a practical min/max gap between preutterance and consonant."""
    role_key = normalize_role(role)
    duration = max(1.0, float(duration_ms))
    min_gap = _resolve_min_pre_cons_gap_ms(role_key)
    max_gap = _resolve_max_pre_cons_gap_ms(role_key)

    offset_abs = _clamp(float(anchors.offset_abs), 0.0, duration)
    overlap_abs = _clamp(float(anchors.overlap_abs), offset_abs, duration)
    pre_abs = _clamp(float(anchors.pre_abs), overlap_abs, duration)
    consonant_abs = _clamp(float(anchors.consonant_abs), pre_abs + 1.0, duration)
    cutoff_abs = _clamp(float(anchors.cutoff_abs), consonant_abs + 1.0, duration)

    gap = float(consonant_abs - pre_abs)
    if gap + 1e-6 >= float(min_gap):
        return AbsoluteOtoAnchors(
            offset_abs=offset_abs,
            overlap_abs=overlap_abs,
            pre_abs=pre_abs,
            consonant_abs=consonant_abs,
            cutoff_abs=cutoff_abs,
            confidence=float(anchors.confidence),
            reason=str(anchors.reason),
        )

    deficit = float(min_gap - gap)
    pre_floor = max(float(active_start_ms), float(overlap_abs), float(offset_abs))
    pull_back = min(deficit, max(0.0, pre_abs - pre_floor))
    pre_abs -= pull_back

    # If still short, extend consonant forward (and cutoff when needed).
    cons_target = pre_abs + float(min_gap)
    if consonant_abs < cons_target:
        consonant_abs = min(duration - 1.0, cons_target)
    if cutoff_abs < consonant_abs + 1.0:
        cutoff_abs = min(duration, consonant_abs + 1.0)

    # Hard clamp upper gap for transition-heavy rows to prevent CV-like spreads.
    gap = float(consonant_abs - pre_abs)
    if gap > float(max_gap) + 1e-6:
        consonant_abs = min(duration - 1.0, pre_abs + float(max_gap))
        if cutoff_abs < consonant_abs + 1.0:
            cutoff_abs = min(duration, consonant_abs + 1.0)

    # Re-apply invariants.
    offset_abs = _clamp(offset_abs, 0.0, duration)
    overlap_abs = _clamp(overlap_abs, offset_abs, duration)
    pre_abs = _clamp(pre_abs, overlap_abs, duration)
    consonant_abs = _clamp(consonant_abs, pre_abs + 1.0, duration)
    cutoff_abs = _clamp(cutoff_abs, consonant_abs + 1.0, duration)
    return AbsoluteOtoAnchors(
        offset_abs=offset_abs,
        overlap_abs=overlap_abs,
        pre_abs=pre_abs,
        consonant_abs=consonant_abs,
        cutoff_abs=cutoff_abs,
        confidence=float(anchors.confidence),
        reason=str(anchors.reason),
    )


def _resolve_gap_caps(role_key: str) -> tuple[float, float, float, float]:
    """Return (cons_min, cons_max, tail_min, tail_max) with env-var overrides.

    The role-specific defaults in `ROLE_GAP_RULES_MS` are conservative for
    short-syllable Japanese CV recordings (cv tail max=86ms). Multi-mora
    Korean CVVC recordings or any voicebank where the user's manual OTO uses
    longer consonant/tail regions can dial the caps up globally without
    rewriting the role table.

    Env vars (all optional, fall back to per-role defaults):
      - `UTOA_BOUNDARY_CONS_MIN_MS`  — consonant region (pre→cons) minimum
      - `UTOA_BOUNDARY_CONS_MAX_MS`  — consonant region maximum
      - `UTOA_BOUNDARY_TAIL_MIN_MS`  — tail (cons→cutoff) minimum
      - `UTOA_BOUNDARY_TAIL_MAX_MS`  — tail maximum
    """
    cons_min_d, cons_max_d, tail_min_d, tail_max_d = ROLE_GAP_RULES_MS.get(
        role_key, ROLE_GAP_RULES_MS["other"]
    )
    cons_min = _env_float("UTOA_BOUNDARY_CONS_MIN_MS", cons_min_d)
    cons_max = _env_float("UTOA_BOUNDARY_CONS_MAX_MS", cons_max_d)
    tail_min = _env_float("UTOA_BOUNDARY_TAIL_MIN_MS", tail_min_d)
    tail_max = _env_float("UTOA_BOUNDARY_TAIL_MAX_MS", tail_max_d)
    # Make sure mins don't exceed maxes after override.
    cons_max = max(cons_min + 1.0, cons_max)
    tail_max = max(tail_min + 1.0, tail_max)
    return cons_min, cons_max, tail_min, tail_max


def _resolve_transition_pre_lead_ms(*, left_anchor_ms: float, right_anchor_ms: float) -> float:
    span = max(1.0, float(right_anchor_ms) - float(left_anchor_ms))
    # Default OFF: aggressive lead hurts offset stability on some KO CVVC banks.
    # Enable/tune per bank via env when needed.
    scale = _env_float("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_SCALE", 0.0)
    lead_min = _env_float("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MIN_MS", 0.0)
    lead_max = _env_float("UTOA_BOUNDARY_TRANSITION_PRE_LEAD_MAX_MS", 0.0)
    return _clamp(span * max(0.0, scale), min(lead_min, lead_max), max(lead_min, lead_max))


def _resolve_min_pre_cons_gap_ms(role_key: str) -> float:
    default_gap = ROLE_MIN_PRE_CONS_GAP_MS.get(role_key, ROLE_MIN_PRE_CONS_GAP_MS["other"])
    base = _env_float("UTOA_BOUNDARY_PRE_CONS_MIN_GAP_MS", default_gap)
    if role_key in TRANSITION_ROLES:
        base = _env_float("UTOA_BOUNDARY_TRANSITION_PRE_CONS_MIN_GAP_MS", base)
    return max(1.0, float(base))


def _resolve_max_pre_cons_gap_ms(role_key: str) -> float:
    default_gap = ROLE_MAX_PRE_CONS_GAP_MS.get(role_key, ROLE_MAX_PRE_CONS_GAP_MS["other"])
    base = _env_float("UTOA_BOUNDARY_PRE_CONS_MAX_GAP_MS", default_gap)
    if role_key in TRANSITION_ROLES:
        base = _env_float("UTOA_BOUNDARY_TRANSITION_PRE_CONS_MAX_GAP_MS", base)
    return max(4.0, float(base))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


__all__ = [
    "ROLE_GAP_RULES_MS",
    "ROLE_MIN_PRE_CONS_GAP_MS",
    "ROLE_MAX_PRE_CONS_GAP_MS",
    "ROLE_RIGHT_GUARD_MARGIN_MS",
    "ROLE_MAX_SPAN_MS",
    "apply_pre_cons_gap_guard",
    "_resolve_gap_caps",
    "apply_right_boundary_guard",
    "apply_span_length_guard",
    "anchors_to_runtime_oto_params",
    "build_absolute_anchors_for_role",
    "stretch_risk_flags",
]
