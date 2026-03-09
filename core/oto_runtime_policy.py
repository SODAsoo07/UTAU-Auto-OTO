from __future__ import annotations


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def resolve_runtime_mapping_policy(
    *,
    ingest_snapshot,
    plan_policy=None,
    mapping_confidence=0.0,
    conf_threshold=0.0,
    format_type="",
    score_a=None,
    score_b=None,
    low_conf_floor=0.58,
    sequence_lock_formats=None,
    abstain_formats=None,
    prefer_sequence=None,
):
    snapshot = ingest_snapshot
    plan = dict(plan_policy or {})
    fmt = str(format_type or "").strip().lower()
    sequence_lock_formats = {str(x).strip().lower() for x in (sequence_lock_formats or set())}
    abstain_formats = {str(x).strip().lower() for x in (abstain_formats or set())}

    trust_score = _clamp01(getattr(snapshot, "textgrid_trust_score", 0.0))
    trust_tier = str(getattr(snapshot, "textgrid_trust_tier", "low") or "low").strip().lower()
    prefer_seq = bool(getattr(snapshot, "prefer_filename_sequence", False)) if prefer_sequence is None else bool(prefer_sequence)
    trust_cap = max(0.0, min(1.0, trust_score + (0.04 if prefer_seq else 0.10)))
    confidence_cap = min(float(plan.get("confidence_cap", 1.0) or 1.0), trust_cap)
    capped_confidence = min(float(mapping_confidence or 0.0), float(confidence_cap))

    if str(plan.get("tier", "")).strip().lower() == "low":
        mapping_tier = "low"
    elif capped_confidence >= (float(conf_threshold or 0.0) + 0.12):
        mapping_tier = "high"
    elif capped_confidence >= float(conf_threshold or 0.0):
        mapping_tier = "mid"
    else:
        mapping_tier = "low"

    force_sequence_lock = bool(
        fmt in sequence_lock_formats
        and (
            mapping_tier == "low"
            or prefer_seq
            or str(plan.get("fallback_mode", "")).strip().lower() == "sequence_lock"
        )
    )
    should_abstain = bool(
        fmt in abstain_formats
        and not bool(plan.get("allow_generation", True))
    )

    score_a_ok = True if score_a is None else float(score_a) >= 48.0
    score_b_ok = True if score_b is None else float(score_b) >= 48.0
    is_low_conf = bool(
        capped_confidence < max(float(conf_threshold or 0.0), float(low_conf_floor))
        or ((score_a is not None and score_b is not None) and (not score_a_ok) and (not score_b_ok))
        or trust_tier == "low"
        or str(plan.get("tier", "")).strip().lower() == "low"
    )

    return {
        "mapping_confidence": float(capped_confidence),
        "trust_cap": float(trust_cap),
        "confidence_cap": float(confidence_cap),
        "mapping_tier": str(mapping_tier),
        "force_sequence_lock": bool(force_sequence_lock),
        "should_abstain": bool(should_abstain),
        "is_low_conf": bool(is_low_conf),
        "prefer_sequence": bool(prefer_seq),
    }


__all__ = ["resolve_runtime_mapping_policy"]
