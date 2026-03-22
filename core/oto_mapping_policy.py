from __future__ import annotations


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def resolve_plan_policy(
    *,
    alignment_trust,
    plan_meta=None,
    expected_count=0,
    planned_count=0,
    format_type="",
    prefer_sequence=False,
):
    trust = _clamp01(alignment_trust)
    meta = dict(plan_meta or {})
    mean_score = float(meta.get("mean_score", 0.0) or 0.0)
    min_score = float(meta.get("min_score", 0.0) or 0.0)
    margin = float(meta.get("margin", 0.0) or 0.0)
    coverage = float(planned_count) / float(max(int(expected_count or 0), 1)) if expected_count else 0.0
    fmt = str(format_type or "").strip().lower()

    tier = "high"
    if trust < 0.55 or mean_score < 70.0 or min_score < 52.0 or coverage < 0.90 or margin < 8.0:
        tier = "mid"
    if trust < 0.40 or mean_score < 58.0 or min_score < 40.0 or coverage < 0.75 or margin < 3.0:
        tier = "low"

    allow_generation = True
    fallback_mode = ""
    if tier == "low":
        fallback_mode = "sequence_lock"
    if trust < 0.28 and coverage < 0.55 and fmt in {"cvvc", "vcv", "cvc", "cv"}:
        allow_generation = False
        fallback_mode = "abstain"

    confidence_cap = 0.96
    if tier == "mid":
        confidence_cap = min(confidence_cap, max(0.40, trust + 0.04))
    if tier == "low":
        confidence_cap = min(confidence_cap, max(0.22, trust + 0.02))
    if prefer_sequence:
        confidence_cap = min(confidence_cap, max(0.20, trust + 0.02))

    return {
        "tier": tier,
        "allow_generation": bool(allow_generation),
        "fallback_mode": str(fallback_mode or ""),
        "confidence_cap": float(confidence_cap),
        "coverage": float(coverage),
        "mean_score": mean_score,
        "min_score": min_score,
        "margin": margin,
    }


__all__ = ["resolve_plan_policy"]
