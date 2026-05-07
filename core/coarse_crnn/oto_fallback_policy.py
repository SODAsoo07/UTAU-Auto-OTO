from __future__ import annotations

from dataclasses import dataclass
import os


_ROLE_BONUS = {
    "vc": 0.18,
    "vv": 0.12,
    "v-cv": 0.10,
    "-cv": 0.08,
}

_TRANSITION_BONUS = {
    "multi": 0.12,
    "cc": 0.08,
    "vc": 0.05,
    "vv": 0.04,
}


@dataclass(frozen=True)
class FallbackSignals:
    confidence: float
    min_anchor_conf: float
    max_anchor_conf: float
    conf_span: float
    alias_type: str
    alias_role: str
    transition_type: str
    row_ratio: float
    is_special: bool


@dataclass(frozen=True)
class FallbackPolicy:
    enabled: bool = False
    score_threshold: float = 0.65
    source_blend: float = 0.60
    conf_max: float = 0.38
    min_anchor_conf_max: float = 0.30
    conf_span_max: float = 0.12
    tail_row_min: float = 0.70
    conf_weight: float = 0.55
    min_anchor_weight: float = 0.35
    span_weight: float = 0.25
    tail_bonus: float = 0.08
    special_bonus: float = 0.08


@dataclass(frozen=True)
class FallbackDecision:
    apply: bool
    score: float
    reasons: tuple[str, ...]


def signals_from_prediction(
    *,
    confidence: float,
    heatmap_confidence: dict[str, float] | None,
    alias_type: object,
    alias_role: object,
    transition_type: object,
    row_ratio: float,
    is_special: bool,
) -> FallbackSignals:
    values = [float(v) for v in (heatmap_confidence or {}).values()]
    if values:
        min_conf = min(values)
        max_conf = max(values)
    else:
        min_conf = float(confidence)
        max_conf = float(confidence)
    return FallbackSignals(
        confidence=_clamp01(float(confidence)),
        min_anchor_conf=_clamp01(float(min_conf)),
        max_anchor_conf=_clamp01(float(max_conf)),
        conf_span=_clamp01(float(max_conf) - float(min_conf)),
        alias_type=str(alias_type or "").strip().lower() or "other",
        alias_role=str(alias_role or "").strip().lower() or "other",
        transition_type=str(transition_type or "").strip().lower() or "other",
        row_ratio=max(0.0, min(1.0, float(row_ratio))),
        is_special=bool(is_special),
    )


def policy_from_env() -> FallbackPolicy:
    return FallbackPolicy(
        enabled=_env_bool("UTOA_OTO_CRNN_FALLBACK_ENABLE", False),
        score_threshold=_env_float("UTOA_OTO_CRNN_FALLBACK_SCORE_THRESHOLD", 0.65),
        source_blend=_env_float("UTOA_OTO_CRNN_FALLBACK_SOURCE_BLEND", 0.60),
        conf_max=_env_float("UTOA_OTO_CRNN_FALLBACK_CONF_MAX", 0.38),
        min_anchor_conf_max=_env_float("UTOA_OTO_CRNN_FALLBACK_MIN_ANCHOR_CONF_MAX", 0.30),
        conf_span_max=_env_float("UTOA_OTO_CRNN_FALLBACK_CONF_SPAN_MAX", 0.12),
        tail_row_min=_env_float("UTOA_OTO_CRNN_FALLBACK_TAIL_ROW_MIN", 0.70),
        conf_weight=_env_float("UTOA_OTO_CRNN_FALLBACK_CONF_WEIGHT", 0.55),
        min_anchor_weight=_env_float("UTOA_OTO_CRNN_FALLBACK_MIN_ANCHOR_WEIGHT", 0.35),
        span_weight=_env_float("UTOA_OTO_CRNN_FALLBACK_SPAN_WEIGHT", 0.25),
        tail_bonus=_env_float("UTOA_OTO_CRNN_FALLBACK_TAIL_BONUS", 0.08),
        special_bonus=_env_float("UTOA_OTO_CRNN_FALLBACK_SPECIAL_BONUS", 0.08),
    )


def decide_fallback(signals: FallbackSignals, policy: FallbackPolicy) -> FallbackDecision:
    if not policy.enabled:
        return FallbackDecision(apply=False, score=0.0, reasons=())
    score = 0.0
    reasons: list[str] = []
    if signals.confidence <= float(policy.conf_max):
        score += _scaled_gap(signals.confidence, float(policy.conf_max)) * float(policy.conf_weight)
        reasons.append("low_conf")
    if signals.min_anchor_conf <= float(policy.min_anchor_conf_max):
        score += _scaled_gap(signals.min_anchor_conf, float(policy.min_anchor_conf_max)) * float(policy.min_anchor_weight)
        reasons.append("low_anchor_conf")
    if signals.conf_span <= float(policy.conf_span_max):
        score += _scaled_gap(signals.conf_span, float(policy.conf_span_max)) * float(policy.span_weight)
        reasons.append("flat_anchor_conf")
    if signals.row_ratio >= float(policy.tail_row_min):
        score += max(0.0, float(policy.tail_bonus))
        reasons.append("tail_row")
    role_bonus = float(_ROLE_BONUS.get(signals.alias_role, 0.0))
    if role_bonus > 0.0:
        score += role_bonus
        reasons.append(f"role:{signals.alias_role}")
    trans_bonus = float(_TRANSITION_BONUS.get(signals.transition_type, 0.0))
    if trans_bonus > 0.0:
        score += trans_bonus
        reasons.append(f"transition:{signals.transition_type}")
    if signals.is_special:
        score += max(0.0, float(policy.special_bonus))
        reasons.append("special_alias")
    apply = score >= max(0.01, float(policy.score_threshold))
    return FallbackDecision(
        apply=bool(apply),
        score=float(score),
        reasons=tuple(reasons),
    )


def blend_with_source(predicted: dict[str, float], source: dict[str, float], *, source_blend: float) -> dict[str, float]:
    ratio = max(0.0, min(1.0, float(source_blend)))
    if ratio <= 0.0:
        return {k: float(v) for k, v in dict(predicted).items()}
    out = {}
    for key in ("offset", "consonant", "preutterance", "overlap"):
        pred = float(predicted.get(key, 0.0))
        src = float(source.get(key, pred))
        out[key] = ((1.0 - ratio) * pred) + (ratio * src)
    pred_cut = float(predicted.get("cutoff", 0.0))
    src_cut = float(source.get("cutoff", pred_cut))
    out["cutoff"] = ((1.0 - ratio) * pred_cut) + (ratio * src_cut)
    # UTAU cutoff is conventionally non-positive.
    if out["cutoff"] > 0.0:
        out["cutoff"] = -abs(out["cutoff"])
    # Keep overlap <= preutterance.
    out["preutterance"] = max(0.0, float(out["preutterance"]))
    out["overlap"] = max(0.0, min(float(out["overlap"]), float(out["preutterance"])))
    out["offset"] = max(0.0, float(out["offset"]))
    out["consonant"] = max(0.0, float(out["consonant"]))
    return out


def _scaled_gap(value: float, max_value: float) -> float:
    cap = max(1e-6, float(max_value))
    v = max(0.0, float(value))
    if v >= cap:
        return 0.0
    return max(0.0, min(1.0, (cap - v) / cap))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except Exception:
        return float(default)


__all__ = [
    "FallbackDecision",
    "FallbackPolicy",
    "FallbackSignals",
    "blend_with_source",
    "decide_fallback",
    "policy_from_env",
    "signals_from_prediction",
]
