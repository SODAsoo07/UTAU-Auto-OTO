from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple


ParamsTuple = Tuple[float, float, float, float, float]


@dataclass
class E2ESelectorConfig:
    mode: str = "hybrid"
    t_low: float = 0.52
    t_high: float = 0.72
    blend_alpha: float = 0.60


@dataclass
class E2ESelectionResult:
    params: ParamsTuple
    selected: str
    score: float
    alpha: float
    reason: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"legacy_only", "e2e_only", "hybrid"}:
        return mode
    return "hybrid"


def _normalize_thresholds(cfg: E2ESelectorConfig) -> tuple[float, float, float]:
    t_low = _clamp(_safe_float(cfg.t_low, 0.52), 0.0, 1.0)
    t_high = _clamp(_safe_float(cfg.t_high, 0.72), 0.0, 1.0)
    if t_high < t_low:
        t_high = t_low
    alpha = _clamp(_safe_float(cfg.blend_alpha, 0.60), 0.0, 1.0)
    return t_low, t_high, alpha


def _cutoff_abs(offset: float, cutoff: float) -> float:
    return float(offset) + abs(float(cutoff))


def _project_params(
    params: ParamsTuple,
    *,
    validate_fn: Callable[..., ParamsTuple] | None = None,
    alias_type: str = "",
) -> ParamsTuple:
    offset, cons, cutoff, pre, ovl = [float(x) for x in params]

    # OTO baseline constraints:
    # - offset: absolute position (do not couple with cons/pre numerically)
    # - consonant/pre/overlap: local timing values
    # - cutoff: should stay negative in common oto.ini conventions
    cons = max(cons, 0.0)
    pre = max(pre, 0.0)
    if pre > cons:
        pre = cons
    ovl = min(ovl, pre)
    cutoff = -abs(cutoff)

    if validate_fn is None:
        return float(offset), float(cons), float(cutoff), float(pre), float(ovl)

    try:
        out = validate_fn(offset, cons, cutoff, pre, ovl, alias_type=str(alias_type or "").strip().lower())
    except TypeError:
        out = validate_fn(offset, cons, cutoff, pre, ovl)
    return float(out[0]), float(out[1]), float(out[2]), float(out[3]), float(out[4])


def _blend_params(legacy: ParamsTuple, e2e: ParamsTuple, alpha: float) -> ParamsTuple:
    a = _clamp(alpha, 0.0, 1.0)
    o_l, c_l, cut_l, p_l, ovl_l = [float(x) for x in legacy]
    o_e, c_e, cut_e, p_e, ovl_e = [float(x) for x in e2e]

    cut_abs_l = _cutoff_abs(o_l, cut_l)
    cut_abs_e = _cutoff_abs(o_e, cut_e)

    offset = (1.0 - a) * o_l + a * o_e
    cons = (1.0 - a) * c_l + a * c_e
    pre = (1.0 - a) * p_l + a * p_e
    ovl = (1.0 - a) * ovl_l + a * ovl_e
    cut_abs = (1.0 - a) * cut_abs_l + a * cut_abs_e
    cutoff = -(cut_abs - offset)
    return float(offset), float(cons), float(cutoff), float(pre), float(ovl)


def select_e2e_or_legacy(
    *,
    legacy_params: ParamsTuple,
    e2e_params: ParamsTuple,
    score: float,
    config: E2ESelectorConfig | None = None,
    validate_fn: Callable[..., ParamsTuple] | None = None,
    alias_type: str = "",
) -> E2ESelectionResult:
    cfg = config or E2ESelectorConfig()
    mode = _normalize_mode(cfg.mode)
    t_low, t_high, alpha = _normalize_thresholds(cfg)
    s = _clamp(_safe_float(score, 0.0), 0.0, 1.0)

    legacy_projected = _project_params(legacy_params, validate_fn=validate_fn, alias_type=alias_type)
    e2e_projected = _project_params(e2e_params, validate_fn=validate_fn, alias_type=alias_type)

    if mode == "legacy_only":
        return E2ESelectionResult(
            params=legacy_projected,
            selected="legacy",
            score=s,
            alpha=0.0,
            reason="mode_legacy_only",
        )

    if mode == "e2e_only":
        return E2ESelectionResult(
            params=e2e_projected,
            selected="e2e",
            score=s,
            alpha=1.0,
            reason="mode_e2e_only",
        )

    if s >= t_high:
        return E2ESelectionResult(
            params=e2e_projected,
            selected="e2e",
            score=s,
            alpha=1.0,
            reason="score_high",
        )

    if s < t_low:
        return E2ESelectionResult(
            params=legacy_projected,
            selected="legacy",
            score=s,
            alpha=0.0,
            reason="score_low",
        )

    blended = _blend_params(legacy_projected, e2e_projected, alpha)
    blended = _project_params(blended, validate_fn=validate_fn, alias_type=alias_type)
    return E2ESelectionResult(
        params=blended,
        selected="blend",
        score=s,
        alpha=alpha,
        reason="score_mid",
    )


__all__ = [
    "E2ESelectionResult",
    "E2ESelectorConfig",
    "select_e2e_or_legacy",
]
