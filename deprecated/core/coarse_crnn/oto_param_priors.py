from __future__ import annotations

from typing import Any

from core.coarse_crnn.alias_role import normalize_role


# Role-based prior table. Numbers chosen from the empirical CV/CVC/CVVC/VCV
# tuning that lives in `oto_timing_prior` below, but indexed on a single
# alias-role axis so the same row gets the same prior regardless of the
# voicebank format wrapper.
_ROLE_PRIOR_TABLE: dict[str, dict[str, float]] = {
    "-cv": dict(cons_gap=58.0, tail=96.0, min_cons_gap=18.0, min_tail=36.0, max_cons_gap=92.0, max_tail=150.0),
    "cv":  dict(cons_gap=48.0, tail=82.0, min_cons_gap=16.0, min_tail=34.0, max_cons_gap=78.0, max_tail=130.0),
    "cv-": dict(cons_gap=34.0, tail=70.0, min_cons_gap=10.0, min_tail=24.0, max_cons_gap=60.0, max_tail=110.0),
    "v":   dict(cons_gap=18.0, tail=48.0, min_cons_gap=4.0,  min_tail=14.0, max_cons_gap=36.0, max_tail=92.0),
    "v-":  dict(cons_gap=18.0, tail=58.0, min_cons_gap=4.0,  min_tail=18.0, max_cons_gap=36.0, max_tail=104.0),
    # vc/vv priors are set to the tighter cvvc-format values since most vc/vv
    # aliases live in cvvc voicebanks. alias_timing_prior("vc") used 28/52 but
    # oto_timing_prior(cvvc,vc) is 24/46, matching real-world cvvc data better.
    "vc":  dict(cons_gap=24.0, tail=46.0, min_cons_gap=8.0,  min_tail=18.0, max_cons_gap=42.0, max_tail=70.0),
    "vv":  dict(cons_gap=38.0, tail=62.0, min_cons_gap=12.0, min_tail=22.0, max_cons_gap=66.0, max_tail=100.0),
    "v-cv":dict(cons_gap=48.0, tail=74.0, min_cons_gap=16.0, min_tail=30.0, max_cons_gap=82.0, max_tail=118.0),
    "endbr":dict(cons_gap=20.0, tail=44.0, min_cons_gap=6.0, min_tail=16.0, max_cons_gap=40.0, max_tail=78.0),
    "br":  dict(cons_gap=16.0, tail=36.0, min_cons_gap=4.0,  min_tail=12.0, max_cons_gap=34.0, max_tail=64.0),
    "other":dict(cons_gap=48.0, tail=82.0, min_cons_gap=14.0, min_tail=30.0, max_cons_gap=82.0, max_tail=128.0),
    # Special phonemes are user-tagged. Treat the special token as a
    # consonant — same range as a generic CV until per-bank tuning lands.
    "special":dict(cons_gap=52.0, tail=86.0, min_cons_gap=18.0, min_tail=32.0, max_cons_gap=88.0, max_tail=134.0),
}

# Diphthongs lengthen the onset region. Multipliers are mild on purpose so
# guarded models don't lurch on edge cases.
_DIPHTHONG_CONS_GAIN = 1.30
_DIPHTHONG_TAIL_GAIN = 1.20


def oto_role_prior(
    alias_role: object,
    *,
    is_diphthong: bool = False,
    is_special: bool = False,
) -> dict[str, float]:
    """Return prior timing values keyed on the role axis.

    The returned dict has the same keys as `oto_timing_prior` so downstream
    decode/clamp code can swap in either source.
    """
    role = "special" if bool(is_special) else normalize_role(alias_role)
    base = _ROLE_PRIOR_TABLE.get(role, _ROLE_PRIOR_TABLE["other"])
    prior = dict(base)
    if bool(is_diphthong) and role in {"cv", "-cv", "cv-", "v-cv", "vv"}:
        prior["cons_gap"] = prior["cons_gap"] * _DIPHTHONG_CONS_GAIN
        prior["tail"] = prior["tail"] * _DIPHTHONG_TAIL_GAIN
        prior["max_cons_gap"] = prior["max_cons_gap"] * _DIPHTHONG_CONS_GAIN
        prior["max_tail"] = prior["max_tail"] * _DIPHTHONG_TAIL_GAIN
    if bool(is_special) and role in {"v", "v-"}:
        cv_prior = _ROLE_PRIOR_TABLE["cv"]
        prior["cons_gap"] = max(prior["cons_gap"], cv_prior["cons_gap"])
        prior["min_cons_gap"] = max(prior["min_cons_gap"], cv_prior["min_cons_gap"])
        prior["max_cons_gap"] = max(prior["max_cons_gap"], cv_prior["max_cons_gap"])
    return prior


def _resolve_prior(
    *,
    alias_role: object,
    is_diphthong: bool,
    is_special: bool,
    format_type: object,
    alias_type: object,
    transition_type: object,
) -> dict[str, float]:
    """Pick the role-based prior when the caller supplies a role; otherwise
    fall back to the legacy format/alias prior so existing callers keep
    their behavior.
    """
    role_text = str(alias_role or "").strip().lower()
    if role_text or bool(is_special):
        return oto_role_prior(alias_role, is_diphthong=is_diphthong, is_special=is_special)
    return oto_timing_prior(format_type=format_type, alias_type=alias_type, transition_type=transition_type)


def oto_timing_prior(
    *,
    format_type: object = "",
    alias_type: object = "",
    transition_type: object = "",
) -> dict[str, float]:
    prior = dict(alias_timing_prior(alias_type))
    fmt = str(format_type or "").strip().lower()
    alias = str(alias_type or "").strip().lower()
    transition = str(transition_type or "").strip().lower()

    if fmt == "cvvc":
        if alias == "vc" or transition == "vc":
            return _prior(cons_gap=24.0, tail=46.0, min_cons_gap=8.0, min_tail=18.0, max_cons_gap=42.0, max_tail=70.0)
        if alias == "vv" or transition == "vv":
            return _prior(cons_gap=38.0, tail=62.0, min_cons_gap=12.0, min_tail=24.0, max_cons_gap=68.0, max_tail=104.0)
        if alias == "vcv":
            return _prior(cons_gap=42.0, tail=64.0, min_cons_gap=14.0, min_tail=24.0, max_cons_gap=72.0, max_tail=104.0)
        if alias == "cv_head":
            return _prior(cons_gap=56.0, tail=86.0, min_cons_gap=18.0, min_tail=34.0, max_cons_gap=88.0, max_tail=132.0)
        if alias == "cv" or transition == "cv":
            return _prior(cons_gap=44.0, tail=72.0, min_cons_gap=14.0, min_tail=30.0, max_cons_gap=72.0, max_tail=116.0)

    if fmt == "vcv":
        if alias == "vcv":
            return _prior(cons_gap=48.0, tail=72.0, min_cons_gap=16.0, min_tail=28.0, max_cons_gap=82.0, max_tail=112.0)
        if alias == "vv" or transition == "vv":
            return _prior(cons_gap=52.0, tail=80.0, min_cons_gap=16.0, min_tail=30.0, max_cons_gap=86.0, max_tail=126.0)
        if alias == "vc" or transition == "vc":
            return _prior(cons_gap=26.0, tail=48.0, min_cons_gap=8.0, min_tail=18.0, max_cons_gap=46.0, max_tail=76.0)

    if fmt == "cvc":
        if alias == "vcv":
            return _prior(cons_gap=44.0, tail=72.0, min_cons_gap=14.0, min_tail=28.0, max_cons_gap=76.0, max_tail=116.0)
        if alias == "vc" or transition == "vc":
            return _prior(cons_gap=28.0, tail=50.0, min_cons_gap=10.0, min_tail=20.0, max_cons_gap=50.0, max_tail=82.0)

    if transition == "vc":
        prior.update({"cons_gap": 26.0, "tail": 50.0, "min_cons_gap": 8.0, "min_tail": 18.0, "max_cons_gap": 48.0, "max_tail": 80.0})
    elif transition == "vv" and alias not in {"vcv", "cv_head"}:
        prior.update({"cons_gap": 52.0, "tail": 82.0, "max_cons_gap": 88.0, "max_tail": 132.0})
    return prior


def alias_timing_prior(alias_type: object) -> dict[str, float]:
    kind = str(alias_type or "").strip().lower()
    if kind == "cv_head":
        return _prior(cons_gap=58.0, tail=96.0, min_cons_gap=18.0, min_tail=36.0, max_cons_gap=92.0, max_tail=150.0)
    if kind == "cv":
        return _prior(cons_gap=48.0, tail=82.0, min_cons_gap=16.0, min_tail=34.0, max_cons_gap=78.0, max_tail=130.0)
    if kind == "vcv":
        return _prior(cons_gap=48.0, tail=74.0, min_cons_gap=16.0, min_tail=30.0, max_cons_gap=82.0, max_tail=118.0)
    if kind == "vc":
        return _prior(cons_gap=28.0, tail=52.0, min_cons_gap=10.0, min_tail=22.0, max_cons_gap=48.0, max_tail=82.0)
    if kind == "vv":
        return _prior(cons_gap=56.0, tail=90.0, min_cons_gap=16.0, min_tail=32.0, max_cons_gap=92.0, max_tail=142.0)
    if kind == "br":
        return _prior(cons_gap=24.0, tail=42.0, min_cons_gap=8.0, min_tail=18.0, max_cons_gap=42.0, max_tail=72.0)
    return _prior(cons_gap=48.0, tail=82.0, min_cons_gap=14.0, min_tail=30.0, max_cons_gap=82.0, max_tail=128.0)


def normalize_relative_oto_target(
    anchors_ms: Any,
    *,
    duration_ms: float,
    alias_type: object,
    format_type: object = "",
    transition_type: object = "",
    alias_role: object = "",
    is_diphthong: bool = False,
    is_special: bool = False,
    active_origin_ms: float | None = None,
    active_span_ms: float | None = None,
) -> list[float]:
    values = [float(v) for v in anchors_ms]
    duration = max(1.0, float(duration_ms))
    offset, overlap, pre, consonant, cutoff = values[:5]
    prior = _resolve_prior(
        alias_role=alias_role,
        is_diphthong=is_diphthong,
        is_special=is_special,
        format_type=format_type,
        alias_type=alias_type,
        transition_type=transition_type,
    )
    overlap_delta = max(0.0, min(max(pre - offset, 1.0), overlap - offset))
    pre_delta = max(overlap_delta, pre - offset)
    cons_gap = _clamp(consonant - pre, prior["min_cons_gap"], prior["max_cons_gap"])
    tail = _clamp(cutoff - consonant, prior["min_tail"], prior["max_tail"])
    offset_origin = max(0.0, float(active_origin_ms)) if active_origin_ms is not None else 0.0
    offset_span = max(1.0, float(active_span_ms)) if active_span_ms is not None else duration
    return [
        _unit((offset - offset_origin) / offset_span),
        _unit(overlap_delta / duration),
        _unit(pre_delta / duration),
        _unit(cons_gap / duration),
        _unit(tail / duration),
    ]


def decode_relative_oto_params(
    scalar_values: Any,
    *,
    duration_ms: float,
    alias_type: object,
    format_type: object = "",
    transition_type: object = "",
    prior_blend: float = 0.45,
    alias_role: object = "",
    is_diphthong: bool = False,
    is_special: bool = False,
    active_origin_ms: float | None = None,
    active_span_ms: float | None = None,
) -> dict[str, float]:
    values = [max(0.0, min(1.0, float(v))) for v in scalar_values]
    duration = max(1.0, float(duration_ms))
    prior = _resolve_prior(
        alias_role=alias_role,
        is_diphthong=is_diphthong,
        is_special=is_special,
        format_type=format_type,
        alias_type=alias_type,
        transition_type=transition_type,
    )
    offset_origin = max(0.0, float(active_origin_ms)) if active_origin_ms is not None else 0.0
    offset_span = max(1.0, float(active_span_ms)) if active_span_ms is not None else duration
    offset = offset_origin + (values[0] * offset_span)
    overlap_delta = values[1] * duration
    pre_delta = max(overlap_delta, values[2] * duration)
    cons_gap_model = _clamp(values[3] * duration, prior["min_cons_gap"], prior["max_cons_gap"])
    tail_model = _clamp(values[4] * duration, prior["min_tail"], prior["max_tail"])
    blend = _unit(prior_blend)
    cons_gap = (cons_gap_model * (1.0 - blend)) + (prior["cons_gap"] * blend)
    tail = (tail_model * (1.0 - blend)) + (prior["tail"] * blend)
    consonant = max(pre_delta + 1.0, pre_delta + cons_gap)
    cutoff = -(consonant + max(1.0, tail))
    overlap = min(overlap_delta, pre_delta)
    return {
        "offset": max(0.0, offset),
        "consonant": max(1.0, consonant),
        "cutoff": -max(1.0, abs(cutoff)),
        "preutterance": max(0.0, pre_delta),
        "overlap": max(0.0, overlap),
    }


def relative_params_to_anchors(params: dict[str, float], *, duration_ms: float) -> tuple[float, float, float, float, float]:
    offset = max(0.0, float(params.get("offset", 0.0)))
    overlap = offset + max(0.0, float(params.get("overlap", 0.0)))
    pre = offset + max(0.0, float(params.get("preutterance", 0.0)))
    consonant = offset + max(1.0, float(params.get("consonant", 1.0)))
    cutoff = offset + max(1.0, abs(float(params.get("cutoff", -1.0))))
    duration = max(1.0, float(duration_ms))
    return (
        min(offset, duration - 1.0),
        min(max(overlap, offset), duration - 1.0),
        min(max(pre, overlap), duration - 1.0),
        min(max(consonant, pre + 1.0), duration - 1.0),
        min(max(cutoff, consonant + 1.0), duration),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _prior(
    *,
    cons_gap: float,
    tail: float,
    min_cons_gap: float,
    min_tail: float,
    max_cons_gap: float,
    max_tail: float,
) -> dict[str, float]:
    return {
        "cons_gap": float(cons_gap),
        "tail": float(tail),
        "min_cons_gap": float(min_cons_gap),
        "min_tail": float(min_tail),
        "max_cons_gap": float(max_cons_gap),
        "max_tail": float(max_tail),
    }


__all__ = [
    "alias_timing_prior",
    "decode_relative_oto_params",
    "normalize_relative_oto_target",
    "oto_role_prior",
    "oto_timing_prior",
    "relative_params_to_anchors",
]
