from __future__ import annotations

from statistics import median


def _clamp(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _mad(values):
    vals = [float(v) for v in (values or [])]
    if not vals:
        return 0.0
    m = float(median(vals))
    return float(median([abs(v - m) for v in vals]))


def _iter_anchor_values(anchor_store):
    if not isinstance(anchor_store, dict):
        return []
    return [v for v in anchor_store.values() if isinstance(v, dict)]


def _collect_anchor_metrics(anchors):
    vowel_lens = []
    cons_gaps = []
    cut_gaps = []
    for row in anchors or []:
        v_len = _safe_float(row.get("vowel_len"), 0.0)
        if v_len > 0.0:
            vowel_lens.append(v_len)
        c_gap = _safe_float(row.get("cons_gap"), 0.0)
        if c_gap > 0.0:
            cons_gaps.append(c_gap)
        cut_gap = _safe_float(row.get("cut_gap"), 0.0)
        if cut_gap > 0.0:
            cut_gaps.append(cut_gap)
    return vowel_lens, cons_gaps, cut_gaps


def derive_cv_bridge_tuning(
    *,
    realized_anchor_by_idx,
    estimated_anchor_by_idx=None,
    trust_tier="",
    mapping_tier="",
    min_samples=8,
):
    """
    CV anchor 통계 기반의 파일 단위 VC bridge 튜닝 파라미터를 반환합니다.
    - sample이 충분하고 신뢰 tier가 낮지 않을 때만 활성화
    - 중앙값+MAD 기반으로 화자/레코딩 조건 변화에 강건하게 반응
    """
    trust = str(trust_tier or "").strip().lower()
    m_tier = str(mapping_tier or "").strip().lower()
    realized_rows = _iter_anchor_values(realized_anchor_by_idx)
    estimated_rows = _iter_anchor_values(estimated_anchor_by_idx)
    source = "realized" if len(realized_rows) >= len(estimated_rows) else "estimated"
    rows = realized_rows if source == "realized" else estimated_rows

    vowel_lens, cons_gaps, cut_gaps = _collect_anchor_metrics(rows)
    sample_count = min(len(vowel_lens), len(cons_gaps), len(cut_gaps))

    out = {
        "enabled": False,
        "source": source,
        "sample_count": int(sample_count),
        "tempo_scale": 1.0,
        "release_scale": 1.0,
        "noise_scale": 0.0,
        "reason": "",
    }
    if trust == "low" or m_tier == "low":
        out["reason"] = "low_tier"
        return out
    if sample_count < int(max(4, min_samples)):
        out["reason"] = "insufficient_samples"
        return out

    med_vowel = max(_safe_float(median(vowel_lens), 0.0), 1.0)
    med_cons = max(_safe_float(median(cons_gaps), 0.0), 1.0)
    med_cut = max(_safe_float(median(cut_gaps), 0.0), 1.0)

    tempo_scale = _clamp(med_vowel / 120.0, 0.85, 1.20)
    release_scale = _clamp((med_cons / 42.0) * 0.52 + (med_cut / 62.0) * 0.48, 0.82, 1.25)

    noise_v = _mad(vowel_lens) / med_vowel
    noise_c = _mad(cut_gaps) / med_cut
    noise_scale = _clamp((noise_v * 0.56) + (noise_c * 0.44), 0.0, 0.35)

    out.update(
        {
            "enabled": True,
            "tempo_scale": float(tempo_scale),
            "release_scale": float(release_scale),
            "noise_scale": float(noise_scale),
            "reason": "ok",
        }
    )
    return out


_JA_PROFILE_TEMPO_KEYS = (
    "offset_pad",
    "offset_pad_min",
    "offset_pad_floor",
    "pre_lead_min",
    "pre_lead_max",
    "tail_margin_base",
    "cons_add_base",
    "cons_add_min",
    "cons_add_max",
    "cut_add_base",
    "cut_add_min",
    "cut_add_max",
    "cut_min_gap",
)
_JA_PROFILE_RELEASE_KEYS = (
    "cons_to_next_margin",
    "cut_to_next_allow",
    "ovl_pre_margin",
)


def apply_ja_bridge_profile_tuning(bridge_profile, bridge_tuning):
    profile = dict(bridge_profile or {})
    tuning = dict(bridge_tuning or {})
    if not profile:
        return profile
    if not bool(tuning.get("enabled")):
        return profile

    tempo = _clamp(_safe_float(tuning.get("tempo_scale", 1.0), 1.0), 0.85, 1.20)
    release = _clamp(_safe_float(tuning.get("release_scale", 1.0), 1.0), 0.82, 1.25)
    noise = _clamp(_safe_float(tuning.get("noise_scale", 0.0), 0.0), 0.0, 0.35)

    for key in _JA_PROFILE_TEMPO_KEYS:
        if key in profile:
            profile[key] = float(profile[key]) * tempo
    for key in _JA_PROFILE_RELEASE_KEYS:
        if key in profile:
            profile[key] = float(profile[key]) * release

    if "offset_len_mul" in profile:
        profile["offset_len_mul"] = _clamp(
            float(profile["offset_len_mul"]) * (1.0 + (tempo - 1.0) * 0.45),
            0.02,
            0.30,
        )
    if "pre_lead_mul" in profile:
        profile["pre_lead_mul"] = _clamp(
            float(profile["pre_lead_mul"]) * (1.0 + (tempo - 1.0) * 0.40),
            0.10,
            0.60,
        )
    if "ovl_ratio" in profile:
        profile["ovl_ratio"] = _clamp(float(profile["ovl_ratio"]) - (noise * 0.08), 0.22, 0.78)
    if "ovl_min_ratio" in profile:
        profile["ovl_min_ratio"] = _clamp(float(profile["ovl_min_ratio"]) - (noise * 0.06), 0.12, 0.70)
    if "ovl_ratio" in profile and "ovl_min_ratio" in profile:
        profile["ovl_min_ratio"] = min(float(profile["ovl_min_ratio"]), float(profile["ovl_ratio"]) - 0.02)

    return profile


def resolve_bridge_tuning_scales(bridge_tuning):
    tuning = dict(bridge_tuning or {})
    if not bool(tuning.get("enabled")):
        return 1.0, 1.0, 0.0
    tempo = _clamp(_safe_float(tuning.get("tempo_scale", 1.0), 1.0), 0.85, 1.20)
    release = _clamp(_safe_float(tuning.get("release_scale", 1.0), 1.0), 0.82, 1.25)
    noise = _clamp(_safe_float(tuning.get("noise_scale", 0.0), 0.0), 0.0, 0.35)
    return float(tempo), float(release), float(noise)


__all__ = [
    "apply_ja_bridge_profile_tuning",
    "derive_cv_bridge_tuning",
    "resolve_bridge_tuning_scales",
]
