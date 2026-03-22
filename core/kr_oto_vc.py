from __future__ import annotations

from core.kr_oto_rules import (
    _canonicalize_kr_coda,
    _extract_vc_right_token,
    _is_kr_plosive_coda_alias,
)
from core.cv_anchor_adaptation import resolve_bridge_tuning_scales
from core.vc_timing_profiles import (
    classify_kr_vc_coda_group,
    get_kr_vc_fallback_profile,
    get_kr_vc_plosive_coda_profile,
)


def _prepare_vc_bounds_from_context(syllables_info, current_w_idx, next_w_idx=None):
    """VC 계산에 필요한 (curr_phones, c_start/c_end/n_start/n_end)를 구성합니다."""
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]

    v_start = curr_phones[-1].minTime * 1000
    v_end = curr_phones[-1].maxTime * 1000
    c_start = v_start
    c_end = v_end

    if next_w_idx is None:
        next_w_idx = current_w_idx + 1

    if next_w_idx < len(syllables_info):
        next_syl = syllables_info[next_w_idx]
        n_start = next_syl["phones"][0].minTime * 1000
        n_end = next_syl["phones"][0].maxTime * 1000
    else:
        n_start = v_end
        n_end = v_end + 100

    return current_w_idx, curr_phones, c_start, c_end, n_start, n_end


def _compute_kr_vc_timing(
    alias,
    alias_type,
    file_format,
    curr_phones,
    current_w_idx,
    syllables_info,
    c_start,
    c_end,
    n_start,
    n_end,
    cv_anchor_by_idx,
    next_w_idx=None,
    prev_cv_anchor=None,
    next_cv_anchor=None,
    bridge_tuning=None,
):
    """한국어 VC 타이밍 계산 핵심 로직입니다."""
    from core.kr_oto_bridge import _compute_vc_from_adjacent_cv, _compute_kr_cvvc_vc_timing_direct
    from core.kr_oto_cv import adaptive_overlap

    c_char = _extract_vc_right_token(alias)
    coda_canon = _canonicalize_kr_coda(c_char)
    coda_group = classify_kr_vc_coda_group(c_char, coda_canon)
    is_stoplike_bridge = bool(coda_group == "stop")
    is_fricative_bridge = bool(coda_group == "fricative")
    is_vc_plosive_coda = (
        alias_type == "vc"
        and file_format in {"cvc", "cvvc", "vc_only"}
        and _is_kr_plosive_coda_alias(alias)
    )
    tempo_scale, release_scale, noise_scale = resolve_bridge_tuning_scales(bridge_tuning)
    tempo_floor = max(0.90, min(1.18, tempo_scale * (1.0 - noise_scale * 0.12)))
    tempo_ceil = max(0.88, min(1.22, tempo_scale * (1.0 + noise_scale * 0.10)))
    release_adj = max(0.84, min(1.28, release_scale * (1.0 + noise_scale * 0.08)))
    resolved_next_w_idx = next_w_idx if next_w_idx is not None else (current_w_idx + 1)
    vc_anchor_params = None
    if resolved_next_w_idx < len(syllables_info):
        prev_cv_anchor = prev_cv_anchor or cv_anchor_by_idx.get(current_w_idx)
        next_cv_anchor = next_cv_anchor or cv_anchor_by_idx.get(resolved_next_w_idx)
        vc_anchor_params = _compute_vc_from_adjacent_cv(
            prev_cv_anchor,
            next_cv_anchor,
            alias_type,
            is_stoplike_bridge,
            bridge_tuning=bridge_tuning,
        )
    if vc_anchor_params is not None:
        offset, consonant, cutoff, pre, ovl = vc_anchor_params
        if is_vc_plosive_coda:
            next_onset_rel = max(n_start - offset, pre + (10.0 * tempo_floor))
            consonant = min(consonant, next_onset_rel - (7.0 * tempo_floor))
            consonant = max(consonant, pre + (6.0 * tempo_floor))
            cutoff_abs = max(abs(float(cutoff)), consonant + (4.0 * tempo_floor))
            cutoff_floor = consonant + (4.0 * tempo_floor)
            cutoff_cap = next_onset_rel - max(0.8, 1.0 * tempo_floor)
            if cutoff_floor > cutoff_cap:
                consonant = max(pre + (6.0 * tempo_floor), cutoff_cap - (4.0 * tempo_floor))
                cutoff_floor = consonant + (4.0 * tempo_floor)
            if cutoff_floor > cutoff_cap:
                cutoff_cap = cutoff_floor + 0.8
            cutoff_abs = _clamp(cutoff_abs, cutoff_floor, cutoff_cap)
            cutoff = -cutoff_abs
        return offset, consonant, cutoff, pre, ovl, bool(is_vc_plosive_coda)

    if is_vc_plosive_coda:
        from core.kr_oto_rules import find_vowel_phone

        stop_prof = get_kr_vc_plosive_coda_profile(coda_canon)
        v_idx = None
        v_phone = None
        if curr_phones:
            v_idx, v_phone = find_vowel_phone(curr_phones)

        vowel_start = (v_phone.minTime * 1000) if v_phone else c_start
        vowel_end = (v_phone.maxTime * 1000) if v_phone else c_end

        coda_start = None
        if v_phone is not None and v_idx is not None and (v_idx + 1) < len(curr_phones):
            coda_start = curr_phones[v_idx + 1].minTime * 1000
        elif (next_w_idx if next_w_idx is not None else (current_w_idx + 1)) < len(syllables_info):
            next_syl = syllables_info[next_w_idx if next_w_idx is not None else (current_w_idx + 1)]
            if next_syl.get("phones"):
                coda_start = next_syl["phones"][0].minTime * 1000

        boundary = max(
            vowel_start + (16.0 * tempo_floor),
            vowel_end - (float(stop_prof.get("boundary_back_ms", 12.0)) * tempo_floor),
        )
        if coda_start is not None:
            coda_margin = float(stop_prof.get("coda_margin_ms", 10.0)) * tempo_floor
            boundary = min(boundary, coda_start - coda_margin)
            boundary = max(
                boundary,
                vowel_start + (float(stop_prof.get("boundary_floor_from_vowel_start", 12.0)) * tempo_floor),
            )

        pre_target = _clamp(
            boundary - vowel_start,
            float(stop_prof.get("pre_min", 44.0)) * tempo_floor,
            float(stop_prof.get("pre_max", 140.0)) * tempo_ceil,
        )
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_char, mode="vc")
        ovl = min(ovl, max(pre - (12.0 * tempo_floor), 0.0))

        tail_floor = float(stop_prof.get("tail_floor", 11.0)) * tempo_floor
        if coda_start is not None:
            tail_room = max(coda_start - boundary, tail_floor)
        else:
            tail_room = max(n_start - boundary, tail_floor)

        cons_mul = float(stop_prof.get("cons_mul", 0.58))
        cons_min = float(stop_prof.get("cons_min", 11.0)) * tempo_floor
        cons_max = float(stop_prof.get("cons_max", 34.0)) * release_adj
        added_cons = _clamp(tail_room * cons_mul, cons_min, cons_max)
        consonant = pre + added_cons
        cut_mul = float(stop_prof.get("cut_mul", 0.62))
        cut_min = float(stop_prof.get("cut_min", 22.0)) * tempo_floor
        cut_max = float(stop_prof.get("cut_max", 58.0)) * release_adj
        cut_gap = _clamp(tail_room * cut_mul, cut_min, cut_max)

        cutoff_abs = consonant + cut_gap
        next_onset_rel = max(
            n_start - offset,
            pre + (float(stop_prof.get("next_onset_pre_floor_gap", 12.0)) * tempo_floor),
        )
        cutoff_soft_cap = next_onset_rel - (
            float(stop_prof.get("cutoff_soft_cap_to_next", 1.0)) * tempo_floor
        )
        cutoff_min_abs = consonant + (float(stop_prof.get("cutoff_min_gap", 9.0)) * tempo_floor)
        if cutoff_soft_cap <= cutoff_min_abs:
            consonant = min(
                consonant,
                max(next_onset_rel - (8.0 * tempo_floor), pre + (6.0 * tempo_floor)),
            )
            cutoff_min_abs = consonant + (
                float(stop_prof.get("cutoff_min_gap_retry", 7.0)) * tempo_floor
            )
            if cutoff_soft_cap <= cutoff_min_abs:
                cutoff_soft_cap = cutoff_min_abs + 0.8
        cutoff_abs = _clamp(cutoff_abs, cutoff_min_abs, cutoff_soft_cap)
        cutoff = -cutoff_abs
        return offset, consonant, cutoff, pre, ovl, True

    if file_format == "cvvc":
        direct_params = _compute_kr_cvvc_vc_timing_direct(
            alias,
            c_start,
            c_end,
            n_start,
            n_end,
            bridge_tuning=bridge_tuning,
        )
        if direct_params is not None:
            offset, consonant, cutoff, pre, ovl = direct_params
            return offset, consonant, cutoff, pre, ovl, False

    vc_target = n_start
    boundary = min(vc_target, c_end + (260.0 * tempo_ceil))
    v_len = c_end - c_start
    n_len = n_end - n_start
    fallback_profile = get_kr_vc_fallback_profile(coda_group)

    offset_padding = float(fallback_profile.get("offset_padding_base", 180.0)) * tempo_floor
    if v_len < offset_padding:
        offset_padding = max(
            v_len * float(fallback_profile.get("offset_padding_ratio_short", 0.80)),
            float(fallback_profile.get("offset_padding_min_short", 50.0)) * tempo_floor,
        )

    offset = boundary - offset_padding
    pre = boundary - offset

    ovl = adaptive_overlap(pre, c_char, mode="vc")

    n_ref = max(n_len, float(fallback_profile.get("n_ref_min", 80.0)) * tempo_floor)
    added_cons = _clamp(
        n_ref * float(fallback_profile.get("cons_mul", 0.50)),
        float(fallback_profile.get("cons_min", 20.0)) * tempo_floor,
        float(fallback_profile.get("cons_max", 100.0)) * release_adj,
    )
    consonant = pre + added_cons
    cut_gap = _clamp(
        n_len * float(fallback_profile.get("cut_mul", 0.50)),
        float(fallback_profile.get("cut_min", 18.0)) * tempo_floor,
        float(fallback_profile.get("cut_max", 80.0)) * release_adj,
    )
    cutoff = -(consonant + cut_gap)

    next_c_onset_rel = max(
        n_start - offset,
        pre + (float(fallback_profile.get("next_onset_pre_floor_gap", 10.0)) * tempo_floor),
    )
    if is_stoplike_bridge:
        cons_cap = next_c_onset_rel - (
            float(fallback_profile.get("cons_cap_before_next", 7.0)) * tempo_floor
        )
        consonant = min(consonant, cons_cap)
        consonant = max(consonant, pre + (float(fallback_profile.get("cons_floor_gap", 6.0)) * tempo_floor))
        cutoff_abs = max(
            consonant + (float(fallback_profile.get("cut_floor_gap", 4.0)) * tempo_floor),
            next_c_onset_rel - (float(fallback_profile.get("cut_seed_before_next", 1.2)) * tempo_floor),
        )
        cutoff_cap = next_c_onset_rel - (
            float(fallback_profile.get("cut_cap_before_next", 1.0)) * tempo_floor
        )
        cutoff_floor = consonant + (float(fallback_profile.get("cut_floor_gap", 4.0)) * tempo_floor)
        if cutoff_floor > cutoff_cap:
            consonant = max(
                pre + (float(fallback_profile.get("cons_floor_gap", 6.0)) * tempo_floor),
                cutoff_cap - (float(fallback_profile.get("cut_floor_gap", 4.0)) * tempo_floor),
            )
            cutoff_floor = consonant + (float(fallback_profile.get("cut_floor_gap", 4.0)) * tempo_floor)
        if cutoff_floor > cutoff_cap:
            cutoff_cap = cutoff_floor + 0.8
        cutoff_abs = min(cutoff_abs, cutoff_cap)
        cutoff_abs = max(cutoff_abs, cutoff_floor)
        cutoff = -cutoff_abs
    elif is_fricative_bridge:
        consonant = min(
            consonant,
            next_c_onset_rel + (float(fallback_profile.get("cons_cap_after_next", 8.0)) * release_adj),
        )
        consonant = max(consonant, pre + (float(fallback_profile.get("cons_floor_gap", 10.0)) * tempo_floor))
        cutoff_abs = min(
            abs(cutoff),
            next_c_onset_rel + (float(fallback_profile.get("cut_cap_after_next", 12.0)) * release_adj),
        )
        cutoff_abs = max(cutoff_abs, consonant + (float(fallback_profile.get("cut_floor_gap", 8.0)) * tempo_floor))
        cutoff = -cutoff_abs
    else:
        consonant = min(
            consonant,
            next_c_onset_rel + (float(fallback_profile.get("cons_cap_after_next", 22.0)) * release_adj),
        )
        consonant = max(consonant, pre + (float(fallback_profile.get("cons_floor_gap", 12.0)) * tempo_floor))
        cutoff_abs = min(
            abs(cutoff),
            next_c_onset_rel + (float(fallback_profile.get("cut_cap_after_next", 24.0)) * release_adj),
        )
        cutoff_abs = max(cutoff_abs, consonant + (float(fallback_profile.get("cut_floor_gap", 8.0)) * tempo_floor))
        cutoff = -cutoff_abs

    return offset, consonant, cutoff, pre, ovl, False


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _uses_kr_vc_context(alias_type):
    """VC 전용 bridge 컨텍스트를 사용할 alias 타입인지 판별합니다."""
    return str(alias_type or "").strip().lower() == "vc"


__all__ = [
    "_prepare_vc_bounds_from_context",
    "_compute_kr_vc_timing",
    "_uses_kr_vc_context",
]
