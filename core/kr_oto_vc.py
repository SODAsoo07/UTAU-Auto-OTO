from __future__ import annotations

from core.kr_oto_rules import (
    _canonicalize_kr_coda,
    _extract_vc_right_token,
    _is_kr_plosive_coda_alias,
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
):
    """한국어 VC 타이밍 계산 핵심 로직입니다."""
    from core.kr_oto_bridge import _compute_vc_from_adjacent_cv, _compute_kr_cvvc_vc_timing_direct
    from core.kr_oto_cv import adaptive_overlap

    c_char = _extract_vc_right_token(alias)
    is_vc_plosive_coda = (
        alias_type == "vc"
        and file_format in {"cvc", "cvvc", "vc_only"}
        and _is_kr_plosive_coda_alias(alias)
    )

    if is_vc_plosive_coda:
        from core.kr_oto_rules import find_vowel_phone

        coda_canon = _canonicalize_kr_coda(c_char)
        is_hard_stop_coda = coda_canon in {"t", "p"}
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

        boundary = max(vowel_start + 16.0, vowel_end - (14.0 if is_hard_stop_coda else 11.0))
        if coda_start is not None:
            coda_margin = 12.0 if is_hard_stop_coda else 9.0
            boundary = min(boundary, coda_start - coda_margin)
            boundary = max(boundary, vowel_start + (12.0 if is_hard_stop_coda else 14.0))

        pre_target = _clamp(
            boundary - vowel_start,
            42.0 if is_hard_stop_coda else 45.0,
            132.0 if is_hard_stop_coda else 145.0,
        )
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_char, mode="vc")
        ovl = min(ovl, max(pre - 12.0, 0.0))

        tail_floor = 10.0 if is_hard_stop_coda else 12.0
        if coda_start is not None:
            tail_room = max(coda_start - boundary, tail_floor)
        else:
            tail_room = max(n_start - boundary, tail_floor)

        cons_mul = 0.54 if is_hard_stop_coda else 0.62
        cons_min = 10.0 if is_hard_stop_coda else 12.0
        cons_max = 30.0 if is_hard_stop_coda else 38.0
        added_cons = _clamp(tail_room * cons_mul, cons_min, cons_max)
        consonant = pre + added_cons
        cut_mul = 0.56 if is_hard_stop_coda else 0.70
        cut_min = 20.0 if is_hard_stop_coda else 24.0
        cut_max = 52.0 if is_hard_stop_coda else 64.0
        cut_gap = _clamp(tail_room * cut_mul, cut_min, cut_max)

        cutoff_abs = consonant + cut_gap
        next_onset_rel = max(n_start - offset, pre + 12.0)
        cutoff_soft_cap = next_onset_rel - (1.2 if is_hard_stop_coda else 0.8)
        cutoff_min_abs = consonant + (8.0 if is_hard_stop_coda else 10.0)
        if cutoff_soft_cap <= cutoff_min_abs:
            consonant = min(consonant, max(next_onset_rel - 8.0, pre + 6.0))
            cutoff_min_abs = consonant + (6.0 if is_hard_stop_coda else 8.0)
            if cutoff_soft_cap <= cutoff_min_abs:
                cutoff_soft_cap = cutoff_min_abs + 0.8
        cutoff_abs = _clamp(cutoff_abs, cutoff_min_abs, cutoff_soft_cap)
        cutoff = -cutoff_abs
        return offset, consonant, cutoff, pre, ovl, True

    if file_format == "cvvc":
        direct_params = _compute_kr_cvvc_vc_timing_direct(alias, c_start, c_end, n_start, n_end)
        if direct_params is not None:
            offset, consonant, cutoff, pre, ovl = direct_params
            return offset, consonant, cutoff, pre, ovl, False

    vc_anchor_params = None
    resolved_next_w_idx = next_w_idx if next_w_idx is not None else (current_w_idx + 1)
    if resolved_next_w_idx < len(syllables_info):
        prev_cv_anchor = prev_cv_anchor or cv_anchor_by_idx.get(current_w_idx)
        next_cv_anchor = next_cv_anchor or cv_anchor_by_idx.get(resolved_next_w_idx)
        is_plosive_sibilant = c_char in ["g", "k", "kk", "gg", "d", "t", "tt", "dd", "b", "p", "bb", "pp", "s", "ss", "h", "j", "jj", "ch"]
        vc_anchor_params = _compute_vc_from_adjacent_cv(
            prev_cv_anchor, next_cv_anchor, alias_type, is_plosive_sibilant
        )

    if vc_anchor_params is not None:
        offset, consonant, cutoff, pre, ovl = vc_anchor_params
        return offset, consonant, cutoff, pre, ovl, False

    vc_target = n_start
    boundary = min(vc_target, c_end + 260)
    v_len = c_end - c_start
    n_len = n_end - n_start

    offset_padding = 180
    if v_len < offset_padding:
        offset_padding = max(v_len * 0.8, 50)

    offset = boundary - offset_padding
    pre = boundary - offset

    is_plosive_sibilant = c_char in ["g", "k", "kk", "gg", "d", "t", "tt", "dd", "b", "p", "bb", "pp", "s", "ss", "h", "j", "jj", "ch"]
    ovl = adaptive_overlap(pre, c_char, mode="vc")

    if is_plosive_sibilant:
        n_ref = max(n_len, 100)
        added_cons = min(max(n_ref * 0.35, 45), 95)
        consonant = pre + added_cons
        cutoff = -(consonant + max(n_len * 0.35, 45))
    else:
        n_ref = max(n_len, 80)
        added_cons = min(max(n_ref * 0.55, 55), 160)
        consonant = pre + added_cons
        cutoff = -(consonant + max(n_len * 0.45, 50))

    next_c_onset_rel = max(n_start - offset, pre + 10.0)
    if is_plosive_sibilant:
        consonant = min(consonant, next_c_onset_rel - 7.0)
        consonant = max(consonant, pre + 10.0)
        cutoff_abs = max(consonant + 8.0, next_c_onset_rel - 1.2)
        cutoff_cap = next_c_onset_rel - 1.0
        cutoff_floor = consonant + 8.0
        if cutoff_floor > cutoff_cap:
            consonant = max(pre + 8.0, cutoff_cap - 8.0)
            cutoff_floor = consonant + 8.0
        if cutoff_floor > cutoff_cap:
            cutoff_cap = cutoff_floor + 0.8
        cutoff_abs = min(cutoff_abs, cutoff_cap)
        cutoff_abs = max(cutoff_abs, cutoff_floor)
        cutoff = -cutoff_abs
    else:
        consonant = min(consonant, next_c_onset_rel + 26.0)
        consonant = max(consonant, pre + 16.0)
        cutoff_abs = min(abs(cutoff), next_c_onset_rel + 30.0)
        cutoff_abs = max(cutoff_abs, consonant + 12.0)
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
