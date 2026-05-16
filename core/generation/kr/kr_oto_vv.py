from __future__ import annotations


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _compute_kr_vowel_stable_anchor(v_start, v_end):
    """모음 onset이 아닌, 비교적 안정적인 중간 지점을 반환합니다."""
    v_start = float(v_start)
    v_end = float(v_end)
    v_len = max(v_end - v_start, 24.0)
    lead = _clamp(v_len * 0.38, 22.0, 82.0)
    tail_margin = _clamp(v_len * 0.26, 18.0, 52.0)
    anchor = v_start + lead
    anchor = max(anchor, v_start + 18.0)
    anchor = min(anchor, max(v_start + 18.0, v_end - tail_margin))
    return anchor


def _compute_kr_noninitial_vowel_timing(v_start, v_end):
    """어두가 아닌 모음 alias가 안정된 모음 중간을 쓰도록 기본 timing을 만듭니다."""
    from core.kr_oto_cv import adaptive_overlap
    from core.oto_generator import validate_oto_params

    v_start = float(v_start)
    v_end = float(v_end)
    v_len = max(v_end - v_start, 40.0)
    offset_pad = _clamp(v_len * 0.18, 12.0, 42.0)
    offset = max(v_start - offset_pad, 0.0)
    anchor_abs = _compute_kr_vowel_stable_anchor(v_start, v_end)
    pre = max(anchor_abs - offset, 22.0)
    ovl_floor = max(adaptive_overlap(pre, "", mode="vv"), min(pre * 0.74, max(pre - 10.0, 0.0)))
    ovl_gap = _clamp(v_len * 0.10, 6.0, 16.0)
    ovl = max(ovl_floor, pre - ovl_gap)
    added_cons = _clamp(v_len * 0.18, 34.0, 104.0)
    consonant = pre + added_cons
    cutoff_abs = consonant + _clamp(v_len * 0.40, 68.0, 220.0)
    return validate_oto_params(offset, consonant, -cutoff_abs, pre, ovl)


def _compute_kr_vv_timing_from_vowel_bounds(prev_v_start, prev_v_end, n_start, n_end):
    """어두가 아닌 VV 연결을 현재 모음의 안정 구간 기준으로 계산합니다."""
    from core.kr_oto_cv import adaptive_overlap
    from core.oto_generator import validate_oto_params

    prev_v_start = float(prev_v_start)
    prev_v_end = float(prev_v_end)
    n_start = float(n_start)
    n_end = float(n_end)
    prev_v_len = max(prev_v_end - prev_v_start, 26.0)
    curr_v_len = max(n_end - n_start, 35.0)

    tail_keep = _clamp(prev_v_len * 0.16, 16.0, 52.0)
    offset = max(prev_v_end - tail_keep, 0.0)
    anchor_abs = _compute_kr_vowel_stable_anchor(n_start, n_end)
    pre = max(anchor_abs - offset, 24.0)
    ovl_floor = max(adaptive_overlap(pre, "", mode="vv"), min(pre * 0.74, max(pre - 10.0, 0.0)))
    ovl_gap = _clamp((prev_v_len * 0.08) + (curr_v_len * 0.04), 5.0, 16.0)
    ovl = max(ovl_floor, pre - ovl_gap)
    added_cons = _clamp(curr_v_len * 0.24, 48.0, 132.0)
    consonant = pre + added_cons

    cut_gap = _clamp(curr_v_len * 0.32, 58.0, 180.0)
    cutoff_abs = consonant + cut_gap
    curr_v_end_rel = max(n_end - offset, consonant + 18.0)
    cutoff_cap = curr_v_end_rel + _clamp(curr_v_len * 0.05, 8.0, 18.0)
    cutoff_abs = min(cutoff_abs, cutoff_cap)
    return validate_oto_params(offset, consonant, -cutoff_abs, pre, ovl)


def _compute_kr_cvvc_vv_timing_direct(current_w_idx, syllables_info, n_start, n_end):
    """한국어 CVVC의 순수 VV 연결을 현재 모음의 안정 구간 기준으로 계산합니다."""
    if current_w_idx <= 0 or current_w_idx >= len(syllables_info or []):
        return None

    prev_syl = syllables_info[current_w_idx - 1]
    prev_phones = prev_syl.get("phones") or []
    if not prev_phones:
        return None

    from core.kr_oto_rules import find_vowel_phone

    _pv_idx, prev_v_phone = find_vowel_phone(prev_phones)
    prev_v_start = float(prev_v_phone.minTime) * 1000.0
    prev_v_end = float(prev_v_phone.maxTime) * 1000.0
    return _compute_kr_vv_timing_from_vowel_bounds(prev_v_start, prev_v_end, n_start, n_end)


__all__ = [
    "_compute_kr_cvvc_vv_timing_direct",
    "_compute_kr_noninitial_vowel_timing",
    "_compute_kr_vowel_stable_anchor",
    "_compute_kr_vv_timing_from_vowel_bounds",
]
