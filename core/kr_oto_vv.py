from __future__ import annotations


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _compute_kr_cvvc_vv_timing_direct(current_w_idx, syllables_info, n_start, n_end):
    """한국어 CVVC의 순수 VV 연결을 이전 모음 tail 기준으로 계산합니다."""
    if current_w_idx <= 0 or current_w_idx >= len(syllables_info or []):
        return None

    prev_syl = syllables_info[current_w_idx - 1]
    prev_phones = prev_syl.get("phones") or []
    if not prev_phones:
        return None

    from core.kr_oto_rules import find_vowel_phone
    from core.kr_oto_cv import adaptive_overlap
    from core.oto_generator import validate_oto_params

    _pv_idx, prev_v_phone = find_vowel_phone(prev_phones)
    prev_v_start = float(prev_v_phone.minTime) * 1000.0
    prev_v_end = float(prev_v_phone.maxTime) * 1000.0
    prev_v_len = max(prev_v_end - prev_v_start, 26.0)
    curr_v_len = max(float(n_end) - float(n_start), 35.0)

    tail_keep = _clamp(prev_v_len * 0.22, 20.0, 76.0)
    offset = max(prev_v_end - tail_keep, 0.0)
    boundary = max(float(n_start), offset + 18.0)
    pre = boundary - offset
    ovl = max(adaptive_overlap(pre, "", mode="vv"), min(pre * 0.68, max(pre - 8.0, 0.0)))
    added_cons = _clamp(curr_v_len * 0.34, 56.0, 156.0)
    consonant = pre + added_cons

    cut_gap = _clamp(curr_v_len * 0.44, 72.0, 240.0)
    cutoff_abs = consonant + cut_gap
    curr_v_end_rel = max(float(n_end) - offset, consonant + 18.0)
    # VV는 다음 모음 연결 내부에서 닫혀야 하므로 현재 모음 끝을 크게 넘지 않게 제한한다.
    cutoff_cap = curr_v_end_rel + _clamp(curr_v_len * 0.08, 8.0, 22.0)
    cutoff_abs = min(cutoff_abs, cutoff_cap)
    cutoff = -cutoff_abs
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


__all__ = ["_compute_kr_cvvc_vv_timing_direct"]
