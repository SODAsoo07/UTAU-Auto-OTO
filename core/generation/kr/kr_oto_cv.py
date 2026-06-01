from __future__ import annotations

import os

from core.kr_oto_rules import (
    _extract_alias_onset,
    _is_sonorant_consonant,
    _is_tense_consonant,
    find_vowel_phone,
    is_plosive_ipa,
    is_plosive_roman,
)

# 마찰음 (onset 소음이 길다)
KR_FRICATIVE_ONSETS = {"s", "ss", "sh", "h", "f"}
# 격음 (기식이 길다)
KR_ASPIRATE_ONSETS = {"k", "t", "p", "ch"}
# Keep this to TextGrid phone labels only. UTAU release aliases such as "R"
# are handled at alias level; bare "r" is a real Korean liquid phone here.
_SILENCE_PHONE_MARKS = {"", "sil", "pau", "sp", "spn", "br", "bre", "breath"}


def _is_fricative_consonant(ipa_hint="", roman_hint=""):
    h = (ipa_hint or "").strip().lower()
    r = (roman_hint or "").strip().lower()
    return h in KR_FRICATIVE_ONSETS or r in KR_FRICATIVE_ONSETS


def _is_aspirate_consonant(ipa_hint="", roman_hint=""):
    h = (ipa_hint or "").strip().lower()
    r = (roman_hint or "").strip().lower()
    return h in KR_ASPIRATE_ONSETS or r in KR_ASPIRATE_ONSETS


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _is_silence_like_phone_mark(mark: str) -> bool:
    m = str(mark or "").strip().lower()
    return m in _SILENCE_PHONE_MARKS


def _normalize_kr_cv_timing_mode(cv_mode="standalone"):
    mode = str(cv_mode or "").strip().lower().replace("-", "_")
    if mode in {"vc_context", "with_vc", "vc_present", "bridge"}:
        return "vc_context"
    if mode in {"vcv_context", "vcv_with_vc", "vcv_bridge", "vcv"}:
        return "vcv_context"
    return "standalone"


def _retune_kr_cv_for_mode(offset, consonant, cutoff, pre, ovl, cv_vowel_len, *, is_diph, cv_mode):
    """
    Separate CV timing behavior by VC context.
    - standalone: keep legacy behavior (CV/VCV or VC-missing banks)
    - vc_context: tighten CV body to leave more room for explicit VC aliases
    """
    mode_norm = _normalize_kr_cv_timing_mode(cv_mode)
    if mode_norm == "standalone":
        return offset, consonant, cutoff, pre, ovl

    pre_old = float(pre)
    cons_gap_old = max(float(consonant) - pre_old, 10.0)
    cut_gap_old = max(abs(float(cutoff)) - float(consonant), 12.0)

    if mode_norm == "vcv_context":
        pre_scale = 0.94
        cons_scale = 0.90 if is_diph else 0.88
        cut_scale = 0.90 if is_diph else 0.86
        pre_min = 48.0 if is_diph else 38.0
        pre_max = 176.0 if is_diph else 160.0
    else:
        pre_scale = 0.90
        cons_scale = 0.84 if is_diph else 0.82
        cut_scale = 0.82 if is_diph else 0.78
        pre_min = 42.0 if is_diph else 34.0
        pre_max = 170.0 if is_diph else 150.0

    pre_new = _clamp(pre_old * pre_scale, pre_min, pre_max)
    shift = max(0.0, pre_old - pre_new)
    offset = max(float(offset) + shift, 0.0)
    pre = pre_new

    if is_diph:
        cons_gap = _clamp(cons_gap_old * cons_scale, 58.0, 170.0)
        cut_gap = _clamp(cut_gap_old * cut_scale, 34.0, 92.0)
    else:
        cons_gap = _clamp(cons_gap_old * cons_scale, 38.0, 142.0)
        cut_gap = _clamp(cut_gap_old * cut_scale, 34.0, 84.0)
    consonant = pre + cons_gap
    cutoff_floor = max((float(cv_vowel_len) * (0.16 if is_diph else 0.18)), 30.0)
    cutoff = -(consonant + max(cut_gap, cutoff_floor))
    ovl = min(float(ovl), max(pre * 0.44, 8.0))
    return offset, consonant, cutoff, pre, ovl


def adaptive_overlap(pre, consonant_hint="", mode="cv"):
    from core.oto_generator import adaptive_overlap as _adaptive_overlap

    return _adaptive_overlap(pre, consonant_hint, mode)


def _compute_kr_cv_timing(
    c_start,
    c_end,
    cv_vowel_len,
    c_hint,
    alias_onset,
    is_diph,
    is_plosive,
    *,
    glide_dur_ms=0.0,
    v_ref_baseline=None,
    cv_mode="standalone",
):
    boundary = c_end
    cv_vowel_len = max(float(cv_vowel_len), 20.0)
    glide_dur_ms = max(float(glide_dur_ms or 0.0), 0.0)
    is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
    is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)
    baseline = 130.0
    try:
        if v_ref_baseline is not None:
            baseline = _clamp(float(v_ref_baseline), 80.0, 160.0)
    except Exception:
        baseline = 130.0

    if is_diph:
        c_len = max(c_end - c_start, 10.0)
        glide_lead = _clamp(glide_dur_ms, 0.0, 120.0) * 0.60
        target_pre = max(64.0, min(136.0, c_len + glide_lead + 14.0))
        if is_tense_cv:
            target_pre = _clamp(target_pre + 3.0, 66.0, 154.0)
        elif is_sonorant_cv:
            target_pre = min(160.0, target_pre + 10.0)
        offset = max(boundary - target_pre, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
        if is_tense_cv:
            ovl = min(ovl, max(pre * 0.32, 9.0))
        elif is_sonorant_cv:
            ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))

        diph_baseline = _clamp(max(baseline * 1.08, 92.0), 92.0, 172.0)
        v_ref = max(cv_vowel_len, diph_baseline)
        if is_tense_cv:
            added_cons = min(max(v_ref * 0.52, 86.0), 196.0)
        elif is_sonorant_cv:
            added_cons = min(max(v_ref * 0.62, 98.0), 240.0)
        else:
            added_cons = min(max(v_ref * 0.55, 90.0), 230.0)
        consonant = pre + added_cons
        cutoff = -(consonant + max(cv_vowel_len * 0.2, 45.0))
        return _retune_kr_cv_for_mode(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            cv_vowel_len,
            is_diph=True,
            cv_mode=cv_mode,
        )

    is_fricative_cv = _is_fricative_consonant(c_hint, alias_onset)
    is_aspirate_cv = _is_aspirate_consonant(c_hint, alias_onset)

    if is_tense_cv:
        lead = 34.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 54.0, 176.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
        ovl = min(ovl, max(pre * 0.32, 9.0))
    elif is_fricative_cv:
        # 마찰음: onset 소음이 길어 pre를 넓게, overlap도 약간 넓게
        lead = 46.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 62.0, 208.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
        ovl = max(ovl, min(pre * 0.42, max(pre - 10.0, 0.0)))
    elif is_aspirate_cv:
        # 격음: 기식이 길어 pre를 넓게, overlap 약간 넓게
        lead = 42.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 58.0, 198.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
        ovl = max(ovl, min(pre * 0.38, max(pre - 10.0, 0.0)))
    elif is_sonorant_cv:
        lead = 44.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 66.0, 216.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
        ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))
    elif is_plosive:
        lead = 32.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 50.0, 164.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")
    else:
        lead = 40.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 58.0, 192.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode="cv")

    v_ref = max(cv_vowel_len, baseline)
    if is_tense_cv:
        added_cons = min(max(v_ref * 0.42, 68.0), 162.0)
    elif is_fricative_cv:
        # 마찰음: 소음 구간이 길어 cons를 넓게
        added_cons = min(max(v_ref * 0.50, 82.0), 200.0)
    elif is_aspirate_cv:
        # 격음: 기식 구간 반영
        added_cons = min(max(v_ref * 0.48, 78.0), 195.0)
    elif is_sonorant_cv:
        added_cons = min(max(v_ref * 0.52, 86.0), 210.0)
    else:
        added_cons = min(max(v_ref * 0.45, 70.0), 180.0)
    consonant = pre + added_cons
    cutoff = -(consonant + max(cv_vowel_len * 0.25, 45.0))
    return _retune_kr_cv_for_mode(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        cv_vowel_len,
        is_diph=False,
        cv_mode=cv_mode,
    )


def _select_kr_cv_onset_slice(curr_phones):
    if not curr_phones:
        return None

    if len(curr_phones) == 1:
        ph = curr_phones[0]
        t = float(ph.minTime) * 1000.0
        return 0, t, t, t, float(ph.maxTime) * 1000.0, 0.0

    v_idx, v_phone = find_vowel_phone(curr_phones)
    v_idx = max(0, int(v_idx))
    n_start = float(v_phone.minTime) * 1000.0
    n_end = float(v_phone.maxTime) * 1000.0

    if v_idx <= 0:
        return 0, n_start, n_start, n_start, n_end, 0.0

    onset_idx = v_idx - 1
    while onset_idx >= 0 and _is_silence_like_phone_mark(curr_phones[onset_idx].mark):
        onset_idx -= 1
    if onset_idx < 0:
        return 0, n_start, n_start, n_start, n_end, 0.0

    glide_dur_ms = 0.0
    last_mark = (curr_phones[onset_idx].mark or "").strip().lower()
    from core.kr_oto_rules import is_glide as _is_glide
    if onset_idx - 1 >= 0 and _is_glide(last_mark):
        glide_phone_idx = onset_idx  # save before overwriting
        prev_idx = onset_idx - 1
        while prev_idx >= 0 and _is_silence_like_phone_mark(curr_phones[prev_idx].mark):
            prev_idx -= 1
        if prev_idx >= 0:
            onset_idx = prev_idx
            glide_start_ms = float(curr_phones[glide_phone_idx].minTime) * 1000.0
            glide_dur_ms = max(0.0, n_start - glide_start_ms)

    c_start = float(curr_phones[onset_idx].minTime) * 1000.0
    c_end = n_start
    return onset_idx, c_start, c_end, n_start, n_end, glide_dur_ms


def _estimate_cv_anchor_from_syllable(syl, ph_intervals, *, cv_mode="standalone"):
    from core.oto_generator import _stabilize_params_to_phone_activity, validate_oto_params, is_diphthong

    curr_phones = (syl or {}).get("phones") or []
    if not curr_phones:
        return None

    roman_tok = (syl.get("roman_cv") or syl.get("roman") or "")
    onset_slice = _select_kr_cv_onset_slice(curr_phones)
    glide_dur_ms = 0.0
    if onset_slice is not None:
        onset_idx, c_start, c_end, n_start, n_end, glide_dur_ms = onset_slice
    else:
        onset_idx = 0
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000
        glide_dur_ms = 0.0

    c_hint = curr_phones[onset_idx].mark if curr_phones and 0 <= onset_idx < len(curr_phones) else ""
    alias_onset = _extract_alias_onset(roman_tok)
    is_diph_syl = is_diphthong(roman_tok)
    cv_vowel_len = max(n_end - n_start, 20.0)
    first_phone_plosive = len(curr_phones) >= 2 and is_plosive_ipa(curr_phones[0].mark)
    is_plosive = (first_phone_plosive or is_plosive_roman(alias_onset)) if alias_onset else first_phone_plosive
    v_ref_baseline = (syl or {}).get("v_ref_baseline_ms")
    offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
        c_start,
        c_end,
        cv_vowel_len,
        c_hint,
        alias_onset,
        is_diph_syl,
        is_plosive,
        glide_dur_ms=glide_dur_ms,
        v_ref_baseline=v_ref_baseline,
        cv_mode=cv_mode,
    )

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    offset, consonant, cutoff, pre, ovl = _stabilize_params_to_phone_activity(
        offset, consonant, cutoff, pre, ovl, ph_intervals, alias_type="cv"
    )
    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return {
        "offset": offset,
        "pre": pre,
        "ovl": ovl,
        "cons": consonant,
        "cutoff": cutoff,
        "pre_abs": offset + pre,
        "cons_abs": offset + consonant,
        "onset_abs": c_start,
        "vowel_start_abs": n_start,
        "vowel_end_abs": n_end,
        "vowel_len": cv_vowel_len,
        "cons_gap": max(consonant - pre, 10.0),
        "cut_gap": max(abs(cutoff) - consonant, 16.0),
    }


def _prepare_cv_head_syllable_timing(syllables_info, current_w_idx, cv_seq_idx, alias, forced_w_idx=None):
    """CV_HEAD 계산용 음절 선택과 기본 timing을 구성합니다."""
    if forced_w_idx is not None:
        current_w_idx = max(0, min(int(forced_w_idx), len(syllables_info) - 1))
        cv_seq_idx = max(cv_seq_idx, current_w_idx + 1)
    elif cv_seq_idx < len(syllables_info):
        current_w_idx = cv_seq_idx
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]

    onset_slice = _select_kr_cv_onset_slice(curr_phones)
    if onset_slice is not None:
        onset_idx, c_start, c_end, n_start, n_end, _glide_dur_ms = onset_slice
    else:
        onset_idx = 0
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000

    c_hint = curr_phones[onset_idx].mark if curr_phones and 0 <= onset_idx < len(curr_phones) else ""
    alias_onset = _extract_alias_onset(alias)
    is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
    is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)

    if is_tense_cv:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 34.0, 50.0, 170.0)
        offset = max(c_end - pre_target, 0.0)
    elif is_sonorant_cv:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 42.0, 62.0, 205.0)
        offset = max(c_end - pre_target, 0.0)
    else:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 32.0, 48.0, 162.0)
        offset = max(c_end - pre_target, 0.0)

    pre = c_end - offset if c_end > c_start else 30
    ovl = adaptive_overlap(pre, c_hint, mode="cv_head")
    if is_tense_cv:
        ovl = min(ovl, max(pre * 0.32, 9.0))
    elif is_sonorant_cv:
        ovl = max(ovl, min(pre * 0.48, max(pre - 10.0, 0.0)))

    cv_vowel_len = n_end - n_start
    if is_tense_cv:
        added_cons = min(max(cv_vowel_len * 0.46, 72), 156)
    elif is_sonorant_cv:
        added_cons = min(max(cv_vowel_len * 0.58, 88), 190)
    else:
        added_cons = min(cv_vowel_len * 0.5, 150)
        if added_cons < 80:
            added_cons = 80

    consonant = pre + added_cons
    cutoff = -(consonant + cv_vowel_len * 0.25)
    return current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl, n_start, n_end


def _prepare_cv_bounds_from_syllable(syllables_info, current_w_idx):
    """CV 계산용 (curr_phones, c_start/c_end/n_start/n_end)를 구성합니다."""
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]

    onset_slice = _select_kr_cv_onset_slice(curr_phones)
    if onset_slice is not None:
        _onset_idx, c_start, c_end, n_start, n_end, _glide_dur_ms = onset_slice
    else:
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000

    return current_w_idx, curr_phones, c_start, c_end, n_start, n_end


__all__ = [
    "_compute_kr_cv_timing",
    "_estimate_cv_anchor_from_syllable",
    "_prepare_cv_bounds_from_syllable",
    "_prepare_cv_head_syllable_timing",
    "_select_kr_cv_onset_slice",
]
