from __future__ import annotations


def build_realized_cv_anchor(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    onset_abs,
    vowel_start_abs,
    vowel_end_abs,
    c_end_abs=None,
    mel_voiced_onset_abs=None,
):
    vowel_start_abs = float(vowel_start_abs)
    vowel_end_abs = float(vowel_end_abs)
    return {
        "offset": float(offset),
        "pre": float(pre),
        "ovl": float(ovl),
        "cons": float(consonant),
        "cutoff": float(cutoff),
        "pre_abs": float(offset + pre),
        "cons_abs": float(offset + consonant),
        "onset_abs": float(onset_abs),
        "vowel_start_abs": vowel_start_abs,
        "vowel_end_abs": vowel_end_abs,
        "vowel_len": max(12.0, float(vowel_end_abs - vowel_start_abs)),
        "cons_gap": max(float(consonant - pre), 10.0),
        "cut_gap": max(float(abs(cutoff) - consonant), 16.0),
        "c_end_abs": None if c_end_abs is None else float(c_end_abs),
        "mel_voiced_onset_abs": None if mel_voiced_onset_abs is None else float(mel_voiced_onset_abs),
    }


def prepare_vcv_syllable_timing(
    syllables_info,
    current_w_idx,
    cv_seq_idx,
    diphthong_cv_consonant_ratio,
    *,
    forced_w_idx=None,
    prepare_cv_bounds_fn,
    adaptive_overlap_fn,
    validate_fn,
):
    if not syllables_info:
        return current_w_idx, cv_seq_idx, 0.0, 50.0, -100.0, 20.0, 12.0

    if forced_w_idx is not None:
        current_w_idx = max(0, min(int(forced_w_idx), len(syllables_info) - 1))
        cv_seq_idx = max(cv_seq_idx, current_w_idx + 1)
    elif cv_seq_idx < len(syllables_info):
        current_w_idx = cv_seq_idx
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    (
        current_w_idx,
        curr_phones,
        c_start,
        c_end,
        n_start,
        n_end,
    ) = prepare_cv_bounds_fn(syllables_info, current_w_idx)

    if current_w_idx > 0:
        (
            _prev_idx,
            _prev_phones,
            _prev_c_start,
            _prev_c_end,
            prev_v_start,
            prev_v_end,
        ) = prepare_cv_bounds_fn(syllables_info, current_w_idx - 1)
    else:
        prev_v_end = max(c_start, n_start)
        prev_v_start = max(0.0, prev_v_end - 100.0)

    c_boundary = c_end
    prev_v_len = max(prev_v_end - prev_v_start, 40.0)
    offset_padding = min(prev_v_len * 0.6, 200.0)
    if offset_padding < 80.0:
        offset_padding = max(prev_v_len * 0.5, 50.0)

    offset = max(prev_v_end - offset_padding, 0.0)
    pre = c_boundary - offset
    c_hint = curr_phones[0].mark if curr_phones else ""
    ovl = adaptive_overlap_fn(pre, c_hint, mode="vcv")

    vowel_len = max(n_end - c_boundary, 20.0)
    added_cons = min(vowel_len * float(diphthong_cv_consonant_ratio), 150.0)
    if added_cons < 50.0:
        added_cons = 50.0
    consonant = pre + added_cons
    cutoff = -(consonant + vowel_len * 0.25)
    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    return current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl


def extract_vcv_anchor_points(syllables_info, current_w_idx):
    if not (0 <= int(current_w_idx) < len(syllables_info or [])):
        return None, None, None
    curr = syllables_info[int(current_w_idx)]
    phones = curr.get("phones") or []
    if not phones:
        return None, None, None
    if len(phones) >= 2:
        anchor = float(phones[-1].minTime) * 1000.0
        return anchor, anchor, float(phones[-1].maxTime) * 1000.0
    anchor = float(phones[0].maxTime) * 1000.0
    return anchor, None, None


__all__ = [
    "build_realized_cv_anchor",
    "extract_vcv_anchor_points",
    "prepare_vcv_syllable_timing",
]
