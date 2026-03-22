from __future__ import annotations


def _clamp(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _resolve_kr_vcv_prev_vowel_tail_ratio(consonant_hint: str) -> float:
    """VCV 앞모음 포함 비율(끝에서부터) 기본값을 onset 계열별로 반환."""
    hint = str(consonant_hint or "").strip().lower()
    hard_like = {
        "k",
        "t",
        "p",
        "ch",
        "c",
        "kk",
        "tt",
        "pp",
        "jj",
        "kh",
        "th",
        "ph",
    }
    fric_like = {"s", "ss", "sh", "h", "f", "z", "x"}
    son_like = {"m", "n", "ng", "l", "r", "y", "w", "ɫ", "ɾ"}
    if hint in hard_like:
        return 0.40
    if hint in fric_like:
        return 0.38
    if hint in son_like:
        return 0.34
    return 0.36


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
    c_hint = curr_phones[0].mark if curr_phones else ""
    prev_tail_ratio = _resolve_kr_vcv_prev_vowel_tail_ratio(c_hint)
    # Korean VCV guideline:
    # keep only tail part of previous vowel (about 1/3 ~ 2/5).
    prev_tail_lo = prev_v_len / 3.0
    prev_tail_hi = prev_v_len * 0.40
    offset_padding = _clamp(prev_v_len * prev_tail_ratio, prev_tail_lo, prev_tail_hi)
    offset = max(prev_v_end - offset_padding, 0.0)
    pre = c_boundary - offset
    ovl = adaptive_overlap_fn(pre, c_hint, mode="vcv")

    vowel_len = max(n_end - c_boundary, 20.0)
    # Include the trailing CV almost entirely in the sampled span.
    cv_end_rel = max(n_end - offset, pre + 24.0)
    cons_tail_keep = _clamp(vowel_len * 0.08, 6.0, 16.0)
    consonant = max(pre + 20.0, cv_end_rel - cons_tail_keep)
    consonant = min(consonant, cv_end_rel - 4.0)
    cutoff_tail_add = _clamp(vowel_len * 0.05, 2.0, 10.0)
    cutoff_abs = max(consonant + 12.0, cv_end_rel + cutoff_tail_add)
    cutoff = -cutoff_abs
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
