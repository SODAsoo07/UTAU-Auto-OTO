from __future__ import annotations


def _clamp(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _resolve_vcv_onset_profile(c_hint: str):
    hint = str(c_hint or "").strip().lower()
    if hint in {"s", "ss", "sh", "h", "ch", "j", "jj", "c", "ts"}:
        return {
            "tail_ratio": 0.34,
            "pre_floor": 46.0,
            "pre_ceil": 176.0,
            "cons_follow_ratio": 0.20,
            "cons_follow_min": 16.0,
            "cons_follow_max": 34.0,
            "tail_cap_min": 54.0,
            "tail_cap_max": 88.0,
        }
    if hint in {"m", "n", "ng", "l", "r", "y", "w", "ny"}:
        return {
            "tail_ratio": 0.38,
            "pre_floor": 58.0,
            "pre_ceil": 210.0,
            "cons_follow_ratio": 0.24,
            "cons_follow_min": 22.0,
            "cons_follow_max": 44.0,
            "tail_cap_min": 64.0,
            "tail_cap_max": 108.0,
        }
    if hint in {"k", "g", "kk", "t", "d", "tt", "p", "b", "pp"}:
        return {
            "tail_ratio": 0.33,
            "pre_floor": 42.0,
            "pre_ceil": 164.0,
            "cons_follow_ratio": 0.18,
            "cons_follow_min": 14.0,
            "cons_follow_max": 30.0,
            "tail_cap_min": 50.0,
            "tail_cap_max": 84.0,
        }
    return {
        "tail_ratio": 0.36,
        "pre_floor": 48.0,
        "pre_ceil": 188.0,
        "cons_follow_ratio": 0.22,
        "cons_follow_min": 18.0,
        "cons_follow_max": 36.0,
        "tail_cap_min": 56.0,
        "tail_cap_max": 96.0,
    }


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

    c_boundary = n_start if float(n_start or 0.0) > float(c_start or 0.0) else c_end
    c_hint = curr_phones[0].mark if curr_phones else ""
    profile = _resolve_vcv_onset_profile(c_hint)

    prev_v_len = max(prev_v_end - prev_v_start, 40.0)
    # Korean VCV: keep only the tail section of the previous vowel (about 1/3~2/5).
    tail_lo = prev_v_len / 3.0
    tail_hi = prev_v_len * 0.40
    offset_padding = _clamp(prev_v_len * float(profile["tail_ratio"]), tail_lo, tail_hi)
    tail_cap = _clamp(prev_v_len * 0.42, float(profile["tail_cap_min"]), float(profile["tail_cap_max"]))
    offset_padding = min(offset_padding, tail_cap)
    offset = max(prev_v_end - offset_padding, 0.0)
    pre = c_boundary - offset
    pre_floor = float(profile["pre_floor"])
    pre_ceil = float(profile["pre_ceil"])
    if pre < pre_floor:
        offset = max(c_boundary - pre_floor, 0.0)
        pre = c_boundary - offset
    elif pre > pre_ceil:
        offset = max(c_boundary - pre_ceil, 0.0)
        pre = c_boundary - offset
    ovl = adaptive_overlap_fn(pre, c_hint, mode="vcv")

    vowel_len = max(n_end - c_boundary, 20.0)
    cv_end_rel = max(n_end - offset, pre + 24.0)
    onset_rel = max(c_boundary - offset, pre + 4.0)
    cons_follow = _clamp(
        vowel_len * float(profile["cons_follow_ratio"]),
        float(profile["cons_follow_min"]),
        float(profile["cons_follow_max"]),
    )
    consonant = max(pre + float(profile["cons_follow_min"]), onset_rel + cons_follow)
    consonant = min(consonant, cv_end_rel - 8.0)
    cutoff_tail_add = _clamp(vowel_len * 0.08, 3.0, 12.0)
    cutoff = -(max(consonant + 12.0, cv_end_rel + cutoff_tail_add))
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
