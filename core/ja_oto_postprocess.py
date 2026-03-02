"""
Japanese OTO postprocess helpers.

Isolates final timing stabilization and CV_HEAD safety guards from the
main generator loop so they can be adjusted independently.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

from core.ja_oto_mapping import _clean_phone_mark

JA_PLOSIVE_CONSONANTS = {
    'k', 'g', 't', 'd', 'b', 'p',
    'kk', 'tt', 'pp', 'dd', 'gg', 'bb',
    'ch', 'ts', 'q', 'c', 'j',
    'ky', 'gy', 'ty', 'dy', 'by', 'py',
}
JA_SIBILANT_ONSETS = {'s', 'z', 'sh', 'j', 'ts', 'dz', 'ch'}


def _nearest_phone_edge_ms(phone_spans_ms: Sequence[Tuple[float, float]], anchor_ms: float) -> Tuple[float, float]:
    nearest = None
    nearest_dist = float("inf")
    for s_ms, e_ms in phone_spans_ms:
        if s_ms <= anchor_ms <= e_ms:
            return anchor_ms, 0.0
        ds = abs(anchor_ms - s_ms)
        de = abs(anchor_ms - e_ms)
        if ds < nearest_dist:
            nearest_dist = ds
            nearest = s_ms
        if de < nearest_dist:
            nearest_dist = de
            nearest = e_ms
    if nearest is None:
        return anchor_ms, 0.0
    return nearest, nearest_dist


def _surrounding_gap_ms(phone_spans_ms: Sequence[Tuple[float, float]], anchor_ms: float) -> Tuple[Optional[float], Optional[float], float]:
    prev_end = None
    next_start = None
    for s_ms, e_ms in phone_spans_ms:
        if s_ms <= anchor_ms <= e_ms:
            return None, None, 0.0
        if e_ms < anchor_ms:
            prev_end = e_ms
        elif s_ms > anchor_ms and next_start is None:
            next_start = s_ms
            break
    if prev_end is None or next_start is None or next_start <= prev_end:
        return None, None, 0.0
    return prev_end, next_start, (next_start - prev_end)


def post_adjust_params(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    ovl: float,
    *,
    alias_type: str = "cv",
    alias_text: str = "",
    local_end_ms: Optional[float] = None,
    local_cut_allow_ms: Optional[float] = None,
    phone_spans_ms: Sequence[Tuple[float, float]],
    timeline_start_ms: float,
    effective_end_ms: float,
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
    recenter_fn: Callable[..., Tuple[float, float, float, float, float]],
    ja_style_enabled: bool = False,
    ja_style_profile=None,
    autotune_profile=None,
    style_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
    autotune_apply_fn: Optional[Callable[..., Tuple[float, float, float, float, float]]] = None,
) -> Tuple[float, float, float, float, float]:
    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)

    pre_abs = offset + pre
    nearest_edge, nearest_dist = _nearest_phone_edge_ms(phone_spans_ms, pre_abs)
    prev_end_ms, next_start_ms, gap_len_ms = _surrounding_gap_ms(phone_spans_ms, pre_abs)
    if gap_len_ms >= 55.0 or nearest_dist > 34.0:
        target = nearest_edge
        if prev_end_ms is not None and next_start_ms is not None:
            if alias_type in ("vc", "vv"):
                target = max(prev_end_ms + 4.0, next_start_ms - 6.0)
            elif alias_type in ("cv", "cv_head"):
                target = max(prev_end_ms + 3.0, next_start_ms - 4.0)
            else:
                target = prev_end_ms
        if abs(target - pre_abs) >= 2.0:
            pre_abs = target

    min_pre_abs = max(timeline_start_ms - (20.0 if alias_type in ("vc", "vv", "vcv") else 10.0), 0.0)
    max_pre_abs = effective_end_ms + (30.0 if alias_type in ("vc", "vv", "vcv") else 80.0)
    pre_abs = max(min_pre_abs, min(pre_abs, max_pre_abs))

    offset_floor = max(timeline_start_ms - (70.0 if alias_type in ("vc", "vv", "vcv") else 40.0), 0.0)
    if offset < offset_floor:
        offset = offset_floor

    if pre_abs - offset > 340.0:
        offset = max(pre_abs - 340.0, 0.0)
    if pre_abs < offset:
        pre_abs = offset + 10.0

    pre = max(pre_abs - offset, 0.0)
    if ovl > pre:
        ovl = pre * 0.72
    if consonant < pre + 25.0:
        consonant = pre + 25.0

    cut_anchor_ms = effective_end_ms if local_end_ms is None else float(local_end_ms)
    cut_allow_ms = 120.0 if local_cut_allow_ms is None else float(local_cut_allow_ms)
    cons_allow_ms = max(40.0, cut_allow_ms - 40.0)
    max_cons_abs = max((cut_anchor_ms + cons_allow_ms) - offset, pre + 40.0)
    consonant = min(consonant, max_cons_abs)
    max_cut_abs = max((cut_anchor_ms + cut_allow_ms) - offset, consonant + 35.0)
    cutoff = -min(abs(cutoff), max_cut_abs)
    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)

    if ja_style_enabled and ja_style_profile and not autotune_profile and style_apply_fn:
        offset, consonant, cutoff, pre, ovl = style_apply_fn(
            alias_type, offset, consonant, cutoff, pre, ovl, ja_style_profile, alias_text=alias_text
        )
    if autotune_profile and autotune_apply_fn:
        offset, consonant, cutoff, pre, ovl = autotune_apply_fn(
            alias_type, offset, consonant, cutoff, pre, ovl, autotune_profile
        )

    return recenter_fn(offset, consonant, cutoff, pre, ovl, alias_type=alias_type, alias_text=alias_text)


def ja_cv_head_min_cutoff_abs(offset: float, consonant: float, pre: float, vowel_start_ms: float, vowel_end_ms: float) -> Optional[float]:
    v_start = float(vowel_start_ms)
    v_end = float(vowel_end_ms)
    v_len = max(0.0, v_end - v_start)
    if v_len < 40.0:
        return None
    vowel_start_rel = max(v_start - float(offset), float(pre) + 8.0)
    keep_v_ms = min(210.0, max(85.0, v_len * 0.36))
    min_from_pre = min(185.0, max(95.0, v_len * 0.26))
    return max(float(consonant) + 12.0, vowel_start_rel + keep_v_ms, float(pre) + min_from_pre)


def guard_ja_cv_cutoff_to_next_onset(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
    *,
    alias_type: str = "cv",
    vowel_start_ms: Optional[float] = None,
    vowel_end_ms: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    if syll_idx is None or syll_idx < 0 or (syll_idx + 1) >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0
    next_syl = syllables_info[syll_idx + 1]
    next_phones = next_syl.get("phones") or []
    if not next_phones:
        return offset, consonant, cutoff, pre, 0.0

    next_mark = _clean_phone_mark(getattr(next_phones[0], "mark", ""))
    hard_next = next_mark in JA_PLOSIVE_CONSONANTS or next_mark in JA_SIBILANT_ONSETS or next_mark in {"ts", "ch", "j", "sh", "s", "z", "h"}
    is_cv_head = str(alias_type or "").strip().lower() == "cv_head"
    safety = 10.0 if is_cv_head else (14.0 if hard_next else 9.0)
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if is_cv_head:
        max_cutoff_abs = next_onset_rel + (10.0 if hard_next else 18.0)
    if max_cutoff_abs <= (pre + 18.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 14.0)
    consonant = max(consonant, pre + 10.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    min_cutoff_abs = None
    if is_cv_head and vowel_start_ms is not None and vowel_end_ms is not None:
        min_cutoff_abs = ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)
        if min_cutoff_abs is not None:
            cutoff_abs = max(cutoff_abs, min(min_cutoff_abs, max_cutoff_abs))
    if cutoff_abs <= (consonant + 8.0):
        cutoff_abs = min(max_cutoff_abs, consonant + 10.0)
        if cutoff_abs <= (consonant + 6.0):
            consonant = max(pre + 8.0, cutoff_abs - 10.0)
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, _ovl = validate_fn(offset, consonant, cutoff, pre, 0.0)
    reduction = max(0.0, original_cutoff_abs - abs(cutoff))
    return offset, consonant, cutoff, pre, reduction


def guard_ja_cv_head_offset_to_onset(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    syll_idx: Optional[int],
    syllables_info: Sequence[dict],
    extract_cv_bounds_fn: Callable[..., Tuple[float, float, float, float]],
    cv_onset_class_fn: Callable[..., Tuple[str, str]],
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
    *,
    alias_text: str = "",
) -> Tuple[float, float, float, float, float]:
    if syll_idx is None or syll_idx < 0 or syll_idx >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0
    curr_syl = syllables_info[syll_idx]
    curr_phones = curr_syl.get("phones") or []
    if not curr_phones:
        return offset, consonant, cutoff, pre, 0.0

    c_hint = curr_phones[0].mark if curr_phones else ""
    c_start, c_end, _n_start, _n_end = extract_cv_bounds_fn(curr_phones, alias_text=alias_text, alias_type="cv_head")
    cls, _onset = cv_onset_class_fn(alias_text, c_hint=c_hint, alias_type="cv_head")
    if cls == "voiceless":
        base_lead = 44.0
    elif cls == "voiced":
        base_lead = 36.0
    elif cls == "nasal":
        base_lead = 30.0
    else:
        base_lead = 34.0
    c_len = max(0.0, float(c_end) - float(c_start))
    lead_cap = min(base_lead, max(20.0, c_len + 16.0))
    offset_floor = max(0.0, float(c_start) - lead_cap)
    if offset >= offset_floor:
        return offset, consonant, cutoff, pre, 0.0

    new_offset = offset_floor
    new_pre = max(float(pre), 8.0)
    new_consonant = max(float(consonant), new_pre + 8.0)
    new_cut_abs = max(abs(float(cutoff)), new_consonant + 12.0)
    new_cutoff = -new_cut_abs
    new_offset, new_consonant, new_cutoff, new_pre, _ovl = validate_fn(new_offset, new_consonant, new_cutoff, new_pre, 0.0)
    reduced_ms = max(0.0, new_offset - offset)
    return new_offset, new_consonant, new_cutoff, new_pre, reduced_ms


def ensure_ja_cv_head_min_vowel_coverage(
    offset: float,
    consonant: float,
    cutoff: float,
    pre: float,
    vowel_start_ms: float,
    vowel_end_ms: float,
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
) -> Tuple[float, float, float, float, float]:
    min_cut_abs = ja_cv_head_min_cutoff_abs(offset, consonant, pre, vowel_start_ms, vowel_end_ms)
    if min_cut_abs is None:
        return offset, consonant, cutoff, pre, 0.0

    cut_abs = abs(float(cutoff))
    if cut_abs >= min_cut_abs:
        return offset, consonant, cutoff, pre, 0.0

    new_cutoff = -min_cut_abs
    offset, consonant, new_cutoff, pre, _ovl = validate_fn(offset, consonant, new_cutoff, pre, 0.0)
    extended_ms = max(0.0, abs(new_cutoff) - cut_abs)
    return offset, consonant, new_cutoff, pre, extended_ms
