from __future__ import annotations


def _clamp_range(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _blend(a, b, weight):
    w = _clamp_range(weight, 0.0, 1.0)
    return (float(a) * (1.0 - w)) + (float(b) * w)


def _enforce_stoplike_vc_before_next_onset(
    pre,
    consonant,
    cutoff_abs,
    next_onset_rel,
    *,
    cons_margin=6.0,
    cut_margin=1.0,
    min_cons_gap=6.0,
    min_cut_gap=4.0,
):
    pre = float(max(pre, 0.0))
    consonant = float(max(consonant, pre + float(min_cons_gap)))
    cutoff_abs = float(max(cutoff_abs, consonant + float(min_cut_gap)))
    next_onset_rel = float(max(next_onset_rel, pre + 2.0))

    hard_cons_cap = float(next_onset_rel) - float(cons_margin)
    if consonant > hard_cons_cap:
        consonant = hard_cons_cap
    consonant = max(consonant, pre + float(min_cons_gap))

    hard_cut_cap = float(next_onset_rel) - float(cut_margin)
    min_cut_abs = consonant + float(min_cut_gap)
    if min_cut_abs > hard_cut_cap:
        consonant = max(pre + float(min_cons_gap), hard_cut_cap - float(min_cut_gap))
        min_cut_abs = consonant + float(min_cut_gap)
    if min_cut_abs > hard_cut_cap:
        hard_cut_cap = min_cut_abs + 0.8

    cutoff_abs = min(float(cutoff_abs), float(hard_cut_cap))
    cutoff_abs = max(float(cutoff_abs), float(min_cut_abs))
    return float(consonant), float(cutoff_abs)


def build_realized_cv_anchor(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    onset_abs,
    vowel_start_abs,
    c_end_abs,
    vowel_end_abs,
    mel_voiced_onset_abs=None,
):
    vowel_len = max(float(vowel_end_abs) - float(vowel_start_abs), 20.0)
    return {
        "offset": float(offset),
        "pre": float(pre),
        "ovl": float(ovl),
        "cons": float(consonant),
        "cutoff": float(cutoff),
        "pre_abs": float(offset) + float(pre),
        "cons_abs": float(offset) + float(consonant),
        "onset_abs": float(onset_abs),
        "vowel_start_abs": float(vowel_start_abs),
        "c_end_abs": float(c_end_abs),
        "vowel_end_abs": float(vowel_end_abs),
        "vowel_len": vowel_len,
        "cons_gap": max(float(consonant) - float(pre), 10.0),
        "cut_gap": max(abs(float(cutoff)) - float(consonant), 16.0),
        "mel_voiced_onset_abs": None if mel_voiced_onset_abs is None else float(mel_voiced_onset_abs),
    }


def estimate_ja_cv_anchor(
    idx,
    syllables_info,
    *,
    extract_cv_bounds_fn,
    cv_offset_and_pre_fn,
    adaptive_overlap_fn,
    validate_fn,
):
    if idx < 0 or idx >= len(syllables_info):
        return None
    syl = syllables_info[idx]
    curr_phones = syl.get("phones") or []
    if not curr_phones:
        return None

    anchor_alias = syl.get("roman_cv") or syl.get("roman") or syl.get("word") or ""
    c_start, c_end, n_start, n_end = extract_cv_bounds_fn(
        curr_phones,
        alias_text=anchor_alias,
        alias_type="cv",
    )

    c_hint = curr_phones[0].mark if curr_phones else ""
    cv_vowel_len = max(float(n_end) - float(n_start), 20.0)
    offset, pre = cv_offset_and_pre_fn(
        c_start,
        c_end,
        anchor_alias,
        c_hint=c_hint,
        alias_type="cv",
        vowel_start=n_start,
        vowel_end=n_end,
    )
    pre = max(float(pre), 10.0)
    ovl = adaptive_overlap_fn(pre, c_hint, mode="cv")
    v_ref = max(cv_vowel_len, 120.0)
    added_cons = min(max(v_ref * 0.45, 70.0), 180.0)
    consonant = pre + added_cons
    cutoff = -(consonant + max(cv_vowel_len * 0.25, 45.0))

    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    return build_realized_cv_anchor(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        onset_abs=c_start,
        vowel_start_abs=n_start,
        c_end_abs=c_end,
        vowel_end_abs=n_end,
    )


def compute_vcv_params_from_virtual_split(
    alias,
    prev_v_start,
    prev_v_end,
    c_boundary,
    n_end,
    *,
    base_shape=None,
    extract_target_syllable_fn,
    split_syllable_fn,
    get_bridge_profile_fn,
    is_n_bridge_fn,
    onset_class_fn,
    validate_fn,
    apply_base_shape_blend_fn,
):
    prev_v_len = max(float(prev_v_end) - float(prev_v_start), 40.0)
    curr_v_len = max(float(n_end) - float(c_boundary), 40.0)
    transition_gap = max(float(c_boundary) - float(prev_v_end), 0.0)

    target = extract_target_syllable_fn(alias)
    onset, _ = split_syllable_fn(target)
    onset = (onset or "").strip().lower()
    profile = get_bridge_profile_fn(onset)
    n_bridge = bool(is_n_bridge_fn(alias, "vcv"))
    onset_cls = onset_class_fn(onset)

    pre_lead = _clamp_range(
        transition_gap * float(profile.get("pre_lead_mul", 0.35)),
        float(profile.get("pre_lead_min", 8.0)),
        float(profile.get("pre_lead_max", 30.0)),
    )
    if n_bridge:
        pre_lead = _clamp_range(
            transition_gap * (0.16 if onset_cls in {"voiced", "nasal"} else 0.20),
            4.0,
            14.0 if onset_cls in {"voiced", "nasal"} else 18.0,
        )
    boundary = max(float(prev_v_end) + 6.0, float(c_boundary) - pre_lead)
    if n_bridge:
        boundary = max(float(prev_v_end) + 2.0, float(c_boundary) - pre_lead)
        late_shift = _clamp_range(curr_v_len * 0.18, 8.0, 24.0)
        boundary = min(float(n_end) - 8.0, boundary + late_shift)

    base_pad = float(profile.get("offset_pad", 86.0))
    dyn_pad = base_pad + max(prev_v_len - 120.0, 0.0) * float(profile.get("offset_len_mul", 0.08))
    pad_lo = float(profile.get("offset_pad_min", 42.0))
    pad_hi = min(260.0, max(prev_v_len * 0.94, base_pad + 40.0))
    offset_padding = _clamp_range(dyn_pad, pad_lo, pad_hi)
    if prev_v_len < offset_padding:
        offset_padding = max(prev_v_len * 0.76, float(profile.get("offset_pad_floor", 36.0)))
    if n_bridge:
        pad_cap = 70.0 if onset_cls in {"voiced", "nasal"} else 82.0
        offset_padding = min(offset_padding, max(prev_v_len * 0.60, pad_cap))

    offset = max(boundary - offset_padding, 0.0)
    pre_floor = 44.0 if n_bridge else 36.0
    pre = max(boundary - offset, pre_floor)

    tail_margin = float(profile.get("tail_margin_base", 10.0)) + prev_v_len * float(profile.get("tail_margin_mul", 0.05))
    tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
    target_ovl_abs = float(prev_v_end) - tail_margin
    upper_ovl = max(pre - float(profile.get("ovl_pre_margin", 6.0)), 0.0)
    lower_ovl = min(pre * float(profile.get("ovl_min_ratio", 0.40)), upper_ovl)
    ovl_anchored = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
    ovl = _blend(ovl_anchored, pre * float(profile.get("ovl_ratio", 0.50)), 0.34)
    ovl = min(upper_ovl, max(lower_ovl, ovl))

    n_ref = max(curr_v_len, 70.0 if n_bridge else 60.0)
    cons_add = float(profile.get("cons_add_base", 36.0)) + max(n_ref - 70.0, 0.0) * float(profile.get("cons_add_mul", 0.12))
    cons_add = _clamp_range(
        cons_add,
        float(profile.get("cons_add_min", 20.0)),
        float(profile.get("cons_add_max", 68.0)),
    )
    if n_bridge:
        cons_add = min(cons_add + 14.0, float(profile.get("cons_add_max", 68.0)) + 18.0)
    consonant = pre + cons_add
    consonant = max(consonant, pre + float(profile.get("cons_floor", 18.0)))
    consonant = min(consonant, pre + max(curr_v_len * 0.72, 96.0))

    cut_add = float(profile.get("cut_add_base", 58.0)) + max(n_ref - 70.0, 0.0) * float(profile.get("cut_add_mul", 0.20))
    cut_add = _clamp_range(
        cut_add,
        float(profile.get("cut_add_min", 34.0)),
        float(profile.get("cut_add_max", 120.0)),
    )
    if n_bridge:
        cut_add = min(cut_add + 20.0, float(profile.get("cut_add_max", 120.0)) + 28.0)
    cutoff_abs = max(
        consonant + float(profile.get("cut_min_gap", 16.0)),
        pre + cut_add,
    )
    end_rel = max(float(n_end) - offset, pre + 40.0)
    cutoff_abs = min(cutoff_abs, end_rel + float(profile.get("cut_to_next_allow", 22.0)))
    if cutoff_abs <= consonant + 10.0:
        cutoff_abs = consonant + 12.0
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    return apply_base_shape_blend_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        base_shape,
        alias_type="vcv",
    )


def compute_ja_vc_from_adjacent_cv(
    prev_cv,
    next_cv,
    *,
    alias_type,
    c_char,
    bridge_profile,
    plosive_consonants,
    validate_fn,
):
    if not prev_cv or not next_cv:
        return None

    profile = dict(bridge_profile or {})
    a_type = str(alias_type or "").strip().lower()
    is_hard_stoplike = a_type == "vc" and str(c_char or "") in set(plosive_consonants or [])
    if a_type == "vc":
        next_onset_abs = float(next_cv["onset_abs"])
        boundary_abs = next_onset_abs
        if is_hard_stoplike:
            raw_transition = max(next_onset_abs - float(prev_cv["vowel_end_abs"]), 12.0)
            onset_lead = _clamp_range(raw_transition * 0.26, 8.0, 20.0)
            boundary_abs = max(float(prev_cv["vowel_end_abs"]) + 4.0, next_onset_abs - onset_lead)
    else:
        boundary_abs = next_cv["pre_abs"]
    pre_target = _clamp_range(_blend(prev_cv["pre"], next_cv["pre"], 0.34), 40.0, 220.0)
    offset = max(float(boundary_abs) - pre_target, 0.0)
    pre = float(boundary_abs) - offset
    if pre <= 0.0:
        return None

    tail_margin = float(profile.get("tail_margin_base", 10.0)) + float(prev_cv["vowel_len"]) * float(profile.get("tail_margin_mul", 0.05))
    tail_margin = _clamp_range(tail_margin, 4.0, 24.0)
    target_ovl_abs = float(prev_cv["vowel_end_abs"]) - tail_margin
    upper_ovl = max(pre - float(profile.get("ovl_pre_margin", 6.0)), 0.0)
    lower_ovl = min(pre * float(profile.get("ovl_min_ratio", 0.40)), upper_ovl)
    ovl = min(upper_ovl, max(lower_ovl, target_ovl_abs - offset))
    ovl = _blend(ovl, pre * float(profile.get("ovl_ratio", 0.50)), 0.20)
    ovl = min(upper_ovl, max(lower_ovl, ovl))

    cons_gap = _clamp_range(_blend(prev_cv["cons_gap"], next_cv["cons_gap"], 0.45), 14.0, 120.0)
    consonant = pre + cons_gap
    next_onset_rel = max(float(next_cv["onset_abs"]) - offset, pre + 10.0)
    next_pre_rel = max(float(next_cv["pre_abs"]) - offset, pre + 16.0)
    next_cons_rel = max(float(next_cv["cons_abs"]) - offset, next_pre_rel + 10.0)

    if a_type == "vc":
        if is_hard_stoplike:
            consonant = min(consonant, next_onset_rel - 7.0)
            consonant = max(consonant, pre + 6.0)
            cutoff_abs = max(consonant + 6.0, next_onset_rel - 1.2)
            consonant, cutoff_abs = _enforce_stoplike_vc_before_next_onset(
                pre,
                consonant,
                cutoff_abs,
                next_onset_rel,
                cons_margin=7.0,
                cut_margin=1.2,
                min_cons_gap=6.0,
                min_cut_gap=4.0,
            )
        else:
            consonant = min(consonant, next_onset_rel + 24.0)
            consonant = max(consonant, pre + 16.0)
            cutoff_abs = max(consonant + 12.0, min(next_cons_rel + 24.0, next_pre_rel + 40.0))
    else:
        consonant = min(max(consonant, pre + 22.0), next_pre_rel + 44.0)
        cutoff_abs = max(consonant + 20.0, next_pre_rel + 10.0)
        cutoff_abs = min(cutoff_abs, next_cons_rel + 54.0)

    cutoff = -float(cutoff_abs)
    return validate_fn(offset, consonant, cutoff, pre, ovl)


def enforce_vcv_cv_entry_guard(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    c_boundary,
    n_end,
    alias_text,
    is_n_bridge_fn,
    extract_target_syllable_fn,
    split_syllable_fn,
    validate_fn,
):
    offset = float(max(offset, 0.0))
    consonant = float(max(consonant, 0.0))
    pre = float(max(pre, 0.0))
    ovl = float(max(ovl, 0.0))
    c_boundary = float(c_boundary)
    n_end = float(max(n_end, c_boundary))

    n_bridge = bool(is_n_bridge_fn(alias_text, "vcv"))
    tok = extract_target_syllable_fn(alias_text)
    onset, _vowel = split_syllable_fn(tok)
    vv_like = not bool((onset or "").strip())

    pre_abs = offset + pre
    if vv_like:
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 4.0, 0.0), c_boundary + 10.0)
        pre_floor = 52.0
    elif n_bridge:
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 2.0, 0.0), c_boundary + 14.0)
        pre_floor = 50.0
    else:
        pre_abs = _clamp_range(pre_abs, max(c_boundary - 8.0, 0.0), c_boundary + 5.0)
        pre_floor = 44.0
    if (pre_abs - offset) < pre_floor:
        offset = max(pre_abs - pre_floor, 0.0)
    pre = max(pre_abs - offset, 0.0)

    vowel_start_rel = max(c_boundary - offset, pre + 8.0)
    vowel_len = max(n_end - c_boundary, 40.0)
    if vv_like:
        min_vowel_keep = _clamp_range(vowel_len * 0.30, 34.0, 88.0)
        min_cons_rel = max(pre + 72.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 86.0, pre + 142.0)
    elif n_bridge:
        min_vowel_keep = _clamp_range(vowel_len * 0.30, 34.0, 96.0)
        min_cons_rel = max(pre + 76.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 96.0, pre + 156.0)
    else:
        min_vowel_keep = _clamp_range(vowel_len * 0.22, 24.0, 76.0)
        min_cons_rel = max(pre + 64.0, vowel_start_rel + min_vowel_keep)
        max_cons_rel = max(vowel_start_rel + min_vowel_keep + 70.0, pre + 110.0)
    consonant = _clamp_range(consonant, min_cons_rel, max_cons_rel)

    cutoff_abs = abs(float(cutoff))
    min_cut_abs = consonant + (34.0 if n_bridge else 28.0)
    max_cut_abs = max((n_end - offset) + 86.0, consonant + 36.0)
    cutoff_abs = _clamp_range(cutoff_abs, min_cut_abs, max_cut_abs)
    cutoff = -cutoff_abs

    if vv_like:
        vv_gap_target = _clamp_range(vowel_len * 0.06, 5.0, 10.0)
        ovl = max(ovl, pre - vv_gap_target)
        ovl = min(ovl, max(pre - 2.0, 0.0))
    if ovl > pre:
        ovl = pre * 0.72
    return validate_fn(offset, consonant, cutoff, pre, ovl)


def enforce_cv_pre_anchor_guard(offset, consonant, cutoff, pre, ovl, *, c_end_abs, alias_type, validate_fn):
    offset = float(max(offset, 0.0))
    pre = float(max(pre, 0.0))
    consonant = float(max(consonant, 0.0))
    ovl = float(max(ovl, 0.0))
    c_end_abs = float(max(c_end_abs, 0.0))

    pre_abs = offset + pre
    win = 10.0 if str(alias_type or "").strip().lower() == "cv_head" else 7.0
    target_abs = _clamp_range(pre_abs, max(c_end_abs - win, 0.0), c_end_abs + win)
    if abs(target_abs - pre_abs) > 2.0:
        new_offset = target_abs - pre
        if new_offset < 0.0:
            offset = 0.0
            pre = target_abs
        else:
            offset = new_offset

    if ovl > pre:
        ovl = pre * 0.72
    return validate_fn(offset, consonant, cutoff, pre, ovl)


__all__ = [
    "build_realized_cv_anchor",
    "compute_ja_vc_from_adjacent_cv",
    "compute_vcv_params_from_virtual_split",
    "enforce_cv_pre_anchor_guard",
    "enforce_vcv_cv_entry_guard",
    "estimate_ja_cv_anchor",
]
