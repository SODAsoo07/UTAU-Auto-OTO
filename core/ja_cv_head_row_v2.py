from __future__ import annotations


def run_ja_cv_head_row(
    *,
    curr_phones,
    current_w_idx,
    alias,
    format_type,
    base_shape,
    mel_ctx_for_file,
    post_ctx,
    real_wav_name,
    final_lines,
    generate_openutau,
    fname,
    log_fn,
    validate_fn,
    extract_cv_bounds_fn,
    cv_offset_and_pre_fn,
    adaptive_overlap_fn,
    soft_guard_fn,
    apply_base_shape_blend_fn,
    apply_anchor_lock_fn,
    target_vowel_from_alias_fn,
    pick_vowel_phone_fn,
    build_anchor_record_fn,
    finalize_row_fn,
    row_builder_fn,
    generate_openutau_aliases_fn,
    alias_out_fn,
    build_guard_messages_fn,
    anchor_store,
    anchor_lock_lite=False,
    alignment_weight=0.0,
    textgrid_trust_tier="",
):
    c_start, c_end, n_start, n_end = extract_cv_bounds_fn(
        curr_phones,
        alias_text=alias,
        alias_type="cv_head",
    )

    c_hint = curr_phones[0].mark if curr_phones else ""
    offset, pre = cv_offset_and_pre_fn(
        c_start,
        c_end,
        alias,
        c_hint=c_hint,
        alias_type="cv_head",
        vowel_start=n_start,
        vowel_end=n_end,
    )
    ovl = adaptive_overlap_fn(pre, c_hint, mode="cv_head")
    cv_vowel_len = n_end - n_start
    added_cons = min(cv_vowel_len * 0.5, 150.0)
    if added_cons < 80.0:
        added_cons = 80.0
    consonant = pre + added_cons
    min_cut_abs = post_ctx.cv_head_min_cutoff(offset, consonant, pre, n_start, n_end)
    if min_cut_abs is None:
        cutoff = -(consonant + cv_vowel_len * 0.25)
    else:
        cutoff = -max(consonant + 14.0, min_cut_abs)

    mel_voiced_onset_ms = None
    if mel_ctx_for_file:
        from core.oto_generator import (
            _estimate_mel_voiced_onset,
            _resolve_mel_onset_weight,
            _apply_mel_voiced_onset_pre_shift,
        )
        pre_abs = float(offset) + float(pre)
        mel_weight = _resolve_mel_onset_weight(alignment_weight, textgrid_trust_tier)
        if mel_weight > 0.0:
            mel_onset = _estimate_mel_voiced_onset(mel_ctx_for_file, pre_abs)
            if mel_onset is not None and abs(float(mel_onset) - pre_abs) <= 120.0:
                (
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    _mel_shift,
                ) = _apply_mel_voiced_onset_pre_shift(
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    mel_onset,
                    weight=mel_weight,
                )
                mel_voiced_onset_ms = float(mel_onset)

    offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = soft_guard_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        "cv_head",
        mel_ctx_for_file,
        onset_hint=c_hint,
        alias_text=alias,
    )
    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
        log_fn(
            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
        )

    offset, consonant, cutoff, pre, ovl = apply_base_shape_blend_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        base_shape,
        alias_type="cv_head",
    )
    adjusted = post_ctx.post_adjust_result(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="cv_head",
        alias_text=alias,
        local_end_ms=n_end,
        local_cut_allow_ms=28.0,
    )
    offset, consonant, cutoff, pre, ovl = adjusted.as_tuple()
    offset, consonant, cutoff, pre, ovl = apply_anchor_lock_fn(
        fname=fname,
        alias_text=alias,
        format_type=format_type,
        alias_type="cv_head",
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        anchor_abs_ms=c_end,
        next_onset_abs_ms=n_start,
        next_vowel_abs_ms=n_end,
        lite=bool(anchor_lock_lite),
        voiced_onset_ms=mel_voiced_onset_ms,
    )
    offset, consonant, cutoff, pre, offset_reduced = post_ctx.guard_cv_head_offset_to_onset(
        offset,
        consonant,
        cutoff,
        pre,
        current_w_idx,
        alias_text=alias,
        format_type=format_type,
    )
    v_cov_start = n_start
    v_cov_end = n_end
    try:
        v_target = target_vowel_from_alias_fn(alias, alias_type="cv_head")
        v_phone, _v_idx = pick_vowel_phone_fn(curr_phones, target_vowel=v_target)
        if v_phone is not None:
            v_cov_start = float(v_phone.minTime) * 1000.0
            v_cov_end = float(v_phone.maxTime) * 1000.0
    except Exception:
        pass
    offset, consonant, cutoff, pre, cutoff_extended = post_ctx.ensure_cv_head_min_vowel_coverage(
        offset,
        consonant,
        cutoff,
        pre,
        v_cov_start,
        v_cov_end,
    )
    offset, consonant, cutoff, pre, cutoff_reduced = post_ctx.guard_cv_cutoff_to_next_onset(
        offset,
        consonant,
        cutoff,
        pre,
        current_w_idx,
        alias_type="cv_head",
        format_type=format_type,
        vowel_start_ms=v_cov_start,
        vowel_end_ms=v_cov_end,
    )
    offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
    anchor_record = build_anchor_record_fn(
        current_w_idx,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        onset_abs=c_start,
        vowel_start_abs=n_start,
        c_end_abs=c_end,
        vowel_end_abs=n_end,
        mel_voiced_onset_abs=mel_voiced_onset_ms,
    )
    finalize_row_fn(
        final_lines=final_lines,
        row_builder_fn=row_builder_fn,
        real_wav_name=real_wav_name,
        alias=alias,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        generate_openutau=generate_openutau,
        generate_openutau_aliases_fn=generate_openutau_aliases_fn,
        alias_out_fn=alias_out_fn,
        anchor_store=anchor_store,
        anchor_record=anchor_record,
        log_fn=log_fn,
        messages=build_guard_messages_fn(
            fname,
            alias,
            cutoff_reduced=cutoff_reduced,
            offset_reduced=offset_reduced,
            cutoff_extended=cutoff_extended,
        ),
    )


__all__ = ["run_ja_cv_head_row"]
