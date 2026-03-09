from __future__ import annotations


def run_ja_general_row(
    *,
    final_lines,
    real_wav_name,
    alias,
    alias_type,
    format_type,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    c_end,
    n_start,
    n_end,
    c_start,
    current_w_idx,
    generate_openutau,
    fname,
    log_fn,
    base_shape,
    mel_ctx_for_file,
    onset_hint_local,
    mapping_tier,
    c_char,
    vc_prev_anchor,
    vc_next_anchor,
    post_ctx,
    validate_fn,
    soft_guard_fn,
    base_shape_blend_fn,
    refine_ja_vc_fn,
    onset_class_fn,
    is_n_bridge_fn,
    limit_pre_anchor_shift_fn,
    apply_anchor_lock_fn,
    anchor_lock_enabled_fn,
    enforce_cv_pre_anchor_guard_fn,
    build_anchor_record_fn,
    finalize_row_fn,
    row_builder_fn,
    generate_openutau_aliases_fn,
    alias_out_fn,
    build_guard_messages_fn,
    anchor_store,
):
    soft_off_shift = 0.0
    soft_cut_shift = 0.0
    cutoff_reduced = 0.0

    if alias_type in {"cv", "cv_head", "vcv"}:
        offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = soft_guard_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            alias_type,
            mel_ctx_for_file,
            onset_hint=onset_hint_local,
            alias_text=alias,
        )
        if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
            log_fn(
                f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
            )

    offset, consonant, cutoff, pre, ovl = base_shape_blend_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        base_shape,
        alias_type=alias_type,
    )
    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type=alias_type,
        alias_text=alias,
        local_end_ms=n_end,
        local_cut_allow_ms=(44.0 if alias_type == "vc" else 40.0 if alias_type == "vv" else 54.0),
    )

    if alias_type == "vc":
        pre_abs_before = offset + pre
        offset, consonant, cutoff, pre, ovl = refine_ja_vc_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            c_char=c_char,
            prev_cv_anchor=vc_prev_anchor,
            next_cv_anchor=vc_next_anchor,
            prev_v_end_abs=c_end,
            next_c_start_abs=n_start,
            next_c_end_abs=n_end,
        )
        pre_abs_after = offset + pre
        if format_type in {"cvvc", "cv"}:
            max_shift = 26.0
            if mapping_tier == "high":
                max_shift = 34.0
            onset_cls = onset_class_fn(c_char)
            if onset_cls in {"voiced", "nasal"}:
                max_shift += 4.0
            if is_n_bridge_fn(alias, "vc"):
                max_shift = min(max_shift, 22.0)
            (
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                pre_abs_after,
                clamped_shift,
            ) = limit_pre_anchor_shift_fn(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                pre_abs_before=pre_abs_before,
                max_shift_ms=max_shift,
            )
            if clamped_shift:
                log_fn(
                    f"🛡️ {fname}: VC-CV 앵커 이동 제한 "
                    f"({pre_abs_before:.1f}->{pre_abs_after:.1f}ms, {alias})"
                )
        if abs(pre_abs_after - pre_abs_before) >= 6.0:
            log_fn(
                f"🧭 {fname}: VC-CV 앵커 재정렬 "
                f"({pre_abs_before:.1f}->{pre_abs_after:.1f}ms, {alias})"
            )

    anchor_abs = c_end
    next_onset_abs = n_start
    next_vowel_abs = n_end
    if alias_type == "vc":
        anchor_abs = n_start
    elif alias_type == "vv":
        anchor_abs = c_end

    offset, consonant, cutoff, pre, ovl = apply_anchor_lock_fn(
        fname=fname,
        alias_text=alias,
        format_type=format_type,
        alias_type=alias_type,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        anchor_abs_ms=anchor_abs,
        next_onset_abs_ms=next_onset_abs,
        next_vowel_abs_ms=next_vowel_abs,
    )
    if alias_type in {"cv", "cv_head"} and not anchor_lock_enabled_fn("japanese", format_type):
        offset, consonant, cutoff, pre, ovl = enforce_cv_pre_anchor_guard_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            c_end_abs=c_end,
            alias_type=alias_type,
        )

    anchor_record = None
    messages = None
    if alias_type in {"cv", "cv_head"}:
        offset, consonant, cutoff, pre, cutoff_reduced = post_ctx.guard_cv_cutoff_to_next_onset(
            offset,
            consonant,
            cutoff,
            pre,
            current_w_idx,
            alias_type=alias_type,
            format_type=format_type,
            vowel_start_ms=(n_start if alias_type == "cv_head" else None),
            vowel_end_ms=(n_end if alias_type == "cv_head" else None),
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
        )
        messages = build_guard_messages_fn(
            fname,
            alias,
            cutoff_reduced=cutoff_reduced,
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
        anchor_store=anchor_store if anchor_record is not None else None,
        anchor_record=anchor_record,
        log_fn=log_fn if messages else None,
        messages=messages,
    )


__all__ = ["run_ja_general_row"]
