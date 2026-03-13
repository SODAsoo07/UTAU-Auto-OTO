from __future__ import annotations


def run_ja_vcv_row(
    *,
    syllables_info,
    current_w_idx,
    stable_vcv_seq_idx,
    cv_seq_idx,
    mapped_idx,
    alias,
    base_shape,
    mel_ctx_for_file,
    onset_hint_local,
    post_ctx,
    real_wav_name,
    final_lines,
    generate_openutau,
    fname,
    log_fn,
    prepare_cv_bounds_fn,
    compute_vcv_params_fn,
    soft_guard_fn,
    enforce_vcv_cv_entry_guard_fn,
    extract_target_syllable_fn,
    split_syllable_fn,
    is_n_bridge_fn,
    apply_anchor_lock_fn,
    finalize_row_fn,
    row_builder_fn,
    generate_openutau_aliases_fn,
    alias_out_fn,
    anchor_lock_lite=False,
):
    current_w_idx = int(mapped_idx)
    post_file_format = str(getattr(post_ctx, "file_format", "") or "").strip().lower()
    if post_file_format == "vcv":
        stable_vcv_seq_idx = min(stable_vcv_seq_idx + 1, max(len(syllables_info) - 1, 0))
        cv_seq_idx = stable_vcv_seq_idx
    else:
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1
    last_vcv_mapped_idx = current_w_idx

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl["phones"]

    if current_w_idx > 0:
        prev_syl = syllables_info[current_w_idx - 1]
        prev_phones = prev_syl["phones"]
        prev_v_start = prev_phones[-1].minTime * 1000.0
        prev_v_end = prev_phones[-1].maxTime * 1000.0
    else:
        prev_v_start = curr_phones[0].minTime * 1000.0 - 100.0
        prev_v_end = curr_phones[0].minTime * 1000.0

    _c_start, c_boundary, n_start, n_end = prepare_cv_bounds_fn(
        curr_phones,
        alias_text=alias,
        alias_type="vcv",
    )
    offset, consonant, cutoff, pre, ovl = compute_vcv_params_fn(
        alias,
        prev_v_start,
        prev_v_end,
        c_boundary,
        n_end,
        base_shape=base_shape,
    )
    offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = soft_guard_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        "vcv",
        mel_ctx_for_file,
        onset_hint=onset_hint_local,
        alias_text=alias,
    )
    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
        log_fn(
            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
        )

    offset, consonant, cutoff, pre, ovl = post_ctx.post_adjust(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="vcv",
        alias_text=alias,
        local_end_ms=n_end,
        local_cut_allow_ms=26.0,
    )
    offset, consonant, cutoff, pre, ovl = enforce_vcv_cv_entry_guard_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        c_boundary=c_boundary,
        n_end=n_end,
        alias_text=alias,
    )
    vcv_tok_for_profile = extract_target_syllable_fn(alias)
    vcv_onset_for_profile, _vcv_vowel_for_profile = split_syllable_fn(vcv_tok_for_profile)
    n_bridge_vcv = is_n_bridge_fn(alias, "vcv")
    vv_like_profile_key = "vcv_vv_like" if not (vcv_onset_for_profile or "").strip() else "vcv"
    if n_bridge_vcv and vv_like_profile_key == "vcv":
        vv_like_profile_key = "vcv_n_bridge"
    offset, consonant, cutoff, pre, ovl = apply_anchor_lock_fn(
        fname=fname,
        alias_text=alias,
        format_type="vcv",
        alias_type="vcv",
        profile_alias_type=vv_like_profile_key,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        anchor_abs_ms=c_boundary,
        next_onset_abs_ms=n_start,
        next_vowel_abs_ms=n_end,
        lite=bool(anchor_lock_lite),
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
    )
    return current_w_idx, cv_seq_idx, stable_vcv_seq_idx, last_vcv_mapped_idx


__all__ = ["run_ja_vcv_row"]
