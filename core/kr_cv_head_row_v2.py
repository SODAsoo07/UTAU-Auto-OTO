from __future__ import annotations

from core.mapping_format_policy import is_kr_order_locked_timing_format


def run_kr_cv_head_row(
    *,
    syllables_info,
    current_w_idx,
    cv_seq_idx,
    alias,
    forced_w_idx,
    file_format,
    real_wav_name,
    final_lines,
    generate_openutau,
    alias_suffix,
    wav_duration_ms,
    timeline_start_ms,
    timeline_end_ms,
    row_mapping_confidence,
    mel_ctx_for_file,
    base_shape,
    ph_intervals,
    fname,
    log_fn,
    validate_fn,
    prepare_cv_head_syllable_timing_fn,
    apply_post_timing_pipeline_fn,
    guard_cv_head_offset_to_current_onset_fn,
    ensure_cv_head_min_vowel_coverage_fn,
    guard_cv_cutoff_to_next_onset_fn,
    prepare_cv_head_anchor_context_fn,
    prepare_cv_bounds_fn,
    apply_anchor_lock_fn,
    build_anchor_record_fn,
    finalize_row_fn,
    row_builder_fn,
    build_guard_messages_fn,
    log_post_timing_events_fn,
    anchor_store,
    anchor_lock_lite=False,
    alignment_weight=0.0,
    textgrid_trust_tier="",
):
    order_locked = bool(is_kr_order_locked_timing_format(file_format))
    min_keep_from_pre_ms = 112.0 if order_locked else 96.0

    (
        selected_w_idx,
        cv_seq_idx,
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        n_start,
        n_end,
    ) = prepare_cv_head_syllable_timing_fn(
        syllables_info,
        current_w_idx,
        cv_seq_idx,
        alias,
        forced_w_idx=forced_w_idx,
    )
    current_w_idx = max(current_w_idx, selected_w_idx)
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
    (
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        soft_off_shift,
        soft_cut_shift,
        cutoff_reduced,
    ) = apply_post_timing_pipeline_fn(
        offset,
        consonant,
        cutoff,
        pre,
        ovl,
        alias_type="cv_head",
        alias_text=alias,
        file_format=file_format,
        mel_ctx_for_file=mel_ctx_for_file,
        base_shape=base_shape,
        ph_intervals=ph_intervals,
        current_w_idx=selected_w_idx,
        syllables_info=syllables_info,
        is_vc_plosive_coda=False,
        enable_stabilize=False,
        enable_cutoff_guard=False,
    )
    (
        offset,
        consonant,
        cutoff,
        pre,
        offset_reduced,
    ) = guard_cv_head_offset_to_current_onset_fn(
        offset,
        consonant,
        cutoff,
        pre,
        selected_w_idx,
        syllables_info,
        file_format=file_format,
    )
    (
        offset,
        consonant,
        cutoff,
        pre,
        cutoff_extended,
    ) = ensure_cv_head_min_vowel_coverage_fn(
        offset,
        consonant,
        cutoff,
        pre,
        n_start,
        n_end,
        file_format=file_format,
    )
    (
        offset,
        consonant,
        cutoff,
        pre,
        cutoff_reduced_after_offset,
    ) = guard_cv_cutoff_to_next_onset_fn(
        offset,
        consonant,
        cutoff,
        pre,
        selected_w_idx,
        syllables_info,
        alias_type="cv_head",
        min_keep_from_pre_ms=min_keep_from_pre_ms,
    )
    cutoff_reduced += cutoff_reduced_after_offset

    cv_head_anchor_ctx = prepare_cv_head_anchor_context_fn(
        syllables_info,
        selected_w_idx,
        fallback_anchor_abs=n_start,
        prepare_cv_bounds_fn=prepare_cv_bounds_fn,
    )
    cvh_anchor = cv_head_anchor_ctx["anchor_abs"]
    if cvh_anchor is not None:
        offset, consonant, cutoff, pre, ovl = apply_anchor_lock_fn(
            fname=fname,
            alias_text=alias,
            format_type=file_format,
            alias_type="cv_head",
            offset=offset,
            consonant=consonant,
            cutoff=cutoff,
            pre=pre,
            ovl=ovl,
            timeline_start_ms=timeline_start_ms,
            timeline_end_ms=timeline_end_ms,
            file_duration_ms=wav_duration_ms,
            anchor_abs_ms=cvh_anchor,
            next_onset_abs_ms=n_start,
            next_vowel_abs_ms=n_end,
            mapping_confidence=row_mapping_confidence,
            lite=bool(anchor_lock_lite),
            voiced_onset_ms=mel_voiced_onset_ms,
        )
        (
            offset,
            consonant,
            cutoff,
            pre,
            offset_reduced_after_anchor,
        ) = guard_cv_head_offset_to_current_onset_fn(
            offset,
            consonant,
            cutoff,
            pre,
            selected_w_idx,
            syllables_info,
            file_format=file_format,
        )
        offset_reduced += offset_reduced_after_anchor
        (
            offset,
            consonant,
            cutoff,
            pre,
            cutoff_extended_after_anchor,
        ) = ensure_cv_head_min_vowel_coverage_fn(
            offset,
            consonant,
            cutoff,
            pre,
            n_start,
            n_end,
            file_format=file_format,
        )
        cutoff_extended += cutoff_extended_after_anchor
        (
            offset,
            consonant,
            cutoff,
            pre,
            cutoff_reduced_after_anchor,
        ) = guard_cv_cutoff_to_next_onset_fn(
            offset,
            consonant,
            cutoff,
            pre,
            selected_w_idx,
            syllables_info,
            alias_type="cv_head",
            min_keep_from_pre_ms=min_keep_from_pre_ms,
        )
        cutoff_reduced += cutoff_reduced_after_anchor
    anchor_record = build_anchor_record_fn(
        selected_w_idx,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        onset_abs=cv_head_anchor_ctx["onset_abs"],
        vowel_start_abs=(
            cv_head_anchor_ctx["vowel_start_abs"]
            if cv_head_anchor_ctx["vowel_start_abs"] is not None
            else n_start
        ),
        vowel_end_abs=(
            cv_head_anchor_ctx["vowel_end_abs"]
            if cv_head_anchor_ctx["vowel_end_abs"] is not None
            else n_end
        ),
        c_end_abs=cv_head_anchor_ctx["c_end_abs"],
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
        alias_suffix=alias_suffix,
        alias_type="cv_head",
        file_format=file_format,
        wav_duration_ms=wav_duration_ms,
        validate_fn=validate_fn,
        log_post_timing_events_fn=log_post_timing_events_fn,
        log_fn=log_fn,
        fname=fname,
        soft_off_shift=soft_off_shift,
        soft_cut_shift=soft_cut_shift,
        cutoff_reduced=cutoff_reduced,
        anchor_store=anchor_store,
        anchor_record=anchor_record,
        messages=build_guard_messages_fn(
            fname,
            alias,
            offset_reduced=offset_reduced,
            cutoff_extended=cutoff_extended,
        ),
    )
    return current_w_idx, cv_seq_idx


__all__ = ["run_kr_cv_head_row"]
