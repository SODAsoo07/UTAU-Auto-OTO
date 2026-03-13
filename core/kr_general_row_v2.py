from __future__ import annotations


def run_kr_general_row(
    *,
    final_lines,
    real_wav_name,
    alias,
    alias_type,
    file_format,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    soft_off_shift,
    soft_cut_shift,
    cutoff_reduced,
    selected_w_idx,
    current_w_idx,
    c_end,
    n_start,
    n_end,
    c_start,
    row_mapping_confidence,
    timeline_start_ms,
    timeline_end_ms,
    wav_duration_ms,
    generate_openutau,
    alias_suffix,
    fname,
    log_fn,
    validate_fn,
    bridge_pair,
    realized_cv_anchor_by_idx,
    cv_anchor_by_idx,
    refine_bridge_fn,
    resolve_anchor_targets_fn,
    apply_anchor_lock_fn,
    build_anchor_record_fn,
    build_bridge_message_fn,
    finalize_row_fn,
    row_builder_fn,
    log_post_timing_events_fn,
    anchor_lock_lite=False,
):
    bridge_shift = 0.0
    if alias_type in {"vc", "vv"}:
        prev_idx = bridge_pair.get("prev_idx")
        next_idx = bridge_pair.get("next_idx")
        if prev_idx is None or next_idx is None:
            if alias_type == "vc":
                prev_idx = current_w_idx
                next_idx = current_w_idx + 1
            elif selected_w_idx is not None and selected_w_idx >= 1:
                prev_idx = selected_w_idx - 1
                next_idx = selected_w_idx
        prev_anchor = bridge_pair.get("prev_anchor")
        next_anchor = bridge_pair.get("next_anchor")
        if prev_anchor is None and prev_idx is not None:
            prev_anchor = realized_cv_anchor_by_idx.get(prev_idx) or cv_anchor_by_idx.get(prev_idx)
        if next_anchor is None and next_idx is not None:
            next_anchor = realized_cv_anchor_by_idx.get(next_idx) or cv_anchor_by_idx.get(next_idx)
        if prev_anchor is not None and next_anchor is not None:
            pre_abs_before = float(offset + pre)
            offset, consonant, cutoff, pre, ovl = refine_bridge_fn(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                alias_type=alias_type,
                alias_text=alias,
                prev_cv=prev_anchor,
                next_cv=next_anchor,
            )
            bridge_shift = float((offset + pre) - pre_abs_before)

    anchor_abs, next_onset_abs, next_vowel_abs = resolve_anchor_targets_fn(
        alias_type,
        c_end,
        n_start,
        n_end,
    )
    offset, consonant, cutoff, pre, ovl = apply_anchor_lock_fn(
        fname=fname,
        alias_text=alias,
        format_type=file_format,
        alias_type=alias_type,
        offset=offset,
        consonant=consonant,
        cutoff=cutoff,
        pre=pre,
        ovl=ovl,
        timeline_start_ms=timeline_start_ms,
        timeline_end_ms=timeline_end_ms,
        file_duration_ms=wav_duration_ms,
        anchor_abs_ms=anchor_abs,
        next_onset_abs_ms=next_onset_abs,
        next_vowel_abs_ms=next_vowel_abs,
        mapping_confidence=row_mapping_confidence,
        lite=bool(anchor_lock_lite),
    )
    anchor_record = None
    if alias_type == "cv" and selected_w_idx is not None:
        anchor_record = build_anchor_record_fn(
            selected_w_idx,
            offset=offset,
            consonant=consonant,
            cutoff=cutoff,
            pre=pre,
            ovl=ovl,
            onset_abs=c_start,
            vowel_start_abs=n_start,
            vowel_end_abs=n_end,
            c_end_abs=c_end,
        )
    bridge_msg = build_bridge_message_fn(
        fname,
        alias,
        alias_type,
        bridge_shift,
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
        alias_type=alias_type,
        file_format=file_format,
        wav_duration_ms=wav_duration_ms,
        validate_fn=validate_fn,
        log_post_timing_events_fn=log_post_timing_events_fn,
        log_fn=log_fn,
        fname=fname,
        soft_off_shift=soft_off_shift,
        soft_cut_shift=soft_cut_shift,
        cutoff_reduced=cutoff_reduced,
        anchor_store=realized_cv_anchor_by_idx,
        anchor_record=anchor_record,
        messages=([bridge_msg] if bridge_msg is not None else []),
    )


__all__ = ["run_kr_general_row"]
