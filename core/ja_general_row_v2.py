from __future__ import annotations

from core.mapping_format_policy import is_ja_sequence_locked_format


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
    anchor_lock_lite=False,
    alignment_weight=0.0,
    textgrid_trust_tier="",
):
    def _clamp_vc_tail_to_next_consonant(
        _offset: float,
        _consonant: float,
        _cutoff: float,
        _pre: float,
        _ovl: float,
    ):
        next_on = float(n_start)
        next_end = max(float(n_end), next_on + 8.0)
        if next_on <= 0.0:
            return _offset, _consonant, _cutoff, _pre, _ovl
        hard_cls = bool(c_char in {"k", "g", "t", "d", "b", "p", "q", "c", "ch", "ts", "dz", "s", "z", "sh", "j", "h", "f", "v", "hy"})
        son_cls = bool(c_char in {"m", "n", "ny", "r", "l", "ry", "w", "y"})

        o = float(_offset)
        p = float(_pre)
        ovl_abs = o + max(float(_ovl), 0.0)
        c_abs = o + float(_consonant)
        cut_abs = o + abs(float(_cutoff))
        next_pre_abs = next_on
        next_pre = 0.0
        if isinstance(vc_next_anchor, dict):
            try:
                next_pre_abs = float(vc_next_anchor.get("pre_abs", next_on) or next_on)
                next_pre = max(float(vc_next_anchor.get("pre", 0.0) or 0.0), 0.0)
            except Exception:
                next_pre_abs = next_on
                next_pre = 0.0
        next_offset_abs = next_pre_abs - next_pre
        if next_offset_abs <= 0.0:
            next_offset_abs = next_on

        pre_abs = o + p
        if hard_cls:
            pre_abs = min(max(pre_abs, next_offset_abs - 7.0), next_offset_abs + 8.0)
        elif son_cls:
            pre_abs = min(max(pre_abs, next_offset_abs - 5.0), next_offset_abs + 12.0)
        else:
            pre_abs = min(max(pre_abs, next_offset_abs - 6.0), next_offset_abs + 10.0)
        o = max(pre_abs - p, 0.0)
        c_abs = o + float(_consonant)
        cut_abs = o + abs(float(_cutoff))
        ovl = max(0.0, min(p, ovl_abs - o))

        if hard_cls:
            # Stop/fricative-like VC should terminate before next onset.
            c_floor = max(p + 6.0 + o, next_on - 12.0)
            c_cap = next_on - 2.0
        elif son_cls:
            c_floor = max(next_on + 4.0, p + 12.0 + o)
            c_cap = next_end - 3.0
        else:
            c_floor = max(next_on + 2.0, p + 10.0 + o)
            c_cap = next_end - 4.0
        if c_cap <= c_floor:
            c_cap = c_floor + 2.0
        c_abs = min(max(c_abs, c_floor), c_cap)

        if hard_cls:
            cut_floor = max(c_abs + 4.0, next_on - 2.0)
            cut_cap = next_on - 0.7
        elif son_cls:
            cut_floor = max(c_abs + 10.0, next_end - 4.0)
            cut_cap = next_end + 6.0
        else:
            cut_floor = max(c_abs + 8.0, next_end - 3.0)
            cut_cap = next_end + 4.0
        if cut_cap <= cut_floor:
            cut_cap = cut_floor + 1.0
        cut_abs = min(max(cut_abs, cut_floor), cut_cap)

        return validate_fn(o, c_abs - o, -(cut_abs - o), p, ovl)

    soft_off_shift = 0.0
    soft_cut_shift = 0.0
    cutoff_reduced = 0.0
    mel_voiced_onset_ms = None

    if alias_type == "cv" and mel_ctx_for_file:
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
    adjusted = post_ctx.post_adjust_result(
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
    offset, consonant, cutoff, pre, ovl = adjusted.as_tuple()
    if alias_type == "cv":
        offset, consonant, cutoff, pre, _offset_adjusted = post_ctx.guard_cv_head_offset_to_onset(
            offset,
            consonant,
            cutoff,
            pre,
            current_w_idx,
            alias_text=alias,
            alias_type="cv",
        )
        offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)

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
        if is_ja_sequence_locked_format(format_type):
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
                alias_type=alias_type,
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

    if alias_type in {"vc", "vv"}:
        mel_cutoff_candidate_ms = None
        next_mel_voiced_onset_ms = None
        if mel_ctx_for_file:
            from core.oto_generator import _estimate_mel_cutoff_candidate

            pre_abs = float(offset) + float(pre)
            cut_abs = float(offset) + abs(float(cutoff))
            mel_cutoff_candidate_ms = _estimate_mel_cutoff_candidate(mel_ctx_for_file, pre_abs, cut_abs)
        if vc_next_anchor is not None:
            try:
                next_mel_voiced_onset_ms = float(vc_next_anchor.get("mel_voiced_onset_abs", 0.0) or 0.0) or None
            except Exception:
                next_mel_voiced_onset_ms = None
        if mel_cutoff_candidate_ms is not None or next_mel_voiced_onset_ms is not None:
            from core.ja_oto_bridge import _apply_vc_vv_mel_cutoff_cap

            offset, consonant, cutoff, pre, ovl = _apply_vc_vv_mel_cutoff_cap(
                offset,
                consonant,
                cutoff,
                pre,
                ovl,
                alias_type=alias_type,
                format_type=format_type,
                mel_cutoff_candidate_ms=mel_cutoff_candidate_ms,
                next_mel_voiced_onset_ms=next_mel_voiced_onset_ms,
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
        lite=bool(anchor_lock_lite),
        voiced_onset_ms=mel_voiced_onset_ms,
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
    if alias_type == "vc":
        offset, consonant, cutoff, pre, ovl = _clamp_vc_tail_to_next_consonant(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
        )
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
            mel_voiced_onset_abs=mel_voiced_onset_ms,
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
