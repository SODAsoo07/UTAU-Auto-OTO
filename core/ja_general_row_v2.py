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
        strict_cvvc = str(format_type or "").strip().lower() == "cvvc"

        o = float(_offset)
        p = float(_pre)
        c_abs = o + float(_consonant)
        cut_abs = o + abs(float(_cutoff))

        if strict_cvvc and hard_cls:
            c_floor = max(p + 6.0 + o, next_on - 14.0)
            c_cap = next_on - 2.5
        elif strict_cvvc and son_cls:
            c_floor = max(next_on - 6.0, p + 9.0 + o)
            c_cap = next_on + 1.5
        elif strict_cvvc:
            c_floor = max(next_on - 6.0, p + 9.0 + o)
            c_cap = next_on + 2.0
        elif hard_cls:
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

        if strict_cvvc and hard_cls:
            cut_floor = max(c_abs + 4.0, next_on - 2.0)
            cut_cap = next_on + 0.8
        elif strict_cvvc and son_cls:
            cut_floor = max(c_abs + 8.0, next_on + 0.4)
            cut_cap = min(next_end + 2.0, next_on + 5.0)
        elif strict_cvvc:
            cut_floor = max(c_abs + 7.0, next_on + 0.6)
            cut_cap = min(next_end + 2.0, next_on + 6.0)
        elif hard_cls:
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

        return validate_fn(o, c_abs - o, -(cut_abs - o), p, float(_ovl))

    def _tighten_cv_overlap_for_cvvc(
        _offset: float,
        _consonant: float,
        _cutoff: float,
        _pre: float,
        _ovl: float,
    ):
        if str(format_type or "").strip().lower() != "cvvc":
            return validate_fn(_offset, _consonant, _cutoff, _pre, _ovl)

        o, c, cut, p, v = validate_fn(_offset, _consonant, _cutoff, _pre, _ovl)
        p = max(float(p), 0.0)
        if p <= 0.0:
            return o, c, cut, p, 0.0

        onset_cls = str(onset_class_fn(c_char) or "").strip().lower()
        vowel_len = max(float(n_end) - float(n_start), 20.0)
        hard_cls = bool(c_char in {"k", "g", "t", "d", "b", "p", "q", "c", "ch", "ts", "dz", "s", "z", "sh", "j", "h", "f", "v", "hy"})

        ratio_cap = 0.66 if alias_type == "cv_head" else 0.62
        if onset_cls == "voiceless":
            ratio_cap -= 0.07
        elif onset_cls == "liquid":
            ratio_cap += 0.03
        elif onset_cls == "nasal":
            ratio_cap += 0.02
        elif onset_cls == "voiced":
            ratio_cap += 0.01
        if hard_cls:
            ratio_cap -= 0.03
        if vowel_len < 90.0:
            ratio_cap -= 0.05
        elif vowel_len > 220.0:
            ratio_cap += 0.02
        max_ratio_cap = 0.75 if onset_cls == "liquid" else 0.73 if onset_cls == "nasal" else 0.72
        ratio_cap = min(max(ratio_cap, 0.46), max_ratio_cap)

        min_gap = 8.0 if alias_type == "cv_head" else 10.0
        if onset_cls == "voiceless" or hard_cls:
            min_gap += 2.0
        elif onset_cls == "liquid":
            min_gap -= 1.0
        if vowel_len < 80.0:
            min_gap += 2.0
        min_gap = min(max(min_gap, 6.0), 20.0)

        ovl_cap = min(p * ratio_cap, max(p - min_gap, 0.0))
        if float(v) > ovl_cap:
            v = ovl_cap
        return validate_fn(o, c, cut, p, v)

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
            format_type=format_type,
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
        if format_type in {"cvvc", "cv"}:
            max_shift = 26.0
            if mapping_tier == "high":
                max_shift = 34.0
            onset_cls = onset_class_fn(c_char)
            if onset_cls == "liquid":
                max_shift += 5.0
            elif onset_cls in {"voiced", "nasal"}:
                max_shift += 3.0
            # Low-confidence CVVC sonorant bridges tend to drift into blank tails.
            # Keep pre-anchor movement tighter to preserve local syllable timing.
            if (
                str(format_type or "").strip().lower() == "cvvc"
                and str(mapping_tier or "").strip().lower() == "low"
                and onset_cls in {"nasal", "liquid"}
            ):
                max_shift = min(max_shift, 14.0)
            if (
                str(format_type or "").strip().lower() == "cvvc"
                and onset_cls in {"nasal", "liquid"}
            ):
                # Sonorants (n/m/r/l family) are easy to snap to a previous syllable.
                # Keep local pre-anchor movement tighter in CVVC to avoid 1-step misplacement.
                if str(mapping_tier or "").strip().lower() == "high":
                    max_shift = min(max_shift, 14.0)
                else:
                    max_shift = min(max_shift, 10.0)
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
        if format_type == "cvvc" and float(n_start) > 0.0:
            onset_cls = onset_class_fn(c_char)
            if onset_cls == "liquid":
                late_allow = 2.8
            elif onset_cls in {"voiced", "nasal"}:
                late_allow = 1.8
            else:
                late_allow = 2.0
            if onset_cls == "liquid":
                early_allow = 3.5
            elif onset_cls == "nasal":
                early_allow = 3.0
            else:
                early_allow = 5.0
            pre_abs_floor = float(n_start) - early_allow
            pre_abs_cap = float(n_start) + late_allow
            if pre_abs_after < pre_abs_floor:
                offset = max(float(offset) + (pre_abs_floor - pre_abs_after), 0.0)
                offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
                pre_abs_after = offset + pre
            if pre_abs_after > pre_abs_cap:
                offset = max(float(offset) - (pre_abs_after - pre_abs_cap), 0.0)
                offset, consonant, cutoff, pre, ovl = validate_fn(offset, consonant, cutoff, pre, ovl)
                pre_abs_after = offset + pre
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
        offset, consonant, cutoff, pre, ovl = _tighten_cv_overlap_for_cvvc(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
        )
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
