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
    voiced_onset_ms=None,
    mel_ctx_for_file=None,
    alignment_weight=0.0,
    textgrid_trust_tier="",
    bridge_tuning=None,
):
    def _clamp_cv_window(
        _offset: float,
        _consonant: float,
        _cutoff: float,
        _pre: float,
        _ovl: float,
    ):
        # Keep CV offset around current consonant onset to avoid VC-like over-early mapping.
        c_len = max(float(c_end) - float(c_start), 0.0)
        lead_cap = max(8.0, min(16.0, c_len * 0.42 + 5.0))
        off_min = max(0.0, float(c_start) - lead_cap)
        off_max = float(c_start) + 3.0

        o = float(_offset)
        c = float(_consonant)
        cut_abs = abs(float(_cutoff))
        p = float(_pre)
        v = float(_ovl)

        if o < off_min - 0.5:
            shift = off_min - o
            o = off_min
            p = max(0.0, p - shift)
            v = max(0.0, v - shift)
            c = max(p + 6.0, c - shift)
            cut_abs = max(c + 8.0, cut_abs - shift)

        if o > off_max + 0.5:
            shift = o - off_max
            o = off_max
            p = max(0.0, p + shift)
            v = max(0.0, v + shift)
            c = max(p + 6.0, c + shift)
            cut_abs = max(c + 8.0, cut_abs + shift)

        return validate_fn(o, c, -cut_abs, p, v)

    def _clamp_vc_window(
        _offset: float,
        _consonant: float,
        _cutoff: float,
        _pre: float,
        _ovl: float,
    ):
        # VC should bridge into next consonant and end near next-consonant tail.
        next_on = float(n_start)
        next_end = max(float(n_end), next_on + 8.0)
        if next_on <= 0.0:
            return _offset, _consonant, _cutoff, _pre, _ovl
        try:
            from core.kr_oto_rules import _canonicalize_kr_coda, _extract_vc_right_token

            coda = _canonicalize_kr_coda(_extract_vc_right_token(alias))
        except Exception:
            coda = ""
        is_stop = coda in {"k", "t", "p", "h"}
        is_sonorant = coda in {"n", "m", "ng", "l", "r"}
        bridge_source = str((bridge_pair or {}).get("source") or "").strip().lower()
        strict_bridge = bridge_source in {"token_invariant", "cv_confirmed"}

        o = float(_offset)
        p = float(_pre)
        ovl_abs = o + max(float(_ovl), 0.0)
        c_abs = o + float(_consonant)
        cut_abs = o + abs(float(_cutoff))

        # Keep preceding vowel include around last 1/3~2/5 when we have reliable CV anchors.
        if strict_bridge and isinstance(prev_anchor, dict):
            try:
                prev_v_start = float(prev_anchor.get("vowel_start_abs", 0.0) or 0.0)
                prev_v_end = float(prev_anchor.get("vowel_end_abs", 0.0) or 0.0)
                prev_v_len = max(prev_v_end - prev_v_start, 30.0)
                off_lo = max(0.0, prev_v_end - (prev_v_len * 0.40))
                off_hi = max(off_lo, prev_v_end - (prev_v_len / 3.0))
                if o < off_lo - 0.5:
                    shift = off_lo - o
                    o = off_lo
                    p = max(0.0, p - shift)
                    ovl_abs = max(o, ovl_abs - shift)
                    c_abs = max(o + p + 8.0, c_abs - shift)
                    cut_abs = max(c_abs + 8.0, cut_abs - shift)
                elif o > off_hi + 0.5:
                    shift = o - off_hi
                    o = off_hi
                    p = max(0.0, p + shift)
                    ovl_abs = ovl_abs + shift
                    c_abs = max(o + p + 8.0, c_abs + shift)
                    cut_abs = max(c_abs + 8.0, cut_abs + shift)
            except Exception:
                pass

        next_pre_abs = next_on
        next_pre = 0.0
        if isinstance(next_anchor, dict):
            try:
                next_pre_abs = float(next_anchor.get("pre_abs", next_on) or next_on)
                next_pre = max(float(next_anchor.get("pre", 0.0) or 0.0), 0.0)
            except Exception:
                next_pre_abs = next_on
                next_pre = 0.0
        next_offset_abs = next_pre_abs - next_pre
        if next_offset_abs <= 0.0:
            next_offset_abs = next_on

        pre_abs = o + p
        if strict_bridge and is_stop:
            pre_abs = min(max(pre_abs, next_offset_abs - 4.0), next_offset_abs + 6.0)
        elif strict_bridge and is_sonorant:
            pre_abs = min(max(pre_abs, next_offset_abs - 3.0), next_offset_abs + 10.0)
        elif strict_bridge:
            pre_abs = min(max(pre_abs, next_offset_abs - 4.0), next_offset_abs + 8.0)
        elif is_stop:
            pre_abs = min(max(pre_abs, next_offset_abs - 7.0), next_offset_abs + 8.0)
        elif is_sonorant:
            pre_abs = min(max(pre_abs, next_offset_abs - 5.0), next_offset_abs + 14.0)
        else:
            pre_abs = min(max(pre_abs, next_offset_abs - 6.0), next_offset_abs + 10.0)
        o = max(pre_abs - p, 0.0)
        c_abs = o + float(_consonant)
        cut_abs = o + abs(float(_cutoff))
        ovl = max(0.0, min(p, ovl_abs - o))

        if is_stop:
            # Stop-like VC should end before next onset to avoid double release.
            c_floor = max(p + 8.0 + o, next_on - 12.0)
            c_cap = next_on - 2.0
        elif is_sonorant:
            c_floor = max(next_on + 4.0, p + 14.0 + o)
            c_cap = next_end - (4.0 if strict_bridge else 3.0)
        else:
            c_floor = max(next_on + 2.0, p + 12.0 + o)
            c_cap = next_end - (5.0 if strict_bridge else 4.0)
        if c_cap <= c_floor:
            c_cap = c_floor + 2.0
        c_abs = min(max(c_abs, c_floor), c_cap)

        if is_stop:
            cut_floor = max(c_abs + 4.0, next_on - 2.2)
            cut_cap = next_on - (0.9 if strict_bridge else 0.7)
        elif is_sonorant:
            cut_floor = max(c_abs + 10.0, next_end - 4.0)
            cut_cap = next_end + (4.0 if strict_bridge else 6.0)
        else:
            cut_floor = max(c_abs + 8.0, next_end - 3.0)
            cut_cap = next_end + (2.5 if strict_bridge else 4.0)
        if cut_cap <= cut_floor:
            cut_cap = cut_floor + 1.0
        cut_abs = min(max(cut_abs, cut_floor), cut_cap)

        return validate_fn(o, c_abs - o, -(cut_abs - o), p, ovl)

    bridge_shift = 0.0
    if alias_type in {"vc", "vv"}:
        mel_cutoff_candidate_ms = None
        next_mel_voiced_onset_ms = None
        if mel_ctx_for_file:
            from core.oto_generator import _estimate_mel_cutoff_candidate

            pre_abs = float(offset + pre)
            cut_abs = float(offset + abs(float(cutoff)))
            mel_cutoff_candidate_ms = _estimate_mel_cutoff_candidate(mel_ctx_for_file, pre_abs, cut_abs)
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
        if next_anchor is not None:
            try:
                next_mel_voiced_onset_ms = float(next_anchor.get("mel_voiced_onset_abs", 0.0) or 0.0) or None
            except Exception:
                next_mel_voiced_onset_ms = None
        if prev_anchor is not None and next_anchor is not None:
            pre_abs_before = float(offset + pre)
            try:
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
                    format_type=file_format,
                    mel_cutoff_candidate_ms=mel_cutoff_candidate_ms,
                    next_mel_voiced_onset_ms=next_mel_voiced_onset_ms,
                    bridge_tuning=bridge_tuning,
                )
            except TypeError as exc:
                if "bridge_tuning" not in str(exc):
                    raise
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
                    format_type=file_format,
                    mel_cutoff_candidate_ms=mel_cutoff_candidate_ms,
                    next_mel_voiced_onset_ms=next_mel_voiced_onset_ms,
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
        voiced_onset_ms=voiced_onset_ms,
    )
    if alias_type == "cv":
        offset, consonant, cutoff, pre, ovl = _clamp_cv_window(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
        )
    elif alias_type == "vc":
        offset, consonant, cutoff, pre, ovl = _clamp_vc_window(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
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
            mel_voiced_onset_abs=voiced_onset_ms,
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
