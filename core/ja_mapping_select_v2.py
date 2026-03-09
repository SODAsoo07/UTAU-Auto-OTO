from __future__ import annotations


def select_ja_vcv_mapping(
    *,
    alias,
    fname,
    format_type,
    stable_vcv_seq_idx,
    cv_seq_idx,
    syllables_info,
    ja_planned_cv_indices,
    last_vcv_mapped_idx,
    mapping_tier,
    mapping_reason_code,
    mapping_confidence_base,
    filename_order_locked,
    syllable_confidence_by_idx,
    log_fn,
    debug_logging,
    resolve_planned_cv_index_fn,
    select_vcv_syllable_index_fn,
    extract_target_syllable_fn,
    find_exact_target_index_fn,
    normalize_syllable_token_fn,
    syllable_info_token_fn,
    split_syllable_fn,
    ja_vowels,
    find_vowel_match_index_fn,
    prefer_vcv_candidate_index_fn,
    should_trace_mapping_decision_fn,
    build_mapping_trace_record_fn,
    append_mapping_trace_fn,
    decide_cv_row_abstain_fn,
    is_cv_syllable_active_fn,
):
    expected_seq_idx = stable_vcv_seq_idx if str(format_type or "").strip().lower() == "vcv" else cv_seq_idx
    if expected_seq_idx < len(syllables_info):
        expected_idx = expected_seq_idx
    else:
        expected_idx = len(syllables_info) - 1

    target_tok_vcv_raw = extract_target_syllable_fn(alias)
    planned_idx_vcv = resolve_planned_cv_index_fn(
        ja_planned_cv_indices,
        expected_seq_idx,
        target_tok_vcv_raw,
        syllables_info,
        alias_type="vcv",
    )
    if planned_idx_vcv is not None:
        mapped_idx = int(planned_idx_vcv)
        if debug_logging and mapped_idx != expected_idx:
            log_fn(
                f"🧭 {fname}: VCV 전역 anchor plan 적용 "
                f"{expected_idx + 1}->{mapped_idx + 1} ({alias})"
            )
    else:
        mapped_idx = int(select_vcv_syllable_index_fn(alias, expected_idx, syllables_info))

    resynced_vcv_exact = False
    resync_idx_vcv = find_exact_target_index_fn(
        target_tok_vcv_raw,
        expected_idx,
        syllables_info,
        search_back=(0 if str(format_type or "").strip().lower() == "vcv" else 3),
        search_fwd=3,
    )
    target_tok_vcv_norm_for_resync = normalize_syllable_token_fn(target_tok_vcv_raw)
    exp_tok_vcv_for_resync = normalize_syllable_token_fn(
        syllable_info_token_fn(syllables_info[expected_idx])
    )
    if (
        resync_idx_vcv is not None
        and resync_idx_vcv < expected_idx
        and exp_tok_vcv_for_resync == target_tok_vcv_norm_for_resync
    ):
        resync_idx_vcv = None
    if resync_idx_vcv is not None and resync_idx_vcv != mapped_idx:
        log_fn(
            f"🧭 {fname}: VCV 순서 드리프트 복구 "
            f"{expected_idx + 1}->{resync_idx_vcv + 1} ({alias})"
        )
        mapped_idx = int(resync_idx_vcv)
        resynced_vcv_exact = True

    target_tok_vcv_norm = normalize_syllable_token_fn(target_tok_vcv_raw)
    mapped_tok_vcv_now = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[mapped_idx]))
    exp_tok_vcv_now = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[expected_idx]))
    _t_on, target_vowel_vcv = split_syllable_fn(target_tok_vcv_norm)
    _m_on, mapped_vowel_vcv = split_syllable_fn(mapped_tok_vcv_now)
    _e_on, expected_vowel_vcv = split_syllable_fn(exp_tok_vcv_now)
    if target_vowel_vcv in ja_vowels and mapped_vowel_vcv and mapped_vowel_vcv != target_vowel_vcv:
        fixed_idx_vcv = find_vowel_match_index_fn(
            target_tok_vcv_norm,
            expected_idx,
            syllables_info,
            search_back=2,
            search_fwd=3,
        )
        if fixed_idx_vcv is not None and fixed_idx_vcv != mapped_idx:
            log_fn(
                f"🧭 {fname}: VCV 모음 불일치 보정 "
                f"{mapped_idx + 1}->{fixed_idx_vcv + 1} ({alias})"
            )
            mapped_idx = int(fixed_idx_vcv)
        elif expected_vowel_vcv == target_vowel_vcv:
            log_fn(
                f"🛡️ {fname}: VCV 모음 불일치 차단 "
                f"({mapped_idx + 1}->{expected_idx + 1}, {alias})"
            )
            mapped_idx = expected_idx

    if mapped_idx > expected_idx:
        target_tok_vcv = normalize_syllable_token_fn(target_tok_vcv_raw)
        exp_tok_vcv = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[expected_idx]))
        mapped_tok_vcv = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[mapped_idx]))
        if mapped_tok_vcv != target_tok_vcv or exp_tok_vcv == target_tok_vcv:
            mapped_idx = expected_idx
    if mapped_idx < expected_idx and (expected_idx - mapped_idx) > 3:
        mapped_idx = expected_idx
    if mapped_idx > (expected_idx + 1):
        if not (resynced_vcv_exact and mapped_idx <= (expected_idx + 2)):
            mapped_idx = expected_idx

    if last_vcv_mapped_idx >= 0:
        if mapped_idx < last_vcv_mapped_idx:
            log_fn(
                f"🛡️ {fname}: VCV 역행 점프 차단 "
                f"{mapped_idx + 1}->{last_vcv_mapped_idx + 1} ({alias})"
            )
            mapped_idx = last_vcv_mapped_idx
        elif mapped_idx > (last_vcv_mapped_idx + 2):
            clamped_idx = last_vcv_mapped_idx + 1
            log_fn(
                f"🛡️ {fname}: VCV 과도 점프 차단 "
                f"{mapped_idx + 1}->{clamped_idx + 1} ({alias})"
            )
            mapped_idx = clamped_idx

    mapped_tok_final = normalize_syllable_token_fn(
        syllable_info_token_fn(syllables_info[mapped_idx])
    )
    if target_tok_vcv_norm and mapped_tok_final != target_tok_vcv_norm:
        retry_idx_vcv = find_vowel_match_index_fn(
            target_tok_vcv_norm,
            expected_idx,
            syllables_info,
            search_back=4,
            search_fwd=4,
        )
        if (
            retry_idx_vcv is not None
            and retry_idx_vcv != mapped_idx
            and abs(retry_idx_vcv - expected_idx) <= 2
        ):
            log_fn(
                f"🧭 {fname}: VCV 최종 재탐색 "
                f"{mapped_idx + 1}->{retry_idx_vcv + 1} ({alias})"
            )
            mapped_idx = int(retry_idx_vcv)
            mapped_tok_final = normalize_syllable_token_fn(
                syllable_info_token_fn(syllables_info[mapped_idx])
            )
        if mapped_tok_final != target_tok_vcv_norm:
            log_fn(
                f"🛡️ {fname}: VCV 토큰 불일치 되돌림 "
                f"({mapped_idx + 1}->{expected_idx + 1}, {alias})"
            )
            mapped_idx = expected_idx

    if str(format_type or "").strip().lower() == "vcv":
        mapped_idx = int(
            prefer_vcv_candidate_index_fn(
                expected_idx,
                mapped_idx,
                target_tok_vcv_norm,
                syllables_info,
                max_delta=1,
            )
        )
    if mapped_idx != expected_idx and abs(mapped_idx - expected_idx) <= 1:
        log_fn(f"🧭 {fname}: VCV 음절 정렬 보정 {expected_idx + 1}->{mapped_idx + 1} ({alias})")

    expected_tok_trace = syllable_info_token_fn(syllables_info[expected_idx])
    mapped_tok_trace = syllable_info_token_fn(syllables_info[mapped_idx])
    local_trace_conf = None
    if syllable_confidence_by_idx:
        conf_idx = max(0, min(expected_idx, len(syllable_confidence_by_idx) - 1))
        local_trace_conf = float(syllable_confidence_by_idx[conf_idx])
    if should_trace_mapping_decision_fn(
        mapping_tier=mapping_tier,
        expected_idx=expected_idx,
        mapped_idx=mapped_idx,
        target_token=normalize_syllable_token_fn(target_tok_vcv_norm),
        mapped_token=normalize_syllable_token_fn(mapped_tok_trace),
    ):
        append_mapping_trace_fn(
            build_mapping_trace_record_fn(
                fname=fname,
                alias=alias,
                alias_type="vcv",
                format_type=format_type,
                target_tok=target_tok_vcv_norm,
                expected_idx=expected_idx,
                mapped_idx=mapped_idx,
                expected_tok=expected_tok_trace,
                mapped_tok=mapped_tok_trace,
                mapping_tier=mapping_tier,
                mapping_reason_code=mapping_reason_code,
                mapping_confidence=mapping_confidence_base,
                filename_order_locked=filename_order_locked,
                local_conf=local_trace_conf,
            )
        )

    row_abstain = decide_cv_row_abstain_fn(
        alias_type="cv",
        format_type=format_type,
        candidate_idx=mapped_idx,
        candidate_count=len(syllables_info),
        candidate_active=(
            is_cv_syllable_active_fn(syllables_info[mapped_idx], require_vowel=True)
            if 0 <= mapped_idx < len(syllables_info)
            else False
        ),
        active_only_formats={"cvvc", "cv"},
    )
    return {
        "expected_idx": int(expected_idx),
        "mapped_idx": int(mapped_idx),
        "target_tok": target_tok_vcv_norm,
        "row_abstain": row_abstain,
    }


def resolve_ja_forced_cv_index(
    *,
    alias,
    alias_type,
    target_tok,
    expected_seq_idx,
    expected_idx,
    syllables_info,
    planned_indices,
    occurrence_map,
    occurrence_state,
    resolve_planned_cv_index_fn,
    resolve_cvvc_occurrence_index_fn,
    remap_forced_cv_index_fn,
    log_fn,
    debug_logging,
    fname,
):
    planned_idx = resolve_planned_cv_index_fn(
        planned_indices,
        expected_seq_idx,
        target_tok,
        syllables_info,
        alias_type=alias_type,
    )
    forced_idx = planned_idx
    if forced_idx is None:
        forced_idx = resolve_cvvc_occurrence_index_fn(
            alias,
            alias_type,
            occurrence_map or {},
            occurrence_state,
        )
    elif debug_logging and forced_idx != expected_idx:
        alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
        log_fn(
            f"🧭 {fname}: {alias_label} 전역 anchor plan 적용 "
            f"{expected_idx + 1}->{int(forced_idx) + 1} ({alias})"
        )
    if forced_idx is not None and not (0 <= int(forced_idx) < len(syllables_info)):
        remapped_idx = remap_forced_cv_index_fn(target_tok, expected_idx, syllables_info)
        if remapped_idx is not None:
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"🧭 {fname}: {alias_label} occurrence 범위 보정 "
                    f"({int(forced_idx) + 1}->{int(remapped_idx) + 1}, {alias})"
                )
            forced_idx = int(remapped_idx)
        else:
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"🛡️ {fname}: {alias_label} occurrence 무효화 "
                    f"(idx={int(forced_idx) + 1}, {alias})"
                )
            forced_idx = None
    return forced_idx


__all__ = ["resolve_ja_forced_cv_index", "select_ja_vcv_mapping"]
