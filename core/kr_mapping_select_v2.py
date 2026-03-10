from __future__ import annotations


def select_kr_vcv_index(
    *,
    target_clean,
    cv_seq_idx,
    current_w_idx,
    romaji_syllables,
    syllables_info,
    file_format,
    row_mapping_confidence,
    row_jump_default,
    row_jump_high_conf,
    file_mapping_conf_th,
    kr_planned_cv_indices,
    resolve_planned_cv_index_fn,
    resolve_cv_syllable_index_fn,
    clamp_cv_index_to_order_fn,
    split_syllable_parts_fn,
    find_cv_vowel_match_index_fn,
    cv_match_score_fn,
    apply_row_confidence_penalty_fn,
    log_fn,
    debug_logging,
    fname,
    alias,
):
    vcv_selected_w_idx = current_w_idx
    if target_clean and cv_seq_idx < len(romaji_syllables):
        expected_vcv_idx = cv_seq_idx
        planned_vcv_idx = resolve_planned_cv_index_fn(
            kr_planned_cv_indices,
            expected_vcv_idx,
            target_clean,
            syllables_info,
            alias_type="vcv",
        )
        if planned_vcv_idx is not None:
            vcv_selected_w_idx = int(planned_vcv_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            vcv_meta = {
                "jump_blocked": 0,
                "raw_chosen_idx": int(vcv_selected_w_idx),
                "chosen_idx": int(vcv_selected_w_idx),
                "best_score": float(cv_match_score_fn(target_clean, romaji_syllables[vcv_selected_w_idx])),
                "expected_score": float(cv_match_score_fn(target_clean, romaji_syllables[expected_vcv_idx]))
                if 0 <= expected_vcv_idx < len(romaji_syllables)
                else -1.0,
            }
            if debug_logging and vcv_selected_w_idx != expected_vcv_idx:
                log_fn(
                    f"🧭 {fname}: KR VCV 전역 anchor plan 적용 "
                    f"({expected_vcv_idx + 1}->{vcv_selected_w_idx + 1}, {alias})"
                )
        else:
            vcv_selected_w_idx, cv_seq_idx, vcv_meta = resolve_cv_syllable_index_fn(
                target_clean,
                romaji_syllables,
                cv_seq_idx,
                current_w_idx,
                mapping_confidence=row_mapping_confidence,
                max_jump_default=row_jump_default,
                max_jump_high_conf=row_jump_high_conf,
                high_conf_threshold=max(float(file_mapping_conf_th), 0.50),
                return_meta=True,
            )
        vcv_jump_blocked = int(vcv_meta.get("jump_blocked", 0) or 0)
        if vcv_jump_blocked:
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.12)
            if debug_logging:
                log_fn(
                    f"🛡️ {fname}: KR VCV 매핑 전진 점프 차단 "
                    f"({int(vcv_meta.get('raw_chosen_idx', vcv_selected_w_idx)) + 1}"
                    f"->{int(vcv_meta.get('chosen_idx', vcv_selected_w_idx)) + 1}, {alias})"
                )
        ordered_vcv_idx = clamp_cv_index_to_order_fn(
            file_format,
            target_clean,
            romaji_syllables,
            expected_vcv_idx,
            vcv_selected_w_idx,
        )
        if ordered_vcv_idx != vcv_selected_w_idx:
            vcv_selected_w_idx = ordered_vcv_idx
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.10)
            if debug_logging:
                log_fn(
                    f"🛡️ {fname}: KR VCV 순서 고정 "
                    f"({expected_vcv_idx + 1}->{ordered_vcv_idx + 1}, {alias})"
                )

        t_onset, t_vowel, _t_coda = split_syllable_parts_fn(target_clean)
        if t_vowel and 0 <= vcv_selected_w_idx < len(romaji_syllables):
            _c_onset, c_vowel, _c_coda = split_syllable_parts_fn(romaji_syllables[vcv_selected_w_idx])
            if c_vowel and c_vowel != t_vowel:
                fixed_idx = find_cv_vowel_match_index_fn(
                    target_clean,
                    romaji_syllables,
                    expected_vcv_idx,
                    search_back=1,
                    search_fwd=2,
                )
                if fixed_idx is not None and fixed_idx >= expected_vcv_idx:
                    max_forward = int(max(0, row_jump_default))
                    max_allowed = int(expected_vcv_idx + max_forward)
                    if fixed_idx > max_allowed:
                        fixed_idx = max_allowed
                    if fixed_idx != vcv_selected_w_idx:
                        if debug_logging:
                            log_fn(
                                f"🧭 {fname}: KR VCV 모음 불일치 보정 "
                                f"{vcv_selected_w_idx + 1}->{fixed_idx + 1} ({alias})"
                            )
                        vcv_selected_w_idx = fixed_idx
                        cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
    return vcv_selected_w_idx, cv_seq_idx, row_mapping_confidence


def select_kr_general_cv_index(
    *,
    alias,
    alias_type,
    fname,
    file_format,
    target_clean,
    current_w_idx,
    cv_seq_idx,
    row_mapping_confidence,
    row_jump_default,
    row_jump_high_conf,
    file_mapping_conf_th,
    file_mapping_low_conf,
    romaji_syllables,
    forced_vv_idx,
    planned_vv_idx,
    planned_cv_idx,
    forced_cvvc_idx,
    remap_forced_cv_index_fn,
    cv_match_score_fn,
    split_syllable_parts_fn,
    apply_row_confidence_penalty_fn,
    resolve_cv_syllable_index_fn,
    should_allow_exact_vowel_fix_fn,
    find_cv_vowel_match_index_fn,
    clamp_cv_index_to_order_fn,
    log_fn,
    debug_logging,
):
    expected_cv_idx = int(cv_seq_idx)
    selected_w_idx = None
    resolve_meta = {}
    row_jump_blocked = 0

    forced_selected_idx = (
        forced_vv_idx
        if forced_vv_idx is not None
        else (
            planned_vv_idx
            if planned_vv_idx is not None
            else (planned_cv_idx if planned_cv_idx is not None else forced_cvvc_idx)
        )
    )
    forced_gate_rejected = False
    if forced_selected_idx is not None and not (0 <= forced_selected_idx < len(romaji_syllables)):
        remapped_idx = remap_forced_cv_index_fn(
            target_clean,
            romaji_syllables,
            expected_cv_idx,
        )
        if remapped_idx is not None:
            if debug_logging:
                log_fn(
                    f"🧭 {fname}: KR 강제 인덱스 범위 보정 "
                    f"({forced_selected_idx + 1}->{remapped_idx + 1}, {alias})"
                )
            forced_selected_idx = remapped_idx
        else:
            if debug_logging:
                log_fn(
                    f"🛡️ {fname}: KR 강제 인덱스 무효화 "
                    f"(idx={forced_selected_idx + 1}, {alias})"
                )
            forced_selected_idx = None
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.12)
    if forced_selected_idx is not None:
        forced_score = -1.0
        expected_score_forced = -1.0
        forced_vowel_mismatch = False
        if target_clean and 0 <= forced_selected_idx < len(romaji_syllables):
            forced_score = float(cv_match_score_fn(target_clean, romaji_syllables[forced_selected_idx]))
            if 0 <= expected_cv_idx < len(romaji_syllables):
                expected_score_forced = float(cv_match_score_fn(target_clean, romaji_syllables[expected_cv_idx]))
            _t_on, _t_v, _t_c = split_syllable_parts_fn(target_clean)
            _f_on, _f_v, _f_c = split_syllable_parts_fn(romaji_syllables[forced_selected_idx])
            forced_vowel_mismatch = bool(_t_v and _f_v and _t_v != _f_v)
        min_forced_score = 64.0 if alias_type == "cv_head" else 60.0
        if file_mapping_low_conf:
            min_forced_score += 6.0
        keep_forced = True
        if forced_score >= 0.0 and forced_score < min_forced_score:
            keep_forced = False
        if forced_vowel_mismatch and file_mapping_low_conf:
            keep_forced = False
        if expected_score_forced >= 0.0 and forced_score >= 0.0 and (forced_score + 14.0) < expected_score_forced:
            keep_forced = False
        if keep_forced:
            selected_w_idx = int(forced_selected_idx)
            cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
        else:
            forced_gate_rejected = True
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
    if forced_selected_idx is None or forced_gate_rejected:
        selected_w_idx, cv_seq_idx, resolve_meta = resolve_cv_syllable_index_fn(
            target_clean,
            romaji_syllables,
            cv_seq_idx,
            current_w_idx,
            mapping_confidence=row_mapping_confidence,
            max_jump_default=row_jump_default,
            max_jump_high_conf=row_jump_high_conf,
            high_conf_threshold=max(float(file_mapping_conf_th), 0.50),
            return_meta=True,
        )
        row_jump_blocked = int(resolve_meta.get("jump_blocked", 0) or 0)
        if row_jump_blocked and debug_logging:
            log_fn(
                f"🛡️ {fname}: KR 매핑 전진 점프 차단 "
                f"({int(resolve_meta.get('raw_chosen_idx', selected_w_idx)) + 1}"
                f"->{int(resolve_meta.get('chosen_idx', selected_w_idx)) + 1}, {alias})"
            )
        if row_jump_blocked:
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.18)
    target_onset, target_vowel, _target_coda = split_syllable_parts_fn(target_clean)
    forced_cv_head_severe_mismatch = False
    if (
        forced_selected_idx is not None
        and alias_type == "cv_head"
        and target_vowel
        and selected_w_idx is not None
        and 0 <= selected_w_idx < len(romaji_syllables)
    ):
        _fc_on, forced_curr_vowel, _fc_coda = split_syllable_parts_fn(romaji_syllables[selected_w_idx])
        forced_cv_head_severe_mismatch = bool(forced_curr_vowel and forced_curr_vowel != target_vowel)
    allow_exact_vowel_fix = (
        should_allow_exact_vowel_fix_fn(
            file_format,
            forced_selected_idx,
            alias_type=alias_type,
            severe_vowel_mismatch=forced_cv_head_severe_mismatch,
        )
        and float(row_mapping_confidence) >= float(file_mapping_conf_th)
    )
    if target_vowel and selected_w_idx is not None and 0 <= selected_w_idx < len(romaji_syllables) and allow_exact_vowel_fix:
        _curr_onset, curr_vowel, _curr_coda = split_syllable_parts_fn(romaji_syllables[selected_w_idx])
        need_exact_vowel_fix = bool(curr_vowel and curr_vowel != target_vowel)
        if file_format == "cvvc" and forced_selected_idx is None and not need_exact_vowel_fix:
            fixed_probe = find_cv_vowel_match_index_fn(
                target_clean,
                romaji_syllables,
                expected_cv_idx,
                search_back=1,
                search_fwd=4,
            )
            if fixed_probe is not None and fixed_probe != selected_w_idx:
                _probe_onset, probe_vowel, _probe_coda = split_syllable_parts_fn(romaji_syllables[fixed_probe])
                if probe_vowel == target_vowel:
                    need_exact_vowel_fix = True
        if need_exact_vowel_fix:
            fixed_search_fwd = 4 if file_format == "cvvc" else 2
            if file_format == "cvvc" and forced_selected_idx is not None and alias_type == "cv_head":
                fixed_search_fwd = 1
            fixed_idx = find_cv_vowel_match_index_fn(
                target_clean,
                romaji_syllables,
                expected_cv_idx,
                search_back=1,
                search_fwd=fixed_search_fwd,
            )
            if fixed_idx is not None and fixed_idx >= expected_cv_idx:
                max_forward = int(max(0, row_jump_default))
                if (
                    float(row_mapping_confidence) >= max(float(file_mapping_conf_th), 0.50)
                    and float(resolve_meta.get("best_score", 0.0) or 0.0) >= 84.0
                ):
                    max_forward = int(max(max_forward, row_jump_high_conf))
                raw_fixed_idx = int(fixed_idx)
                max_allowed_idx = int(expected_cv_idx + max_forward)
                if fixed_idx > max_allowed_idx:
                    fixed_idx = max_allowed_idx
                    row_jump_blocked = 1
                    row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.14)
                    if debug_logging:
                        log_fn(
                            f"🛡️ {fname}: KR 모음 보정 점프 차단 "
                            f"({raw_fixed_idx + 1}->{fixed_idx + 1}, {alias})"
                        )
                if fixed_idx != selected_w_idx and abs(fixed_idx - expected_cv_idx) <= 2:
                    log_fn(
                        f"🧭 {fname}: CV 모음 불일치 보정 "
                        f"{expected_cv_idx + 1}->{fixed_idx + 1} ({alias})"
                    )
                selected_w_idx = int(fixed_idx)
                cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    ordered_idx = clamp_cv_index_to_order_fn(
        file_format,
        target_clean,
        romaji_syllables,
        expected_cv_idx,
        selected_w_idx,
    )
    if ordered_idx != selected_w_idx:
        if debug_logging:
            log_fn(
                f"🛡️ {fname}: KR CV 순서 고정 "
                f"({selected_w_idx + 1}->{ordered_idx + 1}, {alias})"
            )
        row_jump_blocked = 1
        row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.12)
        selected_w_idx = int(ordered_idx)
        cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    return {
        "expected_cv_idx": int(expected_cv_idx),
        "selected_w_idx": int(selected_w_idx) if selected_w_idx is not None else None,
        "cv_seq_idx": int(cv_seq_idx),
        "row_mapping_confidence": float(row_mapping_confidence),
        "resolve_meta": dict(resolve_meta),
        "row_jump_blocked": int(row_jump_blocked),
        "forced_selected_idx": forced_selected_idx,
    }


def resolve_kr_cv_head_forced_index(
    *,
    alias,
    alias_type,
    cv_seq_idx,
    target_clean,
    romaji_syllables,
    syllables_info,
    kr_planned_cv_indices,
    kr_cvvc_occurrence_map,
    kr_cvvc_occurrence_state,
    resolve_planned_cv_index_fn,
    resolve_cvvc_occurrence_index_fn,
    remap_forced_cv_index_fn,
    log_fn,
    debug_logging,
    fname,
):
    planned_cv_head_idx = resolve_planned_cv_index_fn(
        kr_planned_cv_indices,
        cv_seq_idx,
        target_clean,
        syllables_info,
        alias_type="cv_head",
    )
    forced_cvvc_idx = planned_cv_head_idx
    if forced_cvvc_idx is None:
        forced_cvvc_idx = resolve_cvvc_occurrence_index_fn(
            alias,
            alias_type,
            kr_cvvc_occurrence_map or {},
            kr_cvvc_occurrence_state,
        )
    elif debug_logging and forced_cvvc_idx != cv_seq_idx:
        log_fn(
            f"🧭 {fname}: KR CV_HEAD 전역 anchor plan 적용 "
            f"({cv_seq_idx + 1}->{forced_cvvc_idx + 1}, {alias})"
        )
    if forced_cvvc_idx is not None and not (0 <= forced_cvvc_idx < len(romaji_syllables)):
        remapped_idx = remap_forced_cv_index_fn(
            target_clean,
            romaji_syllables,
            cv_seq_idx,
        )
        if remapped_idx is not None:
            if debug_logging:
                log_fn(
                    f"🧭 {fname}: KR CV_HEAD 강제 인덱스 범위 보정 "
                    f"({forced_cvvc_idx + 1}->{remapped_idx + 1}, {alias})"
                )
            forced_cvvc_idx = remapped_idx
        else:
            if debug_logging:
                log_fn(
                    f"🛡️ {fname}: KR CV_HEAD 강제 인덱스 무효화 "
                    f"(idx={forced_cvvc_idx + 1}, {alias})"
                )
            forced_cvvc_idx = None
    return forced_cvvc_idx


__all__ = [
    "resolve_kr_cv_head_forced_index",
    "select_kr_general_cv_index",
    "select_kr_vcv_index",
]
