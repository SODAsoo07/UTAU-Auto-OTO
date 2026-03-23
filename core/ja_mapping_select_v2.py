from __future__ import annotations

from core.ja_oto_mapping import _ja_soft_cv_match_level


def _blank_conf_at(syllables_info, idx):
    if not syllables_info or idx is None:
        return 0.0
    try:
        if idx < 0 or idx >= len(syllables_info):
            return 0.0
        row = syllables_info[idx] or {}
        return max(0.0, min(1.0, float(row.get("blank_confidence", 0.0) or 0.0)))
    except Exception:
        return 0.0


def _mel_conf_at(syllables_info, idx, key, fallback=0.0):
    if not syllables_info or idx is None:
        return float(fallback)
    try:
        if idx < 0 or idx >= len(syllables_info):
            return float(fallback)
        row = syllables_info[idx] or {}
        return max(0.0, min(1.0, float(row.get(key, fallback) or fallback)))
    except Exception:
        return float(fallback)


def _mel_guided_ja_cvvc_adjustment(
    *,
    format_type,
    alias_type,
    target_tok,
    expected_idx,
    selected_idx,
    mapping_confidence,
    mapping_conf_threshold,
    max_search_fwd,
    syllables_info,
    normalize_syllable_token_fn,
    syllable_info_token_fn,
    syllable_confidence_by_idx=None,
    order_locked=False,
    mel_remap_enabled=True,
):
    fmt = str(format_type or "").strip().lower()
    if fmt != "cvvc" or not mel_remap_enabled:
        return selected_idx, False
    a_type = str(alias_type or "").strip().lower()
    if a_type not in {"cv", "cv_head", "vcv"}:
        return selected_idx, False
    if selected_idx is None or not syllables_info:
        return selected_idx, False
    if selected_idx < 0 or selected_idx >= len(syllables_info):
        return selected_idx, False
    if expected_idx < 0:
        expected_idx = 0
    if expected_idx >= len(syllables_info):
        expected_idx = len(syllables_info) - 1

    target_norm = normalize_syllable_token_fn(target_tok)
    if not target_norm:
        return selected_idx, False

    local_conf = None
    if syllable_confidence_by_idx:
        try:
            conf_idx = max(0, min(int(expected_idx), len(syllable_confidence_by_idx) - 1))
            local_conf = float(syllable_confidence_by_idx[conf_idx])
        except Exception:
            local_conf = None
    conf = float(local_conf) if local_conf is not None else float(mapping_confidence or 0.0)
    conf_th = float(mapping_conf_threshold or 0.0)
    selected_blank = _blank_conf_at(syllables_info, selected_idx)
    should_try = bool(conf < max(conf_th + 0.08, 0.62) or selected_blank >= 0.58)
    if not should_try:
        return selected_idx, False

    mel_hint = False
    mel_keys = (
        "mel_voiced_formant_conf",
        "mel_unvoiced_diffuse_conf",
        "mel_silence_sparse_conf",
        "mel_breath_like_conf",
    )
    for check_idx in (selected_idx, expected_idx):
        if check_idx is None:
            continue
        if check_idx < 0 or check_idx >= len(syllables_info):
            continue
        row = syllables_info[check_idx] or {}
        for key in mel_keys:
            try:
                val = float(row.get(key, 0.0) or 0.0)
            except Exception:
                val = 0.0
            if val > 0.0:
                mel_hint = True
                break
        if mel_hint:
            break
    if not mel_hint:
        if _blank_conf_at(syllables_info, selected_idx) > 0.0:
            mel_hint = True
        elif _blank_conf_at(syllables_info, expected_idx) > 0.0:
            mel_hint = True
    if not mel_hint:
        return selected_idx, False

    n = len(syllables_info)
    lo = max(0, expected_idx - 1)
    hi = min(n - 1, expected_idx + int(max(1, max_search_fwd)))
    if order_locked:
        lo = expected_idx
        hi = min(hi, expected_idx + 1)
    if lo >= hi:
        return selected_idx, False

    def _score(idx):
        cand_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[idx]))
        soft = int(_ja_soft_cv_match_level(target_norm, cand_tok) or 0) if cand_tok else 0
        text = (soft * 40.0) + (40.0 if cand_tok == target_norm else 0.0)
        blank = _blank_conf_at(syllables_info, idx)
        sil = _mel_conf_at(
            syllables_info,
            idx,
            "mel_silence_sparse_conf",
            fallback=_mel_conf_at(syllables_info, idx, "blank_confidence", 0.0),
        )
        voiced = _mel_conf_at(syllables_info, idx, "mel_voiced_formant_conf", 0.0)
        unvoiced = _mel_conf_at(syllables_info, idx, "mel_unvoiced_diffuse_conf", 0.0)
        breath = _mel_conf_at(syllables_info, idx, "mel_breath_like_conf", 0.0)
        jump_penalty = abs(idx - expected_idx) * (10.0 if a_type in {"cv", "cv_head"} else 8.0)
        mel_bonus = (9.0 * voiced) + (5.0 * unvoiced)
        mel_penalty = (20.0 * blank) + (12.0 * sil) + (6.0 * breath)
        mismatch_penalty = 0.0
        if soft <= 0:
            mismatch_penalty += 46.0 if a_type in {"cv", "cv_head"} else 34.0
        elif soft == 1:
            mismatch_penalty += 12.0 if a_type in {"cv", "cv_head"} else 6.0
        stability_bonus = 1.5 if idx == selected_idx else 0.0
        return text + mel_bonus + stability_bonus - mel_penalty - jump_penalty - mismatch_penalty

    best_idx = int(selected_idx)
    best_score = float(_score(best_idx))
    for idx in range(lo, hi + 1):
        cand_score = float(_score(idx))
        if cand_score > best_score:
            best_idx = int(idx)
            best_score = cand_score

    if best_idx == int(selected_idx):
        return selected_idx, False

    selected_score = float(_score(int(selected_idx)))
    min_gain = 6.0 if a_type in {"cv", "cv_head"} else 5.0
    if best_score < (selected_score + min_gain):
        return selected_idx, False
    selected_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[int(selected_idx)]))
    best_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[int(best_idx)]))
    selected_soft = int(_ja_soft_cv_match_level(target_norm, selected_tok) or 0) if selected_tok else 0
    best_soft = int(_ja_soft_cv_match_level(target_norm, best_tok) or 0) if best_tok else 0
    if best_soft <= 0:
        return selected_idx, False
    if selected_soft >= 2 and best_soft < selected_soft:
        return selected_idx, False
    if a_type in {"cv", "cv_head"} and best_soft < 2 and best_score < (selected_score + 12.0):
        return selected_idx, False
    expected_blank = _blank_conf_at(syllables_info, expected_idx)
    if _blank_conf_at(syllables_info, best_idx) >= 0.72 and expected_blank <= 0.60:
        return selected_idx, False
    return best_idx, True

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
    mapping_conf_threshold=None,
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
    mel_remap_enabled=True,
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
        search_back=(0 if str(format_type or "").strip().lower() == "vcv" else (1 if mapping_tier == "low" else 3)),
        search_fwd=(1 if mapping_tier == "low" else 3),
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
            search_back=(1 if mapping_tier == "low" else 2),
            search_fwd=(2 if mapping_tier == "low" else 3),
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

    mel_idx, mel_adjusted = _mel_guided_ja_cvvc_adjustment(
        format_type=format_type,
        alias_type="vcv",
        target_tok=target_tok_vcv_norm,
        expected_idx=expected_idx,
        selected_idx=mapped_idx,
        mapping_confidence=mapping_confidence_base,
        mapping_conf_threshold=mapping_conf_threshold,
        max_search_fwd=(2 if str(mapping_tier or "").strip().lower() == "low" else 3),
        syllables_info=syllables_info,
        normalize_syllable_token_fn=normalize_syllable_token_fn,
        syllable_info_token_fn=syllable_info_token_fn,
        syllable_confidence_by_idx=syllable_confidence_by_idx,
        order_locked=bool(filename_order_locked),
        mel_remap_enabled=bool(mel_remap_enabled),
    )
    if mel_adjusted and mel_idx != mapped_idx:
        if debug_logging:
            log_fn(
                f"🧭 {fname}: VCV mel-guided remap "
                f"({mapped_idx + 1}->{mel_idx + 1}, {alias})"
            )
        mapped_idx = int(mel_idx)

    mapped_tok_final = normalize_syllable_token_fn(
        syllable_info_token_fn(syllables_info[mapped_idx])
    )
    if target_tok_vcv_norm and mapped_tok_final != target_tok_vcv_norm:
        retry_idx_vcv = find_vowel_match_index_fn(
            target_tok_vcv_norm,
            expected_idx,
            syllables_info,
            search_back=(2 if mapping_tier == "low" else 4),
            search_fwd=(2 if mapping_tier == "low" else 4),
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
