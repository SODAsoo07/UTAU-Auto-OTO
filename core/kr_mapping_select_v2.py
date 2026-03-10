from __future__ import annotations

import re

from core.kr_oto_rules import _is_kr_glide_vowel


def _extract_vv_pair_tokens(alias, split_syllable_parts_fn):
    parts = [p for p in re.split(r"\s+", (alias or "").strip().lower()) if p]
    if len(parts) != 2:
        return "", ""
    left = re.sub(r"[^a-z]", "", parts[0])
    right = re.sub(r"[^a-z]", "", parts[1])
    lo, lv, lc = split_syllable_parts_fn(left)
    ro, rv, rc = split_syllable_parts_fn(right)
    if not lv or not rv:
        return "", ""
    if lo or ro or lc or rc:
        return "", ""
    return lv, rv


def _vv_pair_matches(idx, romaji_syllables, split_syllable_parts_fn, left_vowel, right_vowel):
    if idx is None or not romaji_syllables:
        return False
    if idx <= 0 or idx >= len(romaji_syllables):
        return False
    lo, lv, lc = split_syllable_parts_fn(romaji_syllables[idx - 1])
    ro, rv, rc = split_syllable_parts_fn(romaji_syllables[idx])
    if lo or ro or lc or rc:
        return False
    return bool(lv and rv and lv == left_vowel and rv == right_vowel)


def _blank_conf_at(syllable_blank_confidences, idx):
    if syllable_blank_confidences is None or idx is None:
        return 0.0
    try:
        if idx < 0 or idx >= len(syllable_blank_confidences):
            return 0.0
        return max(0.0, min(1.0, float(syllable_blank_confidences[idx] or 0.0)))
    except Exception:
        return 0.0


def _effective_jump_limits(row_jump_default, row_jump_high_conf, expected_blank_conf):
    jump_default = int(max(0, row_jump_default))
    jump_high_conf = int(max(jump_default, row_jump_high_conf))
    blank = max(0.0, min(1.0, float(expected_blank_conf)))
    if blank >= 0.70:
        return 0, 0
    if blank >= 0.45:
        return min(jump_default, 1), min(jump_high_conf, 1)
    return jump_default, jump_high_conf


def _blank_guard_idx(expected_idx, selected_idx, syllable_blank_confidences):
    if selected_idx is None:
        return selected_idx, False
    sel_blank = _blank_conf_at(syllable_blank_confidences, selected_idx)
    exp_blank = _blank_conf_at(syllable_blank_confidences, expected_idx)
    if selected_idx > expected_idx and sel_blank >= 0.60 and (exp_blank + 0.10) < sel_blank:
        return expected_idx, True
    if sel_blank >= 0.78 and exp_blank <= 0.62:
        return expected_idx, True
    return selected_idx, False


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
    syllable_blank_confidences=None,
):
    vcv_selected_w_idx = current_w_idx
    if target_clean and cv_seq_idx < len(romaji_syllables):
        expected_vcv_idx = cv_seq_idx
        expected_blank_conf = _blank_conf_at(syllable_blank_confidences, expected_vcv_idx)
        eff_jump_default, eff_jump_high_conf = _effective_jump_limits(
            row_jump_default,
            row_jump_high_conf,
            expected_blank_conf,
        )
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
                max_jump_default=eff_jump_default,
                max_jump_high_conf=eff_jump_high_conf,
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
                    max_forward = int(max(0, eff_jump_default))
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
            elif c_vowel and _is_kr_glide_vowel(c_vowel) != _is_kr_glide_vowel(t_vowel):
                if expected_vcv_idx < len(romaji_syllables):
                    _e_on, e_vowel, _e_coda = split_syllable_parts_fn(romaji_syllables[expected_vcv_idx])
                    if e_vowel == t_vowel and _is_kr_glide_vowel(e_vowel) == _is_kr_glide_vowel(t_vowel):
                        if debug_logging:
                            log_fn(
                                f"🛡️ {fname}: KR VCV 활음 불일치 차단 "
                                f"{vcv_selected_w_idx + 1}->{expected_vcv_idx + 1} ({alias})"
                            )
                        vcv_selected_w_idx = expected_vcv_idx
                        cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
        guarded_idx, guarded = _blank_guard_idx(
            expected_vcv_idx,
            vcv_selected_w_idx,
            syllable_blank_confidences,
        )
        if guarded and guarded_idx != vcv_selected_w_idx:
            if debug_logging:
                log_fn(
                    f"🛡️ {fname}: KR VCV blank guard 적용 "
                    f"({vcv_selected_w_idx + 1}->{guarded_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(guarded_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.10)
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
    syllable_blank_confidences=None,
):
    expected_cv_idx = int(cv_seq_idx)
    expected_blank_conf = _blank_conf_at(syllable_blank_confidences, expected_cv_idx)
    eff_jump_default, eff_jump_high_conf = _effective_jump_limits(
        row_jump_default,
        row_jump_high_conf,
        expected_blank_conf,
    )
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
    vv_left, vv_right = ("", "")
    if alias_type == "vv":
        vv_left, vv_right = _extract_vv_pair_tokens(alias, split_syllable_parts_fn)
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
        if alias_type == "vv" and vv_left and vv_right and not _vv_pair_matches(
            forced_selected_idx, romaji_syllables, split_syllable_parts_fn, vv_left, vv_right
        ):
            keep_forced = False
        forced_blank_conf = _blank_conf_at(syllable_blank_confidences, forced_selected_idx)
        if forced_blank_conf >= 0.66 and (expected_blank_conf + 0.08) < forced_blank_conf:
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
            max_jump_default=eff_jump_default,
            max_jump_high_conf=eff_jump_high_conf,
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
                max_forward = int(max(0, eff_jump_default))
                if (
                    float(row_mapping_confidence) >= max(float(file_mapping_conf_th), 0.50)
                    and float(resolve_meta.get("best_score", 0.0) or 0.0) >= 84.0
                ):
                    max_forward = int(max(max_forward, eff_jump_high_conf))
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
    if alias_type == "vv" and vv_left and vv_right:
        if selected_w_idx is None or not _vv_pair_matches(
            selected_w_idx, romaji_syllables, split_syllable_parts_fn, vv_left, vv_right
        ):
            candidate_idx = None
            if planned_vv_idx is not None and _vv_pair_matches(
                planned_vv_idx, romaji_syllables, split_syllable_parts_fn, vv_left, vv_right
            ):
                candidate_idx = planned_vv_idx
            elif _vv_pair_matches(
                expected_cv_idx, romaji_syllables, split_syllable_parts_fn, vv_left, vv_right
            ):
                candidate_idx = expected_cv_idx
            if candidate_idx is not None and candidate_idx != selected_w_idx:
                if debug_logging:
                    log_fn(
                        f"🛡️ {fname}: VV pair 매칭 보정 "
                        f"{(selected_w_idx + 1) if selected_w_idx is not None else '?'}->{candidate_idx + 1} ({alias})"
                    )
                selected_w_idx = int(candidate_idx)
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
    guarded_idx, guarded = _blank_guard_idx(
        expected_cv_idx,
        selected_w_idx,
        syllable_blank_confidences,
    )
    if guarded and guarded_idx != selected_w_idx:
        if debug_logging:
            log_fn(
                f"🛡️ {fname}: KR blank guard 적용 "
                f"({selected_w_idx + 1}->{guarded_idx + 1}, {alias})"
            )
        row_jump_blocked = 1
        row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.10)
        selected_w_idx = int(guarded_idx)
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
