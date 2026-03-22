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
    if blank >= 0.66:
        return 0, 0
    if blank >= 0.40:
        return min(jump_default, 1), min(jump_high_conf, 1)
    return jump_default, jump_high_conf


def _blank_guard_idx(expected_idx, selected_idx, syllable_blank_confidences, alias_type=""):
    if selected_idx is None:
        return selected_idx, False
    a_type = str(alias_type or "").strip().lower()
    sel_blank = _blank_conf_at(syllable_blank_confidences, selected_idx)
    exp_blank = _blank_conf_at(syllable_blank_confidences, expected_idx)
    if a_type == "cv_head":
        fwd_blank_th = 0.48
        blank_gap_th = 0.06
        hard_blank_th = 0.68
        exp_blank_max = 0.56
    elif a_type == "vcv":
        fwd_blank_th = 0.52
        blank_gap_th = 0.07
        hard_blank_th = 0.72
        exp_blank_max = 0.58
    else:
        fwd_blank_th = 0.54
        blank_gap_th = 0.08
        hard_blank_th = 0.74
        exp_blank_max = 0.60
    if selected_idx > expected_idx and sel_blank >= fwd_blank_th and (exp_blank + blank_gap_th) < sel_blank:
        return expected_idx, True
    if sel_blank >= hard_blank_th and exp_blank <= exp_blank_max:
        return expected_idx, True
    return selected_idx, False


def _global_plan_guard_idx(
    *,
    alias_type,
    file_format,
    target_clean,
    expected_idx,
    planned_idx,
    selected_idx,
    row_mapping_confidence,
    file_mapping_conf_th,
    file_mapping_low_conf,
    romaji_syllables,
    syllable_blank_confidences,
    cv_match_score_fn,
):
    a_type = str(alias_type or "").strip().lower()
    fmt = str(file_format or "").strip().lower()
    if a_type not in {"cv", "cv_head", "vcv"}:
        return selected_idx, False
    if fmt not in {"cvvc", "cvc", "cv", "vcv"}:
        return selected_idx, False
    if planned_idx is None or selected_idx is None:
        return selected_idx, False
    if not romaji_syllables:
        return selected_idx, False

    n = len(romaji_syllables)
    p_idx = int(max(0, min(int(planned_idx), n - 1)))
    s_idx = int(max(0, min(int(selected_idx), n - 1)))
    if s_idx == p_idx:
        return s_idx, False

    conf = float(row_mapping_confidence or 0.0)
    conf_th = float(file_mapping_conf_th or 0.0)
    selected_blank = _blank_conf_at(syllable_blank_confidences, s_idx)
    planned_blank = _blank_conf_at(syllable_blank_confidences, p_idx)

    selected_score = -1.0
    planned_score = -1.0
    if target_clean:
        try:
            selected_score = float(cv_match_score_fn(target_clean, romaji_syllables[s_idx]))
        except Exception:
            selected_score = -1.0
        try:
            planned_score = float(cv_match_score_fn(target_clean, romaji_syllables[p_idx]))
        except Exception:
            planned_score = -1.0

    # 저신뢰/고공백 상황에서는 전역 monotonic planner를 사실상 고정점으로 사용한다.
    lock_blank_th = 0.48 if a_type == "cv_head" else 0.52
    strong_lock = bool(file_mapping_low_conf) or (selected_blank >= lock_blank_th) or (planned_blank >= lock_blank_th)
    conf_gate = max(conf_th + 0.06, 0.68) if a_type == "cv_head" else max(conf_th + 0.04, 0.64)
    if conf < conf_gate:
        strong_lock = True

    if strong_lock:
        return p_idx, True

    # 고신뢰에서도 planner 대비 과도한 전진 점프는 제한한다.
    allowed_forward = 1 if (a_type != "cv_head" and conf >= max(conf_th + 0.16, 0.80)) else 0
    max_idx = min(n - 1, p_idx + allowed_forward)
    if s_idx > max_idx:
        return max_idx, True
    if s_idx < p_idx:
        return p_idx, True

    # planner와의 차이가 작으면 안정성을 우선한다.
    score_gain = selected_score - planned_score
    if (selected_blank >= (planned_blank + 0.08)) or (score_gain < 10.0):
        return p_idx, True
    if abs(s_idx - p_idx) >= 2:
        return p_idx, True
    if abs(s_idx - expected_idx) > abs(p_idx - expected_idx) and score_gain < 16.0:
        return p_idx, True
    return s_idx, False


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


def _is_unvoiced_like_onset(onset: str) -> bool:
    o = str(onset or "").strip().lower()
    if not o:
        return False
    return o.startswith(("s", "sh", "h", "j", "ch", "c", "f", "th"))


def _mel_guided_cvvc_adjustment(
    *,
    file_format,
    alias_type,
    target_clean,
    expected_idx,
    selected_idx,
    row_mapping_confidence,
    file_mapping_conf_th,
    max_search_fwd,
    romaji_syllables,
    syllables_info,
    syllable_blank_confidences,
    split_syllable_parts_fn,
    cv_match_score_fn,
):
    """
    In low-confidence Korean CV-family rows, re-rank nearby candidates with mel class hints.
    This is a local correction only; global monotonic constraints remain unchanged.
    """
    fmt = str(file_format or "").strip().lower()
    a_type = str(alias_type or "").strip().lower()
    if fmt not in {"cv", "cvc", "cvvc", "vcv"}:
        return selected_idx, False
    if a_type not in {"cv", "cv_head", "vcv"}:
        return selected_idx, False
    if fmt == "vcv" and a_type != "vcv":
        return selected_idx, False
    if fmt in {"cv", "cvc"} and a_type == "vcv":
        return selected_idx, False
    if selected_idx is None or not romaji_syllables:
        return selected_idx, False
    if not target_clean or not syllables_info:
        return selected_idx, False
    if selected_idx < 0 or selected_idx >= len(romaji_syllables):
        return selected_idx, False
    if expected_idx < 0:
        expected_idx = 0
    if expected_idx >= len(romaji_syllables):
        expected_idx = len(romaji_syllables) - 1

    if fmt != "cvvc":
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
            if _blank_conf_at(syllable_blank_confidences, selected_idx) > 0.0:
                mel_hint = True
            elif _blank_conf_at(syllable_blank_confidences, expected_idx) > 0.0:
                mel_hint = True
        if not mel_hint:
            return selected_idx, False

    conf = float(row_mapping_confidence or 0.0)
    conf_th = float(file_mapping_conf_th or 0.0)
    selected_blank = _blank_conf_at(syllable_blank_confidences, selected_idx)
    expected_blank = _blank_conf_at(syllable_blank_confidences, expected_idx)
    conf_floor_by_fmt = {
        "cvvc": 0.62,
        "vcv": 0.64,
        "cvc": 0.60,
        "cv": 0.60,
    }
    blank_gate_by_fmt = {
        "cvvc": 0.58,
        "vcv": 0.56,
        "cvc": 0.60,
        "cv": 0.60,
    }
    conf_floor = conf_floor_by_fmt.get(fmt, 0.62)
    blank_gate = blank_gate_by_fmt.get(fmt, 0.58)
    should_try = bool(conf < max(conf_th + 0.08, conf_floor) or selected_blank >= blank_gate)
    if not should_try:
        return selected_idx, False

    n = len(romaji_syllables)
    lo = max(0, expected_idx - 1)
    hi = min(n - 1, expected_idx + int(max(1, max_search_fwd)))
    if lo >= hi:
        return selected_idx, False

    t_onset, _t_vowel, _t_coda = split_syllable_parts_fn(target_clean)
    target_unvoiced = _is_unvoiced_like_onset(t_onset)
    voiced_weight = 10.0 if a_type in {"cv", "cv_head", "vcv"} else 7.0
    unvoiced_weight = 8.0 if target_unvoiced else 3.0

    def _score(idx):
        text = float(cv_match_score_fn(target_clean, romaji_syllables[idx]))
        blank = _blank_conf_at(syllable_blank_confidences, idx)
        sil = _mel_conf_at(
            syllables_info,
            idx,
            "mel_silence_sparse_conf",
            fallback=_mel_conf_at(syllables_info, idx, "blank_confidence", 0.0),
        )
        voiced = _mel_conf_at(syllables_info, idx, "mel_voiced_formant_conf", 0.0)
        unvoiced = _mel_conf_at(syllables_info, idx, "mel_unvoiced_diffuse_conf", 0.0)
        breath = _mel_conf_at(syllables_info, idx, "mel_breath_like_conf", 0.0)
        jump_penalty = abs(idx - expected_idx) * 8.0
        mel_bonus = (voiced_weight * voiced) + (unvoiced_weight * unvoiced)
        mel_penalty = (22.0 * blank) + (14.0 * sil) + (6.0 * breath)
        stability_bonus = 1.5 if idx == selected_idx else 0.0
        return text + mel_bonus + stability_bonus - mel_penalty - jump_penalty

    best_idx = int(selected_idx)
    best_score = float(_score(best_idx))
    for idx in range(lo, hi + 1):
        cand_score = float(_score(idx))
        if cand_score > best_score:
            best_idx = int(idx)
            best_score = cand_score

    if best_idx == int(selected_idx):
        return selected_idx, False

    # Keep changes conservative unless evidence is strong.
    selected_score = float(_score(int(selected_idx)))
    if best_score < (selected_score + 4.0):
        return selected_idx, False
    if _blank_conf_at(syllable_blank_confidences, best_idx) >= 0.72 and expected_blank <= 0.60:
        return selected_idx, False
    return best_idx, True


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
            alias_type="vcv",
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
        mel_idx, mel_adjusted = _mel_guided_cvvc_adjustment(
            file_format=file_format,
            alias_type="vcv",
            target_clean=target_clean,
            expected_idx=expected_vcv_idx,
            selected_idx=vcv_selected_w_idx,
            row_mapping_confidence=row_mapping_confidence,
            file_mapping_conf_th=file_mapping_conf_th,
            max_search_fwd=max(eff_jump_high_conf, 2),
            romaji_syllables=romaji_syllables,
            syllables_info=syllables_info,
            syllable_blank_confidences=syllable_blank_confidences,
            split_syllable_parts_fn=split_syllable_parts_fn,
            cv_match_score_fn=cv_match_score_fn,
        )
        if mel_adjusted and mel_idx != vcv_selected_w_idx:
            if debug_logging:
                log_fn(
                    f"🧭 {fname}: KR VCV mel-guided remap "
                    f"({vcv_selected_w_idx + 1}->{mel_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(mel_idx)
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
    syllable_blank_confidences=None,
    syllables_info=None,
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
        alias_type=alias_type,
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
    mel_idx, mel_adjusted = _mel_guided_cvvc_adjustment(
        file_format=file_format,
        alias_type=alias_type,
        target_clean=target_clean,
        expected_idx=expected_cv_idx,
        selected_idx=selected_w_idx,
        row_mapping_confidence=row_mapping_confidence,
        file_mapping_conf_th=file_mapping_conf_th,
        max_search_fwd=max(eff_jump_high_conf, 2),
        romaji_syllables=romaji_syllables,
        syllables_info=syllables_info,
        syllable_blank_confidences=syllable_blank_confidences,
        split_syllable_parts_fn=split_syllable_parts_fn,
        cv_match_score_fn=cv_match_score_fn,
    )
    if mel_adjusted and mel_idx != selected_w_idx:
        if debug_logging:
            log_fn(
                f"🧭 {fname}: KR {alias_type.upper()} mel-guided remap "
                f"({selected_w_idx + 1}->{mel_idx + 1}, {alias})"
            )
        selected_w_idx = int(mel_idx)
        cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    if planned_cv_idx is not None:
        guarded_plan_idx, plan_guarded = _global_plan_guard_idx(
            alias_type=alias_type,
            file_format=file_format,
            target_clean=target_clean,
            expected_idx=expected_cv_idx,
            planned_idx=planned_cv_idx,
            selected_idx=selected_w_idx,
            row_mapping_confidence=row_mapping_confidence,
            file_mapping_conf_th=file_mapping_conf_th,
            file_mapping_low_conf=file_mapping_low_conf,
            romaji_syllables=romaji_syllables,
            syllable_blank_confidences=syllable_blank_confidences,
            cv_match_score_fn=cv_match_score_fn,
        )
        if plan_guarded and guarded_plan_idx != selected_w_idx:
            if debug_logging:
                log_fn(
                    f"孱・・{fname}: KR global-plan guard ・・圸 "
                    f"({selected_w_idx + 1}->{guarded_plan_idx + 1}, {alias})"
                )
            row_jump_blocked = 1
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.12)
            selected_w_idx = int(guarded_plan_idx)
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
