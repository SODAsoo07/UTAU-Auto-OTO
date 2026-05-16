from __future__ import annotations

import os
import re

from core.ja_oto_mapping import (
    _ja_soft_cv_match_level,
    _ja_special_mora_class,
    _normalize_ja_syllable_token,
)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _resolve_ja_low_tier_forward_window(format_type: str, fallback: int = 1) -> int:
    fmt = str(format_type or "").strip().lower()
    if fmt:
        fmt_key = "UTOA_JA_LOW_TIER_FORWARD_MAX_" + re.sub(r"[^a-z0-9]+", "_", fmt).strip("_").upper()
        fmt_val = _env_int(fmt_key, -1)
        if fmt_val >= 0:
            return int(fmt_val)
    global_val = _env_int("UTOA_JA_LOW_TIER_FORWARD_MAX", int(fallback))
    return max(0, int(global_val))


def _strict_ja_cvvc_occurrence_expected() -> bool:
    return _env_bool("UTOA_JA_CVVC_OCCURRENCE_STRICT_EXPECTED", True)


def _plan_occ_double_guard_enabled() -> bool:
    return _env_bool("UTOA_PLAN_OCCURRENCE_DOUBLE_GUARD_ENABLE", True)


def _plan_occ_blank_improve_delta() -> float:
    return max(
        0.0,
        min(
            0.30,
            _env_float("UTOA_PLAN_OCCURRENCE_BLANK_IMPROVE_DELTA", 0.02),
        ),
    )


def _syllable_info_token(syl_info):
    if not isinstance(syl_info, dict):
        return ""
    for key in ("roman_cv", "roman", "word", "kana"):
        raw = syl_info.get(key, "")
        if raw:
            return str(raw).strip()
    return ""


def _guard_plan_occurrence_candidate_ja(
    *,
    target_tok,
    expected_idx,
    candidate_idx,
    syllables_info,
):
    if not syllables_info:
        return False, "no_syllables"
    n = len(syllables_info)
    if n <= 0:
        return False, "no_syllables"
    e_idx = int(max(0, min(int(expected_idx), n - 1)))
    c_idx = int(max(0, min(int(candidate_idx), n - 1)))
    if c_idx == e_idx:
        return True, "same_as_expected"

    target_norm = _normalize_ja_syllable_token(target_tok)
    cand_norm = _normalize_ja_syllable_token(_syllable_info_token(syllables_info[c_idx]))
    if not target_norm or cand_norm != target_norm:
        return False, f"token_mismatch({cand_norm or '-'}!={target_norm or '-'})"

    expected_blank = _blank_conf_at(syllables_info, e_idx)
    candidate_blank = _blank_conf_at(syllables_info, c_idx)
    min_delta = _plan_occ_blank_improve_delta()
    if candidate_blank > max(0.0, expected_blank - min_delta):
        return False, f"blank_not_improved({candidate_blank:.2f}>={expected_blank:.2f}-{min_delta:.2f})"
    return True, "accepted"


def _token_alpha(tok):
    return re.sub(r"[^a-z]", "", str(tok or "").strip().lower())


def _is_ja_vowel_only_token(tok):
    s = _token_alpha(tok)
    if not s:
        return False
    return all(ch in "aeiou" for ch in s)


def _is_ja_liquid_token(tok):
    s = _token_alpha(tok)
    return bool(s) and s.startswith(("r", "l"))


def _split_ja_token_parts(tok):
    s = _token_alpha(tok)
    if not s:
        return "", ""
    for i, ch in enumerate(s):
        if ch in "aeiou":
            return s[:i], ch
    return s, ""


def _is_ja_weak_boundary_token(tok):
    onset, _vowel = _split_ja_token_parts(tok)
    if not onset:
        return False
    weak_roots = ("m", "n", "l", "r", "j", "y", "w", "ng", "ny", "my", "ry", "ly")
    return any(onset == root or onset.startswith(root) for root in weak_roots)


def _local_transition_contrast(syllables_info, idx):
    if not syllables_info or idx is None:
        return 0.0
    n = len(syllables_info)
    i = int(max(0, min(int(idx), n - 1)))

    def _vec(at_idx):
        return (
            _mel_conf_at(syllables_info, at_idx, "mel_voiced_formant_conf", 0.0),
            _mel_conf_at(syllables_info, at_idx, "mel_unvoiced_diffuse_conf", 0.0),
            _mel_conf_at(
                syllables_info,
                at_idx,
                "mel_silence_sparse_conf",
                fallback=_mel_conf_at(syllables_info, at_idx, "blank_confidence", 0.0),
            ),
            _blank_conf_at(syllables_info, at_idx),
        )

    base = _vec(i)
    contrast = 0.0
    for nb in (i - 1, i + 1):
        if nb < 0 or nb >= n:
            continue
        cur = _vec(nb)
        dist = sum(abs(float(base[k]) - float(cur[k])) for k in range(len(base))) / float(len(base))
        contrast = max(contrast, max(0.0, min(1.0, dist)))
    return float(contrast)


def _estimate_fry_like_conf(syllables_info, idx):
    if not syllables_info or idx is None:
        return 0.0
    explicit = 0.0
    for key in ("mel_vocal_fry_conf", "mel_fry_like_conf", "vocal_fry_conf"):
        explicit = max(explicit, _mel_conf_at(syllables_info, idx, key, 0.0))
    voiced = _mel_conf_at(syllables_info, idx, "mel_voiced_formant_conf", 0.0)
    unvoiced = _mel_conf_at(syllables_info, idx, "mel_unvoiced_diffuse_conf", 0.0)
    sil = _mel_conf_at(
        syllables_info,
        idx,
        "mel_silence_sparse_conf",
        fallback=_mel_conf_at(syllables_info, idx, "blank_confidence", 0.0),
    )
    breath = _mel_conf_at(syllables_info, idx, "mel_breath_like_conf", 0.0)
    blank = _blank_conf_at(syllables_info, idx)
    proxy = max(
        0.0,
        min(1.0, (0.92 * voiced) - (0.52 * unvoiced) - (0.38 * sil) - (0.28 * breath) - (0.20 * blank)),
    )
    return max(explicit, proxy)


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
    target_cls = _ja_special_mora_class(target_norm)
    target_youon = target_cls in {"youon", "inserted"}

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
    selected_tok_now = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[selected_idx]))
    expected_tok_now = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[expected_idx]))
    selected_soft_now = int(_ja_soft_cv_match_level(target_norm, selected_tok_now) or 0) if selected_tok_now else 0
    expected_soft_now = int(_ja_soft_cv_match_level(target_norm, expected_tok_now) or 0) if expected_tok_now else 0
    score_gap_trigger = bool(expected_soft_now >= max(1, selected_soft_now + 1))
    should_try = bool(conf < max(conf_th + 0.08, 0.62) or selected_blank >= 0.58 or score_gap_trigger)
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
    low_tier = bool(conf < max(conf_th, 0.62) or selected_blank >= 0.58)
    if low_tier:
        low_cap_default = 0 if fmt == "cvvc" else 1
        low_cap = _resolve_ja_low_tier_forward_window(fmt, fallback=low_cap_default)
        hi = min(hi, expected_idx + int(max(0, low_cap)))
    if order_locked:
        lo = expected_idx
        hi = min(hi, expected_idx + 1)
    if lo >= hi:
        return selected_idx, False

    weak_target = _is_ja_weak_boundary_token(target_norm)
    reduce_missing_strict = _env_bool("UTOA_WEAK_BOUNDARY_MISSING_REDUCTION_ENABLE", False)
    anti_mismap_strict = _env_bool("UTOA_WEAK_BOUNDARY_ANTIMISMAP_ENABLE", False)
    if weak_target:
        if anti_mismap_strict:
            hi = min(hi, expected_idx + 1)
        elif reduce_missing_strict:
            hi = min(hi, expected_idx + 3)
        else:
            hi = min(hi, expected_idx + 2)
        if lo >= hi:
            return selected_idx, False

    target_onset, target_vowel = _split_ja_token_parts(target_norm)
    target_fry_suppress = bool(_is_ja_vowel_only_token(target_norm) or _is_ja_liquid_token(target_norm))
    fry_weight = -8.5 if target_fry_suppress else 1.2

    def _score(idx):
        cand_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[idx]))
        cand_cls = _ja_special_mora_class(cand_tok)
        cand_youon = cand_cls in {"youon", "inserted"}
        soft = int(_ja_soft_cv_match_level(target_norm, cand_tok) or 0) if cand_tok else 0
        cand_onset, cand_vowel = _split_ja_token_parts(cand_tok)
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
        fry_like = _estimate_fry_like_conf(syllables_info, idx)
        jump_scale = 10.0 if a_type in {"cv", "cv_head"} else 8.0
        if weak_target:
            jump_scale += 2.0
            if anti_mismap_strict:
                jump_scale += 3.0
            elif reduce_missing_strict:
                jump_scale -= 1.0
        jump_penalty = abs(idx - expected_idx) * jump_scale
        mel_bonus = (9.0 * voiced) + (5.0 * unvoiced) + (fry_weight * fry_like)
        blank_weight = 20.0
        sil_weight = 12.0
        if weak_target:
            blank_weight = 14.0
            sil_weight = 9.0
            if anti_mismap_strict:
                blank_weight = 12.0
                sil_weight = 8.0
        mel_penalty = (blank_weight * blank) + (sil_weight * sil) + (6.0 * breath)
        mismatch_penalty = 0.0
        core_bonus = 0.0
        if soft <= 0:
            mismatch_penalty += 46.0 if a_type in {"cv", "cv_head"} else 34.0
        elif soft == 1:
            mismatch_penalty += 12.0 if a_type in {"cv", "cv_head"} else 6.0
        if target_youon != cand_youon and cand_tok != target_norm:
            mismatch_penalty += 52.0 if a_type in {"cv", "cv_head"} else 38.0
        if target_youon and cand_tok != target_norm and soft < 3:
            mismatch_penalty += 22.0
        if (not target_youon) and cand_youon and cand_tok != target_norm:
            mismatch_penalty += 16.0
        if target_vowel:
            if cand_vowel == target_vowel:
                core_bonus += 8.0 if weak_target else 5.0
            else:
                mismatch_penalty += 24.0
                if weak_target:
                    mismatch_penalty += 10.0 if anti_mismap_strict else 6.0
        if target_onset and a_type in {"cv", "cv_head"}:
            if cand_onset == target_onset:
                core_bonus += 3.0
            elif weak_target:
                mismatch_penalty += 8.0 if anti_mismap_strict else 4.0
        if weak_target and idx != expected_idx:
            contrast = _local_transition_contrast(syllables_info, idx)
            contrast_th = 0.18 if anti_mismap_strict else 0.14
            if contrast < contrast_th:
                contrast_penalty = (contrast_th - contrast) * (30.0 if anti_mismap_strict else 18.0)
                if reduce_missing_strict and not anti_mismap_strict:
                    contrast_penalty *= 0.8
                mismatch_penalty += contrast_penalty
        stability_bonus = 1.5 if idx == selected_idx else 0.0
        return text + mel_bonus + stability_bonus + core_bonus - mel_penalty - jump_penalty - mismatch_penalty

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
    if weak_target:
        min_gain += 0.8
        if anti_mismap_strict:
            min_gain += 1.8
        elif reduce_missing_strict:
            min_gain = max(4.0, min_gain - 0.8)
    if best_score < (selected_score + min_gain):
        if weak_target and reduce_missing_strict:
            expected_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[int(expected_idx)]))
            expected_soft = int(_ja_soft_cv_match_level(target_norm, expected_tok) or 0) if expected_tok else 0
            expected_blank = _blank_conf_at(syllables_info, expected_idx)
            if expected_soft > 0 and selected_blank >= 0.62 and expected_blank + 0.06 < selected_blank:
                return int(expected_idx), bool(int(expected_idx) != int(selected_idx))
        return selected_idx, False

    selected_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[int(selected_idx)]))
    selected_cls = _ja_special_mora_class(selected_tok)
    best_tok = normalize_syllable_token_fn(syllable_info_token_fn(syllables_info[int(best_idx)]))
    best_cls = _ja_special_mora_class(best_tok)
    selected_soft = int(_ja_soft_cv_match_level(target_norm, selected_tok) or 0) if selected_tok else 0
    best_soft = int(_ja_soft_cv_match_level(target_norm, best_tok) or 0) if best_tok else 0

    if target_youon and best_cls not in {"youon", "inserted"} and best_tok != target_norm:
        return selected_idx, False
    if (not target_youon) and best_cls in {"youon", "inserted"} and best_tok != target_norm:
        return selected_idx, False
    if target_youon and best_tok != target_norm and best_soft < 3:
        return selected_idx, False

    if best_soft <= 0:
        return selected_idx, False
    if selected_soft >= 2 and best_soft < selected_soft:
        return selected_idx, False
    if a_type in {"cv", "cv_head"} and best_soft < 2 and best_score < (selected_score + 12.0):
        return selected_idx, False
    if weak_target and target_vowel:
        _best_on, best_vowel = _split_ja_token_parts(best_tok)
        _exp_on, exp_vowel = _split_ja_token_parts(expected_tok_now)
        if exp_vowel == target_vowel and best_vowel != target_vowel:
            return selected_idx, False
    if target_youon:
        if selected_tok == target_norm and best_tok != target_norm:
            return selected_idx, False
        if selected_cls in {"youon", "inserted"} and best_cls not in {"youon", "inserted"}:
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

    force_lock_mode = _env_bool("UTOA_LOW_CONF_FORCE_LOCK_MODE", False)
    if force_lock_mode:
        lock_conf_floor = max(
            float(mapping_conf_threshold or 0.0),
            _env_float("UTOA_LOW_CONF_FORCE_LOCK_THRESHOLD", 0.62),
        )
        lock_blank_floor = _env_float("UTOA_LOW_CONF_FORCE_LOCK_BLANK", 0.58)
        expected_blank = _blank_conf_at(syllables_info, expected_idx)
        if (
            float(mapping_confidence_base or 0.0) < lock_conf_floor
            or expected_blank >= lock_blank_floor
        ):
            target_tok_vcv_norm = normalize_syllable_token_fn(extract_target_syllable_fn(alias))
            if debug_logging:
                log_fn(
                    f"[JA] {fname}: VCV 저신뢰 강제 고정 "
                    f"({expected_idx + 1}, conf={float(mapping_confidence_base or 0.0):.2f}, blank={expected_blank:.2f})"
                )
            row_abstain = decide_cv_row_abstain_fn(
                alias_type="cv",
                format_type=format_type,
                candidate_idx=expected_idx,
                candidate_count=len(syllables_info),
                candidate_active=(
                    is_cv_syllable_active_fn(syllables_info[expected_idx], require_vowel=True)
                    if 0 <= expected_idx < len(syllables_info)
                    else False
                ),
                active_only_formats={"cvvc", "cv"},
            )
            return {
                "expected_idx": int(expected_idx),
                "mapped_idx": int(expected_idx),
                "target_tok": target_tok_vcv_norm,
                "row_abstain": row_abstain,
            }

    target_tok_vcv_raw = extract_target_syllable_fn(alias)
    planned_idx_vcv = resolve_planned_cv_index_fn(
        ja_planned_cv_indices,
        expected_seq_idx,
        target_tok_vcv_raw,
        syllables_info,
        alias_type="vcv",
    )
    if planned_idx_vcv is not None and _plan_occ_double_guard_enabled():
        plan_ok, plan_reason = _guard_plan_occurrence_candidate_ja(
            target_tok=target_tok_vcv_raw,
            expected_idx=expected_idx,
            candidate_idx=int(planned_idx_vcv),
            syllables_info=syllables_info,
        )
        if not plan_ok:
            if debug_logging:
                log_fn(
                    f"[JA] {fname}: VCV planned index rejected "
                    f"({expected_idx + 1}->{int(planned_idx_vcv) + 1}, {plan_reason}, {alias})"
                )
            planned_idx_vcv = None
    if planned_idx_vcv is not None:
        mapped_idx = int(planned_idx_vcv)
        if debug_logging and mapped_idx != expected_idx:
            log_fn(
                f"[JA] {fname}: VCV anchor plan applied "
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
            f"[JA] {fname}: VCV order drift recovered "
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
            search_fwd=(1 if mapping_tier == "low" else 3),
        )
        if fixed_idx_vcv is not None and fixed_idx_vcv != mapped_idx:
            log_fn(
                f"[JA] {fname}: VCV vowel mismatch fixed "
                f"{mapped_idx + 1}->{fixed_idx_vcv + 1} ({alias})"
            )
            mapped_idx = int(fixed_idx_vcv)
        elif expected_vowel_vcv == target_vowel_vcv:
            log_fn(
                f"[JA] {fname}: VCV vowel mismatch blocked "
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
                f"[JA] {fname}: VCV backward jump blocked "
                f"{mapped_idx + 1}->{last_vcv_mapped_idx + 1} ({alias})"
            )
            mapped_idx = last_vcv_mapped_idx
        elif mapped_idx > (last_vcv_mapped_idx + 2):
            clamped_idx = last_vcv_mapped_idx + 1
            log_fn(
                f"[JA] {fname}: VCV over-jump blocked "
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
        max_search_fwd=(1 if str(mapping_tier or "").strip().lower() == "low" else 3),
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
                f"[JA] {fname}: VCV mel-guided remap "
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
            search_fwd=(1 if mapping_tier == "low" else 4),
        )
        if (
            retry_idx_vcv is not None
            and retry_idx_vcv != mapped_idx
            and abs(retry_idx_vcv - expected_idx) <= 2
        ):
            log_fn(
                f"[JA] {fname}: VCV final re-search "
                f"{mapped_idx + 1}->{retry_idx_vcv + 1} ({alias})"
            )
            mapped_idx = int(retry_idx_vcv)
            mapped_tok_final = normalize_syllable_token_fn(
                syllable_info_token_fn(syllables_info[mapped_idx])
            )
        if mapped_tok_final != target_tok_vcv_norm:
            log_fn(
                f"[JA] {fname}: VCV token mismatch rollback "
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
        log_fn(f"[JA] {fname}: VCV syllable alignment adjusted {expected_idx + 1}->{mapped_idx + 1} ({alias})")

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
    format_type,
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
    forced_source = "planned"
    planned_idx = resolve_planned_cv_index_fn(
        planned_indices,
        expected_seq_idx,
        target_tok,
        syllables_info,
        alias_type=alias_type,
    )
    forced_idx = planned_idx
    if forced_idx is None:
        forced_source = "occurrence"
        try:
            forced_idx = resolve_cvvc_occurrence_index_fn(
                alias,
                alias_type,
                occurrence_map or {},
                occurrence_state,
                expected_idx=expected_idx,
            )
        except TypeError:
            forced_idx = resolve_cvvc_occurrence_index_fn(
                alias,
                alias_type,
                occurrence_map or {},
                occurrence_state,
            )
    elif debug_logging and forced_idx != expected_idx:
        alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
        log_fn(
            f"[JA] {fname}: {alias_label} anchor plan applied "
            f"{expected_idx + 1}->{int(forced_idx) + 1} ({alias})"
        )
    if forced_idx is not None:
        fmt = str(format_type or "").strip().lower()
        if (
            fmt in {"cvvc", "cv"}
            and _strict_ja_cvvc_occurrence_expected()
            and int(forced_idx) > int(expected_idx)
        ):
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"[JA] {fname}: {alias_label} occurrence forward clamp "
                    f"({int(forced_idx) + 1}->{int(expected_idx) + 1}, {alias})"
                )
            forced_idx = int(expected_idx)
    if forced_idx is not None and _plan_occ_double_guard_enabled():
        guard_ok, guard_reason = _guard_plan_occurrence_candidate_ja(
            target_tok=target_tok,
            expected_idx=expected_idx,
            candidate_idx=int(forced_idx),
            syllables_info=syllables_info,
        )
        if not guard_ok:
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"[JA] {fname}: {alias_label} {forced_source} index rejected "
                    f"({expected_idx + 1}->{int(forced_idx) + 1}, {guard_reason}, {alias})"
                )
            forced_idx = None
    if forced_idx is not None and not (0 <= int(forced_idx) < len(syllables_info)):
        remapped_idx = remap_forced_cv_index_fn(target_tok, expected_idx, syllables_info)
        if remapped_idx is not None:
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"[JA] {fname}: {alias_label} occurrence range remap "
                    f"({int(forced_idx) + 1}->{int(remapped_idx) + 1}, {alias})"
                )
            forced_idx = int(remapped_idx)
        else:
            if debug_logging:
                alias_label = "CV_HEAD" if str(alias_type or "").strip().lower() == "cv_head" else "CV"
                log_fn(
                    f"[JA] {fname}: {alias_label} occurrence invalidated "
                    f"(idx={int(forced_idx) + 1}, {alias})"
                )
            forced_idx = None
    return forced_idx


__all__ = ["resolve_ja_forced_cv_index", "select_ja_vcv_mapping"]
