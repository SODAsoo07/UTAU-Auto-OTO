from __future__ import annotations

import os
import re

from core.kr_oto_rules import _is_kr_glide_vowel


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


def _normalize_kr_token(token: str) -> str:
    return re.sub(r"[^a-z]", "", str(token or "").strip().lower())


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


def _blank_conf_from_rows(syllables_info, idx):
    if not syllables_info or idx is None:
        return 0.0
    try:
        if idx < 0 or idx >= len(syllables_info):
            return 0.0
        row = syllables_info[idx] or {}
        return max(0.0, min(1.0, float(row.get("blank_confidence", 0.0) or 0.0)))
    except Exception:
        return 0.0


def _guard_plan_occurrence_candidate_kr(
    *,
    target_clean,
    expected_idx,
    candidate_idx,
    romaji_syllables,
    syllables_info,
):
    if not romaji_syllables:
        return False, "no_syllables"
    n = len(romaji_syllables)
    if n <= 0:
        return False, "no_syllables"
    e_idx = int(max(0, min(int(expected_idx), n - 1)))
    c_idx = int(max(0, min(int(candidate_idx), n - 1)))
    if c_idx == e_idx:
        return True, "same_as_expected"

    target_norm = _normalize_kr_token(target_clean)
    cand_norm = _normalize_kr_token(romaji_syllables[c_idx])
    if not target_norm or cand_norm != target_norm:
        return False, f"token_mismatch({cand_norm or '-'}!={target_norm or '-'})"

    expected_blank = _blank_conf_from_rows(syllables_info, e_idx)
    candidate_blank = _blank_conf_from_rows(syllables_info, c_idx)
    min_delta = _plan_occ_blank_improve_delta()
    if candidate_blank > max(0.0, expected_blank - min_delta):
        return False, f"blank_not_improved({candidate_blank:.2f}>={expected_blank:.2f}-{min_delta:.2f})"
    return True, "accepted"


def _cv_order_prior_enabled() -> bool:
    if not _env_bool("UTOA_CV_ORDER_PRIOR_ENABLE", True):
        return False
    return _env_bool("UTOA_KR_CV_ORDER_PRIOR_ENABLE", True)


def _cv_order_prior_strength() -> float:
    return max(
        0.0,
        min(
            1.0,
            _env_float(
                "UTOA_KR_CV_ORDER_PRIOR_STRENGTH",
                _env_float("UTOA_CV_ORDER_PRIOR_STRENGTH", 0.56),
            ),
        ),
    )


def _resolve_low_tier_forward_window(file_format: str, fallback: int = 1) -> int:
    fmt = str(file_format or "").strip().lower()
    if fmt:
        fmt_key = "UTOA_KR_LOW_TIER_FORWARD_MAX_" + re.sub(r"[^a-z0-9]+", "_", fmt).strip("_").upper()
        fmt_val = _env_int(fmt_key, -1)
        if fmt_val >= 0:
            return int(fmt_val)
    global_val = _env_int("UTOA_KR_LOW_TIER_FORWARD_MAX", int(fallback))
    return max(0, int(global_val))


def _kr_coda_mismatch_penalty(alias_type: str) -> float:
    a_type = str(alias_type or "").strip().lower()
    base = {
        "vcv": 8.0,
        "cv_head": 12.0,
        "cv": 10.0,
    }.get(a_type, 8.0)
    env_val = _env_float("UTOA_KR_CODA_MISMATCH_PENALTY", -1.0)
    if env_val >= 0.0:
        return float(env_val)
    return float(base)


def _is_kr_mapping_only_enabled(file_format: str, alias_type: str) -> bool:
    if not _env_bool("UTOA_KR_MAPPING_ONLY_ENABLE", False):
        return False
    fmt = str(file_format or "").strip().lower()
    a_type = str(alias_type or "").strip().lower()
    if fmt not in {"cv", "cvc", "cvvc", "vcv"}:
        return False
    return a_type in {"cv", "cv_head", "vcv", "vv", "mono"}


def _pick_kr_mapping_only_idx(
    *,
    alias_type,
    target_clean,
    expected_idx,
    forced_selected_idx,
    planned_cv_idx,
    forced_vv_idx,
    planned_vv_idx,
    romaji_syllables,
    syllables_info,
    syllable_blank_confidences,
    cv_match_score_fn,
    split_syllable_parts_fn,
    vv_left,
    vv_right,
):
    if not romaji_syllables:
        return None
    n = len(romaji_syllables)
    if n <= 0:
        return None

    a_type = str(alias_type or "").strip().lower()
    expected = int(max(0, min(int(expected_idx), n - 1)))
    ordered = []
    if a_type == "vv":
        ordered.extend([planned_vv_idx, forced_vv_idx, planned_cv_idx, forced_selected_idx, expected])
    else:
        ordered.extend([planned_cv_idx, forced_selected_idx, expected])

    candidates = []
    seen = set()
    for idx in ordered:
        if idx is None:
            continue
        cand = int(max(0, min(int(idx), n - 1)))
        if cand in seen:
            continue
        seen.add(cand)
        if a_type == "vv" and vv_left and vv_right:
            if not _vv_pair_matches(cand, romaji_syllables, split_syllable_parts_fn, vv_left, vv_right):
                continue
        candidates.append(cand)
    if not candidates:
        candidates = [expected]

    best_idx = candidates[0]
    best_score = -1.0e12
    for cand in candidates:
        text_score = 0.0
        if target_clean:
            try:
                text_score = float(cv_match_score_fn(target_clean, romaji_syllables[cand]))
            except Exception:
                text_score = 0.0
        blank = _blank_conf_at(syllable_blank_confidences, cand)
        sil = _mel_conf_at(syllables_info, cand, "mel_silence_sparse_conf", fallback=blank)
        # TICKET-008: Onset proximity bonus rewards candidates near a strong mel onset.
        onset_bonus = _mel_onset_bonus(syllables_info, cand)
        score = text_score - (abs(cand - expected) * 10.0) - (blank * 22.0) - (sil * 14.0) + onset_bonus
        if score > best_score:
            best_score = score
            best_idx = cand
    return int(best_idx)


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
    if expected_idx is None:
        return selected_idx, False
    sel_blank = _blank_conf_at(syllable_blank_confidences, selected_idx)
    exp_blank = _blank_conf_at(syllable_blank_confidences, expected_idx)
    if selected_idx != expected_idx and sel_blank >= 0.58 and (exp_blank + 0.06) < sel_blank:
        return expected_idx, True
    if selected_idx > expected_idx and sel_blank >= 0.60 and (exp_blank + 0.10) < sel_blank:
        return expected_idx, True
    if sel_blank >= 0.74 and exp_blank <= 0.64:
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
    if not _cv_order_prior_enabled():
        return selected_idx, False

    prior_strength = _cv_order_prior_strength()
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

    # ・・・ｰ/・・ｵ・ｱ ・・勦・川・・・・・溜 monotonic planner・ｼ ・ｬ・､・・・・菩川愍・・・ｬ・ｩ﨑罹共.
    blank_gate = max(0.52, 0.60 - (0.06 * prior_strength))
    strong_lock = bool(file_mapping_low_conf) or (selected_blank >= blank_gate) or (planned_blank >= blank_gate)
    if conf < max(conf_th + (0.04 + (0.05 * prior_strength)), 0.62 + (0.05 * prior_strength)):
        strong_lock = True

    if strong_lock:
        return p_idx, True

    # ・・・ｰ・川・・・planner ・・・・ｼ・・復 ・・ｧ・・戦売・・・懦復﨑罹共.
    forward_th = max(conf_th + (0.18 + (0.08 * prior_strength)), 0.78 + (0.06 * prior_strength))
    allowed_forward = 1 if (a_type != "cv_head" and conf >= forward_th and prior_strength < 0.90) else 0
    max_idx = min(n - 1, p_idx + allowed_forward)
    if s_idx > max_idx:
        return max_idx, True
    if s_idx < p_idx:
        return p_idx, True

    # planner・・・・ｨ・ｴ・ ・卓愍・ｴ ・溢菩┳・・・ｰ・﨑罹共.
    score_gain = selected_score - planned_score
    if (selected_blank >= (planned_blank + (0.05 + (0.05 * prior_strength)))) or (
        score_gain < (8.0 + (7.0 * prior_strength))
    ):
        return p_idx, True
    if abs(s_idx - p_idx) >= 2:
        return p_idx, True
    if abs(s_idx - expected_idx) > abs(p_idx - expected_idx) and score_gain < (13.0 + (7.0 * prior_strength)):
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


def _is_blank_risky_idx(syllables_info, syllable_blank_confidences, idx):
    blank = _blank_conf_at(syllable_blank_confidences, idx)
    sil = _mel_conf_at(syllables_info, idx, "mel_silence_sparse_conf", fallback=blank)
    voiced = _mel_conf_at(syllables_info, idx, "mel_voiced_formant_conf", 0.0)
    unvoiced = _mel_conf_at(syllables_info, idx, "mel_unvoiced_diffuse_conf", 0.0)
    return bool(blank >= 0.70 or (sil >= 0.68 and (voiced + unvoiced) <= 0.18))


def _mel_onset_bonus(syllables_info, idx) -> float:
    """TICKET-008: Reward candidates whose syllable start sits near a strong mel onset.

    Bonus = mel_onset_energy * 6.0 - mel_onset_distance_ms * 0.05
    Only fires when mel_onset_energy >= UTOA_KR_ONSET_BONUS_MIN_ENERGY (default 0.15).
    """
    import os as _os
    try:
        min_energy = float(str(_os.environ.get("UTOA_KR_ONSET_BONUS_MIN_ENERGY", "0.15")).strip() or "0.15")
    except Exception:
        min_energy = 0.15

    if not syllables_info or idx is None:
        return 0.0
    try:
        if idx < 0 or idx >= len(syllables_info):
            return 0.0
        row = syllables_info[idx] or {}
        onset_energy = float(row.get("mel_onset_energy", 0.0) or 0.0)
        if onset_energy < min_energy:
            return 0.0
        onset_dist_ms = float(row.get("mel_onset_distance_ms", 999.0) or 999.0)
        # Reject candidates whose nearest onset is > 40ms away even if energy passes.
        if onset_dist_ms > 40.0:
            return 0.0
        return onset_energy * 6.0 - onset_dist_ms * 0.05
    except Exception:
        return 0.0


def _find_nonblank_fallback_idx(
    *,
    expected_idx,
    target_clean,
    romaji_syllables,
    syllables_info,
    syllable_blank_confidences,
    cv_match_score_fn,
    search_back=1,
    search_fwd=2,
):
    if expected_idx is None or not romaji_syllables:
        return None
    n = len(romaji_syllables)
    e = max(0, min(int(expected_idx), n - 1))
    lo = max(0, e - max(0, int(search_back)))
    hi = min(n - 1, e + max(0, int(search_fwd)))
    best_idx = None
    best_score = -10**9
    for idx in range(lo, hi + 1):
        if _is_blank_risky_idx(syllables_info, syllable_blank_confidences, idx):
            continue
        text_score = float(cv_match_score_fn(target_clean, romaji_syllables[idx])) if target_clean else 0.0
        blank = _blank_conf_at(syllable_blank_confidences, idx)
        sil = _mel_conf_at(syllables_info, idx, "mel_silence_sparse_conf", fallback=blank)
        # TICKET-008: Onset proximity bonus rewards candidates near a strong mel onset.
        onset_bonus = _mel_onset_bonus(syllables_info, idx)
        score = text_score - (abs(idx - e) * 8.0) - (blank * 20.0) - (sil * 16.0) + onset_bonus
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def _token_invariant_guard_idx(
    *,
    alias_type,
    target_clean,
    expected_idx,
    selected_idx,
    row_mapping_confidence,
    file_mapping_conf_th,
    file_mapping_low_conf,
    romaji_syllables,
    syllable_blank_confidences,
    split_syllable_parts_fn,
    cv_match_score_fn,
):
    if selected_idx is None or expected_idx is None or not romaji_syllables:
        return selected_idx, False
    n = len(romaji_syllables)
    if n <= 0:
        return selected_idx, False
    e = int(max(0, min(int(expected_idx), n - 1)))
    s = int(max(0, min(int(selected_idx), n - 1)))
    if s == e:
        return s, False
    target = str(target_clean or "").strip()
    if not target:
        return s, False

    exp_tok = str(romaji_syllables[e] or "")
    sel_tok = str(romaji_syllables[s] or "")
    exp_score = float(cv_match_score_fn(target, exp_tok))
    sel_score = float(cv_match_score_fn(target, sel_tok))
    exp_blank = _blank_conf_at(syllable_blank_confidences, e)
    sel_blank = _blank_conf_at(syllable_blank_confidences, s)
    conf = float(row_mapping_confidence or 0.0)
    conf_th = float(file_mapping_conf_th or 0.0)
    low_conf = bool(file_mapping_low_conf) or conf < max(conf_th + 0.02, 0.62)

    t_on, t_v, _t_c = split_syllable_parts_fn(target)
    e_on, e_v, _e_c = split_syllable_parts_fn(exp_tok)
    s_on, s_v, _s_c = split_syllable_parts_fn(sel_tok)
    weak_target = _is_weak_boundary_onset(t_on)
    anti_mismap_strict = _env_bool("UTOA_WEAK_BOUNDARY_ANTIMISMAP_ENABLE", False)

    expected_exact = exp_score >= 98.0
    selected_exact = sel_score >= 98.0
    expected_vowel_match = bool(t_v and e_v and t_v == e_v)
    selected_vowel_match = bool(t_v and s_v and t_v == s_v)

    if expected_exact and (not selected_exact):
        return e, True
    if expected_vowel_match and (not selected_vowel_match):
        if weak_target or anti_mismap_strict:
            return e, True
        if low_conf or exp_score >= max(84.0, sel_score + 2.0):
            return e, True
    if alias_type in {"cv", "cv_head"} and t_on and e_on:
        exp_onset_match = bool(t_on == e_on or t_on[:1] == e_on[:1])
        sel_onset_match = bool(t_on == s_on or t_on[:1] == s_on[:1])
        if exp_onset_match and (not sel_onset_match) and exp_score >= (sel_score - 6.0):
            return e, True
    if sel_blank >= 0.66 and (exp_blank + 0.08) < sel_blank and exp_score >= max(84.0, sel_score - 8.0):
        return e, True
    if low_conf and (sel_score + 12.0) < exp_score:
        return e, True
    return s, False


def _is_unvoiced_like_onset(onset: str) -> bool:
    o = str(onset or "").strip().lower()
    if not o:
        return False
    return o.startswith(("s", "sh", "h", "j", "ch", "c", "f", "th"))


def _is_weak_boundary_onset(onset: str) -> bool:
    o = str(onset or "").strip().lower()
    if not o:
        return False
    weak_roots = ("m", "n", "l", "r", "j", "y", "w", "ng", "ny", "my", "ry", "ly")
    return any(o == root or o.startswith(root) for root in weak_roots)


def _is_liquid_or_vowel_target(target_clean, split_syllable_parts_fn):
    try:
        onset, vowel, coda = split_syllable_parts_fn(target_clean)
    except Exception:
        return False
    onset_s = str(onset or "").strip().lower()
    vowel_s = str(vowel or "").strip().lower()
    coda_s = str(coda or "").strip().lower()
    vowel_only = bool(vowel_s and not onset_s and not coda_s)
    liquid = bool(onset_s and onset_s.startswith(("r", "l")))
    return bool(vowel_only or liquid)


def _local_transition_contrast(syllables_info, syllable_blank_confidences, idx):
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
                fallback=_blank_conf_at(syllable_blank_confidences, at_idx),
            ),
            _blank_conf_at(syllable_blank_confidences, at_idx),
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


def _estimate_fry_like_conf(syllables_info, syllable_blank_confidences, idx):
    if not syllables_info or idx is None:
        return 0.0
    explicit = 0.0
    for key in ("mel_vocal_fry_conf", "mel_fry_like_conf", "vocal_fry_conf"):
        explicit = max(explicit, _mel_conf_at(syllables_info, idx, key, 0.0))
    voiced = _mel_conf_at(syllables_info, idx, "mel_voiced_formant_conf", 0.0)
    unvoiced = _mel_conf_at(syllables_info, idx, "mel_unvoiced_diffuse_conf", 0.0)
    sil = _mel_conf_at(syllables_info, idx, "mel_silence_sparse_conf", fallback=_blank_conf_at(syllable_blank_confidences, idx))
    breath = _mel_conf_at(syllables_info, idx, "mel_breath_like_conf", 0.0)
    blank = _blank_conf_at(syllable_blank_confidences, idx)
    proxy = max(
        0.0,
        min(1.0, (0.92 * voiced) - (0.52 * unvoiced) - (0.38 * sil) - (0.28 * breath) - (0.20 * blank)),
    )
    return max(explicit, proxy)


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
    score_gap_trigger = False
    try:
        selected_score_now = float(cv_match_score_fn(target_clean, romaji_syllables[int(selected_idx)]))
        expected_score_now = float(cv_match_score_fn(target_clean, romaji_syllables[int(expected_idx)]))
        if (selected_score_now + 8.0) < expected_score_now:
            score_gap_trigger = True
    except Exception:
        score_gap_trigger = False
    should_try = bool(conf < max(conf_th + 0.08, conf_floor) or selected_blank >= blank_gate or score_gap_trigger)
    if not should_try:
        return selected_idx, False

    n = len(romaji_syllables)
    lo = max(0, expected_idx - 1)
    hi = min(n - 1, expected_idx + int(max(1, max_search_fwd)))
    low_tier = bool(conf < max(conf_th, conf_floor) or selected_blank >= blank_gate)
    if low_tier:
        low_cap_default = {
            "cvvc": 0,
            "vcv": 0,
            "cvc": 0,
            "cv": 0,
        }.get(fmt, 0)
        low_cap = _resolve_low_tier_forward_window(fmt, fallback=low_cap_default)
        hi = min(hi, expected_idx + int(max(0, low_cap)))
    if lo >= hi:
        return selected_idx, False

    t_onset, _t_vowel, _t_coda = split_syllable_parts_fn(target_clean)
    t_onset = str(t_onset or "").strip().lower()
    t_vowel = str(_t_vowel or "").strip().lower()
    t_coda = str(_t_coda or "").strip().lower()
    weak_target = _is_weak_boundary_onset(t_onset)
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
    target_unvoiced = _is_unvoiced_like_onset(t_onset)
    target_fry_suppress = _is_liquid_or_vowel_target(target_clean, split_syllable_parts_fn)
    voiced_weight = 10.0 if a_type in {"cv", "cv_head", "vcv"} else 7.0
    unvoiced_weight = 8.0 if target_unvoiced else 3.0
    fry_weight = -9.0 if target_fry_suppress else 1.2

    def _parts_at(idx):
        try:
            on, vw, cd = split_syllable_parts_fn(romaji_syllables[idx])
            return (
                str(on or "").strip().lower(),
                str(vw or "").strip().lower(),
                str(cd or "").strip().lower(),
            )
        except Exception:
            return "", "", ""

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
        fry_like = _estimate_fry_like_conf(syllables_info, syllable_blank_confidences, idx)
        c_onset, c_vowel, c_coda = _parts_at(idx)
        jump_scale = 10.0
        if weak_target:
            jump_scale = 12.0
            if anti_mismap_strict:
                jump_scale = 15.0
            elif reduce_missing_strict:
                jump_scale = 9.0
        jump_penalty = abs(idx - expected_idx) * jump_scale
        mel_bonus = (voiced_weight * voiced) + (unvoiced_weight * unvoiced) + (fry_weight * fry_like)
        blank_weight = 22.0
        sil_weight = 14.0
        if weak_target:
            blank_weight = 16.0
            sil_weight = 9.0
            if anti_mismap_strict:
                blank_weight = 14.0
                sil_weight = 8.0
        mel_penalty = (blank_weight * blank) + (sil_weight * sil) + (6.0 * breath)
        core_bonus = 0.0
        mismatch_penalty = 0.0
        if t_vowel:
            if c_vowel == t_vowel:
                core_bonus += 9.0
            else:
                mismatch_penalty += 34.0 if a_type == "vcv" else 44.0
                if weak_target:
                    mismatch_penalty += 10.0 if anti_mismap_strict else 6.0
        if t_onset:
            if c_onset == t_onset:
                core_bonus += 4.0
            elif a_type in {"cv", "cv_head"}:
                mismatch_penalty += 12.0
            else:
                mismatch_penalty += 6.0
            if weak_target and c_onset and c_onset != t_onset and a_type in {"cv", "cv_head"}:
                mismatch_penalty += 8.0 if anti_mismap_strict else 4.0
        if t_coda != c_coda:
            coda_penalty = _kr_coda_mismatch_penalty(a_type)
            if (not t_coda) != (not c_coda):
                coda_penalty += 2.0
            mismatch_penalty += coda_penalty
        if weak_target and idx != expected_idx:
            contrast = _local_transition_contrast(syllables_info, syllable_blank_confidences, idx)
            contrast_th = 0.18 if anti_mismap_strict else 0.14
            if contrast < contrast_th:
                contrast_penalty = (contrast_th - contrast) * (34.0 if anti_mismap_strict else 20.0)
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

    # Keep changes conservative unless evidence is strong.
    selected_score = float(_score(int(selected_idx)))
    min_gain = 6.0 if a_type in {"cv", "cv_head"} else 5.0
    if weak_target:
        min_gain += 1.0
        if anti_mismap_strict:
            min_gain += 1.8
        elif reduce_missing_strict:
            min_gain = max(4.2, min_gain - 0.9)
    if best_score < (selected_score + min_gain):
        return selected_idx, False
    if _blank_conf_at(syllable_blank_confidences, best_idx) >= 0.72 and expected_blank <= 0.60:
        return selected_idx, False

    sel_onset, sel_vowel, _sel_coda = _parts_at(int(selected_idx))
    best_onset, best_vowel, _best_coda = _parts_at(best_idx)
    if t_vowel:
        sel_vowel_match = bool(sel_vowel and sel_vowel == t_vowel)
        best_vowel_match = bool(best_vowel and best_vowel == t_vowel)
        if sel_vowel_match and not best_vowel_match:
            return selected_idx, False
        if not best_vowel_match:
            return selected_idx, False
        if weak_target:
            _exp_onset, exp_vowel, _exp_coda = _parts_at(int(expected_idx))
            if exp_vowel == t_vowel and best_vowel != t_vowel:
                return selected_idx, False
    if a_type in {"cv", "cv_head"} and t_onset:
        sel_onset_match = bool(sel_onset and sel_onset == t_onset)
        best_onset_match = bool(best_onset and best_onset == t_onset)
        if sel_onset_match and not best_onset_match:
            return selected_idx, False
        if not best_onset_match and best_score < (selected_score + 10.0):
            return selected_idx, False
    return best_idx, True


def _strong_vcv_monotonic_guard(
    *,
    target_clean,
    expected_idx,
    selected_idx,
    row_mapping_confidence,
    file_mapping_conf_th,
    romaji_syllables,
    syllable_blank_confidences,
    split_syllable_parts_fn,
    cv_match_score_fn,
):
    if selected_idx is None or not romaji_syllables:
        return selected_idx, False, "invalid_input"
    n = len(romaji_syllables)
    if n <= 0:
        return selected_idx, False, "invalid_input"
    e_idx = int(max(0, min(int(expected_idx), n - 1)))
    s_idx = int(max(0, min(int(selected_idx), n - 1)))
    original = int(s_idx)

    # Never move backward for VCV sequence.
    if s_idx < e_idx:
        s_idx = int(e_idx)

    exp_blank = _blank_conf_at(syllable_blank_confidences, e_idx)
    sel_blank = _blank_conf_at(syllable_blank_confidences, s_idx)
    conf = float(row_mapping_confidence or 0.0)
    conf_th = float(file_mapping_conf_th or 0.0)
    low_conf = bool(conf < max(conf_th, 0.62) or sel_blank >= 0.56 or exp_blank >= 0.58)
    max_forward = 0 if low_conf else 1
    allowed_hi = int(min(n - 1, e_idx + max_forward))
    if s_idx > allowed_hi:
        s_idx = int(allowed_hi)

    # Even for +1 forward, require clear gain under stronger VCV monotonic policy.
    if s_idx == (e_idx + 1):
        exp_tok = str(romaji_syllables[e_idx] or "")
        sel_tok = str(romaji_syllables[s_idx] or "")
        exp_score = float(cv_match_score_fn(target_clean, exp_tok)) if target_clean else 0.0
        sel_score = float(cv_match_score_fn(target_clean, sel_tok)) if target_clean else 0.0
        t_on, t_v, _t_c = split_syllable_parts_fn(target_clean)
        _e_on, e_v, _e_c = split_syllable_parts_fn(exp_tok)
        _s_on, s_v, _s_c = split_syllable_parts_fn(sel_tok)
        keep_forward = False
        if (sel_score >= (exp_score + 14.0)) and (sel_blank <= (exp_blank + 0.03)):
            keep_forward = True
        if t_v and (s_v == t_v) and (e_v != t_v) and (sel_blank <= (exp_blank + 0.08)):
            keep_forward = True
        if not keep_forward:
            s_idx = int(e_idx)

    return int(s_idx), bool(int(s_idx) != int(original)), "vcv_monotonic_strong"


def _mel_guided_vcv_adjustment(
    *,
    file_format,
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
    mel_idx, mel_adjusted = _mel_guided_cvvc_adjustment(
        file_format=file_format,
        alias_type="vcv",
        target_clean=target_clean,
        expected_idx=expected_idx,
        selected_idx=selected_idx,
        row_mapping_confidence=row_mapping_confidence,
        file_mapping_conf_th=file_mapping_conf_th,
        max_search_fwd=max_search_fwd,
        romaji_syllables=romaji_syllables,
        syllables_info=syllables_info,
        syllable_blank_confidences=syllable_blank_confidences,
        split_syllable_parts_fn=split_syllable_parts_fn,
        cv_match_score_fn=cv_match_score_fn,
    )
    guarded_idx, guarded, _reason = _strong_vcv_monotonic_guard(
        target_clean=target_clean,
        expected_idx=expected_idx,
        selected_idx=mel_idx,
        row_mapping_confidence=row_mapping_confidence,
        file_mapping_conf_th=file_mapping_conf_th,
        romaji_syllables=romaji_syllables,
        syllable_blank_confidences=syllable_blank_confidences,
        split_syllable_parts_fn=split_syllable_parts_fn,
        cv_match_score_fn=cv_match_score_fn,
    )
    if guarded:
        if guarded_idx == int(selected_idx):
            return int(selected_idx), False
        return int(guarded_idx), True
    return mel_idx, mel_adjusted


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
        force_lock_mode = _env_bool("UTOA_LOW_CONF_FORCE_LOCK_MODE", False)
        if force_lock_mode:
            lock_conf_floor = max(
                float(file_mapping_conf_th or 0.0),
                _env_float("UTOA_LOW_CONF_FORCE_LOCK_THRESHOLD", 0.62),
            )
            lock_blank_floor = _env_float("UTOA_LOW_CONF_FORCE_LOCK_BLANK", 0.58)
            if (
                float(row_mapping_confidence or 0.0) < lock_conf_floor
                or expected_blank_conf >= lock_blank_floor
            ):
                vcv_selected_w_idx = int(expected_vcv_idx)
                cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
                if debug_logging:
                    log_fn(
                        f"[KR] {fname}: VCV 저신뢰 강제 고정 "
                        f"({expected_vcv_idx + 1}, conf={float(row_mapping_confidence or 0.0):.2f}, blank={expected_blank_conf:.2f}, {alias})"
                    )
                return vcv_selected_w_idx, cv_seq_idx, row_mapping_confidence
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
        if planned_vcv_idx is not None and _plan_occ_double_guard_enabled():
            guard_ok, guard_reason = _guard_plan_occurrence_candidate_kr(
                target_clean=target_clean,
                expected_idx=expected_vcv_idx,
                candidate_idx=int(planned_vcv_idx),
                romaji_syllables=romaji_syllables,
                syllables_info=syllables_info,
            )
            if not guard_ok:
                if debug_logging:
                    log_fn(
                        f"[KR] {fname}: VCV planned index rejected "
                        f"({expected_vcv_idx + 1}->{int(planned_vcv_idx) + 1}, {guard_reason}, {alias})"
                    )
                planned_vcv_idx = None
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
                    f"[MAP] {fname}: KR VCV anchor-plan remap "
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
                    f"[MAP] {fname}: KR VCV jump guard remap "
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
                    f"[MAP] {fname}: KR VCV order guard remap "
                    f"({expected_vcv_idx + 1}->{ordered_vcv_idx + 1}, {alias})"
                )
        mono_idx, mono_guarded, _mono_reason = _strong_vcv_monotonic_guard(
            target_clean=target_clean,
            expected_idx=expected_vcv_idx,
            selected_idx=vcv_selected_w_idx,
            row_mapping_confidence=row_mapping_confidence,
            file_mapping_conf_th=file_mapping_conf_th,
            romaji_syllables=romaji_syllables,
            syllable_blank_confidences=syllable_blank_confidences,
            split_syllable_parts_fn=split_syllable_parts_fn,
            cv_match_score_fn=cv_match_score_fn,
        )
        if mono_guarded and mono_idx != vcv_selected_w_idx:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: VCV monotonic strong guard "
                    f"({vcv_selected_w_idx + 1}->{mono_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(mono_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)

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
                                f"[MAP] {fname}: KR VCV vowel-fix remap "
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
                                f"[MAP] {fname}: KR VCV glide guard remap "
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
                    f"[MAP] {fname}: KR VCV blank guard remap "
                    f"({vcv_selected_w_idx + 1}->{guarded_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(guarded_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.10)
        mel_idx, mel_adjusted = _mel_guided_vcv_adjustment(
            file_format=file_format,
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
                    f"[MAP] {fname}: KR VCV mel-guided remap "
                    f"({vcv_selected_w_idx + 1}->{mel_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(mel_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
        invariant_idx, invariant_guarded = _token_invariant_guard_idx(
            alias_type="vcv",
            target_clean=target_clean,
            expected_idx=expected_vcv_idx,
            selected_idx=vcv_selected_w_idx,
            row_mapping_confidence=row_mapping_confidence,
            file_mapping_conf_th=file_mapping_conf_th,
            file_mapping_low_conf=bool(row_mapping_confidence < max(float(file_mapping_conf_th), 0.60)),
            romaji_syllables=romaji_syllables,
            syllable_blank_confidences=syllable_blank_confidences,
            split_syllable_parts_fn=split_syllable_parts_fn,
            cv_match_score_fn=cv_match_score_fn,
        )
        if invariant_guarded and invariant_idx != vcv_selected_w_idx:
            if debug_logging:
                log_fn(
                    f"[MAP] {fname}: KR VCV token-invariant guard remap "
                    f"({vcv_selected_w_idx + 1}->{invariant_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(invariant_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
        final_mono_idx, final_mono_guarded, _final_mono_reason = _strong_vcv_monotonic_guard(
            target_clean=target_clean,
            expected_idx=expected_vcv_idx,
            selected_idx=vcv_selected_w_idx,
            row_mapping_confidence=row_mapping_confidence,
            file_mapping_conf_th=file_mapping_conf_th,
            romaji_syllables=romaji_syllables,
            syllable_blank_confidences=syllable_blank_confidences,
            split_syllable_parts_fn=split_syllable_parts_fn,
            cv_match_score_fn=cv_match_score_fn,
        )
        if final_mono_guarded and final_mono_idx != vcv_selected_w_idx:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: VCV final monotonic guard "
                    f"({vcv_selected_w_idx + 1}->{final_mono_idx + 1}, {alias})"
                )
            vcv_selected_w_idx = int(final_mono_idx)
            cv_seq_idx = max(cv_seq_idx, vcv_selected_w_idx + 1)
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
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
    alias_norm = str(alias_type or "").strip().lower()
    force_lock_mode = _env_bool("UTOA_LOW_CONF_FORCE_LOCK_MODE", False)
    if force_lock_mode and alias_norm in {"cv", "cv_head", "vcv"} and romaji_syllables:
        lock_conf_floor = max(
            float(file_mapping_conf_th or 0.0),
            _env_float("UTOA_LOW_CONF_FORCE_LOCK_THRESHOLD", 0.62),
        )
        lock_blank_floor = _env_float("UTOA_LOW_CONF_FORCE_LOCK_BLANK", 0.58)
        if (
            bool(file_mapping_low_conf)
            or float(row_mapping_confidence or 0.0) < lock_conf_floor
            or expected_blank_conf >= lock_blank_floor
        ):
            selected_w_idx = int(max(0, min(expected_cv_idx, len(romaji_syllables) - 1)))
            cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: CV 저신뢰 강제 고정 "
                    f"({selected_w_idx + 1}, conf={float(row_mapping_confidence or 0.0):.2f}, blank={expected_blank_conf:.2f}, {alias})"
                )
            return {
                "expected_cv_idx": int(expected_cv_idx),
                "selected_w_idx": int(selected_w_idx),
                "cv_seq_idx": int(cv_seq_idx),
                "row_mapping_confidence": float(row_mapping_confidence),
                "resolve_meta": {"forced_lock_mode": 1},
                "row_jump_blocked": 0,
                "forced_selected_idx": int(selected_w_idx),
            }
    mapping_only_enabled = _is_kr_mapping_only_enabled(file_format, alias_type)
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
    forced_source = ""
    if forced_vv_idx is not None:
        forced_source = "planned_vv"
    elif planned_vv_idx is not None:
        forced_source = "fallback_planned_vv"
    elif planned_cv_idx is not None:
        forced_source = "planned_cv"
    elif forced_cvvc_idx is not None:
        forced_source = "occurrence"
    forced_gate_rejected = False
    vv_left, vv_right = ("", "")
    if alias_type == "vv":
        vv_left, vv_right = _extract_vv_pair_tokens(alias, split_syllable_parts_fn)
    if (
        forced_selected_idx is not None
        and alias_norm in {"cv", "cv_head"}
        and forced_source in {"planned_cv", "occurrence"}
        and _plan_occ_double_guard_enabled()
    ):
        guard_ok, guard_reason = _guard_plan_occurrence_candidate_kr(
            target_clean=target_clean,
            expected_idx=expected_cv_idx,
            candidate_idx=int(forced_selected_idx),
            romaji_syllables=romaji_syllables,
            syllables_info=syllables_info,
        )
        if not guard_ok:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: {forced_source} index rejected "
                    f"({expected_cv_idx + 1}->{int(forced_selected_idx) + 1}, {guard_reason}, {alias})"
                )
            forced_selected_idx = None
            forced_gate_rejected = True
    if forced_selected_idx is not None and not (0 <= forced_selected_idx < len(romaji_syllables)):
        remapped_idx = remap_forced_cv_index_fn(
            target_clean,
            romaji_syllables,
            expected_cv_idx,
        )
        if remapped_idx is not None:
            if debug_logging:
                log_fn(
                    f"[MAP] {fname}: KR forced index remap "
                    f"({forced_selected_idx + 1}->{remapped_idx + 1}, {alias})"
                )
            forced_selected_idx = remapped_idx
        else:
            if debug_logging:
                log_fn(
                    f"[MAP] {fname}: KR forced index rejected "
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
        forced_sil_conf = _mel_conf_at(
            syllables_info,
            forced_selected_idx,
            "mel_silence_sparse_conf",
            fallback=forced_blank_conf,
        )
        expected_sil_conf = _mel_conf_at(
            syllables_info,
            expected_cv_idx,
            "mel_silence_sparse_conf",
            fallback=expected_blank_conf,
        )
        fmt_norm = str(file_format or "").strip().lower()
        alias_norm = str(alias_type or "").strip().lower()
        forced_blank_gate = 0.66
        forced_blank_margin = 0.08
        if fmt_norm in {"cvvc", "cvc", "vcv"} and alias_norm in {"cv", "cv_head"}:
            forced_blank_gate = 0.60
            forced_blank_margin = 0.06
        if forced_blank_conf >= forced_blank_gate and (expected_blank_conf + forced_blank_margin) < forced_blank_conf:
            keep_forced = False
        if forced_sil_conf >= (forced_blank_gate + 0.04) and (expected_sil_conf + 0.08) < forced_sil_conf:
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
                f"[MAP] {fname}: KR jump guard remap "
                f"({int(resolve_meta.get('raw_chosen_idx', selected_w_idx)) + 1}"
                f"->{int(resolve_meta.get('chosen_idx', selected_w_idx)) + 1}, {alias})"
            )
        if row_jump_blocked:
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.18)
    if mapping_only_enabled:
        mapping_only_idx = _pick_kr_mapping_only_idx(
            alias_type=alias_type,
            target_clean=target_clean,
            expected_idx=expected_cv_idx,
            forced_selected_idx=forced_selected_idx,
            planned_cv_idx=planned_cv_idx,
            forced_vv_idx=forced_vv_idx,
            planned_vv_idx=planned_vv_idx,
            romaji_syllables=romaji_syllables,
            syllables_info=syllables_info,
            syllable_blank_confidences=syllable_blank_confidences,
            cv_match_score_fn=cv_match_score_fn,
            split_syllable_parts_fn=split_syllable_parts_fn,
            vv_left=vv_left,
            vv_right=vv_right,
        )
        if mapping_only_idx is not None and mapping_only_idx != selected_w_idx:
            if debug_logging:
                log_fn(
                    f"[KR mapping-only] {fname}: idx {int(selected_w_idx) + 1 if selected_w_idx is not None else '?'}"
                    f"->{int(mapping_only_idx) + 1} ({alias})"
                )
            selected_w_idx = int(mapping_only_idx)
            cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
            row_jump_blocked = 1
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.04)
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
            if file_mapping_low_conf:
                fixed_search_fwd = min(
                    int(fixed_search_fwd),
                    int(_resolve_low_tier_forward_window(file_format, fallback=1)),
                )
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
                            f"[MAP] {fname}: KR vowel-fix jump guard remap "
                            f"({raw_fixed_idx + 1}->{fixed_idx + 1}, {alias})"
                        )
                if fixed_idx != selected_w_idx and abs(fixed_idx - expected_cv_idx) <= 2:
                    log_fn(
                        f"[MAP] {fname}: KR CV vowel-fix remap "
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
                        f"[MAP] {fname}: KR VV pair guard remap "
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
                f"[MAP] {fname}: KR CV order guard remap "
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
                f"[MAP] {fname}: KR blank guard remap "
                f"({selected_w_idx + 1}->{guarded_idx + 1}, {alias})"
            )
        row_jump_blocked = 1
        row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.10)
        selected_w_idx = int(guarded_idx)
        cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    mel_idx, mel_adjusted = selected_w_idx, False
    if not mapping_only_enabled:
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
                f"[MAP] {fname}: KR {alias_type.upper()} mel-guided remap "
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
                    f"[MAP] {fname}: KR global-plan guard remap "
                    f"({selected_w_idx + 1}->{guarded_plan_idx + 1}, {alias})"
                )
            row_jump_blocked = 1
            row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.12)
            selected_w_idx = int(guarded_plan_idx)
            cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    invariant_idx, invariant_guarded = _token_invariant_guard_idx(
        alias_type=alias_type,
        target_clean=target_clean,
        expected_idx=expected_cv_idx,
        selected_idx=selected_w_idx,
        row_mapping_confidence=row_mapping_confidence,
        file_mapping_conf_th=file_mapping_conf_th,
        file_mapping_low_conf=file_mapping_low_conf,
        romaji_syllables=romaji_syllables,
        syllable_blank_confidences=syllable_blank_confidences,
        split_syllable_parts_fn=split_syllable_parts_fn,
        cv_match_score_fn=cv_match_score_fn,
    )
    if invariant_guarded and invariant_idx != selected_w_idx:
        if debug_logging:
            log_fn(
                f"[MAP] {fname}: KR {str(alias_type or '').upper()} token-invariant guard remap "
                f"({selected_w_idx + 1}->{invariant_idx + 1}, {alias})"
            )
        row_jump_blocked = 1
        row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
        selected_w_idx = int(invariant_idx)
        cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    # KR CV/CV_HEAD in order-locked formats: prevent occasional +1 forward drift
    # unless the forward index is clearly better.
    if (
        selected_w_idx is not None
        and alias_type in {"cv", "cv_head"}
        and str(file_format or "").strip().lower() in {"cvvc", "cvc", "vcv"}
        and _env_bool("UTOA_KR_CVVC_FORWARD_ONE_STEP_GUARD", True)
        and romaji_syllables
    ):
        n_syl = int(len(romaji_syllables))
        expected_guard_idx = int(max(0, min(int(expected_cv_idx), n_syl - 1)))
        selected_guard_idx = int(max(0, min(int(selected_w_idx), n_syl - 1)))
        if selected_guard_idx == (expected_guard_idx + 1):
            exp_tok = str(romaji_syllables[expected_guard_idx] or "")
            sel_tok = str(romaji_syllables[selected_guard_idx] or "")
            exp_score = float(cv_match_score_fn(target_clean, exp_tok)) if target_clean else 0.0
            sel_score = float(cv_match_score_fn(target_clean, sel_tok)) if target_clean else 0.0
            exp_blank = _blank_conf_at(syllable_blank_confidences, expected_guard_idx)
            sel_blank = _blank_conf_at(syllable_blank_confidences, selected_guard_idx)
            t_on, t_v, _t_c = split_syllable_parts_fn(target_clean)
            e_on, e_v, _e_c = split_syllable_parts_fn(exp_tok)
            s_on, s_v, _s_c = split_syllable_parts_fn(sel_tok)
            keep_forward = False
            if (sel_score >= (exp_score + 12.0)) and (sel_blank <= (exp_blank + 0.03)):
                keep_forward = True
            if t_v and (s_v == t_v) and (e_v != t_v) and (sel_blank <= (exp_blank + 0.08)):
                keep_forward = True
            if t_on and (s_on == t_on) and (e_on != t_on) and (sel_score >= (exp_score + 8.0)):
                keep_forward = True
            if not keep_forward:
                if debug_logging:
                    log_fn(
                        f"[KR] {fname}: CV +1 forward guard "
                        f"({selected_guard_idx + 1}->{expected_guard_idx + 1}, {alias})"
                    )
                row_jump_blocked = 1
                row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
                selected_w_idx = int(expected_guard_idx)
                cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    # KR CV/CV_HEAD in order-locked formats: prevent occasional -1 backward drift
    # unless the backward index is clearly better.
    if (
        selected_w_idx is not None
        and alias_type in {"cv", "cv_head"}
        and str(file_format or "").strip().lower() in {"cvvc", "cvc", "vcv"}
        and _env_bool("UTOA_KR_CVVC_BACKWARD_ONE_STEP_GUARD", True)
        and romaji_syllables
    ):
        n_syl = int(len(romaji_syllables))
        expected_guard_idx = int(max(0, min(int(expected_cv_idx), n_syl - 1)))
        selected_guard_idx = int(max(0, min(int(selected_w_idx), n_syl - 1)))
        if selected_guard_idx == (expected_guard_idx - 1):
            exp_tok = str(romaji_syllables[expected_guard_idx] or "")
            sel_tok = str(romaji_syllables[selected_guard_idx] or "")
            exp_score = float(cv_match_score_fn(target_clean, exp_tok)) if target_clean else 0.0
            sel_score = float(cv_match_score_fn(target_clean, sel_tok)) if target_clean else 0.0
            exp_blank = _blank_conf_at(syllable_blank_confidences, expected_guard_idx)
            sel_blank = _blank_conf_at(syllable_blank_confidences, selected_guard_idx)
            t_on, t_v, _t_c = split_syllable_parts_fn(target_clean)
            e_on, e_v, _e_c = split_syllable_parts_fn(exp_tok)
            s_on, s_v, _s_c = split_syllable_parts_fn(sel_tok)
            keep_backward = False
            if (sel_score >= (exp_score + 16.0)) and (sel_blank <= (exp_blank - 0.05)):
                keep_backward = True
            if t_v and (s_v == t_v) and (e_v != t_v) and (sel_score >= (exp_score + 10.0)):
                keep_backward = True
            if t_on and (s_on == t_on) and (e_on != t_on) and (sel_score >= (exp_score + 10.0)):
                keep_backward = True
            if not keep_backward:
                if debug_logging:
                    log_fn(
                        f"[KR] {fname}: CV -1 backward guard "
                        f"({selected_guard_idx + 1}->{expected_guard_idx + 1}, {alias})"
                    )
                row_jump_blocked = 1
                row_mapping_confidence = apply_row_confidence_penalty_fn(row_mapping_confidence, 0.08)
                selected_w_idx = int(expected_guard_idx)
                cv_seq_idx = max(cv_seq_idx, selected_w_idx + 1)
    if (
        selected_w_idx is not None
        and alias_type in {"cv", "cv_head", "vcv"}
        and _env_bool("UTOA_WEAK_BOUNDARY_MISSING_REDUCTION_ENABLE", False)
    ):
        t_onset_now, _t_vowel_now, _t_coda_now = split_syllable_parts_fn(target_clean)
        if _is_weak_boundary_onset(t_onset_now):
            sel_blank_now = _blank_conf_at(syllable_blank_confidences, selected_w_idx)
            if sel_blank_now >= 0.60:
                fallback_idx = _find_nonblank_fallback_idx(
                    expected_idx=expected_cv_idx,
                    target_clean=target_clean,
                    romaji_syllables=romaji_syllables,
                    syllables_info=syllables_info,
                    syllable_blank_confidences=syllable_blank_confidences,
                    cv_match_score_fn=cv_match_score_fn,
                    search_back=1,
                    search_fwd=(1 if _env_bool("UTOA_WEAK_BOUNDARY_ANTIMISMAP_ENABLE", False) else 2),
                )
                if fallback_idx is not None and int(fallback_idx) != int(selected_w_idx):
                    if debug_logging:
                        log_fn(
                            f"[MAP] {fname}: KR weak-boundary fallback remap "
                            f"({selected_w_idx + 1}->{int(fallback_idx) + 1}, {alias})"
                        )
                    selected_w_idx = int(fallback_idx)
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
    file_format,
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
    forced_source = "planned"
    planned_cv_head_idx = resolve_planned_cv_index_fn(
        kr_planned_cv_indices,
        cv_seq_idx,
        target_clean,
        syllables_info,
        alias_type="cv_head",
    )
    forced_cvvc_idx = planned_cv_head_idx
    if forced_cvvc_idx is None:
        forced_source = "occurrence"
        try:
            forced_cvvc_idx = resolve_cvvc_occurrence_index_fn(
                alias,
                alias_type,
                kr_cvvc_occurrence_map or {},
                kr_cvvc_occurrence_state,
                expected_idx=cv_seq_idx,
            )
        except TypeError:
            forced_cvvc_idx = resolve_cvvc_occurrence_index_fn(
                alias,
                alias_type,
                kr_cvvc_occurrence_map or {},
                kr_cvvc_occurrence_state,
            )
    elif debug_logging and forced_cvvc_idx != cv_seq_idx:
        log_fn(
            f"[KR] {fname}: CV_HEAD anchor plan applied "
            f"({cv_seq_idx + 1}->{forced_cvvc_idx + 1}, {alias})"
        )

    if (
        forced_cvvc_idx is not None
        and str(file_format or "").strip().lower() in {"cvvc", "cvc", "vcv"}
        and _env_bool("UTOA_KR_CVVC_OCCURRENCE_STRICT_EXPECTED", True)
        and int(forced_cvvc_idx) > int(cv_seq_idx)
    ):
        if debug_logging:
            log_fn(
                f"[KR] {fname}: CV_HEAD occurrence forward clamp "
                f"({int(forced_cvvc_idx) + 1}->{int(cv_seq_idx) + 1}, {alias})"
            )
        forced_cvvc_idx = int(cv_seq_idx)

    if forced_cvvc_idx is not None and _plan_occ_double_guard_enabled():
        guard_ok, guard_reason = _guard_plan_occurrence_candidate_kr(
            target_clean=target_clean,
            expected_idx=cv_seq_idx,
            candidate_idx=int(forced_cvvc_idx),
            romaji_syllables=romaji_syllables,
            syllables_info=syllables_info,
        )
        if not guard_ok:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: CV_HEAD {forced_source} index rejected "
                    f"({cv_seq_idx + 1}->{int(forced_cvvc_idx) + 1}, {guard_reason}, {alias})"
                )
            forced_cvvc_idx = None

    if forced_cvvc_idx is not None and not (0 <= forced_cvvc_idx < len(romaji_syllables)):
        remapped_idx = remap_forced_cv_index_fn(
            target_clean,
            romaji_syllables,
            cv_seq_idx,
        )
        if remapped_idx is not None:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: CV_HEAD forced index remap "
                    f"({forced_cvvc_idx + 1}->{remapped_idx + 1}, {alias})"
                )
            forced_cvvc_idx = remapped_idx
        else:
            if debug_logging:
                log_fn(
                    f"[KR] {fname}: CV_HEAD forced index invalid "
                    f"(idx={forced_cvvc_idx + 1}, {alias})"
                )
            forced_cvvc_idx = None
    return forced_cvvc_idx

__all__ = [
    "resolve_kr_cv_head_forced_index",
    "select_kr_general_cv_index",
    "select_kr_vcv_index",
]

