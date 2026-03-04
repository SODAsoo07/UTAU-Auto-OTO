from __future__ import annotations

from core.kr_oto_rules import (
    KR_PLOSIVE_ONSETS,
    KR_SIBILANT_ONSETS,
    KR_SONORANT_CONSONANTS,
    KR_TENSE_CONSONANTS,
    KR_VOICED_ONSETS,
    KR_VOICELESS_ONSETS,
    _canonicalize_kr_coda,
    _extract_alias_onset,
    _extract_vc_right_token,
)


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _safe_ratio(num, den, fallback=0.0):
    if den is None or den == 0:
        return fallback
    return float(num) / float(den)


def _blend(a, b, w):
    w2 = max(0.0, min(1.0, float(w)))
    return (1.0 - w2) * float(a) + w2 * float(b)


def _validate_oto_params(offset, consonant, cutoff, pre, ovl):
    if offset < 0:
        offset = 0
    if pre < 0:
        pre = 0
    if ovl < 0:
        ovl = 0
    if consonant < 0:
        consonant = 0
    if ovl > pre:
        ovl = pre * 0.75
    if consonant < pre:
        consonant = pre + 30
    cutoff_abs = abs(cutoff)
    if cutoff_abs <= consonant:
        cutoff_abs = consonant + 50
    cutoff = -cutoff_abs
    return offset, consonant, cutoff, pre, ovl


def _get_kr_timing_traits_from_alias(alias):
    onset = _extract_alias_onset(alias or "")
    if not onset:
        return {"onset": "", "manner": "other", "voicing": "unknown"}
    c = onset.lower()
    manner = "other"
    if c in KR_SIBILANT_ONSETS:
        manner = "sibilant"
    elif c in KR_PLOSIVE_ONSETS:
        manner = "plosive"

    voicing = "unknown"
    if c in KR_VOICED_ONSETS:
        voicing = "voiced"
    elif c in KR_VOICELESS_ONSETS:
        voicing = "voiceless"
    return {"onset": c, "manner": manner, "voicing": voicing}


def _apply_kr_consonant_timing_shaping(alias, pre, cons, cutoff, ovl):
    traits = _get_kr_timing_traits_from_alias(alias)
    onset = traits["onset"]
    manner = traits["manner"]
    voicing = traits["voicing"]
    coda_canon = _canonicalize_kr_coda(_extract_vc_right_token(alias))
    is_tense_onset = onset in KR_TENSE_CONSONANTS
    is_stop_coda_alias = coda_canon in {"k", "t", "p"}

    pre_mul = 1.0
    cons_gap_mul = 1.0
    cut_gap_mul = 1.0
    ovl_bias = 0.0

    if manner == "plosive":
        pre_mul *= 0.94
        cons_gap_mul *= 0.92
        cut_gap_mul *= 0.94
        ovl_bias -= 0.05
    elif manner == "sibilant":
        pre_mul *= 1.08
        cons_gap_mul *= 1.04
        cut_gap_mul *= 1.03
        ovl_bias += 0.06

    if voicing == "voiced":
        pre_mul *= 1.04
        cons_gap_mul *= 1.05
        cut_gap_mul *= 1.02
        ovl_bias += 0.02
    elif voicing == "voiceless":
        pre_mul *= 0.96
        cons_gap_mul *= 0.95
        cut_gap_mul *= 0.97
        ovl_bias -= 0.02

    if is_tense_onset:
        pre_mul *= 1.08
        cons_gap_mul *= 1.05
        cut_gap_mul *= 0.90
        ovl_bias -= 0.04

    if is_stop_coda_alias:
        if coda_canon in {"t", "p"}:
            cons_gap_mul *= 0.86
            cut_gap_mul *= 0.72
        else:
            cons_gap_mul *= 0.92
            cut_gap_mul *= 0.80
        ovl_bias -= 0.03

    pre_new = _clamp(pre * pre_mul, 20.0, 360.0)
    cons_gap_now = max(cons - pre, 10.0)
    cons_gap_new = _clamp(cons_gap_now * cons_gap_mul, 8.0, 260.0)
    cons_new = pre_new + cons_gap_new

    ovl_ratio_now = _safe_ratio(ovl, pre, fallback=0.30)
    ovl_ratio_new = _clamp(ovl_ratio_now + ovl_bias, 0.08, 0.80)
    ovl_new = max(0.0, pre_new * ovl_ratio_new)

    cut_gap_now = max(abs(cutoff) - cons, 20.0)
    cut_gap_new = _clamp(cut_gap_now * cut_gap_mul, 18.0, 220.0)
    cutoff_new = -(cons_new + cut_gap_new)
    _, cons_new, cutoff_new, pre_new, ovl_new = _validate_oto_params(
        0.0, cons_new, cutoff_new, pre_new, ovl_new
    )
    return pre_new, cons_new, cutoff_new, ovl_new


def _kr_precenter_gap_targets(alias_type, alias_text=""):
    alias_type = str(alias_type or "cv").strip().lower()
    if alias_type == "vv":
        return {
            "ovl_gap": (86.0, 148.0, 112.0),
            "cons_gap": (54.0, 138.0, 84.0),
            "cut_gap": (12.0, 30.0, 18.0),
        }

    if alias_type in {"vc", "vv"}:
        coda = _canonicalize_kr_coda(_extract_vc_right_token(alias_text))
        if coda in {"k", "t", "p", "h"}:
            return {
                "ovl_gap": (78.0, 128.0, 102.0),
                "cons_gap": (26.0, 76.0, 48.0),
                "cut_gap": (8.0, 20.0, 12.0),
            }
        if coda in {"n", "m", "ng", "l"}:
            return {
                "ovl_gap": (92.0, 152.0, 118.0),
                "cons_gap": (40.0, 118.0, 72.0),
                "cut_gap": (10.0, 26.0, 16.0),
            }
        return {
            "ovl_gap": (88.0, 142.0, 110.0),
            "cons_gap": (34.0, 102.0, 62.0),
            "cut_gap": (10.0, 24.0, 14.0),
        }

    traits = _get_kr_timing_traits_from_alias(alias_text)
    onset = traits["onset"]
    manner = traits["manner"]
    is_tense = onset in KR_TENSE_CONSONANTS
    is_sonorant = onset in KR_SONORANT_CONSONANTS

    if is_sonorant:
        return {
            "ovl_gap": (12.0, 28.0, 18.0),
            "cons_gap": (80.0, 210.0, 124.0),
            "cut_gap": (48.0, 130.0, 78.0),
        }
    if manner == "sibilant":
        return {
            "ovl_gap": (16.0, 34.0, 24.0),
            "cons_gap": (72.0, 178.0, 110.0),
            "cut_gap": (42.0, 112.0, 64.0),
        }
    if manner == "plosive" or is_tense:
        return {
            "ovl_gap": (22.0, 42.0, 31.0 if is_tense else 29.0),
            "cons_gap": (56.0, 152.0, 90.0),
            "cut_gap": (34.0, 96.0, 54.0),
        }
    return {
        "ovl_gap": (18.0, 36.0, 26.0),
        "cons_gap": (64.0, 164.0, 100.0),
        "cut_gap": (40.0, 110.0, 62.0),
    }


def _recenter_kr_params_around_pre(offset, consonant, cutoff, pre, ovl, alias_type="cv", alias_text=""):
    targets = _kr_precenter_gap_targets(alias_type, alias_text)
    pre_v = max(float(pre), 0.0)
    if pre_v <= 0.0:
        return _validate_oto_params(offset, consonant, cutoff, pre, ovl)

    alias_type = str(alias_type or "cv").strip().lower()
    if alias_type == "vv":
        ovl_w, cons_w, cut_w = 0.18, 0.12, 0.08
    elif alias_type in {"vc", "vcv"}:
        ovl_w, cons_w, cut_w = 0.16, 0.10, 0.06
    else:
        ovl_w, cons_w, cut_w = 0.22, 0.20, 0.18

    ovl_gap_now = max(pre_v - max(float(ovl), 0.0), 0.0)
    cons_gap_now = max(float(consonant) - pre_v, 8.0)
    cut_gap_now = max(abs(float(cutoff)) - float(consonant), 12.0)

    og_lo, og_hi, og_t = targets["ovl_gap"]
    cg_lo, cg_hi, cg_t = targets["cons_gap"]
    tg_lo, tg_hi, tg_t = targets["cut_gap"]

    ovl_gap_new = max(og_lo, min(og_hi, ovl_gap_now))
    ovl_gap_new = _blend(ovl_gap_new, og_t, ovl_w)

    cons_gap_new = max(cg_lo, min(cg_hi, cons_gap_now))
    cons_gap_new = _blend(cons_gap_new, cg_t, cons_w)

    cut_gap_new = max(tg_lo, min(tg_hi, cut_gap_now))
    cut_gap_new = _blend(cut_gap_new, tg_t, cut_w)

    ovl_new = max(0.0, pre_v - ovl_gap_new)
    cons_new = pre_v + cons_gap_new
    cutoff_new = -(cons_new + cut_gap_new)
    return _validate_oto_params(offset, cons_new, cutoff_new, pre_v, ovl_new)


def _compute_vc_from_adjacent_cv(prev_cv, next_cv, alias_type, is_plosive_sibilant):
    if not prev_cv or not next_cv:
        return None

    boundary_abs = next_cv["onset_abs"] if alias_type == "vc" else next_cv["pre_abs"]
    transition_len = max(boundary_abs - prev_cv["vowel_end_abs"], 14.0)
    if alias_type == "vc":
        if is_plosive_sibilant:
            pre_floor, pre_ceil = 34.0, 108.0
            pre_target = _clamp(transition_len * 0.82, pre_floor, pre_ceil)
        else:
            pre_floor, pre_ceil = 42.0, 132.0
            pre_target = _clamp(transition_len * 0.92, pre_floor, pre_ceil)
        pre_target = _blend(pre_target, min(next_cv["pre"], pre_ceil), 0.22)
    else:
        pre_target = _clamp(_blend(prev_cv["pre"], next_cv["pre"], 0.28), 40.0, 180.0)
    offset = max(boundary_abs - pre_target, 0.0)
    pre = boundary_abs - offset
    if pre <= 0:
        return None

    ovl_tail = _clamp(prev_cv["vowel_len"] * (0.07 if alias_type == "vc" else 0.10), 3.0, 18.0 if alias_type == "vc" else 22.0)
    target_ovl_abs = prev_cv["vowel_end_abs"] - ovl_tail
    ovl = target_ovl_abs - offset
    if is_plosive_sibilant:
        ovl = _clamp(ovl, max(pre - 18.0, pre * 0.62), max(pre - 8.0, 0.0))
    else:
        ovl = _clamp(ovl, max(pre - 14.0, pre * 0.74), max(pre - 6.0, 0.0))

    cons_gap = _clamp(_blend(prev_cv["cons_gap"], next_cv["cons_gap"], 0.32), 14.0, 92.0 if alias_type == "vc" else 120.0)
    consonant = pre + cons_gap
    next_onset_rel = max(next_cv["onset_abs"] - offset, pre + 10.0)
    next_pre_rel = max(next_cv["pre_abs"] - offset, pre + 16.0)
    next_cons_rel = max(next_cv["cons_abs"] - offset, next_pre_rel + 10.0)

    if alias_type == "vc":
        if is_plosive_sibilant:
            consonant = min(consonant, next_onset_rel - 6.0)
            consonant = max(consonant, pre + 12.0)
            cutoff_abs = max(consonant + 10.0, next_onset_rel - 2.0)
            cutoff_abs = min(cutoff_abs, next_onset_rel + 8.0)
        else:
            consonant = min(consonant, next_onset_rel + 16.0)
            consonant = max(consonant, pre + 16.0)
            cutoff_abs = max(consonant + 12.0, min(next_cons_rel + 18.0, next_pre_rel + 30.0))
    else:
        consonant = min(max(consonant, pre + 24.0), next_pre_rel + 48.0)
        cutoff_abs = max(consonant + 20.0, next_pre_rel + 10.0)
        cutoff_abs = min(cutoff_abs, next_cons_rel + 60.0)

    cutoff = -cutoff_abs
    return _validate_oto_params(offset, consonant, cutoff, pre, ovl)


def _refine_kr_bridge_with_adjacent_cv(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    alias_type,
    alias_text="",
    prev_cv=None,
    next_cv=None,
):
    """
    VC/VV를 인접 CV 앵커 기준으로 2차 보정합니다.
    - 기존 계산을 유지하되 인접 CV 기반 후보와 블렌딩해서 연결감을 안정화
    - 다음 onset을 넘는 cutoff 과연장을 억제
    """
    a_type = str(alias_type or "").strip().lower()
    if a_type not in {"vc", "vv"}:
        return _validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if not prev_cv or not next_cv:
        return _validate_oto_params(offset, consonant, cutoff, pre, ovl)

    coda = _canonicalize_kr_coda(_extract_vc_right_token(alias_text))
    is_stop = coda in {"k", "t", "p", "h"}
    is_sonorant = coda in {"n", "m", "ng", "l", "r"}

    is_plosive_sibilant = is_stop
    cand = _compute_vc_from_adjacent_cv(prev_cv, next_cv, a_type, is_plosive_sibilant)
    if cand is None:
        return _validate_oto_params(offset, consonant, cutoff, pre, ovl)
    c_off, c_cons, c_cut, c_pre, c_ovl = cand

    if a_type == "vc":
        if is_stop:
            w_anchor, w_shape = 0.64, 0.60
            pre_lo, pre_hi = 32.0, 154.0
            ovl_gap_lo, ovl_gap_hi = 10.0, 24.0
            cons_gap_lo, cons_gap_hi = 14.0, 56.0
            cut_gap_lo, cut_gap_hi = 8.0, 24.0
            cut_allow_ms = 7.0
        elif is_sonorant:
            w_anchor, w_shape = 0.52, 0.48
            pre_lo, pre_hi = 46.0, 196.0
            ovl_gap_lo, ovl_gap_hi = 7.0, 20.0
            cons_gap_lo, cons_gap_hi = 26.0, 86.0
            cut_gap_lo, cut_gap_hi = 12.0, 36.0
            cut_allow_ms = 22.0
        else:
            w_anchor, w_shape = 0.56, 0.52
            pre_lo, pre_hi = 38.0, 176.0
            ovl_gap_lo, ovl_gap_hi = 8.0, 22.0
            cons_gap_lo, cons_gap_hi = 20.0, 72.0
            cut_gap_lo, cut_gap_hi = 10.0, 30.0
            cut_allow_ms = 16.0
    else:
        w_anchor, w_shape = 0.46, 0.40
        pre_lo, pre_hi = 30.0, 186.0
        ovl_gap_lo, ovl_gap_hi = 4.0, 12.0
        cons_gap_lo, cons_gap_hi = 58.0, 156.0
        cut_gap_lo, cut_gap_hi = 18.0, 98.0
        cut_allow_ms = 34.0

    curr_pre_abs = float(offset) + float(pre)
    cand_pre_abs = float(c_off) + float(c_pre)
    pre_abs_new = _blend(curr_pre_abs, cand_pre_abs, w_anchor)

    pre_new = _blend(float(pre), float(c_pre), w_shape)
    pre_new = _clamp(pre_new, pre_lo, pre_hi)
    offset_new = max(pre_abs_new - pre_new, 0.0)

    curr_ovl_gap = max(float(pre) - float(ovl), 0.0)
    cand_ovl_gap = max(float(c_pre) - float(c_ovl), 0.0)
    ovl_gap_new = _blend(curr_ovl_gap, cand_ovl_gap, w_shape)
    ovl_gap_new = _clamp(ovl_gap_new, ovl_gap_lo, ovl_gap_hi)
    ovl_new = max(0.0, pre_new - ovl_gap_new)

    curr_cons_gap = max(float(consonant) - float(pre), 8.0)
    cand_cons_gap = max(float(c_cons) - float(c_pre), 8.0)
    cons_gap_new = _blend(curr_cons_gap, cand_cons_gap, w_shape)
    cons_gap_new = _clamp(cons_gap_new, cons_gap_lo, cons_gap_hi)
    cons_new = pre_new + cons_gap_new

    curr_cut_gap = max(abs(float(cutoff)) - float(consonant), 10.0)
    cand_cut_gap = max(abs(float(c_cut)) - float(c_cons), 10.0)
    cut_gap_new = _blend(curr_cut_gap, cand_cut_gap, w_shape)
    cut_gap_new = _clamp(cut_gap_new, cut_gap_lo, cut_gap_hi)
    cutoff_abs_new = cons_new + cut_gap_new

    next_onset_abs = float(next_cv.get("onset_abs", 0.0) or 0.0)
    next_pre_abs = float(next_cv.get("pre_abs", next_onset_abs) or next_onset_abs)
    if next_onset_abs > 0.0:
        next_onset_rel = max(next_onset_abs - offset_new, pre_new + 10.0)
        next_pre_rel = max(next_pre_abs - offset_new, next_onset_rel)
        cap = min(next_onset_rel + cut_allow_ms, next_pre_rel + max(8.0, cut_allow_ms * 0.75))
        cutoff_abs_new = min(cutoff_abs_new, max(cons_new + cut_gap_lo, cap))

    cutoff_new = -cutoff_abs_new
    return _validate_oto_params(offset_new, cons_new, cutoff_new, pre_new, ovl_new)


def _compute_kr_cvvc_vc_timing_direct(alias, *args):
    """
    CVVC VC direct timing helper.

    Backward-compatible call forms:
    - _compute_kr_cvvc_vc_timing_direct(alias, c_start, c_end, n_start, n_end)
    - _compute_kr_cvvc_vc_timing_direct(alias, alias_type, c_start, c_end, n_start, n_end)
    """
    if len(args) == 4:
        c_start, c_end, n_start, n_end = args
    elif len(args) == 5:
        _alias_type, c_start, c_end, n_start, n_end = args
    else:
        raise TypeError(
            "_compute_kr_cvvc_vc_timing_direct() expects 5 or 6 total positional arguments"
        )

    if not alias:
        return None

    prev_vowel_len = max(float(c_end) - float(c_start), 24.0)
    next_seg_len = max(float(n_end) - float(n_start), 18.0)
    transition_gap = max(float(n_start) - float(c_end), 10.0)

    right_tok = _extract_vc_right_token(alias)
    coda_canon = _canonicalize_kr_coda(right_tok)
    is_nasal = coda_canon in {"m", "n", "ng"}
    is_liquid = coda_canon in {"l", "r"}
    is_hard = right_tok in {
        "g", "k", "kk", "gg", "d", "t", "tt", "dd", "b", "p", "bb", "pp",
        "s", "ss", "h", "j", "jj", "ch",
    }

    bridge_abs = float(n_start)
    if is_hard:
        tail_keep = _clamp(prev_vowel_len * 0.42, 56.0, 118.0)
        ovl_gap = _clamp(prev_vowel_len * 0.52, 78.0, 122.0)
        cons_gap = _clamp(next_seg_len * 0.44, 34.0, 74.0)
        cut_gap = _clamp(next_seg_len * 0.58, 30.0, 64.0)
        cutoff_margin = 16.0
    elif is_nasal:
        tail_keep = _clamp(prev_vowel_len * 0.54, 82.0, 142.0)
        ovl_gap = _clamp(prev_vowel_len * 0.58, 92.0, 138.0)
        cons_gap = _clamp(next_seg_len * 0.52, 42.0, 88.0)
        cut_gap = _clamp(next_seg_len * 0.72, 38.0, 86.0)
        cutoff_margin = 26.0
    elif is_liquid:
        tail_keep = _clamp(prev_vowel_len * 0.62, 96.0, 166.0)
        ovl_gap = _clamp(prev_vowel_len * 0.64, 104.0, 152.0)
        cons_gap = _clamp(next_seg_len * 0.96, 82.0, 158.0)
        cut_gap = _clamp(next_seg_len * 0.88, 52.0, 118.0)
        cutoff_margin = 42.0
    else:
        tail_keep = _clamp(prev_vowel_len * 0.50, 72.0, 136.0)
        ovl_gap = _clamp(prev_vowel_len * 0.56, 88.0, 132.0)
        cons_gap = _clamp(next_seg_len * 0.58, 40.0, 96.0)
        cut_gap = _clamp(next_seg_len * 0.74, 36.0, 92.0)
        cutoff_margin = 28.0
    offset = max(float(c_end) - tail_keep, 0.0)
    pre_floor = 112.0 if is_hard else (168.0 if is_liquid else 138.0 if is_nasal else 128.0)
    pre_ceil = 214.0 if is_hard else (286.0 if is_liquid else 248.0 if is_nasal else 232.0)
    pre_target = _clamp(transition_gap + tail_keep, pre_floor, pre_ceil)
    pre = max(bridge_abs - offset, pre_target)
    ovl = max(0.0, pre - ovl_gap)
    consonant = pre + cons_gap
    cutoff_abs = consonant + cut_gap
    next_onset_rel = max(float(n_start) - offset, pre + 12.0)
    next_seg_end_rel = max(float(n_end) - offset, next_onset_rel + 8.0)
    consonant = min(consonant, next_onset_rel + (8.0 if is_hard else 22.0 if is_nasal else 42.0 if is_liquid else 28.0))
    min_cons_gap = 28.0 if is_liquid else 18.0 if is_nasal else 16.0
    consonant = max(consonant, pre + min_cons_gap)
    cutoff_cap = min(
        next_onset_rel + cutoff_margin,
        next_seg_end_rel - (8.0 if is_hard else 10.0 if is_nasal else 14.0 if is_liquid else 12.0),
    )
    min_cut_tail = 8.0 if is_hard else 10.0 if is_nasal else 14.0 if is_liquid else 12.0
    if cutoff_cap <= consonant + min_cut_tail:
        consonant = max(pre + min_cons_gap, cutoff_cap - min_cut_tail)
    cutoff_abs = min(max(cutoff_abs, consonant + min_cut_tail), cutoff_cap)

    cutoff = -cutoff_abs
    return _validate_oto_params(offset, consonant, cutoff, pre, ovl)


__all__ = [
    "_apply_kr_consonant_timing_shaping",
    "_compute_kr_cvvc_vc_timing_direct",
    "_compute_vc_from_adjacent_cv",
    "_refine_kr_bridge_with_adjacent_cv",
    "_recenter_kr_params_around_pre",
]
