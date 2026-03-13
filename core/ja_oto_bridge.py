"""
Japanese bridge / CV timing helpers.

This module isolates overlap shaping, pre-centered recentering, and
CV boundary extraction from the main generator.
"""

from __future__ import annotations

import re

from core.ja_lab_generator import split_ja_romaji_syllable
from core.ja_oto_mapping import (
    JA_NASAL_ONSETS,
    JA_VOICED_ONSETS,
    JA_VOICELESS_ONSETS,
    _clean_phone_mark,
    _extract_ja_cv_target_syllable,
    _extract_vcv_target_syllable,
    get_vc_consonant,
)

JA_SIBILANT_ONSETS = {
    "s", "z", "sh", "j", "ch", "ts", "dz", "ss", "zz", "jj",
}
JA_PLOSIVE_ONSETS = {
    "k", "g", "t", "d", "b", "p", "q", "c",
    "kk", "tt", "pp", "dd", "gg", "bb",
    "ky", "gy", "ty", "dy", "by", "py",
}
JA_LIQUID_ONSETS = {"r", "ry", "l"}
JA_GLIDE_ONSETS = {"y", "w"}
JA_FRICATIVE_ONSETS = {"h", "f", "v", "hy", "s", "z", "sh"}


def _ja_bridge_overlap_window(pre, consonant_hint="", mode="vc"):
    p = max(float(pre), 0.0)
    c = _clean_phone_mark(consonant_hint)
    hard = {"k", "g", "t", "d", "p", "b", "q", "c", "ch", "ts", "dz", "ky", "gy", "ty", "dy", "py", "by"}
    fric = {"s", "z", "sh", "h", "f", "v", "hy"}
    sonorant = {"m", "n", "ny", "r", "l", "ry", "w", "y"}

    mode = str(mode or "vc").strip().lower()
    if mode == "vv":
        gap_min, gap_max, gap_target = 4.0, 10.0, 7.0
    elif c in hard or c in fric:
        if mode == "vcv":
            gap_min, gap_max, gap_target = 8.0, 16.0, 11.0
        else:
            gap_min, gap_max, gap_target = 10.0, 20.0, 14.0
    elif c in sonorant:
        if mode == "vcv":
            gap_min, gap_max, gap_target = 5.0, 12.0, 8.0
        else:
            gap_min, gap_max, gap_target = 6.0, 14.0, 9.0
    else:
        if mode == "vcv":
            gap_min, gap_max, gap_target = 6.0, 13.0, 9.0
        else:
            gap_min, gap_max, gap_target = 7.0, 15.0, 10.0

    gap_target = _clamp_range(gap_target, gap_min, gap_max)
    if p <= 0:
        return gap_min, gap_max, gap_target
    gap_max = min(gap_max, max(p - 1.0, gap_min))
    gap_min = min(gap_min, gap_max)
    gap_target = _clamp_range(gap_target, gap_min, gap_max)
    return gap_min, gap_max, gap_target


def _clamp_ja_bridge_overlap(pre, ovl, consonant_hint="", mode="vc"):
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0
    gap_min, gap_max, gap_target = _ja_bridge_overlap_window(p, consonant_hint, mode=mode)
    lower = max(p - gap_max, 0.0)
    upper = max(p - gap_min, 0.0)
    if upper < lower:
        upper = lower
    base = float(ovl) if ovl is not None else (p - gap_target)
    return max(0.0, min(p, min(upper, max(lower, base))))


def _ja_overlap_consonant_hint(alias_text, alias_type="vc"):
    alias_type = str(alias_type or "").strip().lower()
    if alias_type == "vc":
        return get_vc_consonant(alias_text)
    if alias_type == "vcv":
        target = _extract_vcv_target_syllable(alias_text)
        onset, _v = split_ja_romaji_syllable(target)
        return (onset or "").strip().lower()
    return ""


def _ja_precenter_gap_targets(alias_type, alias_text=""):
    alias_type = str(alias_type or "cv").strip().lower()
    if alias_type in {"vc", "vv", "vcv"}:
        if alias_type == "vv":
            return {
                "ovl_gap": (4.0, 10.0, 7.0),
                "cons_gap": (48.0, 132.0, 80.0),
                "cut_gap": (24.0, 96.0, 48.0),
            }

        hint = _ja_overlap_consonant_hint(alias_text, alias_type=alias_type)
        og_lo, og_hi, og_t = _ja_bridge_overlap_window(120.0, hint, mode=("vcv" if alias_type == "vcv" else "vc"))
        if hint in JA_PLOSIVE_ONSETS or hint in JA_SIBILANT_ONSETS or hint in JA_FRICATIVE_ONSETS:
            cons_gap = (20.0, 60.0, 34.0) if alias_type == "vc" else (58.0, 150.0, 92.0)
            cut_gap = (12.0, 46.0, 25.0) if alias_type == "vc" else (34.0, 92.0, 54.0)
        elif hint in JA_NASAL_ONSETS or hint in JA_LIQUID_ONSETS or hint in JA_GLIDE_ONSETS:
            cons_gap = (28.0, 96.0, 52.0) if alias_type == "vc" else (74.0, 180.0, 116.0)
            cut_gap = (18.0, 74.0, 40.0) if alias_type == "vc" else (42.0, 122.0, 70.0)
        else:
            cons_gap = (24.0, 82.0, 44.0) if alias_type == "vc" else (66.0, 164.0, 102.0)
            cut_gap = (16.0, 64.0, 34.0) if alias_type == "vc" else (38.0, 108.0, 62.0)
        return {
            "ovl_gap": (og_lo, og_hi, og_t),
            "cons_gap": cons_gap,
            "cut_gap": cut_gap,
        }

    if alias_type in {"cv", "cv_head"}:
        target = _extract_ja_cv_target_syllable(alias_text, alias_type=alias_type)
        onset, _vowel = split_ja_romaji_syllable(target)
        onset = (onset or "").strip().lower()
        if onset in JA_NASAL_ONSETS or onset in JA_LIQUID_ONSETS or onset in JA_GLIDE_ONSETS:
            return {
                "ovl_gap": (12.0, 28.0, 18.0),
                "cons_gap": (82.0, 192.0, 124.0),
                "cut_gap": (48.0, 128.0, 78.0),
            }
        if onset in JA_PLOSIVE_ONSETS:
            return {
                "ovl_gap": (22.0, 40.0, 29.0),
                "cons_gap": (56.0, 148.0, 88.0),
                "cut_gap": (34.0, 94.0, 54.0),
            }
        if onset in JA_SIBILANT_ONSETS or onset in JA_FRICATIVE_ONSETS:
            return {
                "ovl_gap": (16.0, 34.0, 24.0),
                "cons_gap": (72.0, 176.0, 110.0),
                "cut_gap": (42.0, 110.0, 64.0),
            }
        return {
            "ovl_gap": (18.0, 36.0, 26.0),
            "cons_gap": (64.0, 160.0, 98.0),
            "cut_gap": (40.0, 108.0, 62.0),
        }

    return {
        "ovl_gap": (18.0, 34.0, 24.0),
        "cons_gap": (48.0, 132.0, 80.0),
        "cut_gap": (28.0, 88.0, 48.0),
    }


def _recenter_ja_params_around_pre(offset, consonant, cutoff, pre, ovl, alias_type="cv", alias_text=""):
    from core.ja_oto_generator import validate_oto_params

    targets = _ja_precenter_gap_targets(alias_type, alias_text)
    pre_v = max(float(pre), 0.0)
    if pre_v <= 0.0:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    alias_type = str(alias_type or "cv").strip().lower()
    if alias_type == "vv":
        ovl_w, cons_w, cut_w = 0.50, 0.26, 0.22
    elif alias_type in {"vc", "vcv"}:
        ovl_w, cons_w, cut_w = 0.42, 0.28, 0.24
    else:
        ovl_w, cons_w, cut_w = 0.22, 0.20, 0.18

    ovl_gap_now = max(pre_v - max(float(ovl), 0.0), 0.0)
    cons_gap_now = max(float(consonant) - pre_v, 8.0)
    cut_gap_now = max(abs(float(cutoff)) - float(consonant), 12.0)

    og_lo, og_hi, og_t = targets["ovl_gap"]
    cg_lo, cg_hi, cg_t = targets["cons_gap"]
    tg_lo, tg_hi, tg_t = targets["cut_gap"]

    ovl_gap_new = _clamp_range(ovl_gap_now, og_lo, og_hi)
    ovl_gap_new = _blend(ovl_gap_new, og_t, ovl_w)

    cons_gap_new = _clamp_range(cons_gap_now, cg_lo, cg_hi)
    cons_gap_new = _blend(cons_gap_new, cg_t, cons_w)

    cut_gap_new = _clamp_range(cut_gap_now, tg_lo, tg_hi)
    cut_gap_new = _blend(cut_gap_new, tg_t, cut_w)

    ovl_new = max(0.0, pre_v - ovl_gap_new)
    cons_new = pre_v + cons_gap_new
    cutoff_new = -(cons_new + cut_gap_new)
    return validate_oto_params(offset, cons_new, cutoff_new, pre_v, ovl_new)


def _apply_vc_vv_mel_cutoff_cap(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    *,
    alias_type,
    format_type="",
    mel_cutoff_candidate_ms=None,
    next_mel_voiced_onset_ms=None,
):
    a_type = str(alias_type or "").strip().lower()
    if a_type not in {"vc", "vv"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    max_cut_allow = None
    try:
        from core.timing_anchor_profiles import get_anchor_profile

        prof = get_anchor_profile("japanese", str(format_type or "").strip().lower(), a_type)
        if prof is not None:
            max_cut_allow = prof.max_cut_to_next_voiced_onset_ms
    except Exception:
        max_cut_allow = None
    if max_cut_allow is None:
        max_cut_allow = -6.0

    max_cut_abs = None
    if next_mel_voiced_onset_ms is not None:
        try:
            max_cut_abs = float(next_mel_voiced_onset_ms) + float(max_cut_allow)
        except Exception:
            max_cut_abs = None
    if mel_cutoff_candidate_ms is not None:
        try:
            cand = float(mel_cutoff_candidate_ms)
            max_cut_abs = cand if max_cut_abs is None else min(max_cut_abs, cand)
        except Exception:
            pass

    if max_cut_abs is None:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    max_cut_rel = max_cut_abs - float(offset)
    min_cut_rel = float(consonant) + 20.0
    if max_cut_rel < min_cut_rel:
        max_cut_rel = min_cut_rel
    cut_abs = abs(float(cutoff))
    if cut_abs > max_cut_rel:
        cut_abs = max_cut_rel
        cutoff = -cut_abs
        if float(consonant) > cut_abs - 6.0:
            consonant = max(float(pre) + 8.0, cut_abs - 6.0)
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def _adaptive_ja_overlap(pre, consonant_hint="", mode="cv"):
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0

    c = _clean_phone_mark(consonant_hint)
    hard = {"k", "g", "t", "d", "p", "b", "q", "c", "ch", "ts", "dz", "ky", "gy", "ty", "dy", "py", "by"}
    fric = {"s", "z", "sh", "h", "f", "v", "hy"}
    sonorant = {"m", "n", "ny", "r", "l", "ry", "w", "y"}

    base_by_mode = {
        "cv": 0.46,
        "cv_head": 0.40,
        "vcv": 0.50,
        "vc": 0.44,
        "vv": 0.55,
    }
    ratio = base_by_mode.get(mode, 0.46)

    if c in hard:
        ratio -= 0.16
    elif c in fric:
        ratio += 0.05
    elif c in sonorant:
        ratio += 0.08

    if mode in ("vc", "vv", "vcv"):
        _gap_min, _gap_max, gap_target = _ja_bridge_overlap_window(p, c, mode=mode)
        return _clamp_ja_bridge_overlap(p, p - gap_target, c, mode=mode)

    ratio = max(0.20, min(0.72, ratio))
    ovl = p * ratio
    return max(0.0, min(ovl, p))


def _ja_onset_class(onset):
    o = (onset or "").strip().lower()
    if not o:
        return "other"
    if o in JA_NASAL_ONSETS or o.startswith("m"):
        return "nasal"
    if o in JA_VOICED_ONSETS:
        return "voiced"
    if o in JA_VOICELESS_ONSETS:
        return "voiceless"
    return "other"


def _ja_extract_onset_for_timing(alias_text, alias_type="cv"):
    alias_type = str(alias_type or "cv").strip().lower()
    if alias_type == "vc":
        return get_vc_consonant(alias_text)
    if alias_type == "vcv":
        target = _extract_vcv_target_syllable(alias_text)
        onset, _v = split_ja_romaji_syllable(target)
        return (onset or "").strip().lower()
    target = _extract_ja_cv_target_syllable(alias_text, alias_type=alias_type)
    onset, _v = split_ja_romaji_syllable(target)
    return (onset or "").strip().lower()


def _ja_cv_onset_class(alias_text, c_hint="", alias_type="cv"):
    target_tok = _extract_ja_cv_target_syllable(alias_text, alias_type=alias_type)
    onset = _ja_extract_onset_for_timing(target_tok, alias_type="cv")
    if not onset:
        onset = _ja_extract_onset_for_timing(alias_text, alias_type=alias_type)
    if not onset:
        hint = re.sub(r"[^a-z]", "", (c_hint or "").lower())
        onset = hint
    return _ja_onset_class(onset), onset


def _ja_cv_offset_and_pre(
    c_start,
    c_end,
    alias_text,
    c_hint="",
    alias_type="cv",
    vowel_start=None,
    vowel_end=None,
):
    cls, onset = _ja_cv_onset_class(alias_text, c_hint=c_hint, alias_type=alias_type)
    is_head = alias_type == "cv_head"

    lead = 30.0
    if cls == "nasal":
        lead = 18.0
    elif cls == "voiced":
        lead = 24.0
    elif cls == "voiceless":
        lead = 34.0
    if onset in JA_SIBILANT_ONSETS or onset in JA_FRICATIVE_ONSETS:
        lead += 6.0
    if onset in {"m", "n", "ny", "r", "l", "ry"}:
        lead -= 4.0
    if is_head:
        lead -= 4.0
    lead = _clamp_range(lead, 12.0, 50.0)

    c_anchor = float(c_end)
    late_shift = 0.0
    if vowel_start is not None and vowel_end is not None:
        v_s = float(vowel_start)
        v_e = float(vowel_end)
        v_len = max(v_e - v_s, 0.0)
        c_len = max(float(c_end) - float(c_start), 0.0)
        if (
            alias_type == "cv"
            and v_len >= 620.0
            and c_len <= 45.0
            and onset in {"t", "d", "s", "z", "ts", "dz", "ch", "j"}
        ):
            stable = v_s + _clamp_range(v_len * 0.45, 120.0, 420.0)
            late_shift = max(0.0, stable - c_anchor)

    offset = max(float(c_start) - lead, 0.0)
    pre = c_anchor - offset if c_anchor > c_start else (28.0 if is_head else 24.0)

    if c_anchor > c_start:
        c_len = max(c_anchor - float(c_start), 8.0)
        min_pre = max(20.0, c_len + 8.0)
        if cls in {"nasal", "voiced"}:
            max_pre = c_len + 64.0
        elif cls == "voiceless":
            max_pre = c_len + 54.0
        else:
            max_pre = c_len + 58.0
        if is_head:
            max_pre += 10.0
        pre = _clamp_range(pre, min_pre, max_pre)
        offset = max(c_anchor - pre, 0.0)

    if late_shift > 0.0:
        offset = max(offset + late_shift, 0.0)

    return offset, pre


def _ja_mark_to_vowel(mark):
    clean = re.sub(r"[0-9]", "", (mark or "").strip().lower())
    if not clean:
        return ""
    if clean in {"a", "i", "e", "o"}:
        return clean
    if clean in {"u", "ɯ", "ʉ"}:
        return "u"
    target = _extract_ja_cv_target_syllable(clean, alias_type="cv")
    onset, vowel = split_ja_romaji_syllable(target)
    if vowel in {"a", "i", "u", "e", "o"}:
        return vowel
    return clean if clean in {"a", "i", "u", "e", "o"} else ""


def _ja_target_vowel_from_alias(alias_text, alias_type="cv"):
    tok = _extract_ja_cv_target_syllable(alias_text, alias_type=alias_type)
    if not tok:
        return ""
    onset, vowel = split_ja_romaji_syllable(tok)
    if vowel in {"a", "i", "u", "e", "o"}:
        return vowel
    if tok in {"a", "i", "u", "e", "o"} and not onset:
        return tok
    return ""


def _ja_pick_vowel_phone(curr_phones, target_vowel=""):
    candidates = []
    for i, ph in enumerate(curr_phones or []):
        v = _ja_mark_to_vowel(getattr(ph, "mark", ""))
        if v:
            candidates.append((i, ph, v))
    if not candidates:
        return None, -1
    if target_vowel:
        for i, ph, v in candidates:
            if v == target_vowel:
                return ph, i
    i, ph, _v = candidates[0]
    return ph, i


def _ja_extract_cv_bounds(curr_phones, alias_text="", alias_type="cv"):
    phones = curr_phones or []
    if not phones:
        return 0.0, 0.0, 0.0, 0.0

    target_vowel = _ja_target_vowel_from_alias(alias_text, alias_type=alias_type)
    v_phone, v_idx = _ja_pick_vowel_phone(phones, target_vowel=target_vowel)
    c_start = float(phones[0].minTime) * 1000.0

    if v_phone is not None:
        c_end = float(v_phone.minTime) * 1000.0
        n_start = c_end
        n_end = float(v_phone.maxTime) * 1000.0
        if v_idx > 0:
            prev = phones[v_idx - 1]
            prev_v = _ja_mark_to_vowel(getattr(prev, "mark", ""))
            if not prev_v:
                c_start = float(prev.minTime) * 1000.0
            else:
                c_start = c_end
    else:
        c_start = float(phones[0].minTime) * 1000.0
        c_end = float(phones[-1].minTime) * 1000.0
        n_start = c_end
        n_end = float(phones[-1].maxTime) * 1000.0

    if c_end < c_start:
        c_start = c_end
    return c_start, c_end, n_start, n_end


def _clamp_range(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _blend(current, target, weight):
    w = max(0.0, min(1.0, float(weight)))
    return float(current) * (1.0 - w) + float(target) * w
