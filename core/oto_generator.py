"""
TextGrid를 기반으로 한국어 OTO.ini를 생성합니다.
- phones/words tier를 이용해 OTO 파라미터를 계산합니다.
- CV/VC/VCV/VV/단모음/숨소리(br) 케이스를 처리합니다.
"""

import os
import re
import json
import wave
import datetime
import logging
import unicodedata
import textgrid
import copy

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from core.lab_generator import load_custom_phonemes
from core.textio_utils import load_template_oto_lines
from core.oto_profile_presets import get_kr_profile_preset

logger = logging.getLogger(__name__)

# ==============================================================================
# 접미사 처리 유틸리티
# ==============================================================================

def _normalize_alias_suffix(suffix):
    s = (suffix or "").strip()
    if not s:
        return ""
    return s[1:] if s.startswith("_") else s


def apply_alias_suffix(alias, suffix):
    """에일리어스 끝에 `_<suffix>` 접미사를 붙입니다."""
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return alias
    a = (alias or "").strip()
    if not a:
        return alias
    return f"{a}_{suf}"


def apply_suffix_to_oto_line(line, suffix):
    """`wav=alias,params...` 라인에서 alias에만 접미사를 적용합니다."""
    suf = _normalize_alias_suffix(suffix)
    if not suf:
        return line
    if not line or '=' not in line:
        return line
    left, right = line.split('=', 1)
    if ',' in right:
        alias_part, rest = right.split(',', 1)
        alias_part = apply_alias_suffix(alias_part.strip(), suf)
        return f"{left}={alias_part},{rest}"
    alias_part = apply_alias_suffix(right.strip(), suf)
    return f"{left}={alias_part}"

# ==============================================================================
# 기본 튜닝 파라미터
# ==============================================================================

DEFAULT_PARAMS = {
    'VC_CONSONANT_RATIO': 0.5,
    'VC_VOWEL_START': 0.3,
    'VC_PRE_OFFSET': 25,
    'VC_OVL_RATIO': 0.3,
    'CV_PRE_RATIO': 1.0,
    'CV_OVL_RATIO': 0.4,
    'CV_VOWEL_USE': 0.9,
    'DIPHTHONG_CV_PRE_RATIO': 0.35,
    'DIPHTHONG_CV_CONSONANT_RATIO': 0.6,
    'DIPHTHONG_CV_VOWEL_USE': 0.8,
    'DIPHTHONG_VC_VOWEL_START': 0.3,
    'DIPHTHONG_VC_CONSONANT': 0.5,
    'DIPHTHONG_VC_PRE_EXTEND': 1.2,
}


def normalize_key(name):
    base = os.path.splitext(name)[0]
    clean = re.sub(r"[^a-zA-Z0-9가-힣]", "", base)
    return clean.lower()


def is_diphthong_file(filename):
    """파일명에 이중모음 표식이 포함되는지 확인합니다."""
    clean = filename.lower().replace("'", "").replace(".wav", "")
    diphthong_markers = [
        'ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui',
        'gya','gye','gyeo','nya','nye','nyeo','dya','dye','dyeo',
        'rya','rye','ryeo','mya','mye','myeo','bya','bye','byeo',
        'sya','sye','syeo','jya','jye','jyeo','chya','chye','chyeo',
        'kya','kye','kyeo','tya','tye','tyeo','pya','pye','pyeo',
        'hya','hye','hyeo',
        'gwa','gwe','gwi','gweo','nwa','nwe','nwi','nweo',
        'dwa','dwe','dwi','dweo','rwa','rwe','rwi','rweo',
        'mwa','mwe','mwi','mweo','bwa','bwe','bwi','bweo',
        'swa','swe','swi','sweo','jwa','jwe','jwi','jweo',
        'chwa','chwe','chwi','chweo','kwa','kwe','kwi','kweo',
        'twa','twe','twi','tweo','pwa','pwe','pwi','pweo',
        'hwa','hwe','hwi','hweo',
    ]
    for diph in diphthong_markers:
        if diph in clean:
            return True
    return False


def is_diphthong(alias):
    """에일리어스 문자열에 이중모음 패턴이 있는지 확인합니다."""
    clean = alias.replace(' ', '').lower()
    diphthongs = ['ya','yeo','yo','yu','ye','wa','wo','wi','we','weo','eui','ui']
    for diph in diphthongs:
        if diph in clean:
            return True
    return False


def clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", mark).lower()


def normalize_ipa_mark(mark):
    """IPA 마크를 정규화해 비교 가능한 형태로 만듭니다."""
    base = clean_phone_mark(mark)
    nfd = unicodedata.normalize("NFD", base)
    stripped = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    # 장음/강세/기식 기호를 제거해 비교를 안정화합니다.
    stripped = stripped.replace("ː", "").replace("ˈ", "").replace("ˌ", "").replace("ʰ", "")
    return stripped


def is_glide(phone_mark):
    clean = normalize_ipa_mark(phone_mark)
    return clean in ['j', 'w', 'y', 'ɥ', 'ɰ', 'ʋ']


# MFA phone tier에서 자주 보이는 모음/파열음 표기를 넓게 커버
IPA_VOWELS = {
    'a', 'e', 'i', 'o', 'u', 'y',
    'eo', 'eu', 'ae', 'oe', 'wa', 'we', 'wi', 'wo', 'ui',
    # IPA vowel symbols frequently found in MFA phone tiers
    'ɯ', 'ʌ', 'ɛ', 'ə', 'æ', 'ɑ', 'ɐ', 'ɔ', 'ɪ', 'ʊ', 'ø', 'œ',
}

IPA_PLOSIVES = {
    'k', 'g', 't', 'd', 'p', 'b', 'c', 'q', 'ch', 'j',
}


def find_vowel_phone(phones):
    """음소 시퀀스에서 대표 모음을 찾아 (index, phone)으로 반환합니다."""

    for idx, ph in enumerate(phones):
        mark = normalize_ipa_mark(ph.mark)
        if mark in IPA_VOWELS and not is_glide(mark):
            return idx, ph
    
    phones_list = list(phones)

    for idx, ph in enumerate(phones_list):
        if is_glide(ph.mark) and idx + 1 < len(phones_list):
            return idx + 1, phones_list[idx + 1]
    

    return len(phones_list) - 1, phones_list[-1]


def is_plosive_ipa(phone_mark):
    """IPA 표기가 파열음 계열인지 판별합니다."""
    clean = normalize_ipa_mark(phone_mark)
    if clean in IPA_PLOSIVES:
        return True
    # 기식/장음 기호 제거 후 남는 기본 파열음도 인정
    return clean in {'k', 't', 'p', 'c', 'g', 'd', 'b'}


def is_plosive_roman(consonant_str):
    """로마자 자음 문자열이 파열음 계열인지 판별합니다."""
    return consonant_str.lower() in [
        'g', 'gg', 'd', 'dd', 'b', 'bb', 'j', 'jj',
        'k', 'kk', 't', 'tt', 'p', 'pp', 'ch'
    ]


KR_TENSE_CONSONANTS = {
    "kk", "tt", "pp", "ss", "jj", "gg", "dd", "bb",
}

KR_SONORANT_CONSONANTS = {
    "n", "m", "r", "l", "ng", "y", "w", "ry", "ly", "ny", "my",
}

KR_SIBILANT_ONSETS = {
    "s", "ss", "sh", "h", "f", "z",
}

KR_PLOSIVE_ONSETS = {
    "g", "gg", "k", "kk",
    "d", "dd", "t", "tt",
    "b", "bb", "p", "pp",
    "j", "jj", "ch", "c", "q",
}

KR_VOICED_ONSETS = {
    "g", "d", "b", "j", "z", "v",
    "n", "m", "r", "l", "ng", "y", "w",
}

KR_VOICELESS_ONSETS = {
    "k", "kk", "t", "tt", "p", "pp",
    "s", "ss", "sh", "h", "f", "ch", "c", "q",
}


def _extract_alias_onset(alias):
    """CV 에일리어스에서 초성(로마자 자음군)을 추출합니다."""
    a = re.sub(r"[^a-z]", "", (alias or "").lower())
    if not a:
        return ""
    m = re.match(r"^([^aeiouyw]+)", a)
    return m.group(1) if m else ""


def _is_tense_consonant(ipa_hint="", roman_hint=""):
    i = normalize_ipa_mark(ipa_hint or "")
    r = re.sub(r"[^a-z]", "", (roman_hint or "").lower())
    return i in KR_TENSE_CONSONANTS or r in KR_TENSE_CONSONANTS


def _is_sonorant_consonant(ipa_hint="", roman_hint=""):
    i = normalize_ipa_mark(ipa_hint or "")
    r = re.sub(r"[^a-z]", "", (roman_hint or "").lower())
    return i in KR_SONORANT_CONSONANTS or r in KR_SONORANT_CONSONANTS


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
    """
    한국어 프리셋 적용 후 자음 성질(치찰/파열, 유성/무성)에 맞춰
    pre/overlap/cons_gap/cut_gap을 미세 보정합니다.
    """
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
    _, cons_new, cutoff_new, pre_new, ovl_new = validate_oto_params(
        0.0, cons_new, cutoff_new, pre_new, ovl_new
    )
    return pre_new, cons_new, cutoff_new, ovl_new


def adaptive_overlap(pre, consonant_hint="", mode="cv"):
    """
    자음 성질/에일리어스 타입에 따라 overlap을 동적으로 조정합니다.
    """
    p = max(float(pre), 0.0)
    if p <= 0:
        return 0.0

    hint = normalize_ipa_mark(consonant_hint or "")
    hard = {'k', 'g', 't', 'd', 'p', 'b', 'c', 'ch', 'j', 'kk', 'tt', 'pp', 'gg', 'dd', 'bb'}
    fric = {'s', 'ss', 'sh', 'h', 'f', 'z'}
    sonorant = {'m', 'n', 'ng', 'r', 'l', 'y', 'w'}

    base_by_mode = {
        'cv': 0.46,
        'cv_head': 0.40,
        'vcv': 0.50,
        'vc': 0.44,
        'vv': 0.54,
    }
    ratio = base_by_mode.get(mode, 0.46)

    if hint in hard:
        ratio -= 0.16
    elif hint in fric:
        ratio += 0.05
    elif hint in sonorant:
        ratio += 0.08

    ratio = max(0.20, min(0.72, ratio))
    ovl = p * ratio
    if mode in ('vc', 'vv', 'vcv'):
        ovl = min(ovl, max(p - 14.0, 0.0))
    return max(0.0, min(ovl, p))


def _looks_like_vv_alias(alias):
    """공백 없는 단순 VV 에일리어스 형태인지 판별합니다."""
    if not alias:
        return False
    a = alias.strip().lower()
    if " " in a or len(a) < 2:
        return False
    if a in {"eo", "eu", "ae", "oe", "wi"}:
        return False
    return a[0] in {'a', 'e', 'i', 'o', 'u', 'y', 'w'} and a[-1] in {'a', 'e', 'i', 'o', 'u', 'y', 'w'}


def _compute_kr_cv_timing(c_start, c_end, cv_vowel_len, c_hint, alias_onset, is_diph, is_plosive):
    """
    한국어 CV 타이밍(이중모음/일반)을 계산합니다.
    반환: offset, consonant, cutoff, pre, ovl
    """
    boundary = c_end
    cv_vowel_len = max(float(cv_vowel_len), 20.0)
    is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
    is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)

    if is_diph:
        c_len = max(c_end - c_start, 10.0)
        target_pre = max(64.0, min(136.0, c_len + 14.0))
        if is_tense_cv:
            target_pre = _clamp(target_pre + 3.0, 66.0, 146.0)
        elif is_sonorant_cv:
            target_pre = min(150.0, target_pre + 10.0)
        offset = max(boundary - target_pre, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode='cv')
        if is_tense_cv:
            ovl = min(ovl, max(pre * 0.32, 9.0))
        elif is_sonorant_cv:
            ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))

        v_ref = max(cv_vowel_len, 140.0)
        if is_tense_cv:
            added_cons = min(max(v_ref * 0.52, 86.0), 196.0)
        elif is_sonorant_cv:
            added_cons = min(max(v_ref * 0.62, 98.0), 240.0)
        else:
            added_cons = min(max(v_ref * 0.55, 90.0), 230.0)
        consonant = pre + added_cons
        cutoff = -(consonant + max(cv_vowel_len * 0.2, 45.0))
        return offset, consonant, cutoff, pre, ovl

    if is_tense_cv:
        lead = 34.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 54.0, 176.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode='cv')
        ovl = min(ovl, max(pre * 0.32, 9.0))
    elif is_sonorant_cv:
        lead = 44.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 66.0, 216.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode='cv')
        ovl = max(ovl, min(pre * 0.50, max(pre - 8.0, 0.0)))
    elif is_plosive:
        lead = 32.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 50.0, 164.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode='cv')
    else:
        lead = 40.0
        pre_target = _clamp(max(c_end - c_start, 8.0) + lead, 58.0, 192.0)
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_hint, mode='cv')

    v_ref = max(cv_vowel_len, 130.0)
    if is_tense_cv:
        added_cons = min(max(v_ref * 0.42, 68.0), 162.0)
    elif is_sonorant_cv:
        added_cons = min(max(v_ref * 0.52, 86.0), 210.0)
    else:
        added_cons = min(max(v_ref * 0.45, 70.0), 180.0)
    consonant = pre + added_cons
    cutoff = -(consonant + max(cv_vowel_len * 0.25, 45.0))
    return offset, consonant, cutoff, pre, ovl


def _estimate_cv_anchor_from_syllable(syl, ph_intervals):
    """음절 정보 1개에서 CV anchor(상대 타이밍 shape)를 추정합니다."""
    curr_phones = (syl or {}).get('phones') or []
    if not curr_phones:
        return None

    roman_tok = (syl.get('roman_cv') or syl.get('roman') or "")
    if len(curr_phones) >= 2:
        _v_idx, v_phone = find_vowel_phone(curr_phones)
        c_start = curr_phones[0].minTime * 1000
        c_end = v_phone.minTime * 1000
        n_start = v_phone.minTime * 1000
        n_end = v_phone.maxTime * 1000
    else:
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000

    c_hint = curr_phones[0].mark if curr_phones else ""
    alias_onset = _extract_alias_onset(roman_tok)
    is_diph_syl = is_diphthong(roman_tok)
    cv_vowel_len = max(n_end - n_start, 20.0)
    first_phone_plosive = len(curr_phones) >= 2 and is_plosive_ipa(curr_phones[0].mark)
    is_plosive = (first_phone_plosive or is_plosive_roman(alias_onset)) if alias_onset else first_phone_plosive
    offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
        c_start,
        c_end,
        cv_vowel_len,
        c_hint,
        alias_onset,
        is_diph_syl,
        is_plosive,
    )

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    offset, consonant, cutoff, pre, ovl = _stabilize_params_to_phone_activity(
        offset, consonant, cutoff, pre, ovl, ph_intervals, alias_type="cv"
    )
    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return {
        "offset": offset,
        "pre": pre,
        "ovl": ovl,
        "cons": consonant,
        "cutoff": cutoff,
        "pre_abs": offset + pre,
        "cons_abs": offset + consonant,
        "onset_abs": c_start,
        "vowel_end_abs": n_end,
        "vowel_len": cv_vowel_len,
        "cons_gap": max(consonant - pre, 10.0),
        "cut_gap": max(abs(cutoff) - consonant, 16.0),
    }


def _compute_vc_from_adjacent_cv(prev_cv, next_cv, alias_type, is_plosive_sibilant):
    """인접 CV anchor를 이용해 VC/VV 파라미터를 계산합니다."""
    if not prev_cv or not next_cv:
        return None

    boundary_abs = next_cv["onset_abs"] if alias_type == "vc" else next_cv["pre_abs"]
    pre_target = _clamp(_blend(prev_cv["pre"], next_cv["pre"], 0.35), 45.0, 220.0)
    offset = max(boundary_abs - pre_target, 0.0)
    pre = boundary_abs - offset
    if pre <= 0:
        return None

    ovl_tail = _clamp(prev_cv["vowel_len"] * 0.10, 4.0, 22.0)
    target_ovl_abs = prev_cv["vowel_end_abs"] - ovl_tail
    ovl = target_ovl_abs - offset
    if is_plosive_sibilant:
        ovl = _clamp(ovl, pre * 0.24, max(pre - 12.0, 0.0))
    else:
        ovl = _clamp(ovl, pre * 0.34, max(pre - 10.0, 0.0))

    cons_gap = _clamp(_blend(prev_cv["cons_gap"], next_cv["cons_gap"], 0.45), 16.0, 120.0)
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
            consonant = min(consonant, next_onset_rel + 24.0)
            consonant = max(consonant, pre + 16.0)
            cutoff_abs = max(consonant + 12.0, min(next_cons_rel + 24.0, next_pre_rel + 42.0))
    else:
        consonant = min(max(consonant, pre + 24.0), next_pre_rel + 48.0)
        cutoff_abs = max(consonant + 20.0, next_pre_rel + 10.0)
        cutoff_abs = min(cutoff_abs, next_cons_rel + 60.0)

    cutoff = -cutoff_abs
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def _guard_cv_cutoff_to_next_onset(offset, consonant, cutoff, pre, syll_idx, syllables_info):
    """CV/CV_HEAD cutoff이 다음 음절 onset을 침범하지 않도록 상한을 건다."""
    if syll_idx is None or syll_idx < 0:
        return offset, consonant, cutoff, pre, 0.0
    if (syll_idx + 1) >= len(syllables_info):
        return offset, consonant, cutoff, pre, 0.0

    next_syl = syllables_info[syll_idx + 1]
    next_phones = next_syl.get("phones") or []
    if not next_phones:
        return offset, consonant, cutoff, pre, 0.0

    next_mark = (next_phones[0].mark or "").strip().lower()
    hard_next = is_plosive_ipa(next_mark) or next_mark in {
        "s", "ss", "sh", "ch", "j", "jj", "c", "ts", "h"
    }
    safety = 16.0 if hard_next else 10.0
    next_onset_rel = (next_phones[0].minTime * 1000.0) - offset
    max_cutoff_abs = next_onset_rel - safety
    if max_cutoff_abs <= (pre + 18.0):
        return offset, consonant, cutoff, pre, 0.0

    original_cutoff_abs = abs(cutoff)
    consonant = min(consonant, max_cutoff_abs - 14.0)
    consonant = max(consonant, pre + 10.0)

    cutoff_abs = min(original_cutoff_abs, max_cutoff_abs)
    if cutoff_abs <= (consonant + 8.0):
        cutoff_abs = min(max_cutoff_abs, consonant + 10.0)
        if cutoff_abs <= (consonant + 6.0):
            consonant = max(pre + 8.0, cutoff_abs - 10.0)
    cutoff = -cutoff_abs

    offset, consonant, cutoff, pre, _ovl = validate_oto_params(
        offset, consonant, cutoff, pre, 0.0
    )
    reduction = max(0.0, original_cutoff_abs - abs(cutoff))
    return offset, consonant, cutoff, pre, reduction


def _prepare_vcv_syllable_timing(syllables_info, current_w_idx, cv_seq_idx, diphthong_cv_consonant_ratio):
    """VCV 계산에 필요한 음절 인덱스 갱신과 기본 타이밍을 산출합니다."""
    if cv_seq_idx < len(syllables_info):
        current_w_idx = cv_seq_idx
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl['phones']

    if current_w_idx > 0:
        prev_syl = syllables_info[current_w_idx - 1]
        prev_phones = prev_syl['phones']
        prev_v_start = prev_phones[-1].minTime * 1000
        prev_v_end = prev_phones[-1].maxTime * 1000
    else:
        prev_v_start = curr_phones[0].minTime * 1000 - 100
        prev_v_end = curr_phones[0].minTime * 1000

    if len(curr_phones) >= 2:
        c_boundary = curr_phones[-1].minTime * 1000
        n_end = curr_phones[-1].maxTime * 1000
    else:
        c_boundary = curr_phones[0].minTime * 1000
        n_end = curr_phones[0].maxTime * 1000

    prev_v_len = prev_v_end - prev_v_start
    offset_padding = min(prev_v_len * 0.6, 200)
    if offset_padding < 80:
        offset_padding = max(prev_v_len * 0.5, 50)

    offset = prev_v_end - offset_padding
    if offset < 0:
        offset = 0

    pre = c_boundary - offset
    c_hint = curr_phones[0].mark if curr_phones else ""
    ovl = adaptive_overlap(pre, c_hint, mode='vcv')

    vowel_len = n_end - c_boundary
    added_cons = min(vowel_len * diphthong_cv_consonant_ratio, 150)
    if added_cons < 50:
        added_cons = 50
    consonant = pre + added_cons
    cutoff = -(consonant + vowel_len * 0.25)
    return current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl


def _prepare_cv_head_syllable_timing(syllables_info, current_w_idx, cv_seq_idx, alias):
    """CV_HEAD 계산에 필요한 음절 인덱스 갱신과 기본 타이밍을 산출합니다."""
    if cv_seq_idx < len(syllables_info):
        current_w_idx = cv_seq_idx
        cv_seq_idx = current_w_idx + 1
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl['phones']

    if len(curr_phones) >= 2:
        c_start = curr_phones[0].minTime * 1000
        c_end = curr_phones[-1].minTime * 1000
        n_start = c_end
        n_end = curr_phones[-1].maxTime * 1000
    else:
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000

    c_hint = curr_phones[0].mark if curr_phones else ""
    alias_onset = _extract_alias_onset(alias)
    is_tense_cv = _is_tense_consonant(c_hint, alias_onset)
    is_sonorant_cv = _is_sonorant_consonant(c_hint, alias_onset)

    if is_tense_cv:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 34.0, 50.0, 170.0)
        offset = max(c_end - pre_target, 0.0)
    elif is_sonorant_cv:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 42.0, 62.0, 205.0)
        offset = max(c_end - pre_target, 0.0)
    else:
        pre_target = _clamp(max(c_end - c_start, 8.0) + 32.0, 48.0, 162.0)
        offset = max(c_end - pre_target, 0.0)

    pre = c_end - offset if c_end > c_start else 30
    ovl = adaptive_overlap(pre, c_hint, mode='cv_head')
    if is_tense_cv:
        ovl = min(ovl, max(pre * 0.32, 9.0))
    elif is_sonorant_cv:
        ovl = max(ovl, min(pre * 0.48, max(pre - 10.0, 0.0)))

    cv_vowel_len = n_end - n_start
    if is_tense_cv:
        added_cons = min(max(cv_vowel_len * 0.46, 72), 156)
    elif is_sonorant_cv:
        added_cons = min(max(cv_vowel_len * 0.58, 88), 190)
    else:
        added_cons = min(cv_vowel_len * 0.5, 150)
        if added_cons < 80:
            added_cons = 80

    consonant = pre + added_cons
    cutoff = -(consonant + cv_vowel_len * 0.25)
    return current_w_idx, cv_seq_idx, offset, consonant, cutoff, pre, ovl


def _append_alias_rows(
    final_lines,
    real_wav_name,
    alias,
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    generate_openutau=False,
    alias_suffix="",
):
    """에일리어스(및 OpenUtau 변형)를 OTO 라인으로 누적합니다."""
    aliases_to_write = generate_openutau_aliases(alias) if generate_openutau else [alias]
    for a in aliases_to_write:
        a2 = apply_alias_suffix(a, alias_suffix)
        new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
        final_lines.append(new_line)


def _resolve_cv_syllable_index(target_clean, romaji_syllables, cv_seq_idx, current_w_idx):
    """CV 계열 alias를 words/roman 음절 인덱스에 매핑합니다."""
    if cv_seq_idx >= len(romaji_syllables):
        return current_w_idx, cv_seq_idx

    name_match_idx = None
    best_score = -1
    # diphthong/r-l 혼용 케이스를 위해 탐색 범위를 넓혀 안정적으로 음절 정렬
    scan_start = max(cv_seq_idx - 1, 0)
    scan_end = min(cv_seq_idx + 5, len(romaji_syllables))
    for i in range(scan_start, scan_end):
        score = _cv_match_score(target_clean, romaji_syllables[i])
        score -= abs(i - cv_seq_idx) * 4
        if score > best_score:
            best_score = score
            name_match_idx = i
        if score >= 98:
            break

    expected_score = -1
    if 0 <= cv_seq_idx < len(romaji_syllables):
        expected_score = _cv_match_score(target_clean, romaji_syllables[cv_seq_idx])

    if name_match_idx is not None and best_score >= 62:
        chosen_idx = name_match_idx
        best_gain = best_score - expected_score
        target_onset, target_vowel, _target_coda = _split_kr_syllable_parts(target_clean)
        expected_tok = romaji_syllables[cv_seq_idx] if 0 <= cv_seq_idx < len(romaji_syllables) else ""
        best_tok = romaji_syllables[name_match_idx] if 0 <= name_match_idx < len(romaji_syllables) else ""
        exp_onset, exp_vowel, _exp_coda = _split_kr_syllable_parts(expected_tok)
        best_onset, best_vowel, _best_coda = _split_kr_syllable_parts(best_tok)
        same_vowel_expected = bool(target_vowel and exp_vowel and target_vowel == exp_vowel)
        best_vowel_match = bool(target_vowel and best_vowel and target_vowel == best_vowel)
        same_onset_expected = bool(
            target_onset and exp_onset and (target_onset == exp_onset or target_onset[:1] == exp_onset[:1])
        )
        # 이중모음/종성 포함 토큰에서 발생하는 과도한 앞 점프를 억제한다.
        if (
            name_match_idx > cv_seq_idx
            and expected_score >= max(50, best_score - 20)
        ):
            chosen_idx = cv_seq_idx
        # 한 음절 점프는 충분히 큰 이득이 없으면 보수적으로 유지합니다.
        if abs(name_match_idx - cv_seq_idx) == 1:
            min_gain = 22
            if same_vowel_expected:
                min_gain = 18
            elif same_onset_expected:
                min_gain = 20
            if best_gain < min_gain:
                chosen_idx = cv_seq_idx
            if same_vowel_expected and (not best_vowel_match) and best_gain < 22:
                chosen_idx = cv_seq_idx
            if same_onset_expected and (not (target_onset and best_onset and target_onset[:1] == best_onset[:1])) and best_gain < 24:
                chosen_idx = cv_seq_idx
        # 뒤로 가는 선택도 점수 이득이 충분하지 않으면 방지합니다.
        if name_match_idx < cv_seq_idx and best_gain < 24:
            chosen_idx = cv_seq_idx
        current_w_idx = chosen_idx
    else:
        current_w_idx = cv_seq_idx

    cv_seq_idx = current_w_idx + 1
    return current_w_idx, cv_seq_idx


def _prepare_cv_bounds_from_syllable(syllables_info, current_w_idx):
    """CV 계산에 필요한 (curr_phones, c_start/c_end/n_start/n_end)를 구성합니다."""
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl['phones']

    if len(curr_phones) >= 2:
        _v_idx, v_phone = find_vowel_phone(curr_phones)
        c_start = curr_phones[0].minTime * 1000
        c_end = v_phone.minTime * 1000
        n_start = v_phone.minTime * 1000
        n_end = v_phone.maxTime * 1000
    else:
        c_start = curr_phones[0].minTime * 1000
        c_end = c_start
        n_start = c_start
        n_end = curr_phones[0].maxTime * 1000

    return current_w_idx, curr_phones, c_start, c_end, n_start, n_end


def _prepare_vc_bounds_from_context(syllables_info, current_w_idx):
    """VC 계산에 필요한 (curr_phones, c_start/c_end/n_start/n_end)를 구성합니다."""
    if current_w_idx >= len(syllables_info):
        current_w_idx = len(syllables_info) - 1

    curr_syl = syllables_info[current_w_idx]
    curr_phones = curr_syl['phones']

    v_start = curr_phones[-1].minTime * 1000
    v_end = curr_phones[-1].maxTime * 1000
    c_start = v_start
    c_end = v_end

    if current_w_idx + 1 < len(syllables_info):
        next_syl = syllables_info[current_w_idx + 1]
        n_start = next_syl['phones'][0].minTime * 1000
        n_end = next_syl['phones'][0].maxTime * 1000
    else:
        n_start = v_end
        n_end = v_end + 100

    return current_w_idx, curr_phones, c_start, c_end, n_start, n_end


def _compute_kr_vc_timing(
    alias,
    alias_type,
    file_format,
    curr_phones,
    current_w_idx,
    syllables_info,
    c_start,
    c_end,
    n_start,
    n_end,
    cv_anchor_by_idx,
):
    """한국어 VC/VV 타이밍 계산 핵심 로직입니다."""
    c_char = _extract_vc_right_token(alias)
    is_vc_plosive_coda = (
        alias_type == 'vc'
        and file_format in {'cvc', 'cvvc', 'vc_only'}
        and _is_kr_plosive_coda_alias(alias)
    )

    if is_vc_plosive_coda:
        # 종성 파열음은 받침 구간을 먹지 않도록 모음 끝 직전에 pre를 고정합니다.
        coda_canon = _canonicalize_kr_coda(c_char)
        is_hard_stop_coda = coda_canon in {"t", "p"}
        v_idx = None
        v_phone = None
        if curr_phones:
            v_idx, v_phone = find_vowel_phone(curr_phones)

        vowel_start = (v_phone.minTime * 1000) if v_phone else c_start
        vowel_end = (v_phone.maxTime * 1000) if v_phone else c_end

        coda_start = None
        if v_phone is not None and v_idx is not None and (v_idx + 1) < len(curr_phones):
            coda_start = curr_phones[v_idx + 1].minTime * 1000
        elif current_w_idx + 1 < len(syllables_info):
            next_syl = syllables_info[current_w_idx + 1]
            if next_syl.get('phones'):
                coda_start = next_syl['phones'][0].minTime * 1000

        boundary = max(vowel_start + 16.0, vowel_end - (14.0 if is_hard_stop_coda else 11.0))
        if coda_start is not None:
            coda_margin = 12.0 if is_hard_stop_coda else 9.0
            boundary = min(boundary, coda_start - coda_margin)
            boundary = max(boundary, vowel_start + (12.0 if is_hard_stop_coda else 14.0))

        pre_target = _clamp(
            boundary - vowel_start,
            42.0 if is_hard_stop_coda else 45.0,
            132.0 if is_hard_stop_coda else 145.0,
        )
        offset = max(boundary - pre_target, 0.0)
        pre = boundary - offset
        ovl = adaptive_overlap(pre, c_char, mode='vc')
        ovl = min(ovl, max(pre - 12.0, 0.0))

        tail_floor = 10.0 if is_hard_stop_coda else 12.0
        if coda_start is not None:
            tail_room = max(coda_start - boundary, tail_floor)
        else:
            tail_room = max(n_start - boundary, tail_floor)

        cons_mul = 0.54 if is_hard_stop_coda else 0.62
        cons_min = 10.0 if is_hard_stop_coda else 12.0
        cons_max = 30.0 if is_hard_stop_coda else 38.0
        added_cons = _clamp(tail_room * cons_mul, cons_min, cons_max)
        consonant = pre + added_cons
        cut_mul = 0.56 if is_hard_stop_coda else 0.70
        cut_min = 20.0 if is_hard_stop_coda else 24.0
        cut_max = 52.0 if is_hard_stop_coda else 64.0
        cut_gap = _clamp(tail_room * cut_mul, cut_min, cut_max)

        cutoff_abs = consonant + cut_gap
        next_onset_rel = max(n_start - offset, pre + 12.0)
        cutoff_soft_cap = next_onset_rel + (4.0 if is_hard_stop_coda else 8.0)
        cutoff_min_abs = consonant + (8.0 if is_hard_stop_coda else 10.0)
        if cutoff_soft_cap <= cutoff_min_abs:
            consonant = min(consonant, max(next_onset_rel - 6.0, pre + 8.0))
            cutoff_min_abs = consonant + (6.0 if is_hard_stop_coda else 8.0)
            cutoff_soft_cap = max(cutoff_soft_cap, cutoff_min_abs)
        cutoff_abs = _clamp(cutoff_abs, cutoff_min_abs, cutoff_soft_cap)
        cutoff = -cutoff_abs
        return offset, consonant, cutoff, pre, ovl, True

    vc_anchor_params = None
    if current_w_idx + 1 < len(syllables_info):
        prev_cv_anchor = cv_anchor_by_idx.get(current_w_idx)
        next_cv_anchor = cv_anchor_by_idx.get(current_w_idx + 1)
        is_plosive_sibilant = c_char in ['g', 'k', 'kk', 'gg', 'd', 't', 'tt', 'dd', 'b', 'p', 'bb', 'pp', 's', 'ss', 'h', 'j', 'jj', 'ch']
        vc_anchor_params = _compute_vc_from_adjacent_cv(
            prev_cv_anchor, next_cv_anchor, alias_type, is_plosive_sibilant
        )

    if vc_anchor_params is not None:
        offset, consonant, cutoff, pre, ovl = vc_anchor_params
        return offset, consonant, cutoff, pre, ovl, False

    # VC should attach to the next consonant onset.
    # Consonant-end anchoring often causes awkward late VC timing.
    vc_target = n_start if alias_type == 'vc' else n_end
    boundary = min(vc_target, c_end + 260)
    v_len = c_end - c_start
    n_len = n_end - n_start

    offset_padding = 180
    if v_len < offset_padding:
        offset_padding = max(v_len * 0.8, 50)

    offset = boundary - offset_padding
    pre = boundary - offset

    is_plosive_sibilant = c_char in ['g', 'k', 'kk', 'gg', 'd', 't', 'tt', 'dd', 'b', 'p', 'bb', 'pp', 's', 'ss', 'h', 'j', 'jj', 'ch']
    ovl = adaptive_overlap(pre, c_char, mode='vv' if alias_type == 'vv' else 'vc')

    if is_plosive_sibilant:
        n_ref = max(n_len, 100)
        added_cons = min(max(n_ref * 0.35, 45), 95)
        consonant = pre + added_cons
        cutoff = -(consonant + max(n_len * 0.35, 45))
    else:
        n_ref = max(n_len, 80)
        added_cons = min(max(n_ref * 0.55, 55), 160)
        consonant = pre + added_cons
        cutoff = -(consonant + max(n_len * 0.45, 50))

    if alias_type == 'vc':
        # CVVC 원리: VC는 다음 자음 onset 주변에서 정리해 CV와의 중복 자음을 줄인다.
        next_c_onset_rel = max(n_start - offset, pre + 10.0)
        if is_plosive_sibilant:
            consonant = min(consonant, next_c_onset_rel - 6.0)
            consonant = max(consonant, pre + 12.0)
            cutoff_abs = max(consonant + 10.0, next_c_onset_rel - 2.0)
            cutoff_abs = min(cutoff_abs, next_c_onset_rel + 8.0)
            cutoff = -cutoff_abs
        else:
            consonant = min(consonant, next_c_onset_rel + 26.0)
            consonant = max(consonant, pre + 16.0)
            cutoff_abs = min(abs(cutoff), next_c_onset_rel + 30.0)
            cutoff_abs = max(cutoff_abs, consonant + 12.0)
            cutoff = -cutoff_abs

    return offset, consonant, cutoff, pre, ovl, False


def _apply_post_timing_pipeline(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    alias_type,
    mel_ctx_for_file,
    base_shape,
    ph_intervals,
    current_w_idx,
    syllables_info,
    is_vc_plosive_coda=False,
    enable_stabilize=True,
    enable_cutoff_guard=True,
):
    """후처리 가드(soft mel/base shape/stabilize/cutoff)를 일관 적용합니다."""
    soft_off_shift = 0.0
    soft_cut_shift = 0.0
    cutoff_reduced = 0.0

    if alias_type in {"cv", "cv_head", "vcv"}:
        offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift = _apply_soft_mel_offset_cutoff_guard(
            offset, consonant, cutoff, pre, ovl, alias_type, mel_ctx_for_file
        )

    if not is_vc_plosive_coda:
        offset, consonant, cutoff, pre, ovl = _apply_base_shape_blend(
            offset, consonant, cutoff, pre, ovl, base_shape, alias_type=alias_type
        )

    if enable_stabilize:
        offset, consonant, cutoff, pre, ovl = _stabilize_params_to_phone_activity(
            offset, consonant, cutoff, pre, ovl, ph_intervals, alias_type=alias_type
        )

    if enable_cutoff_guard and alias_type in {"cv", "cv_head"}:
        offset, consonant, cutoff, pre, cutoff_reduced = _guard_cv_cutoff_to_next_onset(
            offset, consonant, cutoff, pre, current_w_idx, syllables_info
        )

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return offset, consonant, cutoff, pre, ovl, soft_off_shift, soft_cut_shift, cutoff_reduced


def _log_post_timing_events(log_fn, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced):
    """후처리 가드에서 발생한 유의미한 이동량을 로그로 기록합니다."""
    if abs(soft_off_shift) > 1.0 or abs(soft_cut_shift) > 1.0:
        log_fn(
            f"🛡️ {fname}: 초기 멜 가드 적용 (offset {soft_off_shift:+.1f}ms, cutoff -{soft_cut_shift:.1f}ms) [{alias}]"
        )
    if cutoff_reduced > 0.5:
        log_fn(f"🛡️ {fname}: CV 컷오프 과연장 보정(-{cutoff_reduced:.1f}ms) [{alias}]")


# ==============================================================================
# 에일리어스 분류용 상수
# ==============================================================================


KR_VOWELS = {
    'a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ae', 'oe', 'wi',
    'ya', 'yeo', 'yo', 'yu', 'ye', 'wa', 'wo', 'we', 'weo', 'eui', 'ui', 'wae'
}


KR_CONSONANTS = {
    'g', 'n', 'd', 'r', 'l', 'm', 'b', 's', 'j', 'ch', 'k', 't', 'p', 'h',
    'gg', 'dd', 'bb', 'ss', 'jj', 'kk', 'tt', 'pp',
    'ng', 'sh', 'f', 'z', 'q', 'c'
}


KR_BATCHIM_MARKERS = {'N', 'L', 'M', 'NG', 'K', 'T', 'P', 'H'}


GLOTTAL_MARKS = {"'", "’", ".", "ʔ"}

KR_CODA_PLOSIVE_MAP = {
    "g": "k", "gg": "k", "k": "k", "kk": "k",
    "d": "t", "dd": "t", "t": "t", "tt": "t",
    "s": "t", "ss": "t", "j": "t", "jj": "t", "ch": "t", "h": "t", "c": "t", "q": "t",
    "b": "p", "bb": "p", "p": "p", "pp": "p",
}


def _detect_glottal_kind(alias):
    """에일리어스의 성문 파열 표식 위치(head/tail)를 판별합니다."""
    a = (alias or "").strip()
    if not a:
        return None
    parts = a.split()
    if len(parts) >= 2:
        if parts[0] in GLOTTAL_MARKS:
            return 'head'
        if parts[-1] in GLOTTAL_MARKS:
            return 'tail'

    if a[0] in GLOTTAL_MARKS:
        return 'head'
    if a[-1] in GLOTTAL_MARKS:
        return 'tail'
    return None


def is_breath(alias):
    """숨소리(br, br1...) 에일리어스인지 판별합니다."""
    clean = alias.strip().lower()
    return bool(re.match(r'^br\d*$', clean))


def _extract_vowel_consonant(text):
    """로마자 음절에서 모음부와 초성부를 분리해 반환합니다."""
    onset, vowel, _ = _split_kr_syllable_parts(text)
    if vowel:
        return vowel, onset
    text = re.sub(r"[^a-z]", "", (text or "").lower())
    return None, text


def _split_kr_syllable_parts(text):
    """
    한국어 로마자 음절을 (초성, 중성, 종성)으로 분해합니다.
    예: nyeong -> (n, yeo, ng), ryeo -> (r, yeo, "")
    """
    t = _normalize_cv_match_token(text)
    if not t:
        return "", "", ""

    vowels = sorted(KR_VOWELS, key=len, reverse=True)
    v_idx = -1
    v_val = ""
    for i in range(len(t)):
        matched = None
        for v in vowels:
            if t.startswith(v, i):
                matched = v
                break
        if matched:
            v_idx = i
            v_val = matched
            break

    if v_idx < 0:
        return "", "", t

    onset = t[:v_idx]
    coda = t[v_idx + len(v_val):]
    return onset, v_val, coda


def _kr_cv_kernel(text):
    """비교용 CV 핵심 문자열(onset+vowel)을 반환합니다."""
    onset, vowel, _ = _split_kr_syllable_parts(text)
    if vowel:
        return f"{onset}{vowel}"
    return _normalize_cv_match_token(text)


def _extract_vc_right_token(alias):
    """VC 에일리어스에서 우측 자음(받침 후보)을 추출합니다."""
    a = (alias or "").strip()
    if not a:
        return ""

    if " " in a:
        parts = [p for p in a.split() if p]
        if len(parts) >= 2:
            return parts[1].strip().rstrip("-")
        return ""

    m = re.match(
        r'^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch|c|q|h)$',
        a.lower()
    )
    if m:
        return m.group(2)
    return ""


def _canonicalize_kr_coda(token):
    """받침 자음 표기를 K/T/P 계열 기준으로 정규화합니다."""
    t = re.sub(r"[^A-Za-z]", "", (token or "")).strip()
    if not t:
        return ""

    t_upper = t.upper()
    if t_upper == "K":
        return "k"
    if t_upper in {"T", "H"}:
        return "t"
    if t_upper == "P":
        return "p"

    return KR_CODA_PLOSIVE_MAP.get(t.lower(), t.lower())


def _is_kr_plosive_coda_alias(alias):
    """VC 에일리어스가 파열음 받침(K/T/P 계열)인지 판별합니다."""
    right = _extract_vc_right_token(alias)
    if not right:
        return False
    canon = _canonicalize_kr_coda(right)
    return canon in {"k", "t", "p"}


def _normalize_cv_match_token(token):
    """CV 음절 매핑 비교용 정규화(r/l 혼용 등)를 적용합니다."""
    t = re.sub(r"[^a-z]", "", (token or "").lower())
    if not t:
        return t
    # ㄹ 초성 표기가 r/l로 혼용되는 경우를 흡수
    if t.startswith("l"):
        t = "r" + t[1:]
    t = t.replace("ly", "ry")
    return t


def _cv_match_score(alias_token, syllable_token):
    """에일리어스 CV와 음절 로마자 간 매핑 점수를 계산합니다."""
    a = _normalize_cv_match_token(alias_token)
    s = _normalize_cv_match_token(syllable_token)
    if not a or not s:
        return 0
    if a == s:
        return 100

    ao, av, _ = _split_kr_syllable_parts(a)
    so, sv, scoda = _split_kr_syllable_parts(s)
    score = 0

    if av and sv:
        if av == sv:
            score += 70
        elif av.startswith(sv) or sv.startswith(av):
            score += 44

    if ao and so:
        if ao == so:
            score += 24
        elif {ao, so} <= {"r", "l"}:
            score += 20
        elif ao.startswith(so) or so.startswith(ao):
            score += 12
        else:
            score -= 16
    elif not ao and not so:
        score += 12
    else:
        score -= 8

    # syllable의 종성이 붙어도 CV 핵심(onset+vowel)이 일치하면 높은 점수 유지
    if av and sv and (ao + av) == (so + sv):
        score += 16
    if scoda and av and (so + sv) and a == (so + sv):
        score += 6

    if a in s or s in a:
        score += 10
    return max(0, min(score, 100))


def _extract_kr_cv_alias_token(alias):
    """CV 매핑용 에일리어스 핵심 토큰을 추출합니다."""
    parts = [p for p in re.split(r"\s+", (alias or "").strip().lower()) if p]
    if not parts:
        return ""
    if parts[0] == "-" and len(parts) >= 2:
        tok = parts[1]
    else:
        tok = parts[-1]
    tok = re.sub(r"[^a-z]", "", tok)
    return _kr_cv_kernel(tok)


def classify_alias(alias, custom_map=None):
    """에일리어스 문자열을 br/cv/cv_head/vc/vv/vcv/mono 타입으로 분류합니다."""
    clean = alias.strip()
    

    if is_breath(clean):
        return 'br'
        

    if custom_map and clean in custom_map:
        mapped_val = custom_map[clean].lower()
        if mapped_val in ['r', 'h', 'sil', 'br']:
            return 'br'
        if mapped_val in KR_VOWELS:
            return 'mono'
        return 'cv'
    

    if ' ' in clean:
        parts = clean.split()
        left = parts[0].strip()
        right = ' '.join(parts[1:]).strip()
        

        if left == '-':
            return 'cv_head'
        

        if right == '-':
            return 'cv'


        left_lower = left.lower()
        right_lower = right.lower().rstrip('-')
        if left in GLOTTAL_MARKS and right_lower in KR_VOWELS:
            return 'cv_head'
        if left_lower in KR_VOWELS and right in GLOTTAL_MARKS:
            return 'vc'
        

        left_upper = left.upper()
        if left_upper in KR_BATCHIM_MARKERS:

            return 'vcv'
        

        left_lower = left.lower()
        right_lower = right.lower().rstrip('-')
        

        if left_lower in KR_VOWELS:

            if right_lower in KR_CONSONANTS or right.upper() in KR_BATCHIM_MARKERS:
                return 'vc'

            if right_lower in KR_VOWELS:
                return 'vv'

            return 'vcv'
        

        return 'vc'
    

    clean_lower = clean.lower().rstrip('-')


    gk = _detect_glottal_kind(clean)
    if gk == 'head':
        return 'cv_head'
    if gk == 'tail':
        return 'vc'
    

    if custom_map and clean_lower in custom_map:
        mapped_val = custom_map[clean_lower].lower()
        if mapped_val in ['sil', 'br', 'r', 'h']: return 'br'
        if mapped_val in KR_VOWELS: return 'mono'
        return 'cv'
    

    if clean_lower in KR_VOWELS:
        return 'mono'
    

    m = re.match(r'^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch)$', clean_lower)
    if m:
        return 'vc'
    

    return 'cv'


def detect_alias_format(alias_list, custom_map=None):
    """파일 단위 에일리어스 목록의 전체 포맷(CVC/CVVC/VCV 등)을 추정합니다."""
    if not alias_list:
        return 'cvc'
    
    types = [classify_alias(a, custom_map) for a in alias_list]
    type_set = set(types)
    

    if type_set == {'br'}:
        return 'br'
    

    if type_set <= {'mono', 'cv_head', 'cv'}:
        return 'mono'
    

    if 'vcv' in type_set:
        return 'vcv'
    

    non_br = type_set - {'br'}
    if non_br <= {'vc', 'vv'}:
        return 'vc_only'
    

    if non_br <= {'cv', 'mono', 'cv_head'}:
        return 'cv_simple'
    

    if 'vv' in type_set:
        return 'cvvc'
    
    return 'cvc'


def _alias_to_cv_target(alias, alias_type):
    a = (alias or "").strip()
    if not a:
        return ""

    if alias_type == "cv":
        tok = re.sub(r"[^a-z]", "", a.lower())
        return _kr_cv_kernel(tok)
    if alias_type == "cv_head":
        parts = a.split()
        if len(parts) >= 2 and parts[0] == "-":
            tok = re.sub(r"[^a-z]", "", parts[1].lower())
            return _kr_cv_kernel(tok)
        tok = re.sub(r"[^a-z]", "", a.lower().lstrip("-"))
        return _kr_cv_kernel(tok)
    if alias_type == "vcv":
        parts = a.split()
        if len(parts) >= 2:
            tok = re.sub(r"[^a-z]", "", parts[1].lower())
            return _kr_cv_kernel(tok)
    if alias_type == "mono":
        tok = re.sub(r"[^a-z]", "", a.lower())
        if tok in KR_VOWELS:
            return tok
    return ""


def _extract_cv_targets_from_lines(lines, custom_map=None):
    targets = []
    for line in lines or []:
        if "=" not in line:
            continue
        alias = line.split("=", 1)[1].split(",", 1)[0].strip()
        if not alias:
            continue
        a_type = classify_alias(alias, custom_map)
        tok = _alias_to_cv_target(alias, a_type)
        if tok:
            targets.append(tok)
    if not targets:
        return []

    collapsed = []
    for t in targets:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    return collapsed


def _is_kr_nucleus_phone_mark(mark):
    m = normalize_ipa_mark(mark)
    if not m:
        return False
    if m in IPA_VOWELS:
        return True
    return m in {"ɯ", "ʌ", "ɛ", "ə", "æ", "ɑ", "ɐ", "ɔ", "ɪ", "ʊ", "ø", "œ"}


def _build_kr_syllables_from_phone_nuclei(ph_intervals, cv_targets):
    if not ph_intervals or not cv_targets:
        return None

    target_n = len(cv_targets)
    nuclei = [
        i for i, p in enumerate(ph_intervals)
        if _is_kr_nucleus_phone_mark(p.mark) and not is_glide(p.mark)
    ]
    if len(nuclei) < target_n:
        return None

    if target_n == 1:
        selected = [nuclei[0]]
    elif len(nuclei) == target_n:
        selected = list(nuclei)
    else:
        n_count = len(nuclei)
        selected_pos = []
        for i in range(target_n):
            ideal = int(round(i * (n_count - 1) / float(target_n - 1)))
            lo = i
            hi = n_count - (target_n - i)
            pos = max(lo, min(hi, ideal))
            selected_pos.append(pos)
        selected = [nuclei[pos] for pos in selected_pos]

    out = []
    global_start = float(ph_intervals[0].minTime)
    global_end = float(ph_intervals[-1].maxTime)
    for i, nuc_idx in enumerate(selected):
        if i == 0:
            s_t = global_start
        else:
            prev_n = selected[i - 1]
            s_t = (float(ph_intervals[prev_n].maxTime) + float(ph_intervals[nuc_idx].minTime)) * 0.5

        if i + 1 < len(selected):
            next_n = selected[i + 1]
            e_t = (float(ph_intervals[nuc_idx].maxTime) + float(ph_intervals[next_n].minTime)) * 0.5
        else:
            e_t = global_end

        phones = [
            p for p in ph_intervals
            if float(p.minTime) >= s_t - 1e-6 and float(p.maxTime) <= e_t + 1e-6
        ]
        if not phones:
            phones = [ph_intervals[nuc_idx]]

        tok = cv_targets[i] if i < len(cv_targets) else ""
        out.append({
            "word": tok,
            "roman": tok,
            "roman_cv": tok,
            "start_time": s_t,
            "end_time": e_t,
            "phones": phones,
        })

    return out


def _score_kr_syllable_mapping(candidate_infos, cv_targets):
    """candidate 음절열이 alias 기반 cv_targets와 얼마나 일치하는지 점수화."""
    if not candidate_infos or not cv_targets:
        return -1.0

    cand = []
    for s in candidate_infos:
        tok = s.get("roman_cv") or s.get("roman") or s.get("word") or ""
        tok = _kr_cv_kernel(tok)
        if tok:
            cand.append(tok)
    if not cand:
        return -1.0

    def _avg_with_shift(shift):
        vals = []
        for i, tgt in enumerate(cv_targets):
            j = i + shift
            if 0 <= j < len(cand):
                vals.append(_cv_match_score(tgt, cand[j]))
        if not vals:
            return -1.0
        coverage = len(vals) / float(max(len(cv_targets), 1))
        length_penalty = abs(len(cand) - len(cv_targets)) * 4.0
        return (sum(vals) / float(len(vals))) * coverage - length_penalty

    return max(_avg_with_shift(0), _avg_with_shift(1), _avg_with_shift(-1))


def validate_oto_params(offset, consonant, cutoff, pre, ovl):
    """UTAU OTO 파라미터를 유효 범위로 보정합니다."""
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


def _nearest_phone_edge_ms(anchor_ms, ph_intervals):
    """anchor_ms에 가장 가까운 phone 경계(ms)를 반환합니다."""
    nearest = None
    nearest_dist = float("inf")
    for ph in ph_intervals or []:
        s = float(ph.minTime) * 1000.0
        e = float(ph.maxTime) * 1000.0
        if s <= anchor_ms <= e:
            return anchor_ms, 0.0
        ds = abs(anchor_ms - s)
        de = abs(anchor_ms - e)
        if ds < nearest_dist:
            nearest_dist = ds
            nearest = s
        if de < nearest_dist:
            nearest_dist = de
            nearest = e
    if nearest is None:
        return anchor_ms, 0.0
    return nearest, nearest_dist


def _surrounding_phone_gap(anchor_ms, ph_intervals):
    """
    anchor_ms가 phone 사이 gap에 있을 때 (prev_end, next_start, gap_len_ms) 반환.
    gap 내부가 아니면 (None, None, 0.0).
    """
    prev_end = None
    next_start = None
    for ph in ph_intervals or []:
        s = float(ph.minTime) * 1000.0
        e = float(ph.maxTime) * 1000.0
        if s <= anchor_ms <= e:
            return None, None, 0.0
        if e < anchor_ms:
            prev_end = e
        elif s > anchor_ms and next_start is None:
            next_start = s
            break
    if prev_end is None or next_start is None or next_start <= prev_end:
        return None, None, 0.0
    return prev_end, next_start, (next_start - prev_end)


def _stabilize_params_to_phone_activity(offset, consonant, cutoff, pre, ovl, ph_intervals, alias_type="cv"):
    """
    선행발성 기준점(pre 절대위치)이 무음 gap에 걸리면
    가장 가까운 유효 phone 경계로 스냅해 빈 공간 정렬을 완화합니다.
    """
    if not ph_intervals:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    pre_abs = float(offset) + float(pre)
    nearest_edge, nearest_dist = _nearest_phone_edge_ms(pre_abs, ph_intervals)
    prev_end, next_start, gap_len = _surrounding_phone_gap(pre_abs, ph_intervals)

    # 충분히 넓은 gap 내부이거나 phone과 멀리 떨어진 경우만 보정
    if gap_len < 55.0 and nearest_dist <= 34.0:
        return offset, consonant, cutoff, pre, ovl

    target = nearest_edge
    if prev_end is not None and next_start is not None:
        if alias_type in {"vc", "vv"}:
            # VC/VV는 다음 음절 입구를 겨냥하되, 경계보다 약간 앞에서 잡는다.
            target = next_start - 6.0
            target = max(target, prev_end + 4.0)
        elif alias_type in {"cv", "cv_head"}:
            # CV 계열은 이전 음절 끝으로 끌어당기지 않고,
            # 현재 자음 onset 직전으로 스냅해 앞 발음 유입을 줄인다.
            target = next_start - 4.0
            target = max(target, prev_end + 3.0)
        else:
            # VCV 등은 이전 음절 끝 경계 근처로 당겨서 공백 정렬을 방지한다.
            target = prev_end

    delta = target - pre_abs
    if abs(delta) < 2.0:
        return offset, consonant, cutoff, pre, ovl

    offset = max(float(offset) + delta, 0.0)
    return validate_oto_params(offset, consonant, cutoff, pre, ovl)


def generate_openutau_aliases(base_alias):
    """OpenUtau 호환을 위해 동일 발음의 대체 에일리어스를 추가 생성합니다."""
    aliases = {base_alias}
    parts = base_alias.split()

    if len(parts) == 2:
        v, c = parts[0], parts[1]
        


        batchim_map = {
            'g': 'k', 'gg': 'k', 'k': 'k',
            'd': 't', 'dd': 't', 's': 't', 'ss': 't', 'j': 't', 'jj': 't', 'ch': 't', 't': 't', 'h': 't',
            'b': 'p', 'bb': 'p', 'p': 'p',
            'r': 'l', 'l': 'l',
            'n': 'n', 'm': 'm', 'ng': 'ng'
        }

        stop_variant_map = {
            'g': ['g', 'k', 'gg', 'kk'],
            'gg': ['gg', 'k', 'g', 'kk'],
            'k': ['k', 'g', 'gg', 'kk'],
            'kk': ['kk', 'k', 'g', 'gg'],
            'd': ['d', 't', 'dd', 'tt'],
            'dd': ['dd', 't', 'd', 'tt'],
            't': ['t', 'd', 'dd', 'tt'],
            'tt': ['tt', 't', 'd', 'dd'],
            'b': ['b', 'p', 'bb', 'pp'],
            'bb': ['bb', 'p', 'b', 'pp'],
            'p': ['p', 'b', 'bb', 'pp'],
            'pp': ['pp', 'p', 'b', 'bb'],
            'j': ['j', 'ch', 'jj'],
            'jj': ['jj', 'j', 'ch'],
            'ch': ['ch', 'j', 'jj'],
        }
        

        aliases.add(f"{v} {c}")
        aliases.add(f"{v}{c}")
        

        if c in ['a','e','i','o','u','eo','eu','ae','oe','wi','ya','yeo','yo','yu','ye','wa','wo','we','weo','eui','ui']:
            aliases.add(f"{v} {c}")
            aliases.add(f"{v}{c}")

            aliases.add(f"{v} -")
            aliases.add(f"{v}-")
        else:

            if c in batchim_map:
                mapped_c = batchim_map[c]

                aliases.add(f"{v} {mapped_c}")
                aliases.add(f"{v}{mapped_c}")
                aliases.add(f"{v} {mapped_c.upper()}")
                aliases.add(f"{v}{mapped_c.upper()}")
                

                aliases.add(f"{v} {mapped_c}-")
                aliases.add(f"{v}{mapped_c}-")
                aliases.add(f"{mapped_c}-")
                if c != mapped_c:
                    aliases.add(f"{v} {c}-")
                    aliases.add(f"{v}{c}-")
                    aliases.add(f"{c}-")

            c_lower = c.lower()
            if c_lower in stop_variant_map:
                for vc in stop_variant_map[c_lower]:
                    aliases.add(f"{v} {vc}")
                    aliases.add(f"{v}{vc}")
                    aliases.add(f"{v} {vc.upper()}")
                    aliases.add(f"{v}{vc.upper()}")

    elif len(parts) == 1:
        cv = parts[0]

        aliases.add(f"- {cv}")
        aliases.add(f"-{cv}")
        

        aliases.add(f"{cv}-")
        aliases.add(f"{cv} -")
        

        if cv == 'eui':
            aliases.add('eu i')
            aliases.update(['ui', '- ui', '-ui', 'ui -', 'ui-'])
        elif cv.startswith('-') and 'eui' in cv:
            aliases.add(cv.replace('eui', 'eu i'))

    return list(aliases)


def _parse_oto_line_profile(line):
    s = (line or "").strip()
    if not s or "=" not in s:
        return None
    wav_name, rest = s.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return None
    try:
        offset = float(parts[1].strip())
        cons = float(parts[2].strip())
        cutoff = float(parts[3].strip())
        pre = float(parts[4].strip())
        ovl = float(parts[5].strip())
    except ValueError:
        return None
    return {
        "wav": wav_name.strip(),
        "alias": parts[0].strip(),
        "offset": offset,
        "cons": cons,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
    }


def _extract_base_timing_shape(line):
    """
    base oto 한 줄에서 상대 타이밍 shape를 추출합니다.
    절대 시각 복사보다 pre/cons_gap/cut_gap/ovl_ratio 블렌딩에 사용.
    """
    p = _parse_oto_line_profile(line)
    if not p:
        return None
    pre = max(float(p["pre"]), 0.0)
    cons = max(float(p["cons"]), 0.0)
    cut_abs = abs(float(p["cutoff"]))
    ovl = max(float(p["ovl"]), 0.0)
    off = max(float(p["offset"]), 0.0)

    # 템플릿 없는 자동 생성(0,0,0,0,0) 라인은 shape로 사용하지 않음.
    if pre < 1.0 and cons < 1.0 and cut_abs < 1.0 and ovl < 1.0:
        return None

    cons_gap = max(cons - pre, 8.0)
    cut_gap = max(cut_abs - cons, 16.0)
    ovl_ratio = (ovl / pre) if pre > 1e-6 else 0.30
    return {
        "offset": off,
        "pre": pre,
        "cons_gap": cons_gap,
        "cut_gap": cut_gap,
        "ovl_ratio": _clamp(ovl_ratio, 0.04, 0.86),
    }


def _apply_base_shape_blend(offset, consonant, cutoff, pre, ovl, base_shape, alias_type="cv"):
    if os.environ.get("UTOA_DISABLE_BASE_SHAPE_BLEND", "").strip().lower() in {"1", "true", "yes", "on"}:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)
    if not base_shape:
        return validate_oto_params(offset, consonant, cutoff, pre, ovl)

    if alias_type in {"vc", "vv"}:
        w = 0.30
    elif alias_type == "vcv":
        w = 0.34
    elif alias_type == "cv_head":
        w = 0.22
    else:
        w = 0.24

    pre_t = _clamp(base_shape.get("pre", pre), 12.0, 420.0)
    cons_gap_t = _clamp(base_shape.get("cons_gap", max(consonant - pre, 10.0)), 8.0, 260.0)
    cut_gap_t = _clamp(base_shape.get("cut_gap", max(abs(cutoff) - consonant, 20.0)), 16.0, 300.0)
    ovl_ratio_t = _clamp(base_shape.get("ovl_ratio", (ovl / pre) if pre > 0 else 0.30), 0.04, 0.86)

    pre_new = _blend(pre, pre_t, w)
    cons_gap_now = max(consonant - pre, 10.0)
    cons_gap_new = _blend(cons_gap_now, cons_gap_t, min(0.42, w + 0.07))
    cons_new = pre_new + cons_gap_new

    cut_gap_now = max(abs(cutoff) - consonant, 20.0)
    cut_gap_new = _blend(cut_gap_now, cut_gap_t, min(0.38, w + 0.03))
    cutoff_new = -(cons_new + cut_gap_new)

    ovl_ratio_now = (ovl / pre) if pre > 1e-6 else 0.30
    ovl_ratio_new = _blend(ovl_ratio_now, ovl_ratio_t, min(0.38, w + 0.04))
    ovl_new = max(0.0, pre_new * _clamp(ovl_ratio_new, 0.04, 0.86))

    return validate_oto_params(offset, cons_new, cutoff_new, pre_new, ovl_new)


def _wav_duration_ms(wav_path):
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
        if sr <= 0:
            return 0.0
        return (n / float(sr)) * 1000.0
    except Exception:
        return 0.0


def _read_wav_mono_np(wav_path):
    if np is None:
        return None, None
    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        return None, None

    if sw == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, None

    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio, sr


def _estimate_f0_voicing_strength(frame, sr):
    """
    간단한 자기상관 기반 유성도(0~1) 추정.
    정확한 F0 추적이 아니라 보정 보조 지표로만 사용합니다.
    """
    if np is None or frame is None or len(frame) < 96 or sr <= 0:
        return 0.0
    x = frame.astype(np.float64)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-4:
        return 0.0

    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if len(ac) < 8:
        return 0.0
    ac0 = float(ac[0])
    if ac0 <= 1e-9:
        return 0.0

    min_lag = max(2, int(sr / 500.0))
    max_lag = min(len(ac) - 1, int(sr / 70.0))
    if max_lag <= min_lag + 1:
        return 0.0

    search = ac[min_lag:max_lag + 1]
    peak_rel = int(np.argmax(search))
    peak = float(search[peak_rel])
    lag = min_lag + peak_rel
    if lag <= 0:
        return 0.0
    f0 = float(sr) / float(lag)
    if f0 < 70.0 or f0 > 500.0:
        return 0.0

    clarity = peak / (ac0 + 1e-9)
    if clarity < 0.25:
        return 0.0
    return float(np.clip((clarity - 0.25) / 0.45, 0.0, 1.0))


def _mel_envelope(audio, sr):
    if np is None or audio is None or sr is None or len(audio) == 0:
        return None
    n_fft = 1024
    hop = max(1, int(sr * 0.005))
    win = min(n_fft, max(256, int(sr * 0.025)))
    window = np.hanning(win).astype(np.float64)

    f_min = 0.0
    f_max = sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, 42)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((40, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, 41):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        if right <= left:
            continue
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / float(center - left)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / float(right - center)

    frames = []
    db_vals = []
    f0_voicing = []
    times = []
    last_voicing = 0.0
    frame_idx = 0
    for st in range(0, max(len(audio) - win + 1, 1), hop):
        fr_raw = audio[st:st + win]
        if len(fr_raw) < win:
            fr_raw = np.concatenate([fr_raw, np.zeros(win - len(fr_raw), dtype=np.float32)], axis=0)

        rms = float(np.sqrt(np.mean(fr_raw.astype(np.float64) ** 2) + 1e-12))
        db_vals.append(20.0 * np.log10(max(rms, 1e-7)))

        # F0는 저가중치 보조 신호만 제공하도록 저밀도 계산.
        if (frame_idx % 3) == 0:
            last_voicing = _estimate_f0_voicing_strength(fr_raw, sr)
        f0_voicing.append(last_voicing)

        fr = fr_raw.astype(np.float64) * window
        if n_fft > win:
            fr = np.pad(fr, (0, n_fft - win))
        spec = np.fft.rfft(fr)
        power = (spec.real ** 2 + spec.imag ** 2)
        mel = fb @ power
        frames.append(np.log1p(np.maximum(mel, 0.0)).mean())
        times.append(st * 1000.0 / sr)
        frame_idx += 1

    if not frames:
        return None

    e = np.array(frames, dtype=np.float64)
    p10 = float(np.percentile(e, 10))
    p90 = float(np.percentile(e, 90))
    span = max(p90 - p10, 1e-6)
    en = (e - p10) / span
    en = np.clip(en, 0.0, 1.0)
    db_arr = np.array(db_vals, dtype=np.float64) if db_vals else np.zeros_like(en)
    f0v_arr = np.array(f0_voicing, dtype=np.float64) if f0_voicing else np.zeros_like(en)
    if len(f0v_arr) >= 3:
        f0v_arr = np.convolve(f0v_arr, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
    f0v_arr = np.clip(f0v_arr, 0.0, 1.0)

    db_p20 = float(np.percentile(db_arr, 20)) if len(db_arr) else -60.0
    db_sil_th = max(-58.0, min(-28.0, db_p20 + 6.0))
    return {
        "times_ms": np.array(times, dtype=np.float64),
        "energy": en,
        "span": span,
        "db_db": db_arr,
        "db_silence_th": float(db_sil_th),
        "f0_voicing": f0v_arr,
    }


def _find_wav_path_for_name(wav_name, wav_dir, wav_index):
    cands = [
        os.path.join(wav_dir, wav_name),
        os.path.join(wav_dir, os.path.basename(wav_name)),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    key = normalize_key(wav_name)
    return wav_index.get(key, "")


def _nearest_time_index(times_ms, t_ms):
    if np is None or times_ms is None or len(times_ms) == 0:
        return -1
    idx = int(np.searchsorted(times_ms, t_ms))
    if idx <= 0:
        return 0
    if idx >= len(times_ms):
        return len(times_ms) - 1
    prev_i = idx - 1
    if abs(float(times_ms[idx]) - t_ms) < abs(float(times_ms[prev_i]) - t_ms):
        return idx
    return prev_i


def _apply_soft_mel_offset_cutoff_guard(
    offset,
    consonant,
    cutoff,
    pre,
    ovl,
    alias_type,
    mel_ctx=None,
    onset_hint="",
    alias_text="",
):
    """
    이른 단계 soft guard:
    - dB 최소 임계값 + mel 에너지로 무음/유음 전이를 감지
    - F0 유성도는 낮은 가중치(보조)로만 반영
    """
    if np is None or not mel_ctx:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
    if alias_type not in {"cv", "cv_head", "vcv"}:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    t_ms = mel_ctx.get("times_ms")
    en = mel_ctx.get("energy")
    db_arr = mel_ctx.get("db_db")
    f0v_arr = mel_ctx.get("f0_voicing")
    db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
    if t_ms is None or en is None or len(t_ms) < 8 or len(en) != len(t_ms):
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0
    if db_arr is None or len(db_arr) != len(en):
        db_arr = np.zeros_like(en, dtype=np.float64)
    if f0v_arr is None or len(f0v_arr) != len(en):
        f0v_arr = np.zeros_like(en, dtype=np.float64)

    pre_abs = float(offset) + float(pre)
    cons_abs = float(offset) + float(consonant)
    cut_abs = float(offset) + abs(float(cutoff))
    if cut_abs <= pre_abs + 18.0:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    sound_mask = (db_arr > (db_sil_th + 1.5)) & (en > 0.14)
    silence_mask = (db_arr <= db_sil_th) | (en <= 0.10)

    off_idx = _nearest_time_index(t_ms, offset)
    pre_idx = _nearest_time_index(t_ms, pre_abs)
    cut_idx = _nearest_time_index(t_ms, cut_abs)
    if min(off_idx, pre_idx, cut_idx) < 0:
        return offset, consonant, cutoff, pre, ovl, 0.0, 0.0

    offset_shift_ms = 0.0
    cutoff_shift_ms = 0.0
    hint = (onset_hint or "").strip().lower()
    if hint in {"ɯ", "a", "i", "u", "e", "o"}:
        hint = ""
    if not hint and alias_text:
        parts = [p.strip().lower() for p in str(alias_text).split() if p.strip()]
        token = ""
        if parts:
            if alias_type in {"cv_head", "vcv"} and len(parts) >= 2:
                token = parts[1]
            else:
                token = parts[-1]
        token = re.sub(r"[^a-z]", "", token)
        if token:
            m = re.match(r"^([bcdfghjklmnpqrstvwxyz]+)", token)
            if m:
                hint = m.group(1)
    # 유성/비음 계열(m,n,r,l,w,y...)은 멜 저역 에너지가 약해
    # offset guard가 모음 시작으로 과도 이동할 수 있어 보수적으로 처리한다.
    low_energy_voiced = hint in {
        "m", "n", "ny", "ng", "r", "l", "ry", "w", "y", "j",
        "g", "d", "b", "z", "dz", "v", "gy", "dy", "by",
        "ɴ", "ŋ", "ɲ", "ɾ", "ɹ",
    } or hint.startswith("m")

    # ---- soft offset guard ----
    if not low_energy_voiced:
        off_silent = bool(silence_mask[off_idx])
        pre_sound = bool(sound_mask[pre_idx] or (en[pre_idx] > 0.20))
        if off_silent and pre_sound:
            lo = max(0, pre_idx - 120)
            seg = sound_mask[lo:pre_idx + 1]
            if np.any(seg):
                rel = int(np.where(seg)[0][0])
                sound_start_idx = lo + rel
                target_offset = float(t_ms[sound_start_idx]) - 12.0
                target_offset = max(0.0, min(pre_abs - 18.0, target_offset))
                new_offset = _blend(offset, target_offset, 0.36)
                offset_shift_ms = float(new_offset - offset)
                offset = new_offset
                pre = max(pre_abs - offset, 0.0)
                consonant = max(cons_abs - offset, pre + 8.0)

    # ---- soft cutoff guard ----
    cut_idx = _nearest_time_index(t_ms, cut_abs)
    cut_sound = bool(sound_mask[cut_idx] or (en[cut_idx] > 0.22))
    if cut_sound:
        start_idx = _nearest_time_index(t_ms, pre_abs + 16.0)
        if start_idx < 0:
            start_idx = 0
        if start_idx < cut_idx:
            seg = np.where(silence_mask[start_idx:cut_idx + 1])[0]
            if len(seg) > 0:
                last_sil_idx = int(start_idx + seg[-1])
                target_cut_abs = float(t_ms[last_sil_idx]) + 4.0
                target_cut_abs = max(pre_abs + 20.0, min(target_cut_abs, cut_abs))
                if target_cut_abs < cut_abs - 8.0:
                    # F0 유성도는 보조(저가중치)로만 반영
                    f0v = float(f0v_arr[last_sil_idx])
                    blend_w = 0.42 - (0.08 * f0v)
                    blend_w = max(0.26, min(0.44, blend_w))
                    new_cut_abs = _blend(cut_abs, target_cut_abs, blend_w)
                    cutoff_shift_ms = float(cut_abs - new_cut_abs)
                    cut_abs = new_cut_abs
                    cutoff = -(cut_abs - offset)
                    consonant = min(consonant, (cut_abs - offset) - 10.0)
                    consonant = max(consonant, pre + 8.0)

    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
    return offset, consonant, cutoff, pre, ovl, offset_shift_ms, cutoff_shift_ms


def _apply_kr_mel_refine_to_oto_file(oto_path, wav_dir, custom_map=None):
    """
    멜 스펙트로그램 에너지 골짜기를 이용해 CV 계열 cutoff 과연장을 후처리 보정합니다.
    """
    if os.environ.get("UTOA_DISABLE_MEL_REFINER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return 0
    if np is None or not oto_path or not os.path.exists(oto_path) or not os.path.isdir(wav_dir):
        return 0

    raw_lines = []
    parsed = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip("\n")
            raw_lines.append(line)
            row = _parse_oto_line_profile(line)
            if row:
                parsed.append((idx, row))
    if not parsed:
        return 0

    wav_index = {}
    try:
        for fn in os.listdir(wav_dir):
            if fn.lower().endswith(".wav"):
                wav_index[normalize_key(fn)] = os.path.join(wav_dir, fn)
    except Exception:
        pass

    by_wav = {}
    for line_idx, row in parsed:
        by_wav.setdefault(row["wav"], []).append((line_idx, row))

    mel_cache = {}
    changed = 0
    for wav_name, rows in by_wav.items():
        wav_path = _find_wav_path_for_name(wav_name, wav_dir, wav_index)
        if not wav_path:
            continue
        mel_ctx = mel_cache.get(wav_path)
        if mel_ctx is None:
            audio, sr = _read_wav_mono_np(wav_path)
            mel_ctx = _mel_envelope(audio, sr)
            mel_cache[wav_path] = mel_ctx
        if not mel_ctx:
            continue

        t_ms = mel_ctx["times_ms"]
        en = mel_ctx["energy"]
        db_arr = mel_ctx.get("db_db")
        f0v_arr = mel_ctx.get("f0_voicing")
        db_sil_th = float(mel_ctx.get("db_silence_th", -42.0))
        if db_arr is None or len(db_arr) != len(en):
            db_arr = np.zeros_like(en, dtype=np.float64)
        if f0v_arr is None or len(f0v_arr) != len(en):
            f0v_arr = np.zeros_like(en, dtype=np.float64)
        if len(t_ms) < 8:
            continue

        for i, (_line_idx, row) in enumerate(rows):
            alias = row["alias"]
            alias_type = classify_alias(alias, custom_map)
            if alias_type not in {"cv", "cv_head"}:
                continue

            off = float(row["offset"])
            pre_abs = off + float(row["pre"])
            cons_abs = off + float(row["cons"])
            cut_abs = off + abs(float(row["cutoff"]))
            if cut_abs <= pre_abs + 24.0:
                continue

            next_anchor = None
            for j in range(i + 1, len(rows)):
                a2 = rows[j][1]["alias"]
                t2 = classify_alias(a2, custom_map)
                if t2 in {"cv", "cv_head", "vcv", "mono"}:
                    next_anchor = float(rows[j][1]["offset"]) + float(rows[j][1]["pre"])
                    break

            search_start = pre_abs + 14.0
            search_end = cut_abs - 8.0
            if next_anchor is not None:
                search_end = min(search_end, next_anchor - 8.0)
            if search_end <= search_start + 25.0:
                continue

            mask = np.where((t_ms >= search_start) & (t_ms <= search_end))[0]
            if len(mask) < 5:
                continue

            local_e = en[mask]
            local_db = db_arr[mask]
            local_f0v = f0v_arr[mask]
            silence_flags = local_db <= db_sil_th
            candidate_mask = mask[silence_flags] if np.any(silence_flags) else mask

            # dB+mel을 주축으로 골짜기 선택, F0는 낮은 가중치(보조)로만 반영.
            best_idx = int(candidate_mask[0])
            best_score = -1e9
            for ci in candidate_mask:
                e_v = float(en[ci])
                db_v = float(db_arr[ci])
                f0_v = float(f0v_arr[ci])
                silence_bonus = 0.28 if db_v <= db_sil_th else 0.0
                score = (1.0 - e_v) + silence_bonus - (0.08 * f0_v)
                if score > best_score:
                    best_score = score
                    best_idx = int(ci)

            valley_t = float(t_ms[best_idx])
            valley_e = float(en[best_idx])
            valley_db = float(db_arr[best_idx])
            valley_f0v = float(f0v_arr[best_idx])
            cut_idx = int(np.argmin(np.abs(t_ms - cut_abs)))
            cut_e = float(en[cut_idx])
            cut_db = float(db_arr[cut_idx])
            contrast = cut_e - valley_e
            db_drop = cut_db - valley_db

            if contrast < 0.12 and db_drop < 2.5:
                continue
            if valley_e > 0.38 and valley_db > (db_sil_th + 6.0):
                continue
            if valley_f0v > 0.70 and contrast < 0.18:
                continue
            if valley_t >= cut_abs - 12.0:
                continue

            target_cut_abs = valley_t + 2.0
            min_cut_abs = pre_abs + 20.0
            if target_cut_abs <= min_cut_abs:
                continue

            new_cons_abs = min(cons_abs, target_cut_abs - 12.0)
            new_cons_abs = max(new_cons_abs, pre_abs + 8.0)
            row["cons"] = max(new_cons_abs - off, 0.0)
            row["cutoff"] = -(target_cut_abs - off)
            changed += 1

    if changed <= 0:
        return 0

    replace_map = {line_idx: row for line_idx, row in parsed}
    out_lines = []
    for i, line in enumerate(raw_lines):
        row = replace_map.get(i)
        if not row:
            out_lines.append(line)
            continue
        o, c, ct, p, ov = validate_oto_params(
            row["offset"], row["cons"], row["cutoff"], row["pre"], row["ovl"]
        )
        out_lines.append(f"{row['wav']}={row['alias']},{o:.2f},{c:.2f},{ct:.2f},{p:.2f},{ov:.2f}")

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _median(vals):
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _safe_ratio(num, den, fallback=0.0):
    if den is None or den == 0:
        return fallback
    return float(num) / float(den)


def _blend(a, b, w):
    w2 = max(0.0, min(1.0, float(w)))
    return (1.0 - w2) * float(a) + w2 * float(b)


def _clamp(v, lo, hi):
    return max(float(lo), min(float(hi), float(v)))


def _default_kr_profile_cache_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profile_dir = os.path.join(base_dir, "assets", "profiles")
    os.makedirs(profile_dir, exist_ok=True)
    return os.path.join(profile_dir, "kr_oto_reference_profile.json")


def _build_kr_reference_profile_from_dirs(ref_dirs, custom_map=None):
    bucket_values = {}
    total_rows = 0
    total_wavs = 0

    for ref_dir in ref_dirs:
        if not ref_dir or not os.path.isdir(ref_dir):
            continue
        oto_path = os.path.join(ref_dir, "oto.ini")
        if not os.path.exists(oto_path):
            continue

        rows_by_wav = {}
        with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                row = _parse_oto_line_profile(raw)
                if not row:
                    continue
                rows_by_wav.setdefault(row["wav"], []).append(row)

        for wav_name, rows in rows_by_wav.items():
            wav_path = os.path.join(ref_dir, wav_name)
            dur_ms = _wav_duration_ms(wav_path)
            if dur_ms <= 0:
                continue
            total_wavs += 1
            for idx, row in enumerate(rows):
                alias_type = classify_alias(row["alias"], custom_map)
                b = bucket_values.setdefault(alias_type, {
                    "pre": [],
                    "cons_gap": [],
                    "cut_gap": [],
                    "ovl_ratio": [],
                    "head_off_ratio": [],
                })
                pre = max(row["pre"], 0.0)
                cons = max(row["cons"], pre)
                cut_abs = abs(row["cutoff"])
                b["pre"].append(_clamp(pre, 0.0, 360.0))
                b["cons_gap"].append(_clamp(max(cons - pre, 10.0), 10.0, 220.0))
                b["cut_gap"].append(_clamp(max(cut_abs - cons, 20.0), 20.0, 180.0))
                b["ovl_ratio"].append(_clamp(_safe_ratio(row["ovl"], pre, fallback=0.0), 0.05, 0.78))
                if idx == 0:
                    b["head_off_ratio"].append(_clamp(_safe_ratio(row["offset"], dur_ms, fallback=0.0), 0.0, 0.35))
                total_rows += 1

    if total_rows < 100:
        return None

    buckets = {}
    for alias_type, vals in bucket_values.items():
        n = len(vals["pre"])
        if n < 8:
            continue
        buckets[alias_type] = {
            "n": n,
            "pre": _median(vals["pre"]),
            "cons_gap": _median(vals["cons_gap"]),
            "cut_gap": _median(vals["cut_gap"]),
            "ovl_ratio": _median(vals["ovl_ratio"]),
            "head_off_ratio": _median(vals["head_off_ratio"]) if vals["head_off_ratio"] else None,
        }

    if not buckets:
        return None

    return {
        "version": 1,
        "language": "korean",
        "source_count": len([d for d in ref_dirs if d and os.path.isdir(d)]),
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rows": total_rows,
        "wavs": total_wavs,
        "buckets": buckets,
    }


def _load_kr_reference_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "buckets" not in obj:
            return None
        return obj
    except Exception:
        return None


def _save_kr_reference_profile(path, profile):
    if not path or not profile:
        return False
    try:
        data = dict(profile)
        # Ensure profile file stores abstract stats only.
        data.pop("source_dirs", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _normalize_alias_for_profile(alias):
    a = re.sub(r"\s+", " ", (alias or "").strip().lower())
    # Treat trailing suffix like _C4 as metadata.
    a = re.sub(r"_[a-z0-9]{1,8}$", "", a)
    return a


def _read_text_with_fallback(path):
    try:
        raw = open(path, "rb").read()
    except Exception:
        return ""
    for enc in ("utf-8-sig", "cp932", "utf-8", "euc-kr", "latin-1"):
        try:
            text = raw.decode(enc)
            if "=" in text:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_kr_oto_rows_for_profile(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    text = _read_text_with_fallback(path)
    for raw in text.splitlines():
        row = _parse_oto_line_profile(raw)
        if not row:
            continue
        row["wav_norm"] = row["wav"].strip().lower()
        row["alias_norm"] = _normalize_alias_for_profile(row["alias"])
        rows.append(row)
    return rows


def _occurrence_map(rows):
    m = {}
    counters = {}
    for row in rows:
        base_key = (row["wav_norm"], row["alias_norm"])
        idx = counters.get(base_key, 0)
        counters[base_key] = idx + 1
        m[(base_key[0], base_key[1], idx)] = row
    return m


def train_kr_autotune_profile(auto_oto_path, manual_oto_path, custom_phonemes_path=""):
    """부분 수동 OTO와 자동 OTO를 매칭해 한국어 델타 기반 프로파일을 학습합니다."""
    auto_rows = _read_kr_oto_rows_for_profile(auto_oto_path)
    manual_rows = _read_kr_oto_rows_for_profile(manual_oto_path)
    if not auto_rows or not manual_rows:
        return None

    custom_map = load_custom_phonemes(custom_phonemes_path)
    auto_map = _occurrence_map(auto_rows)
    manual_map = _occurrence_map(manual_rows)

    fields = ["offset", "cons", "cutoff", "pre", "ovl"]
    buckets = {}
    matched = 0
    for key, a_row in auto_map.items():
        m_row = manual_map.get(key)
        if not m_row:
            continue
        alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", a_row["alias"])
        alias_type = classify_alias(alias_for_cls, custom_map)
        b = buckets.setdefault(alias_type, {f: [] for f in fields})
        for f in fields:
            b[f].append(float(m_row[f]) - float(a_row[f]))
        matched += 1

    if matched < 8:
        return None

    profile = {
        "version": 1,
        "language": "korean",
        "mode": "delta",
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "matched_pairs": matched,
        "source_auto_name": os.path.basename(auto_oto_path),
        "source_manual_name": os.path.basename(manual_oto_path),
        "buckets": {},
    }
    for alias_type, vals in buckets.items():
        n = len(vals["offset"])
        if n < 3:
            continue
        profile["buckets"][alias_type] = {
            "n": n,
            "offset": _median(vals["offset"]),
            "cons": _median(vals["cons"]),
            "cutoff": _median(vals["cutoff"]),
            "pre": _median(vals["pre"]),
            "ovl": _median(vals["ovl"]),
        }
    if not profile["buckets"]:
        return None
    return profile


def save_kr_autotune_profile(path, profile):
    if not path or not profile:
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_kr_autotune_profile(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "buckets" not in obj:
            return None
        return obj
    except Exception:
        return None


def apply_kr_autotune_profile_to_oto(oto_path, profile, custom_phonemes_path=""):
    """학습된 델타 프로파일을 OTO에 적용합니다. 반환값은 변경 라인 수."""
    if not oto_path or not os.path.exists(oto_path):
        return 0
    if isinstance(profile, str):
        profile = load_kr_autotune_profile(profile)
    if not profile:
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    custom_map = load_custom_phonemes(custom_phonemes_path)
    changed = 0
    out_lines = []
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = _parse_oto_line_profile(raw)
            if not row:
                out_lines.append(raw.rstrip("\n"))
                continue

            alias_for_cls = re.sub(r"_[A-Za-z0-9]{1,8}$", "", row["alias"])
            alias_type = classify_alias(alias_for_cls, custom_map)
            stat = buckets.get(alias_type)
            if not stat:
                out_lines.append(raw.rstrip("\n"))
                continue

            n = float(stat.get("n", 0))
            if n < 3:
                out_lines.append(raw.rstrip("\n"))
                continue

            w = min(1.0, max(0.25, n / 24.0))
            offset = row["offset"] + _clamp(stat.get("offset", 0.0), -140.0, 140.0) * w
            cons = row["cons"] + _clamp(stat.get("cons", 0.0), -160.0, 160.0) * w
            cutoff = row["cutoff"] + _clamp(stat.get("cutoff", 0.0), -180.0, 180.0) * w
            pre = row["pre"] + _clamp(stat.get("pre", 0.0), -140.0, 140.0) * w
            ovl = row["ovl"] + _clamp(stat.get("ovl", 0.0), -120.0, 120.0) * w
            offset, cons, cutoff, pre, ovl = validate_oto_params(offset, cons, cutoff, pre, ovl)

            if (
                abs(offset - row["offset"]) > 1e-6
                or abs(cons - row["cons"]) > 1e-6
                or abs(cutoff - row["cutoff"]) > 1e-6
                or abs(pre - row["pre"]) > 1e-6
                or abs(ovl - row["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def _resolve_kr_reference_dirs():
    raw = os.environ.get("UTOA_KR_PROFILE_DIRS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(";") if x.strip()]


def _profile_weight(n):
    # Keep this conservative; profile should nudge, not overwrite.
    return max(0.12, min(0.38, float(n) / 220.0))


def _apply_kr_profile_to_oto_file(oto_path, wav_dir, profile, custom_map=None):
    if not profile or not oto_path or not os.path.exists(oto_path):
        return 0
    buckets = profile.get("buckets") or {}
    if not isinstance(buckets, dict) or not buckets:
        return 0

    rows_by_wav = {}
    with open(oto_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            row = _parse_oto_line_profile(raw)
            if not row:
                continue
            rows_by_wav.setdefault(row["wav"], []).append(row)

    out_lines = []
    changed = 0
    for wav_name, rows in rows_by_wav.items():
        dur_ms = _wav_duration_ms(os.path.join(wav_dir, wav_name))
        for idx, row in enumerate(rows):
            alias_type = classify_alias(row["alias"], custom_map)
            stat = buckets.get(alias_type)
            if not stat:
                out_lines.append(
                    f"{row['wav']}={row['alias']},{row['offset']:.2f},{row['cons']:.2f},{row['cutoff']:.2f},{row['pre']:.2f},{row['ovl']:.2f}"
                )
                continue

            n = float(stat.get("n", 0))
            w = _profile_weight(n)
            offset = row["offset"]
            pre_t = _clamp(stat.get("pre", row["pre"]), 0.0, 360.0)
            pre = _blend(row["pre"], pre_t, w)
            cons_gap_now = max(row["cons"] - row["pre"], 10.0)
            cons_gap_t = _clamp(stat.get("cons_gap", cons_gap_now), 10.0, 220.0)
            cons_gap = _blend(cons_gap_now, cons_gap_t, w)
            cons = pre + cons_gap
            cut_gap_now = max(abs(row["cutoff"]) - row["cons"], 20.0)
            cut_gap_t = _clamp(stat.get("cut_gap", cut_gap_now), 20.0, 180.0)
            cut_w = min(0.12, w * 0.35)
            cut_gap = _blend(cut_gap_now, cut_gap_t, cut_w)
            cutoff = -(cons + cut_gap)
            ovl_ratio_now = _safe_ratio(row["ovl"], row["pre"], fallback=0.0)
            ovl_ratio_t = _clamp(stat.get("ovl_ratio", ovl_ratio_now), 0.05, 0.78)
            ovl_ratio = _blend(ovl_ratio_now, ovl_ratio_t, w)
            ovl = max(0.0, pre * max(0.10, min(0.80, ovl_ratio)))
            pre, cons, cutoff, ovl = _apply_kr_consonant_timing_shaping(
                row.get("alias", ""), pre, cons, cutoff, ovl
            )

            if idx == 0 and dur_ms > 0 and stat.get("head_off_ratio") is not None:
                target_ratio = _clamp(stat["head_off_ratio"], 0.0, 0.35)
                target_off = max(0.0, min(dur_ms * 0.7, dur_ms * target_ratio))
                offset = _blend(offset, target_off, min(0.30, w))

            offset, cons, cutoff, pre, ovl = validate_oto_params(offset, cons, cutoff, pre, ovl)
            if (
                abs(offset - row["offset"]) > 1e-6
                or abs(cons - row["cons"]) > 1e-6
                or abs(cutoff - row["cutoff"]) > 1e-6
                or abs(pre - row["pre"]) > 1e-6
                or abs(ovl - row["ovl"]) > 1e-6
            ):
                changed += 1
            out_lines.append(
                f"{row['wav']}={row['alias']},{offset:.2f},{cons:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
            )

    with open(oto_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    return changed


def generate_oto(
    tg_folder,
    tpl_path,
    out_path,
    params=None,
    generate_openutau=False,
    gen_missing_vowels=False,
    fallback_format='cvvc',
    custom_phonemes_path='',
    alias_suffix='',
    auto_format=None,
    callback=None
):
    """TextGrid와 템플릿/자동 포맷 정보를 사용해 최종 OTO를 생성합니다."""

    try:
        import textgrid
    except ImportError:
        err = "textgrid 모듈이 설치되어 있지 않습니다. `pip install textgrid`를 실행해 주세요."
        logger.error(err)
        if callback:
            callback(err)
        return 0, 0, [err]

    if params is None:
        params = DEFAULT_PARAMS.copy()


    if auto_format:
        af = str(auto_format).strip().lower()
        if af.startswith('cvc'):
            fallback_format = 'cvc'
        elif af.startswith('cvvc'):
            fallback_format = 'cvvc'
        elif af.startswith('vcv'):
            fallback_format = 'vcv'

    auto_gen_format = (fallback_format or "cvvc").strip().lower()
    if auto_gen_format not in {"cvvc", "vcv"}:
        msg = (
            f"⚠ 자동 에일리어스 생성은 현재 CVVC/VCV만 지원합니다. "
            f"{auto_gen_format.upper()} -> CVVC로 전환합니다."
        )
        if callback:
            callback(msg)
        auto_gen_format = "cvvc"

    def log(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    errors = []
    skipped_entries = []

    def _record_unset(reason, fname, line):
        raw = (line or "").rstrip("\n")
        alias = ""
        if "=" in raw:
            rhs = raw.split("=", 1)[1]
            alias = rhs.split(",", 1)[0].strip()
        skipped_entries.append({
            "reason": reason,
            "file": fname or "",
            "alias": alias,
            "line": raw,
        })

    def _record_unset_lines(reason, fname, src_lines):
        for raw in (src_lines or []):
            _record_unset(reason, fname, raw)

    def _log_unset_summary():
        total_unset = len(skipped_entries)
        if total_unset <= 0:
            log("[Auto-OTO] 자동 설정 제외 항목: 0")
            return
        by_reason = {}
        for it in skipped_entries:
            r = it["reason"]
            by_reason[r] = by_reason.get(r, 0) + 1
        log(f"[Auto-OTO] 자동 설정 제외 항목: {total_unset}건")
        for reason, count in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
            log(f"  - {reason}: {count}")
            for it in skipped_entries:
                if it["reason"] != reason:
                    continue
                alias_txt = it["alias"] if it["alias"] else "<empty>"
                log(f"    예시: {it['file']} | alias={alias_txt}")
                break

    kr_profile = None
    profile_path = _default_kr_profile_cache_path()
    try:
        env_profile = os.environ.get("UTOA_KR_PROFILE_JSON", "").strip()
        if env_profile:
            kr_profile = _load_kr_reference_profile(env_profile)
            if kr_profile:
                log(f"[KR-Profile] 외부 프로파일 로드: {env_profile}")

        if not kr_profile:
            kr_profile = _load_kr_reference_profile(profile_path)

        # Optional rebuild path for local experiments only (env-driven).
        if not kr_profile:
            ref_dirs = _resolve_kr_reference_dirs()
            if ref_dirs:
                trained = _build_kr_reference_profile_from_dirs(ref_dirs, custom_map=None)
                if trained and _save_kr_reference_profile(profile_path, trained):
                    kr_profile = trained
                    log(f"[KR-Profile] 로컬 샘플 기반 프로파일 생성: {profile_path} (rows={trained.get('rows', 0)})")

        # Default path: use abstract preset from python module.
        if not kr_profile:
            kr_profile = get_kr_profile_preset()
            _save_kr_reference_profile(profile_path, kr_profile)
            log(f"[KR-Profile] 추상화 프리셋 로드: {profile_path}")

        if kr_profile:
            b_count = len((kr_profile.get("buckets") or {}))
            log(f"[KR-Profile] 기준 프로파일 적용 준비: buckets={b_count}")
    except Exception as e:
        log(f"[KR-Profile] 프로파일 로드 실패: {e}")

    if tpl_path and not os.path.exists(tpl_path):
        log(f"⚠ 템플릿 파일을 찾을 수 없습니다: {tpl_path}")
        log(f"⚡ OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
        tpl_path = ""


    VC_CONSONANT_RATIO = params.get('VC_CONSONANT_RATIO', 0.5) if params else 0.5
    VC_VOWEL_START = params.get('VC_VOWEL_START', 0.3) if params else 0.3
    VC_PRE_OFFSET = params.get('VC_PRE_OFFSET', 25) if params else 25
    VC_OVL_RATIO = params.get('VC_OVL_RATIO', 0.3) if params else 0.3
    CV_PRE_RATIO = params.get('CV_PRE_RATIO', 1.0) if params else 1.0
    CV_OVL_RATIO = params.get('CV_OVL_RATIO', 0.4) if params else 0.4
    DIPHTHONG_CV_PRE_RATIO = params.get('DIPHTHONG_CV_PRE_RATIO', 0.35) if params else 0.35
    DIPHTHONG_CV_CONSONANT_RATIO = params.get('DIPHTHONG_CV_CONSONANT_RATIO', 0.6) if params else 0.6
    DIPHTHONG_VC_VOWEL_START = params.get('DIPHTHONG_VC_VOWEL_START', 0.3) if params else 0.3
    DIPHTHONG_VC_CONSONANT = params.get('DIPHTHONG_VC_CONSONANT', 0.5) if params else 0.5


    template_lines = []
    if tpl_path:
        lines, detected_enc, warning, err = load_template_oto_lines(
            tpl_path,
            require_utf8=True,
            mode_label="한국어 OTO",
        )
        if err:
            log(err)
            log(f"⚡ 템플릿 로드 실패로 OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환합니다.")
            lines = []
        if warning:
            log(warning)
        template_lines = lines or []


    tg_entries = []
    tg_exact_map = {}
    tg_norm_map = {}
    if os.path.exists(tg_folder):
        for f_name in os.listdir(tg_folder):
            if not f_name.lower().endswith('.textgrid'):
                continue
            base = os.path.splitext(f_name)[0]
            info = {
                'path': os.path.join(tg_folder, f_name),
                'real_name': base + '.wav',
                'base_lower': base.lower(),
                'norm_key': normalize_key(f_name),
            }
            tg_entries.append(info)
            if info['base_lower'] not in tg_exact_map:
                tg_exact_map[info['base_lower']] = info
            tg_norm_map.setdefault(info['norm_key'], []).append(info)

    def _resolve_tg_info(fname):
        wav_name = os.path.basename((fname or '').strip())
        base_lower = os.path.splitext(wav_name)[0].lower()
        if base_lower in tg_exact_map:
            return tg_exact_map[base_lower]

        norm_name = normalize_key(wav_name)
        candidates = tg_norm_map.get(norm_name, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            exact_name = wav_name.lower()
            same_name = [c for c in candidates if c['real_name'].lower() == exact_name]
            if len(same_name) == 1:
                return same_name[0]
            log(f"파일명 매핑 충돌: {wav_name} (정규화 키 {norm_name}, 후보 {len(candidates)}개) -> 원본 파일명을 유지합니다.")
            return None
        return None

    def _template_match_stats(lines):
        file_names = set()
        for line in (lines or []):
            if "=" not in line:
                continue
            file_names.add(line.split("=", 1)[0].strip())
        total = len(file_names)
        if total == 0:
            return 0, 0, 0.0
        matched = 0
        for fname in file_names:
            if _resolve_tg_info(fname):
                matched += 1
        return matched, total, (matched / float(total))

    final_lines = []


    custom_map = load_custom_phonemes(custom_phonemes_path)


    use_template = bool(template_lines)
    if use_template:
        t_match, t_total, t_ratio = _template_match_stats(template_lines)
        if t_total == 0 or t_match == 0 or t_ratio < 0.25:
            log(
                f"⚠ 템플릿-TextGrid 매칭률 낮음 ({t_match}/{t_total}, {t_ratio:.1%}) "
                f"-> OpenUtau 호환 {auto_gen_format.upper()} 자동 에일리어스 생성으로 전환"
            )
            use_template = False
        else:
            log(f"📌 템플릿 베이스 OTO 사용 ({t_match}/{t_total}, {t_ratio:.1%})")

    if use_template:
        file_groups = {}
        for line in template_lines:
            fname = line.split('=')[0]
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(line)
    else:

        log(f"⚡ 템플릿 없음/미적합 -> OpenUtau 호환 {auto_gen_format.upper()} 형식으로 에일리어스를 자동 생성합니다.")
        file_groups = {}
        try:
            from core.lab_generator import decompose_hangul_to_roman
        except ImportError:
            pass # fallback
        
        for tg_info in tg_entries:
            tg_path = tg_info['path']
            real_name = tg_info['real_name']
            

            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
                w_tier = None
                for t in tg:
                    if hasattr(t, 'name') and t.name == 'words':
                        w_tier = t
                        break
                if not w_tier:
                    continue
                    
                wd_intervals = [i for i in w_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]
                if not wd_intervals:
                    continue
                    
                lines = []

                base_name = os.path.splitext(real_name)[0].lower()
                is_long = base_name.endswith('long') or len(wd_intervals) == 1
                
                if is_long:

                    for w in wd_intervals:
                        roman_parts = []
                        for ch in w.mark:
                            roman_parts.extend(decompose_hangul_to_roman(ch))
                        roman = "".join(roman_parts).lower()
                        # CV / Mono
                        lines.append(f"{real_name}={roman.capitalize() if roman not in KR_VOWELS else roman},0,0,0,0,0")
                else:
                    coda_alias_map = {
                        "ng": "NG",
                        "n": "N",
                        "l": "L",
                        "r": "L",
                        "m": "M",
                        "k": "K",
                        "g": "K",
                        "t": "T",
                        "d": "T",
                        "s": "T",
                        "ss": "T",
                        "j": "T",
                        "jj": "T",
                        "ch": "T",
                        "h": "H",
                        "p": "P",
                        "b": "P",
                    }

                    for idx, w in enumerate(wd_intervals):
                        roman_parts = []
                        for ch in w.mark:
                            roman_parts.extend(decompose_hangul_to_roman(ch))
                        roman = "".join(roman_parts).lower()
                        
                        if auto_gen_format == 'vcv':
                            if idx == 0:
                                lines.append(f"{real_name}=- {roman},0,0,0,0,0")
                            else:
                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                _, prev_vowel, prev_coda = _split_kr_syllable_parts(prev_roman)
                                prev_coda_alias = coda_alias_map.get((prev_coda or "").lower(), "")
                                
                                if prev_coda_alias:
                                    lines.append(f"{real_name}={prev_coda_alias} {roman},0,0,0,0,0")
                                else:
                                    lines.append(f"{real_name}={(prev_vowel or 'a')} {roman},0,0,0,0,0")
                                
                        elif auto_gen_format == 'cvvc':

                            lines.append(f"{real_name}={roman},0,0,0,0,0")
                            
                            if idx > 0:

                                prev_w = wd_intervals[idx-1]
                                prev_parts = []
                                for ch in prev_w.mark:
                                    prev_parts.extend(decompose_hangul_to_roman(ch))
                                prev_roman = "".join(prev_parts).lower()
                                _, prev_vowel, _ = _split_kr_syllable_parts(prev_roman)
                                prev_vowel = prev_vowel or "a"
                                

                                cur_start_cons = ""
                                for char in roman:
                                    if char in KR_CONSONANTS:
                                        cur_start_cons += char
                                    else:
                                        break
                                
                                if cur_start_cons:
                                    lines.append(f"{real_name}={prev_vowel} {cur_start_cons},0,0,0,0,0")
                                else:

                                    _, cur_vowel, _ = _split_kr_syllable_parts(roman)
                                    lines.append(f"{real_name}={prev_vowel} {(cur_vowel or 'a')},0,0,0,0,0")
                
                if lines:
                    file_groups[real_name] = lines

            except Exception as e:
                log(f"경고: {real_name} 자동 템플릿 생성 실패 ({e})")
                
    processed = 0
    total = len(file_groups)
    wav_root_for_signal = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
    wav_index_for_signal = {}
    try:
        if os.path.isdir(wav_root_for_signal):
            for fn in os.listdir(wav_root_for_signal):
                if fn.lower().endswith(".wav"):
                    wav_index_for_signal[normalize_key(fn)] = os.path.join(wav_root_for_signal, fn)
    except Exception:
        pass
    mel_cache_for_signal = {}

    for fname, lines in file_groups.items():
        tg_info = _resolve_tg_info(fname)

        if not tg_info:
            log(f"경고: {fname}: TextGrid를 찾을 수 없어 원본 라인을 유지합니다.")
            _record_unset_lines("textgrid_missing", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1
            continue
            
        tg_path = tg_info['path']
        real_wav_name = tg_info['real_name']
        wav_path_for_signal = _find_wav_path_for_name(real_wav_name, wav_root_for_signal, wav_index_for_signal)
        mel_ctx_for_file = None
        if wav_path_for_signal:
            mel_ctx_for_file = mel_cache_for_signal.get(wav_path_for_signal)
            if mel_ctx_for_file is None:
                audio_sig, sr_sig = _read_wav_mono_np(wav_path_for_signal)
                mel_ctx_for_file = _mel_envelope(audio_sig, sr_sig)
                mel_cache_for_signal[wav_path_for_signal] = mel_ctx_for_file

        try:
            tg = textgrid.TextGrid.fromFile(tg_path)
            phone_tier = None
            word_tier = None
            for t in tg:
                if isinstance(t, textgrid.IntervalTier):
                    if t.name == 'phones':
                        phone_tier = t
                    elif t.name == 'words':
                        word_tier = t

            if not phone_tier:
                log(f"경고: {fname}: phones tier가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("tier_missing", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue


            ph_intervals_all = [i for i in phone_tier if i.mark.strip() not in ['', 'spn', 'pau']]
            ph_intervals = [i for i in ph_intervals_all if i.mark.strip() not in ['sil']]
            wd_intervals = [i for i in word_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']] if word_tier else []

            if len(ph_intervals) == 0:
                log(f"경고: {fname}: 유효한 음소 구간이 없어 원본 라인을 유지합니다.")
                _record_unset_lines("empty_intervals", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue


            if len(ph_intervals) == 1 and len(wd_intervals) == 1:
                log(f"처리: {fname}: 단모음 파일")
                vowel = ph_intervals[0]
                v_start = vowel.minTime * 1000
                v_end = vowel.maxTime * 1000
                v_len = v_end - v_start
                for line in lines:
                    parts = line.split('=', 1)
                    if len(parts) < 2:
                        _record_unset("malformed_line", fname, line)
                        final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                        continue
                    alias = parts[1].split(',')[0].strip()
                    if not alias:
                        _record_unset("empty_alias", fname, line)
                        preserved = f"{real_wav_name}={parts[1]}"
                        final_lines.append(apply_suffix_to_oto_line(preserved, alias_suffix))
                        continue


                    offset = v_start
                    pre = 0
                    ovl = 0
                    consonant = min(v_len * 0.25, 120)
                    cutoff = -(v_len * 0.8)
                    
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                processed += 1
                continue


            alias_names = []
            for l in lines:
                if "=" not in l:
                    continue
                a = l.split("=", 1)[1].split(",", 1)[0].strip()
                if a:
                    alias_names.append(a)
            if not alias_names:
                log(f"경고: {fname}: 유효한 에일리어스가 없어 원본 라인을 유지합니다.")
                _record_unset_lines("no_valid_alias", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
            file_format = detect_alias_format(alias_names)
            log(f"처리: {fname}: 형식 감지 -> {file_format.upper()}")
            
            is_vc_only = (file_format == 'vc_only')
            is_vcv_file = (file_format == 'vcv')


            try:
                from core.lab_generator import decompose_hangul_to_roman
            except ImportError:
                def decompose_hangul_to_roman(ch):
                    return [ch]
            
            cv_targets = _extract_cv_targets_from_lines(lines, custom_map)
            syllables_info = []
            if wd_intervals:
                for w in wd_intervals:
                    w_start = w.minTime
                    w_end = w.maxTime

                    s_phones = [p for p in ph_intervals if p.minTime >= w_start - 0.01 and p.maxTime <= w_end + 0.01]
                    roman_parts = []
                    for ch in w.mark:
                        roman_parts.extend(decompose_hangul_to_roman(ch))
                    roman_raw = "".join(roman_parts).lower()
                    roman_cv = _kr_cv_kernel(roman_raw)
                    
                    syllables_info.append({
                        'word': w.mark,
                        'roman': roman_raw,
                        'roman_cv': roman_cv,
                        'start_time': w_start,
                        'end_time': w_end,
                        'phones': s_phones
                    })

            alias_based = _build_kr_syllables_from_phone_nuclei(ph_intervals, cv_targets) if cv_targets else None
            if not syllables_info and alias_based:
                syllables_info = alias_based
                log(f"🧭 {fname}: words 티어 없음/실패 → phones 핵 기반 음절 매핑 사용")
            elif syllables_info and alias_based and cv_targets:
                base_score = _score_kr_syllable_mapping(syllables_info, cv_targets)
                alt_score = _score_kr_syllable_mapping(alias_based, cv_targets)
                # TextGrid(words) 결과를 우선 사용하되, alias/filename 기준과 심하게 어긋난 경우만 보정.
                if base_score < 66.0 and alt_score >= 70.0 and alt_score >= (base_score + 8.0):
                    syllables_info = alias_based
                    log(
                        f"🧭 {fname}: 매핑 이탈 보정 적용 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )
                else:
                    log(
                        f"🧭 {fname}: TextGrid(words) 매핑 유지 "
                        f"(base={base_score:.1f}, corrected={alt_score:.1f})"
                    )

            if (not syllables_info) or any(len(s['phones']) == 0 for s in syllables_info):
                log(f"경고: {fname}: 음절-음소 매핑 실패로 원본 라인을 유지합니다.")
                _record_unset_lines("mapping_failed", fname, lines)
                final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
                processed += 1
                continue
                
            romaji_syllables = [s.get('roman_cv') or s.get('roman', '') for s in syllables_info]
            current_w_idx = 0
            cv_seq_idx = 0

            cv_anchor_by_idx = {
                i: _estimate_cv_anchor_from_syllable(syllables_info[i], ph_intervals)
                for i in range(len(syllables_info))
            }
            
            for line_num, line in enumerate(lines):
                parts = line.split('=', 1)
                if len(parts) < 2:
                    _record_unset("malformed_line", fname, line)
                    final_lines.append(apply_suffix_to_oto_line(line, alias_suffix))
                    continue
                alias = parts[1].split(',', 1)[0].strip()
                if not alias:
                    _record_unset("empty_alias", fname, line)
                    preserved = f"{real_wav_name}={parts[1]}"
                    final_lines.append(apply_suffix_to_oto_line(preserved, alias_suffix))
                    continue
                base_shape = _extract_base_timing_shape(line)
                

                alias_type = classify_alias(alias)
                

                if alias_type == 'br':

                    first_ph = ph_intervals[0] if ph_intervals else None
                    last_ph = ph_intervals[-1] if ph_intervals else None
                    if first_ph and last_ph:
                        br_start = first_ph.minTime * 1000
                        br_end = last_ph.maxTime * 1000
                        br_len = br_end - br_start
                    else:
                        br_start = 0
                        br_len = 500
                    offset = max(br_start - 30, 0)
                    pre = 0
                    ovl = 0
                    consonant = min(br_len * 0.3, 100)
                    cutoff = -(br_len * 0.85)
                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                is_vc = alias_type in ('vc', 'vv')
                is_vcv = alias_type == 'vcv'
                is_cv_head = alias_type == 'cv_head'
                is_diph = is_diphthong(alias)
                
                target_clean = _extract_kr_cv_alias_token(alias)




                glottal_kind = _detect_glottal_kind(alias)
                if glottal_kind in ('head', 'tail'):

                    if glottal_kind == 'head':
                        if cv_seq_idx < len(syllables_info):
                            current_w_idx = cv_seq_idx
                            cv_seq_idx = current_w_idx + 1

                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1

                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    _, v_phone = find_vowel_phone(curr_phones)
                    vowel_start = v_phone.minTime * 1000
                    vowel_end = v_phone.maxTime * 1000


                    g_idx = None
                    for i, p in enumerate(ph_intervals):
                        if abs(p.minTime - v_phone.minTime) < 1e-6 and abs(p.maxTime - v_phone.maxTime) < 1e-6:
                            g_idx = i
                            break

                    if glottal_kind == 'tail':
                        next_ph = ph_intervals[g_idx + 1] if g_idx is not None and g_idx + 1 < len(ph_intervals) else None
                        boundary_end = (next_ph.maxTime * 1000) if next_ph else (curr_syl['end_time'] * 1000)
                        glottal_len = max(boundary_end - vowel_end, 60)

                        offset_padding = min(max(glottal_len, 60), 220)
                        offset = max(boundary_end - offset_padding, 0)
                        pre = boundary_end - offset
                        ovl = pre * 0.2
                        consonant = pre + min(glottal_len * 0.3, 30)
                        cutoff = -(consonant + 30)

                    else:
                        prev_ph = ph_intervals[g_idx - 1] if g_idx is not None and g_idx - 1 >= 0 else None
                        glottal_start = (prev_ph.minTime * 1000) if prev_ph else max(vowel_start - 80, 0)
                        boundary = (prev_ph.maxTime * 1000) if prev_ph else vowel_start

                        offset = max(glottal_start - 30, 0)
                        pre = boundary - offset
                        ovl = pre * 0.3
                        vowel_len = max(vowel_end - vowel_start, 80)
                        added_cons = min(max(vowel_len * 0.5, 80), 150)
                        consonant = pre + added_cons
                        cutoff = -(consonant + vowel_len * 0.25)

                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                    a2 = apply_alias_suffix(alias, alias_suffix)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)
                    continue




                breath_tail = None
                a_parts = alias.split()
                if len(a_parts) >= 2 and a_parts[0].lower() in KR_VOWELS and a_parts[1].upper() in {'R', 'H'}:
                    breath_tail = a_parts[1].upper()
                elif alias and alias[-1].upper() in {'R', 'H'} and alias[:-1].lower() in KR_VOWELS:
                    breath_tail = alias[-1].upper()

                if breath_tail:
                    if current_w_idx >= len(syllables_info):
                        current_w_idx = len(syllables_info) - 1
                    curr_syl = syllables_info[current_w_idx]
                    curr_phones = curr_syl['phones']
                    _, v_phone = find_vowel_phone(curr_phones)
                    vowel_start = v_phone.minTime * 1000
                    vowel_end = v_phone.maxTime * 1000
                    vowel_len = max(vowel_end - vowel_start, 80)

                    last_end = (ph_intervals_all[-1].maxTime * 1000) if ph_intervals_all else (curr_syl['end_time'] * 1000)


                    offset_padding = min(max(vowel_len * 0.7, 180), 320)
                    offset = max(vowel_end - offset_padding, 0)


                    pre_abs = max(vowel_end - 20, offset)
                    pre = pre_abs - offset
                    ovl = pre * 0.85


                    consonant = max(vowel_end - offset, pre + 10)


                    cutoff = -(max(last_end - offset, consonant + 80))

                    offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)

                    a2 = apply_alias_suffix(alias, alias_suffix)
                    new_line = f"{real_wav_name}={a2},{offset:.2f},{consonant:.2f},{cutoff:.2f},{pre:.2f},{ovl:.2f}"
                    final_lines.append(new_line)
                    continue
                


                if is_vcv:
                    (
                        current_w_idx,
                        cv_seq_idx,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                    ) = _prepare_vcv_syllable_timing(
                        syllables_info,
                        current_w_idx,
                        cv_seq_idx,
                        DIPHTHONG_CV_CONSONANT_RATIO,
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        soft_off_shift,
                        soft_cut_shift,
                        cutoff_reduced,
                    ) = _apply_post_timing_pipeline(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        alias_type="vcv",
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        current_w_idx=current_w_idx,
                        syllables_info=syllables_info,
                        is_vc_plosive_coda=False,
                        enable_stabilize=False,
                        enable_cutoff_guard=False,
                    )
                    _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                if is_cv_head:
                    (
                        current_w_idx,
                        cv_seq_idx,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                    ) = _prepare_cv_head_syllable_timing(
                        syllables_info,
                        current_w_idx,
                        cv_seq_idx,
                        alias,
                    )
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        soft_off_shift,
                        soft_cut_shift,
                        cutoff_reduced,
                    ) = _apply_post_timing_pipeline(
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        alias_type="cv_head",
                        mel_ctx_for_file=mel_ctx_for_file,
                        base_shape=base_shape,
                        ph_intervals=ph_intervals,
                        current_w_idx=current_w_idx,
                        syllables_info=syllables_info,
                        is_vc_plosive_coda=False,
                        enable_stabilize=False,
                        enable_cutoff_guard=True,
                    )
                    _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)
                    
                    _append_alias_rows(
                        final_lines,
                        real_wav_name,
                        alias,
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        generate_openutau=generate_openutau,
                        alias_suffix=alias_suffix,
                    )
                    continue
                

                if not is_vc:
                    current_w_idx, cv_seq_idx = _resolve_cv_syllable_index(
                        target_clean, romaji_syllables, cv_seq_idx, current_w_idx
                    )
                    current_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_cv_bounds_from_syllable(
                        syllables_info, current_w_idx
                    )
                else:
                    current_w_idx, curr_phones, c_start, c_end, n_start, n_end = _prepare_vc_bounds_from_context(
                        syllables_info, current_w_idx
                    )
                cv_vowel_len = n_end - n_start
                is_vc_plosive_coda = False
                if is_vc:
                    (
                        offset,
                        consonant,
                        cutoff,
                        pre,
                        ovl,
                        is_vc_plosive_coda,
                    ) = _compute_kr_vc_timing(
                        alias,
                        alias_type,
                        file_format,
                        curr_phones,
                        current_w_idx,
                        syllables_info,
                        c_start,
                        c_end,
                        n_start,
                        n_end,
                        cv_anchor_by_idx,
                    )

                else:
                    cv_vowel_len = n_end - n_start
                    
                    if is_diph:
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        alias_onset = _extract_alias_onset(alias)
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            True,
                            False,
                        )
                        
                    elif _looks_like_vv_alias(alias):
                        boundary = c_end
                        v1_len = c_end - c_start
                        
                        offset_padding = 100
                        if v1_len < offset_padding:
                            offset_padding = max(v1_len * 0.8, 50)
                            
                        offset = boundary - offset_padding
                        pre = boundary - offset
                        ovl = adaptive_overlap(pre, "", mode='vv')
                        
                        added_cons = cv_vowel_len * 0.4
                        if added_cons < 60: added_cons = 60
                        consonant = pre + added_cons
                        cutoff = -(consonant + cv_vowel_len * 0.4)
                        
                    else:


                        first_phone_plosive = len(curr_phones) >= 2 and is_plosive_ipa(curr_phones[0].mark)
                        alias_consonant = re.match(r'^([^aeiouyw]+)', alias.lower())
                        roman_plosive = alias_consonant and is_plosive_roman(alias_consonant.group(1)) if alias_consonant else False
                        alias_onset = alias_consonant.group(1) if alias_consonant else ""
                        c_hint = curr_phones[0].mark if curr_phones else ""
                        is_plosive = first_phone_plosive or roman_plosive
                        offset, consonant, cutoff, pre, ovl = _compute_kr_cv_timing(
                            c_start,
                            c_end,
                            cv_vowel_len,
                            c_hint,
                            alias_onset,
                            False,
                            is_plosive,
                        )

                (
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    soft_off_shift,
                    soft_cut_shift,
                    cutoff_reduced,
                ) = _apply_post_timing_pipeline(
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    alias_type=alias_type,
                    mel_ctx_for_file=mel_ctx_for_file,
                    base_shape=base_shape,
                    ph_intervals=ph_intervals,
                    current_w_idx=current_w_idx,
                    syllables_info=syllables_info,
                    is_vc_plosive_coda=is_vc_plosive_coda,
                    enable_stabilize=True,
                    enable_cutoff_guard=True,
                )
                _log_post_timing_events(log, fname, alias, soft_off_shift, soft_cut_shift, cutoff_reduced)

                _append_alias_rows(
                    final_lines,
                    real_wav_name,
                    alias,
                    offset,
                    consonant,
                    cutoff,
                    pre,
                    ovl,
                    generate_openutau=generate_openutau,
                    alias_suffix=alias_suffix,
                )

            processed += 1

        except Exception as e:
            err_msg = f"처리 실패 ({fname}): {e}"
            logger.error(err_msg)
            errors.append(err_msg)
            _record_unset_lines("file_exception", fname, lines)
            final_lines.extend([apply_suffix_to_oto_line(l, alias_suffix) for l in lines])
            processed += 1

        if callback and total > 0 and (processed % 5 == 0 or processed == total):
            callback(f"OTO 생성 중... ({processed}/{total})")


    if gen_missing_vowels:
        log("누락된 단모음 에일리어스 자동 생성을 시작합니다...")
        vowels_list = ['a', 'e', 'i', 'o', 'u', 'eo', 'eu', 'ae', 'oe', 'wi', 'wa', 'we', 'weo', 'ya', 'ye', 'yo', 'yeo', 'yu', 'ui', 'eui']
        template_aliases = set()
        for g_lines in file_groups.values():
            for line in g_lines:
                parts = line.split('=')
                if len(parts) > 1:
                    alias = parts[1].split(',')[0].strip().lower()
                    template_aliases.add(alias)

        for tg_info in tg_entries:

            base_filename = os.path.splitext(tg_info['real_name'])[0].lower()
            

            base_filename = re.sub(r'long$', '', base_filename)
            norm_key_clean = re.sub(r'long$', '', tg_info['norm_key'])
            
            detected_vowel = None
            

            tokens = re.split(r'[-_ ]+', base_filename)
            for token in reversed(tokens):
                clean_token = re.sub(r'[^a-z]', '', token)
                if clean_token in vowels_list:
                    detected_vowel = clean_token
                    break
            

            if not detected_vowel and norm_key_clean in vowels_list:
                detected_vowel = norm_key_clean
                
            if detected_vowel:
                try:
                    tg = textgrid.TextGrid.fromFile(tg_info['path'])
                    w_tier = None
                    for t in tg:
                        if hasattr(t, 'name') and t.name == 'words':
                            w_tier = t
                            break

                    phone_tier = next((t for t in tg if isinstance(t, textgrid.IntervalTier) and t.name == 'phones'), None)
                    if not phone_tier: continue
                    intervals = [i for i in phone_tier if i.mark.strip() not in ['', 'sil', 'spn', 'pau']]
                    
                    if len(intervals) == 1:
                        vowel = intervals[0]
                        v_start = vowel.minTime * 1000
                        v_end = vowel.maxTime * 1000
                        v_len = v_end - v_start
                        
                        alias = detected_vowel
                        if alias not in template_aliases:
                            log(f"추가: 단모음 에일리어스 생성 -> {tg_info['real_name']} [{alias}]")
                            offset = v_start
                            pre = 0
                            ovl = 0
                            consonant = min(v_len * 0.25, 120)
                            cutoff = -(v_len * 0.8)
                            
                            offset, consonant, cutoff, pre, ovl = validate_oto_params(offset, consonant, cutoff, pre, ovl)
                            
                            _append_alias_rows(
                                final_lines,
                                tg_info['real_name'],
                                alias,
                                offset,
                                consonant,
                                cutoff,
                                pre,
                                ovl,
                                generate_openutau=generate_openutau,
                                alias_suffix=alias_suffix,
                            )
                except:
                    continue


    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for line in final_lines:
                f.write(line + "\n")
        log(f"완료: OTO 파일 저장 -> {out_path}")
        wav_dir_for_profile = os.path.dirname(os.path.abspath(tg_folder.rstrip("\\/")))
        if kr_profile:
            changed = _apply_kr_profile_to_oto_file(
                out_path, wav_dir_for_profile, kr_profile, custom_map=custom_map
            )
            if changed > 0:
                log(f"[KR-Profile] 기준 프로파일 보정 적용: {changed} lines")
        mel_changed = _apply_kr_mel_refine_to_oto_file(
            out_path, wav_dir_for_profile, custom_map=custom_map
        )
        if mel_changed > 0:
            log(f"[KR-Mel] 멜 에너지 기반 cutoff 보정 적용: {mel_changed} lines")
    except Exception as e:
        err = f"OTO 파일 저장 실패: {e}"
        logger.error(err)
        errors.append(err)

    _log_unset_summary()
    return processed, total, errors




