"""
한국어 OTO 생성에서 공통으로 쓰는 정적 규칙/분류/토큰화 계층입니다.

이 모듈은 한국어 생성기에서 자주 바뀌는 timing 계산식과 분리해,
형식 감지/에일리어스 분류/음절 토큰 정규화 같은 규칙층을 독립적으로
다룰 수 있게 하기 위한 목적입니다.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

IPA_VOWELS = {
    "a", "e", "i", "o", "u", "y",
    "eo", "eu", "ae", "oe", "wa", "we", "wi", "wo", "ui",
    "ɯ", "ʌ", "ɛ", "ə", "æ", "ɑ", "ɐ", "ɔ", "ɪ", "ʊ", "ø", "œ",
}

IPA_PLOSIVES = {
    "k", "g", "t", "d", "p", "b", "c", "q", "ch", "j",
}

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

KR_VOWELS = {
    "a", "e", "i", "o", "u", "eo", "eu", "ae", "oe", "wi",
    "ya", "yeo", "yo", "yu", "ye", "wa", "wo", "we", "weo", "eui", "ui", "wae",
}

KR_CONSONANTS = {
    "g", "n", "d", "r", "l", "m", "b", "s", "j", "ch", "k", "t", "p", "h",
    "gg", "dd", "bb", "ss", "jj", "kk", "tt", "pp",
    "ng", "sh", "f", "z", "q", "c",
}

KR_BATCHIM_MARKERS = {"N", "L", "M", "NG", "K", "T", "P", "H"}
GLOTTAL_MARKS = {"'", "’", ".", "ʔ"}

KR_CODA_PLOSIVE_MAP = {
    "g": "k", "gg": "k", "k": "k", "kk": "k",
    "d": "t", "dd": "t", "t": "t", "tt": "t",
    "s": "t", "ss": "t", "j": "t", "jj": "t", "ch": "t", "h": "t", "c": "t", "q": "t",
    "b": "p", "bb": "p", "p": "p", "pp": "p",
}


def clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", mark).lower()


def normalize_ipa_mark(mark):
    """IPA 마크를 정규화해 비교 가능한 형태로 만듭니다."""
    base = clean_phone_mark(mark)
    nfd = unicodedata.normalize("NFD", base)
    stripped = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    stripped = stripped.replace("ː", "").replace("ˈ", "").replace("ˌ", "").replace("ʰ", "")
    return stripped


def is_glide(phone_mark):
    clean = normalize_ipa_mark(phone_mark)
    return clean in ["j", "w", "y", "ɥ", "ɰ", "ʋ"]


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
    return clean in {"k", "t", "p", "c", "g", "d", "b"}


def is_plosive_roman(consonant_str):
    """로마자 자음 문자열이 파열음 계열인지 판별합니다."""
    return consonant_str.lower() in [
        "g", "gg", "d", "dd", "b", "bb", "j", "jj",
        "k", "kk", "t", "tt", "p", "pp", "ch",
    ]


@lru_cache(maxsize=32768)
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


def _is_kr_closure_token(token):
    """pcl/tcl/kcl 같은 폐쇄 구간 표기를 한국어 종성형 VC로 취급합니다."""
    t = (token or "").strip().lower().rstrip("-")
    if not t:
        return False
    return bool(re.match(r"^(?:[a-z]{1,3}cl|q)$", t))


def _detect_glottal_kind(alias):
    """에일리어스의 성문 파열 표식 위치(head/tail)를 판별합니다."""
    a = (alias or "").strip()
    if not a:
        return None
    parts = a.split()
    if len(parts) >= 2:
        if parts[0] in GLOTTAL_MARKS:
            return "head"
        if parts[-1] in GLOTTAL_MARKS:
            return "tail"
    if a[0] in GLOTTAL_MARKS:
        return "head"
    if a[-1] in GLOTTAL_MARKS:
        return "tail"
    return None


def is_breath(alias):
    """숨소리(br, br1...) 에일리어스인지 판별합니다."""
    clean = alias.strip().lower()
    return bool(re.match(r"^br\d*$", clean))


def _extract_vowel_consonant(text):
    onset, vowel, _ = _split_kr_syllable_parts(text)
    if vowel:
        return vowel, onset
    text = re.sub(r"[^a-z]", "", (text or "").lower())
    return None, text


@lru_cache(maxsize=65536)
def _normalize_cv_match_token(token):
    """CV 음절 매핑 비교용 정규화(r/l 혼용 등)를 적용합니다."""
    t = re.sub(r"[^a-z]", "", (token or "").lower())
    if not t:
        return t
    if t.startswith("l"):
        t = "r" + t[1:]
    t = t.replace("ly", "ry")
    return t


@lru_cache(maxsize=65536)
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


@lru_cache(maxsize=65536)
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
        r"^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch|c|q|h)$",
        a.lower(),
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


def _is_kr_glide_vowel(vowel):
    v = (vowel or "").strip().lower()
    return v in {"ya", "ye", "yeo", "yo", "yu", "wa", "wae", "we", "weo", "wi", "wo"}


@lru_cache(maxsize=131072)
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
        else:
            alias_glide = _is_kr_glide_vowel(av)
            syl_glide = _is_kr_glide_vowel(sv)
            if alias_glide != syl_glide:
                score -= 28
            else:
                score -= 18

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

    if av and sv and (ao + av) == (so + sv):
        score += 16
    if scoda and av and (so + sv) and a == (so + sv):
        score += 6

    if a in s or s in a:
        score += 10
    return max(0, min(score, 100))


@lru_cache(maxsize=65536)
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
        return "br"

    if custom_map and clean in custom_map:
        mapped_val = custom_map[clean].lower()
        if mapped_val in ["r", "h", "sil", "br"]:
            return "br"
        if mapped_val in KR_VOWELS:
            return "mono"
        return "cv"

    if " " in clean:
        parts = clean.split()
        left = parts[0].strip()
        right = " ".join(parts[1:]).strip()

        if left == "-":
            return "cv_head"

        if right == "-":
            return "cv"

        left_lower = left.lower()
        right_lower = right.lower().rstrip("-")
        if left in GLOTTAL_MARKS and right_lower in KR_VOWELS:
            return "cv_head"
        if left_lower in KR_VOWELS and right in GLOTTAL_MARKS:
            return "vc"

        left_upper = left.upper()
        if left_upper in KR_BATCHIM_MARKERS:
            right_kernel = _kr_cv_kernel(right_lower)
            right_onset, right_vowel, _right_coda = _split_kr_syllable_parts(right_lower)
            if right_lower in KR_CONSONANTS or right.upper() in KR_BATCHIM_MARKERS or _is_kr_closure_token(right):
                return "vc"
            if right_kernel in KR_VOWELS:
                return "vv"
            if right_vowel and (right_onset or right_kernel):
                return "vcv"
            return "vc"

        if left_lower in KR_VOWELS:
            if right_lower in KR_CONSONANTS or right.upper() in KR_BATCHIM_MARKERS or _is_kr_closure_token(right):
                return "vc"
            if right_lower in KR_VOWELS:
                return "vv"
            return "vcv"

        return "vc"

    clean_lower = clean.lower().rstrip("-")

    gk = _detect_glottal_kind(clean)
    if gk == "head":
        return "cv_head"
    if gk == "tail":
        return "vc"

    if custom_map and clean_lower in custom_map:
        mapped_val = custom_map[clean_lower].lower()
        if mapped_val in ["sil", "br", "r", "h"]:
            return "br"
        if mapped_val in KR_VOWELS:
            return "mono"
        return "cv"

    if clean_lower in KR_VOWELS:
        return "mono"

    m = re.match(
        r"^([aoueiwy]+|eo|eu|ae|oe|wa|wo|we|ye|ya|yo|yu|wae|weo|eui|ui)([gknmdrlbsjtph]+|ng|kk|ss|pp|tt|jj|ch)$",
        clean_lower,
    )
    if m:
        return "vc"

    return "cv"


def detect_alias_format(alias_list, custom_map=None):
    """파일 단위 에일리어스 목록의 전체 포맷(CVC/CVVC/VCV 등)을 추정합니다."""
    if not alias_list:
        return "cvc"

    type_cache = {}
    types = []
    for alias in alias_list:
        a_type = type_cache.get(alias)
        if a_type is None:
            a_type = classify_alias(alias, custom_map)
            type_cache[alias] = a_type
        types.append(a_type)
    type_set = set(types)

    if type_set == {"br"}:
        return "br"
    if type_set <= {"mono", "cv_head", "cv"}:
        return "mono"
    if "vcv" in type_set:
        return "vcv"
    if {"cv_head", "cv", "vc"} <= type_set:
        return "cvvc"

    non_br = type_set - {"br"}
    if non_br <= {"vv"}:
        return "vv_only"
    if non_br <= {"vc", "vv"}:
        return "vc_only"
    if non_br <= {"cv", "mono", "cv_head"}:
        return "cv_simple"
    if "vv" in type_set or ("cv_head" in non_br and "vc" in non_br):
        return "cvvc"
    return "cvc"


def _looks_like_vv_alias(alias):
    """공백 없는 단순 VV 에일리어스 형태인지 판별합니다."""
    if not alias:
        return False
    a = alias.strip().lower()
    if " " in a or len(a) < 2:
        return False
    if a in {"eo", "eu", "ae", "oe", "wi"}:
        return False
    return a[0] in {"a", "e", "i", "o", "u", "y", "w"} and a[-1] in {"a", "e", "i", "o", "u", "y", "w"}


__all__ = [
    "GLOTTAL_MARKS",
    "IPA_PLOSIVES",
    "IPA_VOWELS",
    "KR_BATCHIM_MARKERS",
    "KR_CONSONANTS",
    "KR_PLOSIVE_ONSETS",
    "KR_SIBILANT_ONSETS",
    "KR_SONORANT_CONSONANTS",
    "KR_TENSE_CONSONANTS",
    "KR_VOICED_ONSETS",
    "KR_VOICELESS_ONSETS",
    "KR_VOWELS",
    "_canonicalize_kr_coda",
    "_cv_match_score",
    "_detect_glottal_kind",
    "_extract_alias_onset",
    "_extract_kr_cv_alias_token",
    "_extract_vc_right_token",
    "_extract_vowel_consonant",
    "_is_kr_closure_token",
    "_is_kr_glide_vowel",
    "_is_kr_plosive_coda_alias",
    "_is_sonorant_consonant",
    "_is_tense_consonant",
    "_kr_cv_kernel",
    "_looks_like_vv_alias",
    "_normalize_cv_match_token",
    "_split_kr_syllable_parts",
    "classify_alias",
    "clean_phone_mark",
    "detect_alias_format",
    "find_vowel_phone",
    "is_breath",
    "is_glide",
    "is_plosive_ipa",
    "is_plosive_roman",
    "normalize_ipa_mark",
]
