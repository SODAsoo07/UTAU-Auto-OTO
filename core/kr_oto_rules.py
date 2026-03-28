"""
﨑懋ｵｭ・ｴ OTO ・晧┳・川・ ・ｵ奝ｵ・ｼ・・・ｰ・・・菩・・懍ｹ・・・･・奝增ｰ嶹・・・ｸｵ・・笈・､.

・ｴ ・ｨ・溢捩 﨑懋ｵｭ・ｴ ・晧┳・ｰ・川・ ・川｣ｼ ・罷誤株 timing ・・げ・晝ｳｼ ・・ｦｬ﨑ｴ,
嶸菩享 ・川ｧ/・川攵・ｬ・ｴ・､ ・・･・・護・奝增ｰ ・母ｷ懦剩 ・呷捩 ・懍ｹ呷ｸｵ・・・・ｦｽ・・愍・・
・､・ｰ ・・・一ｲ・﨑俾ｸｰ ・・復 ・ｩ・・桿・壱共.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from core.alias_annotation_utils import strip_alias_annotation_suffixes

IPA_VOWELS = {
    "a", "e", "i", "o", "u", "y",
    "eo", "eu", "ae", "oe", "wa", "we", "wi", "wo", "ui",
    "ʌ", "ɯ", "ɛ", "ø", "wʌ",
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

HIRAGANA_RE = re.compile(r"[\u3041-\u3096\u309d-\u309f]")


def clean_phone_mark(mark):
    return re.sub(r"[0-9]", "", mark).lower()


def should_ignore_korean_alias(alias):
    """
    﨑懋ｵｭ・ｴ OTO・川・ 德壱攵・・・alias・・・俯ｪｻ ・樌攤 ・ｼ・ｸ・ｴ ・ｰ・ｴ奓ｰ・・・・｣ｼ﨑俾ｳ ・ｴ・懦復・､.
    """
    return bool(HIRAGANA_RE.search(str(alias or "")))


def _normalize_custom_alias_lookup(alias):
    text = unicodedata.normalize("NFKC", str(alias or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    variants = [text, text.lower()]
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:#|b)?[0-8])$", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:sharp|flat)?[0-8])$", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*[\(\[\{・医疹.*?[\)\]\}・峨曽\s*$", "", stripped).strip()
    if stripped:
        variants.extend([stripped, stripped.lower()])
    out = []
    seen = set()
    for item in variants:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _lookup_custom_alias_value(custom_map, alias):
    if not custom_map:
        return None
    for candidate in _normalize_custom_alias_lookup(alias):
        if candidate in custom_map:
            return custom_map[candidate]
    return None


def normalize_ipa_mark(mark):
    """IPA ・逸〓・ｼ ・母ｷ懦剩﨑ｴ ・・ｵ・・・･﨑・嶸倣・・・・誤働・壱共."""
    base = clean_phone_mark(mark)
    nfd = unicodedata.normalize("NFD", base)
    stripped = "".join(ch for ch in nfd if not unicodedata.combining(ch))
    stripped = stripped.replace("ʰ", "").replace("ː", "").replace("ˑ", "").replace("-", "")
    return stripped


def is_glide(phone_mark):
    clean = normalize_ipa_mark(phone_mark)
    return clean in ["j", "w", "y", "ɥ", "ɰ"]


def find_vowel_phone(phones):
    """・護・ ・懦・､・川・ ・岺・・ｨ・護揆 ・ｾ・・(index, phone)・ｼ・・・倆劍﨑ｩ・壱共."""
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
    """IPA 岺懋ｸｰ・ 甯護龍・・・・龍・ｸ・ 甯尖ｳ・鮒・壱共."""
    clean = normalize_ipa_mark(phone_mark)
    if clean in IPA_PLOSIVES:
        return True
    return clean in {"k", "t", "p", "c", "g", "d", "b"}


def is_plosive_roman(consonant_str):
    """・罹ｧ溢梵 ・川搆 ・ｸ・川龍・ｴ 甯護龍・・・・龍・ｸ・ 甯尖ｳ・鮒・壱共."""
    return consonant_str.lower() in [
        "g", "gg", "d", "dd", "b", "bb", "j", "jj",
        "k", "kk", "t", "tt", "p", "pp", "ch",
    ]


@lru_cache(maxsize=32768)
def _extract_alias_onset(alias):
    """CV ・川攵・ｬ・ｴ・､・川・ ・溢┳(・罹ｧ溢梵 ・川搆・ｰ)・・・肥ｶ懦鮒・壱共."""
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
    """pcl/tcl/kcl ・呷捩 尞川℡ ・ｬ・・岺懋ｸｰ・ｼ 﨑懋ｵｭ・ｴ ・・┳嶸・VC・・・ｨ・駕鮒・壱共."""
    t = (token or "").strip().lower().rstrip("-")
    if not t:
        return False
    return bool(re.match(r"^(?:[a-z]{1,3}cl|q)$", t))


def _detect_glottal_kind(alias):
    """・川攵・ｬ・ｴ・､・・・ｱ・ｸ 甯護龍 岺懍享 ・・ｹ・head/tail)・ｼ 甯尖ｳ・鮒・壱共."""
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
    """Breath alias detector (br, inhale/exhale variants)."""
    clean = unicodedata.normalize("NFKC", str(alias or "")).strip().lower()
    if not clean:
        return False
    parts = [p for p in clean.split() if p]
    if len(parts) >= 2 and parts[0] in KR_VOWELS and parts[-1] in {"r", "h", "・", "·", "fry", "vf"}:
        # VC tail markers should remain VC, not breath.
        return False
    if re.match(r"^br\d*$", clean):
        return True
    if clean in {"bre", "breath", "breathy", "inhale", "exhale", "aspirate", "諱ｯ", "蜷ｸ", "蜷・", "彧｡", "嶸ｸ", "・ｨ"}:
        return True
    if any(ch in clean for ch in ("蜷ｸ", "蜷・", "諱ｯ")) and not (len(parts) >= 2 and parts[0] in KR_VOWELS):
        return True
    return False


def _extract_vowel_consonant(text):
    onset, vowel, _ = _split_kr_syllable_parts(text)
    if vowel:
        return vowel, onset
    text = re.sub(r"[^a-z]", "", (text or "").lower())
    return None, text


@lru_cache(maxsize=65536)
def _normalize_cv_match_token(token):
    """CV ・護・・､﨑・・・ｵ川圸 ・母ｷ懦剩(r/l 嶸ｼ・ｩ ・ｱ)・ｼ ・・圸﨑ｩ・壱共."""
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
    﨑懋ｵｭ・ｴ ・罹ｧ溢梵 ・護溢揆 (・溢┳, ・卓┳, ・・┳)・ｼ・・・・紛﨑ｩ・壱共.
    ・・ nyeong -> (n, yeo, ng), ryeo -> (r, yeo, "")
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
    """・・ｵ川圸 CV 﨑ｵ・ｬ ・ｸ・川龍(onset+vowel)・・・倆劍﨑ｩ・壱共."""
    onset, vowel, _ = _split_kr_syllable_parts(text)
    if vowel:
        return f"{onset}{vowel}"
    return _normalize_cv_match_token(text)


def _extract_vc_right_token(alias):
    """VC ・川攵・ｬ・ｴ・､・川・ ・ｰ・｡ ・川搆(・幗ｹｨ 弡・ｳｴ)・・・肥ｶ懦鮒・壱共."""
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
    """・幗ｹｨ ・川搆 岺懋ｸｰ・ｼ K/T/P ・・龍 ・ｰ・・ｼ・・・母ｷ懦剩﨑ｩ・壱共."""
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
    """VC ・川攵・ｬ・ｴ・､・ 甯護龍・・・幗ｹｨ(K/T/P ・・龍)・ｸ・ 甯尖ｳ・鮒・壱共."""
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
    """・川攵・ｬ・ｴ・､ CV・ ・護・・罹ｧ溢梵 ・・・､﨑・・川・・ｼ ・・げ﨑ｩ・壱共."""
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
    """CV ・､﨑卓圸 ・川攵・ｬ・ｴ・､ 﨑ｵ・ｬ 奝增ｰ・・・肥ｶ懦鮒・壱共."""
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
    """・川攵・ｬ・ｴ・､ ・ｸ・川龍・・br/cv/cv_head/vc/vv/vcv/mono 夋・・愍・・・・･倆鮒・壱共."""
    clean = unicodedata.normalize("NFKC", str(alias or "")).strip()

    if is_breath(clean):
        return "br"

    mapped_val = _lookup_custom_alias_value(custom_map, clean)
    if mapped_val is not None:
        mapped_val = str(mapped_val).lower()
        if mapped_val in ["r", "h", "sil", "br"]:
            return "br"
        if mapped_val in KR_VOWELS:
            return "mono"
        return "cv"

    if " " in clean:
        parts = clean.split()
        left = parts[0].strip()
        right = " ".join(parts[1:]).strip()

        if left in {"-", "・", "·"}:
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
            if right_lower in {"・", "·", "fry", "vf", "h", "r"}:
                return "vc"
            if any(ord(ch) > 127 for ch in right):
                return "vc"
            right_ascii = unicodedata.normalize("NFKD", right_lower)
            right_alpha = "".join(ch for ch in right_ascii if "a" <= ch <= "z")
            if right_alpha and right_alpha not in KR_VOWELS and len(right_alpha) <= 2:
                return "vc"
            if (not right_alpha) and any(not ch.isalnum() for ch in right_lower):
                return "vc"
            if (
                right_lower in KR_CONSONANTS
                or right_alpha in KR_CONSONANTS
                or right.upper() in KR_BATCHIM_MARKERS
                or _is_kr_closure_token(right)
            ):
                return "vc"
            if right_lower in KR_VOWELS or right_alpha in KR_VOWELS:
                return "vv"
            return "vcv"

        return "vc"

    clean_lower = clean.lower().rstrip("-")

    gk = _detect_glottal_kind(clean)
    if gk == "head":
        return "cv_head"
    if gk == "tail":
        return "vc"

    mapped_val = _lookup_custom_alias_value(custom_map, clean_lower)
    if mapped_val is not None:
        mapped_val = str(mapped_val).lower()
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


_CLASSIFY_ALIAS_RAW = classify_alias


def classify_alias(alias, custom_map=None):
    clean = unicodedata.normalize("NFKC", str(alias or "")).strip()
    clean = strip_alias_annotation_suffixes(
        clean,
        classifier=lambda candidate: _CLASSIFY_ALIAS_RAW(candidate, custom_map),
    )
    return _CLASSIFY_ALIAS_RAW(clean, custom_map)


def detect_alias_format(alias_list, custom_map=None):
    """甯護攵 ・ｨ・・・川攵・ｬ・ｴ・､ ・ｩ・晧攪 ・・ｲｴ 尞ｬ・ｷ(CVC/CVVC/VCV ・ｱ)・・・肥倣鮒・壱共."""
    if not alias_list:
        return "cv"

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
        return "cv"
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
        return "cv"
    if "vv" in type_set or ("cv_head" in non_br and "vc" in non_br):
        return "cvvc"
    return "cvc"


def _looks_like_vv_alias(alias):
    """・ｵ・ｱ ・・株 ・ｨ・・VV ・川攵・ｬ・ｴ・､ 嶸倣・・ｸ・ 甯尖ｳ・鮒・壱共."""
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

