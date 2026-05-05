from __future__ import annotations

import re

ALIAS_ROLES: tuple[str, ...] = (
    "-cv",
    "cv",
    "cv-",
    "v",
    "v-",
    "vc",
    "vv",
    "v-cv",
    "endbr",
    "br",
    "other",
    "special",
)

DEFAULT_ROLE = "other"
SPECIAL_ROLE = "special"

# Korean Hangul syllable block decomposition constants.
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3
_JUNG_COUNT = 21
_JONG_COUNT = 28
_JUNGSEONG_TABLE: tuple[str, ...] = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
    "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)

# Korean diphthong jungseong: y-glide, w-glide, and special compounds whose
# nucleus has a glide+vowel (or two-vowel) onset that is longer than a pure CV.
KOREAN_DIPHTHONG_JUNGSEONG: frozenset[str] = frozenset({
    "ㅑ", "ㅒ", "ㅕ", "ㅖ", "ㅛ", "ㅠ",
    "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ",
})

# Japanese small kana that mark yō-on (glide+vowel) — appearing in the alias
# implies a diphthong-like onset (kya, kyu, kyo, sha, etc.).
JAPANESE_YOON_CHARS: frozenset[str] = frozenset({
    "ゃ", "ゅ", "ょ", "ャ", "ュ", "ョ",
})

# Vowel characters used to detect sequential-vowel diphthong patterns
# inside a single Japanese alias token.
_JA_VOWELS: frozenset[str] = frozenset(
    "aiueo"
    "あいうえお"
    "アイウエオ"
    "ぁぃぅぇぉ"
    "ァィゥェォ"
)

_RELEASE_SUFFIX = re.compile(r"\s+[\-‐‑–—―ー]\s*$")
_HEAD_PREFIX = re.compile(r"^[\-‐‑–—―]\s+")
# End-breath markers actually used in voicebanks (a R, eo R, ぁ R, a Br, a 息, a ・).
_ENDBREATH_SUFFIX = re.compile(
    r"\s+(?:R|r|Br|br|BR|息|・|ハ|ヒ|フ|ヘ|ホ)\s*$"
)


def classify_alias_role(
    language: object,
    alias: object,
    *,
    alias_type: object = "",
    transition_type: object = "",
    is_special: bool = False,
) -> str:
    """Map an alias to its rendering role.

    The role is the primary axis the model and prior table should branch on.
    `format_type` (cvvc/vcv/cv/cvc) is intentionally ignored here so that the
    same alias gets the same role no matter which voicebank format wraps it.

    The function never raises — unknown aliases fall back to `other`.
    """
    if bool(is_special):
        return SPECIAL_ROLE

    text = str(alias or "").strip()
    if not text:
        return DEFAULT_ROLE

    a_type = str(alias_type or "").strip().lower()
    t_type = str(transition_type or "").strip().lower()

    if a_type == "br" or t_type == "br":
        return "endbr" if _ENDBREATH_SUFFIX.search(text) else "br"

    head = bool(_HEAD_PREFIX.match(text))
    release = bool(_RELEASE_SUFFIX.search(text))

    if a_type == "cv_head" or head:
        return "-cv"
    if a_type == "vcv":
        return "v-cv"
    if a_type == "vv":
        return "vv"
    if a_type == "vc":
        return "vc"
    if a_type == "cv":
        return "cv-" if release else "cv"
    if a_type in {"vowel", "mono"}:
        return "v-" if release else "v"

    if t_type == "cv":
        return "cv-" if release else "cv"
    if t_type == "vc":
        return "vc"
    if t_type == "vv":
        return "vv"
    if t_type == "vowel":
        return "v-" if release else "v"

    return DEFAULT_ROLE


def is_diphthong(language: object, alias: object) -> bool:
    """Return True when the alias's nucleus behaves CV-like even though it
    is classified as a vowel role.

    The detection is intentionally conservative — only covers patterns that
    are common enough in real voicebanks to matter. False negatives default
    to monophthong handling, which is already what the model expects.
    """
    text = str(alias or "").strip()
    if not text:
        return False
    lang = _normalize_language(language)
    if lang == "korean":
        return _korean_alias_has_diphthong(text)
    if lang == "japanese":
        return _japanese_alias_has_diphthong(text)
    return False


def normalize_role(role: object) -> str:
    text = str(role or "").strip().lower()
    if text in ALIAS_ROLES:
        return text
    aliases = {
        "endbreath": "endbr",
        "end_br": "endbr",
        "end-br": "endbr",
        "vcv": "v-cv",
        "head": "-cv",
        "headcv": "-cv",
        "head_cv": "-cv",
        "cv_head": "-cv",
        "release": "cv-",
        "tail": "cv-",
        "vowel": "v",
        "mono": "v",
        "special_phoneme": "special",
        "special-phoneme": "special",
        "special_phone": "special",
        "special-phone": "special",
        "spcl": "special",
    }
    return aliases.get(text, DEFAULT_ROLE)


def _normalize_language(language: object) -> str:
    text = str(language or "").strip().lower()
    if text in {"ko", "kor", "korean"}:
        return "korean"
    if text in {"ja", "jp", "jpn", "japanese"}:
        return "japanese"
    return text


def _korean_alias_has_diphthong(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if _HANGUL_BASE <= code <= _HANGUL_END:
            jung = _hangul_jungseong(code)
            if jung in KOREAN_DIPHTHONG_JUNGSEONG:
                return True
        elif ch in KOREAN_DIPHTHONG_JUNGSEONG:
            return True
    return False


def _hangul_jungseong(code: int) -> str:
    offset = int(code) - _HANGUL_BASE
    if offset < 0:
        return ""
    jung_index = (offset // _JONG_COUNT) % _JUNG_COUNT
    if 0 <= jung_index < len(_JUNGSEONG_TABLE):
        return _JUNGSEONG_TABLE[jung_index]
    return ""


def _japanese_alias_has_diphthong(text: str) -> bool:
    for ch in text:
        if ch in JAPANESE_YOON_CHARS:
            return True
    cleaned = re.sub(r"\s+", "", text)
    vowel_run = 0
    for ch in cleaned:
        if ch in _JA_VOWELS:
            vowel_run += 1
            if vowel_run >= 2:
                return True
        else:
            vowel_run = 0
    return False


__all__ = [
    "ALIAS_ROLES",
    "DEFAULT_ROLE",
    "JAPANESE_YOON_CHARS",
    "KOREAN_DIPHTHONG_JUNGSEONG",
    "SPECIAL_ROLE",
    "classify_alias_role",
    "is_diphthong",
    "normalize_role",
]
