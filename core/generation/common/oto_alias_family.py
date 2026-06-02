from __future__ import annotations

import re

ALIAS_ATTACHED_PITCH_SUFFIX_RE = re.compile(r"[A-Ga-g](?:#|b)?[0-8](?:[A-Za-z]{1,4})?$")
ROMAJI_VOWELS = {"a", "i", "u", "e", "o", "n", "N", "ん"}
KANA_VOWELS = {
    "あ",
    "い",
    "う",
    "え",
    "お",
    "ぁ",
    "ぃ",
    "ぅ",
    "ぇ",
    "ぉ",
    "ア",
    "イ",
    "ウ",
    "エ",
    "オ",
    "ァ",
    "ィ",
    "ゥ",
    "ェ",
    "ォ",
}
BREATH_ALIASES = {"息", "吸", "br", "breath", "inhale", "exhale"}

_ROMAJI_VOWELS_LOWER = {item.lower() for item in ROMAJI_VOWELS}


def alias_token_is_vowel(token: str) -> bool:
    value = _strip_alias_attached_pitch_suffix_token(token)
    if not value:
        return False
    return value.lower() in _ROMAJI_VOWELS_LOWER or value in KANA_VOWELS


def alias_token_is_consonant_like(token: str) -> bool:
    value = _strip_alias_attached_pitch_suffix_token(token)
    if not value or value == "-":
        return False
    return not alias_token_is_vowel(value)


def _strip_alias_attached_pitch_suffix_token(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    if _is_breath_alias_with_attached_pitch_style_suffix(value):
        return value
    kana_positions = [
        idx for idx, char in enumerate(value)
        if "\u3040" <= char <= "\u30ff"
    ]
    if kana_positions:
        suffix_start = kana_positions[-1] + 1
        suffix = value[suffix_start:]
        if suffix and re.fullmatch(r"[A-Za-z]{0,4}[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}", suffix):
            return value[:suffix_start].strip()
    roman_stripped = _strip_roman_attached_pitch_suffix_token(value)
    if roman_stripped:
        return roman_stripped
    match = ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value)
    if match is None:
        separated_suffix = re.search(r"[_-]([A-Za-z][A-Za-z0-9]{0,4})$", value)
        if separated_suffix is not None and separated_suffix.start() > 0:
            suffix = separated_suffix.group(1)
            if any(char.isdigit() for char in suffix) or suffix != suffix.lower():
                stripped = value[: separated_suffix.start()].rstrip("_- ").strip()
                if stripped:
                    return stripped
        return value
    if match.start() <= 0:
        return ""
    stripped = value[: match.start()].rstrip("_- ").strip()
    return stripped or value


def _strip_roman_attached_pitch_suffix_token(token: str) -> str:
    raw = str(token or "").strip()
    if not raw or not all(char.isascii() for char in raw):
        return ""
    suffix_re = re.compile(r"[A-Za-z]{0,4}[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}")
    for split in range(len(raw) - 1, 0, -1):
        stem = raw[:split].rstrip("_- ").strip()
        suffix = raw[split:]
        if not stem or not suffix:
            continue
        if not suffix_re.fullmatch(suffix):
            continue
        if _roman_alias_stem_is_phone_like(stem):
            return stem
    return ""


def _roman_alias_stem_is_phone_like(stem: str) -> bool:
    text = str(stem or "").strip().lower()
    if not text or not re.fullmatch(r"[a-z]{1,6}", text):
        return False
    vowels = {
        "eui",
        "yeo",
        "yae",
        "wae",
        "weo",
        "eo",
        "eu",
        "ae",
        "ya",
        "ye",
        "yo",
        "yu",
        "wa",
        "wo",
        "we",
        "wi",
        "ui",
        "a",
        "i",
        "u",
        "e",
        "o",
        "n",
    }
    consonants = {
        "b",
        "ch",
        "d",
        "dh",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "ng",
        "ny",
        "p",
        "r",
        "s",
        "sh",
        "ss",
        "t",
        "ts",
        "v",
        "w",
        "y",
        "z",
    }
    yoon = {"ky", "gy", "ny", "hy", "by", "py", "my", "ry", "jy", "dy", "ty"}
    if not any(ch in "aeiou" for ch in text) and text not in (vowels | consonants | yoon):
        return False
    units = tuple(sorted(vowels | consonants | yoon, key=len, reverse=True))
    pos = 0
    while pos < len(text):
        unit = next((item for item in units if text.startswith(item, pos)), None)
        if unit is None:
            return False
        pos += len(unit)
    return True


def _is_breath_alias_with_attached_pitch_style_suffix(alias: str) -> bool:
    raw = str(alias or "").strip()
    if not raw or any(char.isspace() for char in raw) or not raw.isascii():
        return False
    lowered = raw.lower()
    for stem in ("breath", "bre", "br"):
        if not lowered.startswith(stem):
            continue
        suffix = raw[len(stem) :]
        if suffix and re.fullmatch(r"[A-Ga-g](?:#|b)?[0-8][A-Za-z]{0,4}", suffix):
            return True
    return False


def alias_family(alias: str) -> str:
    value = str(alias or "").strip()
    if not value:
        return "blank"
    lowered = value.lower()
    compact = "".join(part for part in lowered.split())
    if (
        lowered in BREATH_ALIASES
        or _is_breath_alias_with_attached_pitch_style_suffix(value)
        or (compact.startswith("br") and compact[2:].isdigit())
        or any(marker in value for marker in ("息", "吸"))
    ):
        return "policy_breath"
    parts = value.split()
    if len(parts) >= 2 and parts[-1] == "-":
        return "terminal_v_dash"
    if value == "-":
        return "terminal_v_dash"
    if parts and parts[0] == "-":
        return "cv_head"
    if len(parts) >= 2:
        left = parts[0]
        right = parts[-1]
        if alias_token_is_vowel(left) and alias_token_is_vowel(right):
            return "vv"
        if alias_token_is_vowel(left) and alias_token_is_consonant_like(right):
            return "vc"
        if alias_token_is_vowel(left):
            return "vcv"
        return "spaced"
    if alias_token_is_vowel(value):
        return "v"
    return "cv"
