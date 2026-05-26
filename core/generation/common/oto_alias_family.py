from __future__ import annotations

import re

ALIAS_ATTACHED_PITCH_SUFFIX_RE = re.compile(r"[A-Ga-g](?:#|b)?[0-8]$")
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
    match = ALIAS_ATTACHED_PITCH_SUFFIX_RE.search(value)
    if match is None:
        return value
    if match.start() <= 0:
        return ""
    stripped = value[: match.start()].rstrip("_- ").strip()
    return stripped or value


def alias_family(alias: str) -> str:
    value = str(alias or "").strip()
    if not value:
        return "blank"
    lowered = value.lower()
    compact = "".join(part for part in lowered.split())
    if (
        lowered in BREATH_ALIASES
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
