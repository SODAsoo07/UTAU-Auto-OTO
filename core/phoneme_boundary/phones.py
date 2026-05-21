from __future__ import annotations

import re

from core.model_context.filename import filename_order_tokens

_VOWELS = {
    "a",
    "i",
    "u",
    "e",
    "o",
    "eo",
    "eu",
    "ae",
    "ya",
    "yeo",
    "yo",
    "yu",
    "ye",
    "wa",
    "wae",
    "weo",
    "we",
    "wi",
    "ui",
    "wo",
    "aa",
    "ii",
    "uu",
    "ee",
    "oo",
}
_SIL = {"", "sil", "pau", "pause", "sp", "spn", "r", "rest", "q"}
_BR = {"br", "bre", "breath"}
_OBS = {"k", "g", "kk", "gg", "kh", "t", "d", "tt", "dd", "th", "p", "b", "pp", "bb", "ph", "c", "ch", "j", "jj", "z", "ts", "dz", "q"}
_FR = {"s", "ss", "sh", "z", "f", "h", "hy", "v"}
_SON = {"m", "n", "ng", "l", "r", "y", "w", "my", "ny", "ry", "ly"}


def normalize_language(language: object) -> str:
    text = str(language or "").strip().lower()
    if text in {"ko", "kor", "korean"}:
        return "korean"
    if text in {"ja", "jp", "jpn", "japanese"}:
        return "japanese"
    return text


def phones_from_text(text: object, language: object = "") -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    tokens = filename_order_tokens(raw, language=normalize_language(language))
    if tokens:
        return [str(token).strip() for token in tokens if str(token).strip()]
    return [item for item in re.split(r"[\s,_\-/|']+", raw) if item]


def coarse_for_phone(phone: object, *, language: str = "") -> str:
    raw = str(phone or "").strip().strip("\"'")
    lowered = raw.lower()
    if lowered in _SIL:
        return "SIL"
    if lowered in _BR:
        return "BR"
    if lowered in _VOWELS:
        return "V"
    if lowered == "j" and normalize_language(language) == "korean":
        return "C_SON"
    if lowered in _OBS:
        return "C_OBS"
    if lowered in _FR:
        return "C_FR"
    if lowered in _SON:
        return "C_SON"
    if re.fullmatch(r"[aeiou]+", lowered):
        return "V"
    if any(v in lowered for v in ("a", "i", "u", "e", "o")) and len(lowered) <= 4:
        return "V" if lowered in _VOWELS else "C_OBS"
    return "SP"


__all__ = ["coarse_for_phone", "normalize_language", "phones_from_text"]
