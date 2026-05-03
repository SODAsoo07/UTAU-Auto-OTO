from __future__ import annotations

import re

COARSE_LABELS = [
    "BLANK",
    "C_OBS",
    "C_FR",
    "C_SON",
    "V",
    "SIL",
    "BR",
    "SP",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(COARSE_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

_SIL = {"", "sil", "pau", "pause", "sp", "spn", "r", "rest", "q"}
_BR = {"br", "bre", "breath", "息", "吸"}
_SP = {"cl", "N", "n_mora", "unknown", "uncertain", "noise", "ap"}

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
    "ɑ",
    "ɐ",
    "ʌ",
    "ə",
    "ɨ",
    "ɯ",
    "ɪ",
    "ɛ",
    "æ",
    "ɔ",
}
_OBS = {
    "k",
    "g",
    "kk",
    "gg",
    "kh",
    "t",
    "d",
    "tt",
    "dd",
    "th",
    "p",
    "b",
    "pp",
    "bb",
    "ph",
    "c",
    "ch",
    "j",
    "jj",
    "z",
    "ts",
    "dz",
    "q",
}
_FR = {"s", "ss", "sh", "z", "f", "h", "hy", "v"}
_SON = {"m", "n", "ng", "ŋ", "l", "ɾ", "r", "y", "w", "my", "ny", "ry", "ly"}

_TOKEN_RE = re.compile(r"[A-Za-z가-힣ぁ-ゖァ-ヺー]+")


def normalize_phone(phone: object) -> str:
    text = str(phone or "").strip()
    if not text:
        return ""
    if "/" in text and re.match(r"^[a-z]{2,3}/", text, flags=re.IGNORECASE):
        text = text.split("/", 1)[1]
    text = text.strip().strip('"').strip("'")
    if text.upper() in {"SP", "SIL"}:
        return text.lower()
    return text


def split_label_tokens(label: object) -> list[str]:
    text = str(label or "").strip()
    if not text:
        return []
    if re.search(r"\s", text):
        return [t for t in re.split(r"\s+", text) if t]
    if text.startswith("ko/") or text.startswith("ja/"):
        return [text]
    return [m.group(0) for m in _TOKEN_RE.finditer(text)] or [text]


def coarse_for_phone(phone: object, *, language: str = "") -> str:
    raw = normalize_phone(phone)
    lowered = raw.lower()
    lang = str(language or "").strip().lower()
    if lowered in _SIL:
        return "SIL"
    if lowered in _BR:
        return "BR"
    if raw in _SP or lowered in _SP:
        return "SP"
    if lowered in _VOWELS:
        return "V"
    if lowered == "j" and lang in {"korean", "ko", "kor"}:
        return "C_SON"
    if lowered in _OBS:
        return "C_OBS"
    if lowered in _FR:
        return "C_FR"
    if lowered in _SON:
        return "C_SON"

    # Common kana/hangul labels should be expanded by the language parser before
    # this point. If a raw timed lab still reaches here, keep it usable instead
    # of crashing the dataset builder.
    if re.fullmatch(r"[aeiou]+", lowered):
        return "V"
    if any(v in lowered for v in ("a", "i", "u", "e", "o")) and len(lowered) <= 4:
        return "V" if lowered in _VOWELS else "C_OBS"
    return "SP"


def coarse_id_for_phone(phone: object, *, language: str = "") -> int:
    return LABEL_TO_ID[coarse_for_phone(phone, language=language)]


def is_usable_training_label(label: object) -> bool:
    return coarse_for_phone(label) in {"C_OBS", "C_FR", "C_SON", "V", "SIL", "BR", "SP"}


__all__ = [
    "COARSE_LABELS",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "coarse_for_phone",
    "coarse_id_for_phone",
    "is_usable_training_label",
    "normalize_phone",
    "split_label_tokens",
]
