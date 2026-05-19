from __future__ import annotations

import unicodedata


_VOWELS = {"a", "i", "u", "e", "o", "n"}

_BASE_PHONES: dict[str, tuple[str, ...]] = {
    "\u3042": ("a",), "\u3044": ("i",), "\u3046": ("u",), "\u3048": ("e",), "\u304a": ("o",),
    "\u3093": ("n",),
    "\u304b": ("k", "a"), "\u304d": ("k", "i"), "\u304f": ("k", "u"), "\u3051": ("k", "e"), "\u3053": ("k", "o"),
    "\u304c": ("g", "a"), "\u304e": ("g", "i"), "\u3050": ("g", "u"), "\u3052": ("g", "e"), "\u3054": ("g", "o"),
    "\u3055": ("s", "a"), "\u3057": ("sh", "i"), "\u3059": ("s", "u"), "\u305b": ("s", "e"), "\u305d": ("s", "o"),
    "\u3056": ("z", "a"), "\u3058": ("j", "i"), "\u305a": ("z", "u"), "\u305c": ("z", "e"), "\u305e": ("z", "o"),
    "\u305f": ("t", "a"), "\u3061": ("ch", "i"), "\u3064": ("ts", "u"), "\u3066": ("t", "e"), "\u3068": ("t", "o"),
    "\u3060": ("d", "a"), "\u3062": ("j", "i"), "\u3065": ("z", "u"), "\u3067": ("d", "e"), "\u3069": ("d", "o"),
    "\u306a": ("n", "a"), "\u306b": ("n", "i"), "\u306c": ("n", "u"), "\u306d": ("n", "e"), "\u306e": ("n", "o"),
    "\u306f": ("h", "a"), "\u3072": ("h", "i"), "\u3075": ("f", "u"), "\u3078": ("h", "e"), "\u307b": ("h", "o"),
    "\u3070": ("b", "a"), "\u3073": ("b", "i"), "\u3076": ("b", "u"), "\u3079": ("b", "e"), "\u307c": ("b", "o"),
    "\u3071": ("p", "a"), "\u3074": ("p", "i"), "\u3077": ("p", "u"), "\u307a": ("p", "e"), "\u307d": ("p", "o"),
    "\u307e": ("m", "a"), "\u307f": ("m", "i"), "\u3080": ("m", "u"), "\u3081": ("m", "e"), "\u3082": ("m", "o"),
    "\u3084": ("y", "a"), "\u3086": ("y", "u"), "\u3088": ("y", "o"),
    "\u3089": ("r", "a"), "\u308a": ("r", "i"), "\u308b": ("r", "u"), "\u308c": ("r", "e"), "\u308d": ("r", "o"),
    "\u308f": ("w", "a"), "\u3092": ("w", "o"),
    "\u3094": ("v", "u"),
    "\u3090": ("w", "i"), "\u3091": ("w", "e"),
}

_SMALL_VOWELS = {
    "\u3041": "a",
    "\u3043": "i",
    "\u3045": "u",
    "\u3047": "e",
    "\u3049": "o",
}

_SMALL_YOON = {
    "\u3083": "a",
    "\u3085": "u",
    "\u3087": "o",
}

_SEPARATORS = {"_", "-", " ", "\u3000", "\u30fb", "\u30fc"}
_ASCII_SKIP = set("0123456789cdefgab#")


def parse_kana_text(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    if not normalized:
        return []
    phones: list[str] = []
    pending_geminate = False
    seen_kana = False
    for raw_char in normalized:
        char = _katakana_to_hiragana(raw_char)
        if char in _SEPARATORS:
            continue
        if raw_char.isascii():
            if raw_char in _ASCII_SKIP:
                continue
            return []
        if char == "\u3063":
            pending_geminate = True
            seen_kana = True
            continue
        if char in _SMALL_YOON:
            if not phones:
                return []
            if phones[-1] in {"i", "e"}:
                phones.pop()
            phones.extend(("y", _SMALL_YOON[char]))
            seen_kana = True
            continue
        if char in _SMALL_VOWELS:
            _apply_small_vowel(phones, _SMALL_VOWELS[char])
            seen_kana = True
            continue
        parsed = _BASE_PHONES.get(char)
        if parsed is None:
            return []
        if pending_geminate:
            first = parsed[0]
            if first not in _VOWELS:
                phones.append(first)
            pending_geminate = False
        phones.extend(parsed)
        seen_kana = True
    return phones if seen_kana else []


def _apply_small_vowel(phones: list[str], vowel: str) -> None:
    if not phones:
        phones.append(vowel)
        return
    last = phones[-1]
    prev = phones[-2] if len(phones) >= 2 else ""
    if last == "i":
        phones.pop()
        phones.extend(("y", vowel))
    elif last == "u" and (not prev or prev in _VOWELS):
        phones.pop()
        phones.extend(("w", vowel))
    elif last in {"a", "i", "u", "e", "o"}:
        phones.pop()
        phones.append(vowel)
    else:
        phones.append(vowel)


def _katakana_to_hiragana(char: str) -> str:
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char


__all__ = ["parse_kana_text"]
