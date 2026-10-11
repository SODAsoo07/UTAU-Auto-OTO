from __future__ import annotations

import unicodedata


_VOWELS = {"a", "i", "u", "e", "o", "n"}
_BASE_PHONES: dict[str, tuple[str, ...]] = {
    "息": ("exh",),
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
    "\u308f": ("w", "a"), "\u3090": ("w", "i"), "\u3091": ("w", "e"), "\u3092": ("w", "o"),
    "\u3094": ("v", "u"),
}
_SMALL_VOWELS = {"\u3041": "a", "\u3043": "i", "\u3045": "u", "\u3047": "e", "\u3049": "o"}
_SMALL_YOON = {"\u3083": "a", "\u3085": "u", "\u3087": "o"}


def parse_kana_slots(text: str) -> list[tuple[str, ...]]:
    """Return one phone tuple per Japanese mora-like filename slot."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    if not normalized:
        return []
    slots: list[list[str]] = []
    pending_geminate = False
    seen_kana = False
    for raw_character in normalized:
        character = _katakana_to_hiragana(raw_character)
        if character in {"_", "-", " ", "\u3000", "\u30fb", "\u30fc", "'"}:
            continue
        if raw_character.isascii():
            if raw_character in set("0123456789cdefgab#"):
                continue
            return []
        if character == "\u3063":
            pending_geminate = True
            seen_kana = True
            continue
        if character in _SMALL_YOON:
            if not slots:
                return []
            if slots[-1] and slots[-1][-1] in {"i", "e"}:
                slots[-1].pop()
            slots[-1].extend(("y", _SMALL_YOON[character]))
            seen_kana = True
            continue
        if character in _SMALL_VOWELS:
            _apply_small_vowel(slots, _SMALL_VOWELS[character])
            seen_kana = True
            continue
        parsed = _BASE_PHONES.get(character)
        if parsed is None:
            return []
        slot = list(parsed)
        if pending_geminate and parsed[0] not in _VOWELS:
            slot.insert(0, parsed[0])
        pending_geminate = False
        slots.append(slot)
        seen_kana = True
    return [tuple(slot) for slot in slots] if seen_kana else []


def _apply_small_vowel(slots: list[list[str]], vowel: str) -> None:
    if not slots:
        slots.append([vowel])
        return
    slot = slots[-1]
    last = slot[-1]
    previous = slot[-2] if len(slot) >= 2 else ""
    if last == "i":
        slot.pop()
        slot.extend(("y", vowel))
    elif last == "u" and previous in {"k", "g"}:
        slot.pop()
        slot.extend(("w", vowel))
    elif last == "u" and previous and previous not in {"f", "v", "ts"}:
        slot.pop()
        slot.extend(("w", vowel))
    elif last == "u" and (not previous or previous in _VOWELS):
        slot.pop()
        slot.extend(("w", vowel))
    elif last in {"a", "i", "u", "e", "o"}:
        slot[-1] = vowel
    else:
        slot.append(vowel)


def _katakana_to_hiragana(character: str) -> str:
    codepoint = ord(character)
    if 0x30A1 <= codepoint <= 0x30F6:
        return chr(codepoint - 0x60)
    return character


__all__ = ["parse_kana_slots"]
