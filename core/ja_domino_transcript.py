from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from core.ja_lab_generator import parse_ja_filename, split_ja_romaji_syllable
from core.textio_utils import read_text_auto


_PAUSE_MARKERS = {"", "pau", "sil", "sp", "spn", "br", "bre", "breath", "r"}
_SPECIAL_TOKEN_MAP = {
    "n": "N",
    "nn": "N",
    "xn": "N",
    "m": "N",
    "q": "cl",
    "cl": "cl",
    "xtu": "cl",
    "xtsu": "cl",
    "ltu": "cl",
    "ltsu": "cl",
    "r": "pau",
    "br": "pau",
    "bre": "pau",
    "breath": "pau",
    "pau": "pau",
    "sil": "pau",
    "sp": "pau",
    "spn": "pau",
}

_ONSET_TO_DOMINO = {
    "k": "k",
    "g": "g",
    "s": "s",
    "z": "z",
    "t": "t",
    "d": "d",
    "n": "n",
    "h": "h",
    "b": "b",
    "p": "p",
    "m": "m",
    "r": "r",
    "l": "r",
    "w": "w",
    "y": "y",
    "j": "j",
    "sh": "sh",
    "ch": "ch",
    "ts": "ts",
    "f": "f",
    "v": "v",
    "ky": "ky",
    "gy": "gy",
    "ny": "ny",
    "hy": "hy",
    "my": "my",
    "ry": "ry",
    "by": "by",
    "py": "py",
    "dy": "dy",
    "sy": "sh",
    "zy": "j",
    "jy": "j",
    "ty": "ch",
    "cy": "ch",
    "fy": "f",
    "vy": "v",
    "kw": "k",
    "gw": "g",
}

_VOWEL_TO_DOMINO = {
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
}


def _normalize_syllable_token(token: str) -> str:
    text = str(token or "").strip().lower()
    if not text:
        return ""
    if text.endswith("諱ｯ"):
        text = text[:-2]
    return text


def _map_onset_to_domino(onset: str) -> str:
    key = str(onset or "").strip().lower()
    if not key:
        return ""
    direct = _ONSET_TO_DOMINO.get(key, "")
    if direct:
        return direct

    # Heuristic fallback for uncommon transliteration variants.
    if key.endswith("y") and len(key) >= 2:
        stem = key[:-1]
        if stem in {"k", "g", "n", "h", "m", "r", "b", "p", "d"}:
            return _ONSET_TO_DOMINO.get(key, _ONSET_TO_DOMINO.get(stem, ""))
    if key in {"si", "shi"}:
        return "sh"
    if key in {"ti", "chi"}:
        return "ch"
    if key in {"tu", "tsu"}:
        return "ts"
    if key in {"zi", "ji"}:
        return "j"
    if key == "c":
        return "ch"
    return ""


def syllable_to_domino_phones(token: str) -> List[str]:
    syllable = _normalize_syllable_token(token)
    if not syllable:
        return []

    special = _SPECIAL_TOKEN_MAP.get(syllable, "")
    if special:
        return [special]

    onset, vowel = split_ja_romaji_syllable(syllable)
    onset = str(onset or "").strip().lower()
    vowel = str(vowel or "").strip().lower()

    if not onset and vowel in _VOWEL_TO_DOMINO:
        return [_VOWEL_TO_DOMINO[vowel]]
    if not onset and syllable in _VOWEL_TO_DOMINO:
        return [_VOWEL_TO_DOMINO[syllable]]

    mapped_onset = _map_onset_to_domino(onset)
    mapped_vowel = _VOWEL_TO_DOMINO.get(vowel, "")

    if mapped_onset and mapped_vowel:
        return [mapped_onset, mapped_vowel]
    if mapped_onset:
        return [mapped_onset]
    if mapped_vowel:
        return [mapped_vowel]

    # Fallback for consonant-only symbols that can appear in some reclists.
    if syllable in _ONSET_TO_DOMINO:
        return [_ONSET_TO_DOMINO[syllable]]
    if syllable in _VOWEL_TO_DOMINO:
        return [_VOWEL_TO_DOMINO[syllable]]

    return []


def build_domino_phone_plan(
    syllables: Sequence[str],
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    phones: List[str] = []
    word_spans: List[Tuple[str, int, int]] = []

    for raw in syllables or []:
        token = _normalize_syllable_token(raw)
        mapped = syllable_to_domino_phones(token)
        if not mapped:
            continue
        start_idx = len(phones)
        phones.extend(mapped)
        end_idx = len(phones) - 1
        if token not in _PAUSE_MARKERS and end_idx >= start_idx:
            word_spans.append((token, start_idx, end_idx))

    if not phones or phones[0] != "pau":
        phones.insert(0, "pau")
        word_spans = [(label, s + 1, e + 1) for (label, s, e) in word_spans]
    if phones[-1] != "pau":
        phones.append("pau")

    return phones, word_spans


def load_ja_syllables_for_wav(wav_path: str) -> Tuple[List[str], str]:
    wav_abs = os.path.abspath(str(wav_path or "").strip())
    base = os.path.splitext(os.path.basename(wav_abs))[0]
    lab_path = os.path.splitext(wav_abs)[0] + ".lab"

    if os.path.isfile(lab_path):
        content, _enc, err = read_text_auto(lab_path)
        if not err:
            tokens = [t for t in str(content or "").split() if t.strip()]
            if tokens:
                return tokens, "lab"

    parsed = parse_ja_filename(base)
    return [t for t in parsed if str(t).strip()], "filename"


__all__ = [
    "build_domino_phone_plan",
    "load_ja_syllables_for_wav",
    "syllable_to_domino_phones",
]
