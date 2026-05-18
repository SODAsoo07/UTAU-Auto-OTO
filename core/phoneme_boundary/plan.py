from __future__ import annotations

import os
from collections.abc import Iterable

from core.coarse_crnn.labels import coarse_for_phone
from core.coarse_crnn.lang import normalize_language, phones_from_text
from core.phoneme_boundary.types import AliasPlan, PhonemePlan, PhoneToken


def build_phoneme_plan(
    *,
    wav_path: str,
    aliases: Iterable[str],
    language: str = "",
    duration_ms: float = 0.0,
) -> PhonemePlan:
    lang = normalize_language(language)
    wav_name = os.path.basename(str(wav_path or ""))
    plans: list[AliasPlan] = []
    global_phone_index = 0
    for alias_index, alias in enumerate([str(a or "").strip() for a in aliases]):
        phones: list[PhoneToken] = []
        for phone in phones_from_alias(alias, lang):
            phones.append(
                PhoneToken(
                    text=phone,
                    index=global_phone_index,
                    alias_index=int(alias_index),
                    coarse=coarse_for_phone(phone, language=lang),
                )
            )
            global_phone_index += 1
        plans.append(
            AliasPlan(
                wav_name=wav_name,
                alias=alias,
                alias_index=int(alias_index),
                phones=tuple(phones),
                language=lang,
            )
        )
    return PhonemePlan(
        wav_path=str(wav_path),
        aliases=tuple(plans),
        duration_ms=max(0.0, float(duration_ms or 0.0)),
        language=lang,
    )


def phones_from_alias(alias: object, language: str = "") -> list[str]:
    text = str(alias or "").strip()
    if not text:
        return []
    # UTAU transition aliases are already phone-like when whitespace exists.
    if any(ch.isspace() for ch in text):
        return [p for p in text.split() if p]
    phones = phones_from_text(text, normalize_language(language))
    if len(phones) == 1 and phones[0].lower() == text.lower():
        split = _split_ascii_cv_token(text)
        if split:
            return split
    return phones


def _split_ascii_cv_token(text: str) -> list[str]:
    token = str(text or "").strip().lower()
    if not token.isascii() or not token.isalpha() or len(token) < 2:
        return []
    vowels = (
        "yae",
        "yeo",
        "wae",
        "weo",
        "ya",
        "ye",
        "yo",
        "yu",
        "wa",
        "we",
        "wi",
        "wo",
        "ui",
        "ae",
        "eo",
        "eu",
        "a",
        "i",
        "u",
        "e",
        "o",
    )
    for vowel in vowels:
        if token.endswith(vowel) and len(token) > len(vowel):
            onset = token[: -len(vowel)]
            return [onset, vowel]
    return []


def load_aliases_text(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


__all__ = [
    "build_phoneme_plan",
    "load_aliases_text",
    "phones_from_alias",
]
