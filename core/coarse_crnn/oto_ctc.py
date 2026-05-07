"""CTC alignment head: phone vocabulary and tokenization helpers.

This module provides a fixed phone vocabulary used by the CTC alignment head
(see [[CRNN-정확도-개선-방안-v2]] section 3-A).

Design notes:
- ``id=0`` is the CTC blank token.
- ``id=1`` is the unknown phone fallback (``<unk>``).
- The vocabulary is a static union of Korean / Japanese / ASCII phones produced
  by :func:`core.coarse_crnn.lang.phones_from_text`. Unknown tokens are mapped
  to ``<unk>`` rather than being dropped, so the CTC loss still sees something
  reasonable for cross-language voicebanks.
- The vocabulary is intentionally small and frozen here. If new tokens appear
  later, append at the end so checkpoints remain backward compatible.
"""
from __future__ import annotations

from typing import Iterable

from core.coarse_crnn.lang import phones_from_text

CTC_BLANK_ID = 0
CTC_UNK_ID = 1

# Static vocab: blank, unk, then phones grouped by family.
# Do NOT reorder existing entries; only append at the end.
CTC_PHONE_VOCAB: tuple[str, ...] = (
    "<blank>",
    "<unk>",
    # Vowels (KR + JA + EN ascii)
    "a", "i", "u", "e", "o",
    "eo", "eu", "ae", "oe", "ya", "ye", "yo", "yu", "wa", "wi", "we", "wo", "ui",
    # Consonants (KR jamo roman + JA onsets)
    "k", "g", "n", "d", "t", "r", "l", "m", "b", "p", "s", "j", "ch", "h",
    "kk", "tt", "pp", "ss", "jj", "ng",
    "sh", "ts", "ky", "gy", "ny", "by", "py", "my", "ry", "hy",
    "f", "v", "w", "y", "z",
    # Special / non-phonemic
    "br", "sil", "pau", "n_special", "cl",
)

CTC_VOCAB_SIZE: int = len(CTC_PHONE_VOCAB)

_VOCAB_INDEX: dict[str, int] = {token: idx for idx, token in enumerate(CTC_PHONE_VOCAB)}


def _normalize_phone_token(token: object) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    # phones_from_text emits "N" for Japanese mora-final n. Map to a
    # non-colliding vocab key so it does not clash with the consonant "n".
    if text == "N":
        return "n_special"
    return text.lower()


def phone_ids_from_alias(alias: object, language: object = "") -> list[int]:
    """Return CTC target phone ids for an alias string.

    Returns an empty list when the alias produces no phones (e.g. blank alias
    rows). The caller is responsible for masking these out of the CTC loss.
    """
    out: list[int] = []
    for phone in phones_from_text(alias, language):
        key = _normalize_phone_token(phone)
        if not key:
            continue
        out.append(_VOCAB_INDEX.get(key, CTC_UNK_ID))
    return out


def phone_ids_from_phones(phones: Iterable[object]) -> list[int]:
    """Convert an already-tokenized phone iterable into CTC ids."""
    out: list[int] = []
    for phone in phones:
        key = _normalize_phone_token(phone)
        if not key:
            continue
        out.append(_VOCAB_INDEX.get(key, CTC_UNK_ID))
    return out


__all__ = [
    "CTC_BLANK_ID",
    "CTC_UNK_ID",
    "CTC_PHONE_VOCAB",
    "CTC_VOCAB_SIZE",
    "phone_ids_from_alias",
    "phone_ids_from_phones",
]
