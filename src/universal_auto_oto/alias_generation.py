"""Generate a format-specific alias list from recording file names.

Used when no oto.ini is given. The aliases follow the common conventions of
hand-made banks (see docs/FEATURE_ROADMAP_2026-10-11.md):

- CV:   ``ka`` per syllable
- CVVC: ``- ka`` head, ``ka``, ``a k`` (vowel to next consonant), ``a i`` (vowel to vowel),
        ``a n``/``n k`` around codas, ``a -`` at the end, ``a R`` before a breath
- VCV:  ``- ka`` head, ``a ka`` for every following syllable, ``a k`` for a final coda,
        otherwise ``a -`` at the end; run-together Korean names are segmented by list structure
- CVC:  CVVC, with codas written as one syllable (``ang``) instead of ``a ng``

Kana file names give kana CV aliases (``か``, ``a か``) and romaji vowels and
consonants (``a k``). The rows only name aliases; timing always comes from the
acoustic decoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .domain import SourceOtoRow
from .kana import parse_kana_slots
from .japanese_cvvc import special_slots
from .planning import recording_tokens
from .voicebank_project_adapter import (
    VoicebankReviewRequired,
    _is_pitch_token,
    _kana_surface_slots,
    _segment_korean_vcv_compact,
    parse_filename_phone_slots,
)


SPECIAL_PHONES = frozenset({"vf", "exh", "q", "sil", "pau", "br"})
_VOWEL_LETTERS = frozenset("aeiou")


@dataclass(frozen=True)
class _Syllable:
    onset: tuple[str, ...]
    vowel: str  # nucleus as written, e.g. "a", "wa", "eu"
    coda: tuple[str, ...]
    surface: str  # CV text for the alias
    after_pause: bool = False


def generate_alias_rows(
    wav_name: str,
    *,
    format_hint: str,
    language_hint: str | None,
) -> list[SourceOtoRow]:
    """Alias rows for one recording; raises VoicebankReviewRequired when the name cannot be parsed."""

    aliases = generate_aliases(wav_name, format_hint=format_hint, language_hint=language_hint)
    return [SourceOtoRow(source_wav=wav_name, alias=alias, source_index=index) for index, alias in enumerate(aliases)]


def generate_aliases(wav_name: str, *, format_hint: str, language_hint: str | None) -> list[str]:
    format_type = format_hint.strip().casefold()
    if format_type not in {"cv", "cvc", "cvvc", "vcv"}:
        raise ValueError(f"unsupported format for alias generation: {format_hint}")
    if format_type == 'cvvc' and (language_hint or '').casefold() in {'ja', 'jp', 'japanese'}:
        special = special_slots(wav_name)
        if special:
            return _japanese_special_aliases(special, wav_name)
    try:
        slots = parse_filename_phone_slots(wav_name, format_hint=format_type, language_hint=language_hint)
    except VoicebankReviewRequired:
        # Run-together Korean VCV names (_gagageogage…K) are segmented by the list structure.
        compact = (_segment_korean_vcv_compact(wav_name)
                   if format_type == "vcv" and (language_hint or "").casefold() in {"ko", "kr", "korean"} else ())
        if not compact:
            raise
        slots = compact
    syllables = _syllables(wav_name, slots, language_hint)
    aliases: list[str] = []
    previous: _Syllable | None = None
    for item in syllables:
        if item == "exh":
            if previous is not None and format_type != "cv":
                aliases.append(f"{_left(previous, format_type)} R")
            previous = None
            continue
        if item is None:  # fry, glottal stop or pause: the phrase restarts
            if previous is not None and format_type != "cv":
                aliases.append(f"{_left(previous, format_type)} -")
            previous = None
            continue
        if format_type == "cv":
            aliases.append(item.surface)
            continue
        if item.after_pause:
            previous = None
        if format_type == "vcv":
            aliases.append(f"{'-' if previous is None else _left(previous, format_type)} {item.surface}")
            # A closing coda is its own row (a k, a ng) and a breath after it too (ng h).
            ends = (_vowel_tail(item.vowel), *item.coda)
            aliases.extend(f"{left} {right}" for left, right in zip(ends, ends[1:]))
            previous = item
            continue
        # CVVC / CVC
        if previous is None:
            aliases.append(f"- {item.surface}")
        else:
            left = _left(previous, format_type)
            if item.onset:
                aliases.append(f"{left} {''.join(item.onset)}")
            else:
                aliases.append(f"{left} {item.surface}")
        aliases.append(item.surface)
        if item.coda:
            aliases.append(_coda_alias(item, format_type))
        previous = item
    if previous is not None and format_type != "cv" and not (format_type == "vcv" and previous.coda):
        aliases.append(f"{_left(previous, format_type)} -")
    return aliases


def _japanese_special_aliases(
    units: Sequence[tuple[tuple[str, ...], str]], wav_name: str,
) -> list[str]:
    if units[0][1].endswith('L'):
        return [units[0][1], f'{units[0][0][-1]} H']
    if any(surface in {'s', 'sh'} for _, surface in units):
        return [f'{units[i-1][1]} {surface}' for i, (_, surface) in enumerate(units)
                if i and units[i-1][1] in {'s', 'sh'}]
    if any(surface.startswith('ん') and len(surface) > 1 for _, surface in units):
        return [alias for i, (_, surface) in enumerate(units)
                if i and surface.startswith('ん') and len(surface) > 1
                for alias in (f'{units[i-1][0][-1]} {surface}', f'n {surface}')]
    aliases = []
    for i, (phones, surface) in enumerate(units):
        if surface == 'ん':
            continue
        if i == 0 or '_' in Path(wav_name).stem.strip('_'):
            aliases.append(f'- {surface}')
        aliases.append(surface)
        if i:
            aliases.append(f'{units[i-1][0][-1]} {surface[0]}')
    return aliases


def generate_bank_alias_rows(
    wav_names: Iterable[str],
    *,
    format_hint: str,
    language_hint: str | None,
) -> tuple[dict[str, list[SourceOtoRow]], list[dict[str, str]]]:
    """Rows per WAV with every alias kept once per bank (first file in name order wins)."""

    seen: set[str] = set()
    rows_by_wav: dict[str, list[SourceOtoRow]] = {}
    skipped: list[dict[str, str]] = []
    for wav_name in wav_names:
        try:
            aliases = generate_aliases(Path(wav_name).name, format_hint=format_hint, language_hint=language_hint)
        except VoicebankReviewRequired as error:
            skipped.append({"wav": wav_name, "reason": error.reason, "detail": str(error.detail or "")})
            continue
        unique = [alias for alias in dict.fromkeys(aliases) if alias not in seen]
        seen.update(unique)
        rows_by_wav[wav_name] = [
            SourceOtoRow(source_wav=wav_name, alias=alias, source_index=index) for index, alias in enumerate(unique)
        ]
    return rows_by_wav, skipped


def _syllables(
    wav_name: str,
    slots: Sequence[Sequence[str]],
    language_hint: str | None = None,
) -> list[_Syllable | str | None]:
    stem = Path(wav_name).stem
    surfaces = list(_kana_surface_slots(stem))
    kana = bool(surfaces)
    if not kana:
        surfaces = [token for token in recording_tokens(Path(wav_name).name)]
        if surfaces and _is_pitch_token(surfaces[-1]):
            surfaces.pop()
        surfaces = [token.strip("_'- ") for token in surfaces if token.strip("_'- ")]
    regular = [slot for slot in slots if not _is_special_slot(slot)]
    use_surfaces = len(surfaces) == len(regular)
    pauses = _pause_positions(stem) if not kana else frozenset()
    if pauses and len(_split_tokens(stem)) != len(regular):
        pauses = frozenset()
    japanese = (language_hint or "").casefold() in {"ja", "jp", "japanese"}
    result: list[_Syllable | str | None] = []
    surface_index = 0
    for slot in slots:
        if _is_special_slot(slot):
            result.append("exh" if tuple(slot) == ("exh",) else None)
            continue
        phones = [phone for phone in slot if phone not in SPECIAL_PHONES]
        vowel_index = next((i for i, phone in enumerate(phones) if _has_vowel(phone)), None)
        if vowel_index is None:
            # Syllabic nasal or liquid (Japanese ん, Korean -M): a nucleus of its own.
            onset: tuple[str, ...] = ()
            vowel = phones[0] if phones else ""
            coda: tuple[str, ...] = ()
        else:
            onset = tuple(phones[:vowel_index])
            vowel = phones[vowel_index]
            coda = tuple(phones[vowel_index + 1:])
        surface = surfaces[surface_index] if use_surfaces else ""
        after_pause = surface_index in pauses
        surface_index += 1
        if not (kana and surface and not coda):
            surface = "".join((*onset, vowel))
            if japanese:
                # Japanese banks name CV aliases in kana even when the files use romaji.
                surface = _romaji_to_kana().get(surface, surface)
        result.append(_Syllable(onset=onset, vowel=vowel, coda=coda, surface=surface, after_pause=after_pause))
    return result


_SEPARATOR_RUN_RE = re.compile(r"([_'\-]+)")


def _split_tokens(stem: str) -> list[str]:
    return [part for part in _SEPARATOR_RUN_RE.split(stem) if part and not _SEPARATOR_RUN_RE.fullmatch(part)]


def _pause_positions(stem: str) -> frozenset[int]:
    """Token indices after a sung pause: a separator run of two or more marks (``'_'``, ``'-'``)."""

    heads, count, pending = set(), 0, False
    for part in _SEPARATOR_RUN_RE.split(stem):
        if not part:
            continue
        if _SEPARATOR_RUN_RE.fullmatch(part):
            pending = pending or (len(part) > 1 and count > 0 and ("-" in part or "_" in part))
            continue
        if pending:
            heads.add(count)
        pending = False
        count += 1
    return frozenset(heads)


@lru_cache(maxsize=1)
def _romaji_to_kana() -> dict[str, str]:
    """Romaji spelling -> hiragana, built from the kana parser so both agree (きゃ -> kya)."""

    base = [chr(code) for code in range(0x3041, 0x3095)]
    smalls = ("ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ")
    small_set = set(smalls) | {"っ"}
    candidates = [kana for kana in base if kana not in small_set]
    candidates += [kana + small for kana in candidates for small in smalls]
    table: dict[str, str] = {}
    best: dict[str, int] = {}
    for kana in candidates:
        slots = parse_kana_slots(kana)
        if len(slots) != 1:
            continue
        key, rank = "".join(slots[0]), _kana_spelling_rank(kana)
        if rank < best.get(key, len(_KANA_SPELLING_RANKS)):
            table[key], best[key] = kana, rank
    return table


# Several kana spell one romaji syllable; the usual one wins (si = すぃ not さぃ, ti = てぃ, tu = とぅ,
# wi = うぃ not ゐ).
_KANA_SPELLING_RANKS = ("plain", "i/u base", "matching e/o base", "other e/o base", "a base", "archaic")


def _kana_spelling_rank(kana: str) -> int:
    if kana in {"ゐ", "ゑ"}:
        return _KANA_SPELLING_RANKS.index("archaic")
    if len(kana) == 1:
        return _KANA_SPELLING_RANKS.index("plain")
    base_vowel = parse_kana_slots(kana[0])[0][-1]
    if base_vowel in {"i", "u"}:
        return _KANA_SPELLING_RANKS.index("i/u base")
    if base_vowel in {"e", "o"}:  # てぃ/でぃ take the e column, とぅ/どぅ the o column
        matching = (kana[1] == "ぅ") == (base_vowel == "o")
        return _KANA_SPELLING_RANKS.index("matching e/o base" if matching else "other e/o base")
    return _KANA_SPELLING_RANKS.index("a base")


def _left(syllable: _Syllable, format_type: str) -> str:
    """Text for the sound a syllable ends on, as the left half of a VC/VCV/VV alias."""

    if syllable.coda:
        return syllable.coda[-1]
    return _vowel_tail(syllable.vowel)


def _coda_alias(syllable: _Syllable, format_type: str) -> str:
    vowel = _vowel_tail(syllable.vowel)
    coda = "".join(syllable.coda)
    return f"{vowel}{coda}" if format_type == "cvc" else f"{vowel} {coda}"


def _vowel_tail(vowel: str) -> str:
    value = vowel.casefold()
    if value in {"ui", "eui"}:
        return "i"
    if len(value) > 1 and value[0] in "yw" and value[1] in _VOWEL_LETTERS:
        value = value[1:]
    return value


def _has_vowel(phone: str) -> bool:
    return any(character in _VOWEL_LETTERS for character in phone.casefold())


def _is_special_slot(slot: Sequence[str]) -> bool:
    return bool(slot) and all(phone in SPECIAL_PHONES for phone in slot)


__all__ = ["generate_alias_rows", "generate_aliases", "generate_bank_alias_rows"]
