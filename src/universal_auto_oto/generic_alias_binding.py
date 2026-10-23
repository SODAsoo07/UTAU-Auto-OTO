"""Bind aliases to phone occurrences with a language pack instead of per-language rules.

An alias is read as an optional marker (file-start head, after-consonant), one
or more space-separated phone groups and an optional end marker.  Its phones
must appear contiguously in the recording; when they appear more than once the
markers and the alias' spacing choose the occurrence:

* head marker: the occurrence at the file start;
* after-consonant marker: an occurrence right after a consonant;
* end marker: the occurrence at the end of the recording;
* glued trailing consonants ("@b"): a coda followed by another consonant;
* spaced trailing consonants ("@ b"): an onset followed by a vowel;
* unmarked consonant-initial aliases avoid the file start when the pack says
  the file-start take is written with the head marker.

Remaining ties take the first occurrence, like the hand-written routes.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from typing import Sequence

from .domain import AliasPlan, SourceOtoRow
from .language_pack import LanguagePack
from .pack_style import PackStyle

_PITCH_SUFFIX_RE = re.compile(r"[ _]?[A-G](?:#|b)?\d+$")


@dataclass(frozen=True)
class ParsedAlias:
    prefix: str
    suffix: str
    groups: tuple[tuple[str, ...], ...]
    shape: str


def parse_pack_alias(pack: LanguagePack, alias: str) -> ParsedAlias | None:
    parsed = _parse(pack, alias.strip())
    if parsed is None:
        stripped = _PITCH_SUFFIX_RE.sub("", alias.strip())
        if stripped != alias.strip():
            parsed = _parse(pack, stripped)
    return parsed


def _parse(pack: LanguagePack, text: str) -> ParsedAlias | None:
    prefix = suffix = ""
    if len(text) > 1 and pack.head_marker and text.startswith(pack.head_marker):
        prefix, text = pack.head_marker, text[len(pack.head_marker):]
    elif len(text) > 1 and pack.after_consonant_marker and text.startswith(pack.after_consonant_marker):
        prefix, text = pack.after_consonant_marker, text[len(pack.after_consonant_marker):]
    if len(text) > 1 and pack.end_marker and text.endswith(pack.end_marker):
        suffix, text = pack.end_marker, text[: -len(pack.end_marker)]
    words = text.split()
    if not words:
        return None
    groups = []
    for word in words:
        phones = pack.tokenize(word)
        if not phones:
            return None
        groups.append(phones)
    return ParsedAlias(prefix, suffix, tuple(groups), pack.shape_of(groups, prefix, suffix))


def _variants(pack: LanguagePack, groups: Sequence[Sequence[str]]) -> list[tuple[tuple[str, ...], tuple[int, ...], int]]:
    """Flattened phone sequences (alias equivalents expanded), group starts and substitution count."""
    per_phone = [[(phone,), *pack.alias_equivalents.get(phone, ())] for group in groups for phone in group]
    sizes = [len(group) for group in groups]
    result = []
    for choice in itertools.product(*per_phone):
        flat: list[str] = []
        starts: list[int] = []
        position = 0
        for size in sizes:
            starts.append(len(flat))
            for part in choice[position:position + size]:
                flat.extend(part)
            position += size
        substitutions = sum(1 for phone, part in zip((p for g in groups for p in g), choice) if part != (phone,))
        result.append((tuple(flat), tuple(starts), substitutions))
    return result


def bind_pack_aliases(
    pack: LanguagePack,
    rows: Sequence[SourceOtoRow],
    phones: Sequence[str],
    wav_name: str,
    *,
    unmapped: list | None,
    style: PackStyle | None = None,
) -> tuple[AliasPlan, ...]:
    result = []
    for row in rows:
        plan, reason = bind_pack_alias(pack, row.alias, phones, style=style)
        if plan is None:
            detail = json.dumps({"alias": row.alias, "reason": reason, "pack": pack.pack_id}, ensure_ascii=False)
            if unmapped is None:
                from .voicebank_project_adapter import VoicebankReviewRequired

                raise VoicebankReviewRequired("unmapped_alias", wav=wav_name, detail=detail)
            unmapped.append((row, detail))
            continue
        result.append(plan)
    return tuple(result)


def bind_pack_alias(pack: LanguagePack, alias: str, phones: Sequence[str], *,
                    style: PackStyle | None = None) -> tuple[AliasPlan | None, str]:
    parsed = parse_pack_alias(pack, alias)
    if parsed is None:
        return None, "alias_not_tokenized"
    rule = pack.rule_for(parsed.shape)
    if rule is None:
        return None, f"no_template_for_shape:{parsed.shape}"
    candidates = []
    for flat, starts, substitutions in _variants(pack, parsed.groups):
        size = len(flat)
        for index in range(len(phones) - size + 1):
            if tuple(phones[index:index + size]) == flat:
                candidates.append((index, size, starts, substitutions))
    if not candidates:
        return None, "phones_not_in_recording"
    transition = rule.role in {"vc", "cc"}
    index, size, starts, _ = min(candidates, key=lambda item: _rank(pack, parsed, phones, *item, transition=transition))
    plan = AliasPlan(alias, tuple(range(index, index + size)), pack.template_ref(rule.template), rule.role,
                     group_starts=starts,
                     anchor_shifts=dict(style.shifts_for(*style_keys(pack, parsed))) if style else {})
    return plan, ""


def style_keys(pack: LanguagePack, parsed: ParsedAlias) -> tuple[str, str]:
    """(shape|class, shape): the class is that of the consonant at the vowel boundary the
    preutterance sits on (before the vowel for CV, after it for VC), else of the first phone."""
    phones = [phone for group in parsed.groups for phone in group]
    nuclei = [i for i, phone in enumerate(phones) if pack.is_nucleus(phone)]
    if nuclei and nuclei[0] > 0:
        pivot = phones[nuclei[0] - 1]
    elif nuclei and nuclei[0] + 1 < len(phones):
        pivot = phones[nuclei[0] + 1]
    else:
        pivot = phones[0]
    return f"{parsed.shape}|{pack.phone_class(pivot)}", parsed.shape


def _rank(pack: LanguagePack, parsed: ParsedAlias, phones: Sequence[str], index: int, size: int,
          _starts: tuple[int, ...], substitutions: int, *, transition: bool) -> tuple:
    end = index + size
    following = phones[end] if end < len(phones) else None
    preceding = phones[index - 1] if index > 0 else None
    misses = 0
    if parsed.prefix == pack.head_marker:
        misses += index != 0
    elif parsed.prefix == pack.after_consonant_marker:
        misses += preceding is None or pack.is_nucleus(preceding)
    if parsed.suffix == pack.end_marker:
        misses += end != len(phones)
        # The phrase-final take is the last one; prefer later ties.
        return (misses, substitutions, -index)
    last_group = parsed.groups[-1]
    # Only transition aliases (VC, CC) choose between coda and onset takes; trailing
    # consonants of a CV or V alias belong to its rhyme unit (b8n, -8n).
    if transition and not pack.is_nucleus(last_group[-1]):
        if len(parsed.groups) > 1 and not any(pack.is_nucleus(phone) for phone in last_group):
            # "@ b": the consonant starts the next syllable.
            misses += following is None or not pack.is_nucleus(following)
        else:
            # "@b": a coda before another consonant, not the phrase end.
            misses += following is None or pack.is_nucleus(following)
    first = parsed.groups[0][0]
    soft = 0
    if not parsed.prefix and pack.unmarked_consonant_start_avoids_file_start and not pack.is_nucleus(first):
        soft += index == 0
    return (misses, substitutions, soft, index)


__all__ = ["ParsedAlias", "bind_pack_alias", "bind_pack_aliases", "parse_pack_alias", "style_keys"]
