"""Explicit mixed kana/romaji CVVC recording conventions (no OTO timing input)."""
from __future__ import annotations

import re
from pathlib import Path

from .kana import parse_kana_slots


def special_slots(name: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Return phones and alias surfaces only for the supported recording families."""
    stem = Path(name).stem.strip('_')
    long = re.fullmatch(r'([あいうえおん])\1hh', stem)
    if long:
        surface = long[1]
        nucleus = tuple(parse_kana_slots(surface)[0])
        return ((nucleus, surface + 'L'), (('sil',), ''), (nucleus, surface), (('exh',), 'H'))
    if re.fullmatch(r'(?:[lx][らりるれろ]|ん|_)+', stem) and stem[0] in 'lx':
        units = re.findall(r'[lx][らりるれろ]|ん', stem)
        result = []
        for unit in units:
            if unit == 'ん':
                result.append((('n',), unit))
            else:
                vowel = parse_kana_slots(unit[1:])[0][-1]
                phones = ('l', vowel) if unit[0] == 'l' else ('trill', 'r', vowel)
                result.append((phones, unit))
        return tuple(result)
    if re.fullmatch(r'[えいおう]ん(?:ng|m|n)[かきくけこぱぴぷぺぽたてとぃぅ]+(?:_[えいおう]ん(?:ng|m|n)[かきくけこぱぴぷぺぽたてとぃぅ]+)+', stem):
        result = []
        for chunk in stem.split('_'):
            match = re.fullmatch(r'([えいおう])ん(ng|m|n)(.+)', chunk)
            if not match:
                return ()
            result.append((tuple(parse_kana_slots(match[1])[0]), match[1]))
            # The tag identifies the target of an internal nasal transition.
            result.append((('n', match[2]), 'ん' + match[2]))
            tail = parse_kana_slots(match[3])
            if len(tail) != 1:
                return ()
            result.append((tuple(tail[0]), match[3]))
        return tuple(result)
    if re.fullmatch(r'あ(?:(?:sh|s)(?:か|き|く|け|こ|た|てぃ|とぅ|て|と))+', stem):
        result = [(('a',), 'あ')]
        for consonant, kana in re.findall(r'(sh|s)(か|き|く|け|こ|た|てぃ|とぅ|て|と)', stem[1:]):
            # No invented vowel between the fricative and following CV.
            result.append(((consonant,), consonant))
            result.append((tuple(parse_kana_slots(kana)[0]), kana))
        return tuple(result)
    return ()


def alias_part_key(part: str) -> str | None:
    match = re.fullmatch(r'([lx])([らりるれろ])', part)
    if match:
        return match[1] + ('r' if match[1] == 'x' else '') + parse_kana_slots(match[2])[0][-1]
    match = re.fullmatch(r'ん(ng|m|n)', part)
    if match:
        return match[1]
    match = re.fullmatch(r'([あいうえおん])L', part)
    if match:
        return parse_kana_slots(match[1])[0][-1] + 'L'
    return None
