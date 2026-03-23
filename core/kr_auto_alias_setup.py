from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable


def _parse_kr_filename_tokens(real_name: str) -> list[str]:
    try:
        from core.lab_generator import _parse_filename
    except Exception:
        return []
    try:
        return list(_parse_filename(os.path.basename(real_name or ""), convert_to_hangul=False) or [])
    except Exception:
        return []


def _build_kr_roman_syllable_expander(split_syllable_parts_fn):
    max_probe = 8
    valid_codas = {
        "",
        "ng", "n", "l", "r", "m", "k", "g", "t", "d", "s", "ss",
        "j", "jj", "ch", "h", "p", "b",
    }

    @lru_cache(maxsize=65536)
    def _segment(token_text: str) -> tuple[str, ...]:
        text = str(token_text or "").strip().lower()
        if not text:
            return tuple()

        @lru_cache(maxsize=None)
        def _walk(start_idx: int):
            if start_idx >= len(text):
                return ((), (0, 0))

            best_tokens = None
            best_score = None
            max_end = min(len(text), start_idx + max_probe)
            for end_idx in range(start_idx + 1, max_end + 1):
                cand = text[start_idx:end_idx]
                onset, vowel, coda = split_syllable_parts_fn(cand)
                rebuilt = f"{onset}{vowel}{coda}".lower()
                if (
                    not vowel
                    or rebuilt != cand
                    or onset.lower() not in _segment.valid_onsets
                    or vowel.lower() not in _segment.valid_vowels
                    or coda.lower() not in valid_codas
                ):
                    continue
                rest = _walk(end_idx)
                if rest is None:
                    continue
                rest_tokens, rest_score = rest
                tokens = (cand,) + rest_tokens
                score = (
                    1 + rest_score[0],
                    len(coda) + rest_score[1],
                )
                if best_score is None or score < best_score:
                    best_tokens = tokens
                    best_score = score

            if best_tokens is None:
                return None
            return best_tokens, best_score

        result = _walk(0)
        if result is None:
            return (text,)
        return tuple(result[0])

    _segment.valid_onsets = set()
    _segment.valid_vowels = set()
    return _segment


def _expand_kr_alias_tokens(tokens, *, split_syllable_parts_fn, kr_vowels=None, kr_consonants=None) -> list[str]:
    segment_roman = _build_kr_roman_syllable_expander(split_syllable_parts_fn)
    segment_roman.valid_onsets = {str(item or "").lower() for item in (kr_consonants or [])}
    segment_roman.valid_onsets.add("")
    segment_roman.valid_vowels = {str(item or "").lower() for item in (kr_vowels or [])}
    out: list[str] = []
    for raw in tokens or []:
        text = str(raw or "").strip().lower()
        if not text:
            continue
        for seg in segment_roman(text):
            onset, vowel, coda = split_syllable_parts_fn(seg)
            rebuilt = f"{onset}{vowel}{coda}".lower()
            if vowel and rebuilt == seg:
                out.append(seg)
    return out


def expand_kr_filename_alias_tokens(real_name: str, *, split_syllable_parts_fn, kr_vowels=None, kr_consonants=None) -> list[str]:
    return _expand_kr_alias_tokens(
        _parse_kr_filename_tokens(real_name),
        split_syllable_parts_fn=split_syllable_parts_fn,
        kr_vowels=kr_vowels,
        kr_consonants=kr_consonants,
    )


def build_kr_auto_file_groups(
    *,
    tg_entries,
    auto_gen_format: str,
    log_fn: Callable[[str], None],
    load_textgrid_fn: Callable[[str], object],
    decompose_hangul_to_roman_fn: Callable[[str], list[str]],
    split_syllable_parts_fn: Callable[[str], tuple[str, str, str]],
    kr_vowels: set[str],
    kr_consonants: set[str],
) -> dict[str, list[str]]:
    file_groups: dict[str, list[str]] = {}
    coda_alias_map = {
        "ng": "NG",
        "n": "N",
        "l": "L",
        "r": "L",
        "m": "M",
        "k": "K",
        "g": "K",
        "t": "T",
        "d": "T",
        "s": "T",
        "ss": "T",
        "j": "T",
        "jj": "T",
        "ch": "T",
        "h": "H",
        "p": "P",
        "b": "P",
    }

    def _word_to_roman(word: str) -> str:
        parts = []
        for ch in word:
            parts.extend(decompose_hangul_to_roman_fn(ch))
        return "".join(parts).lower()

    def _is_low_trust_word_mark(mark: str) -> bool:
        text = str(mark or "").strip().lower()
        return text in {"", "sil", "spn", "pau", "<unk>", "unk"}

    def _select_alias_tokens(real_name: str, word_marks: list[str]) -> list[str]:
        filename_tokens = expand_kr_filename_alias_tokens(
            real_name,
            split_syllable_parts_fn=split_syllable_parts_fn,
            kr_vowels=kr_vowels,
            kr_consonants=kr_consonants,
        )
        word_tokens = _expand_kr_alias_tokens(
            [_word_to_roman(mark) for mark in word_marks if not _is_low_trust_word_mark(mark)],
            split_syllable_parts_fn=split_syllable_parts_fn,
            kr_vowels=kr_vowels,
            kr_consonants=kr_consonants,
        )

        low_trust = any(_is_low_trust_word_mark(mark) for mark in word_marks)
        if filename_tokens and (
            low_trust
            or not word_tokens
            or len(word_marks) <= 1 < len(filename_tokens)
            or len(word_tokens) < len(filename_tokens)
        ):
            return filename_tokens
        return word_tokens or filename_tokens

    for tg_info in tg_entries:
        tg_path = tg_info["path"]
        real_name = tg_info["real_name"]
        try:
            tg = load_textgrid_fn(tg_path)
            word_tier = None
            for tier in tg:
                if hasattr(tier, "name") and tier.name == "words":
                    word_tier = tier
                    break
            if not word_tier:
                continue

            wd_intervals = [i for i in word_tier if str(i.mark or "").strip()]
            if not wd_intervals:
                continue

            lines: list[str] = []
            base_name = os.path.splitext(real_name)[0].lower()
            alias_tokens = _select_alias_tokens(real_name, [i.mark for i in wd_intervals])
            if not alias_tokens:
                continue
            is_long = base_name.endswith("long") or len(alias_tokens) == 1

            if is_long:
                for roman in alias_tokens:
                    alias = roman.capitalize() if roman not in kr_vowels else roman
                    lines.append(f"{real_name}={alias},0,0,0,0,0")
            else:
                for idx, roman in enumerate(alias_tokens):
                    onset, vowel, _coda = split_syllable_parts_fn(roman)
                    if auto_gen_format == "vcv":
                        if idx == 0:
                            lines.append(f"{real_name}=- {roman},0,0,0,0,0")
                        else:
                            prev_roman = alias_tokens[idx - 1]
                            _, prev_vowel, prev_coda = split_syllable_parts_fn(prev_roman)
                            prev_coda_alias = coda_alias_map.get((prev_coda or "").lower(), "")
                            if prev_coda_alias:
                                bridge_right = roman if onset else (vowel or roman)
                                lines.append(f"{real_name}={prev_coda_alias} {bridge_right},0,0,0,0,0")
                            else:
                                lines.append(f"{real_name}={(prev_vowel or 'a')} {roman},0,0,0,0,0")
                    elif auto_gen_format in {"cv", "cvc", "cvvc"}:
                        current_alias = roman
                        if idx > 0 and auto_gen_format in {"cvc", "cvvc"}:
                            prev_roman = alias_tokens[idx - 1]
                            _, _prev_vowel, prev_coda = split_syllable_parts_fn(prev_roman)
                            if not onset and vowel and prev_coda:
                                current_alias = f"{prev_coda.lower()}{vowel}"
                        lines.append(f"{real_name}={current_alias},0,0,0,0,0")
                        if idx > 0 and auto_gen_format in {"cvc", "cvvc"}:
                            prev_roman = alias_tokens[idx - 1]
                            _, prev_vowel, prev_coda = split_syllable_parts_fn(prev_roman)
                            prev_vowel = prev_vowel or "a"
                            prev_coda_alias = coda_alias_map.get((prev_coda or "").lower(), "")
                            if prev_coda_alias:
                                bridge_right = onset or vowel or roman
                                lines.append(f"{real_name}={prev_coda_alias} {bridge_right},0,0,0,0,0")
                            elif onset:
                                lines.append(f"{real_name}={prev_vowel} {onset},0,0,0,0,0")
                            else:
                                lines.append(f"{real_name}={prev_vowel} {(vowel or 'a')},0,0,0,0,0")

            if lines:
                file_groups[real_name] = lines
        except Exception as exc:
            log_fn(f"경고: {real_name} 자동 템플릿 생성 실패 ({exc})")

    return file_groups


__all__ = [
    "build_kr_auto_file_groups",
    "expand_kr_filename_alias_tokens",
]
