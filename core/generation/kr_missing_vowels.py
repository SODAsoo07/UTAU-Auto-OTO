from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from typing import Any, Dict, Tuple


KR_SINGLE_VOWELS = (
    "a",
    "e",
    "i",
    "o",
    "u",
    "eo",
    "eu",
    "ae",
    "oe",
    "wi",
    "wa",
    "we",
    "weo",
    "ya",
    "ye",
    "yo",
    "yeo",
    "yu",
    "ui",
    "eui",
)


def collect_template_aliases(file_groups: Mapping[str, Sequence[str]]) -> set[str]:
    aliases: set[str] = set()
    for group_lines in file_groups.values():
        for line in group_lines:
            parts = str(line).split("=", 1)
            if len(parts) > 1:
                alias = parts[1].split(",", 1)[0].strip().lower()
                aliases.add(alias)
    return aliases


def detect_missing_single_vowel_alias(
    tg_info: Mapping[str, Any],
    *,
    vowels: Sequence[str] = KR_SINGLE_VOWELS,
) -> str:
    vowel_set = set(vowels)
    base_filename = os.path.splitext(str(tg_info.get("real_name", "") or ""))[0].lower()
    base_filename = re.sub(r"long$", "", base_filename)
    norm_key_clean = re.sub(r"long$", "", str(tg_info.get("norm_key", "") or ""))

    tokens = re.split(r"[-_ ]+", base_filename)
    for token in reversed(tokens):
        clean_token = re.sub(r"[^a-z]", "", token)
        if clean_token in vowel_set:
            return clean_token

    if norm_key_clean in vowel_set:
        return norm_key_clean
    return ""


def resolve_single_vowel_span(
    tg_info: Mapping[str, Any],
    *,
    single_vowel_span_by_tg_path: Mapping[str, Tuple[float, float]],
    norm_tg_path_key_fn: Callable[[Any], str],
    load_textgrid_fn: Callable[[Any], Any],
    coerce_interval_tier_fn: Callable[[Any], Any],
    is_interval_like_tier_fn: Callable[[Any], bool],
) -> Tuple[float, float] | None:
    path = tg_info.get("path")
    cached = single_vowel_span_by_tg_path.get(norm_tg_path_key_fn(path))
    if cached is not None:
        return float(cached[0]), float(cached[1])

    try:
        tg = load_textgrid_fn(path)
        phone_tier = next(
            (
                coerce_interval_tier_fn(tier)
                for tier in tg
                if str(getattr(tier, "name", "") or "").strip().lower() == "phones"
                and is_interval_like_tier_fn(tier)
            ),
            None,
        )
        if not phone_tier:
            return None
        intervals = [
            interval
            for interval in phone_tier
            if str(getattr(interval, "mark", "") or "").strip() not in {"", "sil", "spn", "pau"}
        ]
        if len(intervals) != 1:
            return None
        vowel = intervals[0]
        return float(vowel.minTime) * 1000.0, float(vowel.maxTime) * 1000.0
    except Exception:
        return None


def append_kr_missing_single_vowel_aliases(
    *,
    final_lines: MutableSequence[str],
    file_groups: Mapping[str, Sequence[str]],
    tg_entries: Sequence[Mapping[str, Any]],
    single_vowel_span_by_tg_path: Mapping[str, Tuple[float, float]],
    norm_tg_path_key_fn: Callable[[Any], str],
    load_textgrid_fn: Callable[[Any], Any],
    coerce_interval_tier_fn: Callable[[Any], Any],
    is_interval_like_tier_fn: Callable[[Any], bool],
    compute_noninitial_vowel_timing_fn: Callable[[float, float], Tuple[float, float, float, float, float]],
    validate_fn: Callable[[float, float, float, float, float], Tuple[float, float, float, float, float]],
    append_alias_rows_fn: Callable[..., Any],
    generate_openutau: bool,
    alias_suffix: str,
    log_added_fn: Callable[[Mapping[str, Any], str], None],
) -> int:
    template_aliases = collect_template_aliases(file_groups)
    added = 0

    for tg_info in tg_entries:
        alias = detect_missing_single_vowel_alias(tg_info)
        if not alias:
            continue

        v_span = resolve_single_vowel_span(
            tg_info,
            single_vowel_span_by_tg_path=single_vowel_span_by_tg_path,
            norm_tg_path_key_fn=norm_tg_path_key_fn,
            load_textgrid_fn=load_textgrid_fn,
            coerce_interval_tier_fn=coerce_interval_tier_fn,
            is_interval_like_tier_fn=is_interval_like_tier_fn,
        )
        if v_span is None or alias in template_aliases:
            continue

        v_start, v_end = v_span
        log_added_fn(tg_info, alias)
        offset, consonant, cutoff, pre, ovl = compute_noninitial_vowel_timing_fn(
            v_start,
            v_end,
        )
        offset, consonant, cutoff, pre, ovl = validate_fn(
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
        )
        output_name = str(tg_info.get("output_name", tg_info.get("real_name", "")) or "")
        append_alias_rows_fn(
            final_lines,
            output_name or tg_info.get("real_name"),
            alias,
            offset,
            consonant,
            cutoff,
            pre,
            ovl,
            generate_openutau=generate_openutau,
            alias_suffix=alias_suffix,
            alias_type="mono",
            validate_fn=validate_fn,
        )
        added += 1

    return added


__all__ = [
    "KR_SINGLE_VOWELS",
    "append_kr_missing_single_vowel_aliases",
    "collect_template_aliases",
    "detect_missing_single_vowel_alias",
    "resolve_single_vowel_span",
]
