from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Dict


def build_kr_word_syllables(
    *,
    word_intervals: Sequence[Any],
    phone_intervals: Sequence[Any],
    force_words_phone_fill: bool,
    build_interval_lookup_fn: Callable[[Sequence[Any]], Any],
    intervals_within_bounds_fn: Callable[[Any, float, float], Sequence[Any]],
    synthesize_word_phones_fn: Callable[[Any, float, float, Callable[[str], Sequence[str]]], Sequence[Any]],
    decompose_hangul_to_roman_fn: Callable[[str], Sequence[str]],
    kr_cv_kernel_fn: Callable[[str], str],
) -> list[Dict[str, Any]]:
    if not word_intervals:
        return []

    words_phone_lookup = build_interval_lookup_fn(phone_intervals)
    syllables: list[Dict[str, Any]] = []
    for word in word_intervals:
        w_start = word.minTime
        w_end = word.maxTime
        mark = word.mark
        s_phones = list(
            intervals_within_bounds_fn(
                words_phone_lookup,
                float(w_start) - 0.01,
                float(w_end) + 0.01,
            )
        )
        if force_words_phone_fill and not s_phones:
            s_phones = list(
                synthesize_word_phones_fn(
                    mark,
                    float(w_start),
                    float(w_end),
                    decompose_hangul_to_roman_fn,
                )
            )

        roman_parts = []
        for ch in mark:
            roman_parts.extend(decompose_hangul_to_roman_fn(ch))
        roman_raw = "".join(roman_parts).lower()
        syllables.append(
            {
                "word": mark,
                "roman": roman_raw,
                "roman_cv": kr_cv_kernel_fn(roman_raw),
                "start_time": w_start,
                "end_time": w_end,
                "phones": s_phones,
            }
        )
    return syllables


def collect_blank_confidences(syllables_info: Sequence[dict[str, Any]]) -> tuple[list[float], float]:
    values = [
        float((syllable or {}).get("blank_confidence", 0.0) or 0.0)
        for syllable in (syllables_info or [])
    ]
    mean = float(sum(values) / float(len(values))) if values else 0.0
    return values, mean


def log_kr_syllable_source_choice(
    *,
    mapping_reason_code: str,
    fname: str,
    textgrid_trust_tier: str,
    textgrid_trust_score: float,
    low_quality_reasons: Sequence[str],
    base_score: float,
    alt_score: float,
    words_glide_mismatch_ratio: float,
    has_alias_based: bool,
    has_targets_for_build: bool,
    log_fn: Callable[[str], None],
) -> None:
    if mapping_reason_code == "filename_sequence_lock":
        log_fn(
            f"[MAP] {fname}: TextGrid 신뢰도 {str(textgrid_trust_tier).upper()} "
            f"(trust={float(textgrid_trust_score):.2f}) sequence lock kept"
        )
    elif mapping_reason_code == "alias_based_empty_words":
        log_fn(f"[MAP] {fname}: words 기반 phone 매핑 실패 -> alias/phone 기반으로 대체")
    elif mapping_reason_code == "alias_phone_minimal":
        log_fn(f"[MAP] {fname}: words 정보가 부족해 phones 기반으로 보강 매핑")
    elif mapping_reason_code in {
        "order_locked_length_mismatch",
        "order_locked_glide_mismatch",
        "order_locked_low_phone_quality",
    }:
        log_fn(
            f"[MAP] {fname}: CV 순서 잠금 유지 "
            f"(reason={mapping_reason_code}, words={float(base_score):.1f}, alias={float(alt_score):.1f}, "
            f"glide_mismatch={float(words_glide_mismatch_ratio):.2f})"
        )
    elif mapping_reason_code == "words_low_phone_quality":
        log_fn(
            f"[MAP] {fname}: phones 신뢰도 낮음({','.join(low_quality_reasons)}) -> words 보강 매핑 사용 "
            f"(words={float(base_score):.1f}, alias={float(alt_score):.1f})"
        )
    elif mapping_reason_code == "alias_based_cvvc":
        log_fn(
            f"[MAP] {fname}: CVVC에서 alias 기반 매핑 우선 "
            f"(words={float(base_score):.1f}, alias={float(alt_score):.1f})"
        )
    elif mapping_reason_code == "alias_based_recover":
        log_fn(
            f"[MAP] {fname}: 매핑 점수 보정 적용 "
            f"(base={float(base_score):.1f}, corrected={float(alt_score):.1f})"
        )
    elif mapping_reason_code == "words_keep_high_conf":
        log_fn(
            f"[MAP] {fname}: words 매핑 신뢰도 높음 -> words 결과 유지 "
            f"(base={float(base_score):.1f}, corrected={float(alt_score):.1f})"
        )
    elif has_alias_based and has_targets_for_build:
        log_fn(
            f"[MAP] {fname}: TextGrid(words) 기반 재매핑 "
            f"(base={float(base_score):.1f}, corrected={float(alt_score):.1f})"
        )


def prepare_kr_syllable_source_context(
    *,
    phone_intervals: Sequence[Any],
    word_intervals: Sequence[Any],
    targets_for_build: Sequence[Any],
    file_format: str,
    prefer_filename_sequence: bool,
    low_phone_quality: bool,
    force_words_phone_fill: bool,
    mel_ctx_for_file: dict[str, Any] | None,
    fname: str,
    textgrid_trust_tier: str,
    textgrid_trust_score: float,
    low_quality_reasons: Sequence[str],
    log_fn: Callable[[str], None],
    build_interval_lookup_fn: Callable[[Sequence[Any]], Any],
    intervals_within_bounds_fn: Callable[[Any, float, float], Sequence[Any]],
    synthesize_word_phones_fn: Callable[[Any, float, float, Callable[[str], Sequence[str]]], Sequence[Any]],
    decompose_hangul_to_roman_fn: Callable[[str], Sequence[str]],
    kr_cv_kernel_fn: Callable[[str], str],
    build_syllables_from_phone_nuclei_fn: Callable[[Sequence[Any], Sequence[Any]], Sequence[dict[str, Any]]],
    annotate_blank_confidence_fn: Callable[[Sequence[dict[str, Any]] | None, dict[str, Any]], None],
    select_syllable_source_fn: Callable[..., dict[str, Any]],
    score_mapping_fn: Callable[..., Any],
    should_prefer_alias_fn: Callable[..., Any],
    compute_glide_mismatch_fn: Callable[..., Any],
    is_order_locked_format_fn: Callable[[str], bool],
) -> Dict[str, Any]:
    words_syllables = build_kr_word_syllables(
        word_intervals=word_intervals,
        phone_intervals=phone_intervals,
        force_words_phone_fill=force_words_phone_fill,
        build_interval_lookup_fn=build_interval_lookup_fn,
        intervals_within_bounds_fn=intervals_within_bounds_fn,
        synthesize_word_phones_fn=synthesize_word_phones_fn,
        decompose_hangul_to_roman_fn=decompose_hangul_to_roman_fn,
        kr_cv_kernel_fn=kr_cv_kernel_fn,
    )
    used_words_based = len(words_syllables) > 0

    alias_source = (
        build_syllables_from_phone_nuclei_fn(phone_intervals, targets_for_build)
        if targets_for_build
        else None
    )
    alias_syllables = list(alias_source) if alias_source else None
    if mel_ctx_for_file:
        annotate_blank_confidence_fn(words_syllables, mel_ctx_for_file)
        annotate_blank_confidence_fn(alias_syllables, mel_ctx_for_file)

    source_pick = select_syllable_source_fn(
        file_format=file_format,
        prefer_filename_sequence=prefer_filename_sequence,
        low_phone_quality=low_phone_quality,
        used_words_based=used_words_based,
        words_syllables=words_syllables,
        alias_syllables=alias_syllables,
        targets_for_build=targets_for_build,
        score_mapping_fn=score_mapping_fn,
        should_prefer_alias_fn=should_prefer_alias_fn,
        compute_glide_mismatch_fn=compute_glide_mismatch_fn,
        is_order_locked_format_fn=is_order_locked_format_fn,
    )

    syllables_info = list(source_pick.get("syllables_info") or [])
    if mel_ctx_for_file:
        annotate_blank_confidence_fn(syllables_info, mel_ctx_for_file)

    blank_confidences, blank_conf_mean = collect_blank_confidences(syllables_info)
    mapping_reason_code = str(source_pick.get("mapping_reason_code") or "filename_token")
    base_score = float(source_pick.get("base_score", 0.0) or 0.0)
    alt_score = float(source_pick.get("alt_score", 0.0) or 0.0)
    words_glide_mismatch_ratio = float(source_pick.get("words_glide_mismatch_ratio", 0.0) or 0.0)

    log_kr_syllable_source_choice(
        mapping_reason_code=mapping_reason_code,
        fname=fname,
        textgrid_trust_tier=textgrid_trust_tier,
        textgrid_trust_score=textgrid_trust_score,
        low_quality_reasons=low_quality_reasons,
        base_score=base_score,
        alt_score=alt_score,
        words_glide_mismatch_ratio=words_glide_mismatch_ratio,
        has_alias_based=bool(alias_syllables),
        has_targets_for_build=bool(targets_for_build),
        log_fn=log_fn,
    )

    return {
        "syllables_info": syllables_info,
        "words_syllables": words_syllables,
        "alias_syllables": alias_syllables,
        "used_words_based": bool(source_pick.get("used_words_based")),
        "used_alias_based": bool(source_pick.get("used_alias_based")),
        "base_score": base_score,
        "alt_score": alt_score,
        "mapping_reason_code": mapping_reason_code,
        "words_glide_mismatch_ratio": words_glide_mismatch_ratio,
        "syllable_blank_confidences": blank_confidences,
        "blank_conf_mean": blank_conf_mean,
    }


__all__ = [
    "build_kr_word_syllables",
    "collect_blank_confidences",
    "log_kr_syllable_source_choice",
    "prepare_kr_syllable_source_context",
]
