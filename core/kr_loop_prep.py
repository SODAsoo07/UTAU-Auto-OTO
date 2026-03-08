from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class KrLoopPrepResult:
    status: str = "ok"
    meta: dict[str, object] = field(default_factory=dict)
    ph_intervals_raw: list = field(default_factory=list)
    ph_intervals_all: list = field(default_factory=list)
    ph_intervals: list = field(default_factory=list)
    wd_intervals: list = field(default_factory=list)
    alias_names: list[str] = field(default_factory=list)
    file_format: str = ""
    file_mapping_conf_th: float = 0.0
    is_vc_only: bool = False
    is_vcv_file: bool = False
    cv_targets: list = field(default_factory=list)
    filename_cv_targets: list = field(default_factory=list)
    targets_for_build: list = field(default_factory=list)
    expected_syllables: int = 0
    phone_quality: dict[str, object] = field(default_factory=dict)
    low_quality_reasons: list[str] = field(default_factory=list)
    low_phone_quality: bool = False
    force_words_phone_fill: bool = False
    timeline_start_ms: float = 0.0
    timeline_end_ms: float = 0.0
    wav_duration_ms: float = 0.0


def prepare_kr_loop_state(
    *,
    lines: Sequence[str],
    real_wav_name: str,
    phone_tier,
    word_tier,
    wav_duration_ms: float,
    custom_map,
    kr_mapping_words_fallback_enabled: bool,
    kr_mapping_spn_ratio_threshold: float,
    kr_mapping_min_vowel_phone_ratio: float,
    kr_mapping_confidence_threshold,
    debug_log_fn: Callable[[str], None],
    debug_reason_logging: bool,
    decompose_hangul_to_roman_fn: Callable[[str], list[str]],
    synthesize_word_phones_fn: Callable[[str, float, float, Callable[[str], list[str]]], list],
    detect_alias_format_fn: Callable[[list[str], object], str],
    extract_cv_targets_from_lines_fn: Callable[[Sequence[str], object], list],
    extract_cv_targets_from_filename_fn: Callable[[str], list],
    collect_phone_quality_fn: Callable[..., dict],
    resolve_mapping_conf_threshold_fn: Callable[..., float],
) -> KrLoopPrepResult:
    result = KrLoopPrepResult(wav_duration_ms=float(wav_duration_ms or 0.0))

    phone_silence_marks = {"", "sil", "pau", "sp"}
    result.ph_intervals_raw = [i for i in phone_tier if i.mark.strip().lower() not in phone_silence_marks]
    result.ph_intervals_all = [i for i in result.ph_intervals_raw if i.mark.strip().lower() not in {"spn"}]
    result.ph_intervals = [i for i in result.ph_intervals_all if i.mark.strip().lower() not in {"sil"}]
    result.wd_intervals = [i for i in word_tier if i.mark.strip() not in {"", "sil", "spn", "pau"}] if word_tier else []

    if len(result.ph_intervals) == 0:
        if kr_mapping_words_fallback_enabled and result.wd_intervals:
            fallback_phones = []
            for word in result.wd_intervals:
                fallback_phones.extend(
                    synthesize_word_phones_fn(
                        word.mark,
                        float(word.minTime),
                        float(word.maxTime),
                        decompose_hangul_to_roman_fn,
                    )
                )
            result.ph_intervals = fallback_phones
            result.ph_intervals_all = fallback_phones
        if len(result.ph_intervals) == 0:
            result.status = "empty_intervals"
            return result

    result.timeline_start_ms = float(result.ph_intervals[0].minTime * 1000.0)
    result.timeline_end_ms = float(result.ph_intervals[-1].maxTime * 1000.0)
    if result.wd_intervals:
        result.timeline_start_ms = min(result.timeline_start_ms, float(result.wd_intervals[0].minTime * 1000.0))
        result.timeline_end_ms = max(result.timeline_end_ms, float(result.wd_intervals[-1].maxTime * 1000.0))
    if result.wav_duration_ms <= 0.0:
        result.wav_duration_ms = result.timeline_end_ms

    result.alias_names = []
    for line in lines:
        if "=" not in line:
            continue
        alias = line.split("=", 1)[1].split(",", 1)[0].strip()
        if alias:
            result.alias_names.append(alias)
    if not result.alias_names:
        result.status = "no_valid_alias"
        return result

    result.file_format = detect_alias_format_fn(result.alias_names, custom_map)
    debug_log_fn(f"처리: {real_wav_name}: 형식 감지 -> {result.file_format.upper()}")
    result.file_mapping_conf_th = float(
        resolve_mapping_conf_threshold_fn(
            result.file_format,
            override_threshold=kr_mapping_confidence_threshold,
        )
    )
    result.is_vc_only = result.file_format == "vc_only"
    result.is_vcv_file = result.file_format == "vcv"
    result.cv_targets = list(extract_cv_targets_from_lines_fn(lines, custom_map) or [])
    result.filename_cv_targets = list(extract_cv_targets_from_filename_fn(real_wav_name) or [])
    result.targets_for_build = list(result.cv_targets)
    if result.file_format == "cvvc" and result.filename_cv_targets:
        result.targets_for_build = list(result.filename_cv_targets)
    elif (not result.targets_for_build) and result.filename_cv_targets:
        result.targets_for_build = list(result.filename_cv_targets)
    result.expected_syllables = max(len(result.wd_intervals), len(result.cv_targets), len(result.filename_cv_targets))
    result.phone_quality = dict(
        collect_phone_quality_fn(
            phone_tier,
            expected_syllables=result.expected_syllables,
            min_vowel_phone_ratio=kr_mapping_min_vowel_phone_ratio,
        )
        or {}
    )
    result.low_quality_reasons = list(result.phone_quality.get("low_confidence_reasons", []) or [])
    spn_ratio = float(result.phone_quality.get("spn_ratio_in_phone_tier", 0.0))
    if spn_ratio >= float(kr_mapping_spn_ratio_threshold):
        result.low_quality_reasons.append("spn_heavy")
    result.low_quality_reasons = sorted(set(result.low_quality_reasons))
    result.low_phone_quality = bool(result.low_quality_reasons)
    result.force_words_phone_fill = bool(
        kr_mapping_words_fallback_enabled and result.low_phone_quality and result.wd_intervals
    )
    if result.force_words_phone_fill and debug_reason_logging:
        debug_log_fn(
            f"🧭 {real_wav_name}: phones 신뢰도 낮음({','.join(result.low_quality_reasons)}) "
            f"→ words 구간 보강 매핑 사용"
        )
    return result


__all__ = [
    "KrLoopPrepResult",
    "prepare_kr_loop_state",
]
