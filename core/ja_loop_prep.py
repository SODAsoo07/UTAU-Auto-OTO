from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class JaLoopPrepResult:
    status: str = "ok"
    meta: dict[str, object] = field(default_factory=dict)
    wd_intervals: list = field(default_factory=list)
    ph_intervals_raw: list = field(default_factory=list)
    ph_intervals: list = field(default_factory=list)
    words_synth_phones: list = field(default_factory=list)
    filename_syllables: list[str] = field(default_factory=list)
    is_vowel_chain: bool = False
    cv_targets: list = field(default_factory=list)
    format_type: str = ""
    ja_style_profile: dict | None = None
    expected_syllables: int = 0
    phone_quality: dict[str, object] = field(default_factory=dict)
    low_quality_reasons: list[str] = field(default_factory=list)
    low_phone_quality: bool = False
    forced_words_mapping: bool = False
    timeline_start_ms: float = 0.0
    timeline_end_ms: float = 0.0
    effective_end_ms: float = 0.0
    boundary_points_ms: list[float] = field(default_factory=list)
    phone_spans_ms: list[tuple[float, float]] = field(default_factory=list)
    conf_th: float = 0.0


def prepare_ja_loop_state(
    *,
    fname: str,
    lines: Sequence[str],
    real_wav_name: str,
    ph_tier,
    word_tier,
    custom_map,
    forced_format,
    ja_mapping_words_fallback_enabled: bool,
    ja_mapping_spn_ratio_threshold: float,
    ja_mapping_min_vowel_phone_ratio: float,
    ja_mapping_confidence_threshold,
    debug_reason_logging: bool,
    log_fn: Callable[[str], None],
    parse_filename_fn: Callable[[str], list[str]],
    is_vowel_chain_fn: Callable[[list[str]], bool],
    extract_cv_targets_fn: Callable[[Sequence[str], object], list],
    detect_alias_format_fn: Callable[[list[str], object], str],
    get_profile_fn: Callable[[str], dict | None],
    collect_phone_quality_fn: Callable[..., dict],
    build_words_synth_phones_fn: Callable[[list, list[str]], list],
    resolve_conf_threshold_fn: Callable[..., float],
) -> JaLoopPrepResult:
    result = JaLoopPrepResult()
    silence_marks = {"", "sil", "spn", "pau", "sp"}
    phone_silence_marks = {"", "sil", "pau", "sp"}
    result.wd_intervals = [i for i in word_tier if i.mark.strip().lower() not in silence_marks] if word_tier else []
    result.ph_intervals_raw = [i for i in ph_tier if i.mark.strip().lower() not in phone_silence_marks]
    result.ph_intervals = [i for i in result.ph_intervals_raw if i.mark.strip().lower() != "spn"]
    result.words_synth_phones = []

    base_name = os.path.splitext(real_wav_name)[0]
    result.filename_syllables = list(parse_filename_fn(base_name) or [])
    result.is_vowel_chain = bool(is_vowel_chain_fn(result.filename_syllables))
    result.cv_targets = list(extract_cv_targets_fn(lines, custom_map) or [])

    alias_names = []
    for line in lines:
        if "=" not in line:
            continue
        alias = line.split("=", 1)[1].split(",", 1)[0].strip()
        if alias:
            alias_names.append(alias)
    if not alias_names:
        result.status = "no_valid_alias"
        return result

    detected_format = detect_alias_format_fn(alias_names, custom_map)
    result.format_type = forced_format or detected_format
    result.ja_style_profile = get_profile_fn(result.format_type)
    if result.is_vowel_chain:
        prev_format = result.format_type
        result.format_type = "vcv"
        result.ja_style_profile = get_profile_fn(result.format_type)
        log_fn(f"🎵 {fname}: 모음 연속음 파일 감지 → VCV 강제 적용 (기존: {prev_format.upper()})")
    elif forced_format:
        log_fn(f"🎵 {fname}: 포맷 수동 지정 → {result.format_type.upper()} (자동 감지: {detected_format.upper()})")
    else:
        log_fn(f"🎵 {fname}: 포맷 감지 → {result.format_type.upper()}")

    result.expected_syllables = max(len(result.filename_syllables), len(result.cv_targets), len(result.wd_intervals))
    result.phone_quality = dict(
        collect_phone_quality_fn(
            ph_tier,
            expected_syllables=result.expected_syllables,
            min_vowel_phone_ratio=ja_mapping_min_vowel_phone_ratio,
        )
        or {}
    )
    result.low_quality_reasons = list(result.phone_quality.get("low_confidence_reasons", []) or [])
    spn_ratio = float(result.phone_quality.get("spn_ratio_in_phone_tier", 0.0))
    if spn_ratio >= float(ja_mapping_spn_ratio_threshold):
        result.low_quality_reasons.append("spn_heavy")
    result.low_quality_reasons = sorted(set(result.low_quality_reasons))
    result.low_phone_quality = bool(result.low_quality_reasons)

    if (
        result.format_type in {"vcv", "cvvc", "cv"}
        and ja_mapping_words_fallback_enabled
        and result.low_phone_quality
    ):
        if result.wd_intervals:
            result.words_synth_phones = list(build_words_synth_phones_fn(result.wd_intervals, result.filename_syllables) or [])
            if result.words_synth_phones:
                result.ph_intervals = result.words_synth_phones
                result.forced_words_mapping = True
                if debug_reason_logging:
                    log_fn(
                        f"🧭 {fname}: phones 신뢰도 낮음({','.join(result.low_quality_reasons)}) "
                        f"→ words 기반 합성 phone 우선 사용"
                    )
        elif debug_reason_logging:
            log_fn(f"⚠️ {fname}: phones 신뢰도 낮음({','.join(result.low_quality_reasons)}), words 티어 없음")

    if not result.ph_intervals and result.wd_intervals:
        result.words_synth_phones = list(build_words_synth_phones_fn(result.wd_intervals, result.filename_syllables) or [])
        if result.words_synth_phones:
            result.ph_intervals = result.words_synth_phones

    if not result.ph_intervals:
        result.status = "empty_intervals"
        result.meta = {
            "diag_hint": f"spn_ratio={spn_ratio:.2f}",
            "phone_quality": result.phone_quality,
            "forced_words_mapping": result.forced_words_mapping,
        }
        return result

    result.timeline_start_ms = float(result.ph_intervals[0].minTime * 1000.0)
    result.timeline_end_ms = float(result.ph_intervals[-1].maxTime * 1000.0)
    if result.wd_intervals:
        result.timeline_start_ms = min(result.timeline_start_ms, float(result.wd_intervals[0].minTime * 1000.0))
        result.timeline_end_ms = max(result.timeline_end_ms, float(result.wd_intervals[-1].maxTime * 1000.0))

    result.effective_end_ms = result.timeline_end_ms
    if len(result.ph_intervals) >= 2:
        prev_p = result.ph_intervals[-2]
        last_p = result.ph_intervals[-1]
        gap_ms = float((last_p.minTime - prev_p.maxTime) * 1000.0)
        last_len_ms = float((last_p.maxTime - last_p.minTime) * 1000.0)
        if gap_ms > 450.0 and last_len_ms < 100.0:
            result.effective_end_ms = float(prev_p.maxTime * 1000.0)

    boundary_points_ms = set()
    for phone in result.ph_intervals:
        boundary_points_ms.add(float(phone.minTime * 1000.0))
        boundary_points_ms.add(float(phone.maxTime * 1000.0))
    if len(boundary_points_ms) < 4 and result.wd_intervals:
        for word in result.wd_intervals:
            boundary_points_ms.add(float(word.minTime * 1000.0))
            boundary_points_ms.add(float(word.maxTime * 1000.0))
    result.boundary_points_ms = sorted(boundary_points_ms)
    result.phone_spans_ms = [
        (float(phone.minTime * 1000.0), float(phone.maxTime * 1000.0))
        for phone in result.ph_intervals
    ]
    result.conf_th = float(
        resolve_conf_threshold_fn(
            result.format_type,
            override_threshold=ja_mapping_confidence_threshold,
        )
    )
    return result


__all__ = [
    "JaLoopPrepResult",
    "prepare_ja_loop_state",
]
