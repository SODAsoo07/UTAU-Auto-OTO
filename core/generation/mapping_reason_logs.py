from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .mapping_reason_codes import (
    COMMON_REASON_ALIAS_PHONE_FALLBACK,
    COMMON_REASON_ALIAS_RECOVER,
    COMMON_REASON_FILENAME_TOKEN,
    COMMON_REASON_PHONES_SPLIT_FALLBACK,
    JA_REASON_ALIAS_CVVC_LENGTH_MISMATCH,
    JA_REASON_ALIAS_CVVC_LOW_PHONE_QUALITY,
    JA_REASON_ALIAS_CVVC_YOUON_MISMATCH,
    JA_REASON_ALIAS_WORDS_FALLBACK,
    JA_REASON_FILENAME_LINEAR_FALLBACK,
    JA_REASON_FILENAME_WORDS_LINEAR_LOCK,
    JA_REASON_FILENAME_WORDS_LOCK,
    JA_REASON_WORDS_SPARSE_SYNTH,
    JA_REASON_WORDS_TIER,
    KR_REASON_ALIAS_BASED_CVVC,
    KR_REASON_ALIAS_BASED_EMPTY_WORDS,
    KR_REASON_ALIAS_BASED_RECOVER,
    KR_REASON_ALIAS_PHONE_MINIMAL,
    KR_REASON_FILENAME_SEQUENCE_LOCK,
    KR_REASON_ORDER_LOCKED_GLIDE_MISMATCH,
    KR_REASON_ORDER_LOCKED_LENGTH_MISMATCH,
    KR_REASON_ORDER_LOCKED_LOW_PHONE_QUALITY,
    KR_REASON_WORDS_KEEP_HIGH_CONF,
    KR_REASON_WORDS_LOW_PHONE_QUALITY,
    normalize_mapping_reason_code,
)


def build_kr_mapping_reason_log(
    *,
    mapping_reason_code: str,
    base_score: float,
    alt_score: float,
    words_glide_mismatch_ratio: float,
    low_quality_reasons: Iterable[str],
    alias_syllables_present: bool,
    targets_for_build: Sequence[str],
    textgrid_trust_tier: str,
    textgrid_trust_score: float,
) -> Optional[str]:
    reason = normalize_mapping_reason_code(
        mapping_reason_code,
        language="kr",
        fallback_code=COMMON_REASON_FILENAME_TOKEN,
    )
    if reason == KR_REASON_FILENAME_SEQUENCE_LOCK:
        return (
            f"TextGrid 신뢰도 {str(textgrid_trust_tier or '').upper()} "
            f"(trust={float(textgrid_trust_score or 0.0):.2f}) → 파일명 순서 기반 매핑 고정"
        )
    if reason == KR_REASON_ALIAS_BASED_EMPTY_WORDS:
        return "words 매핑에 빈 phone 구간 존재 → alias/phone 기반 음절 매핑 사용"
    if reason == KR_REASON_ALIAS_PHONE_MINIMAL:
        return "words 티어 없음/실패 → phones 핵 기반 음절 매핑 사용"
    if reason in {
        KR_REASON_ORDER_LOCKED_LENGTH_MISMATCH,
        KR_REASON_ORDER_LOCKED_GLIDE_MISMATCH,
        KR_REASON_ORDER_LOCKED_LOW_PHONE_QUALITY,
    }:
        return (
            "CV 계열 순서 고정 포맷 보정 "
            f"(reason={reason}, words={float(base_score or 0.0):.1f}, "
            f"alias={float(alt_score or 0.0):.1f}, "
            f"glide_mismatch={float(words_glide_mismatch_ratio or 0.0):.2f})"
        )
    if reason == KR_REASON_WORDS_LOW_PHONE_QUALITY:
        quality = ",".join(str(x) for x in (low_quality_reasons or []))
        return (
            f"phones 저신뢰({quality}) → words 매핑 고정 "
            f"(words={float(base_score or 0.0):.1f}, alias={float(alt_score or 0.0):.1f})"
        )
    if reason == KR_REASON_ALIAS_BASED_CVVC:
        return (
            "CVVC는 alias 기반 음절 매핑 우선 "
            f"(words={float(base_score or 0.0):.1f}, alias={float(alt_score or 0.0):.1f})"
        )
    if reason == KR_REASON_ALIAS_BASED_RECOVER:
        return (
            "매핑 이탈 보정 적용 "
            f"(base={float(base_score or 0.0):.1f}, corrected={float(alt_score or 0.0):.1f})"
        )
    if reason == KR_REASON_WORDS_KEEP_HIGH_CONF:
        return (
            "words 매핑 신뢰도 높음 → alias 보정 생략 "
            f"(base={float(base_score or 0.0):.1f}, corrected={float(alt_score or 0.0):.1f})"
        )
    if bool(alias_syllables_present) and bool(targets_for_build):
        return (
            "TextGrid(words) 매핑 유지 "
            f"(base={float(base_score or 0.0):.1f}, corrected={float(alt_score or 0.0):.1f})"
        )
    return None


def build_ja_mapping_reason_log(
    *,
    selected_candidate,
    mapping_reason_code: str,
    filename_syllable_count: int,
    cv_target_count: int,
    phone_interval_count: int,
    word_interval_count: int,
    spn_ratio: float,
    base_score: float,
    alt_score: float,
) -> Optional[str]:
    if not selected_candidate:
        return None
    reason = normalize_mapping_reason_code(
        mapping_reason_code,
        language="ja",
        fallback_code=COMMON_REASON_FILENAME_TOKEN,
    )
    if reason in {JA_REASON_FILENAME_WORDS_LOCK, JA_REASON_FILENAME_WORDS_LINEAR_LOCK}:
        return (
            "words 합성 phone 기반 매핑 고정 "
            f"({int(filename_syllable_count or 0)}음절, spn_ratio={float(spn_ratio or 0.0):.2f})"
        )
    if reason == JA_REASON_ALIAS_WORDS_FALLBACK:
        return (
            "words 합성 phone 기반 filename 실패 → alias/phone 매핑 사용 "
            f"({int(cv_target_count or 0)}음절)"
        )
    if reason == COMMON_REASON_FILENAME_TOKEN:
        return f"파일명 우선 음절 매핑 사용 ({int(filename_syllable_count or 0)}음절)"
    if reason == JA_REASON_FILENAME_LINEAR_FALLBACK:
        return f"파일명 선형 fallback 음절 매핑 사용 ({int(filename_syllable_count or 0)}음절)"
    if reason == COMMON_REASON_ALIAS_PHONE_FALLBACK:
        return f"파일명 매핑 실패 → alias/phone 기반 음절 매핑 사용 ({int(cv_target_count or 0)}음절)"
    if reason == JA_REASON_WORDS_SPARSE_SYNTH:
        return f"phones 희소 감지({int(phone_interval_count or 0)}개) → words 기반 합성 phone 매핑 사용"
    if reason == JA_REASON_WORDS_TIER:
        return f"words 티어 기반 음절 매핑 사용 ({int(word_interval_count or 0)}구간)"
    if reason == COMMON_REASON_PHONES_SPLIT_FALLBACK:
        return "phones split fallback 음절 매핑 사용"
    if reason == COMMON_REASON_ALIAS_RECOVER:
        return (
            "매핑 이탈 보정 적용 "
            f"(base={float(base_score or 0.0):.1f}, corrected={float(alt_score or 0.0):.1f})"
        )
    if reason in {
        JA_REASON_ALIAS_CVVC_LENGTH_MISMATCH,
        JA_REASON_ALIAS_CVVC_YOUON_MISMATCH,
        JA_REASON_ALIAS_CVVC_LOW_PHONE_QUALITY,
    }:
        return (
            "CVVC 보수 가드로 alias/phone 매핑 전환 "
            f"(base={float(base_score or 0.0):.1f}, corrected={float(alt_score or 0.0):.1f}, reason={reason})"
        )
    return None
