from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Set

# Common reason codes shared by KR/JA paths.
COMMON_REASON_FILENAME_TOKEN = "filename_token"
COMMON_REASON_ALIAS_PHONE_FALLBACK = "alias_phone_fallback"
COMMON_REASON_ALIAS_RECOVER = "alias_recover"
COMMON_REASON_PHONES_SPLIT_FALLBACK = "phones_split_fallback"

# KR-specific reason codes.
KR_REASON_WORDS_KEEP = "words_keep"
KR_REASON_FILENAME_SEQUENCE_LOCK = "filename_sequence_lock"
KR_REASON_ALIAS_BASED_EMPTY_WORDS = "alias_based_empty_words"
KR_REASON_ALIAS_PHONE_MINIMAL = "alias_phone_minimal"
KR_REASON_ORDER_LOCKED_LENGTH_MISMATCH = "order_locked_length_mismatch"
KR_REASON_ORDER_LOCKED_GLIDE_MISMATCH = "order_locked_glide_mismatch"
KR_REASON_ORDER_LOCKED_LOW_PHONE_QUALITY = "order_locked_low_phone_quality"
KR_REASON_WORDS_LOW_PHONE_QUALITY = "words_low_phone_quality"
KR_REASON_ALIAS_BASED_CVVC = "alias_based_cvvc"
KR_REASON_ALIAS_BASED_RECOVER = "alias_based_recover"
KR_REASON_WORDS_KEEP_HIGH_CONF = "words_keep_high_conf"

# JA-specific reason codes.
JA_REASON_FILENAME_WORDS_LOCK = "filename_words_lock"
JA_REASON_FILENAME_WORDS_LINEAR_LOCK = "filename_words_linear_lock"
JA_REASON_ALIAS_WORDS_FALLBACK = "alias_words_fallback"
JA_REASON_WORDS_SPARSE_SYNTH = "words_sparse_synth"
JA_REASON_WORDS_TIER = "words_tier"
JA_REASON_ALIAS_CVVC_LENGTH_MISMATCH = "alias_cvvc_length_mismatch"
JA_REASON_ALIAS_CVVC_YOUON_MISMATCH = "alias_cvvc_youon_mismatch"
JA_REASON_ALIAS_CVVC_LOW_PHONE_QUALITY = "alias_cvvc_low_phone_quality"
JA_REASON_FILENAME_LINEAR_FALLBACK = "filename_linear_fallback"
JA_REASON_FILENAME_LINEAR_LOW_CONF = "filename_linear_low_conf"
JA_REASON_FILENAME_LOW_CONF = "filename_low_conf"


COMMON_MAPPING_REASON_CODES = frozenset(
    {
        COMMON_REASON_FILENAME_TOKEN,
        COMMON_REASON_ALIAS_PHONE_FALLBACK,
        COMMON_REASON_ALIAS_RECOVER,
        COMMON_REASON_PHONES_SPLIT_FALLBACK,
    }
)

KR_MAPPING_REASON_CODES = frozenset(
    {
        KR_REASON_WORDS_KEEP,
        KR_REASON_FILENAME_SEQUENCE_LOCK,
        KR_REASON_ALIAS_BASED_EMPTY_WORDS,
        KR_REASON_ALIAS_PHONE_MINIMAL,
        KR_REASON_ORDER_LOCKED_LENGTH_MISMATCH,
        KR_REASON_ORDER_LOCKED_GLIDE_MISMATCH,
        KR_REASON_ORDER_LOCKED_LOW_PHONE_QUALITY,
        KR_REASON_WORDS_LOW_PHONE_QUALITY,
        KR_REASON_ALIAS_BASED_CVVC,
        KR_REASON_ALIAS_BASED_RECOVER,
        KR_REASON_WORDS_KEEP_HIGH_CONF,
    }
)

JA_MAPPING_REASON_CODES = frozenset(
    {
        JA_REASON_FILENAME_WORDS_LOCK,
        JA_REASON_FILENAME_WORDS_LINEAR_LOCK,
        JA_REASON_ALIAS_WORDS_FALLBACK,
        JA_REASON_WORDS_SPARSE_SYNTH,
        JA_REASON_WORDS_TIER,
        JA_REASON_ALIAS_CVVC_LENGTH_MISMATCH,
        JA_REASON_ALIAS_CVVC_YOUON_MISMATCH,
        JA_REASON_ALIAS_CVVC_LOW_PHONE_QUALITY,
        JA_REASON_FILENAME_LINEAR_FALLBACK,
        JA_REASON_FILENAME_LINEAR_LOW_CONF,
        JA_REASON_FILENAME_LOW_CONF,
    }
)

_REASON_SCHEMA: Dict[str, Dict[str, str]] = {
    COMMON_REASON_FILENAME_TOKEN: {
        "stage": "candidate_select",
        "cause": "filename_token_alignment",
        "recovery": "use_filename_tokens",
    },
    COMMON_REASON_ALIAS_PHONE_FALLBACK: {
        "stage": "candidate_select",
        "cause": "filename_mapping_unstable",
        "recovery": "fallback_alias_phone",
    },
    COMMON_REASON_ALIAS_RECOVER: {
        "stage": "candidate_select",
        "cause": "alias_candidate_superior",
        "recovery": "promote_alias_candidate",
    },
    COMMON_REASON_PHONES_SPLIT_FALLBACK: {
        "stage": "candidate_select",
        "cause": "words_tier_missing",
        "recovery": "split_phone_sequence",
    },
    KR_REASON_WORDS_KEEP: {
        "stage": "source_select",
        "cause": "words_mapping_usable",
        "recovery": "keep_words_mapping",
    },
    KR_REASON_FILENAME_SEQUENCE_LOCK: {
        "stage": "source_select",
        "cause": "prefer_filename_sequence",
        "recovery": "lock_filename_order",
    },
    KR_REASON_ALIAS_BASED_EMPTY_WORDS: {
        "stage": "source_select",
        "cause": "empty_word_phone_span",
        "recovery": "switch_alias_mapping",
    },
    KR_REASON_ALIAS_PHONE_MINIMAL: {
        "stage": "source_select",
        "cause": "words_tier_unavailable",
        "recovery": "minimal_alias_phone_split",
    },
    KR_REASON_ORDER_LOCKED_LENGTH_MISMATCH: {
        "stage": "source_select",
        "cause": "order_locked_length_mismatch",
        "recovery": "enforce_alias_ordered_path",
    },
    KR_REASON_ORDER_LOCKED_GLIDE_MISMATCH: {
        "stage": "source_select",
        "cause": "order_locked_glide_mismatch",
        "recovery": "enforce_alias_ordered_path",
    },
    KR_REASON_ORDER_LOCKED_LOW_PHONE_QUALITY: {
        "stage": "source_select",
        "cause": "order_locked_low_phone_quality",
        "recovery": "enforce_alias_ordered_path",
    },
    KR_REASON_WORDS_LOW_PHONE_QUALITY: {
        "stage": "source_select",
        "cause": "low_phone_quality",
        "recovery": "keep_words_to_stabilize",
    },
    KR_REASON_ALIAS_BASED_CVVC: {
        "stage": "source_select",
        "cause": "cvvc_alias_preference",
        "recovery": "prefer_alias_cvvc",
    },
    KR_REASON_ALIAS_BASED_RECOVER: {
        "stage": "source_select",
        "cause": "alias_score_advantage",
        "recovery": "switch_alias_mapping",
    },
    KR_REASON_WORDS_KEEP_HIGH_CONF: {
        "stage": "source_select",
        "cause": "high_confidence_words_path",
        "recovery": "keep_words_mapping",
    },
    JA_REASON_FILENAME_WORDS_LOCK: {
        "stage": "candidate_select",
        "cause": "forced_words_mapping_filename_lock",
        "recovery": "lock_filename_candidate",
    },
    JA_REASON_FILENAME_WORDS_LINEAR_LOCK: {
        "stage": "candidate_select",
        "cause": "forced_words_mapping_linear_lock",
        "recovery": "lock_filename_linear_candidate",
    },
    JA_REASON_ALIAS_WORDS_FALLBACK: {
        "stage": "candidate_select",
        "cause": "forced_words_mapping_filename_unavailable",
        "recovery": "fallback_alias_candidate",
    },
    JA_REASON_WORDS_SPARSE_SYNTH: {
        "stage": "candidate_select",
        "cause": "sparse_phone_sequence",
        "recovery": "synthesize_words_from_phones",
    },
    JA_REASON_WORDS_TIER: {
        "stage": "candidate_select",
        "cause": "words_tier_available",
        "recovery": "prefer_words_tier",
    },
    JA_REASON_ALIAS_CVVC_LENGTH_MISMATCH: {
        "stage": "candidate_select",
        "cause": "cvvc_length_mismatch",
        "recovery": "switch_alias_cvvc_guard",
    },
    JA_REASON_ALIAS_CVVC_YOUON_MISMATCH: {
        "stage": "candidate_select",
        "cause": "cvvc_youon_mismatch",
        "recovery": "switch_alias_cvvc_guard",
    },
    JA_REASON_ALIAS_CVVC_LOW_PHONE_QUALITY: {
        "stage": "candidate_select",
        "cause": "cvvc_low_phone_quality",
        "recovery": "switch_alias_cvvc_guard",
    },
    JA_REASON_FILENAME_LINEAR_FALLBACK: {
        "stage": "candidate_select",
        "cause": "filename_token_match_weak",
        "recovery": "use_filename_linear_fallback",
    },
    JA_REASON_FILENAME_LINEAR_LOW_CONF: {
        "stage": "runtime_lock",
        "cause": "low_confidence_force_sequence_lock",
        "recovery": "relock_filename_linear",
    },
    JA_REASON_FILENAME_LOW_CONF: {
        "stage": "runtime_lock",
        "cause": "low_confidence_force_sequence_lock",
        "recovery": "relock_filename_token",
    },
}


def _normalize_language(language: Optional[str]) -> str:
    token = str(language or "").strip().lower()
    if token in {"kr", "ko", "korean"}:
        return "kr"
    if token in {"ja", "jp", "japanese"}:
        return "ja"
    return "all"


def canonicalize_mapping_reason_code(reason_code: Optional[str]) -> str:
    return str(reason_code or "").strip().lower()


def get_known_mapping_reason_codes(language: Optional[str] = None) -> Set[str]:
    lang = _normalize_language(language)
    if lang == "kr":
        return set(COMMON_MAPPING_REASON_CODES | KR_MAPPING_REASON_CODES)
    if lang == "ja":
        return set(COMMON_MAPPING_REASON_CODES | JA_MAPPING_REASON_CODES)
    return set(COMMON_MAPPING_REASON_CODES | KR_MAPPING_REASON_CODES | JA_MAPPING_REASON_CODES)


def is_known_mapping_reason_code(reason_code: Optional[str], language: Optional[str] = None) -> bool:
    code = canonicalize_mapping_reason_code(reason_code)
    return bool(code) and code in get_known_mapping_reason_codes(language)


def normalize_mapping_reason_code(
    reason_code: Optional[str],
    *,
    language: Optional[str] = None,
    fallback_code: Optional[str] = None,
) -> str:
    code = canonicalize_mapping_reason_code(reason_code)
    if code:
        return code
    if fallback_code:
        fallback = canonicalize_mapping_reason_code(fallback_code)
        if fallback and is_known_mapping_reason_code(fallback, language):
            return fallback
    return "unknown"


def resolve_mapping_reason_schema(
    reason_code: Optional[str],
    *,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    code = canonicalize_mapping_reason_code(reason_code)
    lang = _normalize_language(language)
    known = bool(code) and code in get_known_mapping_reason_codes(lang)
    meta: Mapping[str, str] = _REASON_SCHEMA.get(code, {})
    return {
        "code": code or "unknown",
        "language": lang,
        "known": bool(known),
        "stage": str(meta.get("stage") or "unknown"),
        "cause": str(meta.get("cause") or "unknown"),
        "recovery": str(meta.get("recovery") or "inspect_runtime_log"),
    }
