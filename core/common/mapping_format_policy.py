from __future__ import annotations


def normalize_format(value) -> str:
    return str(value or "").strip().lower()


JA_SEQUENCE_LOCK_FORMATS = frozenset({"cvvc", "cv", "vcv"})
KR_SEQUENCE_LOCK_FORMATS = frozenset({"cvvc", "cvc", "cv", "vcv"})

# Sequence lock fallback shares the same format scope as filename-order preference.
JA_FILENAME_SEQUENCE_FORMATS = JA_SEQUENCE_LOCK_FORMATS
KR_FILENAME_SEQUENCE_FORMATS = KR_SEQUENCE_LOCK_FORMATS

# Row-level abstain should follow the same language format policy.
JA_ROW_GUARD_ACTIVE_FORMATS = JA_SEQUENCE_LOCK_FORMATS
JA_ROW_GUARD_MARGIN_FORMATS = JA_SEQUENCE_LOCK_FORMATS
KR_ROW_GUARD_ACTIVE_FORMATS = frozenset({"cvvc", "cv", "vcv"})
KR_ROW_GUARD_MARGIN_FORMATS = frozenset({"cvvc", "cv", "vcv"})
KR_ROW_GUARD_BLANK_FORMATS = KR_SEQUENCE_LOCK_FORMATS

# Some KR timing guards stay stricter than mapping sequence lock.
KR_ORDER_LOCKED_TIMING_FORMATS = frozenset({"cvvc", "cvc"})

JA_RUNTIME_ABSTAIN_FORMATS = JA_SEQUENCE_LOCK_FORMATS
KR_RUNTIME_ABSTAIN_FORMATS = KR_SEQUENCE_LOCK_FORMATS
JA_RUNTIME_STRICT_FORMATS = frozenset({"cvvc"})
KR_RUNTIME_STRICT_FORMATS = frozenset({"cvvc"})

ORDER_PRESERVING_CANDIDATE_FORMATS = JA_SEQUENCE_LOCK_FORMATS


def is_ja_sequence_locked_format(file_format) -> bool:
    return normalize_format(file_format) in JA_SEQUENCE_LOCK_FORMATS


def is_kr_sequence_locked_format(file_format) -> bool:
    return normalize_format(file_format) in KR_SEQUENCE_LOCK_FORMATS


def is_ja_filename_sequence_format(file_format) -> bool:
    return normalize_format(file_format) in JA_FILENAME_SEQUENCE_FORMATS


def is_kr_filename_sequence_format(file_format) -> bool:
    return normalize_format(file_format) in KR_FILENAME_SEQUENCE_FORMATS


def is_kr_order_locked_timing_format(file_format) -> bool:
    return normalize_format(file_format) in KR_ORDER_LOCKED_TIMING_FORMATS


__all__ = [
    "JA_FILENAME_SEQUENCE_FORMATS",
    "JA_ROW_GUARD_ACTIVE_FORMATS",
    "JA_ROW_GUARD_MARGIN_FORMATS",
    "JA_RUNTIME_ABSTAIN_FORMATS",
    "JA_RUNTIME_STRICT_FORMATS",
    "JA_SEQUENCE_LOCK_FORMATS",
    "KR_FILENAME_SEQUENCE_FORMATS",
    "KR_ORDER_LOCKED_TIMING_FORMATS",
    "KR_ROW_GUARD_ACTIVE_FORMATS",
    "KR_ROW_GUARD_BLANK_FORMATS",
    "KR_ROW_GUARD_MARGIN_FORMATS",
    "KR_RUNTIME_ABSTAIN_FORMATS",
    "KR_RUNTIME_STRICT_FORMATS",
    "KR_SEQUENCE_LOCK_FORMATS",
    "ORDER_PRESERVING_CANDIDATE_FORMATS",
    "is_ja_filename_sequence_format",
    "is_ja_sequence_locked_format",
    "is_kr_filename_sequence_format",
    "is_kr_order_locked_timing_format",
    "is_kr_sequence_locked_format",
    "normalize_format",
]
