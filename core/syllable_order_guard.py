from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "n"}

_SUPPORTED_FORMATS = {
    "korean": {"cv", "cvc", "coc", "cvvc", "vcv"},
    "japanese": {"cv", "cvvc", "vcv"},
}


@dataclass(frozen=True)
class SyllableOrderGuardDecision:
    enabled: bool
    applied: bool
    reason: str
    language: str
    format_type: str
    expected_count: int


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return bool(default)


def _normalize_language(language: str) -> str:
    raw = str(language or "").strip().lower()
    if raw.startswith("ko") or "korean" in raw:
        return "korean"
    if raw.startswith("ja") or "japanese" in raw:
        return "japanese"
    return raw


def _normalize_format(format_type: str) -> str:
    fmt = str(format_type or "").strip().lower()
    if fmt == "coc":
        return "cvc"
    return fmt


def is_syllable_order_guard_enabled(language: str = "") -> bool:
    if _env_bool("UTOA_DISABLE_SYLLABLE_ORDER_GUARD", False):
        return False
    lang = _normalize_language(language)
    if lang == "korean":
        return not _env_bool("UTOA_DISABLE_KR_SYLLABLE_ORDER_GUARD", False)
    if lang == "japanese":
        return not _env_bool("UTOA_DISABLE_JA_SYLLABLE_ORDER_GUARD", False)
    return bool(lang)


def evaluate_syllable_order_guard(
    *,
    language: str,
    format_type: str,
    expected_tokens: Sequence[str] | None,
    alignment_weight: float = 0.0,
    low_phone_quality: bool = False,
    textgrid_trust_tier: str = "",
) -> SyllableOrderGuardDecision:
    lang = _normalize_language(language)
    fmt = _normalize_format(format_type)
    tokens = [str(x).strip() for x in (expected_tokens or []) if str(x).strip()]
    enabled = is_syllable_order_guard_enabled(lang)
    if not enabled:
        return SyllableOrderGuardDecision(
            enabled=False,
            applied=False,
            reason="disabled",
            language=lang,
            format_type=fmt,
            expected_count=len(tokens),
        )

    supported = _SUPPORTED_FORMATS.get(lang, set())
    if fmt not in supported:
        return SyllableOrderGuardDecision(
            enabled=True,
            applied=False,
            reason="unsupported_format",
            language=lang,
            format_type=fmt,
            expected_count=len(tokens),
        )
    if len(tokens) < 2:
        return SyllableOrderGuardDecision(
            enabled=True,
            applied=False,
            reason="insufficient_expected_tokens",
            language=lang,
            format_type=fmt,
            expected_count=len(tokens),
        )

    tier = str(textgrid_trust_tier or "").strip().lower()
    try:
        weight = float(alignment_weight or 0.0)
    except Exception:
        weight = 0.0
    reason = "sequence_guard"
    if tier == "low" or weight < 0.55:
        reason = "sequence_guard_low_trust"
    elif bool(low_phone_quality):
        reason = "sequence_guard_low_phone_quality"

    return SyllableOrderGuardDecision(
        enabled=True,
        applied=True,
        reason=reason,
        language=lang,
        format_type=fmt,
        expected_count=len(tokens),
    )


__all__ = [
    "SyllableOrderGuardDecision",
    "evaluate_syllable_order_guard",
    "is_syllable_order_guard_enabled",
]
