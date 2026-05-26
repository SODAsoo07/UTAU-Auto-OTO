from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.model_context.manual_oto_anchor import classify_manual_oto_alias_role


STYLE_TAIL_VC = "tail_vc"
STYLE_FOLLOWING_ONSET_VC = "following_onset_vc"
STYLE_MIXED = "mixed"
STYLE_UNKNOWN = "unknown"

KOREAN_CVVC_VOWELS = frozenset(
    {
        "a",
        "ae",
        "e",
        "eo",
        "eu",
        "i",
        "o",
        "u",
        "ui",
        "wa",
        "we",
        "wi",
        "wo",
        "ya",
        "ye",
        "yeo",
        "yo",
        "yu",
    }
)


@dataclass(frozen=True)
class ReclistConvention:
    vc_pre_style: str
    confidence: float
    role_counts: Mapping[str, int]
    evidence: Mapping[str, float | int | bool]
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, object]:
        return {
            "vc_pre_style": self.vc_pre_style,
            "confidence": float(self.confidence),
            "role_counts": dict(self.role_counts),
            "evidence": dict(self.evidence),
            "warnings": list(self.warnings),
        }


def classify_korean_cvvc_alias_role(
    alias: object,
    *,
    language: str,
    format_type: str,
    fallback: str | None = None,
) -> str:
    role = str(
        fallback
        if fallback is not None
        else classify_manual_oto_alias_role(alias, language=language, format_type=format_type)
    )
    language_text = str(language or "").strip().lower()
    format_text = str(format_type or "").strip().lower()
    if language_text not in {"ko", "kr", "korean"} or format_text != "cvvc":
        return role

    text = str(alias or "").strip()
    if not text:
        return role
    if text.startswith(("-", "'")):
        return "cv"
    if text.endswith(("-", "*")):
        return "v"

    terms = korean_cvvc_alias_terms(text)
    if len(terms) >= 2:
        left = terms[0]
        right = terms[-1]
        if is_korean_cvvc_vowel_term(left) and is_korean_cvvc_vowel_term(right):
            return "vv"
        return "vc"
    if len(terms) == 1:
        term = terms[0]
        if is_korean_cvvc_vowel_term(term):
            return "v"
        if contains_korean_cvvc_vowel(term):
            return "cv"
    return role


def analyze_reclist_convention(
    aliases: Sequence[object],
    *,
    roles: Sequence[str] | None = None,
    language: str,
    format_type: str,
) -> ReclistConvention:
    language_text = str(language or "").strip().lower()
    format_text = str(format_type or "").strip().lower()
    if language_text not in {"ko", "kr", "korean"} or format_text != "cvvc":
        return ReclistConvention(
            vc_pre_style=STYLE_UNKNOWN,
            confidence=0.0,
            role_counts={},
            evidence={"is_korean_cvvc": False, "total_rows": len(aliases)},
            warnings=("unsupported_format",),
        )

    if roles is None:
        resolved_roles = [
            classify_korean_cvvc_alias_role(alias, language=language, format_type=format_type)
            for alias in aliases
        ]
    else:
        resolved_roles = [str(role or "").strip().lower() for role in roles]
    count = len(resolved_roles)
    role_counts = Counter(resolved_roles)
    vc_indices = [idx for idx, role in enumerate(resolved_roles) if role == "vc"]
    voiced_indices = [idx for idx, role in enumerate(resolved_roles) if role in {"cv", "v", "vv"}]
    evidence: dict[str, float | int | bool] = {
        "is_korean_cvvc": True,
        "total_rows": count,
        "vc_rows": len(vc_indices),
        "voiced_rows": len(voiced_indices),
    }
    if not vc_indices:
        return ReclistConvention(
            vc_pre_style=STYLE_UNKNOWN,
            confidence=0.45,
            role_counts=dict(role_counts),
            evidence=evidence,
            warnings=("no_vc_rows",),
        )

    first_vc = min(vc_indices)
    suffix_len = max(1, count - first_vc)
    suffix_vc_count = sum(1 for idx in range(first_vc, count) if resolved_roles[idx] == "vc")
    suffix_vc_purity = float(suffix_vc_count) / float(suffix_len)
    leading_voiced_count = sum(1 for idx in range(0, first_vc) if resolved_roles[idx] in {"cv", "v", "vv"})
    early_limit = max(1, count // 2)
    early_vc_count = sum(1 for idx in vc_indices if idx < early_limit)
    coda_vc_count = sum(1 for idx in vc_indices if _vc_left_term_is_coda(aliases[idx]))
    coda_vc_ratio = float(coda_vc_count) / float(max(1, len(vc_indices)))
    before_vc_roles = resolved_roles[:first_vc]
    all_voiced_before_vc = bool(before_vc_roles) and all(role in {"cv", "v", "vv"} for role in before_vc_roles)
    evidence.update(
        {
            "first_vc_index": first_vc,
            "leading_voiced_rows": leading_voiced_count,
            "suffix_vc_purity": suffix_vc_purity,
            "early_vc_count": early_vc_count,
            "coda_vc_ratio": coda_vc_ratio,
            "all_voiced_before_vc": all_voiced_before_vc,
        }
    )

    following_score = 0.0
    if leading_voiced_count >= 3 and first_vc >= 3 and len(vc_indices) >= 3:
        following_score += 0.35
        if suffix_vc_purity >= 0.75:
            following_score += 0.30
        if all_voiced_before_vc:
            following_score += 0.20
        following_score += 0.10

    tail_score = 0.0
    if first_vc <= 2:
        tail_score += 0.35
    if early_vc_count >= max(1, min(3, len(vc_indices) // 3)):
        tail_score += 0.25
    if coda_vc_ratio >= 0.25:
        tail_score += 0.20
    if suffix_vc_purity < 0.70:
        tail_score += 0.10

    evidence["following_onset_score"] = following_score
    evidence["tail_score"] = tail_score

    warnings: list[str] = []
    if following_score >= 0.70 and following_score >= tail_score + 0.18:
        style = STYLE_FOLLOWING_ONSET_VC
        confidence = min(0.97, 0.55 + following_score - (tail_score * 0.35))
    elif tail_score >= 0.55 and tail_score >= following_score + 0.12:
        style = STYLE_TAIL_VC
        confidence = min(0.96, 0.55 + tail_score - (following_score * 0.35))
    elif max(following_score, tail_score) >= 0.45:
        style = STYLE_MIXED
        confidence = min(0.72, 0.45 + abs(following_score - tail_score))
        warnings.append("mixed_or_ambiguous_vc_convention")
    else:
        style = STYLE_UNKNOWN
        confidence = max(following_score, tail_score)
        warnings.append("low_vc_convention_evidence")
    if confidence < 0.65:
        warnings.append("low_vc_convention_confidence")

    return ReclistConvention(
        vc_pre_style=style,
        confidence=float(max(0.0, min(1.0, confidence))),
        role_counts=dict(role_counts),
        evidence=evidence,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def korean_cvvc_alias_terms(alias: object) -> list[str]:
    text = str(alias or "").strip().lower().split("_", 1)[0]
    parts = [part for part in text.replace("-", " - ").split() if part and part != "-"]
    out: list[str] = []
    for part in parts:
        term = normalize_korean_cvvc_token(part)
        if term:
            out.append(term)
    return out


def normalize_korean_cvvc_token(token: object) -> str:
    text = str(token or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def is_korean_cvvc_vowel_term(term: object) -> bool:
    return normalize_korean_cvvc_token(term) in KOREAN_CVVC_VOWELS


def contains_korean_cvvc_vowel(term: object) -> bool:
    text = normalize_korean_cvvc_token(term)
    return any(vowel in text for vowel in sorted(KOREAN_CVVC_VOWELS, key=len, reverse=True))


def _vc_left_term_is_coda(alias: object) -> bool:
    terms = korean_cvvc_alias_terms(alias)
    if not terms:
        return False
    return not is_korean_cvvc_vowel_term(terms[0]) and not contains_korean_cvvc_vowel(terms[0])
