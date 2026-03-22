from __future__ import annotations


def normalize_language_name(language: str) -> str:
    lang = str(language or "").strip().lower()
    if lang in {"japanese", "jp", "ja", "jpn"}:
        return "japanese"
    if lang in {"korean", "kr", "ko", "kor"}:
        return "korean"
    if lang in {"english", "en", "eng"}:
        return "english"
    return lang


def _normalize_token(text: str) -> str:
    token = str(text or "").strip().lower()
    token = token.replace("\\", "/")
    token = token.replace("-", "_")
    token = token.replace(" ", "")
    return token


def normalize_format_type(language: str, format_type: str) -> str:
    lang = normalize_language_name(language)
    raw = str(format_type or "").strip()
    fmt = _normalize_token(raw)
    if not fmt:
        return ""
    if fmt == "general":
        return "general"

    if lang == "korean":
        if fmt.startswith("cmpx"):
            return "cmpx"
        if fmt.startswith("cvvc"):
            return "cvvc"
        if fmt.startswith("vcv"):
            return "vcv"
        if fmt.startswith("cvc"):
            return "cvc"
        if fmt.startswith("cv") or fmt.startswith("mono"):
            return "cv"
        return fmt

    if lang == "japanese":
        if fmt.startswith("cvvc"):
            return "cvvc"
        if fmt.startswith("vcv"):
            return "vcv"
        if fmt.startswith("cvc") or fmt.startswith("cv") or fmt.startswith("mono"):
            return "cv"
        return fmt

    if lang == "english":
        if fmt.startswith("cvvc"):
            return "cvvc"
        if fmt.startswith("vcv"):
            return "vcv"
        if fmt.startswith("cvc") or fmt.startswith("cv") or fmt.startswith("mono"):
            return "cv"
        return fmt

    return fmt


def normalize_auto_format_value(language: str, auto_format: str) -> str:
    lang = normalize_language_name(language)
    raw = str(auto_format or "").strip()
    if not raw:
        return ""

    normalized = _normalize_token(raw)
    if normalized.startswith("auto") or "자동" in raw:
        return ""
    if normalized.startswith("cvvc"):
        return "cvvc"
    if normalized.startswith("vcv"):
        return "vcv"
    if lang == "korean" and normalized.startswith("cmpx"):
        return "cmpx"

    if lang == "korean":
        if normalized.startswith("cvc"):
            return "cvc"
        if normalized.startswith("cv") or normalized.startswith("mono"):
            return "cv"
    elif lang == "english":
        if normalized.startswith("cvc") or normalized.startswith("cv") or normalized.startswith("mono"):
            return "cv"
    else:
        if normalized.startswith("cvc") or normalized.startswith("cv") or normalized.startswith("mono"):
            return "cv"

    return normalize_format_type(lang, raw)


__all__ = ["normalize_auto_format_value", "normalize_format_type", "normalize_language_name"]
