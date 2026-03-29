from __future__ import annotations

import re
import unicodedata
from typing import Callable


_ATTACHED_SUFFIX_RE = re.compile(r"([A-Za-z가-힣ぁ-んァ-ヶ一-龯]+)$")
_SUFFIX_SCRIPT_RE = re.compile(r"[가-힣ぁ-んァ-ヶ一-龯]")
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d{1,3})$")
_TRAILING_ROMAN_SUFFIX_RE = re.compile(r"^(.*?)([A-Z]{1,4}\d{1,3})$")
_TRAILING_ROMAN_V_RE = re.compile(r"^(.*?)([A-Za-z]{1,8}?)(V\d{0,3})$")
_DEGENERATE_ALIAS_MARKERS = {"-", "r", "h", "R", "H", "'", "."}


def _accept_candidate_type(base_type: str, candidate_type: str, *, require_same_type: bool) -> bool:
    ctype = str(candidate_type or "")
    if not ctype:
        return False
    if not require_same_type:
        return True
    btype = str(base_type or "")
    return bool(btype and ctype == btype)


def _is_degenerate_alias_candidate(text: str) -> bool:
    compact = str(text or "").strip()
    return compact in _DEGENERATE_ALIAS_MARKERS


def strip_known_pitch_suffix(text: str) -> str:
    stripped = re.sub(r"(?:[_\-\s]+)(?:[A-G](?:#|b)?[0-8])$", "", text)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[A-G](?:sharp|flat)?[0-8])$", "", stripped)
    return stripped.strip()


def strip_bracket_suffix(text: str) -> str:
    return re.sub(r"\s*[\(\[\{（【].*?[\)\]\}）】]\s*$", "", text).strip()


def _try_strip_trailing_separator_suffix(
    text: str,
    classifier: Callable[[str], str],
    base_type: str,
) -> str:
    for sep in ("_", " "):
        if sep not in text:
            continue
        parts = text.rsplit(sep, 1)
        if len(parts) != 2:
            continue
        prefix = parts[0].strip()
        suffix = parts[1].strip()
        if not prefix or not suffix:
            continue
        if _is_degenerate_alias_candidate(prefix):
            continue
        prefix_type = str(classifier(prefix) or "")
        if prefix_type == base_type and prefix_type:
            return prefix
    return text


def _try_strip_trailing_roman_numeric_suffix(
    text: str,
    classifier: Callable[[str], str],
    base_type: str,
) -> str:
    def _check(candidate: str, *, require_same_type: bool) -> str:
        cand = str(candidate or "").rstrip(" _-").strip()
        if not cand or cand == text:
            return ""
        if _is_degenerate_alias_candidate(cand):
            return ""
        candidate_type = str(classifier(cand) or "")
        if _accept_candidate_type(base_type, candidate_type, require_same_type=require_same_type):
            return cand
        return ""

    if " " not in text and "_" not in text:
        match = _TRAILING_ROMAN_V_RE.match(text)
        if match:
            stripped = _check(match.group(1) + match.group(2), require_same_type=False)
            if stripped:
                return stripped

        match = _TRAILING_ROMAN_SUFFIX_RE.match(text)
        if match:
            stripped = _check(match.group(1), require_same_type=False)
            if stripped:
                return stripped

        match = _TRAILING_DIGITS_RE.match(text)
        if match:
            stripped = _check(match.group(1), require_same_type=False)
            if stripped:
                return stripped

    for sep in ("_", " "):
        if sep not in text:
            continue
        prefix, suffix = text.rsplit(sep, 1)
        prefix = prefix.strip()
        suffix = suffix.strip()
        if not prefix or not suffix:
            continue

        suffix_match = _TRAILING_ROMAN_V_RE.match(suffix)
        if suffix_match:
            candidate = prefix
            core = (suffix_match.group(1) + suffix_match.group(2)).strip()
            if core:
                candidate = f"{prefix}{sep}{core}"
            stripped = _check(candidate, require_same_type=False)
            if stripped:
                return stripped

        if re.fullmatch(r"[A-Za-z]{1,8}\d{1,3}", suffix):
            core = re.sub(r"\d{1,3}$", "", suffix).strip()
            candidate = prefix
            if core:
                candidate = f"{prefix}{sep}{core}"
            stripped = _check(candidate, require_same_type=False)
            if stripped:
                return stripped

    return text


def _try_strip_attached_suffix(
    text: str,
    classifier: Callable[[str], str],
    base_type: str,
) -> str:
    match = _ATTACHED_SUFFIX_RE.search(text)
    if not match:
        return text
    suffix = match.group(1)
    if not _SUFFIX_SCRIPT_RE.search(suffix):
        return text
    if len(suffix) <= 1 and not re.search(r"[一-龯]", suffix):
        return text
    run_start = len(text) - len(suffix)
    first_suffix_idx = next(
        (idx for idx, ch in enumerate(suffix) if _SUFFIX_SCRIPT_RE.match(ch)),
        None,
    )
    if first_suffix_idx is None:
        return text
    for cut in range(run_start + first_suffix_idx, len(text)):
        prefix = text[:cut].rstrip(" _-")
        if not prefix:
            continue
        if _is_degenerate_alias_candidate(prefix):
            continue
        prefix_type = str(classifier(prefix) or "")
        if prefix_type == base_type and prefix_type:
            return prefix
    return text


def strip_alias_annotation_suffixes(text: str, classifier: Callable[[str], str]) -> str:
    current = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not current:
        return ""
    base_type = str(classifier(current) or "")
    while True:
        previous = current
        current = strip_known_pitch_suffix(current)
        current = strip_bracket_suffix(current)
        current = _try_strip_trailing_roman_numeric_suffix(current, classifier, base_type)
        current = _try_strip_trailing_separator_suffix(current, classifier, base_type)
        current = _try_strip_attached_suffix(current, classifier, base_type)
        if current == previous:
            break
    return current


__all__ = [
    "strip_alias_annotation_suffixes",
    "strip_bracket_suffix",
    "strip_known_pitch_suffix",
]
