from __future__ import annotations

import re
import unicodedata
from typing import Callable


_ATTACHED_SUFFIX_RE = re.compile(r"([A-Za-z가-힣ぁ-んァ-ヶ一-龯]+)$")
_SUFFIX_SCRIPT_RE = re.compile(r"[가-힣ぁ-んァ-ヶ一-龯]")


def strip_known_pitch_suffix(text: str) -> str:
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:#|b)?[0-8])$", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:sharp|flat)?[0-8])$", "", stripped, flags=re.IGNORECASE)
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
        prefix_type = str(classifier(prefix) or "")
        if prefix_type == base_type and prefix_type:
            return prefix
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
