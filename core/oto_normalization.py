from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Optional

from core.ja_lab_generator import parse_ja_filename, repair_japanese_mojibake_text


def _katakana_to_hiragana(text: str) -> str:
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def normalize_wav_key(name: str) -> str:
    base = os.path.splitext(os.path.basename(str(name or "")))[0]
    base = repair_japanese_mojibake_text(base)
    if not base:
        return ""
    text = unicodedata.normalize("NFKC", base).strip().lower()
    text = _katakana_to_hiragana(text)
    text = re.sub(r"[\s_\-]+", "", text)
    text = re.sub(r"[`~!@#$%^&*()+={}\[\]|\\:;\"'<>,.?/・｡､。、「」『』（）［］｛｝]", "", text)
    text = re.sub(r"[^0-9a-z\u3041-\u3096\u30fc\u31f0-\u31ff\u3400-\u9fff\uac00-\ud7a3]+", "", text)
    return text


def _classify_alias_for_normalization(language: str, alias: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    lang = str(language or "").strip().lower()
    if lang == "japanese":
        from core.ja_oto_mapping import classify_ja_alias

        return str(classify_ja_alias(alias, custom_map=custom_map) or "")
    from core.kr_oto_rules import classify_alias

    return str(classify_alias(alias, custom_map=custom_map) or "")


def _strip_known_pitch_suffix(text: str) -> str:
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:#|b)?[0-8])$", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"(?:[_\-\s]+)(?:[a-g](?:sharp|flat)?[0-8])$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _strip_bracket_suffix(text: str) -> str:
    return re.sub(r"\s*[\(\[\{（【].*?[\)\]\}）】]\s*$", "", text).strip()


def _try_strip_trailing_separator_suffix(text: str, language: str, custom_map: Optional[Dict[str, str]], base_type: str) -> str:
    for sep in ("_", " "):
        if sep not in text:
            continue
        parts = text.rsplit(sep, 1)
        if len(parts) != 2:
            continue
        prefix, suffix = parts[0].strip(), parts[1].strip()
        if not prefix or not suffix:
            continue
        prefix_type = _classify_alias_for_normalization(language, prefix, custom_map=custom_map)
        if prefix_type == base_type and prefix_type:
            return prefix
    return text


def _try_strip_attached_suffix(text: str, language: str, custom_map: Optional[Dict[str, str]], base_type: str) -> str:
    match = re.search(r"([A-Za-z가-힣ぁ-んァ-ヶ一-龯]+)$", text)
    if not match:
        return text
    suffix = match.group(1)
    if not re.search(r"[가-힣ぁ-んァ-ヶ一-龯]", suffix):
        return text
    if len(suffix) <= 1 and not re.search(r"[一-龯]", suffix):
        return text
    run_start = len(text) - len(suffix)
    first_suffix_idx = next(
        (idx for idx, ch in enumerate(suffix) if re.match(r"[가-힣ぁ-んァ-ヶ一-龯]", ch)),
        None,
    )
    if first_suffix_idx is None:
        return text
    for cut in range(run_start + first_suffix_idx, len(text)):
        prefix = text[:cut].rstrip(" _-")
        if not prefix:
            continue
        prefix_type = _classify_alias_for_normalization(language, prefix, custom_map=custom_map)
        if prefix_type == base_type and prefix_type:
            return prefix
    return text


def _normalize_ja_alias_token(token: str) -> str:
    raw = repair_japanese_mojibake_text(str(token or ""))
    text = _katakana_to_hiragana(unicodedata.normalize("NFKC", raw).strip())
    if not text:
        return ""
    if text in {"-", "r", "h"}:
        return text.lower()
    for sep in ("_", "-"):
        if sep in text:
            head = text.split(sep, 1)[0].strip()
            if head and parse_ja_filename(head):
                text = head
                break
    syllables = parse_ja_filename(text)
    if syllables:
        return "".join(syllables).lower()
    return text.lower()


def canonicalize_alias_for_matching(language: str, alias: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    lang = str(language or "").strip().lower() or "korean"
    text = unicodedata.normalize("NFKC", str(alias or "")).strip()
    if not text:
        return ""
    if lang == "japanese":
        parts = [_normalize_ja_alias_token(part) for part in re.split(r"\s+", text) if part.strip()]
        text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if not text:
        return ""
    base_type = _classify_alias_for_normalization(lang, text, custom_map=custom_map)
    while True:
        prev = text
        text = _strip_known_pitch_suffix(text)
        text = _strip_bracket_suffix(text)
        text = _try_strip_trailing_separator_suffix(text, lang, custom_map, base_type)
        text = _try_strip_attached_suffix(text, lang, custom_map, base_type)
        if text == prev:
            break
    return text


__all__ = ["canonicalize_alias_for_matching", "normalize_wav_key"]
