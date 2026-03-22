from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Optional

from core.alias_annotation_utils import strip_alias_annotation_suffixes
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


def _normalize_ja_alias_token(token: str) -> str:
    raw = repair_japanese_mojibake_text(str(token or ""))
    text = _katakana_to_hiragana(unicodedata.normalize("NFKC", raw).strip())
    for marker in ("\u30fb", "\u00b7", "\uff65", "\u2022", "\u2019", "\u02bc", "\u0294", "\u02c0", "`"):
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
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


def _strip_attached_pitch_suffix_token(token: str) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    stripped = re.sub(r"([A-G](?:[#♯]|[b♭])?[0-8])$", "", text, flags=re.IGNORECASE).strip()
    stripped = stripped.rstrip("_- ").strip()
    if stripped:
        return stripped
    return text


def canonicalize_alias_for_matching(language: str, alias: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    lang = str(language or "").strip().lower() or "korean"
    text = unicodedata.normalize("NFKC", str(alias or "")).strip()
    if not text:
        return ""
    raw_base_type = _classify_alias_for_normalization(lang, text, custom_map=custom_map)
    if raw_base_type == "br":
        return "br"
    if lang == "japanese":
        parts = [_normalize_ja_alias_token(part) for part in re.split(r"\s+", text) if part.strip()]
        text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = " ".join(_strip_attached_pitch_suffix_token(part) for part in text.split(" ") if part.strip()).strip()
    if not text:
        return ""
    text = strip_alias_annotation_suffixes(
        text,
        classifier=lambda candidate: _classify_alias_for_normalization(lang, candidate, custom_map=custom_map),
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    base_type = _classify_alias_for_normalization(lang, text, custom_map=custom_map)
    if base_type == "br":
        return "br"
    return text


__all__ = ["canonicalize_alias_for_matching", "normalize_wav_key"]
