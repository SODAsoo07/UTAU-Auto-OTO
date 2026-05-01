from __future__ import annotations

import os
import re

from core.coarse_crnn.labels import coarse_for_phone

_HANGUL_RE = re.compile(r"[가-힣]")
_KANA_RE = re.compile(r"[\u3041-\u3096\u30A1-\u30FA\u30FC]")
_ASCII_RE = re.compile(r"[A-Za-z]+")
_SEP_RE = re.compile(r"[\s,_\-/|']+")


def normalize_language(language: object) -> str:
    text = str(language or "").strip().lower()
    if text in {"ko", "kor", "korean", "한국어"}:
        return "korean"
    if text in {"ja", "jp", "jpn", "japanese", "일본어"}:
        return "japanese"
    return text or ""


def short_language(language: object) -> str:
    lang = normalize_language(language)
    if lang == "korean":
        return "ko"
    if lang == "japanese":
        return "ja"
    return lang


def infer_language_from_path(path: object) -> str:
    text = str(path or "").replace("\\", "/").lower()
    parts = [p for p in text.split("/") if p]
    joined = "/".join(parts)
    if any(p in {"korean", "ko", "kor"} for p in parts) or "한국" in joined or "다화자" in joined or "다음색" in joined:
        return "korean"
    if any(p in {"japanese", "ja", "jp", "jpn"} for p in parts) or "utagoe" in joined or "kiritan" in joined:
        return "japanese"
    if "tiger_ja" in joined or "波音" in str(path or ""):
        return "japanese"
    if "english" in parts or "en" in parts or "_en" in joined or "eng" in joined:
        return ""
    return ""


def phones_from_text(text: object, language: object = "") -> list[str]:
    lang = normalize_language(language)
    raw = str(text or "").strip()
    if not raw:
        return []
    if lang == "japanese" or _KANA_RE.search(raw):
        return _phones_from_japanese(raw)
    if lang == "korean" or _HANGUL_RE.search(raw):
        return _phones_from_korean(raw)
    return _phones_from_ascii(raw)


def phones_from_wav_name(path: object, language: object = "") -> list[str]:
    base = os.path.splitext(os.path.basename(str(path or "")))[0]
    return phones_from_text(base, language)


def _phones_from_ascii(text: str) -> list[str]:
    out: list[str] = []
    for token in [t for t in _SEP_RE.split(text) if t]:
        if coarse_for_phone(token) != "SP" or token.lower() in {"cl", "n"}:
            out.append(token)
            continue
        out.extend(_ASCII_RE.findall(token.lower()))
    return [p for p in out if p]


def _phones_from_korean(text: str) -> list[str]:
    out: list[str] = []
    try:
        from core.lab_generator import decompose_hangul_to_roman
    except Exception:
        decompose_hangul_to_roman = None

    for token in [t for t in _SEP_RE.split(text) if t]:
        if not _HANGUL_RE.search(token):
            out.extend(_phones_from_ascii(token))
            continue
        for ch in token:
            if _HANGUL_RE.fullmatch(ch) and decompose_hangul_to_roman is not None:
                try:
                    out.extend([p for p in decompose_hangul_to_roman(ch) if str(p or "").strip()])
                except Exception:
                    pass
            else:
                out.extend(_phones_from_ascii(ch))
    return [p for p in out if p]


def _phones_from_japanese(text: str) -> list[str]:
    syllables: list[str] = []
    try:
        from core.ja_lab_generator import parse_ja_filename, split_ja_romaji_syllable

        syllables = list(parse_ja_filename(text) or [])
    except Exception:
        split_ja_romaji_syllable = None

    if not syllables:
        syllables = [t for t in _SEP_RE.split(text) if t]

    out: list[str] = []
    for syllable in syllables:
        s = str(syllable or "").strip()
        if not s:
            continue
        if s.lower() in {"r", "br", "breath", "息"}:
            out.append("br" if s.lower() != "r" else "sil")
            continue
        if split_ja_romaji_syllable is not None:
            onset, vowel = split_ja_romaji_syllable(s)
            if onset:
                out.append(str(onset).lower())
            if vowel:
                out.append("N" if str(vowel).lower() == "n" else str(vowel).lower())
            continue
        out.extend(_phones_from_ascii(s))
    return [p for p in out if p]


__all__ = [
    "infer_language_from_path",
    "normalize_language",
    "phones_from_text",
    "phones_from_wav_name",
    "short_language",
]
