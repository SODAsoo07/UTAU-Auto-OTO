"""
Text I/O helpers for encoding-aware reads.
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional, Tuple


ENCODING_CANDIDATES = (
    "utf-8-sig",
    "utf-8",
    "cp932",
    "shift_jis",
    "cp949",
    "euc-kr",
)


def _looks_east_asian(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana/Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0x1100 <= code <= 0x11FF  # Hangul jamo
    )


def _text_quality_score(text: str) -> float:
    score = 0.0
    east_asian = 0
    suspicious = 0
    for ch in text:
        if ch in "\r\n\t":
            continue
        if ch == "\ufffd":
            score -= 8.0
            suspicious += 1
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            score -= 5.0
            suspicious += 1
            continue
        if _looks_east_asian(ch):
            east_asian += 1
            score += 2.2
            continue
        code = ord(ch)
        if 0x80 <= code <= 0x024F:
            suspicious += 1
            score -= 0.8

    line_bonus = 0.0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "=" in s and s.count(",") >= 5:
            line_bonus += 1.5
        elif "=" in s:
            line_bonus += 0.6
    score += line_bonus
    score += min(east_asian, 20) * 0.1
    score -= min(suspicious, 40) * 0.05
    return score


def read_text_auto(path: str, encodings=ENCODING_CANDIDATES) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Read text file with fallback encodings.
    Returns: (text, encoding, error_message)
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception as e:
        return None, None, f"파일 읽기 실패: {e}"

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig"), "utf-8-sig", None
        except UnicodeDecodeError:
            pass

    decoded_candidates = []
    for enc in encodings:
        try:
            decoded_candidates.append((enc, raw.decode(enc)))
        except UnicodeDecodeError:
            continue

    if not decoded_candidates:
        return None, None, "지원 인코딩(UTF-8/CP932/CP949)으로 해석할 수 없습니다."

    if len(decoded_candidates) == 1:
        enc, text = decoded_candidates[0]
        return text, enc, None

    if any(b >= 0x80 for b in raw):
        utf8_text = None
        cp932_text = None
        for enc, text in decoded_candidates:
            if enc == "utf-8":
                utf8_text = text
            elif enc == "cp932":
                cp932_text = text
        if utf8_text is not None and cp932_text is not None:
            utf8_score = _text_quality_score(utf8_text)
            cp932_score = _text_quality_score(cp932_text)
            if cp932_score >= utf8_score + 2.0:
                return cp932_text, "cp932", None

    best_enc, best_text = max(decoded_candidates, key=lambda item: _text_quality_score(item[1]))
    return best_text, best_enc, None


def load_template_oto_lines(
    tpl_path: str,
    require_utf8: bool = False,
    mode_label: str = "",
) -> Tuple[Optional[List[str]], Optional[str], Optional[str], Optional[str]]:
    """
    Load `wav=alias,...` lines from template oto.ini with encoding detection.
    Returns: (lines, detected_encoding, warning_message, error_message)
    """
    text, enc, err = read_text_auto(tpl_path)
    if err:
        return None, None, None, f"❌ 템플릿 OTO 인코딩 판독 실패: {err}"

    if require_utf8 and enc not in ("utf-8", "utf-8-sig"):
        label = mode_label or "현재 모드"
        return (
            None,
            enc,
            None,
            f"❌ 템플릿 OTO 인코딩 감지: {enc}. {label}는 UTF-8 인코딩의 oto.ini만 지원됩니다.",
        )

    warning = None
    if enc not in ("utf-8", "utf-8-sig"):
        warning = (
            f"⚠ 템플릿 OTO 인코딩 감지: {enc}. "
            "UTF-8이 아니어서 일부 문자가 깨질 수 있습니다."
        )

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" in line:
            lines.append(line)

    return lines, enc, warning, None
