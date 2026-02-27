"""
Text I/O helpers for encoding-aware reads.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


ENCODING_CANDIDATES = (
    "utf-8-sig",
    "utf-8",
    "cp932",
    "shift_jis",
    "cp949",
    "euc-kr",
)


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

    for enc in encodings:
        try:
            return raw.decode(enc), enc, None
        except UnicodeDecodeError:
            continue

    return None, None, "지원 인코딩(UTF-8/CP932/CP949)으로 해석할 수 없습니다."


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

