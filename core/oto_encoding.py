from __future__ import annotations

import os
from typing import Sequence

from core.textio_utils import read_text_auto


def is_shift_jis_family_encoding(encoding: str) -> bool:
    enc = str(encoding or "").strip().lower().replace("-", "_")
    return enc in {"shift_jis", "sjis", "cp932", "ms932"}


def normalize_shift_jis_encoding(encoding: str) -> str:
    enc = str(encoding or "").strip().lower().replace("-", "_")
    if enc in {"shift_jis", "sjis"}:
        return "shift_jis"
    if enc in {"cp932", "ms932"}:
        return "cp932"
    return ""


def detect_text_file_encoding(path: str) -> str:
    text, enc, _err = read_text_auto(path)
    if text is None:
        return ""
    return str(enc or "").strip().lower()


def resolve_oto_write_encoding(
    oto_path: str,
    *,
    language: str = "",
    preferred_encoding: str = "",
) -> str:
    lang = str(language or "").strip().lower()
    preferred = normalize_shift_jis_encoding(preferred_encoding)
    if preferred:
        return preferred
    if lang != "japanese":
        return "utf-8"
    path = str(oto_path or "").strip()
    if not path or not os.path.isfile(path):
        return "utf-8"
    detected = detect_text_file_encoding(path)
    normalized = normalize_shift_jis_encoding(detected)
    return normalized or "utf-8"


def write_oto_lines_with_encoding(
    out_path: str,
    lines: Sequence[str],
    *,
    language: str = "",
    preferred_encoding: str = "",
) -> str:
    encoding = resolve_oto_write_encoding(
        out_path,
        language=language,
        preferred_encoding=preferred_encoding,
    )
    normalized = normalize_shift_jis_encoding(encoding)
    target = normalized or "utf-8"
    line_rows = [str(line) for line in (lines or [])]
    try:
        with open(out_path, "w", encoding=target) as f:
            for line in line_rows:
                f.write(line + "\n")
        return target
    except UnicodeEncodeError:
        if target != "shift_jis":
            raise
        with open(out_path, "w", encoding="cp932") as f:
            for line in line_rows:
                f.write(line + "\n")
        return "cp932"


__all__ = [
    "detect_text_file_encoding",
    "is_shift_jis_family_encoding",
    "normalize_shift_jis_encoding",
    "resolve_oto_write_encoding",
    "write_oto_lines_with_encoding",
]

