"""
Text I/O helpers for encoding-aware reads.
"""

from __future__ import annotations

import re
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


def _normalize_alias_suffix_token(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def _split_oto_line_alias(line: str) -> tuple[str, str, str] | None:
    if not line or "=" not in line:
        return None
    left, right = line.split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        return left, alias.strip(), rest
    return left, right.strip(), ""


def _replace_oto_line_alias(line: str, alias: str) -> str:
    split = _split_oto_line_alias(line)
    if split is None:
        return line
    left, _old_alias, rest = split
    if rest:
        return f"{left}={alias},{rest}"
    return f"{left}={alias}"


def strip_template_alias_suffixes(
    lines: List[str],
    *,
    alias_suffix: str = "",
) -> tuple[List[str], str, int]:
    """
    Strip pitch/style suffixes from template aliases for internal matching.
    Returns: (normalized_lines, stripped_suffix, changed_count)
    """
    suffix = _normalize_alias_suffix_token(alias_suffix)
    if suffix:
        suffixes = [suffix]
    else:
        found: dict[str, int] = {}
        suffix_pattern = re.compile(r"_(?P<suffix>[A-Ga-g][#b]?\d)$")
        for line in lines or []:
            split = _split_oto_line_alias(line)
            if split is None:
                continue
            _left, alias, _rest = split
            match = suffix_pattern.search(alias)
            if not match:
                continue
            key = match.group("suffix")
            found[key] = int(found.get(key, 0)) + 1
        total_aliases = max(1, sum(1 for line in (lines or []) if _split_oto_line_alias(line) is not None))
        suffixes = [
            key for key, count in found.items()
            if count >= 1 and (count / float(total_aliases)) >= 0.10
        ]
        suffixes.sort(key=len, reverse=True)
    if not suffixes:
        return list(lines or []), "", 0

    out: List[str] = []
    changed = 0
    stripped_label = suffixes[0]
    for line in lines or []:
        split = _split_oto_line_alias(line)
        if split is None:
            out.append(line)
            continue
        _left, alias, _rest = split
        new_alias = alias
        for suffix_item in suffixes:
            needle = "_" + suffix_item
            if new_alias.endswith(needle) and len(new_alias) > len(needle):
                new_alias = new_alias[: -len(needle)].rstrip()
                stripped_label = suffix_item
                break
        if new_alias != alias:
            out.append(_replace_oto_line_alias(line, new_alias))
            changed += 1
        else:
            out.append(line)
    return out, stripped_label, changed

