from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Optional


def read_text_with_fallback(path: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return ""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            if "=" in text:
                return text
        except Exception:
            continue
    candidates: list[tuple[float, str]] = []
    for enc in ("cp932", "cp949", "euc-kr", "latin-1"):
        try:
            text = raw.decode(enc, errors="replace")
        except Exception:
            continue
        if "=" not in text:
            continue
        candidates.append((_decoded_text_score(text), text))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return raw.decode("utf-8", errors="replace")


def _decoded_text_score(text: str) -> float:
    replacements = text.count("\ufffd")
    c1_controls = sum(1 for char in text if 0x80 <= ord(char) <= 0x9F)
    japanese = sum(1 for char in text if 0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF)
    hangul = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7AF or 0x3130 <= ord(char) <= 0x318F)
    line_markers = text.count("=") + text.count(",")
    return float(line_markers) + (0.8 * float(japanese)) + (0.4 * float(hangul)) - (40.0 * float(replacements)) - (
        8.0 * float(c1_controls)
    )


def parse_oto_line(
    line: str,
    *,
    alias_filter: Optional[Callable[[str], bool]] = None,
) -> Optional[Dict[str, float]]:
    s = (line or "").strip()
    if not s or "=" not in s:
        return None
    wav_name, rest = s.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 6:
        return None
    try:
        offset = _parse_oto_number(parts[1])
        cons = _parse_oto_number(parts[2])
        cutoff = _parse_oto_number(parts[3])
        pre = _parse_oto_number(parts[4])
        ovl = _parse_oto_number(parts[5])
    except ValueError:
        return None
    alias = parts[0].strip()
    if alias_filter and alias_filter(alias):
        return None
    return {
        "wav": wav_name.strip(),
        "alias": alias,
        "offset": offset,
        "cons": cons,
        "cutoff": cutoff,
        "pre": pre,
        "ovl": ovl,
    }


def _parse_oto_number(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    return float(text)


def extract_base_timing_shape(
    line: str,
    *,
    alias_filter: Optional[Callable[[str], bool]] = None,
) -> Optional[Dict[str, float]]:
    row = parse_oto_line(line, alias_filter=alias_filter)
    if not row:
        return None
    pre = max(float(row["pre"]), 0.0)
    cons = max(float(row["cons"]), 0.0)
    cut_abs = abs(float(row["cutoff"]))
    ovl = max(float(row["ovl"]), 0.0)
    off = max(float(row["offset"]), 0.0)
    if pre < 1.0 and cons < 1.0 and cut_abs < 1.0 and ovl < 1.0:
        return None
    cons_gap = max(cons - pre, 8.0)
    cut_gap = max(cut_abs - cons, 16.0)
    ovl_ratio = (ovl / pre) if pre > 1e-6 else 0.30
    return {
        "offset": off,
        "pre": pre,
        "cons_gap": cons_gap,
        "cut_gap": cut_gap,
        "ovl_ratio": max(0.04, min(0.86, ovl_ratio)),
    }


def read_oto_rows_for_profile(
    path: str,
    *,
    alias_filter: Optional[Callable[[str], bool]] = None,
    wav_normalizer: Optional[Callable[[str], str]] = None,
    alias_normalizer: Optional[Callable[[str], str]] = None,
) -> list[Dict[str, object]]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    text = read_text_with_fallback(path)
    for raw in text.splitlines():
        row = parse_oto_line(raw, alias_filter=alias_filter)
        if not row:
            continue
        if wav_normalizer is not None:
            row["wav_norm"] = wav_normalizer(str(row["wav"]))
        if alias_normalizer is not None:
            row["alias_norm"] = alias_normalizer(str(row["alias"]))
        rows.append(row)
    return rows


def build_occurrence_map(
    rows: Iterable[Dict[str, object]],
    *,
    wav_key_field: str = "wav_norm",
    alias_key_field: str = "alias_norm",
) -> Dict[tuple[str, str, int], Dict[str, object]]:
    mapped: Dict[tuple[str, str, int], Dict[str, object]] = {}
    counters: Dict[tuple[str, str], int] = {}
    for row in rows:
        base_key = (str(row[wav_key_field]), str(row[alias_key_field]))
        idx = counters.get(base_key, 0)
        counters[base_key] = idx + 1
        mapped[(base_key[0], base_key[1], idx)] = row
    return mapped


def normalize_alias_suffix(suffix: str) -> str:
    text = str(suffix or "").strip()
    if not text:
        return ""
    return text[1:] if text.startswith("_") else text


def apply_alias_suffix(line: str, suffix: str) -> str:
    normalized = normalize_alias_suffix(suffix)
    if not normalized or "=" not in str(line or ""):
        return line
    left, right = str(line).split("=", 1)
    if "," in right:
        alias, rest = right.split(",", 1)
        alias = alias.strip()
        if alias:
            alias = f"{alias}_{normalized}"
        return f"{left}={alias},{rest}"
    alias = right.strip()
    if alias:
        alias = f"{alias}_{normalized}"
    return f"{left}={alias}"


def _normalize_encoding_name(encoding: str) -> str:
    name = str(encoding or "").strip().lower().replace("-", "_")
    aliases = {
        "shift_jis": "cp932",
        "shiftjis": "cp932",
        "sjis": "cp932",
        "ms932": "cp932",
        "windows_31j": "cp932",
        "utf8": "utf-8",
        "utf_8": "utf-8",
        "utf8_sig": "utf-8-sig",
        "utf_8_sig": "utf-8-sig",
        "uhc": "cp949",
        "ms949": "cp949",
        "euckr": "euc-kr",
    }
    return aliases.get(name, str(encoding or "").strip() or "utf-8")


def detect_oto_text_encoding(path: str) -> str:
    """Best-effort detection of an oto.ini's on-disk text encoding.

    Returns a Python codec name suitable for re-writing the file. UTF-8 (with or
    without BOM) is preferred; otherwise the first legacy codec that decodes the
    bytes into oto-looking text wins (notably ``cp932`` for Shift-JIS voicebanks).
    Falls back to ``utf-8`` when the file is missing or undecodable.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except Exception:
        return "utf-8"
    if not raw:
        return "utf-8"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        if "=" in raw.decode("utf-8"):
            return "utf-8"
    except UnicodeDecodeError:
        pass
    for enc in ("cp932", "cp949", "euc-kr"):
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        if "=" in text:
            return enc
    return "utf-8"


def reencode_oto_file(path: str, target_encoding: str) -> tuple[bool, str]:
    """Rewrite an oto.ini in ``target_encoding`` if it is not already.

    Returns ``(changed, encoding_used)``. If the text cannot be represented in the
    target encoding the file is left untouched (returns ``(False, current)``) so a
    voicebank is never corrupted by a lossy conversion.
    """
    target = _normalize_encoding_name(target_encoding)
    if not path or not os.path.isfile(path) or not target:
        return (False, "")
    current = detect_oto_text_encoding(path)
    if _normalize_encoding_name(current) == target:
        return (False, current)
    text = read_text_with_fallback(path)
    if not text:
        return (False, current)
    try:
        data = text.encode(target)
    except (LookupError, UnicodeEncodeError):
        return (False, current)
    with open(path, "wb") as handle:
        handle.write(data)
    return (True, target)


def _oto_line_alias(line: str) -> str:
    if "=" not in str(line or ""):
        return ""
    right = str(line).split("=", 1)[1]
    alias = right.split(",", 1)[0] if "," in right else right
    return alias.strip()


def apply_alias_suffix_to_line_idempotent(line: str, suffix: str) -> str:
    """Append the suffix to an oto line's alias unless it is already present.

    Unlike :func:`apply_alias_suffix`, this is safe to run on an oto file that may
    already carry the suffix (e.g. re-applying after generation), so it never
    double-appends ``_C4_C4``.
    """
    normalized = normalize_alias_suffix(suffix)
    if not normalized:
        return line
    alias = _oto_line_alias(line)
    if not alias or alias.endswith(f"_{normalized}"):
        return line
    return apply_alias_suffix(line, normalized)


def apply_alias_suffix_to_oto_text(text: str, suffix: str) -> tuple[str, int]:
    """Append ``suffix`` to every alias in oto text, idempotently.

    Returns the rewritten text and the number of alias lines that changed.
    Newline style and non-alias lines (comments, blanks) are preserved.
    """
    normalized = normalize_alias_suffix(suffix)
    src = str(text or "")
    if not normalized or not src:
        return src, 0
    newline = "\r\n" if "\r\n" in src else "\n"
    changed = 0
    out: list[str] = []
    for raw in src.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        new_line = apply_alias_suffix_to_line_idempotent(line, normalized)
        if new_line != line:
            changed += 1
        out.append(new_line)
    return newline.join(out), changed


def apply_alias_suffix_to_oto_file(path: str, suffix: str) -> tuple[int, int]:
    """Re-apply ``suffix`` to an already-generated oto.ini in place.

    Returns ``(changed_count, alias_line_count)``. A no-op (returns ``(0, 0)``)
    when the suffix is empty, the path is missing, or the file is unreadable.
    """
    normalized = normalize_alias_suffix(suffix)
    if not normalized or not path or not os.path.isfile(path):
        return (0, 0)
    text = read_text_with_fallback(path)
    if not text:
        return (0, 0)
    alias_line_count = sum(1 for ln in text.split("\n") if _oto_line_alias(ln))
    new_text, changed = apply_alias_suffix_to_oto_text(text, normalized)
    if changed:
        # Preserve the file's existing on-disk encoding (e.g. Shift-JIS voicebanks)
        # rather than forcing UTF-8.
        encoding = _normalize_encoding_name(detect_oto_text_encoding(path))
        try:
            data = new_text.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            data = new_text.encode("utf-8")
        with open(path, "wb") as handle:
            handle.write(data)
    return (changed, alias_line_count)


__all__ = [
    "apply_alias_suffix",
    "apply_alias_suffix_to_line_idempotent",
    "apply_alias_suffix_to_oto_file",
    "apply_alias_suffix_to_oto_text",
    "build_occurrence_map",
    "detect_oto_text_encoding",
    "extract_base_timing_shape",
    "normalize_alias_suffix",
    "parse_oto_line",
    "read_oto_rows_for_profile",
    "read_text_with_fallback",
    "reencode_oto_file",
]
