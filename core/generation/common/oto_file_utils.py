from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Optional


def read_text_with_fallback(path: str) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return ""
    for enc in ("utf-8-sig", "cp932", "utf-8", "euc-kr", "latin-1"):
        try:
            text = raw.decode(enc)
            if "=" in text:
                return text
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


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
        offset = float(parts[1].strip())
        cons = float(parts[2].strip())
        cutoff = float(parts[3].strip())
        pre = float(parts[4].strip())
        ovl = float(parts[5].strip())
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


__all__ = [
    "apply_alias_suffix",
    "build_occurrence_map",
    "extract_base_timing_shape",
    "normalize_alias_suffix",
    "parse_oto_line",
    "read_oto_rows_for_profile",
    "read_text_with_fallback",
]
