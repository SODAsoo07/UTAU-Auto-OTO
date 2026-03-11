from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from core.textio_utils import read_text_auto


NOTE_RE = re.compile(r"([A-Ga-g](?:#|b)?\d)")


@dataclass(frozen=True)
class PrefixMapRule:
    note: str
    prefix: str
    suffix: str


def normalize_note_name(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not raw:
        return ""
    m = NOTE_RE.search(raw)
    if not m:
        return ""
    note = m.group(1)
    return note[0].upper() + note[1:]


def _iter_parent_dirs(path: str) -> Iterable[str]:
    current = os.path.abspath(path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    seen = set()
    while current and current not in seen:
        seen.add(current)
        yield current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        current = parent


def _iter_map_candidates(directory: str) -> Iterable[str]:
    preferred = os.path.join(directory, "prefix.map")
    if os.path.isfile(preferred):
        yield preferred

    try:
        entries = sorted(os.listdir(directory), key=lambda s: s.lower())
    except OSError:
        return

    for name in entries:
        if name.lower() == "prefix.map":
            continue
        if not name.lower().endswith(".map"):
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            yield candidate


def _split_map_line(raw: str) -> List[str]:
    line = raw.strip("\ufeff").rstrip()
    if not line or line.startswith("#") or line.startswith(";"):
        return []
    if "\t" in line:
        return [p.strip() for p in line.split("\t")]
    parts = [p.strip() for p in re.split(r"\s{2,}", line)]
    if len(parts) >= 3:
        return parts
    return [p.strip() for p in re.split(r"\s+", line, maxsplit=2)]


def _is_pitch_affix(text: str) -> bool:
    s = str(text or "")
    if not s:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    if re.search(r"[_\-\+\[\]\(\){}#]", s):
        return True
    if re.search(r"[A-Z]", s):
        return True
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", s):
        return True
    return len(s) >= 2


def _looks_like_affix_value(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if value == "-":
        return True
    return _is_pitch_affix(value)


def _parse_prefix_map_rules(text: str) -> Tuple[PrefixMapRule, ...]:
    rules: List[PrefixMapRule] = []
    for raw in text.splitlines():
        parts = _split_map_line(raw)
        if not parts:
            continue
        note = normalize_note_name(parts[0])
        if not note:
            continue
        prefix = parts[1] if len(parts) >= 2 else ""
        suffix = parts[2] if len(parts) >= 3 else ""
        if not (_looks_like_affix_value(prefix) and _looks_like_affix_value(suffix)):
            continue
        rules.append(PrefixMapRule(note=note, prefix=prefix, suffix=suffix))
    return tuple(rules)


def find_prefix_map_path(*paths: str) -> str:
    seen_dirs = set()
    for path in paths:
        if not path:
            continue
        for directory in _iter_parent_dirs(path):
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            for candidate in _iter_map_candidates(directory):
                if load_prefix_map(candidate):
                    return candidate
                if os.path.basename(candidate).lower() == "prefix.map":
                    return candidate
    return ""


@lru_cache(maxsize=256)
def load_prefix_map(path: str) -> Tuple[PrefixMapRule, ...]:
    if not path or not os.path.isfile(path):
        return ()
    text, _enc, err = read_text_auto(path)
    if err or text is None:
        return ()
    return _parse_prefix_map_rules(text)


def _extract_context_notes(*values: str) -> List[str]:
    notes: List[str] = []
    seen = set()
    for value in values:
        if not value:
            continue
        abs_value = os.path.abspath(value)
        candidates = [os.path.basename(abs_value), os.path.splitext(os.path.basename(abs_value))[0]]
        parts = [p for p in re.split(r"[\\/]+", abs_value) if p]
        candidates.extend(parts[-6:])
        for cand in candidates:
            note = normalize_note_name(cand)
            if note and note not in seen:
                seen.add(note)
                notes.append(note)
    return notes


def _candidate_rules_for_context(rules: Sequence[PrefixMapRule], wav_name: str = "", context_paths: Sequence[str] = ()) -> List[PrefixMapRule]:
    by_note: Dict[str, List[PrefixMapRule]] = {}
    for rule in rules:
        by_note.setdefault(rule.note, []).append(rule)

    selected: List[PrefixMapRule] = []
    seen = set()
    notes = _extract_context_notes(wav_name, *context_paths)
    for note in notes:
        for rule in by_note.get(note, []):
            key = (rule.prefix, rule.suffix)
            if key not in seen:
                selected.append(rule)
                seen.add(key)

    if selected:
        return selected

    unique_rules = []
    for rule in rules:
        key = (rule.prefix, rule.suffix)
        if key in seen:
            continue
        seen.add(key)
        unique_rules.append(rule)
    if len(unique_rules) == 1:
        return unique_rules
    return []


def strip_prefix_map_affixes(
    alias: str,
    prefix_map_path: str = "",
    wav_name: str = "",
    context_paths: Sequence[str] = (),
) -> str:
    text = unicodedata.normalize("NFKC", str(alias or "")).strip()
    if not text or not prefix_map_path:
        return text
    rules = load_prefix_map(prefix_map_path)
    if not rules:
        return text

    candidates = _candidate_rules_for_context(rules, wav_name=wav_name, context_paths=context_paths)
    best = text
    best_delta = 0
    for rule in candidates:
        cur = text
        removed = 0
        prefix = str(rule.prefix or "")
        suffix = str(rule.suffix or "")
        if prefix and _is_pitch_affix(prefix) and cur.startswith(prefix):
            cur = cur[len(prefix):].lstrip()
            removed += len(prefix)
        if suffix and _is_pitch_affix(suffix) and cur.endswith(suffix):
            cur = cur[:-len(suffix)].rstrip()
            removed += len(suffix)
        if removed <= 0 or not cur:
            continue
        if removed > best_delta:
            best = cur
            best_delta = removed
    return best


__all__ = [
    "PrefixMapRule",
    "find_prefix_map_path",
    "load_prefix_map",
    "normalize_note_name",
    "strip_prefix_map_affixes",
]
