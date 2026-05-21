from __future__ import annotations

import os
import re
from typing import Iterable, Sequence

from core.model_context.types import FilenameTokenContext, ReclistMatchContext

_ORDER_TOKEN_RE = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3]+")
DEFAULT_IGNORE_LABELS: frozenset[str] = frozenset({"sil", "pau", "sp", "br", "endbr"})

_ROMAJI_CANON: dict[str, str] = {
    "shi": "sh",
    "si": "sh",
    "chi": "ch",
    "ti": "ch",
    "ji": "j",
    "zi": "j",
    "tsu": "ts",
    "tu": "ts",
    "fu": "f",
    "hu": "f",
    "nn": "n",
    "xn": "n",
    "ltsu": "q",
    "ltu": "q",
    "xtsu": "q",
    "xtu": "q",
    "cl": "q",
}

_JA_ONSET_VOWELS = ("a", "i", "u", "e", "o")
_JA_ONSET_SPLIT: dict[str, tuple[str, ...]] = {
    "ky": ("k", "y"),
    "gy": ("g", "y"),
    "ny": ("n", "y"),
    "hy": ("h", "y"),
    "my": ("m", "y"),
    "ry": ("r", "y"),
    "by": ("b", "y"),
    "py": ("p", "y"),
    "dy": ("d", "y"),
    "ty": ("t", "y"),
}


def canonicalize_order_token(token: object) -> str:
    text = str(token or "").strip().lower().strip("*-_")
    return _ROMAJI_CANON.get(text, text)


def _plain_order_token(token: object) -> str:
    return str(token or "").strip().lower().strip("*-_")


def filename_order_tokens(wav_name: object) -> list[str]:
    stem = os.path.splitext(os.path.basename(str(wav_name or "")))[0]
    out: list[str] = []
    for raw in _ORDER_TOKEN_RE.findall(stem):
        token = _plain_order_token(raw)
        if token:
            out.append(token)
    return out


def canonicalize_order_tokens(
    tokens: Iterable[object],
    *,
    ignore_labels: Iterable[str] = DEFAULT_IGNORE_LABELS,
) -> list[str]:
    ignore = {canonicalize_order_token(label) for label in ignore_labels}
    out: list[str] = []
    for token in tokens:
        text = canonicalize_order_token(token)
        if text and text not in ignore:
            out.append(text)
    return out


def read_lab_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 3:
                    tokens.append(" ".join(parts[2:]))
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp932", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 3:
                    tokens.append(parts[2])
    return tokens


def _explode_ja_romaji_onset(token: str) -> list[str]:
    text = canonicalize_order_token(token)
    if not text:
        return []
    if text in _JA_ONSET_VOWELS:
        return [text]
    try:
        from core.ja_lab_generator import split_ja_romaji_syllable

        onset, vowel = split_ja_romaji_syllable(text)
    except Exception:
        onset, vowel = "", text
    onset = canonicalize_order_token(onset)
    vowel = canonicalize_order_token(vowel)
    if onset and vowel and onset != vowel:
        return [*_JA_ONSET_SPLIT.get(onset, (onset,)), vowel]
    return [text]


def filename_expected_tokens(base: str, *, language: str = "ja") -> list[str]:
    lang = str(language or "").strip().lower()
    if lang in {"ja", "japanese", "jp"}:
        try:
            from core.ja_lab_generator import parse_ja_filename

            parsed = parse_ja_filename(base)
        except Exception:
            parsed = {}
        items: list[object] = []
        if isinstance(parsed, dict):
            items = list(parsed.get("syllables") or [])
        elif isinstance(parsed, (list, tuple)):
            items = list(parsed)
        if items:
            out: list[str] = []
            for syllable in items:
                if canonicalize_order_token(syllable) in {"r", "br", "bre", "breath"}:
                    continue
                out.extend(_explode_ja_romaji_onset(str(syllable)))
            return out
    return filename_order_tokens(base)


def ordered_token_match(expected: Sequence[object], observed: Sequence[object]) -> tuple[int, float]:
    exp = [canonicalize_order_token(item) for item in expected if canonicalize_order_token(item)]
    obs = [canonicalize_order_token(item) for item in observed if canonicalize_order_token(item)]
    if not exp or not obs:
        return 0, 0.0
    pos = 0
    hits = 0
    for token in exp:
        while pos < len(obs) and obs[pos] != token:
            pos += 1
        if pos >= len(obs):
            break
        hits += 1
        pos += 1
    return hits, hits / max(1, len(exp))


def edit_distance(a: Sequence[object], b: Sequence[object]) -> int:
    aa = [canonicalize_order_token(item) for item in a]
    bb = [canonicalize_order_token(item) for item in b]
    prev = list(range(len(bb) + 1))
    for i, ca in enumerate(aa, 1):
        cur = [i]
        for j, cb in enumerate(bb, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def parse_filename_context(
    wav_name: object,
    *,
    language: str = "",
    format_type: str = "",
) -> FilenameTokenContext:
    tokens = tuple(filename_order_tokens(wav_name))
    canonical = tuple(canonicalize_order_tokens(tokens))
    return FilenameTokenContext(
        raw=str(wav_name or ""),
        tokens=tokens,
        canonical_tokens=canonical,
        token_count=len(tokens),
        slot_count=max(1, len(tokens)),
        parse_confidence=1.0 if tokens else 0.0,
        warnings=() if tokens else ("empty_filename_tokens",),
    )


def classify_filename_lab_match(
    base: str,
    lab_tokens: Iterable[object],
    *,
    language: str = "ja",
    ignore_labels: Iterable[str] = DEFAULT_IGNORE_LABELS,
    loose_threshold: float = 0.5,
) -> ReclistMatchContext:
    expected = canonicalize_order_tokens(filename_expected_tokens(base, language=language), ignore_labels=ignore_labels)
    observed = canonicalize_order_tokens(lab_tokens, ignore_labels=ignore_labels)
    if not expected:
        status = "boundary_only"
        ratio = 0.0
        reason = "empty_expected_tokens"
    elif expected == observed:
        status = "match"
        ratio = 1.0
        reason = ""
    else:
        _hits, ratio = ordered_token_match(expected, observed)
        if ratio >= 0.999:
            status = "ordered_match"
            reason = ""
        elif ratio >= float(loose_threshold):
            status = "loose_match"
            reason = "partial_order_match"
        else:
            status = "mismatch"
            reason = "token_order_mismatch"
    return ReclistMatchContext(
        match_status=status,
        match_ratio=float(ratio),
        mismatch_reason=reason,
        expected_tokens=tuple(expected),
        observed_tokens=tuple(observed),
    )
