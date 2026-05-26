from __future__ import annotations

import os
from typing import Iterable

from core.model_context.filename import (
    DEFAULT_IGNORE_LABELS,
    canonicalize_order_tokens,
    filename_expected_tokens,
    filename_order_tokens,
)


def collect_alignment_token_sequence(corpus_dir: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    if not os.path.isdir(corpus_dir):
        return tokens
    for filename in sorted(os.listdir(corpus_dir)):
        low = filename.lower()
        if not (low.endswith(".lab") or low.endswith(".txt")):
            continue
        path = os.path.join(corpus_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except Exception:
            continue
        for raw in str(text or "").replace("\n", " ").replace("\t", " ").split():
            token = str(raw or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
    return tokens


def filename_alignment_tokens(
    wav_path: str,
    *,
    language: str = "",
    ignore_labels: Iterable[str] = DEFAULT_IGNORE_LABELS,
) -> list[str]:
    base = os.path.splitext(os.path.basename(str(wav_path or "")))[0]
    lang = str(language or "").strip().lower()
    if lang in {"ja", "japanese", "jp"}:
        return canonicalize_order_tokens(filename_expected_tokens(base, language="ja"), ignore_labels=ignore_labels)
    return canonicalize_order_tokens(filename_order_tokens(base), ignore_labels=ignore_labels)
