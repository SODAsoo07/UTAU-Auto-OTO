from __future__ import annotations

import os
from typing import Callable, Iterator


def iter_files_with_suffix(
    root: str,
    suffix: str,
    *,
    recursive: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield ``(dirpath, filename)`` for files under root matching suffix."""
    if not root or not os.path.isdir(root):
        return
    suffix_norm = str(suffix or "").lower()
    if recursive:
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    if filename.lower().endswith(suffix_norm):
                        yield dirpath, filename
        except Exception:
            return
        return

    try:
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if os.path.isfile(path) and filename.lower().endswith(suffix_norm):
                yield root, filename
    except Exception:
        return


def iter_textgrid_files(root: str, *, recursive: bool = True) -> Iterator[tuple[str, str]]:
    yield from iter_files_with_suffix(root, ".textgrid", recursive=recursive)


def iter_wav_files(root: str, *, recursive: bool = True) -> Iterator[tuple[str, str]]:
    yield from iter_files_with_suffix(root, ".wav", recursive=recursive)


def has_direct_wav_files(root: str) -> bool:
    try:
        return any(True for _dirpath, _filename in iter_wav_files(root, recursive=False))
    except Exception:
        return False


def list_wav_names(root: str, *, recursive: bool = False) -> set[str]:
    names: set[str] = set()
    for _dirpath, filename in iter_wav_files(root, recursive=recursive):
        names.add(filename)
    return names


def build_wav_index(
    root: str,
    *,
    normalize_key_fn: Callable[[str], str],
    recursive: bool = True,
) -> dict[str, str]:
    wav_index: dict[str, str] = {}
    for dirpath, filename in iter_wav_files(root, recursive=recursive):
        key = normalize_key_fn(filename)
        if key:
            wav_index.setdefault(key, os.path.join(dirpath, filename))
    return wav_index


__all__ = [
    "build_wav_index",
    "has_direct_wav_files",
    "iter_files_with_suffix",
    "iter_textgrid_files",
    "iter_wav_files",
    "list_wav_names",
]
