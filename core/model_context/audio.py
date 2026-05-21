from __future__ import annotations

import os
from typing import Callable

from core.generation.file_index import iter_wav_files
from core.model_context.oto_params import wav_duration_ms
from core.model_context.types import AudioContext


def build_wav_index(
    root: str,
    *,
    normalize_key_fn: Callable[[str], str] | None = None,
    recursive: bool = True,
    include_relative: bool = False,
    include_stem: bool = False,
) -> dict[str, str]:
    key_fn = normalize_key_fn or (lambda value: str(value or "").strip().lower())
    index: dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return index
    for dirpath, filename in iter_wav_files(root, recursive=recursive):
        path = os.path.join(dirpath, filename)
        keys = [filename]
        if include_stem:
            keys.append(os.path.splitext(filename)[0])
        if include_relative:
            rel = os.path.relpath(path, root).replace("\\", "/")
            keys.append(rel)
            keys.append(os.path.splitext(rel)[0])
        for raw_key in keys:
            key = key_fn(raw_key)
            if key:
                index.setdefault(key, path)
    return index


def build_audio_context(wav_path: str, *, mode: str = "duration") -> AudioContext:
    duration = wav_duration_ms(wav_path)
    if str(mode or "").strip().lower() not in {"light", "candidates"}:
        return AudioContext(
            wav_path=str(wav_path or ""),
            duration_ms=duration,
            active_start_ms=0.0,
            active_end_ms=duration,
            audio_reliability=0.4 if duration <= 1.0 else 0.6,
            warnings=() if duration > 1.0 else ("missing_or_unreadable_wav",),
        )
    return AudioContext(
        wav_path=str(wav_path or ""),
        duration_ms=duration,
        active_start_ms=0.0,
        active_end_ms=duration,
        audio_reliability=0.55 if duration > 1.0 else 0.3,
        warnings=("light_audio_candidates_deprecated",),
    )
