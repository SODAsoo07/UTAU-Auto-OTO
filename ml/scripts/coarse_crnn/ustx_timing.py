"""Parse OpenUtau .ustx project files into note-level timing for evaluation.

The listening UI needs to know where each alias appears in the rendered wav.
A USTx file is YAML with this relevant subset::

    bpm: 120                       # initial bpm, used if no tempos entries cover position 0
    resolution: 480                # ticks per beat
    tempos:                        # list of tempo changes, sorted by position
      - {position: 0, bpm: 140}
      - {position: 7680, bpm: 90}
    voice_parts:
      - position: 0                # part offset in ticks
        notes:
          - {position: 1920, duration: 160, lyric: "ta", ...}

Converting `position_ticks` to absolute milliseconds requires honoring tempo
changes. This module produces (lyric, start_ms, end_ms) tuples for every
note across every voice part so the HTML labeling page can drive playback
with `audio.currentTime`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class NoteTiming:
    lyric: str
    start_ms: float
    end_ms: float
    part_index: int
    note_index: int
    position_ticks: int
    duration_ticks: int


def parse_ustx_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"USTx file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"USTx root must be a mapping, got {type(data).__name__}")
    return data


def position_ticks_to_ms(
    ticks: int,
    *,
    resolution: int,
    tempos: list[dict[str, Any]],
    base_bpm: float,
) -> float:
    """Convert a tick position to absolute ms, honoring tempo changes.

    Tempo entries are inserted at position 0 with `base_bpm` if missing, so
    files with only an initial top-level `bpm` work without explicit tempos.
    """
    if int(resolution) <= 0:
        raise ValueError("resolution must be > 0")
    sorted_tempos = sorted(
        list(tempos or []),
        key=lambda t: int(t.get("position", 0) or 0),
    )
    if not sorted_tempos or int(sorted_tempos[0].get("position", 0) or 0) > 0:
        sorted_tempos = [{"position": 0, "bpm": float(base_bpm)}, *sorted_tempos]

    ticks_pos = float(ticks)
    ms = 0.0
    last_pos = float(sorted_tempos[0].get("position", 0) or 0)
    last_bpm = float(sorted_tempos[0].get("bpm", base_bpm) or base_bpm)

    for entry in sorted_tempos[1:]:
        t_pos = float(entry.get("position", 0) or 0)
        if ticks_pos < t_pos:
            break
        ms += (t_pos - last_pos) / float(resolution) * 60000.0 / max(1e-6, last_bpm)
        last_pos = t_pos
        last_bpm = float(entry.get("bpm", base_bpm) or base_bpm)

    ms += (ticks_pos - last_pos) / float(resolution) * 60000.0 / max(1e-6, last_bpm)
    return float(ms)


def collect_notes_with_timing(
    ustx: dict[str, Any],
) -> list[NoteTiming]:
    """Flatten every voice part's notes into NoteTiming records with ms timing."""
    resolution = int(ustx.get("resolution", 480) or 480)
    base_bpm = float(ustx.get("bpm", 120) or 120)
    tempos = list(ustx.get("tempos") or [])
    parts = list(ustx.get("voice_parts") or [])

    out: list[NoteTiming] = []
    for part_idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        part_position = int(part.get("position", 0) or 0)
        notes = list(part.get("notes") or [])
        for note_idx, note in enumerate(notes):
            if not isinstance(note, dict):
                continue
            n_pos = int(note.get("position", 0) or 0)
            n_dur = int(note.get("duration", 0) or 0)
            lyric = str(note.get("lyric", "") or "").strip()
            if n_dur <= 0:
                continue
            abs_start = part_position + n_pos
            abs_end = abs_start + n_dur
            start_ms = position_ticks_to_ms(
                abs_start, resolution=resolution, tempos=tempos, base_bpm=base_bpm
            )
            end_ms = position_ticks_to_ms(
                abs_end, resolution=resolution, tempos=tempos, base_bpm=base_bpm
            )
            out.append(
                NoteTiming(
                    lyric=lyric,
                    start_ms=float(start_ms),
                    end_ms=float(end_ms),
                    part_index=int(part_idx),
                    note_index=int(note_idx),
                    position_ticks=int(abs_start),
                    duration_ticks=int(n_dur),
                )
            )
    return out


def index_notes_by_lyric(notes: list[NoteTiming]) -> dict[str, list[NoteTiming]]:
    """Group notes by exact lyric match. Preserves note order within each group."""
    out: dict[str, list[NoteTiming]] = {}
    for note in notes:
        out.setdefault(note.lyric, []).append(note)
    return out


def total_duration_ms(notes: list[NoteTiming]) -> float:
    if not notes:
        return 0.0
    return float(max(n.end_ms for n in notes))


__all__ = [
    "NoteTiming",
    "collect_notes_with_timing",
    "index_notes_by_lyric",
    "parse_ustx_file",
    "position_ticks_to_ms",
    "total_duration_ms",
]
