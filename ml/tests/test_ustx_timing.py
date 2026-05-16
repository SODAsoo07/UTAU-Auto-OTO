"""Unit tests for ml.scripts.coarse_crnn.ustx_timing (Phase E listening UI)."""
from __future__ import annotations

import os
import sys

import pytest
import yaml


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.scripts.coarse_crnn.ustx_timing import (  # noqa: E402
    NoteTiming,
    collect_notes_with_timing,
    index_notes_by_lyric,
    parse_ustx_file,
    position_ticks_to_ms,
    total_duration_ms,
)


def test_position_ticks_to_ms_single_tempo_uses_base_bpm():
    # 120 BPM, 480 ticks/beat → 1 beat = 500 ms → 480 ticks = 500 ms
    ms = position_ticks_to_ms(480, resolution=480, tempos=[], base_bpm=120)
    assert pytest.approx(ms) == 500.0


def test_position_ticks_to_ms_zero_returns_zero():
    ms = position_ticks_to_ms(0, resolution=480, tempos=[], base_bpm=120)
    assert ms == 0.0


def test_position_ticks_to_ms_uses_tempo_entry_at_position_zero():
    # explicit tempo at position 0 with 140 BPM overrides base_bpm
    tempos = [{"position": 0, "bpm": 140}]
    # 480 ticks at 140 BPM = (480/480) * 60000/140 = 428.571 ms
    ms = position_ticks_to_ms(480, resolution=480, tempos=tempos, base_bpm=120)
    assert pytest.approx(ms, rel=1e-3) == 428.571


def test_position_ticks_to_ms_handles_tempo_change_mid_song():
    # 120 BPM for first 960 ticks (=2 beats=1000 ms), then 60 BPM
    tempos = [{"position": 0, "bpm": 120}, {"position": 960, "bpm": 60}]
    # Position 960 = 1000 ms (still in 120 BPM region)
    ms_at_change = position_ticks_to_ms(960, resolution=480, tempos=tempos, base_bpm=120)
    assert pytest.approx(ms_at_change) == 1000.0
    # Position 1440 = 1000 ms + (480 ticks at 60 BPM = 1 beat = 1000 ms) = 2000 ms
    ms_after = position_ticks_to_ms(1440, resolution=480, tempos=tempos, base_bpm=120)
    assert pytest.approx(ms_after) == 2000.0


def test_position_ticks_to_ms_inserts_base_bpm_when_first_tempo_late():
    # Tempo entry only at tick 960; before that we should use base_bpm
    tempos = [{"position": 960, "bpm": 60}]
    # Position 480 should be (480/480)*60000/120 = 500 ms using base 120
    ms = position_ticks_to_ms(480, resolution=480, tempos=tempos, base_bpm=120)
    assert pytest.approx(ms) == 500.0


def test_position_ticks_to_ms_rejects_invalid_resolution():
    with pytest.raises(ValueError):
        position_ticks_to_ms(100, resolution=0, tempos=[], base_bpm=120)


def test_collect_notes_with_timing_flattens_parts():
    ustx = {
        "resolution": 480,
        "bpm": 120,
        "voice_parts": [
            {
                "position": 0,
                "notes": [
                    {"position": 0, "duration": 480, "lyric": "a"},
                    {"position": 480, "duration": 480, "lyric": "k a"},
                ],
            },
            {
                "position": 1920,
                "notes": [
                    {"position": 0, "duration": 240, "lyric": "eo n"},
                ],
            },
        ],
    }
    notes = collect_notes_with_timing(ustx)
    assert len(notes) == 3
    # First note: 0 ticks → 0 ms ; duration 480 ticks → 500 ms
    assert notes[0].lyric == "a"
    assert pytest.approx(notes[0].start_ms) == 0.0
    assert pytest.approx(notes[0].end_ms) == 500.0
    # Second note: starts at 480 ticks → 500 ms
    assert notes[1].lyric == "k a"
    assert pytest.approx(notes[1].start_ms) == 500.0
    # Third note (second part offset 1920): 1920 ticks = 2000 ms
    assert notes[2].lyric == "eo n"
    assert pytest.approx(notes[2].start_ms) == 2000.0
    assert pytest.approx(notes[2].end_ms) == 2250.0


def test_collect_notes_skips_zero_duration_and_non_dict_entries():
    ustx = {
        "resolution": 480,
        "bpm": 120,
        "voice_parts": [
            {
                "position": 0,
                "notes": [
                    {"position": 0, "duration": 0, "lyric": "skip"},  # zero duration
                    None,  # malformed
                    {"position": 0, "duration": 240, "lyric": "keep"},
                ],
            },
            "not a part",  # malformed part
        ],
    }
    notes = collect_notes_with_timing(ustx)
    assert len(notes) == 1
    assert notes[0].lyric == "keep"


def test_collect_notes_uses_tempos_when_present():
    ustx = {
        "resolution": 480,
        "bpm": 120,
        "tempos": [{"position": 0, "bpm": 60}],
        "voice_parts": [
            {
                "position": 0,
                "notes": [{"position": 0, "duration": 480, "lyric": "x"}],
            },
        ],
    }
    notes = collect_notes_with_timing(ustx)
    # 60 BPM with 480 ticks/beat: 480 ticks = 1 beat = 1000 ms
    assert pytest.approx(notes[0].end_ms) == 1000.0


def test_collect_notes_strips_lyric_whitespace():
    ustx = {
        "resolution": 480,
        "bpm": 120,
        "voice_parts": [
            {"position": 0, "notes": [{"position": 0, "duration": 100, "lyric": "  a k  "}]},
        ],
    }
    notes = collect_notes_with_timing(ustx)
    assert notes[0].lyric == "a k"


def test_index_notes_by_lyric_groups_repeats_in_order():
    notes = [
        NoteTiming(lyric="a", start_ms=0, end_ms=100, part_index=0, note_index=0, position_ticks=0, duration_ticks=100),
        NoteTiming(lyric="b", start_ms=100, end_ms=200, part_index=0, note_index=1, position_ticks=100, duration_ticks=100),
        NoteTiming(lyric="a", start_ms=200, end_ms=300, part_index=0, note_index=2, position_ticks=200, duration_ticks=100),
    ]
    grouped = index_notes_by_lyric(notes)
    assert set(grouped.keys()) == {"a", "b"}
    assert len(grouped["a"]) == 2
    assert grouped["a"][0].start_ms == 0
    assert grouped["a"][1].start_ms == 200


def test_total_duration_ms_returns_max_end():
    notes = [
        NoteTiming(lyric="a", start_ms=0, end_ms=500, part_index=0, note_index=0, position_ticks=0, duration_ticks=480),
        NoteTiming(lyric="b", start_ms=200, end_ms=1500, part_index=0, note_index=1, position_ticks=200, duration_ticks=1248),
        NoteTiming(lyric="c", start_ms=500, end_ms=800, part_index=0, note_index=2, position_ticks=500, duration_ticks=288),
    ]
    assert total_duration_ms(notes) == 1500.0


def test_total_duration_ms_empty():
    assert total_duration_ms([]) == 0.0


def test_parse_ustx_file_reads_yaml(tmp_path):
    path = tmp_path / "x.ustx"
    payload = {"bpm": 100, "resolution": 480, "voice_parts": []}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    out = parse_ustx_file(str(path))
    assert out["bpm"] == 100


def test_parse_ustx_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_ustx_file(str(tmp_path / "nope.ustx"))


def test_parse_ustx_file_rejects_non_mapping(tmp_path):
    path = tmp_path / "bad.ustx"
    path.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_ustx_file(str(path))
