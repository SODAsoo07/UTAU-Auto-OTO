from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.domain import RenderedOtoRow
from universal_auto_oto.stable_window import apply_stable_window, stability_track, steady_span

RATE = 44100


def _row(role: str, alias: str, offset: float, pre: float, fixed: float, cutoff: float) -> RenderedOtoRow:
    return RenderedOtoRow(
        wav_name="a.wav", alias=alias, offset=offset, consonant=fixed - offset, cutoff=-(cutoff - offset),
        preutterance=pre - offset, overlap=max(0.0, pre - offset - 30.0), role=role, slot_index=0, confidence=1.0,
        split="review", reasons=(), source_index=0,
        absolute={"offset_abs": offset, "overlap_abs": pre - 30.0, "pre_abs": pre, "fixed_end_abs": fixed,
                  "cutoff_abs": cutoff},
    )


def _vowel_with_wobble(path: Path) -> Path:
    """Silence, an 80 ms rising onset at 200 ms, steady 220 Hz to 900 ms, a 6 Hz wobble to 1300 ms, decay to 1500."""
    t = np.arange(int(1.8 * RATE)) / RATE
    cents = np.zeros_like(t)
    gain = np.zeros_like(t)
    onset = (t >= 0.2) & (t < 0.28)
    cents[onset] = -300.0 * (0.28 - t[onset]) / 0.08
    gain[onset] = (t[onset] - 0.2) / 0.08
    gain[(t >= 0.28) & (t < 1.3)] = 1.0
    wobble = (t >= 0.9) & (t < 1.3)
    cents[wobble] = 150.0 * np.sin(2 * np.pi * 6.0 * (t[wobble] - 0.9))
    gain[wobble] *= 10 ** (6.0 * np.sin(2 * np.pi * 6.0 * (t[wobble] - 0.9)) / 20.0)
    decay = (t >= 1.3) & (t < 1.5)
    gain[decay] = 10 ** (-40.0 * (t[decay] - 1.3) / 0.2 / 20.0)
    phase = 2 * np.pi * np.cumsum(220.0 * 2 ** (cents / 1200.0)) / RATE
    voice = sum(np.sin(k * phase) / k for k in range(1, 12))
    noise = np.random.default_rng(0).normal(0.0, 1e-4, len(t))
    sf.write(path, 0.2 * gain * voice + noise, RATE)
    return path


def test_stretch_span_stops_before_the_wobble_and_fixed_waits_for_the_onset(tmp_path: Path) -> None:
    wav = _vowel_with_wobble(tmp_path / "a.wav")
    fixed, cutoff, reason = steady_span(stability_track(wav), 200.0, 230.0, 1600.0)
    assert reason == "stable"
    assert 260.0 <= fixed <= 360.0
    assert 820.0 <= cutoff <= 960.0


def test_vowel_row_is_only_shortened(tmp_path: Path) -> None:
    wav = _vowel_with_wobble(tmp_path / "a.wav")
    row = _row("cv", "ka", 150.0, 200.0, 230.0, 1600.0)
    new, info = apply_stable_window(row, wav)
    assert info["applied"] is True
    assert new.offset == row.offset and new.preutterance == row.preutterance
    assert new.offset - new.cutoff < 1000.0
    assert new.consonant > row.consonant
    # A cutoff counted from the file end is left alone.
    end_row = _row("cv", "ka", 150.0, 200.0, 230.0, 1600.0)
    end_row = RenderedOtoRow(**{**end_row.__dict__, "cutoff": 100.0})
    assert apply_stable_window(end_row, wav)[0] is end_row


def test_vc_keeps_the_last_fifth_of_the_vowel(tmp_path: Path) -> None:
    row = _row("vc", "a k", 140.0, 600.0, 640.0, 660.0)
    new, info = apply_stable_window(row, tmp_path / "unused.wav", vowel_start_ms=100.0)
    assert info["vc_lead_ms"] == 100.0
    assert new.offset == 500.0 and new.preutterance == 100.0 and new.overlap == 50.0
    assert new.offset + new.consonant == 640.0 and new.offset - new.cutoff == 660.0
    # Short vowels keep at least 50 ms, long ones at most 160 ms.
    assert apply_stable_window(row, tmp_path / "unused.wav", vowel_start_ms=450.0)[0].preutterance == 50.0
    assert apply_stable_window(_row("vc", "a k", 0.0, 1500.0, 1540.0, 1560.0), tmp_path / "unused.wav",
                               vowel_start_ms=0.0)[0].preutterance == 160.0


def test_other_roles_are_unchanged(tmp_path: Path) -> None:
    row = _row("breath", "br", 100.0, 150.0, 200.0, 400.0)
    assert apply_stable_window(row, tmp_path / "unused.wav")[0] is row
