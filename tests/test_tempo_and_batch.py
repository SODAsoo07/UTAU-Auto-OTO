from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from universal_auto_oto.batch import discover_bank_folders, run_batch
from universal_auto_oto.tempo import beats_for_period, estimate_tempo, period_to_bpm, syllable_nuclei_ms

RATE = 22050


def _syllables(path: Path, onsets_ms: list[float], length_ms: float = 260.0, total_ms: float | None = None) -> None:
    total = total_ms or onsets_ms[-1] + 900.0
    signal = np.zeros(int(RATE * total / 1000.0))
    t = np.arange(int(RATE * length_ms / 1000.0)) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 700.0 * t) * np.hanning(len(t))
    for onset in onsets_ms:
        start = int(RATE * onset / 1000.0)
        signal[start:start + len(tone)] += tone
    sf.write(path, signal, RATE)


def test_period_to_bpm_folds_into_usual_range() -> None:
    assert period_to_bpm(500.0) == (120.0, 1.0)
    bpm, per_beat = period_to_bpm(1000.0)
    assert bpm == 120.0 and per_beat == 0.5
    assert beats_for_period(250.0, 120.0) == 2.0


def test_nuclei_follow_steady_syllables(tmp_path: Path) -> None:
    path = tmp_path / "ka_ki_ku_ke.wav"
    _syllables(path, [300.0, 800.0, 1300.0, 1800.0])
    nuclei = syllable_nuclei_ms(path, 4)
    assert nuclei is not None
    assert np.median(np.diff(nuclei)) == pytest.approx(500.0, abs=15.0)
    assert syllable_nuclei_ms(path, 6) is None


def test_estimate_tempo_from_steady_files(tmp_path: Path) -> None:
    wavs = []
    for index in range(6):
        path = tmp_path / f"f{index}.wav"
        _syllables(path, [200.0 + 500.0 * k for k in range(5)])
        wavs.append(path)
    estimate = estimate_tempo(wavs, lambda wav: 5)
    assert estimate.ok
    assert estimate.bpm == pytest.approx(120.0, abs=4.0)


def test_estimate_tempo_refuses_irregular_recordings(tmp_path: Path) -> None:
    wavs = []
    for index in range(6):
        path = tmp_path / f"f{index}.wav"
        _syllables(path, [200.0 + (400.0 + 120.0 * index) * k for k in range(5)])
        wavs.append(path)
    estimate = estimate_tempo(wavs, lambda wav: 5)
    assert not estimate.ok
    assert estimate.bpm is None


def test_discover_folders_and_batch_continue_after_failure(tmp_path: Path) -> None:
    root = tmp_path / "Bank (JP VCV)"
    for name in ("A4", "C4"):
        (root / name).mkdir(parents=True)
        for stem in ("_ああいあうえあ", "_かかきかくかけ"):
            _syllables(root / name / f"{stem}.wav", [200.0, 700.0])
    (root / "readme").mkdir()
    folders = discover_bank_folders(root, language="japanese")
    assert [folder.relative for folder in folders] == ["A4", "C4"]
    assert {folder.format for folder in folders} == {"vcv"}

    def generate_one(folder, output):
        if folder.relative == "A4":
            raise RuntimeError("boom")
        return {"generated_rows": 7}

    summary = run_batch(folders, root=root, output_root=tmp_path / "out", in_place=False, generate_one=generate_one)
    assert summary["succeeded"] == 1 and summary["failed"] == 1
    assert summary["generated_rows"] == 7
    assert summary["results"][1]["output"] == str(tmp_path / "out" / "C4")
