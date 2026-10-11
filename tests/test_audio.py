from pathlib import Path

import numpy as np
import soundfile as sf

from universal_auto_oto.audio import analyze_wav


def test_activity_region_ignores_isolated_early_noise(tmp_path: Path):
    sample_rate = 16000
    audio = np.zeros(sample_rate * 5, dtype=np.float32)
    audio[int(0.20 * sample_rate) : int(0.23 * sample_rate)] = 0.04
    start = int(1.0 * sample_rate)
    end = int(4.2 * sample_rate)
    time = np.arange(end - start, dtype=np.float32) / sample_rate
    audio[start:end] = 0.12 * np.sin(2.0 * np.pi * 180.0 * time)
    path = tmp_path / "a_i_u_e_o.wav"
    sf.write(path, audio, sample_rate)

    timeline = analyze_wav(path, slot_count=5)

    assert 850.0 <= timeline.active_start_ms <= 1100.0
    assert 4100.0 <= timeline.active_end_ms <= 4350.0


def test_activity_region_spans_repeated_vowels_with_long_gaps(tmp_path: Path):
    sample_rate = 16000
    audio = np.zeros(sample_rate * 6, dtype=np.float32)
    for start_seconds in (1.0, 1.8, 2.6, 3.4, 4.2):
        start = int(start_seconds * sample_rate)
        end = start + int(0.28 * sample_rate)
        time = np.arange(end - start, dtype=np.float32) / sample_rate
        audio[start:end] = 0.12 * np.sin(2.0 * np.pi * 190.0 * time)
    path = tmp_path / "a_i_u_e_o.wav"
    sf.write(path, audio, sample_rate)

    timeline = analyze_wav(path, slot_count=5)

    assert 850.0 <= timeline.active_start_ms <= 1100.0
    assert 4400.0 <= timeline.active_end_ms <= 4650.0
