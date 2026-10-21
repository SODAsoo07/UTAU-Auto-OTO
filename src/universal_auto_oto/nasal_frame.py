"""Frame-level nasal-murmur probability, fitted on the banks' hand-made oto.ini.

A hand-written cue (the local rise of the low-band share) also fired on other voiced consonants and
vowel transitions, so the probability comes from a small model fitted on hand-OTO boundaries:
murmur before nasal-onset CV rows and after nasal-coda VC rows against the same positions of other
voiced onsets and against vowels (run_outputs/nasal_frame_20261021).

Features at each frame time: the spectral shape (log mel bands minus the frame's mean, so level free)
at t-25, t and t+25 ms, the low/mid band balance, and the level relative to the file's loud frames.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from . import analysis_cache

WINDOW_S = 0.030
BANDS = 20
LOW_HZ, HIGH_HZ = 60.0, 6000.0
CONTEXT_MS = 25.0
PROFILE = Path(__file__).resolve().parent / "profiles" / "nasal_frame_v1.json"


def _mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _band_matrix(freqs: np.ndarray) -> np.ndarray:
    edges_mel = np.linspace(_mel(LOW_HZ), _mel(HIGH_HZ), BANDS + 2)
    edges = 700.0 * (10.0 ** (edges_mel / 2595.0) - 1.0)
    matrix = np.zeros((BANDS, len(freqs)))
    for b in range(BANDS):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        rise = (freqs - lo) / max(mid - lo, 1e-9)
        fall = (hi - freqs) / max(hi - mid, 1e-9)
        matrix[b] = np.clip(np.minimum(rise, fall), 0.0, None)
    return matrix


def _spectra(mono: np.ndarray, rate: int, times_ms: np.ndarray):
    window = max(16, int(round(WINDOW_S * rate)))
    padded = np.concatenate((np.zeros(window), mono, np.zeros(window)))
    starts = np.clip(np.round(np.asarray(times_ms) / 1000.0 * rate).astype(int) - window // 2 + window,
                     0, len(padded) - window)
    frames = padded[starts[:, None] + np.arange(window)[None, :]] * np.hanning(window)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return power, np.fft.rfftfreq(window, 1.0 / rate)


def _shape(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    bands = np.log(power @ _band_matrix(freqs).T + 1e-10)
    return bands - bands.mean(axis=1, keepdims=True)


def frame_features(wav_path: Path, times_ms: np.ndarray) -> np.ndarray:
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    times = np.asarray(times_ms, dtype=np.float64)
    power, freqs = _spectra(mono, rate, times)
    total = power.sum(axis=1) + 1e-12
    balance = (power[:, freqs < 400.0].sum(axis=1) - power[:, (freqs >= 800.0) & (freqs < 3000.0)].sum(axis=1)) / total
    level = 10.0 * np.log10(total)
    grid = np.arange(0.0, len(mono) / rate * 1000.0, 10.0)
    grid_power, _ = _spectra(mono, rate, grid)
    reference = np.percentile(10.0 * np.log10(grid_power.sum(axis=1) + 1e-12), 95) if len(grid) else 0.0
    shape = _shape(power, freqs)
    before = _shape(_spectra(mono, rate, times - CONTEXT_MS)[0], freqs)
    after = _shape(_spectra(mono, rate, times + CONTEXT_MS)[0], freqs)
    relative = np.clip(level - reference, -60.0, 10.0) / 20.0
    return np.column_stack((shape, before, after, balance, relative))


@lru_cache(maxsize=1)
def _model():
    data = json.loads(PROFILE.read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.float64) if isinstance(value, list) else value
            for key, value in data.items()}


def model_available() -> bool:
    return PROFILE.is_file()


def nasal_probability(wav_path: Path, times_ms: np.ndarray) -> np.ndarray:
    times = np.asarray(times_ms, dtype=np.float64)
    return analysis_cache.cached("nasal_frame_v1", wav_path, (times.tobytes(),),
                                 lambda: _nasal_probability_uncached(wav_path, times))


def _nasal_probability_uncached(wav_path: Path, times_ms: np.ndarray) -> np.ndarray:
    model = _model()
    x = (frame_features(wav_path, times_ms) - model["mean"]) / model["std"]
    hidden = np.tanh(x @ model["w1"] + model["b1"])
    logit = hidden @ model["w2"] + float(model["b2"])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
