"""Frame voicing shared by the ending-voicing offsets and the row risk model.

A frame (30 ms window, 5 ms hop, mean removed) is voiced when its normalised autocorrelation
peak for F0 70-600 Hz is at least 0.5 and its level is within 45 dB of the file's loudest
frame. The autocorrelation (two FFTs per frame) is computed only for the frames asked for and
remembered per file while ``analysis_cache`` is active, so later identity passes and the row
risk model reuse it. Every frame is computed exactly as in the whole-file version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import analysis_cache


class FrameVoicing:
    def __init__(self, wav_path: Path) -> None:
        self.wav_path = Path(wav_path)
        mono, rate = self._read()
        self.rate = rate
        self.hop = max(1, int(round(rate * 0.005)))
        self.window = max(4, int(round(rate * 0.030)))
        frames = self._frames(mono)
        level = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
        self.loud = level > level.max() - 45.0 if len(level) else np.zeros(0, dtype=bool)
        self.times = (np.arange(len(frames)) * self.hop + self.window / 2.0) / rate * 1000.0
        self.sample_count = len(mono)
        self._voiced = np.zeros(len(frames), dtype=bool)
        self._known = np.zeros(len(frames), dtype=bool)

    def _read(self) -> tuple[np.ndarray, int]:
        samples, rate = sf.read(str(self.wav_path), dtype="float64", always_2d=True)
        return samples.mean(axis=1), rate

    def _frames(self, mono: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(mono, self.window)[::self.hop]
        return frames - frames.mean(axis=1, keepdims=True)

    def voiced(self, needed: np.ndarray | None = None) -> np.ndarray:
        """The voicing array; frames not in ``needed`` (all when None) may still be False/unknown."""
        indices = np.arange(len(self._voiced)) if needed is None else np.asarray(needed, dtype=np.int64)
        missing = indices[~self._known[indices]] if len(indices) else indices
        if len(missing):
            subset = self._frames(self._read()[0])[missing]
            spectrum = np.fft.rfft(subset, n=2 * self.window, axis=1)
            autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :self.window]
            lo, hi = max(1, int(self.rate / 600.0)), min(self.window - 1, int(self.rate / 70.0))
            strength = autocorr[:, lo:hi].max(axis=1) / (autocorr[:, 0] + 1e-12)
            self._voiced[missing] = (strength >= 0.5) & self.loud[missing]
            self._known[missing] = True
        return self._voiced


def frame_voicing(wav_path: Path) -> FrameVoicing:
    return analysis_cache.state("frame_voicing", wav_path, lambda: FrameVoicing(wav_path))


__all__ = ["FrameVoicing", "frame_voicing"]
