"""Recording tempo (BPM) estimation for list recordings.

Each multi-syllable file is split into syllable nuclei with the 300-3000 Hz
energy envelope (the decoder-independent syllable map used in
docs/BANK_ACCURACY_2026-10-09.md). A file counts only when it shows exactly
one nucleus per filename syllable at a steady pace. The bank tempo is the
median syllable period of those files; it is reported only when enough files
agree, so recordings made without a guide (irregular pace) give no estimate
instead of a wrong one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf


MIN_BPM = 60.0
MAX_BPM = 240.0
TYPICAL_BPM = (100.0, 160.0)
MIN_STEADY_FILES = 5
MAX_PERIOD_SPREAD = 0.12  # interquartile range / median of the per-file periods
MAX_FILE_GAP_CV = 0.25  # gap std / mean inside one file
MAX_ANALYSED_FILES = 80


@dataclass(frozen=True)
class TempoEstimate:
    bpm: float | None
    syllable_period_ms: float | None
    syllables_per_beat: float | None
    steady_files: int
    analysed_files: int
    period_spread: float | None
    reason: str
    file_periods_ms: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.bpm is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "bpm": None if self.bpm is None else round(self.bpm, 1),
            "syllable_period_ms": None if self.syllable_period_ms is None else round(self.syllable_period_ms, 1),
            "syllables_per_beat": self.syllables_per_beat,
            "steady_files": self.steady_files,
            "analysed_files": self.analysed_files,
            "period_spread": None if self.period_spread is None else round(self.period_spread, 3),
            "reason": self.reason,
        }


def bpm_to_period_ms(bpm: float, syllables_per_beat: float = 1.0) -> float:
    return 60000.0 / float(bpm) / float(syllables_per_beat)


def beats_for_period(period_ms: float, bpm: float) -> float:
    """Syllables per beat (0.5, 1, 2 or 4) that best explains a measured syllable period at a given BPM."""

    beat = 60000.0 / float(bpm)
    return min((0.5, 1.0, 2.0, 4.0), key=lambda count: abs(np.log((beat / count) / period_ms)))


def period_to_bpm(period_ms: float) -> tuple[float, float]:
    """Return (bpm, syllables per beat), folding the tempo into the usual 100-160 BPM range."""

    bpm = 60000.0 / period_ms
    per_beat = 1.0
    while bpm < TYPICAL_BPM[0] and per_beat > 0.25:
        bpm *= 2.0
        per_beat /= 2.0
    while bpm > TYPICAL_BPM[1] * 1.25 and per_beat < 4.0:
        bpm /= 2.0
        per_beat *= 2.0
    return bpm, per_beat


def syllable_nuclei_ms(path: Path, expected: int) -> list[float] | None:
    """Times of the syllable nuclei when the file shows exactly ``expected`` steady ones."""

    times, energy = _envelope(path)
    if len(energy) < 5:
        return None
    floor = np.percentile(energy, 20)
    loud = energy > floor + 0.5 * (energy.max() - floor)
    candidates = [
        i for i in range(1, len(energy) - 1)
        if energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1] and loud[i]
    ]
    chosen: list[int] = []
    for dip in (6.0, 5.0, 4.0, 3.0):
        chosen = []
        for index in sorted(candidates, key=lambda item: -energy[item]):
            if any(abs(times[index] - times[other]) < 150.0 for other in chosen):
                continue
            if all(
                energy[min(index, other):max(index, other) + 1].min() <= min(energy[index], energy[other]) - dip
                for other in chosen
            ):
                chosen.append(index)
        if len(chosen) >= expected:
            break
    if len(chosen) != expected:
        return None
    chosen.sort()
    nuclei = [float(times[index]) for index in chosen]
    gaps = np.diff(nuclei)
    if len(gaps) == 0 or gaps.mean() <= 0.0 or gaps.std() / gaps.mean() > MAX_FILE_GAP_CV:
        return None
    return nuclei


def estimate_tempo(
    wavs: Sequence[Path],
    syllable_count: Callable[[Path], int | None],
    *,
    max_files: int = MAX_ANALYSED_FILES,
) -> TempoEstimate:
    """Estimate the bank tempo; ``syllable_count`` gives the filename syllable count (None to skip)."""

    usable = []
    for wav in wavs:
        count = syllable_count(wav)
        if count is not None and count >= 3:
            usable.append((wav, count))
    if not usable:
        return TempoEstimate(None, None, None, 0, 0, None, "여러 음절 목록 파일이 없어 BPM을 추정할 수 없습니다.")
    step = max(1, len(usable) // max_files)
    sample = usable[::step][:max_files]
    periods: dict[str, float] = {}
    for wav, count in sample:
        try:
            nuclei = syllable_nuclei_ms(wav, count)
        except (OSError, RuntimeError, ValueError):
            continue
        if nuclei is not None:
            periods[wav.name] = float(np.median(np.diff(nuclei)))
    values = sorted(periods.values())
    if len(values) < MIN_STEADY_FILES:
        return TempoEstimate(
            None, None, None, len(values), len(sample), None,
            f"박자가 일정한 파일이 {len(values)}개뿐이라 BPM을 추정하지 않았습니다(가이드 없이 녹음한 경우 정상).",
            periods,
        )
    median = statistics.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    spread = float((q3 - q1) / median)
    if spread > MAX_PERIOD_SPREAD:
        return TempoEstimate(
            None, None, None, len(values), len(sample), spread,
            f"파일마다 박자가 달라(편차 {spread:.0%}) BPM을 추정하지 않았습니다.",
            periods,
        )
    bpm, per_beat = period_to_bpm(median)
    if not MIN_BPM <= bpm <= MAX_BPM:
        return TempoEstimate(None, None, None, len(values), len(sample), spread,
                             f"추정 BPM {bpm:.0f}이 범위를 벗어났습니다.", periods)
    return TempoEstimate(
        bpm, median, per_beat, len(values), len(sample), spread,
        f"박자가 일정한 파일 {len(values)}/{len(sample)}개에서 추정했습니다.",
        periods,
    )


def _envelope(path: Path) -> tuple[np.ndarray, np.ndarray]:
    signal, rate = sf.read(str(path), always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    hop, window = int(rate * 0.010), int(rate * 0.040)
    if len(signal) < window + hop:
        return np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(signal, window)[::hop] * np.hanning(window)
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(window, 1.0 / rate)
    band = spectrum[:, (frequencies >= 300.0) & (frequencies < 3000.0)].sum(axis=1)
    energy = np.convolve(10.0 * np.log10(band + 1e-12), np.ones(5) / 5.0, "same")
    times = (np.arange(len(energy)) * hop + window / 2.0) / rate * 1000.0
    return times, energy


__all__ = [
    "TempoEstimate",
    "beats_for_period",
    "bpm_to_period_ms",
    "estimate_tempo",
    "period_to_bpm",
    "syllable_nuclei_ms",
]
