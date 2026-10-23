"""Acoustic landmarks added to language-pack timelines after alignment.

The HSMM puts a vowel's end where its states hand over, which drifts into the
silent closure of a following stop.  ``vowel_offset`` is the steepest energy
fall near that end: where the voice actually stops.  Templates read it as
``phone[i].vowel_offset`` with a fallback to ``phone[i].end``.

After a nasal, liquid or glide the HSMM tends to start the vowel inside the
sonorant (a held word-initial murmur is scored as vowel).  ``vowel_onset`` is
where the low band stops dominating and the 0.8-3 kHz band rises: the
sonorant-to-vowel release.  On three hand-timed English VCCV banks the hand
preutterance sat at median 0 ms from it (72-93% within 30 ms) against
32-70% for the aligned boundary.  Only recordings
routed through a language pack get these landmarks, so the Korean/Japanese
routes are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from .domain import Landmark, LandmarkTimeline, PhoneDefinition

PACK_LANDMARK_VERSION = "pack_landmarks_v5"
# Sub-bass rumble (mic bumps, wind) is loud in the RMS but carries no speech.
PACK_RUMBLE_HIGHPASS_HZ = 70.0
HOP_MS = 2.5
WINDOW_MS = 20.0
HALF_MS = 15.0
SEARCH_BEFORE_MS = 120.0
SEARCH_AFTER_MS = 60.0
MIN_VOWEL_MS = 40.0
MIN_FALL_DB = 6.0
# A drop this far below the vowel's own level that lasts SUSTAIN_MS ends the
# vowel even when the aligner kept a following stop closure inside it.
DEEP_DROP_DB = 15.0
SUSTAIN_MS = 30.0
# vowel_onset after a sonorant
SPECTRUM_WINDOW_MS = 25.0
ONSET_HALF_MS = 12.0
ONSET_AFTER_VOWEL_MS = 200.0
ONSET_VOICING_DB = 12.0       # "voiced" = within this of the local peak energy
ONSET_AFTER_VOICING_MS = 25.0  # the release comes after the murmur has started


def add_pack_landmarks(
    timeline: LandmarkTimeline,
    wav_path: Path,
    inventory: Mapping[str, PhoneDefinition],
) -> LandmarkTimeline:
    if timeline.provenance.get("pack_landmarks") == PACK_LANDMARK_VERSION:
        return timeline
    energy = _frame_energy_db(wav_path)
    bands = None
    segments = sorted(timeline.phone_segments, key=lambda item: item.phone_index)
    added: list[Landmark] = []
    for position, segment in enumerate(segments):
        if not _is_nucleus(inventory.get(segment.phone_id)):
            continue
        preceding = segments[position - 1] if position else None
        if preceding is not None and _is_sonorant_consonant(inventory.get(preceding.phone_id)):
            if bands is None:
                bands = _band_tracks_db(wav_path)
            onset = _sonorant_release(bands, preceding.start_ms + 15.0, segment.start_ms + ONSET_AFTER_VOWEL_MS)
            if onset is not None:
                added.append(Landmark(
                    label="vowel_onset", time_ms=onset[0], phone_index=segment.phone_index,
                    confidence=float(min(1.0, onset[1] / 40.0)), source="pack_sonorant_release",
                    raw_time_ms=segment.start_ms, refinement_reason=f"band_change_{onset[1]:.1f}dB",
                ))
        following = segments[position + 1] if position + 1 < len(segments) else None
        # Into a vowel or a sonorant (l, m, n, r ...) the voice goes on; no fall marks the end.
        if following is not None and not _is_obstruent(inventory.get(following.phone_id)):
            continue
        upper = segment.end_ms + SEARCH_AFTER_MS
        if following is not None:
            upper = min(upper, following.end_ms)
        found = _deep_drop(energy, segment.start_ms + MIN_VOWEL_MS, upper)
        if found is None:
            lower = max(segment.start_ms + MIN_VOWEL_MS, segment.end_ms - SEARCH_BEFORE_MS)
            found = _steepest_fall(energy, lower, upper)
        if found is None:
            continue
        time_ms, fall_db = found
        added.append(Landmark(
            label="vowel_offset",
            time_ms=time_ms,
            phone_index=segment.phone_index,
            confidence=float(min(1.0, fall_db / 20.0)),
            source="pack_energy_fall",
            raw_time_ms=segment.end_ms,
            refinement_reason=f"energy_fall_{fall_db:.1f}dB",
        ))
    landmarks = tuple(sorted((*timeline.landmarks, *added), key=lambda item: (
        item.time_ms, item.phone_index if item.phone_index is not None else -1, item.label)))
    provenance = {**dict(timeline.provenance), "pack_landmarks": PACK_LANDMARK_VERSION,
                  "pack_vowel_offsets": sum(item.label == "vowel_offset" for item in added),
                  "pack_vowel_onsets": sum(item.label == "vowel_onset" for item in added)}
    return replace(timeline, landmarks=landmarks, provenance=provenance)


def _is_obstruent(definition: PhoneDefinition | None) -> bool:
    return bool(definition and not _is_nucleus(definition) and not definition.features.get("sonorant")
                and definition.features.get("manner") in {"stop", "affricate", "fricative"})


def _is_sonorant_consonant(definition: PhoneDefinition | None) -> bool:
    return bool(definition and not _is_nucleus(definition) and definition.features.get("sonorant"))


def _is_nucleus(definition: PhoneDefinition | None) -> bool:
    return bool(definition and (definition.features.get("syllabic") or definition.features.get("vowel_like")))


def _read_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Mono samples without sub-bass rumble, matching the aligner's pack evidence."""
    samples, sample_rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    sos = butter(4, PACK_RUMBLE_HIGHPASS_HZ, "highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, samples.mean(axis=1)), int(sample_rate)


def _frame_energy_db(wav_path: Path) -> np.ndarray:
    mono, sample_rate = _read_mono(wav_path)
    window = max(1, int(sample_rate * WINDOW_MS / 1000.0))
    hop = max(1, int(sample_rate * HOP_MS / 1000.0))
    padded = np.pad(mono, (window // 2, window))
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop]
    power = np.mean((frames * np.hanning(window)) ** 2, axis=1)
    return 10.0 * np.log10(power + 1e-10)


def _band_tracks_db(wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low<400 Hz minus 0.8-3 kHz, 0.8-3 kHz, total) in dB per HOP_MS frame."""
    mono, sample_rate = _read_mono(wav_path)
    window = max(8, int(sample_rate * SPECTRUM_WINDOW_MS / 1000.0))
    hop = max(1, int(sample_rate * HOP_MS / 1000.0))
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(mono, (window // 2, window)), window)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(window), axis=1)) ** 2
    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    low = 10.0 * np.log10(spectrum[:, freqs < 400.0].sum(axis=1) + 1e-10)
    mid = 10.0 * np.log10(spectrum[:, (freqs >= 800.0) & (freqs < 3000.0)].sum(axis=1) + 1e-10)
    total = 10.0 * np.log10(spectrum.sum(axis=1) + 1e-10)
    return low - mid, mid, total


def _sonorant_release(bands, lower_ms: float, upper_ms: float) -> tuple[float, float] | None:
    """Largest (low/mid ratio fall + mid-band rise) once the sonorant is voiced."""
    ratio, mid, total = bands
    start, stop = int(max(0.0, lower_ms - 65.0) / HOP_MS), min(len(total), int(upper_ms / HOP_MS))
    if stop <= start:
        return None
    voiced = np.flatnonzero(total[start:stop] > total[start:stop].max() - ONSET_VOICING_DB)
    if len(voiced):
        lower_ms = max(lower_ms, (start + voiced[0]) * HOP_MS + ONSET_AFTER_VOICING_MS)
    half = int(ONSET_HALF_MS / HOP_MS)
    first, last = max(half, int(lower_ms / HOP_MS)), min(len(ratio) - half - 1, int(upper_ms / HOP_MS))
    if last <= first:
        return None
    ratio_sum = np.concatenate(([0.0], np.cumsum(ratio)))
    mid_sum = np.concatenate(([0.0], np.cumsum(mid)))
    centers = np.arange(first, last)
    fall = (ratio_sum[centers] - ratio_sum[centers - half] - ratio_sum[centers + half] + ratio_sum[centers]) / half
    rise = (mid_sum[centers + half] - mid_sum[centers] - mid_sum[centers] + mid_sum[centers - half]) / half
    score = fall + rise
    best = int(np.argmax(score))
    return float(centers[best] * HOP_MS), float(score[best])


def _deep_drop(energy: np.ndarray, lower_ms: float, upper_ms: float) -> tuple[float, float] | None:
    """First sustained fall DEEP_DROP_DB below the vowel's level, refined to its steepest point."""
    first, last = int(lower_ms / HOP_MS), min(len(energy), int(upper_ms / HOP_MS))
    if last - first < 8:
        return None
    level = float(np.median(energy[first:first + max(4, (last - first) // 3)]))
    sustain = int(SUSTAIN_MS / HOP_MS)
    below = energy[first:last] < level - DEEP_DROP_DB
    for index in range(len(below) - sustain + 1):
        if below[index:index + sustain].all():
            crossing = (first + index) * HOP_MS
            return _steepest_fall(energy, crossing - 40.0, crossing + 10.0)
    return None


def _steepest_fall(energy: np.ndarray, lower_ms: float, upper_ms: float) -> tuple[float, float] | None:
    half = int(HALF_MS / HOP_MS)
    first = max(half, int(lower_ms / HOP_MS))
    last = min(len(energy) - half - 1, int(upper_ms / HOP_MS))
    if last <= first:
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    centers = np.arange(first, last)
    before = (cumulative[centers] - cumulative[centers - half]) / half
    after = (cumulative[centers + half] - cumulative[centers]) / half
    fall = before - after
    best = int(np.argmax(fall))
    if fall[best] < MIN_FALL_DB:
        return None
    return float(centers[best] * HOP_MS), float(fall[best])


__all__ = ["PACK_LANDMARK_VERSION", "add_pack_landmarks"]
