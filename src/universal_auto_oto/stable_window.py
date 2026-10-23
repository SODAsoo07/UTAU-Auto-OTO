"""Keep an OTO row's fixed and stretch span on audio whose level and pitch hold steady.

UTAU plays [offset, fixed] once and loops/stretches [fixed, cutoff]. The templates end a
vowel row at the decoded vowel end, so the stretch span can take in the decay, room
silence, or a wobble of the voice, and every long note repeats it. Hand OTOs keep it
about 150-250 ms (2026-10-23, 70k rows: generated CV spans ran 2-3x longer).

- Vowel-ending rows (V, CV, VV, VCV): fixed moves to where the onset's level/pitch change
  settles; cutoff stops before the first sustained wobble, level drop, or loss of voicing.
- VC rows: the preceding vowel keeps only its last fifth (the user's reference keeps
  ~70 ms of a ~350 ms vowel); a held sonorant coda (n, m, ng, l) gets the same cutoff rule.

Rows are only ever shortened: offset never moves earlier, cutoff never later.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import analysis_cache
from .domain import RenderedOtoRow

VOWEL_END_ROLES = frozenset({"v", "v_head", "v_blend", "cv", "cv_head", "cvvc", "vv", "vcv"})
VC_ROLES = frozenset({"vc"})
SONORANT_CODAS = frozenset({"n", "m", "ng", "l", "N", "M", "NG", "L", "nn", "ll", "r"})

SAMPLE_RATE = 16000
HOP_MS = 5.0
WINDOW_MS = 40.0
SPAN_FRAMES = 13  # ~60 ms window for the local wobble measures
F0_FLOOR_HZ, F0_CEILING_HZ = 70.0, 700.0


@dataclass(frozen=True)
class StableWindowSettings:
    f0_range_cents: float = 60.0  # p90-p10 of pitch inside a ~60 ms window
    level_range_db: float = 4.0  # max-min of smoothed level inside the window
    level_drop_db: float = 9.0  # below the row's vowel peak level
    f0_drift_cents: float = 90.0  # from the pitch where the stable part starts
    level_drift_db: float = 6.0
    # Timbre settles when the cepstral change drops to ~1.6x the vowel's own steady value
    # (hand fixed ~6.5, mid stretch ~4.6, 20 ms after pre ~14; 600 hand rows).
    spectral_settle_ratio: float = 1.6
    spectral_settle_min: float = 5.0
    spectral_settle_max: float = 9.0
    spectral_hold_ratio: float = 2.5  # the stretch span ends at a larger timbre change
    settle_ms: float = 30.0  # stable this long before fixed counts as settled
    fixed_min_ms: float = 20.0  # fixed at least this far after pre
    fixed_max_ms: float = 180.0
    gap_ms: float = 25.0  # shorter unstable blips are bridged
    stretch_min_ms: float = 60.0
    cutoff_margin_ms: float = 5.0
    vc_vowel_share: float = 0.2
    vc_lead_min_ms: float = 50.0
    vc_lead_max_ms: float = 160.0


DEFAULT_SETTINGS = StableWindowSettings()


@dataclass(frozen=True)
class StabilityTrack:
    times_ms: np.ndarray
    level_db: np.ndarray
    voiced: np.ndarray
    cents: np.ndarray  # 0 where unvoiced
    f0_range: np.ndarray  # local p90-p10 cents (inf where too little voicing)
    level_range: np.ndarray
    spectral_change: np.ndarray  # mel-cepstral distance across +-10 ms (formant movement)

    def frame(self, time_ms: float) -> int:
        return int(np.clip(round((time_ms - self.times_ms[0]) / HOP_MS), 0, len(self.times_ms) - 1))


def stability_track(wav_path: Path) -> StabilityTrack:
    return analysis_cache.cached("stable_window_track", Path(wav_path), (SAMPLE_RATE, HOP_MS, WINDOW_MS),
                                 lambda: _compute_track(Path(wav_path)))


def _compute_track(wav_path: Path) -> StabilityTrack:
    samples, rate = sf.read(str(wav_path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, int(rate) // divisor)
    hop = int(SAMPLE_RATE * HOP_MS / 1000.0)
    window = int(SAMPLE_RATE * WINDOW_MS / 1000.0)
    if len(mono) < window:
        mono = np.pad(mono, (0, window - len(mono)))
    frames = np.lib.stride_tricks.sliding_window_view(mono, window)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    level = 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    f0, strength = _pitch(frames)
    voiced = (strength >= 0.45) & (level > level.max() - 45.0)
    cents = np.where(voiced, _octave_folded(1200.0 * np.log2(np.maximum(f0, 1.0) / 55.0), voiced), 0.0)
    smooth = np.convolve(level, np.ones(5) / 5.0, mode="same")
    times = (np.arange(len(frames)) * hop + window / 2.0) * 1000.0 / SAMPLE_RATE
    return StabilityTrack(times, smooth, voiced, cents, *_local_ranges(cents, voiced, smooth),
                          _spectral_change(frames))


def _spectral_change(frames: np.ndarray) -> np.ndarray:
    window = frames.shape[1]
    power = np.abs(np.fft.rfft(frames * np.hanning(window), axis=1)) ** 2
    freqs = np.fft.rfftfreq(window, 1.0 / SAMPLE_RATE)
    mel = 2595.0 * np.log10(1.0 + freqs / 700.0)
    edges = np.linspace(2595.0 * np.log10(1.0 + 80.0 / 700.0), 2595.0 * np.log10(1.0 + 6000.0 / 700.0), 28)
    bank = np.maximum(0.0, 1.0 - np.abs(mel[None, :] - edges[1:-1, None]) / (edges[1] - edges[0]))
    log_mel = np.log(power @ bank.T + 1e-10)
    count = log_mel.shape[1]
    dct = np.cos(np.pi / count * (np.arange(count)[None, :] + 0.5) * np.arange(1, 13)[:, None])
    cepstra = log_mel @ dct.T
    change = np.zeros(len(frames))
    if len(frames) > 4:
        change[2:-2] = np.linalg.norm(cepstra[4:] - cepstra[:-4], axis=1)
    return np.convolve(change, np.ones(3) / 3.0, mode="same")


def _pitch(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = frames.shape[1]
    lo = max(1, int(SAMPLE_RATE / F0_CEILING_HZ))
    hi = min(window - 2, int(SAMPLE_RATE / F0_FLOOR_HZ))
    f0 = np.zeros(len(frames))
    strength = np.zeros(len(frames))
    taper = np.hanning(window)
    for start in range(0, len(frames), 2048):
        chunk = frames[start:start + 2048] * taper
        spectrum = np.fft.rfft(chunk, n=2 * window, axis=1)
        autocorr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :window]
        autocorr /= np.maximum(autocorr[:, :1], 1e-12)
        search = autocorr[:, lo:hi + 1]
        best = np.argmax(search, axis=1)
        peak = search[np.arange(len(chunk)), best]
        lag = best + lo
        # A subharmonic (twice the period) often wins by a hair; prefer the half lag when close.
        half = lag // 2
        half_ok = half >= lo
        half_peak = np.where(half_ok, autocorr[np.arange(len(chunk)), np.maximum(half, lo)], 0.0)
        use_half = half_ok & (half_peak >= 0.85 * peak)
        lag = np.where(use_half, half, lag)
        # Integer lags are ~30 cents apart at 300 Hz; refine with a parabola through the peak.
        rows = np.arange(len(chunk))
        left = autocorr[rows, np.maximum(lag - 1, 0)]
        mid = autocorr[rows, lag]
        right = autocorr[rows, np.minimum(lag + 1, window - 1)]
        curve = left - 2.0 * mid + right
        shift = np.zeros(len(chunk))
        np.divide(0.5 * (left - right), curve, out=shift, where=np.abs(curve) > 1e-9)
        refined = lag + np.clip(shift, -0.5, 0.5)
        f0[start:start + len(chunk)] = SAMPLE_RATE / np.maximum(refined, 1.0)
        strength[start:start + len(chunk)] = np.clip(np.where(use_half, half_peak, peak), 0.0, 1.0)
    return f0, strength


def _octave_folded(cents: np.ndarray, voiced: np.ndarray, frames: int = 21) -> np.ndarray:
    """Move octave errors of the autocorrelation pick back to the octave of the local median."""
    if len(cents) < frames or not voiced.any():
        return cents
    pad = frames // 2
    padded = np.pad(np.where(voiced, cents, np.nan), pad, constant_values=np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are unvoiced
        median = np.nanmedian(np.lib.stride_tricks.sliding_window_view(padded, frames), axis=1)
    median = np.where(np.isfinite(median), median, cents)
    return cents - 1200.0 * np.round((cents - median) / 1200.0)


def _local_ranges(cents: np.ndarray, voiced: np.ndarray, level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = SPAN_FRAMES // 2
    count = len(cents)
    f0_range = np.full(count, np.inf)
    level_range = np.full(count, np.inf)
    if count < SPAN_FRAMES:
        return f0_range, level_range
    c = np.lib.stride_tricks.sliding_window_view(np.where(voiced, cents, np.nan), SPAN_FRAMES)
    v = np.lib.stride_tricks.sliding_window_view(voiced, SPAN_FRAMES)
    lv = np.lib.stride_tricks.sliding_window_view(level, SPAN_FRAMES)
    enough = v.mean(axis=1) >= 0.8
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are unvoiced
        spread = np.nanpercentile(c, 90, axis=1) - np.nanpercentile(c, 10, axis=1)
    f0_range[pad:pad + len(spread)] = np.where(enough, spread, np.inf)
    level_range[pad:pad + len(spread)] = lv.max(axis=1) - lv.min(axis=1)
    return f0_range, level_range


def steady_span(track: StabilityTrack, pre_abs: float, fixed_abs: float, cutoff_abs: float,
                settings: StableWindowSettings = DEFAULT_SETTINGS) -> tuple[float, float, str]:
    """(fixed, cutoff, reason) for a vowel stretch between pre and cutoff."""
    first, last = track.frame(pre_abs), track.frame(cutoff_abs)
    if last - first < 8:
        return fixed_abs, cutoff_abs, "too_short"
    level = track.level_db[first:last + 1]
    voiced = track.voiced[first:last + 1]
    if voiced.mean() < 0.3:
        return fixed_abs, cutoff_abs, "unvoiced"
    peak = float(np.percentile(level[voiced], 90))
    local = ((track.f0_range[first:last + 1] <= settings.f0_range_cents)
             & (track.level_range[first:last + 1] <= settings.level_range_db)
             & (level >= peak - settings.level_drop_db) & voiced)
    spectral = track.spectral_change[first:last + 1]
    steady_spectral = float(np.median(spectral[local])) if local.any() else float(np.median(spectral))
    # Noisy takes have a high steady value of their own; the cap gives way to 1.15x of it.
    settle_limit = max(min(settings.spectral_settle_ratio * steady_spectral, settings.spectral_settle_max),
                       1.15 * steady_spectral, settings.spectral_settle_min)
    settled = local & (spectral <= settle_limit)
    local = local & (spectral <= settings.spectral_hold_ratio * max(steady_spectral, 1.0))
    settle = max(1, int(settings.settle_ms / HOP_MS))
    lo = int(settings.fixed_min_ms / HOP_MS)
    hi = min(len(local) - settle, int(settings.fixed_max_ms / HOP_MS))
    start = next((i for i in range(lo, hi + 1) if settled[i:i + settle].all()), None)
    gap = int(settings.gap_ms / HOP_MS)
    if start is None:
        # A take that wobbles throughout keeps its fixed; only the decay and silence are cut.
        sounding = voiced & (level >= peak - settings.level_drop_db)
        end = _run_end(sounding, max(track.frame(fixed_abs) - first, 0), gap)
        new_cutoff = max(float(track.times_ms[first + end]) - settings.cutoff_margin_ms,
                         fixed_abs + settings.stretch_min_ms)
        return fixed_abs, min(new_cutoff, cutoff_abs), "no_stable_onset"
    ref_cents = float(np.median(track.cents[first + start:first + start + settle]))
    ref_level = float(np.median(level[start:start + settle]))
    steady = (local
              & (np.abs(track.cents[first:last + 1] - ref_cents) <= settings.f0_drift_cents)
              & (np.abs(level - ref_level) <= settings.level_drift_db))
    end = _run_end(steady, start, gap)
    new_fixed = float(track.times_ms[first + start])
    new_cutoff = float(track.times_ms[first + end]) - settings.cutoff_margin_ms
    new_cutoff = max(new_cutoff, new_fixed + settings.stretch_min_ms)
    reason = "stable" if end < len(steady) - 1 else "stable_to_end"
    return new_fixed, min(new_cutoff, cutoff_abs), reason


def _run_end(mask: np.ndarray, begin: int, gap: int) -> int:
    """Last True frame of the run from begin, bridging False gaps of up to gap frames."""
    end, run_gap = begin, 0
    for i in range(begin, len(mask)):
        if mask[i]:
            end, run_gap = i, 0
        else:
            run_gap += 1
            if run_gap > gap:
                break
    return end


def apply_stable_window(row: RenderedOtoRow, wav_path: Path, *, vowel_start_ms: float | None = None,
                        settings: StableWindowSettings = DEFAULT_SETTINGS) -> tuple[RenderedOtoRow, dict]:
    """The row with a steadier fixed/cutoff (and VC vowel lead); info for the row report."""
    if row.cutoff > 0.0:
        return row, {"applied": False, "reason": "file_end_cutoff"}
    parts = row.alias.split()
    offset, pre = row.offset, row.offset + row.preutterance
    overlap, fixed, cutoff = row.offset + row.overlap, row.offset + row.consonant, row.offset - row.cutoff
    info: dict[str, object] = {"applied": False}
    if row.role in VOWEL_END_ROLES:
        new_fixed, new_cutoff, reason = steady_span(stability_track(wav_path), pre, fixed, cutoff, settings)
        info["reason"] = reason
        fixed, cutoff = max(pre, new_fixed), max(min(new_cutoff, cutoff), pre + 5.0)
    elif row.role in VC_ROLES:
        vowel_start = vowel_start_ms if vowel_start_ms is not None else offset
        lead = min(max(settings.vc_vowel_share * (pre - vowel_start), settings.vc_lead_min_ms),
                   settings.vc_lead_max_ms)
        if pre - offset > lead:
            offset = pre - lead
            overlap = offset + 0.5 * lead
            info["vc_lead_ms"] = lead
        info["reason"] = "vc_lead"
        if len(parts) == 2 and parts[1] in SONORANT_CODAS and cutoff - pre > 80.0:
            new_fixed, new_cutoff, reason = steady_span(stability_track(wav_path), pre, fixed, cutoff, settings)
            info["reason"] = f"vc_lead+coda_{reason}"
            if reason.startswith("stable"):
                fixed, cutoff = max(pre, min(new_fixed, fixed)), max(min(new_cutoff, cutoff), pre + 5.0)
    else:
        return row, {"applied": False, "reason": "role"}
    fixed = min(max(fixed, pre), cutoff)
    overlap = min(max(offset, overlap), pre)
    changed = any(abs(a - b) > 0.5 for a, b in (
        (offset, row.offset), (fixed, row.offset + row.consonant), (cutoff, row.offset - row.cutoff)))
    if not changed:
        return row, info
    info["applied"] = True
    absolute = dict(row.absolute)
    absolute.update({"offset_abs": offset, "overlap_abs": overlap, "pre_abs": pre,
                     "fixed_end_abs": fixed, "cutoff_abs": cutoff})
    return replace(row, offset=offset, overlap=overlap - offset, preutterance=pre - offset,
                   consonant=fixed - offset, cutoff=-(cutoff - offset), absolute=absolute), info


__all__ = ["DEFAULT_SETTINGS", "StableWindowSettings", "StabilityTrack", "apply_stable_window", "stability_track",
           "steady_span"]
