"""Beat-grid prior for syllable anchor placement.

Voicebanks recorded with a fixed-tempo guide BGM have syllable onsets that
fall on beat positions. When the BPM is known (or can be inferred), the
expected anchor times become a strong structural prior that closes the gap
on categories the acoustic boundary scorer struggles with — most notably
vowel-only VV transitions (no transient to lock onto) and coda-bridge VC
(``NG g``/``L H``) where the model places the anchor at the plosive
closure instead of the release burst.

Two-tier estimation:

1. **Filename-derived** (most reliable for CVVC banks). The wav duration
   minus a detected lead-in, divided by the slot count (= mora count from
   the apostrophe-separated filename tokens), gives the syllable interval.
   Multiplied by ``mora_per_beat`` and divided into 60_000 gives BPM.
2. **Acoustic autocorrelation** of an RMS-derived onset envelope. Lag
   search is restricted to the user-hinted BPM range (default 100–150),
   which eliminates the octave error that usually plagues BPM detectors
   on short / sparse signals.

When both estimators agree within ±5 BPM, the wav gets confidence 0.95.
Filename-only → 0.80, acoustic-only → ≤ 0.7. Snap is gated on confidence
≥ ``min_confidence`` so banks recorded without BGM (random IOI → low
autocorr peak, filename estimate out of range) silently skip the snap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.coarse_crnn.audio import load_wav_mono


_DEFAULT_BPM_RANGE: tuple[float, float] = (100.0, 150.0)
_DEFAULT_FRAME_MS: float = 20.0
_DEFAULT_TARGET_SR: int = 16000


@dataclass(frozen=True)
class BpmEstimate:
    """BPM estimate plus the side info needed to lay out a beat grid."""

    bpm: float
    mora_per_beat: int
    lead_in_ms: float
    confidence: float
    source: str  # "manual" | "filename+acoustic" | "filename" | "acoustic"


def detect_lead_in_ms(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float = _DEFAULT_FRAME_MS,
    floor_percentile: float = 5.0,
    above_floor_ratio: float = 3.0,
    absolute_floor_db: float = -40.0,
) -> float:
    """Return the time (ms) of the first frame whose RMS exceeds a noise floor.

    Threshold = max(5th-percentile RMS × 3, 10 ** (−40 dB / 20)). The
    multiplicative floor catches recordings where the silent region is not
    truly silent (room tone) and the absolute floor prevents very-quiet
    wavs from triggering on hiss.
    """
    if audio.size == 0 or sr <= 0:
        return 0.0
    frame_samples = max(1, int(round(sr * frame_ms / 1000.0)))
    if audio.size < frame_samples * 2:
        return 0.0
    n_frames = audio.size // frame_samples
    trimmed = audio[: n_frames * frame_samples]
    frames = trimmed.reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    noise_floor = float(np.percentile(rms, float(floor_percentile)))
    abs_floor = 10.0 ** (float(absolute_floor_db) / 20.0)
    threshold = max(noise_floor * float(above_floor_ratio), abs_floor)
    above = np.where(rms > threshold)[0]
    if above.size == 0:
        return 0.0
    return float(above[0]) * float(frame_ms)


def _onset_envelope(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float = _DEFAULT_FRAME_MS,
) -> np.ndarray:
    """Half-wave-rectified frame-to-frame RMS difference (cheap spectral flux)."""
    if audio.size == 0 or sr <= 0:
        return np.zeros(0, dtype=np.float32)
    frame_samples = max(1, int(round(sr * frame_ms / 1000.0)))
    if audio.size < frame_samples * 4:
        return np.zeros(0, dtype=np.float32)
    n_frames = audio.size // frame_samples
    trimmed = audio[: n_frames * frame_samples]
    frames = trimmed.reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    diff = np.diff(rms, prepend=rms[0])
    return np.maximum(diff, 0.0).astype(np.float32)


def estimate_bpm_acoustic(
    audio: np.ndarray,
    sr: int,
    *,
    bpm_range: tuple[float, float] = _DEFAULT_BPM_RANGE,
    frame_ms: float = _DEFAULT_FRAME_MS,
) -> tuple[float | None, float]:
    """Estimate BPM via autocorrelation of the onset envelope.

    Lag search is restricted to ``[60000/bpm_hi, 60000/bpm_lo]`` ms so the
    autocorrelation cannot fall into an octave-error trap (the 100–150 BPM
    default span covers only 1.5 octaves, well below the half/double
    ambiguity threshold).

    Returns ``(bpm, confidence)`` where confidence is the normalized
    distance of the peak above the median of the in-range autocorrelation
    — high when the peak is sharply isolated, near zero when the envelope
    is too sparse or too uniform to lock onto a period.
    """
    envelope = _onset_envelope(audio, sr, frame_ms=frame_ms)
    if envelope.size < 30:
        return None, 0.0
    bpm_lo, bpm_hi = float(bpm_range[0]), float(bpm_range[1])
    if bpm_lo <= 0.0 or bpm_hi <= bpm_lo:
        return None, 0.0
    lag_min = max(1, int(round(60_000.0 / bpm_hi / float(frame_ms))))
    lag_max = min(envelope.size - 1, int(round(60_000.0 / bpm_lo / float(frame_ms))))
    if lag_min >= lag_max:
        return None, 0.0
    env_zm = envelope - float(envelope.mean())
    full_ac = np.correlate(env_zm, env_zm, mode="full")
    ac = full_ac[full_ac.size // 2 :]
    if ac.size <= lag_max or ac[0] <= 0.0:
        return None, 0.0
    ac_norm = ac / float(ac[0])
    window = ac_norm[lag_min : lag_max + 1]
    if window.size == 0:
        return None, 0.0
    best_idx = int(np.argmax(window))
    peak = float(window[best_idx])
    if peak <= 0.0:
        return None, 0.0
    median_val = float(np.median(window))
    confidence = max(0.0, min(1.0, (peak - median_val) / max(1e-6, peak)))
    best_lag_frames = lag_min + best_idx
    bpm = 60_000.0 / (best_lag_frames * float(frame_ms))
    return float(bpm), float(confidence)


def estimate_bpm_filename(
    duration_ms: float,
    slot_count: int,
    *,
    lead_in_ms: float = 250.0,
    tail_buffer_ms: float = 150.0,
    bpm_range: tuple[float, float] = _DEFAULT_BPM_RANGE,
    mora_per_beat_options: Iterable[int] = (2, 1),
) -> list[tuple[float, int]]:
    """Return all (BPM, mora_per_beat) pairs that fall in ``bpm_range``.

    Pairs are returned in ``mora_per_beat_options`` order; the caller can
    pick the first as the default. For KO CVVC banks recorded around
    120 BPM, ``mora_per_beat=2`` (eighth-note syllables) is far more
    common than 1 (quarter-note), so it is preferred by default.
    """
    if slot_count <= 1 or duration_ms <= 0.0:
        return []
    effective = max(50.0, float(duration_ms) - float(lead_in_ms) - float(tail_buffer_ms))
    syllable_ms = effective / float(slot_count)
    out: list[tuple[float, int]] = []
    for mora_per_beat in mora_per_beat_options:
        beat_ms = syllable_ms * float(mora_per_beat)
        if beat_ms <= 0.0:
            continue
        bpm = 60_000.0 / beat_ms
        if bpm_range[0] <= bpm <= bpm_range[1]:
            out.append((float(bpm), int(mora_per_beat)))
    return out


def estimate_bpm(
    wav_path: str,
    slot_count: int,
    *,
    bpm_range: tuple[float, float] = _DEFAULT_BPM_RANGE,
    target_sr: int = _DEFAULT_TARGET_SR,
    frame_ms: float = _DEFAULT_FRAME_MS,
    agreement_tolerance_bpm: float = 5.0,
    min_acoustic_confidence: float = 0.3,
) -> BpmEstimate | None:
    """Hybrid filename + acoustic BPM estimator.

    Returns ``None`` when no estimator yields a BPM in ``bpm_range`` with
    sufficient confidence — callers should treat that as "this wav was
    not BGM-recorded" and skip the beat-grid snap entirely.
    """
    try:
        audio, sr, duration = load_wav_mono(str(wav_path), target_sr=int(target_sr))
    except Exception:
        return None
    if audio.size == 0 or sr <= 0:
        return None
    duration_ms = float(duration) * 1000.0
    lead_in_ms = detect_lead_in_ms(audio, sr, frame_ms=frame_ms)
    fn_candidates = estimate_bpm_filename(
        duration_ms,
        slot_count,
        lead_in_ms=lead_in_ms,
        bpm_range=bpm_range,
    )
    bpm_ac, conf_ac = estimate_bpm_acoustic(audio, sr, bpm_range=bpm_range, frame_ms=frame_ms)
    # Priority: high-confidence acoustic wins (filename heuristic depends on
    # tail-buffer assumption which is unreliable). Use filename candidates
    # only to pick mora_per_beat, not the BPM value itself.
    # 2026-05-17 diagnostic: filename gave 102 BPM (tail-buffer 150 ms vs
    # actual ~870 ms) while acoustic correctly returned 120 BPM @ conf 0.90.
    if bpm_ac is not None and conf_ac >= 0.70:
        chosen_mora = 2  # default for 100–150 BPM range
        # If a filename candidate is within agreement tolerance, take its
        # mora_per_beat as the structural hint.
        if fn_candidates:
            best_match = None
            best_diff = float("inf")
            for fbpm, mora in fn_candidates:
                diff = abs(float(fbpm) - float(bpm_ac))
                if diff < best_diff:
                    best_diff = diff
                    best_match = (fbpm, mora)
            if best_match is not None and best_diff <= 20.0:
                chosen_mora = int(best_match[1])
                if best_diff <= float(agreement_tolerance_bpm):
                    return BpmEstimate(
                        bpm=float(bpm_ac),
                        mora_per_beat=chosen_mora,
                        lead_in_ms=float(lead_in_ms),
                        confidence=0.95,
                        source="acoustic+filename",
                    )
                return BpmEstimate(
                    bpm=float(bpm_ac),
                    mora_per_beat=chosen_mora,
                    lead_in_ms=float(lead_in_ms),
                    confidence=0.85,
                    source="acoustic+filename(mora-only)",
                )
        # No filename agreement — pick mora_per_beat by checking which one
        # makes 8-syllable singing fit within the wav minus lead-in.
        for mora_try in (1, 2):
            beat_ms = 60_000.0 / float(bpm_ac) / float(mora_try)
            sung_ms = beat_ms * int(slot_count)
            if 0.55 * (duration_ms - lead_in_ms) <= sung_ms <= 1.10 * (duration_ms - lead_in_ms):
                chosen_mora = mora_try
                break
        return BpmEstimate(
            bpm=float(bpm_ac),
            mora_per_beat=int(chosen_mora),
            lead_in_ms=float(lead_in_ms),
            confidence=float(conf_ac) * 0.85,
            source="acoustic",
        )
    # Filename only (no strong acoustic) — modest confidence
    if fn_candidates:
        fbpm, mora = fn_candidates[0]
        return BpmEstimate(
            bpm=float(fbpm),
            mora_per_beat=int(mora),
            lead_in_ms=float(lead_in_ms),
            confidence=0.65,
            source="filename",
        )
    # Acoustic with low confidence — gate
    if bpm_ac is not None and conf_ac >= float(min_acoustic_confidence):
        return BpmEstimate(
            bpm=float(bpm_ac),
            mora_per_beat=2,
            lead_in_ms=float(lead_in_ms),
            confidence=float(conf_ac) * 0.6,
            source="acoustic(low-conf)",
        )
    return None


def compute_beat_grid(
    estimate: BpmEstimate,
    slot_count: int,
    *,
    duration_ms: float = 0.0,
) -> list[float]:
    """Return the expected onset time (ms) of each syllable slot."""
    if estimate.bpm <= 0.0 or slot_count <= 0:
        return []
    beat_interval_ms = 60_000.0 / float(estimate.bpm) / float(estimate.mora_per_beat)
    grid = [float(estimate.lead_in_ms + i * beat_interval_ms) for i in range(int(slot_count))]
    if duration_ms > 0.0:
        # Clamp any beat that overshoots the wav (degenerate inputs)
        grid = [min(b, float(duration_ms) - 10.0) for b in grid]
    return grid


def snap_offset_to_grid(
    offset_ms: float,
    beat_grid: list[float],
    *,
    tolerance_ms: float = 80.0,
) -> float:
    """Snap ``offset_ms`` to nearest beat within ``tolerance_ms``, else pass through."""
    if not beat_grid:
        return float(offset_ms)
    nearest = min(beat_grid, key=lambda b: abs(b - float(offset_ms)))
    if abs(nearest - float(offset_ms)) <= float(tolerance_ms):
        return float(nearest)
    return float(offset_ms)


def parse_manual_bpm_env(env_map: dict[str, str] | None = None) -> BpmEstimate | None:
    """Build a manual BpmEstimate from environment variables.

    Looks at ``UTOA_BOUNDARY_BPM_MS`` (required) plus optional overrides for
    mora_per_beat, lead_in_ms, confidence. Returns ``None`` when BPM env is
    unset or non-numeric.
    """
    env = env_map if env_map is not None else os.environ
    raw = str(env.get("UTOA_BOUNDARY_BPM_MS", "") or "").strip()
    if not raw:
        return None
    try:
        bpm = float(raw)
    except ValueError:
        return None
    if bpm <= 0.0:
        return None

    def _opt_float(key: str, default: float) -> float:
        v = str(env.get(key, "") or "").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    def _opt_int(key: str, default: int) -> int:
        v = str(env.get(key, "") or "").strip()
        if not v:
            return default
        try:
            return int(float(v))
        except ValueError:
            return default

    return BpmEstimate(
        bpm=bpm,
        mora_per_beat=_opt_int("UTOA_BOUNDARY_BPM_MORA_PER_BEAT", 2),
        lead_in_ms=_opt_float("UTOA_BOUNDARY_BPM_LEAD_IN_MS", 250.0),
        confidence=1.0,
        source="manual",
    )


def bpm_prior_enabled(env_map: dict[str, str] | None = None) -> bool:
    env = env_map if env_map is not None else os.environ
    raw = str(env.get("UTOA_BOUNDARY_BPM_PRIOR_ENABLE", "on") or "").strip().lower()
    return raw in {"on", "1", "true", "yes"}


def bpm_range_from_env(env_map: dict[str, str] | None = None) -> tuple[float, float]:
    env = env_map if env_map is not None else os.environ

    def _opt(key: str, default: float) -> float:
        v = str(env.get(key, "") or "").strip()
        if not v:
            return default
        try:
            return float(v)
        except ValueError:
            return default

    lo = _opt("UTOA_BOUNDARY_BPM_RANGE_MIN", _DEFAULT_BPM_RANGE[0])
    hi = _opt("UTOA_BOUNDARY_BPM_RANGE_MAX", _DEFAULT_BPM_RANGE[1])
    if hi <= lo:
        return _DEFAULT_BPM_RANGE
    return (lo, hi)


def bpm_snap_tolerance_ms(env_map: dict[str, str] | None = None) -> float:
    env = env_map if env_map is not None else os.environ
    raw = str(env.get("UTOA_BOUNDARY_BPM_SNAP_TOLERANCE_MS", "80") or "").strip()
    if not raw:
        return 80.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 80.0


def bpm_min_confidence(env_map: dict[str, str] | None = None) -> float:
    env = env_map if env_map is not None else os.environ
    raw = str(env.get("UTOA_BOUNDARY_BPM_MIN_CONFIDENCE", "0.5") or "").strip()
    if not raw:
        return 0.5
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.5


def role_snap_target_slot(role: str, slot_index: int, slot_count: int, alias: str = "") -> int | None:
    """Return the slot index whose beat the row's offset should align with.

    Per-role policy (2026-05-17 BPM prior design):

    - ``-cv`` / ``cv`` / ``v`` / ``vowel``: head of a syllable → slot_index
    - ``vv``: V→V transition lives at the start of the next syllable → slot_index+1
    - ``v-cv``: 3-token VCV center is the next-syllable consonant → slot_index+1
    - ``vc`` **coda-bridge** (``NG g``/``L H``): release burst is at next-syllable
      onset → slot_index+1
    - ``vc`` **onset** (``a bw``/``eu g``): boundary is *between* beats inside
      the syllable → snap target is ``None`` (skip snap)
    - Other roles → ``None`` (skip)
    """
    role_low = str(role or "").strip().lower()
    if role_low in {"-cv", "cv", "cv-", "v", "vowel"}:
        return int(max(0, min(slot_count - 1, slot_index)))
    if role_low in {"vv", "v-cv"}:
        return int(max(0, min(slot_count - 1, slot_index + 1)))
    if role_low == "vc":
        # Distinguish coda-bridge (left = KO coda) from onset VC (left = vowel).
        parts = str(alias or "").strip().split()
        if len(parts) == 2:
            left = parts[0].split("_", 1)[0]
            if left and left[0] not in "aeiou":
                # Treat as coda-bridge — snap to start of NEXT syllable
                return int(max(0, min(slot_count - 1, slot_index + 1)))
        # Onset VC — no clean beat alignment
        return None
    if role_low == "other":
        # 'other' is the catchall lane for bare KO CV aliases (`geu`/`ga`/
        # `kkya`) that ``_infer_alias_type`` could not decompose (no KO
        # syllable splitter in ``_phones_from_ascii``). Structurally these
        # ARE CV — their offset should land at the start of their own
        # syllable. Detect single-token aliases starting with a consonant
        # letter.
        text = str(alias or "").strip()
        if text and " " not in text and text[0].isalpha() and text[0] not in "aeiou-":
            return int(max(0, min(slot_count - 1, slot_index)))
        return None
    return None


__all__ = [
    "BpmEstimate",
    "bpm_min_confidence",
    "bpm_prior_enabled",
    "bpm_range_from_env",
    "bpm_snap_tolerance_ms",
    "compute_beat_grid",
    "detect_lead_in_ms",
    "estimate_bpm",
    "estimate_bpm_acoustic",
    "estimate_bpm_filename",
    "parse_manual_bpm_env",
    "role_snap_target_slot",
    "snap_offset_to_grid",
]
