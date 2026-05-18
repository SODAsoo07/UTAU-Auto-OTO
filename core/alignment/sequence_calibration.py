"""Voicebank-level calibration helpers for the sequence aligner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class WavCalibSample:
    duration_ms: float
    rms_p05: float
    rms_p18: float
    rms_p40: float
    rms_p60: float
    rms_p90: float
    leading_silence_ms: float
    tail_silence_ms: float
    word_count: int
    words: Tuple[str, ...] = field(default_factory=tuple)
    alias_key: str = ""


@dataclass
class VoicebankCalibration:
    sample_count: int = 0
    silence_floor_rms: float = 0.0
    voiced_floor_rms: float = 0.0
    leading_silence_ms_p50: float = 0.0
    leading_silence_ms_p90: float = 0.0
    tail_silence_ms_p50: float = 0.0
    tail_silence_ms_p90: float = 0.0
    alias_duration_ms: Dict[str, List[float]] = field(default_factory=dict)
    alias_duration_median_ms: Dict[str, float] = field(default_factory=dict)
    reclist_kind: str = "unknown"
    word_count_median: float = 0.0
    notes: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return int(self.sample_count) >= 1


def _percentile(values: Sequence[float], p: float) -> float:
    arr = sorted(float(v) for v in values if v is not None and np.isfinite(float(v)))
    if not arr:
        return 0.0
    if p <= 0.0:
        return float(arr[0])
    if p >= 1.0:
        return float(arr[-1])
    idx = int(round(float(p) * (len(arr) - 1)))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _active_ratio_guard(
    threshold: float,
    values: Sequence[float] | np.ndarray | None,
    *,
    min_active_ratio: float | None,
    max_active_ratio: float | None,
) -> float:
    if values is None:
        return float(max(1e-6, threshold))
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        return float(max(1e-6, threshold))
    lo = 0.0 if min_active_ratio is None else float(np.clip(min_active_ratio, 0.01, 0.95))
    hi = 1.0 if max_active_ratio is None else float(np.clip(max_active_ratio, lo + 0.02, 0.99))
    out = float(max(1e-6, threshold))
    for _ in range(8):
        active_ratio = float(np.mean(arr >= out))
        if active_ratio > hi:
            out *= 1.13
            continue
        if active_ratio < lo:
            out *= 0.84
            continue
        break
    return float(max(1e-6, out))


def extract_wav_calib_sample(
    rms: np.ndarray,
    *,
    hop_sec: float,
    silence_threshold: float,
    words: Sequence[str] | None = None,
    alias_key: str = "",
) -> WavCalibSample:
    r = np.asarray(rms, dtype=np.float32).reshape(-1)
    n = int(r.size)
    hop_s = max(float(hop_sec), 1e-6)
    word_list = tuple(str(w or "").strip() for w in (words or ()) if str(w or "").strip())
    if n <= 0:
        return WavCalibSample(
            duration_ms=0.0,
            rms_p05=0.0,
            rms_p18=0.0,
            rms_p40=0.0,
            rms_p60=0.0,
            rms_p90=0.0,
            leading_silence_ms=0.0,
            tail_silence_ms=0.0,
            word_count=len(word_list),
            words=word_list,
            alias_key=str(alias_key or ""),
        )

    duration_ms = float(n) * hop_s * 1000.0
    threshold = max(1e-6, float(silence_threshold))
    active = r >= threshold

    leading_frames = 0
    for v in active:
        if not bool(v):
            leading_frames += 1
        else:
            break
    tail_frames = 0
    for v in active[::-1]:
        if not bool(v):
            tail_frames += 1
        else:
            break

    return WavCalibSample(
        duration_ms=duration_ms,
        rms_p05=float(np.quantile(r, 0.05)),
        rms_p18=float(np.quantile(r, 0.18)),
        rms_p40=float(np.quantile(r, 0.40)),
        rms_p60=float(np.quantile(r, 0.60)),
        rms_p90=float(np.quantile(r, 0.90)),
        leading_silence_ms=float(leading_frames) * hop_s * 1000.0,
        tail_silence_ms=float(tail_frames) * hop_s * 1000.0,
        word_count=len(word_list),
        words=word_list,
        alias_key=str(alias_key or ""),
    )


def estimate_reclist_kind(words_per_wav: Sequence[Sequence[str]]) -> str:
    if not words_per_wav:
        return "unknown"
    counts: List[int] = []
    starts_with_v = 0
    long_chain = 0
    total = 0
    for words in words_per_wav:
        toks = [str(w or "").strip().lower() for w in words if str(w or "").strip()]
        if not toks:
            continue
        total += 1
        counts.append(len(toks))
        first = toks[0]
        if first and first[0] in {"a", "i", "u", "e", "o"} and len(first) <= 2:
            starts_with_v += 1
        if len(toks) >= 4:
            long_chain += 1
    if total <= 0:
        return "unknown"

    avg = float(np.mean(counts)) if counts else 0.0
    sv_ratio = float(starts_with_v) / float(total)
    long_ratio = float(long_chain) / float(total)
    if avg < 1.4:
        return "cv"
    if long_ratio > 0.30:
        return "delta"
    if sv_ratio > 0.55 and avg <= 2.6:
        return "vcv"
    if avg >= 2.0:
        return "cvvc"
    return "unknown"


def aggregate_calibration(samples: Sequence[WavCalibSample]) -> VoicebankCalibration:
    if not samples:
        return VoicebankCalibration()

    sample_count = len(samples)
    silence_floor_rms = float(np.median([s.rms_p05 for s in samples]))
    voiced_floor_rms = float(np.median([s.rms_p40 for s in samples]))
    leading = [float(s.leading_silence_ms) for s in samples]
    tail = [float(s.tail_silence_ms) for s in samples]

    alias_duration_ms: Dict[str, List[float]] = {}
    for s in samples:
        key = str(s.alias_key or "").strip().lower()
        if key:
            alias_duration_ms.setdefault(key, []).append(float(s.duration_ms))
    alias_duration_median = {k: float(np.median(v)) for k, v in alias_duration_ms.items() if v}

    word_counts = [int(s.word_count) for s in samples]
    notes: List[str] = []
    if silence_floor_rms <= 0.0:
        notes.append("voicebank silence floor unavailable")
    if voiced_floor_rms <= silence_floor_rms:
        notes.append("voiced floor not separable from silence floor")

    return VoicebankCalibration(
        sample_count=sample_count,
        silence_floor_rms=silence_floor_rms,
        voiced_floor_rms=voiced_floor_rms,
        leading_silence_ms_p50=_percentile(leading, 0.50),
        leading_silence_ms_p90=_percentile(leading, 0.90),
        tail_silence_ms_p50=_percentile(tail, 0.50),
        tail_silence_ms_p90=_percentile(tail, 0.90),
        alias_duration_ms=alias_duration_ms,
        alias_duration_median_ms=alias_duration_median,
        reclist_kind=estimate_reclist_kind([list(s.words) for s in samples]),
        word_count_median=float(np.median(word_counts)) if word_counts else 0.0,
        notes=notes,
    )


def resolve_effective_silence_threshold(
    wav_threshold: float,
    *,
    calibration: VoicebankCalibration | None,
    floor_weight: float = 0.7,
    values: Sequence[float] | np.ndarray | None = None,
    min_active_ratio: float | None = None,
    max_active_ratio: float | None = None,
) -> float:
    """Blend a per-wav threshold with the voicebank floor, then keep activity sane."""
    base = float(max(1e-6, wav_threshold))
    if calibration is None or not calibration.is_active():
        return base
    vb_floor = float(calibration.silence_floor_rms)
    if vb_floor <= 0.0:
        return base

    w = float(np.clip(floor_weight, 0.0, 1.0))
    blended = base * (1.0 - w) + max(vb_floor, 1e-6) * w
    blended = max(blended, vb_floor * 0.9)
    blended = _active_ratio_guard(
        blended,
        values,
        min_active_ratio=min_active_ratio,
        max_active_ratio=max_active_ratio,
    )
    return float(max(1e-6, blended))


def reclist_duration_bias(
    *,
    calibration: VoicebankCalibration | None,
    alias_key: str,
    nominal_total_frames: int,
    hop_sec: float,
    min_occurrences: int = 2,
) -> float:
    if (
        calibration is None
        or not calibration.is_active()
        or nominal_total_frames <= 0
        or hop_sec <= 0.0
    ):
        return 1.0
    key = str(alias_key or "").strip().lower()
    if not key:
        return 1.0
    durations = calibration.alias_duration_ms.get(key)
    if not durations or len(durations) < max(1, int(min_occurrences)):
        return 1.0
    observed_frames = float(np.median(durations)) / 1000.0 / float(hop_sec)
    if observed_frames <= 0.0:
        return 1.0
    return float(np.clip(observed_frames / float(nominal_total_frames), 0.5, 1.5))


def collect_voicebank_calibration_from_samples(
    samples: Iterable[WavCalibSample],
) -> VoicebankCalibration:
    return aggregate_calibration(list(samples))


__all__ = [
    "VoicebankCalibration",
    "WavCalibSample",
    "aggregate_calibration",
    "collect_voicebank_calibration_from_samples",
    "estimate_reclist_kind",
    "extract_wav_calib_sample",
    "reclist_duration_bias",
    "resolve_effective_silence_threshold",
]
