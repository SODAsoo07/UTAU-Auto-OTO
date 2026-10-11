from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from .domain import Slot, Timeline


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_wav(path: Path, slot_count: int, *, decoder_mode: str = "local") -> Timeline:
    if slot_count < 1:
        raise ValueError("slot_count는 1 이상이어야 합니다")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"빈 WAV입니다: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float64)
    mono = np.nan_to_num(mono, copy=False)
    duration_ms = 1000.0 * len(mono) / float(sample_rate)
    frame_length = max(128, int(round(sample_rate * 0.025)))
    hop_length = max(32, int(round(sample_rate * 0.005)))
    if len(mono) < frame_length:
        mono = np.pad(mono, (0, frame_length - len(mono)))
    remainder = (len(mono) - frame_length) % hop_length
    if remainder:
        mono = np.pad(mono, (0, hop_length - remainder))
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame_length)[::hop_length]
    window = np.hanning(frame_length).astype(np.float64)
    windowed = frames * window
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-8)
    magnitude = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-9
    normalized_mag = magnitude / np.maximum(np.sum(magnitude, axis=1, keepdims=True), 1e-9)
    positive_delta = np.maximum(normalized_mag[1:] - normalized_mag[:-1], 0.0)
    flux = np.concatenate(([0.0], np.sqrt(np.sum(positive_delta * positive_delta, axis=1))))
    flatness = np.exp(np.mean(np.log(magnitude), axis=1)) / np.maximum(np.mean(magnitude, axis=1), 1e-9)
    periodicity = np.clip(1.0 - flatness, 0.0, 1.0)
    frame_times_ms = (np.arange(len(frames)) * hop_length + frame_length / 2.0) * 1000.0 / sample_rate

    peak_db = float(np.percentile(rms_db, 99.0))
    noise_db = float(np.percentile(rms_db, 15.0))
    active_threshold_db = max(noise_db + 8.0, peak_db - 35.0)
    active = rms_db >= active_threshold_db
    active = np.convolve(active.astype(np.int8), np.ones(5, dtype=np.int8), mode="same") >= 2
    warnings: list[str] = []
    if np.any(active):
        activity_indices = _select_main_activity_region(
            active,
            rms_db,
            active_threshold_db,
            max_gap_frames=max(1, int(round(120.0 / 5.0))),
            expected_regions=slot_count,
        )
        active_start_ms = max(0.0, float(frame_times_ms[activity_indices[0]]) - 40.0)
        active_end_ms = min(duration_ms, float(frame_times_ms[activity_indices[-1]]) + 60.0)
    else:
        active_start_ms = 0.0
        active_end_ms = duration_ms
        warnings.append("no_reliable_activity")
    usable_duration = active_end_ms - active_start_ms
    if usable_duration < slot_count * 60.0:
        active_start_ms = 0.0
        active_end_ms = duration_ms
        usable_duration = duration_ms
        warnings.append("activity_too_short_for_slots")

    energy_norm = _robust_normalize(rms_db)
    flux_norm = _robust_normalize(flux)
    periodicity_norm = _robust_normalize(periodicity)
    energy_rise = np.maximum(np.gradient(energy_norm), 0.0)
    periodicity_rise = np.maximum(np.gradient(periodicity_norm), 0.0)
    boundary_score = _smooth(0.55 * flux_norm + 0.30 * energy_rise + 0.15 * periodicity_rise, 3)
    transition_score = _smooth(
        0.42 * periodicity_rise + 0.28 * energy_rise + 0.20 * periodicity_norm + 0.10 * flux_norm,
        3,
    )

    normalized_decoder = decoder_mode.strip().casefold()
    if normalized_decoder not in {"local", "monotonic"}:
        raise ValueError(f"지원하지 않는 boundary decoder입니다: {decoder_mode}")
    if normalized_decoder == "monotonic":
        boundaries, boundary_confidences = _select_boundaries(
            frame_times_ms,
            boundary_score,
            active_start_ms,
            active_end_ms,
            slot_count,
        )
        decoder_version = "monotonic_boundary_dp_v1"
    else:
        boundaries, boundary_confidences = _select_boundaries_local(
            frame_times_ms,
            boundary_score,
            active_start_ms,
            active_end_ms,
            slot_count,
        )
        decoder_version = "local_boundary_v1"
    slots: list[Slot] = []
    transition_confidences: list[float] = []
    for index, (start_ms, end_ms) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        transition_ms, transition_confidence = _select_transition(
            frame_times_ms,
            transition_score,
            periodicity_norm,
            start_ms,
            end_ms,
        )
        stable_ms = min(end_ms - 15.0, transition_ms + max(35.0, min(90.0, (end_ms - start_ms) * 0.12)))
        stable_ms = max(transition_ms, stable_ms)
        boundary_confidence = boundary_confidences[min(index, len(boundary_confidences) - 1)]
        slots.append(
            Slot(
                index=index,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                transition_ms=float(transition_ms),
                stable_ms=float(stable_ms),
                boundary_confidence=float(boundary_confidence),
                transition_confidence=float(transition_confidence),
            )
        )
        transition_confidences.append(transition_confidence)

    contrast_confidence = float(np.clip((peak_db - active_threshold_db) / 24.0, 0.0, 1.0))
    boundary_mean = float(np.mean(boundary_confidences)) if boundary_confidences else 0.5
    transition_mean = float(np.mean(transition_confidences)) if transition_confidences else 0.5
    confidence = float(np.clip(0.35 * contrast_confidence + 0.35 * boundary_mean + 0.30 * transition_mean, 0.0, 1.0))
    if confidence < 0.42:
        warnings.append("low_alignment_confidence")
    slot_durations = np.diff(boundaries)
    if len(slot_durations) > 1 and float(np.std(slot_durations) / max(1.0, np.mean(slot_durations))) > 0.28:
        warnings.append("irregular_slot_duration")

    return Timeline(
        wav_path=path,
        wav_sha256=file_sha256(path),
        sample_rate=int(sample_rate),
        duration_ms=float(duration_ms),
        active_start_ms=float(active_start_ms),
        active_end_ms=float(active_end_ms),
        slots=tuple(slots),
        confidence=confidence,
        decoder_version=decoder_version,
        warnings=tuple(dict.fromkeys(warnings)),
        feature_summary={
            "peak_db": peak_db,
            "noise_db": noise_db,
            "active_threshold_db": active_threshold_db,
            "activity_contrast_db": peak_db - active_threshold_db,
            "frame_ms": 25.0,
            "hop_ms": 5.0,
        },
    )


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 10.0))
    high = float(np.percentile(values, 95.0))
    if high - low < 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="same")


def _select_main_activity_region(
    active: np.ndarray,
    rms_db: np.ndarray,
    threshold_db: float,
    *,
    max_gap_frames: int,
    expected_regions: int = 1,
) -> np.ndarray:
    """Return the dominant connected activity region, ignoring isolated noise.

    Repeated-syllable recordings contain short gaps between moras, so nearby
    active runs are joined. A click or room-noise spike far from the sustained
    recording can no longer pull the left/right activity boundary outward.
    """
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return indices
    contiguous_splits = np.flatnonzero(np.diff(indices) > 1) + 1
    contiguous_regions = np.split(indices, contiguous_splits)

    if expected_regions > 1:
        minimum_run_frames = max(4, int(round(40.0 / 5.0)))
        candidates = [region for region in contiguous_regions if len(region) >= minimum_run_frames]
        if candidates:
            strengths = [
                len(region) * float(np.mean(np.maximum(rms_db[region] - threshold_db, 0.0)))
                for region in candidates
            ]
            strongest = max(strengths)
            reliable = [
                region
                for region, strength in zip(candidates, strengths)
                if strongest <= 1e-9 or strength >= strongest * 0.06
            ]
            if reliable:
                return np.array([reliable[0][0], reliable[-1][-1]], dtype=np.int64)

    split_points = np.flatnonzero(np.diff(indices) > max_gap_frames) + 1
    regions = np.split(indices, split_points)

    def score(region: np.ndarray) -> tuple[float, float, int]:
        span = int(region[-1] - region[0] + 1)
        excess_energy = float(np.sum(np.maximum(rms_db[region] - threshold_db, 0.0)))
        return float(span), excess_energy, len(region)

    return max(regions, key=score)


def _select_boundaries(
    times_ms: np.ndarray,
    score: np.ndarray,
    active_start_ms: float,
    active_end_ms: float,
    slot_count: int,
) -> tuple[np.ndarray, list[float]]:
    if slot_count == 1:
        return np.array([active_start_ms, active_end_ms], dtype=np.float64), [0.7]
    decoded = _select_boundaries_monotonic(
        times_ms,
        score,
        active_start_ms,
        active_end_ms,
        slot_count,
    )
    if decoded is not None:
        return decoded
    return _select_boundaries_local(times_ms, score, active_start_ms, active_end_ms, slot_count)


def _select_boundaries_local(
    times_ms: np.ndarray,
    score: np.ndarray,
    active_start_ms: float,
    active_end_ms: float,
    slot_count: int,
) -> tuple[np.ndarray, list[float]]:
    step = max(1.0, (active_end_ms - active_start_ms) / slot_count)
    boundaries = [active_start_ms]
    confidences: list[float] = []
    for boundary_index in range(1, slot_count):
        target = active_start_ms + boundary_index * step
        left = max(boundaries[-1] + step * 0.45, target - step * 0.30)
        right = min(active_end_ms - (slot_count - boundary_index) * step * 0.45, target + step * 0.30)
        indices = np.flatnonzero((times_ms >= left) & (times_ms <= right))
        if len(indices) == 0:
            chosen = target
            confidence = 0.2
        else:
            local_times = times_ms[indices]
            prior = np.exp(-0.5 * ((local_times - target) / max(10.0, step * 0.18)) ** 2)
            combined = 0.72 * score[indices] + 0.28 * prior
            local_index = int(np.argmax(combined))
            chosen = float(local_times[local_index])
            peak = float(score[indices[local_index]])
            baseline = float(np.median(score[indices]))
            confidence = float(np.clip(0.45 + 0.9 * (peak - baseline), 0.0, 1.0))
        boundaries.append(chosen)
        confidences.append(confidence)
    boundaries.append(active_end_ms)
    if confidences:
        confidences.append(confidences[-1])
    else:
        confidences.append(0.7)
    return np.array(boundaries, dtype=np.float64), confidences


def _select_boundaries_monotonic(
    times_ms: np.ndarray,
    score: np.ndarray,
    active_start_ms: float,
    active_end_ms: float,
    slot_count: int,
) -> tuple[np.ndarray, list[float]] | None:
    """Decode all syllable boundaries as one monotonic path.

    Boundary peaks are acoustic evidence, while asymmetric duration priors keep
    a sustained slot from being shortened solely to match a uniform grid.
    """
    active_indices = np.flatnonzero((times_ms > active_start_ms) & (times_ms < active_end_ms))
    if len(active_indices) < slot_count - 1:
        return None
    step = max(1.0, (active_end_ms - active_start_ms) / slot_count)
    local_peak_mask = np.zeros(len(score), dtype=bool)
    if len(score) >= 3:
        local_peak_mask[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:])
    peaks = active_indices[local_peak_mask[active_indices]]
    peak_limit = max(32, 12 * (slot_count - 1))
    if len(peaks) > peak_limit:
        peaks = peaks[np.argsort(score[peaks])[-peak_limit:]]
    candidate_indices: set[int] = {int(index) for index in peaks}
    for boundary_index in range(1, slot_count):
        target = active_start_ms + boundary_index * step
        window = active_indices[np.abs(times_ms[active_indices] - target) <= step * 0.48]
        if len(window):
            best = window[np.argsort(score[window])[-min(4, len(window)):]]
            candidate_indices.update(int(index) for index in best)
            closest = int(window[np.argmin(np.abs(times_ms[window] - target))])
            candidate_indices.add(closest)
    candidates = sorted(candidate_indices, key=lambda index: float(times_ms[index]))
    if len(candidates) < slot_count - 1:
        return None

    min_duration = step * 0.28
    max_duration = step * 2.15
    sigma = max(20.0, step * 0.52)
    previous: dict[int, tuple[float, tuple[int, ...]]] = {-1: (0.0, ())}
    for boundary_index in range(1, slot_count):
        target = active_start_ms + boundary_index * step
        remaining_slots = slot_count - boundary_index
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        for candidate_position, frame_index in enumerate(candidates):
            boundary_time = float(times_ms[frame_index])
            remaining_duration = active_end_ms - boundary_time
            if remaining_duration < remaining_slots * min_duration or remaining_duration > remaining_slots * max_duration:
                continue
            best_state: tuple[float, tuple[int, ...]] | None = None
            for previous_position, (previous_score, path) in previous.items():
                previous_time = active_start_ms if previous_position < 0 else float(times_ms[candidates[previous_position]])
                duration = boundary_time - previous_time
                if duration < min_duration or duration > max_duration:
                    continue
                z = (duration - step) / sigma
                duration_penalty = -0.5 * z * z
                if duration > step:
                    duration_penalty *= 0.42
                target_penalty = -0.08 * ((boundary_time - target) / step) ** 2
                acoustic = 1.15 * float(score[frame_index])
                total = previous_score + acoustic + duration_penalty + target_penalty
                if best_state is None or total > best_state[0]:
                    best_state = (total, (*path, candidate_position))
            if best_state is not None:
                current[candidate_position] = best_state
        if not current:
            return None
        if len(current) > 96:
            current = dict(sorted(current.items(), key=lambda item: item[1][0], reverse=True)[:96])
        previous = current

    best_final: tuple[float, tuple[int, ...]] | None = None
    for previous_position, (path_score, path) in previous.items():
        last_time = float(times_ms[candidates[previous_position]])
        duration = active_end_ms - last_time
        if duration < min_duration or duration > max_duration:
            continue
        z = (duration - step) / sigma
        duration_penalty = -0.5 * z * z
        if duration > step:
            duration_penalty *= 0.42
        total = path_score + duration_penalty
        if best_final is None or total > best_final[0]:
            best_final = (total, path)
    if best_final is None:
        return None

    chosen_indices = [candidates[position] for position in best_final[1]]
    boundaries = np.asarray(
        [active_start_ms, *(float(times_ms[index]) for index in chosen_indices), active_end_ms],
        dtype=np.float64,
    )
    confidences: list[float] = []
    for frame_index in chosen_indices:
        time_value = float(times_ms[frame_index])
        local = np.flatnonzero(np.abs(times_ms - time_value) <= max(25.0, step * 0.18))
        baseline = float(np.median(score[local])) if len(local) else 0.0
        confidence = float(np.clip(0.45 + 0.9 * (float(score[frame_index]) - baseline), 0.0, 1.0))
        confidences.append(confidence)
    confidences.append(confidences[-1] if confidences else 0.7)
    return boundaries, confidences


def _select_transition(
    times_ms: np.ndarray,
    score: np.ndarray,
    periodicity: np.ndarray,
    start_ms: float,
    end_ms: float,
) -> tuple[float, float]:
    duration = max(1.0, end_ms - start_ms)
    target = start_ms + duration * 0.28
    left = start_ms + min(12.0, duration * 0.05)
    right = start_ms + duration * 0.62
    indices = np.flatnonzero((times_ms >= left) & (times_ms <= right))
    if len(indices) == 0:
        return target, 0.2
    local_times = times_ms[indices]
    prior = np.exp(-0.5 * ((local_times - target) / max(12.0, duration * 0.18)) ** 2)
    combined = 0.58 * score[indices] + 0.24 * periodicity[indices] + 0.18 * prior
    local_index = int(np.argmax(combined))
    chosen = float(local_times[local_index])
    peak = float(combined[local_index])
    baseline = float(np.median(combined))
    confidence = float(np.clip(0.4 + 0.9 * (peak - baseline), 0.0, 1.0))
    return chosen, confidence
