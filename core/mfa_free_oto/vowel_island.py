from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manual_oto_candidates import ManualOtoCandidateTracks


@dataclass(frozen=True)
class VowelIsland:
    start_ms: float
    nucleus_ms: float
    end_ms: float
    confidence: float
    left_valley_ms: float
    right_valley_ms: float
    start_index: int
    nucleus_index: int
    end_index: int


@dataclass(frozen=True)
class SlotIslandAssignment:
    slot_index: int
    island_index: int | None
    score: float
    margin: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VowelIslandDecode:
    islands: tuple[VowelIsland, ...]
    assignments: tuple[SlotIslandAssignment, ...]
    score: float
    margin: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VowelIslandConfig:
    high_threshold: float = 0.34
    low_threshold: float = 0.20
    min_duration_ms: float = 35.0
    merge_gap_ms: float = 45.0
    max_duration_ms: float = 680.0
    min_confidence: float = 0.18
    slot_distance_weight: float = 2.1
    reuse_penalty: float = 0.38
    skip_penalty: float = 1.25
    low_margin_threshold: float = 0.12
    count_mismatch_ratio: float = 0.0
    local_offset_window_ms: float = 190.0
    local_offset_min_gap_ms: float = 18.0


def extract_vowel_islands(
    tracks: ManualOtoCandidateTracks,
    *,
    config: VowelIslandConfig | None = None,
) -> tuple[VowelIsland, ...]:
    cfg = config or VowelIslandConfig()
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size == 0:
        return ()
    energy = vowel_energy_track(tracks)
    if energy.size != times.size:
        energy = np.resize(energy, times.shape).astype(np.float32)
    high = max(float(cfg.high_threshold), float(np.percentile(energy, 64.0)) if energy.size else 0.0)
    low = min(float(cfg.low_threshold), high * 0.72)
    segments = _hysteresis_segments(energy, high=high, low=low)
    segments = _merge_segments(segments, times, max_gap_ms=float(cfg.merge_gap_ms))
    segments = _split_long_segments(segments, times, energy, max_duration_ms=float(cfg.max_duration_ms))
    out: list[VowelIsland] = []
    for start, end in segments:
        if end <= start:
            continue
        start_ms = float(times[start])
        end_ms = float(times[min(end, times.size - 1)])
        if end_ms - start_ms < float(cfg.min_duration_ms):
            continue
        local = energy[start : end + 1]
        if local.size == 0:
            continue
        nucleus_rel = int(np.argmax(local))
        nucleus_idx = start + nucleus_rel
        confidence = float(np.percentile(local, 85.0))
        if confidence < float(cfg.min_confidence):
            continue
        left_valley_idx = _local_min_index(energy, max(0, start - 4), nucleus_idx)
        right_valley_idx = _local_min_index(energy, nucleus_idx, min(int(times.size) - 1, end + 4))
        out.append(
            VowelIsland(
                start_ms=start_ms,
                nucleus_ms=float(times[nucleus_idx]),
                end_ms=end_ms,
                confidence=confidence,
                left_valley_ms=float(times[left_valley_idx]),
                right_valley_ms=float(times[right_valley_idx]),
                start_index=int(start),
                nucleus_index=int(nucleus_idx),
                end_index=int(end),
            )
        )
    return tuple(out)


def assign_slots_to_islands(
    islands: tuple[VowelIsland, ...],
    *,
    slot_count: int,
    duration_ms: float,
    config: VowelIslandConfig | None = None,
) -> VowelIslandDecode:
    cfg = config or VowelIslandConfig()
    n = max(1, int(slot_count))
    m = len(islands)
    warnings: list[str] = []
    if m == 0:
        return VowelIslandDecode(
            islands=islands,
            assignments=tuple(
                SlotIslandAssignment(slot_index=i, island_index=None, score=-float(cfg.skip_penalty), margin=0.0, warnings=("missing_island",))
                for i in range(n)
            ),
            score=-float(cfg.skip_penalty) * float(n),
            margin=0.0,
            warnings=("missing_islands",),
        )
    mismatch_ratio = abs(float(m) - float(n)) / max(1.0, float(n))
    if m != n and mismatch_ratio >= float(cfg.count_mismatch_ratio):
        warnings.append("island_count_mismatch")

    # DP state: slot -> selected island. Nondecreasing island indices are allowed so
    # dense OTO rows can share one vowel island without forcing a false split.
    dp = np.full((n, m), -1e9, dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int32)
    for j in range(m):
        dp[0, j] = _slot_island_score(0, n, j, islands, duration_ms, cfg)
    for i in range(1, n):
        for j in range(m):
            best_score = -1e9
            best_prev = -1
            for prev in range(j + 1):
                penalty = float(cfg.reuse_penalty) if prev == j else 0.0
                score = float(dp[i - 1, prev]) - penalty
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i, j] = best_score + _slot_island_score(i, n, j, islands, duration_ms, cfg)
            back[i, j] = best_prev
    ranked = np.argsort(dp[n - 1])[::-1]
    best_last = int(ranked[0])
    best_score = float(dp[n - 1, best_last])
    second_score = float(dp[n - 1, int(ranked[1])]) if ranked.size > 1 else best_score - 1.0
    margin = float(best_score - second_score)
    if margin < float(cfg.low_margin_threshold):
        warnings.append("low_dp_margin")

    indices = [best_last]
    for i in range(n - 1, 0, -1):
        indices.append(int(back[i, indices[-1]]))
    indices.reverse()

    assignments: list[SlotIslandAssignment] = []
    for i, island_idx in enumerate(indices):
        local_warnings: list[str] = []
        island = islands[int(island_idx)]
        if island.confidence < float(cfg.min_confidence) + 0.08:
            local_warnings.append("low_assignment_confidence")
        assignments.append(
            SlotIslandAssignment(
                slot_index=int(i),
                island_index=int(island_idx),
                score=float(_slot_island_score(i, n, int(island_idx), islands, duration_ms, cfg)),
                margin=margin,
                warnings=tuple([*warnings, *local_warnings]),
            )
        )
    return VowelIslandDecode(
        islands=islands,
        assignments=tuple(assignments),
        score=best_score,
        margin=margin,
        warnings=tuple(warnings),
    )


def fit_islands_to_slot_count(
    tracks: ManualOtoCandidateTracks,
    islands: tuple[VowelIsland, ...],
    *,
    slot_count: int,
    config: VowelIslandConfig | None = None,
) -> tuple[VowelIsland, ...]:
    cfg = config or VowelIslandConfig()
    target = max(1, int(slot_count))
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size == 0:
        return islands
    energy = vowel_energy_track(tracks)
    current = sorted(list(islands), key=lambda item: item.start_ms)
    if not current:
        return current if target <= 0 else tuple(current)
    while len(current) < target:
        inserted = _insert_likely_missing_gap_island(current, tracks, times, energy)
        if inserted is not None:
            current = inserted
            continue
        durations = [float(item.end_ms) - float(item.start_ms) for item in current]
        idx = int(np.argmax(durations))
        item = current[idx]
        if item.end_index - item.start_index < 4:
            break
        left, right = _split_island(item, times, energy, cfg)
        if left is None or right is None:
            break
        current[idx : idx + 1] = [left, right]
    while len(current) > target and len(current) > 1:
        merge_idx = _closest_island_pair(current)
        merged = _merge_island_pair(current[merge_idx], current[merge_idx + 1], times, energy)
        current[merge_idx : merge_idx + 2] = [merged]
    return tuple(sorted(current, key=lambda item: item.start_ms))


def vowel_energy_track(tracks: ManualOtoCandidateTracks) -> np.ndarray:
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    size = int(times.size)
    if size == 0:
        return np.zeros((0,), dtype=np.float32)
    nucleus = _unit(_track(tracks, "nucleus", size))
    voicing = _unit(_track(tracks, "voicing", size))
    rms = _unit(_track(tracks, "rms", size))
    stability_raw = _track(tracks, "world_stability", size)
    stability = _unit(stability_raw) if np.any(stability_raw) else np.full((size,), 0.5, dtype=np.float32)
    silence = np.clip(_track(tracks, "silence", size), 0.0, 1.0)
    product = np.power(np.clip(voicing, 0.0, 1.0), 0.75)
    product *= np.power(np.clip(rms, 0.0, 1.0), 0.45)
    product *= np.power(np.clip(stability, 0.0, 1.0), 0.35)
    product *= 0.45 + 0.55 * np.clip(nucleus, 0.0, 1.0)
    energy = _unit(0.62 * _unit(product) + 0.23 * nucleus + 0.15 * (1.0 - silence))
    return _smooth(energy, radius=2)


def local_offset_from_preutterance(
    tracks: ManualOtoCandidateTracks,
    *,
    preutterance_ms: float,
    expected_gap_ms: float,
    config: VowelIslandConfig | None = None,
) -> float:
    cfg = config or VowelIslandConfig()
    times = np.asarray(tracks.times_ms, dtype=np.float32)
    if times.size == 0:
        return max(0.0, float(preutterance_ms) - float(expected_gap_ms))
    expected = float(preutterance_ms) - max(float(cfg.local_offset_min_gap_ms), float(expected_gap_ms))
    start = max(0.0, expected - float(cfg.local_offset_window_ms))
    end = min(float(preutterance_ms) - float(cfg.local_offset_min_gap_ms), expected + float(cfg.local_offset_window_ms))
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return min(float(preutterance_ms), max(0.0, expected))
    idxs = np.flatnonzero(mask)
    size = int(times.size)
    offset_score = _unit(np.asarray(tracks.anchor_scores.get("offset", np.zeros((size,), dtype=np.float32)), dtype=np.float32))
    transition = _unit(_track(tracks, "transition", size))
    activity_edge = _unit(_track(tracks, "activity_edge", size))
    distance = np.abs(times[idxs] - expected) / max(1.0, float(cfg.local_offset_window_ms))
    score = 0.48 * offset_score[idxs] + 0.30 * transition[idxs] + 0.22 * activity_edge[idxs] - 0.42 * distance
    best = int(idxs[int(np.argmax(score))])
    return float(times[best])


def assignment_is_safe(assignment: SlotIslandAssignment | None, island: VowelIsland | None) -> bool:
    if assignment is None or island is None or assignment.island_index is None:
        return False
    warnings = set(assignment.warnings)
    if {"missing_island", "island_count_mismatch", "low_dp_margin", "low_assignment_confidence"} & warnings:
        return False
    return float(island.start_ms) <= float(island.nucleus_ms) <= float(island.end_ms)


def island_overlay_events(decode: VowelIslandDecode) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for idx, island in enumerate(decode.islands):
        events.extend(
            [
                {"label": "vowel_island_start", "time_ms": float(island.start_ms), "island_index": idx},
                {"label": "vowel_island_nucleus", "time_ms": float(island.nucleus_ms), "island_index": idx},
                {"label": "vowel_island_end", "time_ms": float(island.end_ms), "island_index": idx},
            ]
        )
    return events


def _island_from_indices(start: int, end: int, times: np.ndarray, energy: np.ndarray) -> VowelIsland | None:
    if times.size == 0:
        return None
    start = max(0, min(int(start), int(times.size) - 1))
    end = max(start, min(int(end), int(times.size) - 1))
    local = energy[start : end + 1]
    if local.size == 0:
        return None
    nucleus_idx = start + int(np.argmax(local))
    left_valley_idx = _local_min_index(energy, start, nucleus_idx)
    right_valley_idx = _local_min_index(energy, nucleus_idx, end)
    return VowelIsland(
        start_ms=float(times[start]),
        nucleus_ms=float(times[nucleus_idx]),
        end_ms=float(times[end]),
        confidence=float(np.percentile(local, 85.0)),
        left_valley_ms=float(times[left_valley_idx]),
        right_valley_ms=float(times[right_valley_idx]),
        start_index=int(start),
        nucleus_index=int(nucleus_idx),
        end_index=int(end),
    )


def _split_island(
    island: VowelIsland,
    times: np.ndarray,
    energy: np.ndarray,
    cfg: VowelIslandConfig,
) -> tuple[VowelIsland | None, VowelIsland | None]:
    start = int(island.start_index)
    end = int(island.end_index)
    if end - start < 4:
        return None, None
    mid = (start + end) // 2
    radius = max(2, min(18, (end - start) // 3))
    lo = max(start + 2, mid - radius)
    hi = min(end - 2, mid + radius)
    if hi <= lo:
        split = mid
    else:
        split = lo + int(np.argmin(energy[lo : hi + 1]))
    left = _island_from_indices(start, max(start, split), times, energy)
    right = _island_from_indices(min(end, split + 1), end, times, energy)
    if left is None or right is None:
        return None, None
    if float(left.end_ms) - float(left.start_ms) < max(20.0, float(cfg.min_duration_ms) * 0.5):
        return None, None
    if float(right.end_ms) - float(right.start_ms) < max(20.0, float(cfg.min_duration_ms) * 0.5):
        return None, None
    return left, right


def _closest_island_pair(islands: list[VowelIsland]) -> int:
    best_idx = 0
    best_gap = float("inf")
    for idx in range(len(islands) - 1):
        gap = float(islands[idx + 1].start_ms) - float(islands[idx].end_ms)
        if gap < best_gap:
            best_gap = gap
            best_idx = idx
    return best_idx


def _merge_island_pair(
    left: VowelIsland,
    right: VowelIsland,
    times: np.ndarray,
    energy: np.ndarray,
) -> VowelIsland:
    merged = _island_from_indices(int(left.start_index), int(right.end_index), times, energy)
    if merged is None:
        return VowelIsland(
            start_ms=float(left.start_ms),
            nucleus_ms=float(left.nucleus_ms if left.confidence >= right.confidence else right.nucleus_ms),
            end_ms=float(right.end_ms),
            confidence=max(float(left.confidence), float(right.confidence)),
            left_valley_ms=float(left.left_valley_ms),
            right_valley_ms=float(right.right_valley_ms),
            start_index=int(left.start_index),
            nucleus_index=int(left.nucleus_index if left.confidence >= right.confidence else right.nucleus_index),
            end_index=int(right.end_index),
        )
    return merged


def _insert_likely_missing_gap_island(
    islands: list[VowelIsland],
    tracks: ManualOtoCandidateTracks,
    times: np.ndarray,
    energy: np.ndarray,
) -> list[VowelIsland] | None:
    if len(islands) < 2 or times.size == 0:
        return None
    nucleus_gaps = [
        float(islands[idx + 1].nucleus_ms) - float(islands[idx].nucleus_ms)
        for idx in range(len(islands) - 1)
        if float(islands[idx + 1].nucleus_ms) > float(islands[idx].nucleus_ms)
    ]
    median_nucleus_gap = float(np.median(nucleus_gaps)) if nucleus_gaps else 0.0
    min_gap_ms = max(170.0, min(360.0, 0.58 * median_nucleus_gap if median_nucleus_gap > 0.0 else 210.0))
    candidates: list[tuple[float, int, int, int]] = []
    size = int(times.size)
    stable = _unit(_track(tracks, "stable_vowel", size))
    nucleus = _unit(_track(tracks, "nucleus", size))
    rms = _unit(_track(tracks, "rms", size))
    silence = np.clip(_track(tracks, "silence", size), 0.0, 1.0)
    weak_vowel_score = _unit(0.46 * _unit(energy) + 0.24 * stable + 0.18 * nucleus + 0.08 * rms + 0.04 * (1.0 - silence))
    for idx in range(len(islands) - 1):
        left = islands[idx]
        right = islands[idx + 1]
        gap_ms = float(right.start_ms) - float(left.end_ms)
        if gap_ms < min_gap_ms:
            continue
        start_idx = max(0, int(left.end_index) + 1)
        end_idx = min(size - 1, int(right.start_index) - 1)
        if end_idx - start_idx < 3:
            continue
        local = weak_vowel_score[start_idx : end_idx + 1]
        if local.size == 0:
            continue
        peak_rel = int(np.argmax(local))
        peak_idx = start_idx + peak_rel
        peak_score = float(local[peak_rel])
        if peak_score < 0.12:
            continue
        edge_gap = min(float(times[peak_idx]) - float(left.end_ms), float(right.start_ms) - float(times[peak_idx]))
        if edge_gap < 28.0:
            continue
        candidates.append((gap_ms * max(0.35, peak_score), idx, peak_idx, start_idx))
    if not candidates:
        return None
    _score, insert_after, peak_idx, _start_idx = max(candidates, key=lambda item: item[0])
    left = islands[insert_after]
    right = islands[insert_after + 1]
    threshold = max(0.10, min(0.42, float(weak_vowel_score[peak_idx]) * 0.46))
    start = int(peak_idx)
    while start > int(left.end_index) + 1 and weak_vowel_score[start - 1] >= threshold:
        if float(times[peak_idx]) - float(times[start - 1]) > 190.0:
            break
        start -= 1
    end = int(peak_idx)
    while end + 1 < int(right.start_index) and weak_vowel_score[end + 1] >= threshold:
        if float(times[end + 1]) - float(times[peak_idx]) > 210.0:
            break
        end += 1
    if end <= start:
        start = max(int(left.end_index) + 1, int(peak_idx) - 2)
        end = min(int(right.start_index) - 1, int(peak_idx) + 2)
    inserted = _island_from_indices(start, end, times, weak_vowel_score)
    if inserted is None:
        return None
    if float(inserted.end_ms) - float(inserted.start_ms) < 18.0:
        return None
    out = list(islands)
    out.insert(insert_after + 1, inserted)
    return sorted(out, key=lambda item: item.start_ms)


def _slot_island_score(
    slot_index: int,
    slot_count: int,
    island_index: int,
    islands: tuple[VowelIsland, ...],
    duration_ms: float,
    cfg: VowelIslandConfig,
) -> float:
    island = islands[int(island_index)]
    duration = max(1.0, float(duration_ms))
    if len(islands) > 1 and slot_count > 1:
        observed_start = float(islands[0].nucleus_ms)
        observed_end = float(islands[-1].nucleus_ms)
        observed_span = max(1.0, observed_end - observed_start)
        expected_time = observed_start + (float(slot_index) / float(slot_count - 1)) * observed_span
    else:
        expected_time = (float(slot_index) + 0.5) / max(1.0, float(slot_count)) * duration
    distance = abs(float(island.nucleus_ms) - expected_time) / duration
    if len(islands) > 1 and slot_count > 1:
        expected_order = float(slot_index) / float(slot_count - 1)
        island_order = float(island_index) / float(len(islands) - 1)
        order_distance = abs(island_order - expected_order)
    else:
        order_distance = 0.0
    return float(island.confidence) - float(cfg.slot_distance_weight) * distance - 0.65 * order_distance


def _track(tracks: ManualOtoCandidateTracks, key: str, size: int) -> np.ndarray:
    values = np.asarray(tracks.tracks.get(key, np.zeros((size,), dtype=np.float32)), dtype=np.float32)
    if values.size != size:
        return np.resize(values, (size,)).astype(np.float32)
    return values


def _unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 5.0))
    hi = float(np.percentile(arr, 95.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(values: np.ndarray, *, radius: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or radius <= 0:
        return arr
    kernel = np.ones((radius * 2 + 1,), dtype=np.float32)
    kernel /= float(kernel.sum())
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def _hysteresis_segments(values: np.ndarray, *, high: float, low: float) -> list[tuple[int, int]]:
    arr = np.asarray(values, dtype=np.float32)
    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < arr.size:
        if arr[idx] < float(high):
            idx += 1
            continue
        start = idx
        while start > 0 and arr[start - 1] >= float(low):
            start -= 1
        end = idx
        while end + 1 < arr.size and arr[end + 1] >= float(low):
            end += 1
        segments.append((start, end))
        idx = end + 1
    return segments


def _merge_segments(segments: list[tuple[int, int]], times: np.ndarray, *, max_gap_ms: float) -> list[tuple[int, int]]:
    if not segments:
        return []
    out = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = out[-1]
        gap = float(times[start]) - float(times[prev_end])
        if gap <= float(max_gap_ms):
            out[-1] = (prev_start, end)
        else:
            out.append((start, end))
    return out


def _split_long_segments(
    segments: list[tuple[int, int]],
    times: np.ndarray,
    energy: np.ndarray,
    *,
    max_duration_ms: float,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in segments:
        duration = float(times[end]) - float(times[start])
        if duration <= float(max_duration_ms):
            out.append((start, end))
            continue
        count = max(2, int(round(duration / max(1.0, float(max_duration_ms)))))
        edges = np.linspace(start, end + 1, count + 1, dtype=np.int32)
        for left, right in zip(edges[:-1], edges[1:]):
            seg_start = int(left)
            seg_end = max(seg_start, int(right) - 1)
            out.append((seg_start, seg_end))
    out.sort()
    return _merge_segments(out, times, max_gap_ms=12.0)


def _local_min_index(values: np.ndarray, start: int, end: int) -> int:
    lo = max(0, min(int(start), int(end)))
    hi = min(int(values.size) - 1, max(int(start), int(end)))
    if hi <= lo:
        return lo
    local = np.asarray(values[lo : hi + 1], dtype=np.float32)
    return int(lo + int(np.argmin(local)))


__all__ = [
    "SlotIslandAssignment",
    "VowelIsland",
    "VowelIslandConfig",
    "VowelIslandDecode",
    "assign_slots_to_islands",
    "assignment_is_safe",
    "extract_vowel_islands",
    "fit_islands_to_slot_count",
    "island_overlay_events",
    "local_offset_from_preutterance",
    "vowel_energy_track",
]
