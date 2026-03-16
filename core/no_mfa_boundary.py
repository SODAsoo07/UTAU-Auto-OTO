from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values)) / float(len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = _clamp(float(q), 0.0, 1.0) * float(len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = float(pos - lo)
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _smooth(values: Sequence[float], radius: int = 2) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - int(radius))
        hi = min(len(values), idx + int(radius) + 1)
        out.append(_mean(values[lo:hi]))
    return out


def _frame_signal(
    samples: Sequence[float],
    sr: int,
    *,
    win_ms: float = 25.0,
    hop_ms: float = 5.0,
) -> Dict[str, List[float]]:
    out = {
        "times_ms": [],
        "energy": [],
        "zcr": [],
    }
    if not samples or sr <= 0:
        return out
    win = max(8, int(float(sr) * float(win_ms) / 1000.0))
    hop = max(1, int(float(sr) * float(hop_ms) / 1000.0))
    sample_count = len(samples)
    for start in range(0, sample_count, hop):
        end = min(sample_count, start + win)
        frame = samples[start:end]
        if not frame:
            break
        abs_vals = [abs(float(v)) for v in frame]
        energy = _mean(abs_vals)
        zcr = 0.0
        if len(frame) > 1:
            changes = 0
            prev = float(frame[0])
            for val in frame[1:]:
                cur = float(val)
                if (prev < 0.0 <= cur) or (prev > 0.0 >= cur):
                    changes += 1
                prev = cur
            zcr = float(changes) / float(max(len(frame) - 1, 1))
        out["times_ms"].append(float(start) * 1000.0 / float(sr))
        out["energy"].append(float(energy))
        out["zcr"].append(float(zcr))
    return out


def _contiguous_runs(mask: Sequence[bool]) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = -1
    for idx, flag in enumerate(mask):
        if flag and start < 0:
            start = idx
            continue
        if not flag and start >= 0:
            runs.append((start, idx))
            start = -1
    if start >= 0:
        runs.append((start, len(mask)))
    return runs


def _pick_best_run(runs: Sequence[tuple[int, int]], mid_idx: float) -> Optional[tuple[int, int]]:
    if not runs:
        return None
    best_run = None
    best_score = None
    for start, end in runs:
        length = max(1, end - start)
        center = (float(start) + float(end)) * 0.5
        score = abs(center - float(mid_idx)) - (float(length) * 0.35)
        if best_score is None or score < best_score:
            best_run = (start, end)
            best_score = score
    return best_run


def _uniform_segments(
    duration_ms: float,
    token_count: int,
    *,
    pause_mask: Optional[Sequence[bool]] = None,
) -> List[Dict[str, float]]:
    if token_count <= 0 or duration_ms <= 0.0:
        return []
    pauses = list(bool(x) for x in (pause_mask or []))
    while len(pauses) < token_count:
        pauses.append(False)
    out: List[Dict[str, float]] = []
    for idx in range(token_count):
        start = float(idx) * float(duration_ms) / float(token_count)
        end = float(idx + 1) * float(duration_ms) / float(token_count)
        span = max(1.0, end - start)
        nucleus_start = start + (span * 0.25)
        nucleus_end = start + (span * 0.75)
        boundary_conf = 0.34
        if pauses[idx]:
            boundary_conf = 0.26
        out.append(
            {
                "onset_ms": float(start),
                "tail_ms": float(end),
                "voiced_onset_ms": float(start),
                "nucleus_start_ms": float(nucleus_start),
                "nucleus_end_ms": float(nucleus_end),
                "blank_confidence": 0.72 if pauses[idx] else 0.58,
                "voiced_confidence": 0.12 if pauses[idx] else 0.32,
                "unvoiced_confidence": 0.18 if pauses[idx] else 0.28,
                "boundary_confidence": float(boundary_conf),
                "source_detail": "uniform_split",
            }
        )
    return out


def _first_true(mask: Sequence[bool], start: int, end: int) -> int:
    for idx in range(max(0, start), min(len(mask), end)):
        if mask[idx]:
            return idx
    return max(0, start)


def _last_true(mask: Sequence[bool], start: int, end: int) -> int:
    hi = min(len(mask), end)
    lo = max(0, start)
    for idx in range(hi - 1, lo - 1, -1):
        if mask[idx]:
            return idx
    return max(lo, hi - 1)


def _segment_payload(
    *,
    start_idx: int,
    end_idx: int,
    times_ms: Sequence[float],
    hop_ms: float,
    energy: Sequence[float],
    active_mask: Sequence[bool],
    voiced_mask: Sequence[bool],
    unvoiced_mask: Sequence[bool],
    noise_floor: float,
    energy_spread: float,
    file_boundary_conf: float,
    source_detail: str,
    pause_like: bool,
) -> Dict[str, float]:
    safe_start = max(0, min(int(start_idx), len(times_ms) - 1))
    safe_end = max(safe_start + 1, min(int(end_idx), len(times_ms)))
    onset_idx = _first_true(active_mask, safe_start, safe_end)
    tail_idx = _last_true(active_mask, safe_start, safe_end)
    if tail_idx <= onset_idx:
        onset_idx = safe_start
        tail_idx = max(safe_start, safe_end - 1)

    seg_energy = [float(v) for v in energy[safe_start:safe_end]]
    seg_active = list(bool(v) for v in active_mask[safe_start:safe_end])
    seg_voiced = list(bool(v) for v in voiced_mask[safe_start:safe_end])
    seg_unvoiced = list(bool(v) for v in unvoiced_mask[safe_start:safe_end])
    seg_count = max(len(seg_energy), 1)
    blank_ratio = float(sum(1 for flag in seg_active if not flag)) / float(seg_count)
    voiced_ratio = float(sum(1 for flag in seg_voiced if flag)) / float(seg_count)
    unvoiced_ratio = float(sum(1 for flag in seg_unvoiced if flag)) / float(seg_count)
    voiced_runs = _contiguous_runs(seg_voiced)
    local_mid = (float(safe_start) + float(safe_end)) * 0.5 - float(safe_start)
    run = _pick_best_run(voiced_runs, local_mid)

    onset_ms = float(times_ms[onset_idx])
    tail_ms = float(times_ms[tail_idx] + float(hop_ms))
    if run is not None:
        run_start = safe_start + int(run[0])
        run_end = safe_start + int(run[1])
        voiced_onset_ms = float(times_ms[run_start])
        nucleus_start_ms = float(times_ms[run_start])
        nucleus_end_ms = float(times_ms[min(len(times_ms) - 1, max(run_start, run_end - 1))] + float(hop_ms))
    else:
        span_ms = max(1.0, tail_ms - onset_ms)
        voiced_onset_ms = float(onset_ms)
        nucleus_start_ms = float(onset_ms + (span_ms * 0.25))
        nucleus_end_ms = float(onset_ms + (span_ms * 0.75))

    local_peak = max(seg_energy) if seg_energy else 0.0
    local_energy_conf = _clamp((local_peak - float(noise_floor)) / max(float(energy_spread), 1e-6), 0.0, 1.0)
    density = 1.0 - blank_ratio
    boundary_conf = _clamp(
        0.22 + (float(file_boundary_conf) * 0.38) + (density * 0.20) + (local_energy_conf * 0.20) + (voiced_ratio * 0.12),
        0.05,
        0.98,
    )
    if source_detail == "uniform_split":
        boundary_conf *= 0.72
    if pause_like:
        blank_ratio = max(blank_ratio, 0.55)
        voiced_ratio *= 0.35
        boundary_conf *= 0.82

    return {
        "onset_ms": float(max(0.0, onset_ms)),
        "tail_ms": float(max(onset_ms + 1.0, tail_ms)),
        "voiced_onset_ms": float(max(onset_ms, voiced_onset_ms)),
        "nucleus_start_ms": float(max(onset_ms, nucleus_start_ms)),
        "nucleus_end_ms": float(max(nucleus_start_ms + 1.0, min(tail_ms, nucleus_end_ms))),
        "blank_confidence": float(_clamp(blank_ratio, 0.0, 1.0)),
        "voiced_confidence": float(_clamp(voiced_ratio * 1.6, 0.0, 1.0)),
        "unvoiced_confidence": float(_clamp(unvoiced_ratio * 1.6, 0.0, 1.0)),
        "boundary_confidence": float(_clamp(boundary_conf, 0.0, 1.0)),
        "source_detail": str(source_detail),
    }


def estimate_token_boundaries(
    samples: Sequence[float],
    sr: int,
    *,
    token_count: int,
    pause_mask: Optional[Sequence[bool]] = None,
    duration_ms: float = 0.0,
) -> List[Dict[str, float]]:
    if token_count <= 0:
        return []
    if not samples or sr <= 0:
        return _uniform_segments(float(duration_ms), int(token_count), pause_mask=pause_mask)

    frame = _frame_signal(samples, sr)
    times_ms = list(frame.get("times_ms") or [])
    energy = _smooth(frame.get("energy") or [], radius=2)
    zcr = _smooth(frame.get("zcr") or [], radius=2)
    if not times_ms or not energy or len(times_ms) != len(energy):
        return _uniform_segments(float(duration_ms), int(token_count), pause_mask=pause_mask)

    hop_ms = float(times_ms[1] - times_ms[0]) if len(times_ms) > 1 else 5.0
    noise_floor = _percentile(energy, 0.15)
    peak_energy = _percentile(energy, 0.95)
    energy_spread = max(peak_energy - noise_floor, 1e-6)
    speech_thr = max(noise_floor + (energy_spread * 0.18), noise_floor * 1.35, 0.0035)
    voiced_thr = max(noise_floor + (energy_spread * 0.32), speech_thr + (energy_spread * 0.08), 0.0060)
    zcr_voiced_thr = min(0.22, max(0.06, _percentile(zcr, 0.50) * 0.95))

    active_mask = [float(v) >= float(speech_thr) for v in energy]
    voiced_mask = [
        bool(active_mask[idx]) and float(energy[idx]) >= float(voiced_thr) and float(zcr[idx]) <= float(zcr_voiced_thr)
        for idx in range(len(energy))
    ]
    unvoiced_mask = [bool(active_mask[idx]) and not bool(voiced_mask[idx]) for idx in range(len(energy))]

    active_runs = _contiguous_runs(active_mask)
    duration_est_ms = float(duration_ms) if float(duration_ms) > 0.0 else float(times_ms[-1] + hop_ms)
    if not active_runs:
        return _uniform_segments(duration_est_ms, int(token_count), pause_mask=pause_mask)

    longest_active = max(active_runs, key=lambda item: item[1] - item[0])
    active_start_idx, active_end_idx = int(longest_active[0]), int(longest_active[1])
    if active_end_idx <= active_start_idx:
        return _uniform_segments(duration_est_ms, int(token_count), pause_mask=pause_mask)

    speech_start_ms = float(times_ms[active_start_idx])
    speech_end_ms = float(times_ms[min(len(times_ms) - 1, active_end_idx - 1)] + hop_ms)
    speech_span_ms = max(1.0, speech_end_ms - speech_start_ms)
    active_ratio = float(sum(1 for flag in active_mask if flag)) / float(max(len(active_mask), 1))
    energy_contrast = _clamp((peak_energy - noise_floor) / max(peak_energy, 1e-6), 0.0, 1.0)
    file_boundary_conf = _clamp(
        0.26 + (energy_contrast * 0.42) + (min(speech_span_ms / max(duration_est_ms, 1.0), 1.0) * 0.16) + ((1.0 - abs(active_ratio - 0.55)) * 0.16),
        0.08,
        0.98,
    )

    pauses = list(bool(x) for x in (pause_mask or []))
    while len(pauses) < token_count:
        pauses.append(False)

    frame_span = active_end_idx - active_start_idx
    min_gap_frames = max(2, int(round(max(18.0, speech_span_ms / max(token_count * 3.2, 1.0)) / max(hop_ms, 1.0))))
    if frame_span < max(token_count, 1) * min_gap_frames:
        return _uniform_segments(duration_est_ms, int(token_count), pause_mask=pauses)

    if token_count == 1:
        return [
            _segment_payload(
                start_idx=active_start_idx,
                end_idx=active_end_idx,
                times_ms=times_ms,
                hop_ms=hop_ms,
                energy=energy,
                active_mask=active_mask,
                voiced_mask=voiced_mask,
                unvoiced_mask=unvoiced_mask,
                noise_floor=noise_floor,
                energy_spread=energy_spread,
                file_boundary_conf=file_boundary_conf,
                source_detail="boundary_guided",
                pause_like=pauses[0],
            )
        ]

    boundaries: List[int] = []
    prev_idx = active_start_idx
    for boundary_num in range(1, token_count):
        target_ms = speech_start_ms + (speech_span_ms * float(boundary_num) / float(token_count))
        target_idx = min(range(len(times_ms)), key=lambda idx: abs(float(times_ms[idx]) - float(target_ms)))
        remaining_tokens = token_count - boundary_num
        lo = max(prev_idx + min_gap_frames, target_idx - max(min_gap_frames * 3, 4))
        hi = min(active_end_idx - (remaining_tokens * min_gap_frames), target_idx + max(min_gap_frames * 3, 4))
        if hi <= lo:
            candidate = max(prev_idx + min_gap_frames, min(active_end_idx - (remaining_tokens * min_gap_frames), target_idx))
            candidate = min(candidate, active_end_idx - 1)
            boundaries.append(candidate)
            prev_idx = candidate
            continue
        best_idx = lo
        best_score = None
        for idx in range(lo, hi + 1):
            valley_score = float(energy[idx])
            voiced_penalty = 0.30 if voiced_mask[idx] else 0.0
            dist_penalty = abs(idx - target_idx) / float(max(hi - lo + 1, 1))
            score = valley_score + voiced_penalty + (dist_penalty * 0.16)
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
        boundaries.append(best_idx)
        prev_idx = best_idx

    out: List[Dict[str, float]] = []
    seg_starts = [active_start_idx] + boundaries
    seg_ends = boundaries + [active_end_idx]
    for idx in range(token_count):
        source_detail = "boundary_guided"
        if pauses[idx]:
            source_detail = "boundary_guided_pause"
        out.append(
            _segment_payload(
                start_idx=seg_starts[idx],
                end_idx=seg_ends[idx],
                times_ms=times_ms,
                hop_ms=hop_ms,
                energy=energy,
                active_mask=active_mask,
                voiced_mask=voiced_mask,
                unvoiced_mask=unvoiced_mask,
                noise_floor=noise_floor,
                energy_spread=energy_spread,
                file_boundary_conf=file_boundary_conf,
                source_detail=source_detail,
                pause_like=pauses[idx],
            )
        )
    return out


__all__ = ["estimate_token_boundaries"]
