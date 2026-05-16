from __future__ import annotations

from collections import defaultdict
from statistics import mean

from core.coarse_crnn.boundary_types import BoundaryCandidate, BoundaryFrameScores, normalize_boundary_label
from core.coarse_crnn.oto_audio_candidates import AudioCandidates

LABEL_MIN_SCORE_OFFSETS: dict[str, float] = {
    "syllable_onset": 0.00,
    "consonant_onset": 0.01,
    "vowel_start": 0.00,
    "vowel_stable": 0.03,
    "vowel_end": 0.02,
    "transition_peak": 0.04,
    "next_onset": 0.03,
    "silence_boundary": 0.06,
}


def peak_candidates_from_scores(
    score_map: BoundaryFrameScores,
    *,
    min_score: float = 0.32,
    min_gap_ms: float = 16.0,
) -> list[BoundaryCandidate]:
    candidates: list[BoundaryCandidate] = []
    global_quality = _global_quality(score_map)
    gap_scale = _clamp(1.20 - (0.45 * global_quality), 0.80, 1.35)
    for label, values in score_map.scores.items():
        label_key = normalize_boundary_label(label)
        label_floor = _label_min_score(label_key, base_min_score=float(min_score), global_quality=global_quality)
        label_gap = float(min_gap_ms) * gap_scale
        times = list(score_map.times_ms)
        if len(values) < 3 or len(values) != len(times):
            continue
        picked: list[float] = []
        for idx in range(1, len(values) - 1):
            local_quality = _quality_at_frame(score_map, idx, default=global_quality)
            local_floor = label_floor * _clamp(1.15 - (0.55 * local_quality), 0.85, 1.30)
            v = float(values[idx])
            if v < local_floor:
                continue
            if v < float(values[idx - 1]) or v < float(values[idx + 1]):
                continue
            t = float(times[idx])
            if any(abs(t - prev) <= label_gap for prev in picked):
                continue
            picked.append(t)
            score = _clamp01(v * (0.66 + (0.70 * local_quality)))
            candidates.append(BoundaryCandidate(time_ms=t, kind=label_key, score=score, source="model"))
    candidates.sort(key=lambda item: item.time_ms)
    return candidates


def audio_candidates_to_boundary_candidates(audio: AudioCandidates) -> list[BoundaryCandidate]:
    out: list[BoundaryCandidate] = []
    for peak in audio.onset_peaks:
        score = max(0.0, min(1.0, float(peak.strength)))
        out.append(BoundaryCandidate(time_ms=float(peak.time_ms), kind="syllable_onset", score=score, source="audio_onset"))
        out.append(BoundaryCandidate(time_ms=float(peak.time_ms), kind="consonant_onset", score=score * 0.92, source="audio_onset"))
        out.append(BoundaryCandidate(time_ms=float(peak.time_ms), kind="next_onset", score=score * 0.88, source="audio_onset"))
    for seg in audio.stable_vowel_segments:
        s = max(0.0, min(1.0, float(seg.vowel_confidence) * 0.7 + float(seg.stability_score) * 0.3))
        out.append(BoundaryCandidate(time_ms=float(seg.start_ms), kind="vowel_start", score=s, source="audio_vowel"))
        out.append(BoundaryCandidate(time_ms=float(seg.center_ms), kind="vowel_stable", score=s, source="audio_vowel"))
        out.append(BoundaryCandidate(time_ms=float(seg.end_ms), kind="vowel_end", score=s, source="audio_vowel"))
    for peak in audio.mel_delta_peaks:
        out.append(
            BoundaryCandidate(
                time_ms=float(peak.time_ms),
                kind="transition_peak",
                score=max(0.0, min(1.0, float(peak.strength))),
                source="audio_transition",
            )
        )
    out.append(BoundaryCandidate(time_ms=float(audio.active_start_ms), kind="silence_boundary", score=0.75, source="audio_active"))
    out.append(BoundaryCandidate(time_ms=float(audio.active_end_ms), kind="silence_boundary", score=0.75, source="audio_active"))
    out.sort(key=lambda item: item.time_ms)
    return out


def merge_candidates(
    *,
    model_candidates: list[BoundaryCandidate],
    audio_candidates: list[BoundaryCandidate],
    merge_ms: float = 20.0,
    model_weight: float = 0.70,
    audio_weight: float = 0.55,
    model_quality: float | None = None,
    audio_reliability: float | None = None,
) -> list[BoundaryCandidate]:
    resolved_model_quality = _clamp01(
        float(model_quality) if model_quality is not None else _estimate_model_quality(model_candidates)
    )
    resolved_audio_reliability = _clamp01(
        float(audio_reliability) if audio_reliability is not None else _estimate_audio_reliability(audio_candidates)
    )
    quality_blend = 0.5 * (resolved_model_quality + resolved_audio_reliability)
    merge_ms_eff = float(merge_ms) * _clamp(1.18 - (0.40 * quality_blend), 0.85, 1.30)
    model_weight_eff = float(model_weight) * _clamp(0.75 + (0.55 * resolved_model_quality), 0.45, 1.25)
    audio_weight_eff = float(audio_weight) * _clamp(0.65 + (0.70 * resolved_audio_reliability), 0.35, 1.35)
    if resolved_audio_reliability < 0.32:
        audio_weight_eff *= 0.82
    if resolved_model_quality < 0.32:
        model_weight_eff *= 0.86

    grouped: dict[str, list[BoundaryCandidate]] = defaultdict(list)
    for item in model_candidates:
        grouped[item.kind].append(item)
    for item in audio_candidates:
        grouped[item.kind].append(item)
    merged: list[BoundaryCandidate] = []
    for kind, items in grouped.items():
        rows = sorted(items, key=lambda row: row.time_ms)
        bucket: list[BoundaryCandidate] = []
        for row in rows:
            if not bucket:
                bucket = [row]
                continue
            if abs(float(row.time_ms) - float(bucket[-1].time_ms)) <= merge_ms_eff:
                bucket.append(row)
                continue
            merged.append(
                _collapse_bucket(
                    kind=kind,
                    bucket=bucket,
                    model_weight=model_weight_eff,
                    audio_weight=audio_weight_eff,
                )
            )
            bucket = [row]
        if bucket:
            merged.append(
                _collapse_bucket(
                    kind=kind,
                    bucket=bucket,
                    model_weight=model_weight_eff,
                    audio_weight=audio_weight_eff,
                )
            )
    merged.sort(key=lambda item: item.time_ms)
    return merged


def _collapse_bucket(*, kind: str, bucket: list[BoundaryCandidate], model_weight: float, audio_weight: float) -> BoundaryCandidate:
    total_w = 0.0
    total_t = 0.0
    total_s = 0.0
    sources: list[str] = []
    for item in bucket:
        w = float(model_weight) if item.source.startswith("model") else float(audio_weight)
        s = max(0.0, min(1.0, float(item.score)))
        total_w += w
        total_t += float(item.time_ms) * w
        total_s += s * w
        sources.append(item.source)
    if total_w <= 0.0:
        total_w = 1.0
    merged_source = "merged:" + "+".join(sorted(set(sources)))
    return BoundaryCandidate(
        time_ms=float(total_t / total_w),
        kind=kind,
        score=max(0.0, min(1.0, float(total_s / total_w))),
        source=merged_source,
    )


def _label_min_score(label: str, *, base_min_score: float, global_quality: float) -> float:
    offset = float(LABEL_MIN_SCORE_OFFSETS.get(str(label or "").strip().lower(), 0.0))
    quality_scale = _clamp(1.18 - (0.58 * float(global_quality)), 0.88, 1.35)
    return max(0.05, float(base_min_score) * quality_scale + offset)


def _quality_at_frame(score_map: BoundaryFrameScores, frame_index: int, *, default: float) -> float:
    values = list(score_map.quality_scores or [])
    if not values:
        return _clamp01(default)
    idx = min(max(0, int(frame_index)), len(values) - 1)
    return _clamp01(float(values[idx]))


def _global_quality(score_map: BoundaryFrameScores) -> float:
    values = [float(v) for v in list(score_map.quality_scores or []) if v is not None]
    if not values:
        return 0.50
    return _clamp01(float(mean(values)))


def _estimate_audio_reliability(audio_candidates: list[BoundaryCandidate]) -> float:
    scores = [float(item.score) for item in audio_candidates if str(item.source).startswith("audio")]
    if not scores:
        return 0.50
    return _clamp01(float(mean(scores)))


def _estimate_model_quality(model_candidates: list[BoundaryCandidate]) -> float:
    scores = [float(item.score) for item in model_candidates if str(item.source).startswith("model")]
    if not scores:
        return 0.50
    return _clamp01(float(mean(scores)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


__all__ = [
    "audio_candidates_to_boundary_candidates",
    "merge_candidates",
    "peak_candidates_from_scores",
]
