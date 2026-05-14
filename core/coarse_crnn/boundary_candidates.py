from __future__ import annotations

from collections import defaultdict

from core.coarse_crnn.boundary_types import BoundaryCandidate, BoundaryFrameScores, normalize_boundary_label
from core.coarse_crnn.oto_audio_candidates import AudioCandidates


def peak_candidates_from_scores(
    score_map: BoundaryFrameScores,
    *,
    min_score: float = 0.32,
    min_gap_ms: float = 16.0,
) -> list[BoundaryCandidate]:
    candidates: list[BoundaryCandidate] = []
    for label, values in score_map.scores.items():
        times = list(score_map.times_ms)
        if len(values) < 3 or len(values) != len(times):
            continue
        picked: list[float] = []
        for idx in range(1, len(values) - 1):
            v = float(values[idx])
            if v < float(min_score):
                continue
            if v < float(values[idx - 1]) or v < float(values[idx + 1]):
                continue
            t = float(times[idx])
            if any(abs(t - prev) <= float(min_gap_ms) for prev in picked):
                continue
            picked.append(t)
            candidates.append(BoundaryCandidate(time_ms=t, kind=normalize_boundary_label(label), score=v, source="model"))
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
) -> list[BoundaryCandidate]:
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
            if abs(float(row.time_ms) - float(bucket[-1].time_ms)) <= float(merge_ms):
                bucket.append(row)
                continue
            merged.append(_collapse_bucket(kind=kind, bucket=bucket, model_weight=model_weight, audio_weight=audio_weight))
            bucket = [row]
        if bucket:
            merged.append(_collapse_bucket(kind=kind, bucket=bucket, model_weight=model_weight, audio_weight=audio_weight))
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


__all__ = [
    "audio_candidates_to_boundary_candidates",
    "merge_candidates",
    "peak_candidates_from_scores",
]

