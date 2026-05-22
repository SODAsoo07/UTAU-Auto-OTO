from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import FeatureBatch, extract_features
from .manifest import load_manifest_jsonl, write_manifest_jsonl


@dataclass(frozen=True)
class AcousticNucleusConfig:
    edge_margin_ms: float = 28.0
    min_edge_margin_ratio: float = 0.18
    min_confidence: float = 0.20
    nucleus_weight: float = 0.30
    voicing_weight: float = 0.24
    periodicity_weight: float = 0.20
    stability_weight: float = 0.16
    rms_weight: float = 0.10
    silence_penalty: float = 0.22


@dataclass(frozen=True)
class AcousticNucleusSummary:
    rows: int
    vowel_segments: int
    relabelled_events: int
    low_confidence_events: int
    mean_abs_shift_ms: float | None
    median_abs_shift_ms: float | None
    p90_abs_shift_ms: float | None
    max_abs_shift_ms: float | None


def relabel_vowel_nuclei_manifest(
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    encoder: str = "acoustic_world_v1",
    device: str | None = None,
    config: AcousticNucleusConfig | None = None,
) -> tuple[list[dict], AcousticNucleusSummary]:
    rows = load_manifest_jsonl(manifest_path, require_manual=True, require_labels=True)
    relabelled_rows: list[dict] = []
    row_summaries: list[dict] = []
    for row in rows:
        new_row, row_summary = relabel_vowel_nuclei(
            row,
            encoder=encoder,
            device=device,
            config=config,
        )
        relabelled_rows.append(new_row)
        row_summaries.append(row_summary)
    write_manifest_jsonl(out_path, relabelled_rows)
    summary = summarize_nucleus_relabel(row_summaries)
    return relabelled_rows, summary


def relabel_vowel_nuclei(
    row: dict,
    *,
    encoder: str = "acoustic_world_v1",
    device: str | None = None,
    config: AcousticNucleusConfig | None = None,
) -> tuple[dict, dict]:
    batch = extract_features(row["wav_path"], encoder=encoder, device=device)
    new_row, row_summary = relabel_vowel_nuclei_from_batch(
        row,
        batch,
        config=config,
    )
    policy = dict(new_row.get("label_policy") or {})
    policy["vowel_nucleus"] = {
        "source": "acoustic_recomputed",
        "encoder": encoder,
        "lab_midpoint_kept_as_auxiliary_event": True,
        "source_oto_timing_used": False,
        "config": asdict(config or AcousticNucleusConfig()),
    }
    new_row["label_policy"] = policy
    return new_row, row_summary


def relabel_vowel_nuclei_from_batch(
    row: dict,
    batch: FeatureBatch,
    *,
    config: AcousticNucleusConfig | None = None,
) -> tuple[dict, dict]:
    cfg = config or AcousticNucleusConfig()
    vowel_segments = [
        segment
        for segment in row.get("frame_labels") or []
        if str(segment.get("label") or "") == "vowel"
    ]
    auxiliary_events: list[dict] = list(row.get("auxiliary_events") or [])
    nucleus_events_by_key: dict[tuple[str, float, float], dict] = {}
    recomputed_events: list[dict] = []
    diagnostics: list[dict] = []
    low_confidence = 0
    shifts: list[float] = []

    for segment in vowel_segments:
        start_ms = float(segment.get("start_ms", 0.0))
        end_ms = float(segment.get("end_ms", 0.0))
        phone = str(segment.get("phone") or "")
        lab_midpoint = (start_ms + end_ms) * 0.5
        candidate = select_acoustic_nucleus(
            batch,
            start_ms=start_ms,
            end_ms=end_ms,
            config=cfg,
        )
        selected_ms = float(candidate["time_ms"])
        confidence = float(candidate["confidence"])
        if confidence < float(cfg.min_confidence):
            low_confidence += 1
        shift = selected_ms - lab_midpoint
        shifts.append(abs(shift))
        event = {
            "label": "vowel_nucleus",
            "time_ms": selected_ms,
            "phone": phone,
            "confidence": confidence,
            "source": "acoustic_recomputed",
            "lab_midpoint_time_ms": lab_midpoint,
            "lab_midpoint_shift_ms": shift,
            "vowel_start_ms": start_ms,
            "vowel_end_ms": end_ms,
            "selection": {key: value for key, value in candidate.items() if key != "time_ms"},
        }
        key = (phone, round(start_ms, 3), round(end_ms, 3))
        nucleus_events_by_key[key] = event
        recomputed_events.append(event)
        auxiliary_events.append(
            {
                "label": "vowel_nucleus_lab_midpoint",
                "time_ms": lab_midpoint,
                "phone": phone,
                "confidence": float(segment.get("confidence", 1.0)),
                "source": "htk_lab_segment_midpoint",
                "vowel_start_ms": start_ms,
                "vowel_end_ms": end_ms,
            }
        )
        auxiliary_events.append(
            {
                "label": "vowel_nucleus_acoustic",
                "time_ms": selected_ms,
                "phone": phone,
                "confidence": confidence,
                "source": "acoustic_recomputed",
                "vowel_start_ms": start_ms,
                "vowel_end_ms": end_ms,
            }
        )
        diagnostics.append(
            {
                "phone": phone,
                "vowel_start_ms": start_ms,
                "vowel_end_ms": end_ms,
                "lab_midpoint_time_ms": lab_midpoint,
                "acoustic_time_ms": selected_ms,
                "shift_ms": shift,
                "confidence": confidence,
                "score": float(candidate["score"]),
            }
        )

    events: list[dict] = []
    used_nucleus_keys: set[tuple[str, float, float]] = set()
    segment_lookup = {
        (
            str(segment.get("phone") or ""),
            round(float(segment.get("start_ms", 0.0)), 3),
            round(float(segment.get("end_ms", 0.0)), 3),
        ): segment
        for segment in vowel_segments
    }
    for event in row.get("events") or []:
        if str(event.get("label") or "") != "vowel_nucleus":
            events.append(dict(event))
            continue
        phone = str(event.get("phone") or "")
        time_ms = float(event.get("time_ms", 0.0))
        matched_key = _nearest_segment_key(segment_lookup, phone=phone, time_ms=time_ms)
        if matched_key is None:
            events.append(dict(event))
        else:
            events.append(dict(nucleus_events_by_key[matched_key]))
            used_nucleus_keys.add(matched_key)

    for key, event in nucleus_events_by_key.items():
        if key not in used_nucleus_keys:
            events.append(dict(event))

    new_row = dict(row)
    new_row["events"] = events
    new_row["auxiliary_events"] = auxiliary_events
    new_row["vowel_nucleus_diagnostics"] = diagnostics
    metadata = dict(new_row.get("landmark_metadata") or {})
    metadata["vowel_nucleus_relabelled"] = True
    metadata["vowel_nucleus_source"] = "acoustic_recomputed"
    metadata["vowel_nucleus_lab_midpoint_preserved"] = True
    new_row["landmark_metadata"] = metadata

    row_summary = {
        "row_id": row.get("row_id") or row.get("wav_name"),
        "vowel_segments": len(vowel_segments),
        "relabelled_events": len(recomputed_events),
        "low_confidence_events": low_confidence,
        "abs_shift_ms": shifts,
        "diagnostics": diagnostics,
    }
    return new_row, row_summary


def select_acoustic_nucleus(
    batch: FeatureBatch,
    *,
    start_ms: float,
    end_ms: float,
    config: AcousticNucleusConfig | None = None,
) -> dict:
    cfg = config or AcousticNucleusConfig()
    times = np.asarray(batch.times_ms, dtype=np.float32)
    if times.size == 0:
        midpoint = (float(start_ms) + float(end_ms)) * 0.5
        return {
            "time_ms": midpoint,
            "score": 0.0,
            "confidence": 0.0,
            "reason": "empty_feature_batch",
        }
    segment_start = float(start_ms)
    segment_end = float(end_ms)
    midpoint = (segment_start + segment_end) * 0.5
    if segment_end <= segment_start:
        idx = _nearest_time_index(times, midpoint)
        return {
            "time_ms": float(times[idx]),
            "score": 0.0,
            "confidence": 0.0,
            "reason": "invalid_vowel_segment",
        }

    margin = _effective_edge_margin(segment_start, segment_end, cfg)
    inner_start = segment_start + margin
    inner_end = segment_end - margin
    if inner_end <= inner_start:
        inner_start = segment_start
        inner_end = segment_end
    mask = (times >= inner_start) & (times <= inner_end)
    if not np.any(mask):
        mask = (times >= segment_start) & (times <= segment_end)
    if not np.any(mask):
        idx = _nearest_time_index(times, midpoint)
        mask = np.zeros_like(times, dtype=bool)
        mask[idx] = True

    score = acoustic_nucleus_score(batch, config=cfg)
    smoothed = _smooth(score)
    masked_indices = np.where(mask)[0]
    local_scores = smoothed[masked_indices]
    best_local = int(np.argmax(local_scores))
    best_idx = int(masked_indices[best_local])
    best_score = float(local_scores[best_local])
    median_score = float(np.median(local_scores)) if local_scores.size else 0.0
    p75_score = float(np.percentile(local_scores, 75)) if local_scores.size else median_score
    confidence = float(np.clip((0.65 * best_score) + (0.35 * max(0.0, best_score - median_score + p75_score)), 0.0, 1.0))
    return {
        "time_ms": float(times[best_idx]),
        "score": best_score,
        "confidence": confidence,
        "local_median_score": median_score,
        "local_p75_score": p75_score,
        "inner_start_ms": inner_start,
        "inner_end_ms": inner_end,
        "reason": "acoustic_score_peak",
    }


def acoustic_nucleus_score(batch: FeatureBatch, *, config: AcousticNucleusConfig | None = None) -> np.ndarray:
    cfg = config or AcousticNucleusConfig()
    frame_count = int(len(batch.times_ms))
    nucleus = _score_track(batch, frame_count, "world_nucleus", "nucleus_likelihood", "voicing")
    voicing = _score_track(batch, frame_count, "world_voicing", "voicing")
    periodicity = _score_track(batch, frame_count, "world_periodicity", "voicing")
    stability = _score_track(batch, frame_count, "world_spectral_stability")
    rms = _score_track(batch, frame_count, "rms")
    silence = _score_track(batch, frame_count, "silence_likelihood")
    score = (
        float(cfg.nucleus_weight) * nucleus
        + float(cfg.voicing_weight) * voicing
        + float(cfg.periodicity_weight) * periodicity
        + float(cfg.stability_weight) * stability
        + float(cfg.rms_weight) * rms
        - float(cfg.silence_penalty) * silence
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def summarize_nucleus_relabel(row_summaries: Sequence[dict]) -> AcousticNucleusSummary:
    shifts: list[float] = []
    rows = len(row_summaries)
    vowel_segments = 0
    relabelled = 0
    low_conf = 0
    for row in row_summaries:
        vowel_segments += int(row.get("vowel_segments") or 0)
        relabelled += int(row.get("relabelled_events") or 0)
        low_conf += int(row.get("low_confidence_events") or 0)
        shifts.extend(float(value) for value in row.get("abs_shift_ms") or [])
    arr = np.asarray(shifts, dtype=np.float32)
    return AcousticNucleusSummary(
        rows=rows,
        vowel_segments=vowel_segments,
        relabelled_events=relabelled,
        low_confidence_events=low_conf,
        mean_abs_shift_ms=(float(np.mean(arr)) if arr.size else None),
        median_abs_shift_ms=(float(np.median(arr)) if arr.size else None),
        p90_abs_shift_ms=(float(np.percentile(arr, 90)) if arr.size else None),
        max_abs_shift_ms=(float(np.max(arr)) if arr.size else None),
    )


def write_summary_json(path: str | Path, summary: AcousticNucleusSummary | dict) -> None:
    payload = asdict(summary) if isinstance(summary, AcousticNucleusSummary) else summary
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _score_track(batch: FeatureBatch, frame_count: int, *keys: str) -> np.ndarray:
    for key in keys:
        values = np.asarray(batch.acoustic_scores.get(key, []), dtype=np.float32)
        if values.shape[0] == frame_count:
            return _robust_unit(values)
    return np.zeros((frame_count,), dtype=np.float32)


def _robust_unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi <= lo + 1e-7:
        hi = float(np.max(arr))
        lo = float(np.min(arr))
    if hi <= lo + 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _smooth(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        return arr
    return np.convolve(arr, np.asarray([0.25, 0.50, 0.25], dtype=np.float32), mode="same").astype(np.float32)


def _effective_edge_margin(start_ms: float, end_ms: float, cfg: AcousticNucleusConfig) -> float:
    duration = max(0.0, float(end_ms) - float(start_ms))
    by_ratio = duration * max(0.0, min(0.45, float(cfg.min_edge_margin_ratio)))
    return min(float(cfg.edge_margin_ms), by_ratio)


def _nearest_time_index(times: np.ndarray, time_ms: float) -> int:
    if times.size == 0:
        return 0
    return int(np.argmin(np.abs(times.astype(np.float32) - float(time_ms))))


def _nearest_segment_key(
    segment_lookup: dict[tuple[str, float, float], dict],
    *,
    phone: str,
    time_ms: float,
) -> tuple[str, float, float] | None:
    candidates: list[tuple[float, tuple[str, float, float]]] = []
    for key, segment in segment_lookup.items():
        seg_phone = key[0]
        if seg_phone and phone and seg_phone != phone:
            continue
        start_ms = float(segment.get("start_ms", 0.0))
        end_ms = float(segment.get("end_ms", 0.0))
        if start_ms <= float(time_ms) <= end_ms:
            midpoint = (start_ms + end_ms) * 0.5
            candidates.append((abs(float(time_ms) - midpoint), key))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
