from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.coarse_crnn.deprecated.coarse_alignment.decode import viterbi_align_phones
from core.coarse_crnn.audio import estimate_active_region_for_wav
from core.coarse_crnn.deprecated.coarse_alignment.inference import predict_outputs
from core.coarse_crnn.deprecated.coarse_alignment.lab_io import frame_targets_from_segments, load_aligned_segments
from core.coarse_crnn.deprecated.coarse_alignment.model import load_checkpoint
from core.coarse_crnn.deprecated.coarse_alignment.reclist import estimate_reclist_phone_bounds
from core.coarse_crnn.deprecated.coarse_alignment.types import PhoneSegment, Segment


@dataclass
class AlignmentEvalConfig:
    model_path: str
    device: str = "auto"
    max_items: int = 0
    max_per_language: int = 0
    source: str = ""
    seed: int = 1337
    min_gold_segments: int = 1


def evaluate_manifest(rows: list[dict[str, Any]], config: AlignmentEvalConfig) -> dict[str, Any]:
    torch = __import__("torch")
    from core.coarse_crnn.torch_utils import resolve_torch_device

    device = resolve_torch_device(torch, config.device)
    model, model_config, meta = load_checkpoint(config.model_path, map_location=str(device))
    model = model.to(device).eval()

    selected = _select_rows(rows, config)
    file_results: list[dict[str, Any]] = []
    boundary_errors: list[float] = []
    boundary_hits_20 = 0
    boundary_hits_50 = 0
    boundary_count = 0
    frame_correct = 0
    frame_count = 0
    confidence_values: list[float] = []
    vowel_onset_errors: list[float] = []
    cv_transition_errors: list[float] = []
    failures: list[str] = []

    for idx, row in enumerate(selected, start=1):
        try:
            result = _evaluate_one(row, model=model, model_config=model_config, device=str(device))
        except Exception as exc:
            failures.append(f"{row.get('audio', '')}: {exc}")
            continue
        file_results.append(result)
        boundary_errors.extend(result["boundary_errors_sec"])
        boundary_hits_20 += int(result["boundary_hits_20"])
        boundary_hits_50 += int(result["boundary_hits_50"])
        boundary_count += int(result["boundary_count"])
        frame_correct += int(result["coarse_frame_correct"])
        frame_count += int(result["coarse_frame_count"])
        vowel_onset_errors.extend(result.get("vowel_onset_errors_sec") or [])
        cv_transition_errors.extend(result.get("cv_transition_errors_sec") or [])
        if result["mean_confidence"] is not None:
            confidence_values.append(float(result["mean_confidence"]))

    summary = {
        "model_path": os.path.abspath(config.model_path),
        "model_meta": meta,
        "device": str(device),
        "requested_items": len(selected),
        "evaluated_items": len(file_results),
        "failed_items": len(failures),
        "boundary_count": int(boundary_count),
        "boundary_mae_ms": _mean(boundary_errors) * 1000.0 if boundary_errors else None,
        "boundary_median_ae_ms": _median(boundary_errors) * 1000.0 if boundary_errors else None,
        "boundary_acc_20ms": float(boundary_hits_20) / float(boundary_count) if boundary_count else None,
        "boundary_acc_50ms": float(boundary_hits_50) / float(boundary_count) if boundary_count else None,
        "coarse_frame_acc": float(frame_correct) / float(frame_count) if frame_count else None,
        "vowel_onset_count": len(vowel_onset_errors),
        "vowel_onset_mae_ms": _mean(vowel_onset_errors) * 1000.0 if vowel_onset_errors else None,
        "vowel_onset_acc_50ms": _hit_rate(vowel_onset_errors, 0.050),
        "cv_transition_count": len(cv_transition_errors),
        "cv_transition_mae_ms": _mean(cv_transition_errors) * 1000.0 if cv_transition_errors else None,
        "cv_transition_acc_50ms": _hit_rate(cv_transition_errors, 0.050),
        "mean_confidence": _mean(confidence_values) if confidence_values else None,
        "by_language": _aggregate_by(file_results, "language"),
        "by_source": _aggregate_by(file_results, "source"),
        "failures": failures[:20],
        "files": file_results,
    }
    return summary


def _select_rows(rows: list[dict[str, Any]], config: AlignmentEvalConfig) -> list[dict[str, Any]]:
    source = str(config.source or "").strip()
    candidates = [row for row in rows if not source or str(row.get("source", "")) == source]
    rng = random.Random(int(config.seed))
    rng.shuffle(candidates)
    if int(config.max_per_language) > 0:
        by_lang: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            by_lang.setdefault(str(row.get("language", "") or "unknown"), []).append(row)
        selected: list[dict[str, Any]] = []
        for lang in sorted(by_lang):
            selected.extend(by_lang[lang][: int(config.max_per_language)])
        candidates = selected
    if int(config.max_items) > 0:
        candidates = candidates[: int(config.max_items)]
    return candidates


def _evaluate_one(row: dict[str, Any], *, model, model_config, device: str) -> dict[str, Any]:
    wav_path = str(row.get("audio", "") or "")
    language = str(row.get("language", "") or "")
    label_path = str(row.get("label_path", "") or "")
    label_format = str(row.get("label_format", "") or "")
    gold_all = load_aligned_segments(label_path, label_format)
    gold = [seg for seg in gold_all if str(seg.label or "").strip()]
    if not gold:
        raise ValueError("gold alignment is empty")

    phones = [str(seg.label).strip() for seg in gold]
    probs, boundary_probs, hop_sec, duration_sec = predict_outputs(model, model_config, wav_path, device=device)
    active_region = estimate_active_region_for_wav(wav_path, target_sr=int(model_config.sample_rate))
    boundary_priors = estimate_reclist_phone_bounds(
        wav_path,
        phones,
        language=language,
        active_range_sec=active_region,
    )
    pred = viterbi_align_phones(
        probs,
        phones,
        labels=list(model_config.labels),
        hop_sec=hop_sec,
        language=language,
        duration_sec=duration_sec,
        active_range_sec=active_region,
        boundary_priors_sec=boundary_priors,
        boundary_probs=boundary_probs,
    )
    boundary_errors = _boundary_errors(gold, pred)
    vowel_onset_errors = _vowel_onset_errors(gold, pred)
    cv_transition_errors = _cv_transition_errors(gold, pred)
    targets = frame_targets_from_segments(
        gold_all,
        frame_count=int(probs.shape[0]),
        hop_sec=float(hop_sec),
        duration_sec=float(duration_sec),
        language=language,
    )
    pred_ids = np.argmax(probs, axis=1).astype(np.int64) if probs.size else np.zeros((0,), dtype=np.int64)
    valid = targets != -100
    common = min(int(pred_ids.shape[0]), int(targets.shape[0]))
    if common > 0:
        valid = valid[:common]
        frame_correct = int(np.sum(pred_ids[:common][valid] == targets[:common][valid]))
        frame_count = int(np.sum(valid))
    else:
        frame_correct = 0
        frame_count = 0
    conf_values = [float(seg.confidence) for seg in pred if seg.confidence is not None]
    return {
        "audio": wav_path,
        "language": language,
        "source": str(row.get("source", "")),
        "gold_segments": len(gold),
        "pred_segments": len(pred),
        "duration_sec": float(duration_sec),
        "boundary_count": len(boundary_errors),
        "boundary_mae_ms": _mean(boundary_errors) * 1000.0 if boundary_errors else None,
        "boundary_acc_20ms": _hit_rate(boundary_errors, 0.020),
        "boundary_acc_50ms": _hit_rate(boundary_errors, 0.050),
        "boundary_hits_20": sum(1 for value in boundary_errors if value <= 0.020),
        "boundary_hits_50": sum(1 for value in boundary_errors if value <= 0.050),
        "boundary_errors_sec": boundary_errors,
        "vowel_onset_errors_sec": vowel_onset_errors,
        "vowel_onset_mae_ms": _mean(vowel_onset_errors) * 1000.0 if vowel_onset_errors else None,
        "vowel_onset_acc_50ms": _hit_rate(vowel_onset_errors, 0.050),
        "cv_transition_errors_sec": cv_transition_errors,
        "cv_transition_mae_ms": _mean(cv_transition_errors) * 1000.0 if cv_transition_errors else None,
        "cv_transition_acc_50ms": _hit_rate(cv_transition_errors, 0.050),
        "coarse_frame_correct": frame_correct,
        "coarse_frame_count": frame_count,
        "coarse_frame_acc": float(frame_correct) / float(frame_count) if frame_count else None,
        "mean_confidence": _mean(conf_values) if conf_values else None,
    }


def _boundary_errors(gold: list[Segment], pred: list[PhoneSegment]) -> list[float]:
    n = min(len(gold), len(pred))
    if n <= 0:
        return []
    errors: list[float] = []
    # Compare starts and ends for each matched phone. This intentionally includes
    # edge boundaries because UTAU offset/cutoff quality depends on them too.
    for idx in range(n):
        errors.append(abs(float(pred[idx].start) - float(gold[idx].start)))
        errors.append(abs(float(pred[idx].end) - float(gold[idx].end)))
    return errors


def _vowel_onset_errors(gold: list[Segment], pred: list[PhoneSegment]) -> list[float]:
    n = min(len(gold), len(pred))
    errors: list[float] = []
    for idx in range(n):
        if str(pred[idx].coarse) == "V":
            errors.append(abs(float(pred[idx].start) - float(gold[idx].start)))
    return errors


def _cv_transition_errors(gold: list[Segment], pred: list[PhoneSegment]) -> list[float]:
    n = min(len(gold), len(pred))
    errors: list[float] = []
    for idx in range(1, n):
        if str(pred[idx].coarse) == "V" and str(pred[idx - 1].coarse).startswith("C_"):
            errors.append(abs(float(pred[idx].start) - float(gold[idx].start)))
    return errors


def _mean(values: list[float]) -> float:
    return float(sum(values) / float(len(values))) if values else float("nan")


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _hit_rate(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return float(sum(1 for value in values if float(value) <= float(threshold))) / float(len(values))


def _aggregate_by(file_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in file_results:
        group = str(row.get(key, "") or "unknown")
        bucket = out.setdefault(
            group,
            {"files": 0, "boundary_errors": [], "vowel_onset_errors": [], "cv_transition_errors": [], "frame_correct": 0, "frame_count": 0},
        )
        bucket["files"] += 1
        bucket["boundary_errors"].extend(row.get("boundary_errors_sec") or [])
        bucket["vowel_onset_errors"].extend(row.get("vowel_onset_errors_sec") or [])
        bucket["cv_transition_errors"].extend(row.get("cv_transition_errors_sec") or [])
        bucket["frame_correct"] += int(row.get("coarse_frame_correct") or 0)
        bucket["frame_count"] += int(row.get("coarse_frame_count") or 0)
    final: dict[str, Any] = {}
    for group, bucket in out.items():
        errors = list(bucket["boundary_errors"])
        frame_count = int(bucket["frame_count"])
        final[group] = {
            "files": int(bucket["files"]),
            "boundary_mae_ms": _mean(errors) * 1000.0 if errors else None,
            "boundary_acc_20ms": _hit_rate(errors, 0.020),
            "boundary_acc_50ms": _hit_rate(errors, 0.050),
            "vowel_onset_mae_ms": _mean(bucket["vowel_onset_errors"]) * 1000.0 if bucket["vowel_onset_errors"] else None,
            "vowel_onset_acc_50ms": _hit_rate(bucket["vowel_onset_errors"], 0.050),
            "cv_transition_mae_ms": _mean(bucket["cv_transition_errors"]) * 1000.0 if bucket["cv_transition_errors"] else None,
            "cv_transition_acc_50ms": _hit_rate(bucket["cv_transition_errors"], 0.050),
            "coarse_frame_acc": float(bucket["frame_correct"]) / float(frame_count) if frame_count else None,
        }
    return final


def write_eval_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = ["AlignmentEvalConfig", "evaluate_manifest", "write_eval_json"]

