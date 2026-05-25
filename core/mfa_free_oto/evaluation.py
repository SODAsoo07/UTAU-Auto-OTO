from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .decode import decode_monotonic_events
from .features import extract_features, feature_timebase_metadata
from .manifest import load_manifest_jsonl
from .metrics import boundary_error_metrics, frame_classification_metrics, gate_report, row_failure_metrics
from .model import MfaFreeFrameModelConfig, build_frame_model
from .slot_viterbi import assign_slots_viterbi, slot_assignments_to_decoded_events
from .targets import IGNORE_INDEX, rasterize_targets
from .types import EVENT_LABELS, FRAME_LABELS, FramePosterior


def load_checkpoint(checkpoint_path: str | Path, *, device: str | None = None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint inference requires torch.") from exc
    checkpoint = torch.load(str(checkpoint_path), map_location=device or "cpu")
    cfg = MfaFreeFrameModelConfig.from_dict(checkpoint["model_config"])
    model = build_frame_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()
    return checkpoint, model, target_device


def predict_row_posterior(
    row: dict,
    checkpoint_path: str | Path,
    *,
    encoder: str | None = None,
    device: str | None = None,
) -> FramePosterior:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint inference requires torch.") from exc
    checkpoint, model, target_device = load_checkpoint(checkpoint_path, device=device)
    encoder_name = encoder or str(checkpoint.get("encoder") or "acoustic")
    batch = extract_features(row["wav_path"], encoder=encoder_name, device=device)
    with torch.no_grad():
        x = torch.from_numpy(batch.features[None, :, :]).float().to(target_device)
        output = model(x)
        frame_probs = torch.softmax(output["frame_logits"], dim=-1)[0].detach().cpu().numpy()
        event_scores = torch.sigmoid(output["event_logits"])[0].detach().cpu().numpy()
    return FramePosterior(
        wav_path=row["wav_path"],
        times_ms=batch.times_ms.tolist(),
        class_probs={label: frame_probs[:, idx].astype(float).tolist() for idx, label in enumerate(FRAME_LABELS)},
        event_scores={label: event_scores[:, idx].astype(float).tolist() for idx, label in enumerate(EVENT_LABELS)},
        acoustic_scores={key: value.astype(float).tolist() for key, value in batch.acoustic_scores.items()},
        metadata={"encoder": encoder_name, **feature_timebase_metadata(batch)},
    )


def evaluate_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    *,
    encoder: str | None = None,
    device: str | None = None,
    boundary_tolerance_ms: float = 100.0,
    nucleus_tolerance_ms: float = 100.0,
    use_slot_viterbi: bool = True,
) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint evaluation requires torch.") from exc
    rows = load_manifest_jsonl(manifest_path, require_manual=True, require_labels=True)
    checkpoint, model, target_device = load_checkpoint(checkpoint_path, device=device)
    encoder_name = encoder or str(checkpoint.get("encoder") or "acoustic")
    y_true: list[int] = []
    y_pred: list[int] = []
    all_reference_cv: list[float] = []
    all_predicted_cv: list[float] = []
    all_reference_vn: list[float] = []
    all_predicted_vn: list[float] = []
    row_results: list[dict] = []
    predictions: list[dict] = []
    for row in rows:
        try:
            batch = extract_features(row["wav_path"], encoder=encoder_name, device=device)
            targets = rasterize_targets(row, batch.times_ms)
            with torch.no_grad():
                x = torch.from_numpy(batch.features[None, :, :]).float().to(target_device)
                output = model(x)
                frame_probs = torch.softmax(output["frame_logits"], dim=-1)[0].detach().cpu().numpy()
                event_scores = torch.sigmoid(output["event_logits"])[0].detach().cpu().numpy()
            valid = targets.frame_class != IGNORE_INDEX
            y_true.extend(targets.frame_class[valid].astype(int).tolist())
            y_pred.extend(np.argmax(frame_probs[valid], axis=-1).astype(int).tolist())
            posterior = FramePosterior(
                wav_path=row["wav_path"],
                times_ms=batch.times_ms.tolist(),
                class_probs={label: frame_probs[:, idx].astype(float).tolist() for idx, label in enumerate(FRAME_LABELS)},
                event_scores={label: event_scores[:, idx].astype(float).tolist() for idx, label in enumerate(EVENT_LABELS)},
                acoustic_scores={key: value.astype(float).tolist() for key, value in batch.acoustic_scores.items()},
                metadata={"encoder": encoder_name, **feature_timebase_metadata(batch)},
            )
            expected_phones = _expected_phones(row)
            slot_result = assign_slots_viterbi(posterior, expected_phones=expected_phones) if use_slot_viterbi and expected_phones else None
            decoded = (
                slot_assignments_to_decoded_events(slot_result)
                if slot_result is not None and slot_result.assignments
                else decode_monotonic_events(posterior, expected_phones=expected_phones)
            )
            reference_cv = [
                float(event["time_ms"])
                for event in row.get("events") or []
                if str(event.get("label")) == "cv_boundary"
            ]
            predicted_cv = [event.selected_time_ms for event in decoded if event.label == "cv_boundary"]
            reference_vn = [
                float(event["time_ms"])
                for event in row.get("events") or []
                if str(event.get("label")) == "vowel_nucleus"
            ]
            predicted_vn = [event.selected_time_ms for event in decoded if event.label == "vowel_nucleus"]
            all_reference_cv.extend(reference_cv)
            all_predicted_cv.extend(predicted_cv)
            all_reference_vn.extend(reference_vn)
            all_predicted_vn.extend(predicted_vn)
            row_error = boundary_error_metrics(reference_cv, predicted_cv, tolerance_ms=boundary_tolerance_ms)
            row_nucleus_error = boundary_error_metrics(reference_vn, predicted_vn, tolerance_ms=nucleus_tolerance_ms)
            hard_failure = bool(reference_cv and not predicted_cv)
            catastrophic = hard_failure or (
                row_error["median_error_ms"] is not None and float(row_error["median_error_ms"]) > boundary_tolerance_ms
            )
            failure_tags = _failure_tags(row_error, slot_result, reference_cv, predicted_cv)
            row_results.append(
                {
                    "row_id": row.get("row_id") or row.get("wav_name"),
                    "hard_failure": hard_failure,
                    "catastrophic": catastrophic,
                    "failure_tags": failure_tags,
                    "cv_boundary": row_error,
                    "vowel_nucleus": row_nucleus_error,
                }
            )
            predictions.append(
                {
                    "row_id": row.get("row_id") or row.get("wav_name"),
                    "times_ms": posterior.times_ms,
                    "class_probs": posterior.class_probs,
                    "event_scores": posterior.event_scores,
                    "acoustic_scores": posterior.acoustic_scores,
                    "decoded_events": [
                        {
                            "label": event.label,
                            "selected_time_ms": event.selected_time_ms,
                            "score": event.score,
                            "expected_phone": event.expected_phone,
                        }
                        for event in decoded
                    ],
                    "slot_assignments": [
                        {
                            "slot_index": assignment.slot_index,
                            "phone_index": assignment.phone_index,
                            "phone": assignment.phone,
                            "role": assignment.role,
                            "event_label": assignment.event_label,
                            "selected_time_ms": assignment.selected_time_ms,
                            "score": assignment.score,
                            "frame_index": assignment.frame_index,
                            "expected_time_ms": assignment.expected_time_ms,
                        }
                        for assignment in (slot_result.assignments if slot_result is not None else ())
                    ],
                    "slot_warnings": list(slot_result.warnings if slot_result is not None else ()),
                }
            )
        except Exception as exc:  # Keep evaluation usable on partially broken goldsets.
            row_results.append(
                {
                    "row_id": row.get("row_id") or row.get("wav_name"),
                    "hard_failure": True,
                    "catastrophic": True,
                    "error": str(exc),
                }
            )
    frame_metrics = frame_classification_metrics(y_true, y_pred, labels=FRAME_LABELS)
    boundary_metrics = boundary_error_metrics(
        all_reference_cv,
        all_predicted_cv,
        tolerance_ms=boundary_tolerance_ms,
    )
    nucleus_metrics = boundary_error_metrics(
        all_reference_vn,
        all_predicted_vn,
        tolerance_ms=nucleus_tolerance_ms,
    )
    failure_metrics = row_failure_metrics(row_results)
    metrics = {
        "rows": len(rows),
        "encoder": encoder_name,
        "use_slot_viterbi": use_slot_viterbi,
        "frame_macro_f1": frame_metrics["macro_f1"],
        "frame": frame_metrics,
        "cv_boundary_median_error_ms": boundary_metrics["median_error_ms"],
        "cv_boundary_p90_error_ms": boundary_metrics["p90_error_ms"],
        "cv_boundary": boundary_metrics,
        "vowel_nucleus_median_error_ms": nucleus_metrics["median_error_ms"],
        "vowel_nucleus_p90_error_ms": nucleus_metrics["p90_error_ms"],
        "vowel_nucleus": nucleus_metrics,
        **failure_metrics,
        "gate_report": {},
        "row_results": row_results,
        "predictions": predictions,
    }
    gate_input = dict(metrics)
    gate_input["cv_boundary_median_error_ms"] = gate_input["cv_boundary_median_error_ms"] or 1e9
    gate_input["cv_boundary_p90_error_ms"] = gate_input["cv_boundary_p90_error_ms"] or 1e9
    metrics["gate_report"] = gate_report(gate_input)
    return metrics


def save_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _expected_phones(row: dict) -> list[str]:
    phones: list[str] = []
    for item in row.get("expected_phones") or []:
        if isinstance(item, dict):
            phone = item.get("phone")
            if phone:
                phones.append(str(phone))
        else:
            phones.append(str(item))
    return phones


def _failure_tags(row_error: dict, slot_result, reference_cv: Sequence[float], predicted_cv: Sequence[float]) -> list[str]:
    tags: set[str] = set()
    warnings = list(slot_result.warnings if slot_result is not None else ())
    for warning in warnings:
        if warning.startswith("low_energy_slot"):
            tags.add("low_energy")
        elif warning.startswith("weak_flux_slot"):
            tags.add("weak_flux")
        elif warning.startswith("weak_voicing_slot"):
            tags.add("weak_voicing")
        elif warning.startswith("tight_gap"):
            tags.add("slot_gap")
        elif warning.startswith("low_score_slot"):
            tags.add("low_posterior_slot")
        elif warning.startswith("hard_"):
            tags.add(warning.split(":", 1)[0])
    if row_error.get("missed", 0):
        tags.add("missed_slot")
    if row_error.get("false_positive", 0):
        tags.add("false_slot")
    median = row_error.get("median_error_ms")
    if reference_cv and predicted_cv and len(reference_cv) == len(predicted_cv) and median is not None and float(median) >= 40.0:
        tags.add("possible_one_step_shift")
    if row_error.get("missed", 0) and row_error.get("false_positive", 0):
        tags.add("posterior_conflict")
    return sorted(tags)
