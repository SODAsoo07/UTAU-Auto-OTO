from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import EVENT_LABELS, EVENT_LABEL_TO_ID, FRAME_LABELS, FRAME_LABEL_TO_ID

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingTargets:
    frame_class: np.ndarray
    frame_mask: np.ndarray
    event_targets: np.ndarray
    slot_event_targets: np.ndarray
    slot_event_mask: np.ndarray


def frame_times_ms(frame_count: int, hop_ms: float) -> np.ndarray:
    return np.arange(frame_count, dtype=np.float32) * float(hop_ms)


def rasterize_targets(
    row: dict,
    times_ms: Sequence[float],
    *,
    event_sigma_ms: float = 18.0,
    slot_event_bins: int = 0,
) -> TrainingTargets:
    times = np.asarray(times_ms, dtype=np.float32)
    frame_class = np.full((times.shape[0],), IGNORE_INDEX, dtype=np.int64)
    frame_mask = np.zeros((times.shape[0],), dtype=np.float32)
    for segment in row.get("frame_labels") or []:
        label = str(segment["label"])
        start_ms = float(segment["start_ms"])
        end_ms = float(segment["end_ms"])
        confidence = float(segment.get("confidence", 1.0))
        mask = (times >= start_ms) & (times < end_ms)
        frame_class[mask] = FRAME_LABEL_TO_ID[label]
        frame_mask[mask] = max(0.0, min(1.0, confidence))

    event_targets = np.zeros((times.shape[0], len(EVENT_LABELS)), dtype=np.float32)
    slot_bins = max(0, int(slot_event_bins))
    slot_event_targets = np.zeros(
        (times.shape[0], slot_bins, len(EVENT_LABELS)),
        dtype=np.float32,
    )
    slot_event_mask = np.zeros((slot_bins,), dtype=np.float32)
    sigma = max(1.0, float(event_sigma_ms))
    events = list(row.get("events") or [])
    inferred_slots = infer_event_slot_indices(events)
    for event, inferred_slot in zip(events, inferred_slots):
        label = str(event["label"])
        time_ms = float(event["time_ms"])
        confidence = float(event.get("confidence", 1.0))
        col = EVENT_LABEL_TO_ID[label]
        peak = np.exp(-0.5 * ((times - time_ms) / sigma) ** 2).astype(np.float32)
        event_targets[:, col] = np.maximum(event_targets[:, col], peak * confidence)
        slot_index = _event_slot_index(event, inferred_slot)
        if 0 <= slot_index < slot_bins:
            slot_event_targets[:, slot_index, col] = np.maximum(
                slot_event_targets[:, slot_index, col],
                peak * confidence,
            )
            slot_event_mask[slot_index] = 1.0
    return TrainingTargets(
        frame_class=frame_class,
        frame_mask=frame_mask,
        event_targets=event_targets,
        slot_event_targets=slot_event_targets,
        slot_event_mask=slot_event_mask,
    )


def infer_event_slot_indices(events: Sequence[dict]) -> list[int]:
    out: list[int] = []
    current_slot = -1
    for event in events:
        explicit = event.get("slot_index")
        if explicit is not None:
            try:
                current_slot = max(0, int(explicit))
                out.append(current_slot)
                continue
            except (TypeError, ValueError):
                pass
        current_slot += 1
        out.append(current_slot)
    return out


def _event_slot_index(event: dict, inferred_slot: int) -> int:
    try:
        return int(event.get("slot_index", inferred_slot))
    except (TypeError, ValueError):
        return int(inferred_slot)


def labels_from_class_ids(class_ids: Sequence[int]) -> list[str]:
    labels: list[str] = []
    for class_id in class_ids:
        if int(class_id) == IGNORE_INDEX:
            labels.append("ignore")
        else:
            labels.append(FRAME_LABELS[int(class_id)])
    return labels
