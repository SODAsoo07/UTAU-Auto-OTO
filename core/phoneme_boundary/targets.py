from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.phoneme_boundary.types import (
    BOUNDARY_LABELS,
    CONSONANT_LABELS,
    BoundaryEvent,
    IGNORE_INDEX,
    PHONE_STATE_LABELS,
    VOWEL_LABELS,
    event_from_mapping,
    label_sigma_ms,
    normalize_boundary_label,
)
from core.phoneme_boundary.phones import coarse_for_phone


@dataclass(frozen=True)
class PhoneAuxTargetMap:
    state_target: np.ndarray
    state_mask: np.ndarray
    consonant_target: np.ndarray
    consonant_mask: np.ndarray
    vowel_target: np.ndarray
    vowel_mask: np.ndarray


def build_boundary_target_map(
    events: list[BoundaryEvent],
    *,
    duration_ms: float,
    hop_ms: float,
    frame_count: int | None = None,
    labels: tuple[str, ...] = BOUNDARY_LABELS,
) -> tuple[list[float], np.ndarray]:
    hop = max(1e-6, float(hop_ms))
    if frame_count is None:
        frame_count = max(1, int(round(float(duration_ms) / hop)) + 1)
    frames = int(max(0, frame_count))
    times = (np.arange(frames, dtype=np.float32) * hop).tolist()
    target = np.zeros((frames, len(labels)), dtype=np.float32)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    if frames <= 0:
        return times, target
    for event in events:
        label = normalize_boundary_label(event.label)
        idx = label_to_idx.get(label)
        if idx is None:
            continue
        center = float(event.time_ms)
        sigma = max(1.0, label_sigma_ms(label))
        confidence = max(0.0, min(1.0, float(event.confidence)))
        lo = max(0, int((center - 4.0 * sigma) // hop))
        hi = min(frames - 1, int((center + 4.0 * sigma) // hop) + 1)
        for frame_idx in range(lo, hi + 1):
            dt = (frame_idx * hop) - center
            value = confidence * float(np.exp(-0.5 * (dt / sigma) ** 2))
            if value > float(target[frame_idx, idx]):
                target[frame_idx, idx] = value
    return times, target


def events_from_manifest_row(row: dict[str, Any]) -> list[BoundaryEvent]:
    events: list[BoundaryEvent] = []
    for key in ("events", "boundaries", "boundary_events"):
        raw_items = row.get(key)
        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, dict):
                    event = event_from_mapping(item)
                    if event is not None:
                        events.append(event)
    events.extend(_events_from_explicit_time_fields(row))
    events.extend(_events_from_phone_intervals(row))
    return _dedupe_events(events)


def build_phone_aux_target_map(
    row: dict[str, Any],
    *,
    frame_count: int,
    hop_ms: float,
    phone_state_labels: tuple[str, ...] = PHONE_STATE_LABELS,
    consonant_labels: tuple[str, ...] = CONSONANT_LABELS,
    vowel_labels: tuple[str, ...] = VOWEL_LABELS,
) -> PhoneAuxTargetMap:
    frames = int(max(0, frame_count))
    state_target = np.full((frames,), IGNORE_INDEX, dtype=np.int64)
    state_mask = np.zeros((frames,), dtype=np.float32)
    consonant_target = np.full((frames,), IGNORE_INDEX, dtype=np.int64)
    consonant_mask = np.zeros((frames,), dtype=np.float32)
    vowel_target = np.full((frames,), IGNORE_INDEX, dtype=np.int64)
    vowel_mask = np.zeros((frames,), dtype=np.float32)
    phones = row.get("phones", row.get("phone_intervals", None))
    if not isinstance(phones, list) or frames <= 0:
        return PhoneAuxTargetMap(state_target, state_mask, consonant_target, consonant_mask, vowel_target, vowel_mask)
    state_to_idx = {label: idx for idx, label in enumerate(phone_state_labels)}
    consonant_to_idx = {label: idx for idx, label in enumerate(consonant_labels)}
    vowel_to_idx = {label: idx for idx, label in enumerate(vowel_labels)}
    language = str(row.get("language", "") or "")
    hop = max(1e-6, float(hop_ms))
    for item in phones:
        if not isinstance(item, dict):
            continue
        phone = str(item.get("phone", item.get("label", "")) or "")
        try:
            start_ms = float(item.get("start_ms", item.get("begin_ms", item.get("start", 0.0))))
            end_ms = float(item.get("end_ms", item.get("stop_ms", item.get("end", start_ms))))
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        coarse = coarse_for_phone(phone, language=language)
        state = "silence"
        if coarse.startswith("C_"):
            state = "consonant"
        elif coarse == "V":
            state = "vowel"
        state_idx = state_to_idx.get(state)
        lo = max(0, int(start_ms // hop))
        hi = min(frames - 1, int(end_ms // hop))
        if state_idx is not None:
            state_target[lo : hi + 1] = int(state_idx)
            state_mask[lo : hi + 1] = 1.0
        if state == "consonant":
            idx = consonant_to_idx.get(_clean_phone_label(phone), consonant_to_idx.get("other"))
            if idx is not None:
                consonant_target[lo : hi + 1] = int(idx)
                consonant_mask[lo : hi + 1] = 1.0
        elif state == "vowel":
            idx = vowel_to_idx.get(_clean_phone_label(phone), vowel_to_idx.get("other"))
            if idx is not None:
                vowel_target[lo : hi + 1] = int(idx)
                vowel_mask[lo : hi + 1] = 1.0
    return PhoneAuxTargetMap(state_target, state_mask, consonant_target, consonant_mask, vowel_target, vowel_mask)


def read_boundary_manifest(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    if str(path).lower().endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows
    with open(path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("items", None))
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return [payload]
    return []


def resolve_wav_path(row: dict[str, Any], *, manifest_dir: str = "") -> str:
    raw = str(row.get("wav_path", row.get("wav", row.get("audio", ""))) or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    if manifest_dir:
        return os.path.abspath(os.path.join(manifest_dir, raw))
    return os.path.abspath(raw)


def _events_from_explicit_time_fields(row: dict[str, Any]) -> list[BoundaryEvent]:
    mappings = {
        "phone_start_ms": "phone_start",
        "phone_end_ms": "phone_end",
        "consonant_onset_ms": "consonant_onset",
        "cv_vowel_onset_ms": "vowel_onset",
        "vowel_onset_ms": "vowel_onset",
        "vowel_nucleus_ms": "vowel_nucleus",
        "vv_vowel_nucleus_ms": "vowel_nucleus",
        "vowel_end_ms": "vowel_end",
        "silence_boundary_ms": "silence_boundary",
    }
    out: list[BoundaryEvent] = []
    for key, label in mappings.items():
        if key not in row:
            continue
        try:
            time_ms = float(row[key])
        except Exception:
            continue
        out.append(
            BoundaryEvent(
                label=label,
                time_ms=max(0.0, time_ms),
                alias_index=_safe_int(row.get("alias_index", row.get("line_index", -1)), -1),
                phone_index=_safe_int(row.get("phone_index", -1), -1),
                phone=str(row.get("phone", "") or ""),
                confidence=max(0.0, min(1.0, _safe_float(row.get("confidence", 1.0), 1.0))),
                source="explicit",
            )
        )
    return out


def _events_from_phone_intervals(row: dict[str, Any]) -> list[BoundaryEvent]:
    phones = row.get("phones", row.get("phone_intervals", None))
    if not isinstance(phones, list):
        return []
    language = str(row.get("language", "") or "")
    out: list[BoundaryEvent] = []
    for idx, item in enumerate(phones):
        if not isinstance(item, dict):
            continue
        phone = str(item.get("phone", item.get("label", "")) or "")
        try:
            start_ms = float(item.get("start_ms", item.get("begin_ms", item.get("start", 0.0))))
            end_ms = float(item.get("end_ms", item.get("stop_ms", item.get("end", start_ms))))
        except Exception:
            continue
        confidence = max(0.0, min(1.0, _safe_float(item.get("confidence", 1.0), 1.0)))
        alias_index = _safe_int(item.get("alias_index", row.get("alias_index", -1)), -1)
        coarse = coarse_for_phone(phone, language=language)
        out.append(BoundaryEvent("phone_start", start_ms, alias_index, idx, phone, confidence, "phone_interval"))
        out.append(BoundaryEvent("phone_end", end_ms, alias_index, idx, phone, confidence, "phone_interval"))
        if coarse.startswith("C_"):
            out.append(BoundaryEvent("consonant_onset", start_ms, alias_index, idx, phone, confidence, "phone_interval"))
        elif coarse == "V":
            nucleus = item.get("nucleus_ms", item.get("center_ms", None))
            if nucleus is None:
                nucleus_ms = 0.5 * (start_ms + end_ms)
            else:
                nucleus_ms = _safe_float(nucleus, 0.5 * (start_ms + end_ms))
            out.append(BoundaryEvent("vowel_onset", start_ms, alias_index, idx, phone, confidence, "phone_interval"))
            out.append(BoundaryEvent("vowel_nucleus", nucleus_ms, alias_index, idx, phone, confidence, "phone_interval"))
            out.append(BoundaryEvent("vowel_end", end_ms, alias_index, idx, phone, confidence, "phone_interval"))
        elif coarse in {"SIL", "BR"}:
            out.append(BoundaryEvent("silence_boundary", start_ms, alias_index, idx, phone, confidence, "phone_interval"))
            out.append(BoundaryEvent("silence_boundary", end_ms, alias_index, idx, phone, confidence, "phone_interval"))
    return out


def _dedupe_events(events: list[BoundaryEvent]) -> list[BoundaryEvent]:
    best: dict[tuple[str, int, int, int], BoundaryEvent] = {}
    for event in events:
        label = normalize_boundary_label(event.label)
        if label not in BOUNDARY_LABELS:
            continue
        bucket = int(round(float(event.time_ms) / 2.0))
        key = (label, int(event.alias_index), int(event.phone_index), bucket)
        prev = best.get(key)
        normalized = BoundaryEvent(
            label=label,
            time_ms=max(0.0, float(event.time_ms)),
            alias_index=int(event.alias_index),
            phone_index=int(event.phone_index),
            phone=str(event.phone or ""),
            confidence=max(0.0, min(1.0, float(event.confidence))),
            source=str(event.source or ""),
            meta=dict(event.meta or {}),
        )
        if prev is None or normalized.confidence > prev.confidence:
            best[key] = normalized
    return sorted(best.values(), key=lambda item: (float(item.time_ms), item.label))


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clean_phone_label(phone: object) -> str:
    text = str(phone or "").strip().lower()
    if "/" in text:
        text = text.split("/", 1)[-1]
    return text or "other"


__all__ = [
    "build_boundary_target_map",
    "build_phone_aux_target_map",
    "events_from_manifest_row",
    "PhoneAuxTargetMap",
    "read_boundary_manifest",
    "resolve_wav_path",
]
