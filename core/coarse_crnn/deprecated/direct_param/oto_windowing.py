from __future__ import annotations

from typing import Any

import numpy as np
from core.coarse_crnn.alias_role import normalize_role


def should_use_vcv_target_window(
    format_type: object,
    *,
    enabled: bool = True,
    formats: tuple[str, ...] | list[str] | None = None,
    alias_type: object = "",
    alias_role: object = "",
) -> bool:
    fmt = str(format_type or "").strip().lower()
    if fmt == "cvvc":
        return False
    allowed = tuple(str(item or "").strip().lower() for item in (formats or ("vcv",)))
    return bool(enabled) and fmt in allowed


def target_window_frames_for(
    config: object,
    format_type: object,
    default_frames: int,
    *,
    alias_role: object = "",
) -> int:
    fmt = str(format_type or "").strip().lower()
    role_text = normalize_role(alias_role)
    frames = int(default_frames)
    for item in getattr(config, "target_window_role_frame_overrides", ()) or ():
        text = str(item or "").strip()
        if "=" not in text:
            continue
        key, raw = text.split("=", 1)
        key_text = key.strip().lower()
        key_fmt = ""
        key_role = ""
        if "|" in key_text:
            key_fmt, key_role = key_text.split("|", 1)
            key_fmt = key_fmt.strip().lower()
            key_role = normalize_role(key_role)
        else:
            key_fmt = fmt
            key_role = normalize_role(key_text)
        if key_fmt != fmt or key_role != role_text:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            return max(1, frames)
    for item in getattr(config, "target_window_frame_overrides", ()) or ():
        text = str(item or "").strip()
        if "=" not in text:
            continue
        key, raw = text.split("=", 1)
        if key.strip().lower() != fmt:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            return max(1, frames)
    return max(1, frames)


def crop_oto_target_window(
    features: np.ndarray,
    anchors_ms: np.ndarray | None,
    *,
    hop_sec: float,
    duration_ms: float,
    row_index_in_wav: int,
    file_row_count: int,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray | None, float, int]:
    n = int(features.shape[0])
    width = int(window_frames)
    if width <= 0 or n <= width:
        clipped = _clip_anchors(anchors_ms, duration_ms=duration_ms) if anchors_ms is not None else None
        return features, clipped, max(float(duration_ms), 1.0), 0
    start = target_window_start_frame(
        frame_count=n,
        row_index_in_wav=row_index_in_wav,
        file_row_count=file_row_count,
        window_frames=width,
    )
    end = start + width
    hop_ms = max(float(hop_sec) * 1000.0, 1e-3)
    crop_duration_ms = float(width) * hop_ms
    if anchors_ms is None:
        return features[start:end], None, crop_duration_ms, int(start)
    shifted = np.asarray(anchors_ms, dtype=np.float32) - float(start) * hop_ms
    shifted = np.clip(shifted, 0.0, crop_duration_ms).astype(np.float32)
    return features[start:end], shifted, crop_duration_ms, int(start)


def target_window_start_frame(
    *,
    frame_count: int,
    row_index_in_wav: int,
    file_row_count: int,
    window_frames: int,
) -> int:
    n = max(1, int(frame_count))
    width = max(1, min(int(window_frames), n))
    count = max(1, int(file_row_count))
    idx = max(0, min(count - 1, int(row_index_in_wav)))
    if count <= 1:
        center = float(n) * 0.5
    else:
        center = (float(idx) + 0.5) * float(n) / float(count)
    start = int(round(center - float(width) * 0.5))
    return max(0, min(n - width, start))


def row_window_args(row: dict[str, Any]) -> tuple[int, int]:
    return (
        int(float(row.get("row_index_in_wav", 0) or 0)),
        max(1, int(float(row.get("file_row_count", 1) or 1))),
    )


def _clip_anchors(anchors_ms: np.ndarray, *, duration_ms: float) -> np.ndarray:
    return np.clip(np.asarray(anchors_ms, dtype=np.float32), 0.0, max(float(duration_ms), 1.0)).astype(np.float32)


__all__ = [
    "crop_oto_target_window",
    "row_window_args",
    "should_use_vcv_target_window",
    "target_window_frames_for",
    "target_window_start_frame",
]
