"""Zoomable time viewport and per-pixel waveform reduction shared by the review and preview UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MIN_WAVEFORM_ZOOM = 1.0
MAX_WAVEFORM_ZOOM = 64.0


@dataclass(frozen=True)
class WaveformViewport:
    duration_ms: float
    zoom: float = MIN_WAVEFORM_ZOOM
    start_ms: float = 0.0

    def __post_init__(self) -> None:
        duration = max(1e-9, float(self.duration_ms))
        zoom = max(MIN_WAVEFORM_ZOOM, min(MAX_WAVEFORM_ZOOM, float(self.zoom)))
        span = duration / zoom
        start = max(0.0, min(duration - span, float(self.start_ms)))
        object.__setattr__(self, "duration_ms", duration)
        object.__setattr__(self, "zoom", zoom)
        object.__setattr__(self, "start_ms", start)

    @property
    def span_ms(self) -> float:
        return self.duration_ms / self.zoom

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.span_ms

    @property
    def center_ms(self) -> float:
        return self.start_ms + self.span_ms / 2.0

    @property
    def scrollbar_range(self) -> tuple[float, float]:
        return self.start_ms / self.duration_ms, self.end_ms / self.duration_ms

    def time_to_x(self, time_ms: float, width: float) -> float:
        return (float(time_ms) - self.start_ms) / self.span_ms * max(1.0, float(width))

    def x_to_time(self, x: float, width: float) -> float:
        resolved_width = max(1.0, float(width))
        clamped_x = max(0.0, min(resolved_width, float(x)))
        return self.start_ms + clamped_x / resolved_width * self.span_ms

    def contains(self, time_ms: float) -> bool:
        return self.start_ms <= float(time_ms) <= self.end_ms

    def zoomed(
        self,
        zoom: float,
        *,
        anchor_time_ms: float | None = None,
        anchor_fraction: float = 0.5,
    ) -> "WaveformViewport":
        resolved_zoom = max(MIN_WAVEFORM_ZOOM, min(MAX_WAVEFORM_ZOOM, float(zoom)))
        fraction = max(0.0, min(1.0, float(anchor_fraction)))
        anchor = self.center_ms if anchor_time_ms is None else float(anchor_time_ms)
        new_span = self.duration_ms / resolved_zoom
        return WaveformViewport(self.duration_ms, resolved_zoom, anchor - fraction * new_span)

    def shifted(self, delta_ms: float) -> "WaveformViewport":
        return WaveformViewport(self.duration_ms, self.zoom, self.start_ms + float(delta_ms))

    def moved_to_fraction(self, fraction: float) -> "WaveformViewport":
        return WaveformViewport(self.duration_ms, self.zoom, float(fraction) * self.duration_ms)

    def centered_on(self, time_ms: float) -> "WaveformViewport":
        return WaveformViewport(self.duration_ms, self.zoom, float(time_ms) - self.span_ms / 2.0)


def visible_waveform_columns(
    envelope: Sequence[Sequence[float]],
    viewport: WaveformViewport,
    *,
    canvas_width: int,
) -> tuple[tuple[float, float, float], ...]:
    """Reduce the visible envelope to at most one min/max segment per pixel."""

    values = np.asarray(envelope, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        return tuple()
    width = max(1, int(canvas_width))
    count = values.shape[0]
    start_index = max(0, min(count - 1, int(np.floor(viewport.start_ms / viewport.duration_ms * count))))
    end_index = max(start_index + 1, min(count, int(np.ceil(viewport.end_ms / viewport.duration_ms * count))))
    visible_count = end_index - start_index
    column_count = min(width, visible_count)
    edges = np.linspace(start_index, end_index, column_count + 1, dtype=np.int64)
    result: list[tuple[float, float, float]] = []
    for index in range(column_count):
        left = int(edges[index])
        right = max(left + 1, int(edges[index + 1]))
        block = values[left:right]
        time_ms = (left + right - 1) / 2.0 / max(1, count - 1) * viewport.duration_ms
        result.append((viewport.time_to_x(time_ms, width), float(np.min(block[:, 0])), float(np.max(block[:, 1]))))
    return tuple(result)
