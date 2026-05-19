from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import read_wav_mono, spectrogram_image
from .types import DecodedEvent, EVENT_LABELS, FRAME_LABELS, FramePosterior

FRAME_COLORS = {
    "silence": "#9ca3af",
    "consonant": "#f97316",
    "vowel": "#2563eb",
    "other": "#8b5cf6",
}
EVENT_COLORS = {
    "cv_boundary": "#dc2626",
    "vowel_nucleus": "#16a34a",
    "phone_change": "#0f766e",
}


def render_review_html(
    row: dict,
    *,
    posterior: FramePosterior | None = None,
    decoded_events: Sequence[DecodedEvent | dict] | None = None,
    generated_oto_rows: Sequence[dict] | None = None,
    width: int = 1280,
) -> str:
    wav_path = str(row["wav_path"])
    samples, sample_rate = read_wav_mono(wav_path, target_sample_rate=16000)
    duration_ms = max(1.0, 1000.0 * float(samples.shape[0]) / float(sample_rate))
    spec_times, spec = spectrogram_image(samples, sample_rate, bins=48)
    waveform_svg = _render_waveform(samples, duration_ms, width=width, height=120)
    spectrogram_svg = _render_spectrogram(spec_times, spec, duration_ms, width=width, height=180)
    labels_svg = _render_frame_labels(row.get("frame_labels") or [], duration_ms, width=width, height=56)
    events_svg = _render_events(row.get("events") or [], duration_ms, width=width, height=72)
    posterior_svg = _render_posterior(posterior, duration_ms, width=width, height=160) if posterior else ""
    decoded_svg = _render_decoded_events(decoded_events or [], duration_ms, width=width, height=64)
    generated_oto_svg = _render_generated_oto_rows(generated_oto_rows or [], duration_ms, width=width, height=92)
    row_json = html.escape(json.dumps(_compact_row(row), ensure_ascii=False, indent=2))
    title = html.escape(str(row.get("row_id") or row.get("wav_name") or wav_path))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MFA-free OTO review - {title}</title>
<style>
body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; color: #111827; background: #f8fafc; }}
main {{ max-width: {width + 48}px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 20px; margin: 0 0 16px; }}
h2 {{ font-size: 14px; margin: 20px 0 8px; color: #374151; }}
svg {{ display: block; width: 100%; background: white; border: 1px solid #d1d5db; }}
pre {{ background: #111827; color: #e5e7eb; padding: 12px; overflow: auto; font-size: 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; margin-bottom: 8px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: -2px; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<div class="legend">{_legend_html()}</div>
<h2>Waveform</h2>
{waveform_svg}
<h2>Spectrogram</h2>
{spectrogram_svg}
<h2>Manual frame labels</h2>
{labels_svg}
<h2>Manual events</h2>
{events_svg}
{posterior_svg}
<h2>Decoded OTO anchors</h2>
{decoded_svg}
<h2>Generated OTO params</h2>
{generated_oto_svg}
<h2>Manifest row</h2>
<pre>{row_json}</pre>
</main>
</body>
</html>
"""


def write_review_html(
    path: str | Path,
    row: dict,
    *,
    posterior: FramePosterior | None = None,
    decoded_events: Sequence[DecodedEvent | dict] | None = None,
    generated_oto_rows: Sequence[dict] | None = None,
    width: int = 1280,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_review_html(
            row,
            posterior=posterior,
            decoded_events=decoded_events,
            generated_oto_rows=generated_oto_rows,
            width=width,
        ),
        encoding="utf-8",
    )


def posterior_from_json(data: dict, wav_path: str) -> FramePosterior:
    return FramePosterior(
        wav_path=wav_path,
        times_ms=data["times_ms"],
        class_probs=data.get("class_probs") or {},
        event_scores=data.get("event_scores") or {},
        acoustic_scores=data.get("acoustic_scores") or {},
        metadata=data.get("metadata") or {},
    )


def _render_waveform(samples: np.ndarray, duration_ms: float, *, width: int, height: int) -> str:
    bins = max(1, min(width, 900))
    edges = np.linspace(0, samples.shape[0], bins + 1, dtype=np.int64)
    points: list[str] = []
    center = height / 2.0
    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        segment = samples[start:end]
        amp = float(np.max(np.abs(segment))) if segment.size else 0.0
        x = idx * width / max(1, bins - 1)
        y_top = center - amp * center
        y_bottom = center + amp * center
        points.append(f'<line x1="{x:.2f}" y1="{y_top:.2f}" x2="{x:.2f}" y2="{y_bottom:.2f}" stroke="#111827" stroke-width="1"/>')
    ticks = _time_ticks(duration_ms, width, height)
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(points)}{ticks}</svg>'


def _render_spectrogram(times_ms: np.ndarray, spec: np.ndarray, duration_ms: float, *, width: int, height: int) -> str:
    if spec.size == 0:
        return f'<svg viewBox="0 0 {width} {height}"></svg>'
    frames, bins = spec.shape
    cell_w = width / max(1, frames)
    cell_h = height / max(1, bins)
    rects: list[str] = []
    for t_idx in range(frames):
        for b_idx in range(bins):
            value = float(spec[t_idx, b_idx])
            shade = int(255 - value * 230)
            color = f"rgb({shade},{shade},{shade})"
            x = t_idx * cell_w
            y = height - (b_idx + 1) * cell_h
            rects.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.5:.2f}" height="{cell_h + 0.5:.2f}" fill="{color}"/>')
    ticks = _time_ticks(duration_ms, width, height)
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(rects)}{ticks}</svg>'


def _render_frame_labels(labels: Sequence[dict], duration_ms: float, *, width: int, height: int) -> str:
    rects: list[str] = []
    for segment in labels:
        label = str(segment.get("label"))
        x = float(segment.get("start_ms", 0.0)) / duration_ms * width
        end_x = float(segment.get("end_ms", 0.0)) / duration_ms * width
        color = FRAME_COLORS.get(label, "#64748b")
        rects.append(f'<rect x="{x:.2f}" y="0" width="{max(1.0, end_x - x):.2f}" height="{height}" fill="{color}" opacity="0.72"/>')
        rects.append(f'<text x="{x + 4:.2f}" y="{height / 2 + 4:.2f}" font-size="12" fill="white">{html.escape(label)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(rects)}{_time_ticks(duration_ms, width, height)}</svg>'


def _render_events(events: Sequence[dict], duration_ms: float, *, width: int, height: int) -> str:
    items: list[str] = []
    for event in events:
        label = str(event.get("label"))
        time_ms = float(event.get("time_ms", 0.0))
        x = time_ms / duration_ms * width
        color = EVENT_COLORS.get(label, "#111827")
        items.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" stroke="{color}" stroke-width="2"/>')
        items.append(f'<text x="{x + 4:.2f}" y="16" font-size="12" fill="{color}">{html.escape(label)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(items)}{_time_ticks(duration_ms, width, height)}</svg>'


def _render_posterior(posterior: FramePosterior, duration_ms: float, *, width: int, height: int) -> str:
    rows: list[str] = ['<h2>Predicted posteriors</h2>']
    per_track = height / float(len(FRAME_LABELS) + len(EVENT_LABELS))
    y = 0.0
    curves: list[str] = []
    for label in FRAME_LABELS:
        curves.append(_posterior_polyline(posterior.times_ms, posterior.class_probs.get(label, []), duration_ms, width, y, per_track, FRAME_COLORS.get(label, "#111827"), label))
        y += per_track
    for label in EVENT_LABELS:
        curves.append(_posterior_polyline(posterior.times_ms, posterior.event_scores.get(label, []), duration_ms, width, y, per_track, EVENT_COLORS.get(label, "#111827"), label))
        y += per_track
    rows.append(f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(curves)}{_time_ticks(duration_ms, width, height)}</svg>')
    return "\n".join(rows)


def _posterior_polyline(times: Sequence[float], values: Sequence[float], duration_ms: float, width: int, y: float, row_h: float, color: str, label: str) -> str:
    vals = list(values)
    if not vals:
        return ""
    pts = []
    for time_ms, value in zip(times, vals):
        x = float(time_ms) / duration_ms * width
        yy = y + row_h - max(0.0, min(1.0, float(value))) * (row_h - 6.0)
        pts.append(f"{x:.2f},{yy:.2f}")
    text = f'<text x="4" y="{y + 12:.2f}" font-size="11" fill="{color}">{html.escape(label)}</text>'
    return f'{text}<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/>'


def _render_decoded_events(events: Sequence[DecodedEvent | dict], duration_ms: float, *, width: int, height: int) -> str:
    items: list[str] = []
    for event in events:
        if isinstance(event, DecodedEvent):
            label = event.label
            time_ms = event.selected_time_ms
            score = event.score
        else:
            label = str(event.get("label"))
            time_ms = float(event.get("selected_time_ms", event.get("time_ms", 0.0)))
            score = float(event.get("score", 0.0))
        x = time_ms / duration_ms * width
        color = EVENT_COLORS.get(label, "#111827")
        items.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" stroke="{color}" stroke-width="2" stroke-dasharray="4 3"/>')
        items.append(f'<text x="{x + 4:.2f}" y="16" font-size="12" fill="{color}">{html.escape(label)} {score:.2f}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(items)}{_time_ticks(duration_ms, width, height)}</svg>'


def _render_generated_oto_rows(rows: Sequence[dict], duration_ms: float, *, width: int, height: int) -> str:
    if not rows:
        return f'<svg viewBox="0 0 {width} {height}" role="img">{_time_ticks(duration_ms, width, height)}</svg>'
    colors = {
        "offset_abs": "#64748b",
        "overlap_abs": "#0ea5e9",
        "preutterance_abs": "#dc2626",
        "consonant_abs": "#f97316",
        "cutoff_abs": "#111827",
    }
    labels = {
        "offset_abs": "offset",
        "overlap_abs": "overlap",
        "preutterance_abs": "preutterance",
        "consonant_abs": "consonant",
        "cutoff_abs": "cutoff",
    }
    items: list[str] = []
    row_h = max(20.0, height / max(1, len(rows)))
    for row_idx, row in enumerate(rows):
        absolute = row.get("absolute") if isinstance(row, dict) else None
        if not isinstance(absolute, dict):
            absolute = row
        y0 = row_idx * row_h
        alias = html.escape(str(row.get("alias", f"row{row_idx}") if isinstance(row, dict) else f"row{row_idx}"))
        items.append(f'<text x="4" y="{y0 + 13:.2f}" font-size="11" fill="#374151">{alias}</text>')
        for key, color in colors.items():
            if key not in absolute:
                continue
            time_ms = float(absolute.get(key) or 0.0)
            x = time_ms / duration_ms * width
            items.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{min(height, y0 + row_h):.2f}" stroke="{color}" stroke-width="2"/>')
            items.append(f'<text x="{x + 3:.2f}" y="{y0 + row_h - 4:.2f}" font-size="10" fill="{color}">{labels[key]}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(items)}{_time_ticks(duration_ms, width, height)}</svg>'


def _time_ticks(duration_ms: float, width: int, height: int) -> str:
    ticks: list[str] = []
    step = 100.0 if duration_ms <= 1200.0 else 500.0
    tick = 0.0
    while tick <= duration_ms + 1e-5:
        x = tick / duration_ms * width
        ticks.append(f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" stroke="#e5e7eb" stroke-width="1"/>')
        ticks.append(f'<text x="{x + 2:.2f}" y="{height - 4:.2f}" font-size="10" fill="#6b7280">{tick / 1000.0:.2f}s</text>')
        tick += step
    return "".join(ticks)


def _legend_html() -> str:
    items = []
    for label, color in {**FRAME_COLORS, **EVENT_COLORS}.items():
        items.append(f'<span><span class="swatch" style="background:{color}"></span>{html.escape(label)}</span>')
    return "".join(items)


def _compact_row(row: dict) -> dict:
    keep = [
        "row_id",
        "wav_name",
        "wav_path",
        "duration_ms",
        "language",
        "format_type",
        "aliases",
        "expected_phones",
        "frame_labels",
        "events",
        "label_source",
    ]
    return {key: row.get(key) for key in keep if key in row}
