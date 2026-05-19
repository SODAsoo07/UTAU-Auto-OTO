from __future__ import annotations

import argparse
import html
import json
import os
import wave
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_types import BOUNDARY_LABELS, BoundaryCandidate, BoundaryFrameScores, DecodedOtoRow
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.boundary_targets import load_row_specs_from_source_oto
from core.coarse_crnn.wav_decoder import decode_wav_rows


LABEL_COLORS = {
    "syllable_onset": "#1f77b4",
    "consonant_onset": "#d62728",
    "vowel_start": "#2ca02c",
    "vowel_stable": "#98df8a",
    "vowel_end": "#ff7f0e",
    "transition_peak": "#9467bd",
    "next_onset": "#8c564b",
    "silence_boundary": "#7f7f7f",
}

ROLE_COLORS = {
    "-cv": "#1f77b4",
    "cv": "#17becf",
    "v": "#2ca02c",
    "vc": "#bcbd22",
    "vv": "#7f7f7f",
    "v-cv": "#ff7f0e",
    "v-": "#9467bd",
    "cv-": "#8c564b",
    "br": "#444444",
    "endbr": "#444444",
    "other": "#607d8b",
    "special": "#e377c2",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render Boundary Scorer posterior and decoded OTO anchors for one wav as dependency-free SVG."
    )
    ap.add_argument("--model", required=True, help="Boundary scorer checkpoint.")
    ap.add_argument("--out", required=True, help="Output SVG path.")
    ap.add_argument("--bank", default="", help="Voicebank/staged folder containing wav files and oto.ini.")
    ap.add_argument("--source-oto", default="", help="Explicit source oto.ini. Defaults to --bank/oto.ini.")
    ap.add_argument("--wav-dir", default="", help="Explicit wav folder. Defaults to --bank.")
    ap.add_argument("--wav", default="", help="Wav basename or absolute path to inspect. Defaults to the first decodable wav.")
    ap.add_argument("--language", default="", help="korean or japanese. Inferred from path when omitted.")
    ap.add_argument("--format-type", default="", help="cv/cvvc/vcv/cvc/other. Inferred from path when omitted.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=90)
    args = ap.parse_args()

    source_oto, wav_dir = _resolve_inputs(args.bank, args.source_oto, args.wav_dir)
    language = str(args.language or "").strip() or _infer_language(source_oto)
    format_type = str(args.format_type or "").strip() or _infer_format(source_oto)
    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=source_oto,
        wav_dir=wav_dir,
        language=language,
        format_type=format_type,
    )
    if not rows_by_wav:
        raise SystemExit(f"no decodable OTO rows found: {source_oto}")
    wav_path, row_specs = _select_wav(rows_by_wav, args.wav)
    if int(args.max_rows) > 0 and len(row_specs) > int(args.max_rows):
        row_specs = row_specs[: int(args.max_rows)]

    torch = __import__("torch")
    device = _resolve_device(torch, args.device)
    model, config, meta = load_boundary_checkpoint(args.model, map_location=str(device))
    score_map = infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(device))
    audio = compute_audio_candidates(wav_path)
    model_candidates = peak_candidates_from_scores(score_map)
    audio_candidates = audio_candidates_to_boundary_candidates(audio)
    merged = merge_candidates(
        model_candidates=model_candidates,
        audio_candidates=audio_candidates,
        model_quality=_mean(score_map.quality_scores),
        audio_reliability=None,
    )
    decoded = decode_wav_rows(
        wav_path=wav_path,
        duration_ms=float(audio.duration_ms),
        row_specs=row_specs,
        candidates=merged,
        active_start_ms=float(audio.active_start_ms),
        active_end_ms=float(audio.active_end_ms),
        model_quality=_mean(score_map.quality_scores),
        audio_reliability=None,
        posterior_scores=score_map.scores,
        posterior_times_ms=score_map.times_ms,
        cvs_scores=score_map.cvs_scores,
        consonant_scores=score_map.consonant_scores,
        vowel_scores=score_map.vowel_scores,
        consonant_family_scores=score_map.consonant_family_scores,
        vowel_nucleus_scores=score_map.vowel_nucleus_scores,
        vowel_glide_scores=score_map.vowel_glide_scores,
    )

    svg = render_svg(
        wav_path=wav_path,
        model_path=os.path.abspath(args.model),
        source_oto=source_oto,
        score_map=score_map,
        model_candidates=model_candidates,
        merged_candidates=merged,
        decoded_rows=decoded.rows,
        anchor_timeline=decoded.anchor_timeline_ms,
        meta=meta,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path.resolve()),
                "wav": wav_path,
                "rows": len(decoded.rows),
                "model_candidates": len(model_candidates),
                "merged_candidates": len(merged),
                "fallback_rows": int(decoded.fallback_count),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def render_svg(
    *,
    wav_path: str,
    model_path: str,
    source_oto: str,
    score_map: BoundaryFrameScores,
    model_candidates: list[BoundaryCandidate],
    merged_candidates: list[BoundaryCandidate],
    decoded_rows: list[DecodedOtoRow],
    anchor_timeline: list[float],
    meta: dict[str, Any],
) -> str:
    samples, sr = _load_wav_for_plot(wav_path)
    duration_ms = max(_duration_ms(samples, sr), max(score_map.times_ms or [0.0]), 1.0)
    width = 1500
    left = 125
    right = 30
    top = 58
    panel_gap = 18
    wave_h = 160
    curve_h = 78
    row_h = 18
    rows_h = max(90, min(430, 24 + row_h * max(1, len(decoded_rows))))
    labels = [label for label in BOUNDARY_LABELS if score_map.scores.get(label)]
    height = top + wave_h + panel_gap + (curve_h + panel_gap) * len(labels) + rows_h + 60
    plot_w = width - left - right

    def x(ms: float) -> float:
        return left + (max(0.0, min(duration_ms, float(ms))) / duration_ms) * plot_w

    parts: list[str] = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:Segoe UI,Malgun Gothic,Yu Gothic,Arial,sans-serif} .small{font-size:10px;fill:#333} .tiny{font-size:8px;fill:#333}</style>",
        "<rect width='100%' height='100%' fill='#fbfbfd'/>",
        f"<text x='18' y='24' font-size='15' font-weight='600'>Boundary decode visualization</text>",
        f"<text x='18' y='42' class='small'>wav={_e(os.path.basename(wav_path))} | model={_e(os.path.basename(model_path))} | source={_e(os.path.basename(source_oto))}</text>",
    ]
    y0 = top
    parts.append(_panel_rect(left, y0, plot_w, wave_h))
    parts.append(f"<text x='18' y='{y0 + 18}' class='small'>waveform</text>")
    parts.append(_waveform_polyline(samples, sr, x, y0, wave_h, duration_ms))
    for t in anchor_timeline:
        parts.append(_vline(x(t), y0, y0 + wave_h, "#555", 0.20, dash="3,4"))
    for row in decoded_rows:
        color = ROLE_COLORS.get(str(row.spec.role).lower(), "#607d8b")
        parts.append(_vline(x(row.selected_time_ms), y0, y0 + wave_h, color, 0.48))

    candidate_by_kind = _group_candidates(model_candidates)
    merged_by_kind = _group_candidates(merged_candidates)
    y = y0 + wave_h + panel_gap
    for label in labels:
        color = LABEL_COLORS.get(label, "#444444")
        parts.append(_panel_rect(left, y, plot_w, curve_h))
        parts.append(f"<text x='18' y='{y + 18}' class='small'>{_e(label)}</text>")
        parts.append(_score_polyline(score_map.times_ms, score_map.scores[label], x, y, curve_h, color))
        for cand in candidate_by_kind.get(label, []):
            cx = x(cand.time_ms)
            cy = y + curve_h - (max(0.0, min(1.0, cand.score)) * (curve_h - 16)) - 8
            parts.append(f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='2.2' fill='{color}' fill-opacity='0.78'/>")
        for cand in merged_by_kind.get(label, []):
            parts.append(_vline(x(cand.time_ms), y + 5, y + curve_h - 5, "#111", 0.18, dash="2,3"))
        y += curve_h + panel_gap

    parts.append(_panel_rect(left, y, plot_w, rows_h))
    parts.append(f"<text x='18' y='{y + 18}' class='small'>decoded rows</text>")
    parts.append(
        f"<text x='{left}' y='{y - 5}' class='tiny'>row span=offset..cutoff, selected=center line, O=overlap, P=pre, C=consonant</text>"
    )
    row_top = y + 24
    for idx, row in enumerate(decoded_rows):
        yy = row_top + idx * row_h
        if yy > y + rows_h - 12:
            break
        anchors = row.anchors
        color = ROLE_COLORS.get(str(row.spec.role).lower(), "#607d8b")
        x_off = x(anchors.offset_abs)
        x_cut = x(anchors.cutoff_abs)
        x_ovl = x(anchors.overlap_abs)
        x_pre = x(anchors.pre_abs)
        x_cons = x(anchors.consonant_abs)
        x_sel = x(row.selected_time_ms)
        parts.append(f"<rect x='{x_off:.2f}' y='{yy - 5:.2f}' width='{max(1.0, x_cut - x_off):.2f}' height='10' fill='{color}' fill-opacity='0.42'/>")
        parts.append(_vline(x_ovl, yy - 7, yy + 7, "#4b5563", 0.85))
        parts.append(_vline(x_pre, yy - 7, yy + 7, "#16a34a", 0.95))
        parts.append(_vline(x_cons, yy - 7, yy + 7, "#f97316", 0.95))
        parts.append(_vline(x_sel, yy - 8, yy + 8, "#000", 0.85, dash="3,2" if row.fallback_used else ""))
        label = f"{idx:02d} {row.spec.alias} [{row.spec.role}] score={row.quality_score:.2f}"
        parts.append(f"<text x='18' y='{yy + 3:.2f}' class='tiny'>{_e(label)}</text>")

    footer = {
        "model_meta": meta,
        "model_candidates": len(model_candidates),
        "merged_candidates": len(merged_candidates),
    }
    parts.append(f"<text x='18' y='{height - 20}' class='tiny'>{_e(json.dumps(footer, ensure_ascii=False, default=str)[:900])}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def _resolve_inputs(bank: str, source_oto: str, wav_dir: str) -> tuple[str, str]:
    bank_path = Path(bank).resolve() if str(bank or "").strip() else None
    source = Path(source_oto).resolve() if str(source_oto or "").strip() else None
    wav_root = Path(wav_dir).resolve() if str(wav_dir or "").strip() else None
    if source is None and bank_path is not None:
        source = bank_path / "oto.ini"
    if wav_root is None and bank_path is not None:
        wav_root = bank_path
    if source is None or not source.is_file():
        raise SystemExit("provide --bank with oto.ini or explicit --source-oto")
    if wav_root is None or not wav_root.is_dir():
        raise SystemExit("provide --bank or explicit --wav-dir")
    return str(source), str(wav_root)


def _select_wav(rows_by_wav: dict[str, list[Any]], wav: str) -> tuple[str, list[Any]]:
    if str(wav or "").strip():
        needle = str(wav).strip()
        needle_abs = os.path.abspath(needle) if os.path.isabs(needle) else ""
        needle_base = os.path.basename(needle).lower()
        for path, rows in sorted(rows_by_wav.items()):
            if (needle_abs and os.path.abspath(path) == needle_abs) or os.path.basename(path).lower() == needle_base:
                return path, rows
        raise SystemExit(f"wav not found in source OTO rows: {wav}")
    for path, rows in sorted(rows_by_wav.items()):
        if rows:
            return path, rows
    raise SystemExit("no wav rows found")


def _load_wav_for_plot(path: str) -> tuple[np.ndarray, int]:
    with closing(wave.open(path, "rb")) as wf:
        frames = int(wf.getnframes())
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        width = int(wf.getsampwidth())
        raw = wf.readframes(frames)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        data = np.zeros((1,), dtype=np.float32)
    if channels > 1 and data.size >= channels:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _duration_ms(samples: np.ndarray, sr: int) -> float:
    if int(sr) <= 0:
        return 1.0
    return float(len(samples)) * 1000.0 / float(sr)


def _waveform_polyline(samples: np.ndarray, sr: int, x_fn, y: float, h: float, duration_ms: float) -> str:
    if samples.size <= 0:
        return ""
    return _waveform_envelope(samples, sr, x_fn, y, h, duration_ms)


def _waveform_envelope(samples: np.ndarray, sr: int, x_fn, y: float, h: float, duration_ms: float) -> str:
    max_columns = 1800
    step = max(1, int(samples.size // max_columns))
    mid = y + h * 0.5
    amp = h * 0.42
    lines: list[str] = []
    for start in range(0, samples.size, step):
        chunk = samples[start : start + step]
        if chunk.size <= 0:
            continue
        t = float(start + (chunk.size * 0.5)) * 1000.0 / float(max(1, sr))
        if t > duration_ms:
            break
        lo = float(np.clip(np.min(chunk), -1.0, 1.0))
        hi = float(np.clip(np.max(chunk), -1.0, 1.0))
        xx = x_fn(t)
        y_hi = mid - hi * amp
        y_lo = mid - lo * amp
        lines.append(
            f"<line x1='{xx:.2f}' x2='{xx:.2f}' y1='{y_hi:.2f}' y2='{y_lo:.2f}' "
            "stroke='#222' stroke-width='0.75' stroke-opacity='0.86'/>"
        )
    return "\n".join(lines)


def _score_polyline(times: list[float], values: list[float], x_fn, y: float, h: float, color: str) -> str:
    if not times or not values:
        return ""
    n = min(len(times), len(values))
    step = max(1, n // 1800)
    pts = []
    for idx in range(0, n, step):
        v = max(0.0, min(1.0, float(values[idx])))
        pts.append(f"{x_fn(float(times[idx])):.2f},{(y + h - 8.0 - v * (h - 16.0)):.2f}")
    return f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='1.1'/>"


def _panel_rect(x: float, y: float, w: float, h: float) -> str:
    return f"<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='#ffffff' stroke='#d8dde8' stroke-width='1'/>"


def _vline(x: float, y1: float, y2: float, color: str, opacity: float, *, dash: str = "") -> str:
    dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
    return f"<line x1='{x:.2f}' x2='{x:.2f}' y1='{y1:.2f}' y2='{y2:.2f}' stroke='{color}' stroke-width='1'{dash_attr} stroke-opacity='{opacity:.3f}'/>"


def _group_candidates(candidates: list[BoundaryCandidate]) -> dict[str, list[BoundaryCandidate]]:
    out: dict[str, list[BoundaryCandidate]] = {}
    for item in candidates:
        out.setdefault(str(item.kind), []).append(item)
    return out


def _mean(values: list[float] | None) -> float:
    vals = [float(v) for v in list(values or []) if v is not None]
    return sum(vals) / max(1, len(vals)) if vals else 0.5


def _infer_language(path: str) -> str:
    text = str(path or "").replace("\\", "/").lower()
    if "/korean/" in text or "/ko/" in text:
        return "korean"
    if "/japanese/" in text or "/ja/" in text or "/jp/" in text:
        return "japanese"
    return "korean"


def _infer_format(path: str) -> str:
    text = str(path or "").replace("\\", "/").lower()
    for token in ("cvvc", "vcv", "cvc", "cv"):
        if f"/{token}/" in text or f"\\{token}\\" in text:
            return token
    return "other"


def _resolve_device(torch, device: str) -> str:
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return text


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
