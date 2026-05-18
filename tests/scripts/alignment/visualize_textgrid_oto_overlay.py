from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import wave
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Sequence

import numpy as np

try:
    import textgrid  # type: ignore
except Exception:  # pragma: no cover
    textgrid = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.oto_file_utils import parse_oto_line, read_text_with_fallback


_PAUSE_MARKS = {"", "sp", "spn", "sil", "pau", "ap", "br", "bre", "breath", "r"}


@dataclass
class PhoneInterval:
    start_ms: float
    end_ms: float
    mark: str


@dataclass
class OtoAnchorRow:
    alias: str
    offset_ms: float
    pre_ms: float
    cons_ms: float
    cutoff_ms: float
    ovl_ms: float


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _norm_wav_key(name: str) -> str:
    base = os.path.splitext(os.path.basename(str(name or "")))[0]
    return str(base).strip().lower()


def _list_top_level_wavs(path: str) -> List[str]:
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if str(name).lower().endswith(".wav"):
            out.append(os.path.join(root, name))
    return out


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_ch = int(wf.getnchannels() or 1)
        sw = int(wf.getsampwidth() or 2)
        sr = int(wf.getframerate() or 44100)
        n = int(wf.getnframes() or 0)
        raw = wf.readframes(n)

    if n <= 0:
        return np.zeros((0,), dtype=np.float32), sr

    if sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_ch > 1:
        frame_count = data.size // n_ch
        data = data[: frame_count * n_ch].reshape(frame_count, n_ch).mean(axis=1)
    return np.clip(data, -1.0, 1.0), sr


def _load_phone_intervals(textgrid_path: str, tier_name: str) -> List[PhoneInterval]:
    if textgrid is None or not os.path.isfile(textgrid_path):
        return []
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except Exception:
        return []

    target = None
    want = str(tier_name or "phones").strip().lower()
    name_candidates = [want, "phones", "phone", "phonemes", "phoneme"]
    for name in name_candidates:
        for tier in list(getattr(tg, "tiers", []) or []):
            tier_key = str(getattr(tier, "name", "") or "").strip().lower()
            if tier_key == name:
                target = tier
                break
        if target is not None:
            break
    if target is None:
        for tier in list(getattr(tg, "tiers", []) or []):
            if hasattr(tier, "intervals"):
                target = tier
                break
    if target is None:
        return []

    out: List[PhoneInterval] = []
    for it in list(getattr(target, "intervals", []) or []):
        mark = str(getattr(it, "mark", "") or "").strip()
        st = float(getattr(it, "minTime", 0.0) or 0.0) * 1000.0
        ed = float(getattr(it, "maxTime", 0.0) or 0.0) * 1000.0
        if ed <= st:
            continue
        out.append(PhoneInterval(start_ms=float(st), end_ms=float(ed), mark=mark))
    return out


def _load_oto_rows_by_wav(oto_path: str) -> Dict[str, List[OtoAnchorRow]]:
    rows_by_wav: Dict[str, List[OtoAnchorRow]] = {}
    if not os.path.isfile(oto_path):
        return rows_by_wav
    text = read_text_with_fallback(oto_path)
    for raw in text.splitlines():
        parsed = parse_oto_line(raw)
        if not parsed:
            continue
        wav_name = str(parsed.get("wav", "") or "")
        key = _norm_wav_key(wav_name)
        if not key:
            continue
        offset = float(parsed.get("offset", 0.0) or 0.0)
        pre_rel = max(0.0, float(parsed.get("pre", 0.0) or 0.0))
        cons_rel = max(pre_rel, float(parsed.get("cons", pre_rel) or pre_rel))
        cutoff_raw = float(parsed.get("cutoff", 0.0) or 0.0)
        cutoff_rel = abs(cutoff_raw) if cutoff_raw < 0.0 else max(0.0, cutoff_raw)
        ovl_rel = max(0.0, float(parsed.get("ovl", 0.0) or 0.0))

        row = OtoAnchorRow(
            alias=str(parsed.get("alias", "") or ""),
            offset_ms=float(offset),
            pre_ms=float(offset + pre_rel),
            cons_ms=float(offset + cons_rel),
            cutoff_ms=float(offset + cutoff_rel),
            ovl_ms=float(offset + min(ovl_rel, pre_rel)),
        )
        rows_by_wav.setdefault(key, []).append(row)
    return rows_by_wav


def _phone_boundary_points_ms(intervals: Sequence[PhoneInterval]) -> List[float]:
    vals: List[float] = []
    for it in intervals:
        vals.append(float(it.start_ms))
        vals.append(float(it.end_ms))
    vals = sorted(vals)
    uniq: List[float] = []
    for v in vals:
        if (not uniq) or abs(v - uniq[-1]) > 1e-4:
            uniq.append(float(v))
    return uniq


def _nearest_delta_ms(value_ms: float, points_ms: Sequence[float]) -> float:
    if not points_ms:
        return 999999.0
    return min(abs(float(value_ms) - float(p)) for p in points_ms)


def _p90(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    idx = max(0, min(len(vals) - 1, int(round((len(vals) - 1) * 0.9))))
    return float(vals[idx])


def _analyze_alignment(rows: Sequence[OtoAnchorRow], intervals: Sequence[PhoneInterval]) -> Dict[str, float]:
    points = _phone_boundary_points_ms(intervals)
    if not rows or not points:
        return {}
    pre_d = [_nearest_delta_ms(r.pre_ms, points) for r in rows]
    cons_d = [_nearest_delta_ms(r.cons_ms, points) for r in rows]
    return {
        "rows": float(len(rows)),
        "phones": float(len(intervals)),
        "pre_nearest_mean_ms": float(statistics.mean(pre_d)),
        "pre_nearest_p90_ms": float(_p90(pre_d)),
        "cons_nearest_mean_ms": float(statistics.mean(cons_d)),
        "cons_nearest_p90_ms": float(_p90(cons_d)),
    }


def _build_svg(
    wav_path: str,
    intervals: Sequence[PhoneInterval],
    rows: Sequence[OtoAnchorRow],
    *,
    max_rows: int,
) -> str:
    audio, sr = _read_wav_mono(wav_path)
    if audio.size <= 0:
        return ""

    duration_ms = (float(audio.size) / float(max(1, sr))) * 1000.0
    if duration_ms <= 1e-6:
        return ""

    width = 1800
    height = 840
    left = 72
    right = 26
    top = 30
    bottom = 38
    wave_top = top + 18
    wave_h = 420
    lane_top = wave_top + wave_h + 26
    lane_h = height - lane_top - bottom
    plot_w = float(width - left - right)

    def x_map(ms: float) -> float:
        clamped = min(max(0.0, float(ms)), duration_ms)
        return left + (clamped / duration_ms) * plot_w

    def y_wave(v: float) -> float:
        vv = min(max(-1.0, float(v)), 1.0)
        return wave_top + ((1.0 - ((vv + 1.0) * 0.5)) * wave_h)

    step = max(1, int(round(audio.size / 15000)))
    y_plot = np.clip(audio[::step], -1.0, 1.0)
    x_plot = np.arange(0, y_plot.size, dtype=np.float64) * float(step) / float(max(1, sr)) * 1000.0
    wave_points = " ".join(f"{x_map(float(x)):.2f},{y_wave(float(y)):.2f}" for x, y in zip(x_plot, y_plot))

    sampled_rows: List[OtoAnchorRow] = list(rows)
    sampled = False
    if len(sampled_rows) > int(max_rows):
        sampled = True
        idxs = np.linspace(0, len(sampled_rows) - 1, int(max_rows), dtype=int).tolist()
        sampled_rows = [sampled_rows[i] for i in idxs]

    phone_lines: List[str] = []
    phone_labels: List[str] = []
    for idx, it in enumerate(intervals):
        xs = x_map(it.start_ms)
        xe = x_map(it.end_ms)
        phone_lines.append(
            f"<line x1='{xs:.2f}' y1='{wave_top}' x2='{xs:.2f}' y2='{wave_top + wave_h}' "
            "stroke='#3778D8' stroke-width='1' stroke-dasharray='4,4' opacity='0.68' />"
        )
        if idx < len(intervals) - 1:
            mid = (float(xs) + float(xe)) * 0.5
            mark = escape(str(it.mark or ""))
            if mark and mark.lower() not in _PAUSE_MARKS:
                phone_labels.append(
                    f"<text x='{mid:.2f}' y='{wave_top + 16}' text-anchor='middle' font-size='10' "
                    "font-family='Segoe UI, sans-serif' fill='#2B4F8C'>{}</text>".format(mark)
                )

    anchor_styles = [
        ("offset", "#888888", "2,2"),
        ("ovl", "#00A9A5", "1,0"),
        ("pre", "#FF8C00", "1,0"),
        ("cons", "#2E9F59", "1,0"),
        ("cutoff", "#7C4DFF", "5,3"),
    ]

    row_lines: List[str] = []
    row_markers: List[str] = []
    row_count = max(1, len(sampled_rows))
    lane_step = lane_h / float(row_count + 1)
    for idx, row in enumerate(sampled_rows, start=1):
        y = lane_top + (idx * lane_step)
        left_seg = x_map(min(row.offset_ms, row.cutoff_ms))
        right_seg = x_map(max(row.offset_ms, row.cutoff_ms))
        row_lines.append(
            f"<line x1='{left_seg:.2f}' y1='{y:.2f}' x2='{right_seg:.2f}' y2='{y:.2f}' "
            "stroke='#C9CDD5' stroke-width='1' opacity='0.75' />"
        )
        values = {
            "offset": row.offset_ms,
            "ovl": row.ovl_ms,
            "pre": row.pre_ms,
            "cons": row.cons_ms,
            "cutoff": row.cutoff_ms,
        }
        for key, color, dash in anchor_styles:
            x = x_map(values[key])
            row_markers.append(
                f"<line x1='{x:.2f}' y1='{y - 4:.2f}' x2='{x:.2f}' y2='{y + 4:.2f}' "
                f"stroke='{color}' stroke-width='1.4' stroke-dasharray='{dash}' opacity='0.95' />"
            )

    legend_x = left + 12
    legend_items: List[str] = []
    for idx, (name, color, dash) in enumerate(anchor_styles):
        ly = lane_top - 10 + (idx * 16)
        legend_items.append(
            f"<line x1='{legend_x}' y1='{ly}' x2='{legend_x + 18}' y2='{ly}' stroke='{color}' "
            f"stroke-width='1.6' stroke-dasharray='{dash}' />"
        )
        legend_items.append(
            f"<text x='{legend_x + 24}' y='{ly + 4}' font-size='11' font-family='Segoe UI, sans-serif' fill='#333'>{name}</text>"
        )

    sample_note = ""
    if sampled:
        sample_note = f" (sampled {len(sampled_rows)}/{len(rows)} rows)"

    title = escape(os.path.basename(wav_path))
    svg: List[str] = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    svg.append("<rect width='100%' height='100%' fill='#FFFFFF'/>")
    svg.append(
        f"<rect x='{left}' y='{wave_top}' width='{plot_w:.2f}' height='{wave_h}' fill='#FBFCFE' stroke='#D7DDE7' stroke-width='1'/>"
    )
    svg.append(
        f"<rect x='{left}' y='{lane_top}' width='{plot_w:.2f}' height='{lane_h:.2f}' fill='#FDFDFD' stroke='#E3E6EB' stroke-width='1'/>"
    )
    svg.append(f"<polyline points='{wave_points}' fill='none' stroke='#5A5A5A' stroke-width='1' />")
    svg.extend(phone_lines)
    svg.extend(phone_labels)
    svg.extend(row_lines)
    svg.extend(row_markers)
    svg.extend(legend_items)
    svg.append(
        f"<text x='{left}' y='{top + 4}' font-size='15' font-family='Segoe UI, sans-serif' fill='#1F2D3D'>{title}</text>"
    )
    svg.append(
        f"<text x='{width - right}' y='{top + 4}' text-anchor='end' font-size='12' font-family='Segoe UI, sans-serif' fill='#555'>"
        f"phones={len(intervals)} | oto_rows={len(rows)}{escape(sample_note)}</text>"
    )
    svg.append(
        f"<text x='{left}' y='{height - 12}' font-size='11' font-family='Segoe UI, sans-serif' fill='#666'>0 ms</text>"
    )
    svg.append(
        f"<text x='{width - right}' y='{height - 12}' text-anchor='end' font-size='11' font-family='Segoe UI, sans-serif' fill='#666'>"
        f"{duration_ms:.1f} ms</text>"
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _write_html(path: str, payload: Dict[str, object]) -> None:
    items = list(payload.get("items", []) or [])
    stats = dict(payload.get("stats", {}) or {})
    sections: List[str] = []
    for item in items:
        svg_rel = str(item.get("svg_rel", "") or "")
        if not svg_rel:
            continue
        metrics = item.get("metrics", {}) or {}
        sections.append(
            "<section class='card'>"
            f"<h3>{escape(str(item.get('base', '')))}</h3>"
            f"<p class='meta'>phones={int(item.get('phones', 0) or 0)} | oto_rows={int(item.get('oto_rows', 0) or 0)}"
            f" | pre_mean={float(metrics.get('pre_nearest_mean_ms', 0.0) or 0.0):.2f}ms"
            f" | pre_p90={float(metrics.get('pre_nearest_p90_ms', 0.0) or 0.0):.2f}ms</p>"
            f"<img src='{escape(svg_rel)}' alt='{escape(str(item.get('base', '')))} overlay' />"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TextGrid + OTO Overlay</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 16px; color: #222; background: #fafafa; }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    .meta {{ color: #555; font-size: 13px; margin: 6px 0 12px 0; }}
    .summary {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 6px; padding: 12px; margin-bottom: 14px; }}
    .card {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
    .card h3 {{ margin: 0 0 6px 0; font-size: 16px; }}
    .card img {{ width: 100%; border: 1px solid #e7eaf0; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>TextGrid Phones + OTO Anchors</h1>
  <div class="summary">
    <div class="meta">voicebank={escape(str(payload.get("voicebank", "")))}</div>
    <div class="meta">textgrid_dir={escape(str(payload.get("textgrid_dir", "")))}</div>
    <div class="meta">source_oto={escape(str(payload.get("source_oto", "")))}</div>
    <div class="meta">files={int(stats.get("files", 0) or 0)} | compared={int(stats.get("compared", 0) or 0)}
      | pre_mean={float(stats.get("pre_nearest_mean_ms", 0.0) or 0.0):.2f}ms
      | pre_p90={float(stats.get("pre_nearest_p90_ms", 0.0) or 0.0):.2f}ms</div>
  </div>
  {"".join(sections)}
</body>
</html>
"""
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize TextGrid phone boundaries and source OTO parameter anchors together."
    )
    parser.add_argument("--voicebank", required=True, help="Voicebank wav root")
    parser.add_argument("--textgrid-dir", default="", help="TextGrid folder (default: <voicebank>/textgrids)")
    parser.add_argument("--source-oto", default="", help="Source oto.ini (default: <voicebank>/oto.ini)")
    parser.add_argument("--tier", default="phones", help="Phone tier name")
    parser.add_argument("--max-files", type=int, default=40, help="Max wav files to render")
    parser.add_argument("--max-rows-per-wav", type=int, default=120, help="Max OTO rows rendered per wav")
    parser.add_argument("--out-root", default="", help="Output root folder")
    args = parser.parse_args()

    voicebank = os.path.abspath(str(args.voicebank or "").strip())
    if not os.path.isdir(voicebank):
        raise SystemExit(f"voicebank not found: {voicebank}")
    if textgrid is None:
        raise SystemExit("python-textgrid is required but not available")

    textgrid_dir = (
        os.path.abspath(str(args.textgrid_dir))
        if str(args.textgrid_dir or "").strip()
        else os.path.join(voicebank, "textgrids")
    )
    source_oto = (
        os.path.abspath(str(args.source_oto))
        if str(args.source_oto or "").strip()
        else os.path.join(voicebank, "oto.ini")
    )
    if not os.path.isdir(textgrid_dir):
        raise SystemExit(f"textgrid-dir not found: {textgrid_dir}")
    if not os.path.isfile(source_oto):
        raise SystemExit(f"source-oto not found: {source_oto}")

    out_root = (
        os.path.abspath(str(args.out_root))
        if str(args.out_root or "").strip()
        else os.path.join(os.getcwd(), "temp", f"tg_oto_overlay_{os.path.basename(voicebank)}")
    )
    plot_dir = _safe_mkdir(os.path.join(out_root, "plots"))

    wav_files = _list_top_level_wavs(voicebank)[: max(1, int(args.max_files))]
    rows_by_wav = _load_oto_rows_by_wav(source_oto)

    items: List[Dict[str, object]] = []
    pre_all: List[float] = []
    cons_all: List[float] = []
    compared = 0
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        key = _norm_wav_key(base)
        tg_path = os.path.join(textgrid_dir, base + ".TextGrid")
        intervals = _load_phone_intervals(tg_path, str(args.tier or "phones"))
        oto_rows = rows_by_wav.get(key, [])

        svg = _build_svg(
            wav_path,
            intervals,
            oto_rows,
            max_rows=max(10, int(args.max_rows_per_wav)),
        )
        svg_rel = ""
        if svg:
            svg_name = f"{base}.svg"
            svg_path = os.path.join(plot_dir, svg_name)
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write(svg)
            svg_rel = os.path.relpath(svg_path, out_root)

        metrics = _analyze_alignment(oto_rows, intervals)
        if metrics:
            compared += 1
            pre_all.append(float(metrics.get("pre_nearest_mean_ms", 0.0) or 0.0))
            cons_all.append(float(metrics.get("cons_nearest_mean_ms", 0.0) or 0.0))

        items.append(
            {
                "base": base,
                "wav_path": wav_path,
                "textgrid_path": tg_path,
                "phones": int(len(intervals)),
                "oto_rows": int(len(oto_rows)),
                "svg_rel": svg_rel,
                "metrics": metrics,
            }
        )

    stats = {
        "files": int(len(wav_files)),
        "compared": int(compared),
        "pre_nearest_mean_ms": float(statistics.mean(pre_all)) if pre_all else 0.0,
        "pre_nearest_p90_ms": float(_p90(pre_all)) if pre_all else 0.0,
        "cons_nearest_mean_ms": float(statistics.mean(cons_all)) if cons_all else 0.0,
        "cons_nearest_p90_ms": float(_p90(cons_all)) if cons_all else 0.0,
    }
    payload: Dict[str, object] = {
        "voicebank": voicebank,
        "textgrid_dir": textgrid_dir,
        "source_oto": source_oto,
        "tier": str(args.tier or "phones"),
        "stats": stats,
        "items": items,
    }

    _safe_mkdir(out_root)
    summary_json = os.path.join(out_root, "summary.json")
    summary_html = os.path.join(out_root, "summary.html")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_html(summary_html, payload)

    print(f"[DONE] out_root={out_root}")
    print(f"[DONE] summary_json={summary_json}")
    print(f"[DONE] summary_html={summary_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
