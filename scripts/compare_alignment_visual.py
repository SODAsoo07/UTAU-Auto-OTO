from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
import wave
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import textgrid  # type: ignore
except Exception:  # pragma: no cover
    textgrid = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.alignment_pipeline import run_alignment_with_fallback
from core.sequence_aligner import run_sequence_align
from core.sinsy_label_ingest import load_sinsy_label_entries

_SKIP_MARKS = {"", "spn", "sil", "pau", "sp", "bre", "breath", "r"}
_PLOT_EXT = "png" if plt is not None else "svg"
_DEFAULT_SODA_D4 = (
    r"C:\Users\oyh57\SODAsoo1\VocalSynth\UTAU\Singer\SODAsoo_CVVC_Test-copy\SODA_D4"
)
_DEFAULT_TRUTH_LAB_DIR = os.path.join(
    _REPO_ROOT,
    "aligner_sample",
    "sinsy_lab",
    "sequence_words",
    "lab",
)


@dataclass
class Interval:
    start: float
    end: float
    mark: str


def _log(msg: str) -> None:
    print(msg, flush=True)


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _list_top_level_wavs(path: str) -> List[str]:
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".wav"):
            out.append(os.path.join(root, name))
    return out


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_ch = int(wf.getnchannels() or 1)
        sw = int(wf.getsampwidth() or 2)
        sr = int(wf.getframerate() or 44100)
        n = int(wf.getnframes() or 0)
        raw = wf.readframes(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), sr

    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_ch > 1:
        frame_count = data.size // n_ch
        data = data[: frame_count * n_ch].reshape(frame_count, n_ch).mean(axis=1)

    return np.clip(data, -1.0, 1.0), sr


def _read_active_intervals(textgrid_path: str, tier_name: str = "words") -> List[Interval]:
    if textgrid is None or (not os.path.isfile(textgrid_path)):
        return []
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except Exception:
        return []

    target = None
    for tier in tg.tiers:
        if str(getattr(tier, "name", "")).strip().lower() == str(tier_name).strip().lower():
            target = tier
            break
    if target is None:
        return []

    out: List[Interval] = []
    for it in list(target.intervals):
        mark = str(getattr(it, "mark", "") or "").strip()
        if mark.lower() in _SKIP_MARKS:
            continue
        out.append(
            Interval(
                float(getattr(it, "minTime", 0.0) or 0.0),
                float(getattr(it, "maxTime", 0.0) or 0.0),
                mark,
            )
        )
    return out


def _interval_boundaries(items: Sequence[Interval]) -> List[float]:
    if not items:
        return []
    vals: List[float] = []
    for it in items:
        vals.append(float(it.start))
        vals.append(float(it.end))
    vals = sorted(vals)
    uniq: List[float] = []
    for v in vals:
        if (not uniq) or abs(v - uniq[-1]) > 1e-4:
            uniq.append(v)
    return uniq


def _pairwise_delta_ms(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    if (not a) or (not b):
        return 999999.0
    if len(a) != len(b):
        return 999999.0
    vals: List[float] = []
    for left, right in zip(a, b):
        vals.append(abs(float(left.start) - float(right.start)) * 1000.0)
        vals.append(abs(float(left.end) - float(right.end)) * 1000.0)
    return float(statistics.mean(vals)) if vals else 999999.0


def _sofa_style_boundary_metrics(a: Sequence[Interval], b: Sequence[Interval]) -> Dict[str, float]:
    if (not a) or (not b) or len(a) != len(b):
        return {}

    err_sec: List[float] = []
    iou_vals: List[float] = []
    total_ref_duration = 0.0
    for left, right in zip(a, b):
        ls = float(left.start)
        le = float(left.end)
        rs = float(right.start)
        re = float(right.end)
        err_sec.append(abs(ls - rs))
        err_sec.append(abs(le - re))
        total_ref_duration += max(0.0, le - ls)
        inter = max(0.0, min(le, re) - max(ls, rs))
        union = max(1e-7, max(le, re) - min(ls, rs))
        iou_vals.append(float(inter / union))

    if not err_sec:
        return {}
    total_ref_duration = max(1e-7, float(total_ref_duration))
    err_sum = float(np.sum(np.asarray(err_sec, dtype=np.float64)))
    return {
        "boundary_edit_ratio": float(err_sum / total_ref_duration),
        "ber_10ms": float(np.mean([1.0 if x > 0.010 else 0.0 for x in err_sec])),
        "ber_20ms": float(np.mean([1.0 if x > 0.020 else 0.0 for x in err_sec])),
        "ber_50ms": float(np.mean([1.0 if x > 0.050 else 0.0 for x in err_sec])),
        "interval_iou": float(np.mean(iou_vals)) if iou_vals else 0.0,
    }


def _read_truth_lab_intervals(lab_path: str) -> List[Interval]:
    if not os.path.isfile(lab_path):
        return []
    try:
        rows = load_sinsy_label_entries(explicit_path=lab_path)
    except Exception:
        rows = []
    out: List[Interval] = []
    for row in rows:
        mark = str(getattr(row, "label", "") or "").strip()
        if mark.lower() in _SKIP_MARKS:
            continue
        start = float(getattr(row, "start_ms", 0.0) or 0.0) / 1000.0
        end = float(getattr(row, "end_ms", 0.0) or 0.0) / 1000.0
        if end <= start:
            continue
        out.append(Interval(start=start, end=end, mark=mark))
    return out


def _load_reference_words(*, base: str, ref_kind: str, ref_dir: str) -> List[Interval]:
    if ref_kind == "truth":
        return _read_truth_lab_intervals(os.path.join(ref_dir, base + ".lab"))
    return _read_active_intervals(os.path.join(ref_dir, base + ".TextGrid"), "words")


def _plot_overlay(
    wav_path: str,
    ref_words: Sequence[Interval],
    tgt_words: Sequence[Interval],
    *,
    out_path: str,
    title: str,
    delta_ms: float,
    ref_label: str,
    tgt_label: str,
) -> bool:
    if plt is None:
        return _plot_overlay_svg(
            wav_path,
            ref_words,
            tgt_words,
            out_path=out_path,
            title=title,
            delta_ms=delta_ms,
            ref_label=ref_label,
            tgt_label=tgt_label,
        )
    return _plot_overlay_matplotlib(
        wav_path,
        ref_words,
        tgt_words,
        out_path=out_path,
        title=title,
        delta_ms=delta_ms,
        ref_label=ref_label,
        tgt_label=tgt_label,
    )


def _plot_overlay_matplotlib(
    wav_path: str,
    ref_words: Sequence[Interval],
    tgt_words: Sequence[Interval],
    *,
    out_path: str,
    title: str,
    delta_ms: float,
    ref_label: str,
    tgt_label: str,
) -> bool:
    if plt is None:
        return False
    samples, sr = _read_wav(wav_path)
    if samples.size <= 0:
        return False

    x = np.arange(samples.size, dtype=np.float32) / float(max(sr, 1))
    step = max(1, int(round(samples.size / 14000)))
    x_plot = x[::step]
    y_plot = samples[::step]

    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=120)
    ax.plot(x_plot, y_plot, color="#555555", linewidth=0.8, alpha=0.9)
    ax.axhline(0.0, color="#AFAFAF", linewidth=0.6)

    ref_bounds = _interval_boundaries(ref_words)
    tgt_bounds = _interval_boundaries(tgt_words)

    first_ref = True
    for t in ref_bounds:
        ax.axvline(
            t,
            color="#D64A4A",
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
            label=f"{ref_label} boundary" if first_ref else None,
        )
        first_ref = False

    first_tgt = True
    for t in tgt_bounds:
        ax.axvline(
            t,
            color="#2E6FE3",
            linewidth=1.1,
            alpha=0.75,
            label=f"{tgt_label} boundary" if first_tgt else None,
        )
        first_tgt = False

    ax.set_xlim(0.0, float(x[-1]) if x.size else 1.0)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title} | mean boundary delta={delta_ms:.2f} ms")
    ax.grid(alpha=0.16, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    _safe_mkdir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _plot_overlay_svg(
    wav_path: str,
    ref_words: Sequence[Interval],
    tgt_words: Sequence[Interval],
    *,
    out_path: str,
    title: str,
    delta_ms: float,
    ref_label: str,
    tgt_label: str,
) -> bool:
    samples, sr = _read_wav(wav_path)
    if samples.size <= 0:
        return False
    duration = float(samples.size) / float(max(sr, 1))
    if duration <= 1e-6:
        return False

    width = 1500
    height = 420
    left = 62
    right = 24
    top = 30
    bottom = 44
    plot_w = float(width - left - right)
    plot_h = float(height - top - bottom)

    step = max(1, int(round(samples.size / 12000)))
    y_plot = np.clip(samples[::step], -1.0, 1.0)
    x_plot = np.arange(0, y_plot.size, dtype=np.float32) * float(step) / float(max(sr, 1))

    def x_map(t: float) -> float:
        tt = min(max(0.0, float(t)), duration)
        return left + (tt / duration) * plot_w

    def y_map(v: float) -> float:
        vv = min(max(-1.0, float(v)), 1.0)
        return top + ((1.0 - ((vv + 1.0) * 0.5)) * plot_h)

    points = " ".join(f"{x_map(float(t)):.2f},{y_map(float(v)):.2f}" for t, v in zip(x_plot, y_plot))
    ref_bounds = _interval_boundaries(ref_words)
    tgt_bounds = _interval_boundaries(tgt_words)

    def boundary_lines(bounds: Sequence[float], color: str, dash: str) -> List[str]:
        lines: List[str] = []
        for t in bounds:
            x = x_map(float(t))
            lines.append(
                f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{height-bottom}' "
                f"stroke='{color}' stroke-width='1.2' stroke-dasharray='{dash}' opacity='0.78' />"
            )
        return lines

    svg_lines: List[str] = []
    svg_lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    svg_lines.append("<rect width='100%' height='100%' fill='#FFFFFF'/>")
    svg_lines.append(f"<rect x='{left}' y='{top}' width='{plot_w:.2f}' height='{plot_h:.2f}' fill='#FAFAFA' stroke='#D3D3D3' stroke-width='1'/>")
    svg_lines.append(f"<polyline points='{points}' fill='none' stroke='#5A5A5A' stroke-width='1' />")
    svg_lines.extend(boundary_lines(ref_bounds, "#D64A4A", "6,4"))
    svg_lines.extend(boundary_lines(tgt_bounds, "#2E6FE3", "1,0"))

    zero_y = y_map(0.0)
    svg_lines.append(f"<line x1='{left}' y1='{zero_y:.2f}' x2='{width-right}' y2='{zero_y:.2f}' stroke='#B6B6B6' stroke-width='1'/>")
    svg_lines.append(f"<text x='{left}' y='18' font-size='14' font-family='Segoe UI, sans-serif' fill='#222'>{escape(title)}</text>")
    svg_lines.append(
        f"<text x='{width-right}' y='18' text-anchor='end' font-size='12' font-family='Segoe UI, sans-serif' fill='#444'>"
        f"mean boundary delta={delta_ms:.2f} ms</text>"
    )
    svg_lines.append(f"<text x='{left}' y='{height-14}' font-size='11' font-family='Segoe UI, sans-serif' fill='#555'>0s</text>")
    svg_lines.append(
        f"<text x='{width-right}' y='{height-14}' text-anchor='end' font-size='11' font-family='Segoe UI, sans-serif' fill='#555'>"
        f"{duration:.2f}s</text>"
    )
    svg_lines.append(f"<text x='{left+8}' y='{top+16}' font-size='11' font-family='Segoe UI, sans-serif' fill='#D64A4A'>{escape(ref_label)} boundary</text>")
    svg_lines.append(f"<text x='{left+180}' y='{top+16}' font-size='11' font-family='Segoe UI, sans-serif' fill='#2E6FE3'>{escape(tgt_label)} boundary</text>")
    svg_lines.append("</svg>")

    _safe_mkdir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines) + "\n")
    return True


def _percentile(values: Sequence[float], p: float) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return 0.0
    idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * float(p)))))
    return float(arr[idx])


def _target_label(target: str) -> str:
    t = str(target or "").strip().lower()
    if t == "mfa":
        return "MFA"
    if t == "sequence":
        return "Sequence"
    return t.upper() or "Target"


def _run_target_align(
    target: str,
    *,
    voicebank: str,
    dictionary_path: str,
    out_dir: str,
    language: str,
) -> Tuple[bool, str]:
    t = str(target or "").strip().lower()
    if t == "sequence":
        return run_sequence_align(
            "",
            voicebank,
            dictionary_path,
            out_dir,
            language=language,
            callback=_log,
        )
    return False, f"unsupported target: {target}"


def _collect_comparison_rows(
    *,
    wav_files: Sequence[str],
    ref_kind: str,
    ref_dir: str,
    target_dir: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        tgt_tg = os.path.join(target_dir, base + ".TextGrid")
        if not os.path.isfile(tgt_tg):
            continue

        ref_words = _load_reference_words(base=base, ref_kind=ref_kind, ref_dir=ref_dir)
        tgt_words = _read_active_intervals(tgt_tg, "words")
        if (not ref_words) or (not tgt_words):
            continue
        delta_ms = _pairwise_delta_ms(ref_words, tgt_words)
        sofa_metrics = _sofa_style_boundary_metrics(ref_words, tgt_words)
        rows.append(
            {
                "file": base,
                "wav_path": wav_path,
                "ref_words": len(ref_words),
                "target_words": len(tgt_words),
                "delta_ms": float(delta_ms),
                "mismatch": int(len(ref_words) != len(tgt_words)),
                **sofa_metrics,
            }
        )
    return rows


def _build_target_report(
    *,
    target: str,
    wav_files: Sequence[str],
    ref_kind: str,
    ref_dir: str,
    ref_label: str,
    target_dir: str,
    out_root: str,
    max_plots: int,
) -> Dict[str, object]:
    label = _target_label(target)
    rows = _collect_comparison_rows(
        wav_files=wav_files,
        ref_kind=ref_kind,
        ref_dir=ref_dir,
        target_dir=target_dir,
    )
    if not rows:
        return {
            "target": target,
            "target_label": label,
            "ref_label": ref_label,
            "stats": {"compared_files": 0},
            "top_diffs": [],
        }

    rows_sorted = sorted(rows, key=lambda x: (int(x["mismatch"]), float(x["delta_ms"])), reverse=True)
    top = rows_sorted[: max(1, int(max_plots))]

    plot_dir = _safe_mkdir(os.path.join(out_root, "plots", str(target).strip().lower()))
    top_with_plot: List[Dict[str, object]] = []
    for item in top:
        base = str(item["file"])
        wav_path = str(item["wav_path"])
        tgt_tg = os.path.join(target_dir, base + ".TextGrid")
        ref_words = _load_reference_words(base=base, ref_kind=ref_kind, ref_dir=ref_dir)
        tgt_words = _read_active_intervals(tgt_tg, "words")

        plot_name = f"{base}_overlay.{_PLOT_EXT}"
        plot_path = os.path.join(plot_dir, plot_name)
        ok_plot = _plot_overlay(
            wav_path,
            ref_words,
            tgt_words,
            out_path=plot_path,
            title=f"{base} ({label})",
            delta_ms=float(item["delta_ms"]),
            ref_label=ref_label,
            tgt_label=label,
        )

        enriched = dict(item)
        enriched["plot"] = plot_path if ok_plot else ""
        enriched["plot_rel"] = os.path.relpath(plot_path, out_root) if ok_plot else ""
        top_with_plot.append(enriched)

    deltas = [float(r["delta_ms"]) for r in rows if float(r["delta_ms"]) < 900000.0]
    token_match_rate = statistics.mean([1.0 if int(r["mismatch"]) == 0 else 0.0 for r in rows]) if rows else 0.0
    stats = {
        "compared_files": int(len(rows)),
        "token_count_match_rate": float(token_match_rate),
        "mean_delta_ms": float(statistics.mean(deltas)) if deltas else 0.0,
        "p90_delta_ms": float(_percentile(deltas, 0.9)) if deltas else 0.0,
    }
    sofa_rows = [r for r in rows if "boundary_edit_ratio" in r]
    if sofa_rows:
        stats.update(
            {
                "boundary_edit_ratio": float(
                    statistics.mean([float(r.get("boundary_edit_ratio", 0.0) or 0.0) for r in sofa_rows])
                ),
                "ber_10ms": float(statistics.mean([float(r.get("ber_10ms", 0.0) or 0.0) for r in sofa_rows])),
                "ber_20ms": float(statistics.mean([float(r.get("ber_20ms", 0.0) or 0.0) for r in sofa_rows])),
                "ber_50ms": float(statistics.mean([float(r.get("ber_50ms", 0.0) or 0.0) for r in sofa_rows])),
                "interval_iou": float(
                    statistics.mean([float(r.get("interval_iou", 0.0) or 0.0) for r in sofa_rows])
                ),
            }
        )

    return {
        "target": target,
        "target_label": label,
        "ref_label": ref_label,
        "stats": stats,
        "top_diffs": top_with_plot,
    }


def _write_summary_md(path: str, payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Alignment Comparison Report")
    lines.append("")
    lines.append(f"- voicebank: `{payload.get('voicebank', '')}`")
    lines.append(f"- language: `{payload.get('language', '')}`")
    lines.append(f"- selected_targets: `{', '.join(payload.get('selected_targets', []) or [])}`")
    lines.append(f"- reference_mode: `{payload.get('reference_mode', '')}`")
    lines.append(f"- reference_label: `{payload.get('reference_label', '')}`")
    lines.append(f"- reference_source: `{payload.get('reference_source', '')}`")
    lines.append(f"- truth_lab_dir: `{payload.get('truth_lab_dir', '')}`")
    lines.append(f"- mfa_source: `{payload.get('mfa_source', '')}`")
    lines.append(f"- mfa_ok: `{payload.get('mfa_ok', False)}`")
    lines.append("")

    target_results = payload.get("targets", {}) or {}
    for target_key in payload.get("selected_targets", []) or []:
        report = target_results.get(target_key, {}) if isinstance(target_results, dict) else {}
        label = str(report.get("target_label") or _target_label(str(target_key)))
        ref_label = str(report.get("ref_label") or payload.get("reference_label") or "Reference")
        lines.append(f"## {ref_label} vs {label}")
        lines.append("")
        lines.append(f"- target_ok: `{bool(report.get('ok', False))}`")
        lines.append(f"- target_message: `{str(report.get('message', '') or '')}`")
        stats = report.get("stats", {}) or {}
        lines.append(f"- compared_files: `{stats.get('compared_files', 0)}`")
        lines.append(f"- token_count_match_rate: `{float(stats.get('token_count_match_rate', 0.0)):.3f}`")
        lines.append(f"- mean_delta_ms: `{float(stats.get('mean_delta_ms', 0.0)):.2f}`")
        lines.append(f"- p90_delta_ms: `{float(stats.get('p90_delta_ms', 0.0)):.2f}`")
        if "boundary_edit_ratio" in stats:
            lines.append(f"- boundary_edit_ratio: `{float(stats.get('boundary_edit_ratio', 0.0)):.3f}`")
            lines.append(f"- ber_10ms: `{float(stats.get('ber_10ms', 0.0)):.3f}`")
            lines.append(f"- ber_20ms: `{float(stats.get('ber_20ms', 0.0)):.3f}`")
            lines.append(f"- ber_50ms: `{float(stats.get('ber_50ms', 0.0)):.3f}`")
            lines.append(f"- interval_iou: `{float(stats.get('interval_iou', 0.0)):.3f}`")
        lines.append("")
        lines.append("| file | delta(ms) | mfa_words | target_words | plot |")
        lines.append("|---|---:|---:|---:|---|")
        for item in list(report.get("top_diffs", []) or []):
            lines.append(
                "| {file} | {delta:.2f} | {ref_words} | {target_words} | {plot} |".format(
                    file=str(item.get("file", "")),
                    delta=float(item.get("delta_ms", 0.0) or 0.0),
                    ref_words=int(item.get("ref_words", 0) or 0),
                    target_words=int(item.get("target_words", 0) or 0),
                    plot=str(item.get("plot_rel", "")),
                )
            )
        lines.append("")

    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _resolve_targets(raw_target: str) -> List[str]:
    _ = str(raw_target or "sequence").strip().lower()
    return ["sequence"]


def _resolve_reference_mode(raw_mode: str) -> str:
    mode = str(raw_mode or "truth").strip().lower()
    if mode in {"mfa", "truth"}:
        return mode
    return "auto"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MFA with Sequence alignments visually.")
    parser.add_argument("--voicebank", default=_DEFAULT_SODA_D4, help="Voicebank wav root folder")
    parser.add_argument("--language", default="japanese", choices=["japanese", "korean"], help="Alignment language")
    parser.add_argument("--target", default="sequence", choices=["sequence"], help="Target engine to compare against MFA")
    parser.add_argument(
        "--reference",
        default="truth",
        choices=["truth", "mfa", "auto"],
        help="Reference for comparison (truth=manual sinsy lab)",
    )
    parser.add_argument(
        "--truth-lab-dir",
        default=_DEFAULT_TRUTH_LAB_DIR,
        help="Manual Sinsy lab folder used as GT when reference=truth/auto",
    )
    parser.add_argument("--max-plots", type=int, default=12, help="How many difference plots to generate per target")
    parser.add_argument("--mfa-profile", default="default", help="MFA profile code")
    parser.add_argument("--out-root", default="", help="Optional explicit output root")
    args = parser.parse_args()

    voicebank = os.path.abspath(args.voicebank)
    if not os.path.isdir(voicebank):
        raise SystemExit(f"voicebank not found: {voicebank}")

    selected_targets = _resolve_targets(args.target)
    reference_mode = _resolve_reference_mode(args.reference)
    truth_lab_dir = (
        os.path.abspath(str(args.truth_lab_dir))
        if str(args.truth_lab_dir or "").strip()
        else ""
    )

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_root = os.path.join(os.getcwd(), "temp", f"alignment_compare_{os.path.basename(voicebank)}_{stamp}")
    out_root = os.path.abspath(args.out_root) if str(args.out_root or "").strip() else auto_root
    _safe_mkdir(out_root)

    ref_dir = _safe_mkdir(os.path.join(out_root, "mfa_textgrids"))
    target_dirs = {"sequence": _safe_mkdir(os.path.join(out_root, "sequence_textgrids"))}

    dictionary_path = os.path.join(
        voicebank,
        "japanese_dict.txt" if args.language == "japanese" else "korean_dict.txt",
    )

    _log(f"[1/4] MFA align run -> {ref_dir}")
    mfa_result = run_alignment_with_fallback(
        language=args.language,
        wav_folder=voicebank,
        dictionary_path=dictionary_path,
        output_folder=ref_dir,
        primary_aligner="mfa",
        fallback_aligner="",
        mfa_path="",
        mfa_align_profile=str(args.mfa_profile or "default"),
        callback=_log,
    )
    mfa_ok = bool(mfa_result.get("ok", False))
    mfa_source = "fresh_mfa"

    existing_ref = os.path.join(voicebank, "textgrids")
    if (not mfa_ok) and os.path.isdir(existing_ref):
        _log("[WARN] MFA run failed. Falling back to existing voicebank/textgrids as MFA reference.")
        ref_dir = existing_ref
        mfa_source = "existing_textgrids"

    ref_kind = "mfa"
    reference_label = "MFA"
    reference_source = ref_dir
    if reference_mode in {"truth", "auto"}:
        if truth_lab_dir and os.path.isdir(truth_lab_dir):
            ref_kind = "truth"
            reference_label = "ManualGT"
            reference_source = truth_lab_dir
        elif reference_mode == "truth":
            _log(f"[WARN] truth-lab-dir not found. fallback to MFA reference: {truth_lab_dir}")
    comparison_ref_dir = truth_lab_dir if ref_kind == "truth" else ref_dir

    report_targets = list(selected_targets)
    if ref_kind == "truth":
        report_targets = ["mfa"] + [t for t in report_targets if t != "mfa"]

    target_runtime: Dict[str, Dict[str, object]] = {}
    for idx, target in enumerate(selected_targets, start=2):
        out_dir = target_dirs[target]
        _log(f"[{idx}/4] {target.upper()} align run -> {out_dir}")
        ok, msg = _run_target_align(
            target,
            voicebank=voicebank,
            dictionary_path=dictionary_path,
            out_dir=out_dir,
            language=args.language,
        )
        target_runtime[target] = {
            "ok": bool(ok),
            "message": str(msg or ""),
            "output_dir": out_dir,
        }
    target_runtime["mfa"] = {
        "ok": bool(mfa_ok),
        "message": str(mfa_result.get("message", "") or ""),
        "output_dir": ref_dir,
    }

    _log("[4/4] Compare boundaries + generate plots")
    if plt is None:
        _log("[INFO] matplotlib not available. Using built-in SVG overlay renderer.")
    wav_files = _list_top_level_wavs(voicebank)

    target_reports: Dict[str, Dict[str, object]] = {}
    for target in report_targets:
        target_dir = ref_dir if target == "mfa" else target_dirs[target]
        report = _build_target_report(
            target=target,
            wav_files=wav_files,
            ref_kind=ref_kind,
            ref_dir=comparison_ref_dir,
            ref_label=reference_label,
            target_dir=target_dir,
            out_root=out_root,
            max_plots=max(1, int(args.max_plots)),
        )
        runtime = target_runtime.get(target, {})
        report["ok"] = bool(runtime.get("ok", False))
        report["message"] = str(runtime.get("message", "") or "")
        report["output_dir"] = str(runtime.get("output_dir", "") or "")
        target_reports[target] = report

    payload: Dict[str, object] = {
        "voicebank": voicebank,
        "language": args.language,
        "selected_targets": report_targets,
        "reference_mode": reference_mode,
        "reference_label": reference_label,
        "reference_source": reference_source,
        "truth_lab_dir": truth_lab_dir,
        "mfa_ok": bool(mfa_ok),
        "mfa_source": mfa_source,
        "mfa_result": mfa_result,
        "targets": target_reports,
        "out_root": out_root,
    }

    json_path = os.path.join(out_root, "summary.json")
    md_path = os.path.join(out_root, "summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_summary_md(md_path, payload)

    _log("[DONE] Alignment comparison complete")
    _log(f"Output root: {out_root}")
    _log(f"Summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
