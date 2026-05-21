"""Plot boundary-score and phone-aware pseudo-target diagnostics for one wav.

This is a manual inspection tool for the boundary-decoder OTO stack. It makes
the hidden training signal visible before a long phone-aware auxiliary-head
training run: mel, boundary posterior/target, source OTO row spans, CVS target,
and phone identity target/prediction are rendered into one PNG.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import BoundaryScorerConfig, load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import (
    build_boundary_target_map,
    build_phone_aware_target_map,
    phone_aware_role_family,
    training_rows_to_wav_groups,
)
from core.coarse_crnn.boundary_types import (
    BOUNDARY_LABELS,
    PHONE_AWARE_CONSONANT_LABELS,
    PHONE_AWARE_CVS_LABELS,
    PHONE_AWARE_IGNORE_INDEX,
    PHONE_AWARE_VOWEL_LABELS,
    AbsoluteOtoAnchors,
    BoundaryFrameScores,
    OtoRowSpec,
)


DEFAULT_ROLE_FILTER = "vc,vv,v-cv"
DEFAULT_FORMAT_FILTER = "cvvc"
BOUNDARY_PLOT_LABELS = (
    "consonant_onset",
    "vowel_start",
    "vowel_end",
    "transition_peak",
    "next_onset",
    "silence_boundary",
)
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
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a phone-aware boundary-decoder debug PNG for one wav from an OTO JSONL manifest."
    )
    parser.add_argument("--manifest", required=True, help="OTO training/eval JSONL manifest.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--model", default="", help="Optional boundary scorer checkpoint. Omit for target-only plot.")
    parser.add_argument("--device", default="cpu", help="Torch device for model inference. Default: cpu.")
    parser.add_argument("--wav", default="", help="Exact wav path or wav basename to inspect.")
    parser.add_argument("--alias", default="", help="Optional alias text to choose the target wav.")
    parser.add_argument("--line-index", type=int, default=None, help="Optional manifest/source OTO line_index to choose the target row.")
    parser.add_argument("--role-filter", default=DEFAULT_ROLE_FILTER, help="Comma-separated alias_role filter for auto-pick.")
    parser.add_argument("--format-filter", default=DEFAULT_FORMAT_FILTER, help="Comma-separated format_type filter for auto-pick. Empty disables.")
    parser.add_argument("--max-rows-per-wav", type=int, default=80, help="Maximum rows to draw if a wav has many aliases.")
    parser.add_argument("--dpi", type=int, default=140, help="PNG DPI.")
    args = parser.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")

    selection = _select_manifest_rows(
        manifest,
        wav=args.wav,
        alias=args.alias,
        line_index=args.line_index,
        role_filter=_parse_csv(args.role_filter),
        format_filter=_parse_csv(args.format_filter),
        max_rows_per_wav=max(1, int(args.max_rows_per_wav)),
    )
    if not selection.rows:
        raise SystemExit("no matching rows found in manifest")

    groups = training_rows_to_wav_groups(selection.rows)
    wav_path = os.path.abspath(selection.wav_path)
    row_pairs = groups.get(wav_path)
    if row_pairs is None:
        # Some manifests normalize paths differently; fall back by basename.
        wav_base = os.path.basename(wav_path).lower()
        row_pairs = next((pairs for key, pairs in groups.items() if os.path.basename(key).lower() == wav_base), None)
    if not row_pairs:
        raise SystemExit(f"selected rows could not be converted to OTO row specs: {wav_path}")

    config = BoundaryScorerConfig()
    score_map: BoundaryFrameScores | None = None
    model_meta: dict[str, Any] = {}
    if str(args.model or "").strip():
        model_path = Path(args.model)
        if not model_path.is_file():
            raise SystemExit(f"model not found: {model_path}")
        model, config, model_meta = load_boundary_checkpoint(str(model_path), map_location="cpu")
        score_map = infer_boundary_scores_with_model(model=model, config=config, wav_path=wav_path, device=str(args.device))

    samples, sr, duration_sec = load_wav_mono(wav_path, target_sr=int(config.sample_rate))
    mel, hop_sec = log_mel_spectrogram(
        samples,
        sr,
        n_mels=int(config.n_mels),
        frame_ms=float(config.frame_ms),
        hop_ms=float(config.hop_ms),
    )
    duration_ms = max(float(duration_sec) * 1000.0, _max_anchor_ms(row_pairs), float(selection.duration_ms or 0.0), 1.0)
    frame_count = int(mel.shape[0])
    hop_ms = float(hop_sec) * 1000.0
    target_times, boundary_target = build_boundary_target_map(
        row_pairs,
        duration_ms=duration_ms,
        hop_ms=hop_ms,
        frame_count=frame_count,
    )
    phone_times, phone_target = build_phone_aware_target_map(
        row_pairs,
        duration_ms=duration_ms,
        hop_ms=hop_ms,
        frame_count=frame_count,
    )

    _render_plot(
        out_path=Path(args.out),
        wav_path=wav_path,
        selection=selection,
        row_pairs=row_pairs,
        mel=mel,
        duration_ms=duration_ms,
        target_times=target_times,
        boundary_target=boundary_target,
        phone_times=phone_times,
        phone_target=phone_target,
        score_map=score_map,
        model_path=str(args.model or ""),
        model_meta=model_meta,
        dpi=int(args.dpi),
    )
    print(f"[phone-aware-debug] wrote {Path(args.out).resolve()}")
    return 0


class _Selection:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        target_row: dict[str, Any],
        wav_path: str,
        duration_ms: float,
        reason: str,
    ) -> None:
        self.rows = rows
        self.target_row = target_row
        self.wav_path = wav_path
        self.duration_ms = duration_ms
        self.reason = reason


def _select_manifest_rows(
    manifest: Path,
    *,
    wav: str,
    alias: str,
    line_index: int | None,
    role_filter: set[str],
    format_filter: set[str],
    max_rows_per_wav: int,
) -> _Selection:
    target = _find_target_row(
        manifest,
        wav=wav,
        alias=alias,
        line_index=line_index,
        role_filter=role_filter,
        format_filter=format_filter,
    )
    if target is None:
        raise SystemExit("target row not found; relax --role-filter/--format-filter or pass --wav")
    wav_path = os.path.abspath(str(target.get("audio", "") or ""))
    if not wav_path:
        raise SystemExit("target row is missing audio path")
    rows = _collect_wav_rows(manifest, wav_path=wav_path, wav_name=str(target.get("wav", "") or ""))
    rows.sort(key=_row_line_index)
    if len(rows) > max_rows_per_wav:
        rows = _window_rows(rows, target_line_index=_row_line_index(target), max_rows=max_rows_per_wav)
    return _Selection(
        rows=rows,
        target_row=target,
        wav_path=wav_path,
        duration_ms=float(target.get("duration_ms", 0.0) or 0.0),
        reason=_selection_reason(target, explicit_wav=wav),
    )


def _find_target_row(
    manifest: Path,
    *,
    wav: str,
    alias: str,
    line_index: int | None,
    role_filter: set[str],
    format_filter: set[str],
) -> dict[str, Any] | None:
    explicit_wav = str(wav or "").strip()
    for row in _iter_jsonl(manifest):
        if explicit_wav and not _row_matches_wav(row, explicit_wav):
            continue
        if alias and str(row.get("alias", "") or "") != str(alias):
            continue
        if line_index is not None and _row_line_index(row, default=-1) != int(line_index):
            continue
        if not explicit_wav:
            role = str(row.get("alias_role", "") or "").strip().lower()
            fmt = str(row.get("format_type", "") or "").strip().lower()
            if role_filter and role not in role_filter:
                continue
            if format_filter and fmt not in format_filter:
                continue
        return row
    return None


def _collect_wav_rows(manifest: Path, *, wav_path: str, wav_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    wav_abs = os.path.abspath(wav_path)
    wav_base = os.path.basename(str(wav_name or wav_path)).lower()
    for row in _iter_jsonl(manifest):
        row_audio = os.path.abspath(str(row.get("audio", "") or ""))
        row_wav = os.path.basename(str(row.get("wav", "") or row_audio)).lower()
        if row_audio == wav_abs or (wav_base and row_wav == wav_base):
            out.append(row)
    return out


def _window_rows(rows: list[dict[str, Any]], *, target_line_index: int, max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    nearest = min(range(len(rows)), key=lambda idx: abs(_row_line_index(rows[idx]) - target_line_index))
    half = max_rows // 2
    start = max(0, nearest - half)
    end = min(len(rows), start + max_rows)
    start = max(0, end - max_rows)
    return rows[start:end]


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _row_line_index(row: dict[str, Any], *, default: int = 0) -> int:
    try:
        return int(row.get("line_index", default))
    except Exception:
        return int(default)


def _parse_csv(text: str) -> set[str]:
    return {item.strip().lower() for item in str(text or "").split(",") if item.strip()}


def _row_matches_wav(row: dict[str, Any], wav: str) -> bool:
    needle = os.path.abspath(wav) if os.path.isabs(wav) else str(wav).strip().lower()
    row_audio = str(row.get("audio", "") or "")
    row_wav = str(row.get("wav", "") or "")
    if os.path.isabs(wav):
        return os.path.abspath(row_audio) == needle
    return row_wav.lower() == needle or os.path.basename(row_audio).lower() == needle or row_audio.lower() == needle


def _selection_reason(row: dict[str, Any], *, explicit_wav: str) -> str:
    if explicit_wav:
        return "explicit wav"
    return (
        f"auto-picked role={row.get('alias_role', '')} format={row.get('format_type', '')} "
        f"alias={row.get('alias', '')} line={row.get('line_index', '')}"
    )


def _render_plot(
    *,
    out_path: Path,
    wav_path: str,
    selection: _Selection,
    row_pairs: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]],
    mel: np.ndarray,
    duration_ms: float,
    target_times: list[float],
    boundary_target: np.ndarray,
    phone_times: list[float],
    phone_target,
    score_map: BoundaryFrameScores | None,
    model_path: str,
    model_meta: dict[str, Any],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _configure_plot_fonts(plt)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.25, 1.15, 1.1, 1.1, 1.1]},
    )
    title = (
        f"{os.path.basename(wav_path)} | {selection.reason} | rows={len(row_pairs)} | "
        f"model={os.path.basename(model_path) if model_path else 'target-only'}"
    )
    fig.suptitle(title, fontsize=11)

    _plot_mel(axes[0], mel=mel, duration_ms=duration_ms)
    _plot_anchor_lines(axes[0], row_pairs=row_pairs, annotate=True)
    axes[0].set_ylabel("mel")

    _plot_boundary_panel(
        axes[1],
        target_times=target_times,
        boundary_target=boundary_target,
        score_map=score_map,
    )
    axes[1].set_title("Boundary posterior vs target (solid=model, dashed=source target)", fontsize=10)

    ax_rows = axes[2]
    ax_rows.set_title("Source OTO rows used for pseudo-targets", fontsize=10)
    for idx, (spec, anchors) in enumerate(row_pairs):
        color = ROLE_COLORS.get(str(spec.role).lower(), "#607d8b")
        left = max(0.0, float(anchors.offset_abs))
        right = min(duration_ms, max(left + 1.0, float(anchors.cutoff_abs)))
        ax_rows.add_patch(Rectangle((left, idx - 0.42), right - left, 0.84, color=color, alpha=0.75, lw=0))
        label = f"{spec.alias} [{spec.role}/{phone_aware_role_family(spec)}]"
        ax_rows.text(0.5 * (left + right), idx, label, ha="center", va="center", fontsize=6, color="white", clip_on=True)
    ax_rows.set_ylim(-0.8, max(1.0, len(row_pairs) - 0.2))
    ax_rows.set_yticks([])
    ax_rows.grid(True, axis="x", alpha=0.18)

    _plot_cvs_panel(
        axes[3],
        times=phone_times,
        target=phone_target.cvs_target,
        mask=phone_target.cvs_mask,
        score_dict=score_map.cvs_scores if score_map is not None else None,
    )
    axes[3].set_title("CVS phone class (target mask + optional aux posterior)", fontsize=10)

    _plot_identity_panel(
        axes[4],
        title="Consonant identity",
        times=phone_times,
        target=phone_target.consonant_target,
        mask=phone_target.consonant_mask,
        labels=PHONE_AWARE_CONSONANT_LABELS,
        score_dict=score_map.consonant_scores if score_map is not None else None,
    )
    _plot_identity_panel(
        axes[5],
        title="Vowel identity",
        times=phone_times,
        target=phone_target.vowel_target,
        mask=phone_target.vowel_mask,
        labels=PHONE_AWARE_VOWEL_LABELS,
        score_dict=score_map.vowel_scores if score_map is not None else None,
    )

    for ax in axes:
        ax.set_xlim(0.0, duration_ms)
        ax.grid(True, axis="x", alpha=0.16)
    axes[-1].set_xlabel("time (ms)")

    footer = {
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wav": wav_path,
        "target_alias": selection.target_row.get("alias", ""),
        "target_role": selection.target_row.get("alias_role", ""),
        "model_aux": bool(score_map and score_map.cvs_scores and score_map.consonant_scores and score_map.vowel_scores),
        "model_meta": model_meta,
    }
    fig.text(0.01, 0.005, json.dumps(footer, ensure_ascii=False, default=str)[:1400], fontsize=6, color="#555555")
    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=max(72, int(dpi)))
    plt.close(fig)


def _configure_plot_fonts(plt) -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return
    preferred = ("Malgun Gothic", "Yu Gothic", "Meiryo", "Noto Sans CJK KR", "DejaVu Sans")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = [name for name in preferred if name in installed]
    if family:
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def _plot_mel(ax, *, mel: np.ndarray, duration_ms: float) -> None:
    arr = np.asarray(mel, dtype=np.float32)
    if arr.size <= 0:
        ax.text(0.5, 0.5, "empty mel", transform=ax.transAxes, ha="center", va="center")
        return
    vmax = float(np.quantile(arr, 0.995))
    vmin = float(np.quantile(arr, 0.05))
    ax.imshow(
        arr.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, duration_ms, 0.0, float(arr.shape[1])],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )


def _plot_anchor_lines(ax, *, row_pairs: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]], annotate: bool = False) -> None:
    for spec, anchors in row_pairs:
        for label, value, color, style in (
            ("off", anchors.offset_abs, "#9e9e9e", ":"),
            ("pre", anchors.pre_abs, "#00c853", "-"),
            ("cons", anchors.consonant_abs, "#ffd600", "-"),
            ("cut", anchors.cutoff_abs, "#90a4ae", ":"),
        ):
            ax.axvline(float(value), color=color, linestyle=style, lw=0.8, alpha=0.75)
            if annotate and label in {"pre", "cons"}:
                ax.text(float(value), 0.98, label, transform=ax.get_xaxis_transform(), fontsize=6, rotation=90, va="top", color=color)


def _plot_boundary_panel(
    ax,
    *,
    target_times: list[float],
    boundary_target: np.ndarray,
    score_map: BoundaryFrameScores | None,
) -> None:
    times = np.asarray(target_times, dtype=np.float32)
    target = np.asarray(boundary_target, dtype=np.float32)
    for label in BOUNDARY_PLOT_LABELS:
        if label not in BOUNDARY_LABELS:
            continue
        idx = BOUNDARY_LABELS.index(label)
        line = ax.plot(times, target[:, idx], linestyle="--", linewidth=1.0, alpha=0.75, label=f"{label} target")[0]
        if score_map is not None and label in score_map.scores:
            score_times = np.asarray(score_map.times_ms, dtype=np.float32)
            ax.plot(score_times, np.asarray(score_map.scores[label], dtype=np.float32), color=line.get_color(), linewidth=1.1, label=f"{label} pred")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("prob")
    ax.legend(loc="upper right", fontsize=6, ncol=3)


def _plot_cvs_panel(
    ax,
    *,
    times: list[float],
    target: np.ndarray,
    mask: np.ndarray,
    score_dict: dict[str, list[float]] | None,
) -> None:
    t = np.asarray(times, dtype=np.float32)
    target_idx = _masked_target_series(target, mask)
    if score_dict:
        for label in PHONE_AWARE_CVS_LABELS:
            vals = np.asarray(score_dict.get(label, []), dtype=np.float32)
            if vals.size:
                ax.plot(t[: vals.size], vals, linewidth=1.2, label=f"{label} pred")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("prob")
        ax2 = ax.twinx()
        ax2.step(t, target_idx, where="mid", color="#111111", linewidth=1.0, alpha=0.8, label="target idx")
        ax2.set_ylim(-0.3, len(PHONE_AWARE_CVS_LABELS) - 0.7)
        ax2.set_yticks(range(len(PHONE_AWARE_CVS_LABELS)))
        ax2.set_yticklabels(PHONE_AWARE_CVS_LABELS, fontsize=7)
        ax.legend(loc="upper right", fontsize=7)
        return
    ax.step(t, target_idx, where="mid", color="#111111", linewidth=1.2, label="target idx")
    ax.set_ylim(-0.3, len(PHONE_AWARE_CVS_LABELS) - 0.7)
    ax.set_yticks(range(len(PHONE_AWARE_CVS_LABELS)))
    ax.set_yticklabels(PHONE_AWARE_CVS_LABELS, fontsize=7)
    ax.legend(loc="upper right", fontsize=7)


def _plot_identity_panel(
    ax,
    *,
    title: str,
    times: list[float],
    target: np.ndarray,
    mask: np.ndarray,
    labels: tuple[str, ...],
    score_dict: dict[str, list[float]] | None,
) -> None:
    t = np.asarray(times, dtype=np.float32)
    target_idx = _masked_target_series(target, mask)
    ax.step(t, target_idx, where="mid", color="#111111", linewidth=1.1, label="target", alpha=0.9)
    if score_dict:
        pred, conf = _argmax_score_series(score_dict, labels)
        if pred.size:
            ax.plot(t[: pred.size], pred, color="#d62728", linewidth=1.0, alpha=0.85, label="pred argmax")
            confident = np.where(conf >= 0.45, pred, np.nan)
            ax.scatter(t[: confident.size], confident, s=5, color="#d62728", alpha=0.45, label="pred conf>=0.45")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    step = 1 if len(labels) <= 22 else 2
    ticks = list(range(0, len(labels), step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[idx] for idx in ticks], fontsize=6)
    ax.legend(loc="upper right", fontsize=7)


def _masked_target_series(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    tgt = np.asarray(target)
    m = np.asarray(mask, dtype=np.float32) > 0.0
    out = np.full((tgt.shape[0],), np.nan, dtype=np.float32)
    valid = m & (tgt != PHONE_AWARE_IGNORE_INDEX)
    out[valid] = tgt[valid].astype(np.float32)
    return out


def _argmax_score_series(score_dict: dict[str, list[float]], labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    cols = []
    for label in labels:
        vals = np.asarray(score_dict.get(label, []), dtype=np.float32)
        if vals.size <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        cols.append(vals)
    arr = np.stack(cols, axis=1)
    pred = np.argmax(arr, axis=1).astype(np.float32)
    conf = np.max(arr, axis=1).astype(np.float32)
    return pred, conf


def _max_anchor_ms(row_pairs: list[tuple[OtoRowSpec, AbsoluteOtoAnchors]]) -> float:
    value = 0.0
    for _spec, anchors in row_pairs:
        value = max(
            value,
            float(anchors.offset_abs),
            float(anchors.pre_abs),
            float(anchors.consonant_abs),
            float(anchors.cutoff_abs),
        )
    return value


if __name__ == "__main__":
    raise SystemExit(main())
