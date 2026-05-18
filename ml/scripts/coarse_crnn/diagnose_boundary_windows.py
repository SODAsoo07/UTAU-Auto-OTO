"""Render per-row diagnostic PNGs of the boundary-decoder window decisions.

Unlike `plot_phone_aware_debug.py` (which renders training targets), this script
runs the **inference pipeline** end-to-end and visualizes what the decoder
actually produces for OTO window edges (offset/overlap/preutterance/consonant/
cutoff) on top of the mel spectrogram and frame-level boundary scores.

Use this to localize the cause when CV/VC windows fail to enclose the alias
content in listening tests:

  - Are the model's score peaks already in the wrong place?
    → encoder/head training issue.
  - Are the score peaks roughly correct but the decoder picks wrong frames?
    → decoder heuristic issue (most common).
  - Is the window peak-tight without safety padding for attack/release?
    → decoder padding policy issue.

Inputs match the production runtime exactly: wav_dir + source OTO + model path.
Source OTO numerics never enter the rendered window (source contract is
enforced in `boundary_generator.py`), so the predicted window is purely
model+decoder output.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core.coarse_crnn.audio import load_wav_mono, log_mel_spectrogram
from core.coarse_crnn.boundary_audio_features import (
    DEFAULT_GAP_MS as LWC_DEFAULT_GAP_MS,
    DEFAULT_PEAK_MIN_GAP_MS as LWC_DEFAULT_MIN_GAP_MS,
    DEFAULT_THRESHOLD as LWC_DEFAULT_THRESHOLD,
    compute_long_window_candidates,
    lwc_enabled_by_env,
    lwc_float_from_env,
    lwc_scales_from_env,
)
from core.coarse_crnn.boundary_candidates import (
    audio_candidates_to_boundary_candidates,
    merge_candidates,
    peak_candidates_from_scores,
)
from core.coarse_crnn.boundary_scorer_inference import infer_boundary_scores_with_model
from core.coarse_crnn.boundary_scorer_model import load_boundary_checkpoint
from core.coarse_crnn.boundary_targets import (
    absolute_anchors_to_oto_params,
    load_row_specs_from_source_oto,
)
from core.coarse_crnn.oto_audio_candidates import compute_audio_candidates
from core.coarse_crnn.oto_predictor_generator import resolve_boundary_scorer_model_path
from core.coarse_crnn.training import resolve_torch_device
from core.coarse_crnn.wav_decoder import decode_wav_rows
from core.no_mfa_oto_builder import resolve_no_mfa_source_oto


_PLOT_SCORE_LABELS = (
    "syllable_onset",
    "consonant_onset",
    "vowel_start",
    "vowel_stable",
    "vowel_end",
    "transition_peak",
    "next_onset",
    "silence_boundary",
)
_SCORE_COLORS = {
    "syllable_onset": "#1f77b4",
    "consonant_onset": "#d62728",
    "vowel_start": "#2ca02c",
    "vowel_stable": "#17becf",
    "vowel_end": "#9467bd",
    "transition_peak": "#8c564b",
    "next_onset": "#e377c2",
    "silence_boundary": "#7f7f7f",
}


def _mean_quality(quality_scores) -> float:
    if not quality_scores:
        return 1.0
    arr = np.asarray(quality_scores, dtype=np.float32)
    if arr.size == 0:
        return 1.0
    return float(np.clip(arr.mean(), 0.0, 1.0))


def _estimate_audio_reliability(audio) -> float:
    # Re-use the same heuristic as the production generator's import.
    from core.coarse_crnn.boundary_generator import _estimate_audio_reliability as _impl

    return float(_impl(audio))


def _run_pipeline_for_wav(
    *,
    model,
    config,
    device,
    wav_path: str,
    specs,
):
    """Mirror of `generate_oto_with_boundary_decoder`'s per-wav inner block,
    short of the residual/BPM/shift post-processing. Returns the raw decoded
    rows + frame-level scores + computed audio metadata so the plot has access
    to everything the production pipeline saw.
    """
    score_map = infer_boundary_scores_with_model(
        model=model,
        config=config,
        wav_path=wav_path,
        device=str(device),
    )
    audio = compute_audio_candidates(wav_path)
    model_quality = _mean_quality(score_map.quality_scores)
    audio_reliability = _estimate_audio_reliability(audio)
    model_cands = peak_candidates_from_scores(score_map)
    audio_cands = audio_candidates_to_boundary_candidates(audio)
    if lwc_enabled_by_env(default=True):
        try:
            lwc = compute_long_window_candidates(
                wav_path=wav_path,
                scales_ms=lwc_scales_from_env(),
                gap_ms=lwc_float_from_env("UTOA_BOUNDARY_LWC_GAP_MS", LWC_DEFAULT_GAP_MS),
                threshold=lwc_float_from_env("UTOA_BOUNDARY_LWC_THRESHOLD", LWC_DEFAULT_THRESHOLD),
                min_gap_ms=lwc_float_from_env("UTOA_BOUNDARY_LWC_MIN_GAP_MS", LWC_DEFAULT_MIN_GAP_MS),
                active_start_ms=float(audio.active_start_ms),
                active_end_ms=float(audio.active_end_ms),
            )
            audio_cands = audio_cands + lwc
        except Exception:
            pass
    merged = merge_candidates(
        model_candidates=model_cands,
        audio_candidates=audio_cands,
        model_quality=model_quality,
        audio_reliability=audio_reliability,
    )
    decoded = decode_wav_rows(
        wav_path=wav_path,
        duration_ms=float(audio.duration_ms),
        row_specs=specs,
        candidates=merged,
        active_start_ms=float(audio.active_start_ms),
        active_end_ms=float(audio.active_end_ms),
        model_quality=model_quality,
        audio_reliability=audio_reliability,
        posterior_scores=score_map.scores,
        posterior_times_ms=score_map.times_ms,
        cvs_scores=score_map.cvs_scores,
        consonant_scores=score_map.consonant_scores,
        vowel_scores=score_map.vowel_scores,
        consonant_family_scores=score_map.consonant_family_scores,
        vowel_nucleus_scores=score_map.vowel_nucleus_scores,
        vowel_glide_scores=score_map.vowel_glide_scores,
    )
    return score_map, audio, decoded


def _mel_for_plot(wav_path: str, sample_rate: int) -> tuple[np.ndarray, float]:
    samples, sr, _duration = load_wav_mono(wav_path, target_sr=int(sample_rate))
    features, hop_sec = log_mel_spectrogram(samples, sr)
    return np.asarray(features, dtype=np.float32), float(hop_sec)


def _render_wav_panel(
    *,
    wav_path: str,
    score_map,
    audio,
    decoded,
    sample_rate: int,
    out_dir: Path,
    model_tag: str,
    max_rows: int,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel, hop_sec = _mel_for_plot(wav_path, sample_rate)
    duration_ms = float(audio.duration_ms)
    times = np.asarray(score_map.times_ms, dtype=np.float32)
    rows = list(decoded.rows)
    if not rows:
        return []
    if len(rows) > max_rows:
        rows = rows[:max_rows]

    out_paths: list[Path] = []
    base = os.path.splitext(os.path.basename(wav_path))[0]

    for row in rows:
        anchors = row.anchors
        params = absolute_anchors_to_oto_params(anchors, duration_ms=duration_ms)
        # Window window: pad ±150ms around offset..cutoff for context.
        pad_ms = 150.0
        win_start = max(0.0, float(anchors.offset_abs) - pad_ms)
        win_end = min(duration_ms, float(anchors.cutoff_abs) + pad_ms)
        win_start_s = win_start / 1000.0
        win_end_s = win_end / 1000.0

        fig, (ax_mel, ax_scores) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6]}
        )
        # Mel spectrogram (in seconds).
        n_frames_mel = mel.shape[0]
        mel_times = np.arange(n_frames_mel) * hop_sec
        # crop to window for clarity
        mask = (mel_times >= win_start_s) & (mel_times <= win_end_s)
        if mask.sum() < 4:
            mask = np.ones_like(mel_times, dtype=bool)
        mel_crop = mel[mask].T
        extent = [
            float(mel_times[mask].min()),
            float(mel_times[mask].max()),
            0.0,
            float(mel_crop.shape[0]),
        ]
        ax_mel.imshow(mel_crop, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax_mel.set_ylabel("mel bin")
        ax_mel.set_title(
            f"{os.path.basename(wav_path)}  ·  alias={row.spec.alias!r}  "
            f"role={row.spec.role}  slot={row.spec.slot_index}/{row.spec.slot_count}  "
            f"model={model_tag}",
            fontsize=10,
        )

        # Score curves (only labels that exist).
        score_dict = score_map.scores or {}
        for label in _PLOT_SCORE_LABELS:
            values = score_dict.get(label)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float32)
            n = min(arr.shape[0], times.shape[0])
            if n <= 0:
                continue
            t_s = times[:n] / 1000.0
            m = (t_s >= win_start_s) & (t_s <= win_end_s)
            if m.sum() < 2:
                continue
            ax_scores.plot(
                t_s[m],
                arr[:n][m],
                color=_SCORE_COLORS.get(label, None),
                label=label,
                linewidth=1.1,
                alpha=0.85,
            )
        ax_scores.set_ylim(-0.02, 1.02)
        ax_scores.set_ylabel("posterior")
        ax_scores.set_xlabel("time (s)")
        ax_scores.legend(loc="upper right", fontsize=7, ncols=2)

        # Active audio region (RMS-based start/end from compute_audio_candidates).
        for ax in (ax_mel, ax_scores):
            ax.axvspan(
                float(audio.active_start_ms) / 1000.0,
                float(audio.active_end_ms) / 1000.0,
                color="#cfd8dc",
                alpha=0.20,
                zorder=0,
            )

        # Vertical lines for OTO window edges as DECODED by the inference pipeline.
        # All values are absolute ms; convert to s.
        anchor_lines = [
            ("offset", anchors.offset_abs, "#1976d2", "-", 2.0),
            ("overlap", anchors.overlap_abs, "#42a5f5", "--", 1.4),
            ("preutterance", anchors.pre_abs, "#388e3c", "--", 1.4),
            ("consonant", anchors.consonant_abs, "#fb8c00", "--", 1.4),
            ("cutoff", anchors.cutoff_abs, "#c62828", "-", 2.0),
        ]
        for name, t_ms, color, ls, lw in anchor_lines:
            t_s = float(t_ms) / 1000.0
            for ax in (ax_mel, ax_scores):
                ax.axvline(t_s, color=color, linestyle=ls, linewidth=lw, alpha=0.95, zorder=3)
            ax_scores.annotate(
                name,
                xy=(t_s, 1.02),
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=7,
                color=color,
                rotation=75,
                ha="left",
                va="bottom",
            )

        # Bottom legend with raw param values + window length.
        win_len = float(anchors.cutoff_abs) - float(anchors.offset_abs)
        consonant_dur = float(anchors.consonant_abs) - float(anchors.pre_abs)
        ax_scores.text(
            0.0,
            -0.32,
            (
                f"params: offset={params['offset']:.1f}  overlap={params['overlap']:.1f}  "
                f"preutt={params['preutterance']:.1f}  consonant={params['consonant']:.1f}  "
                f"cutoff={params['cutoff']:.1f}  ·  "
                f"window={win_len:.1f}ms  consonant_dur={consonant_dur:.1f}ms  "
                f"selected_peak={row.selected_time_ms:.1f}ms  "
                f"fallback={row.fallback_used}  reason={anchors.reason!r}"
            ),
            transform=ax_scores.transAxes,
            fontsize=8,
            family="monospace",
            color="#444444",
        )

        plt.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.18, hspace=0.06)

        safe_alias = "".join(c if c.isalnum() else "_" for c in str(row.spec.alias or "alias"))[:24]
        out_name = (
            f"diag_{model_tag}_{base}_{safe_alias}_li{int(row.spec.line_index)}.png"
        )
        out_path = out_dir / out_name
        fig.savefig(str(out_path), dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def _model_short_tag(model_path: str) -> str:
    name = os.path.splitext(os.path.basename(model_path))[0]
    for prefix in ("oto_boundary_scorer_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name[:40]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav-dir", required=True, help="Directory containing the voicebank wav files.")
    ap.add_argument(
        "--source-oto",
        default="",
        help="Source oto.ini whose alias/filename rows drive inference. If empty, auto-discover with resolve_no_mfa_source_oto.",
    )
    ap.add_argument(
        "--model",
        default="",
        help="Boundary scorer .pt path. Empty = auto-resolve (production default).",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--language", default="korean")
    ap.add_argument("--format-type", default="")
    ap.add_argument("--alias-suffix", default="")
    ap.add_argument(
        "--out-dir",
        default=os.path.join("ml_workspace", "coarse_crnn", "diagnose_windows"),
    )
    ap.add_argument("--max-wavs", type=int, default=5, help="Number of wavs to render.")
    ap.add_argument("--max-rows-per-wav", type=int, default=4)
    ap.add_argument("--role-filter", default="", help='Comma-separated alias roles to include. Empty = all. Example: "cv,vc,v-cv".')
    ap.add_argument("--alias-substring", default="", help="If set, only rows whose alias contains this substring are rendered.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    random.seed(int(args.seed))

    source = resolve_no_mfa_source_oto(
        wav_dir=str(args.wav_dir), source_hint=str(args.source_oto or "")
    )
    if not source:
        raise SystemExit("source oto.ini를 찾지 못했습니다. --source-oto로 명시하세요.")
    model_path = resolve_boundary_scorer_model_path(str(args.model or ""))
    if not model_path:
        raise SystemExit("boundary scorer 모델 경로를 찾지 못했습니다.")
    print(f"[diagnose] source = {source}")
    print(f"[diagnose] model  = {model_path}")

    import torch

    torch_device = resolve_torch_device(torch, str(args.device))
    model, config, _meta = load_boundary_checkpoint(model_path, map_location=str(torch_device))
    model = model.to(torch_device).eval()

    rows_by_wav = load_row_specs_from_source_oto(
        source_oto_path=source,
        wav_dir=str(args.wav_dir),
        language=str(args.language),
        format_type=str(args.format_type),
        alias_suffix=str(args.alias_suffix),
    )
    if not rows_by_wav:
        raise SystemExit("source oto에서 row가 추출되지 않았습니다.")

    role_filter = {item.strip().lower() for item in str(args.role_filter or "").split(",") if item.strip()}
    alias_substr = str(args.alias_substring or "").strip()

    wav_items = list(rows_by_wav.items())
    random.shuffle(wav_items)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = _model_short_tag(model_path)

    rendered = 0
    for wav_path, specs in wav_items:
        if rendered >= int(args.max_wavs):
            break
        kept = []
        for spec in specs:
            if role_filter and str(spec.role or "").lower() not in role_filter:
                continue
            if alias_substr and alias_substr not in str(spec.alias or ""):
                continue
            kept.append(spec)
        if not kept:
            continue
        try:
            score_map, audio, decoded = _run_pipeline_for_wav(
                model=model,
                config=config,
                device=torch_device,
                wav_path=wav_path,
                specs=kept,
            )
        except Exception as exc:
            print(f"[diagnose] {os.path.basename(wav_path)}: failed — {exc}")
            continue
        paths = _render_wav_panel(
            wav_path=wav_path,
            score_map=score_map,
            audio=audio,
            decoded=decoded,
            sample_rate=int(config.sample_rate),
            out_dir=out_dir,
            model_tag=model_tag,
            max_rows=int(args.max_rows_per_wav),
            dpi=int(args.dpi),
        )
        for path in paths:
            print(f"[diagnose] wrote {path}")
        if paths:
            rendered += 1

    print(f"[diagnose] done. rendered_wavs={rendered}  out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
