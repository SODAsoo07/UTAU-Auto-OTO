from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.sequence_aligner import (
    _expand_activity_with_consonant_cues,
    _framewise_consonant_cues,
    _framewise_rms,
    _read_pcm_as_float_mono,
    _smooth_activity_mask,
    _resolve_silence_threshold,
)

PAUSE_LABELS = {"", "sp", "spn", "sil", "pau", "ap", "br", "bre", "breath", "r"}


@dataclass
class Sample:
    base: str
    wav_path: str
    clean_lab_path: str
    duration_sec: float
    target_start_sec: float
    target_end_sec: float
    token_count: int
    rms: np.ndarray
    zcr: np.ndarray
    hf_ratio: np.ndarray
    flux: np.ndarray
    hop_sec: float


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = str(raw or "").strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _is_pause(label: str) -> bool:
    return str(label or "").strip().lower() in PAUSE_LABELS


def _load_target_span(clean_lab_path: str) -> Tuple[float, float]:
    rows: List[Tuple[int, int, str]] = []
    with open(clean_lab_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            s = str(raw or "").strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                st = int(float(parts[0]))
                ed = int(float(parts[1]))
            except Exception:
                continue
            rows.append((st, ed, " ".join(parts[2:])))
    non_pause = [(st, ed, lb) for st, ed, lb in rows if not _is_pause(lb)]
    if not non_pause:
        return 0.0, 0.0
    start = float(non_pause[0][0]) / 10_000_000.0
    end = float(non_pause[-1][1]) / 10_000_000.0
    return float(max(0.0, start)), float(max(start, end))


def _load_samples(manifest_path: str, *, frame_ms: int, hop_ms: int) -> List[Sample]:
    rows = _read_jsonl(manifest_path)
    out: List[Sample] = []
    for row in rows:
        wav_path = os.path.abspath(str(row.get("wav_path", "") or ""))
        lab_path = os.path.abspath(str(row.get("clean_lab_path", "") or ""))
        base = str(row.get("base", "") or os.path.splitext(os.path.basename(wav_path))[0])
        if (not os.path.isfile(wav_path)) or (not os.path.isfile(lab_path)):
            continue
        target_start, target_end = _load_target_span(lab_path)
        signal, sr, duration_sec = _read_pcm_as_float_mono(wav_path)
        if signal.size <= 0 or duration_sec <= 0.0:
            continue
        rms, hop_sec = _framewise_rms(signal, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        if rms.size <= 0 or hop_sec <= 0.0:
            continue
        zcr, hf_ratio, flux = _framewise_consonant_cues(
            signal,
            sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )
        if not (
            zcr.size == rms.size
            and hf_ratio.size == rms.size
            and flux.size == rms.size
        ):
            continue
        token_count = int(row.get("expected_words", 0) or 0)
        out.append(
            Sample(
                base=base,
                wav_path=wav_path,
                clean_lab_path=lab_path,
                duration_sec=float(duration_sec),
                target_start_sec=float(target_start),
                target_end_sec=float(target_end),
                token_count=max(0, token_count),
                rms=rms,
                zcr=zcr,
                hf_ratio=hf_ratio,
                flux=flux,
                hop_sec=float(hop_sec),
            )
        )
    return out


def _predict_active_span(sample: Sample, params: Dict[str, float]) -> Tuple[float, float]:
    local_min = float(params["active_ratio_min"])
    local_max = float(params["active_ratio_max"])
    token_count = int(sample.token_count)
    if token_count > 0 and token_count <= 2:
        local_min = max(local_min, float(params["mono_active_ratio_min"]))
        local_max = max(local_max, float(params["mono_active_ratio_max"]))
    elif token_count >= 6:
        local_max = min(local_max, float(params["multi_active_ratio_max"]))

    threshold = _resolve_silence_threshold(
        sample.rms,
        base_quantile=float(params["silence_q"]),
        scale=float(params["silence_scale"]),
        min_active_ratio=local_min,
        max_active_ratio=local_max,
    )
    min_active_frames = max(
        1,
        int(round((float(params["min_active_ms"]) / 1000.0) / max(sample.hop_sec, 1e-6))),
    )
    max_gap_frames = max(
        0,
        int(round((float(params["max_gap_ms"]) / 1000.0) / max(sample.hop_sec, 1e-6))),
    )
    mask = _smooth_activity_mask(
        sample.rms,
        threshold=threshold,
        min_active_frames=min_active_frames,
        max_gap_frames=max_gap_frames,
    )
    lookback_frames = max(
        1,
        int(round((float(params["onset_lookback_ms"]) / 1000.0) / max(sample.hop_sec, 1e-6))),
    )
    mask = _expand_activity_with_consonant_cues(
        mask,
        rms=sample.rms,
        zcr=sample.zcr,
        hf_ratio=sample.hf_ratio,
        flux=sample.flux,
        silence_threshold=float(threshold),
        lookback_frames=lookback_frames,
    )
    idx = np.where(mask)[0]
    if idx.size <= 0:
        return 0.0, float(sample.duration_sec)
    start = float(idx[0]) * float(sample.hop_sec)
    end = float(idx[-1] + 1) * float(sample.hop_sec)
    start = max(0.0, min(start, sample.duration_sec))
    end = max(start, min(end, sample.duration_sec))
    return start, end


def _evaluate(samples: Sequence[Sample], params: Dict[str, float]) -> Dict[str, float]:
    if not samples:
        return {
            "count": 0.0,
            "mean_start_mae_ms": 0.0,
            "mean_end_mae_ms": 0.0,
            "mean_total_mae_ms": 0.0,
            "p90_total_mae_ms": 0.0,
            "score": 0.0,
        }
    start_errs: List[float] = []
    end_errs: List[float] = []
    total_errs: List[float] = []
    for sample in samples:
        pred_start, pred_end = _predict_active_span(sample, params)
        start_mae = abs(pred_start - sample.target_start_sec) * 1000.0
        end_mae = abs(pred_end - sample.target_end_sec) * 1000.0
        total = (start_mae * 0.65) + (end_mae * 0.35)
        start_errs.append(float(start_mae))
        end_errs.append(float(end_mae))
        total_errs.append(float(total))
    p90_idx = max(0, min(len(total_errs) - 1, int(round((len(total_errs) - 1) * 0.9))))
    p90 = sorted(total_errs)[p90_idx]
    score = float(statistics.mean(total_errs))
    return {
        "count": float(len(samples)),
        "mean_start_mae_ms": float(statistics.mean(start_errs)),
        "mean_end_mae_ms": float(statistics.mean(end_errs)),
        "mean_total_mae_ms": float(statistics.mean(total_errs)),
        "p90_total_mae_ms": float(p90),
        "score": score,
    }


def _grid_candidates() -> Iterable[Dict[str, float]]:
    silence_q = [0.12, 0.16, 0.20, 0.24]
    silence_scale = [0.80, 0.90, 1.00]
    active_ratio_min = [0.14, 0.20]
    active_ratio_max = [0.86, 0.92, 0.96]
    min_active_ms = [16.0, 24.0, 32.0]
    max_gap_ms = [16.0, 28.0, 40.0]
    onset_lookback_ms = [24.0, 36.0, 42.0, 54.0, 66.0]
    mono_active_ratio_min = [0.55]
    mono_active_ratio_max = [0.95]
    multi_active_ratio_max = [0.88]

    keys = [
        "silence_q",
        "silence_scale",
        "active_ratio_min",
        "active_ratio_max",
        "min_active_ms",
        "max_gap_ms",
        "onset_lookback_ms",
        "mono_active_ratio_min",
        "mono_active_ratio_max",
        "multi_active_ratio_max",
    ]
    values = [
        silence_q,
        silence_scale,
        active_ratio_min,
        active_ratio_max,
        min_active_ms,
        max_gap_ms,
        onset_lookback_ms,
        mono_active_ratio_min,
        mono_active_ratio_max,
        multi_active_ratio_max,
    ]
    for combo in itertools.product(*values):
        item = {k: float(v) for k, v in zip(keys, combo)}
        if item["active_ratio_min"] >= item["active_ratio_max"]:
            continue
        yield item


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_profile_env_ps1(path: str, params: Dict[str, float]) -> None:
    lines = [
        f"$env:UTOA_SEQUENCE_ALIGN_SILENCE_Q = \"{params['silence_q']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_SILENCE_SCALE = \"{params['silence_scale']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MIN = \"{params['active_ratio_min']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_ACTIVE_RATIO_MAX = \"{params['active_ratio_max']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_MIN_ACTIVE_MS = \"{params['min_active_ms']:.1f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_MAX_GAP_MS = \"{params['max_gap_ms']:.1f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_ONSET_LOOKBACK_MS = \"{params['onset_lookback_ms']:.1f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MIN = \"{params['mono_active_ratio_min']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_MONO_ACTIVE_RATIO_MAX = \"{params['mono_active_ratio_max']:.4f}\"",
        f"$env:UTOA_SEQUENCE_ALIGN_MULTI_ACTIVE_RATIO_MAX = \"{params['multi_active_ratio_max']:.4f}\"",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_report_md(path: str, payload: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Sequence Aligner Tuning Report")
    lines.append("")
    lines.append(f"- train_manifest: `{payload.get('train_manifest', '')}`")
    lines.append(f"- val_manifest: `{payload.get('val_manifest', '')}`")
    lines.append(f"- frame_ms: `{payload.get('frame_ms', 0)}`")
    lines.append(f"- hop_ms: `{payload.get('hop_ms', 0)}`")
    lines.append("")
    lines.append("## Best Params")
    lines.append("")
    best = payload.get("best_params", {}) or {}
    for key in sorted(best.keys()):
        lines.append(f"- {key}: `{best[key]}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for split in ("train_metrics", "val_metrics", "all_metrics"):
        m = payload.get(split, {}) or {}
        lines.append(f"### {split}")
        lines.append(f"- mean_start_mae_ms: `{float(m.get('mean_start_mae_ms', 0.0)):.2f}`")
        lines.append(f"- mean_end_mae_ms: `{float(m.get('mean_end_mae_ms', 0.0)):.2f}`")
        lines.append(f"- mean_total_mae_ms: `{float(m.get('mean_total_mae_ms', 0.0)):.2f}`")
        lines.append(f"- p90_total_mae_ms: `{float(m.get('p90_total_mae_ms', 0.0)):.2f}`")
        lines.append("")
    lines.append("## Top Candidates")
    lines.append("")
    lines.append("| rank | score | silence_q | silence_scale | ar_min | ar_max | min_active_ms | max_gap_ms | onset_lookback_ms |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(list(payload.get("top_candidates", []) or []), start=1):
        lines.append(
            "| {rank} | {score:.2f} | {q:.2f} | {scale:.2f} | {mn:.2f} | {mx:.2f} | {ma:.1f} | {mg:.1f} | {ol:.1f} |".format(
                rank=idx,
                score=float(row.get("score", 0.0) or 0.0),
                q=float(row.get("silence_q", 0.0) or 0.0),
                scale=float(row.get("silence_scale", 0.0) or 0.0),
                mn=float(row.get("active_ratio_min", 0.0) or 0.0),
                mx=float(row.get("active_ratio_max", 0.0) or 0.0),
                ma=float(row.get("min_active_ms", 0.0) or 0.0),
                mg=float(row.get("max_gap_ms", 0.0) or 0.0),
                ol=float(row.get("onset_lookback_ms", 0.0) or 0.0),
            )
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune sequence aligner parameters from preprocessed Sinsy labels.")
    parser.add_argument("--train-manifest", required=True, help="Train manifest jsonl")
    parser.add_argument("--val-manifest", required=True, help="Validation manifest jsonl")
    parser.add_argument("--frame-ms", type=int, default=22, help="Frame size (ms)")
    parser.add_argument("--hop-ms", type=int, default=5, help="Hop size (ms)")
    parser.add_argument("--language", default="japanese", choices=["japanese", "korean"], help="Language hint")
    parser.add_argument("--out-root", default="", help="Output root folder")
    args = parser.parse_args()

    train_manifest = os.path.abspath(str(args.train_manifest or "").strip())
    val_manifest = os.path.abspath(str(args.val_manifest or "").strip())
    if not os.path.isfile(train_manifest):
        raise SystemExit(f"train manifest not found: {train_manifest}")
    if not os.path.isfile(val_manifest):
        raise SystemExit(f"val manifest not found: {val_manifest}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_root or "").strip():
        out_root = os.path.abspath(str(args.out_root))
    else:
        out_root = os.path.abspath(
            os.path.join(
                os.getcwd(),
                "models",
                "sequence_aligner_tuning",
                f"{args.language}_{stamp}",
            )
        )
    os.makedirs(out_root, exist_ok=True)

    train_samples = _load_samples(train_manifest, frame_ms=int(args.frame_ms), hop_ms=int(args.hop_ms))
    val_samples = _load_samples(val_manifest, frame_ms=int(args.frame_ms), hop_ms=int(args.hop_ms))
    all_samples = list(train_samples) + list(val_samples)

    if not train_samples:
        raise SystemExit("train samples are empty after loading.")

    top_candidates: List[Dict[str, float]] = []
    best_params: Dict[str, float] | None = None
    best_score = float("inf")
    candidate_count = 0

    for params in _grid_candidates():
        candidate_count += 1
        train_metrics = _evaluate(train_samples, params)
        score = float(train_metrics["score"])
        row = dict(params)
        row["score"] = score
        top_candidates.append(row)
        if score < best_score:
            best_score = score
            best_params = dict(params)

    top_candidates = sorted(top_candidates, key=lambda x: float(x.get("score", 0.0)))[:20]
    if not best_params:
        raise SystemExit("failed to find best params")

    train_metrics = _evaluate(train_samples, best_params)
    val_metrics = _evaluate(val_samples, best_params) if val_samples else _evaluate([], best_params)
    all_metrics = _evaluate(all_samples, best_params)

    profile_payload: Dict[str, object] = {
        "version": 1,
        "profile_name": f"sequence_sinsy_tuned_{args.language}_{stamp}",
        "language": args.language,
        "frame_ms": int(args.frame_ms),
        "hop_ms": int(args.hop_ms),
        "params": dict(best_params),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
        "source": {
            "train_manifest": train_manifest,
            "val_manifest": val_manifest,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "grid_candidates": candidate_count,
        },
    }

    result_payload: Dict[str, object] = {
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "language": args.language,
        "frame_ms": int(args.frame_ms),
        "hop_ms": int(args.hop_ms),
        "best_params": dict(best_params),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
        "top_candidates": top_candidates,
        "grid_candidates": int(candidate_count),
        "out_root": out_root,
    }

    profile_json = os.path.join(out_root, f"sequence_aligner_profile_{args.language}.json")
    report_json = os.path.join(out_root, "tuning_report.json")
    report_md = os.path.join(out_root, "tuning_report.md")
    env_ps1 = os.path.join(out_root, "apply_env.ps1")

    _write_json(profile_json, profile_payload)
    _write_json(report_json, result_payload)
    _write_report_md(report_md, result_payload)
    _write_profile_env_ps1(env_ps1, best_params)

    print(f"[TUNE] train_samples={len(train_samples)} val_samples={len(val_samples)}")
    print(f"[TUNE] grid_candidates={candidate_count}")
    print(f"[TUNE] best_score={float(train_metrics.get('mean_total_mae_ms', 0.0)):.2f} ms")
    print(f"[TUNE] profile_json={profile_json}")
    print(f"[TUNE] report_md={report_md}")
    print(f"[TUNE] env_ps1={env_ps1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
