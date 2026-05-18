"""Render a phoneme-boundary prediction JSON (from predict_phoneme_boundary_dir.py)
into a stacked PNG: waveform on top, per-label score curves below, detected
events drawn as vertical ticks. Use --batch with a folder of *.boundary_pred.json
to render every wav in one shot."""
from __future__ import annotations

import argparse
import json
import os
import wave
from contextlib import closing

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib required: {exc}")


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with closing(wave.open(path, "rb")) as wf:
        n = wf.getnframes()
        rate = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, int(rate)


LABEL_COLOR = {
    "phone_start": "#1f77b4",
    "phone_end": "#aec7e8",
    "consonant_onset": "#d62728",
    "vowel_onset": "#2ca02c",
    "vowel_nucleus": "#98df8a",
    "vowel_end": "#ff7f0e",
    "silence_boundary": "#7f7f7f",
}


def render(json_path: str, out_png: str) -> None:
    with open(json_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    wav_path = payload["wav_path"]
    times = np.asarray(payload.get("times_ms", []), dtype=np.float32)
    scores: dict = payload.get("scores", {})
    events = payload.get("events", [])
    labels = list(scores.keys())

    samples, sr = load_wav(wav_path)
    wav_t_ms = np.arange(len(samples), dtype=np.float32) * 1000.0 / float(sr)
    duration_ms = float(wav_t_ms[-1]) if len(wav_t_ms) else 0.0

    n_panels = 1 + len(labels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 1.0 + 0.9 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    ax0 = axes[0]
    ax0.plot(wav_t_ms, samples, linewidth=0.4, color="#222")
    ax0.set_ylabel("waveform", fontsize=8)
    ax0.set_title(f"{os.path.basename(wav_path)}    model={os.path.basename(payload.get('model_path',''))}",
                  fontsize=9)
    ax0.set_xlim(0, duration_ms)
    ax0.tick_params(labelsize=7)

    for ev in events:
        color = LABEL_COLOR.get(ev["label"], "#444")
        ax0.axvline(ev["time_ms"], color=color, alpha=0.45, linewidth=0.6)

    for idx, label in enumerate(labels):
        ax = axes[1 + idx]
        curve = np.asarray(scores[label], dtype=np.float32)
        color = LABEL_COLOR.get(label, "#444")
        if len(curve) == len(times):
            ax.fill_between(times, 0.0, curve, color=color, alpha=0.35)
            ax.plot(times, curve, color=color, linewidth=0.8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=7)
        for ev in events:
            if ev["label"] == label:
                ax.axvline(ev["time_ms"], color="#000", alpha=0.55, linewidth=0.7)
                ax.text(ev["time_ms"], 0.92, f"{ev['score']:.2f}", fontsize=6, ha="center", color="#000")
        ax.axhline(payload.get("threshold", 0.45), color="#888", linewidth=0.4, linestyle="--")

    axes[-1].set_xlabel("time (ms)", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="", help="Single prediction JSON.")
    ap.add_argument("--batch", default="", help="Directory containing *.boundary_pred.json.")
    ap.add_argument("--out-dir", default="", help="Output directory for PNGs. Defaults next to JSON.")
    args = ap.parse_args()

    targets: list[str] = []
    if args.json:
        targets.append(os.path.abspath(args.json))
    if args.batch:
        for f in sorted(os.listdir(args.batch)):
            if f.endswith(".boundary_pred.json"):
                targets.append(os.path.abspath(os.path.join(args.batch, f)))
    if not targets:
        ap.error("provide --json or --batch")

    rendered: list[str] = []
    errors: list[dict] = []
    for jp in targets:
        stem = os.path.splitext(os.path.basename(jp))[0]
        if args.out_dir:
            out_png = os.path.join(os.path.abspath(args.out_dir), stem + ".png")
        else:
            out_png = os.path.join(os.path.dirname(jp), stem + ".png")
        try:
            render(jp, out_png)
            rendered.append(out_png)
        except Exception as exc:
            errors.append({"json": jp, "error": str(exc)})
    print(json.dumps({"rendered": len(rendered), "errors": errors, "out_samples": rendered[:5]},
                     ensure_ascii=False, indent=2))
    return 0 if rendered else 2


if __name__ == "__main__":
    raise SystemExit(main())
