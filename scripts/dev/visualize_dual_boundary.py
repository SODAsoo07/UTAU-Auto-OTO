"""Render dual_pred.json (BoundaryScorer + PhonemeBoundary) into one PNG that
stacks the equivalent label curves on the same time axis. Optional --lab to
overlay manual HTK lab boundaries as vertical reference lines."""
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

HTK = 1.0 / 10000.0
SIL = {"AP", "SP", "sil", "pau", "br", "vf", "axh", "exh", "cl"}

# Label pairings (PB → BS). Each tuple drives one stacked comparison panel.
PAIRS = [
    ("vowel_onset",      "vowel_start",      "#2ca02c"),
    ("vowel_nucleus",    "vowel_stable",     "#98df8a"),
    ("vowel_end",        "vowel_end",        "#ff7f0e"),
    ("consonant_onset",  "consonant_onset",  "#d62728"),
    ("phone_start",      "syllable_onset",   "#1f77b4"),
    ("silence_boundary", "silence_boundary", "#7f7f7f"),
]


def load_wav(path):
    with closing(wave.open(path, "rb")) as w:
        n = w.getnframes(); sr = w.getframerate(); ch = w.getnchannels(); width = w.getsampwidth()
        raw = w.readframes(n)
    if width == 2: data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
    elif width == 4: data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)/2147483648.0
    else: data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0)/128.0
    if ch > 1: data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def parse_lab(path):
    out = []
    for raw in open(path, encoding="utf-8-sig", errors="replace"):
        p = raw.strip().split()
        if len(p) < 3: continue
        try: s = float(p[0]) * HTK; e = float(p[1]) * HTK
        except: continue
        if e <= s: continue
        ph = p[2]
        is_sil = ph in SIL or ph.lower() in {x.lower() for x in SIL}
        out.append((s, e, ph, is_sil))
    return out


def render(json_path: str, out_png: str, lab_path: str = "") -> None:
    payload = json.load(open(json_path, encoding="utf-8"))
    wav = payload["wav_path"]
    bs = payload["bs"]; pb = payload["pb"]
    bs_t = np.asarray(bs["times_ms"], dtype=np.float32)
    pb_t = np.asarray(pb["times_ms"], dtype=np.float32)
    samples, sr = load_wav(wav)
    wav_t = np.arange(len(samples), dtype=np.float32) * 1000.0 / sr
    duration_ms = float(wav_t[-1]) if len(wav_t) else 0.0

    lab = parse_lab(lab_path) if lab_path and os.path.isfile(lab_path) else []

    n_panels = 1 + len(PAIRS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, 1.2 + 1.0 * n_panels), sharex=True)
    ax0 = axes[0]
    ax0.plot(wav_t, samples, linewidth=0.4, color="#222")
    ax0.set_xlim(0, duration_ms)
    ax0.set_ylabel("waveform", fontsize=8); ax0.tick_params(labelsize=7)
    ax0.set_title(f"{os.path.basename(wav)}    BS={os.path.basename(bs['model'])}    "
                  f"PB={os.path.basename(pb['model'])}", fontsize=9)
    # lab markers on waveform
    for s, e, ph, is_sil in lab:
        col = "#bbbbbb" if is_sil else "#000000"
        ax0.axvline(s, color=col, alpha=0.4 if is_sil else 0.6, linewidth=0.6)
        if not is_sil:
            ax0.text(s, 0.85, ph, fontsize=6, color="#000", ha="left",
                     transform=ax0.get_xaxis_transform())

    for idx, (pb_lbl, bs_lbl, color) in enumerate(PAIRS):
        ax = axes[1 + idx]
        pb_curve = np.asarray(pb["scores"].get(pb_lbl, []), dtype=np.float32)
        bs_curve = np.asarray(bs["scores"].get(bs_lbl, []), dtype=np.float32)
        if len(pb_curve) == len(pb_t):
            ax.fill_between(pb_t, 0, pb_curve, color=color, alpha=0.35, label=f"PB.{pb_lbl}")
            ax.plot(pb_t, pb_curve, color=color, linewidth=0.7)
        if len(bs_curve) == len(bs_t):
            ax.plot(bs_t, bs_curve, color="#111", linewidth=0.8, linestyle="--", label=f"BS.{bs_lbl}")
        ax.set_ylim(0, 1.05); ax.set_ylabel(f"{pb_lbl}\nvs {bs_lbl}", fontsize=7)
        ax.tick_params(labelsize=7); ax.legend(loc="upper right", fontsize=6)
        ax.axhline(0.45, color="#888", linewidth=0.3, linestyle=":")
        # lab markers as light verticals (only non-sil for label panels)
        for s, e, ph, is_sil in lab:
            if pb_lbl == "silence_boundary":
                if is_sil:
                    ax.axvline(s, color="#666", alpha=0.4, linewidth=0.5)
                    ax.axvline(e, color="#666", alpha=0.4, linewidth=0.5)
            elif not is_sil:
                if pb_lbl in {"vowel_onset", "phone_start", "consonant_onset"}:
                    ax.axvline(s, color="#000", alpha=0.35, linewidth=0.5)
                elif pb_lbl == "vowel_end":
                    ax.axvline(e, color="#000", alpha=0.35, linewidth=0.5)

    axes[-1].set_xlabel("time (ms)", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=""); ap.add_argument("--batch", default="")
    ap.add_argument("--lab-dir", default="", help="Optional dir with *.lab matched by stem")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    targets = []
    if args.json: targets.append(os.path.abspath(args.json))
    if args.batch:
        for f in sorted(os.listdir(args.batch)):
            if f.endswith(".dual_pred.json"):
                targets.append(os.path.abspath(os.path.join(args.batch, f)))
    if not targets:
        ap.error("provide --json or --batch")

    rendered = []
    for jp in targets:
        stem = os.path.splitext(os.path.basename(jp))[0]
        if stem.endswith(".dual_pred"):
            base = stem[:-len(".dual_pred")]
        else:
            base = stem
        out_png = os.path.join(os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(jp),
                               base + ".dual.png")
        lab = os.path.join(args.lab_dir, base + ".lab") if args.lab_dir else ""
        try:
            render(jp, out_png, lab_path=lab)
            rendered.append(out_png)
        except Exception as exc:
            print(f"# error {jp}: {exc}")
    print(json.dumps({"rendered": len(rendered), "samples": rendered[:3]}, ensure_ascii=False, indent=2))
    return 0 if rendered else 2


if __name__ == "__main__":
    raise SystemExit(main())
