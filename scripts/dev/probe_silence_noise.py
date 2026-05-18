"""For each wav, compute RMS in regions labeled as silence by the HTK lab and
contrast it with non-silence regions. Then, for every model-predicted
silence_boundary, sample the local RMS to see whether the model is pointing at
a real acoustic transition that the coarse SP/exh label is hiding."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import wave
from contextlib import closing

import numpy as np

HTK = 1.0 / 10_000.0
SIL = {"AP", "SP", "sil", "pau", "br", "vf", "axh", "exh", "cl"}
SIL_LOW = {x.lower() for x in SIL}


def parse_lab(path):
    out = []
    for raw in open(path, encoding="utf-8-sig", errors="replace"):
        p = raw.strip().split()
        if len(p) < 3: continue
        try: s = float(p[0]) * HTK; e = float(p[1]) * HTK
        except: continue
        if e <= s: continue
        ph = p[2]
        norm = "sil" if (ph in SIL or ph.lower() in SIL_LOW) else ph
        out.append((s, e, ph, norm))
    return out


def load_wav(path):
    with closing(wave.open(path, "rb")) as w:
        n = w.getnframes(); sr = w.getframerate()
        ch = w.getnchannels(); width = w.getsampwidth()
        raw = w.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def rms_db(samples: np.ndarray) -> float:
    if samples.size == 0: return -120.0
    val = float(np.sqrt(np.mean(samples * samples)) + 1e-12)
    return 20.0 * math.log10(val)


def rms_at(samples, sr, t_ms, window_ms):
    half = int(round(sr * (window_ms / 1000.0) / 2))
    center = int(round(sr * t_ms / 1000.0))
    lo = max(0, center - half); hi = min(len(samples), center + half)
    return rms_db(samples[lo:hi])


def analyze(wav, lab, pred_json, *, win_ms=40.0):
    samples, sr = load_wav(wav)
    rows = parse_lab(lab)
    sil_segs = [(s, e, raw) for (s, e, raw, n) in rows if n == "sil" and (e - s) >= 30.0]
    voiced_segs = [(s, e, raw) for (s, e, raw, n) in rows if n != "sil"]
    payload = json.load(open(pred_json, encoding="utf-8"))
    events = payload.get("events") or []
    pred_sil = [(float(e["time_ms"]), float(e["score"])) for e in events
                if e["label"] == "silence_boundary"]

    # whole-segment RMS distribution
    sil_rms = []
    for s, e, raw in sil_segs:
        lo = int(round(sr * s / 1000.0)); hi = int(round(sr * e / 1000.0))
        if hi > lo:
            sil_rms.append((raw, rms_db(samples[lo:hi])))
    voi_rms = []
    for s, e, raw in voiced_segs:
        lo = int(round(sr * s / 1000.0)); hi = int(round(sr * e / 1000.0))
        if hi > lo:
            voi_rms.append((raw, rms_db(samples[lo:hi])))

    # for each predicted silence_boundary, what's the RMS at that point?
    pred_local = []
    for t, sc in pred_sil:
        # is this point inside a labeled sil interval?
        inside_raw = None
        for s, e, raw in sil_segs:
            if s <= t <= e:
                inside_raw = raw; break
        # nearest gold sil-boundary distance
        gold_pts = []
        for s, e, raw in sil_segs:
            gold_pts.append((s, raw, "start"))
            gold_pts.append((e, raw, "end"))
        nearest = min(gold_pts, key=lambda p: abs(p[0] - t)) if gold_pts else None
        nearest_err = abs(nearest[0] - t) if nearest else float("inf")
        pred_local.append({
            "t_ms": round(t, 1),
            "score": round(sc, 3),
            "rms_db_at_pred": round(rms_at(samples, sr, t, win_ms), 1),
            "inside_lab_sil": inside_raw,
            "nearest_gold_err_ms": round(nearest_err, 1),
        })

    return {
        "wav": os.path.basename(wav),
        "duration_ms": round(len(samples) * 1000.0 / sr, 1),
        "sil_segments": [{"phone": r, "dur_ms": round(e - s, 1),
                          "rms_db_segment": round(v, 1)}
                         for ((s, e, r), (_, v)) in zip(sil_segs, sil_rms)],
        "voiced_segments_count": len(voi_rms),
        "voiced_rms_db_median": round(float(np.median([v for _, v in voi_rms])), 1) if voi_rms else None,
        "voiced_rms_db_min": round(float(np.min([v for _, v in voi_rms])), 1) if voi_rms else None,
        "predicted_silence_boundaries": pred_local,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--viz-dir", required=True)
    ap.add_argument("--lab-dir", required=True)
    ap.add_argument("--wav-dir", required=True, help="Where the actual wav files live.")
    ap.add_argument("--wavs", nargs="+", required=True, help="wav basenames (stems w/o .wav) to probe.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    reports = []
    for stem in args.wavs:
        wav = os.path.join(args.wav_dir, stem + ".wav")
        lab = os.path.join(args.lab_dir, stem + ".lab")
        jf = os.path.join(args.viz_dir, stem + ".boundary_pred.json")
        for p in (wav, lab, jf):
            if not os.path.isfile(p):
                print(f"# missing: {p}"); break
        else:
            reports.append(analyze(wav, lab, jf))
    text = json.dumps(reports, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
