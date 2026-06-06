"""Build a training manifest from voicebank OTO files using acoustic snapping.

The repo's lab gold set is tiny/incomplete, so this derives *pseudo* frame/event
labels straight from established voicebank ``oto.ini`` files (offset/preutterance
anchors), snapped to nearby acoustic landmarks. Only high-confidence regions are
labelled; everything else is left to the IGNORE class so the noisy parts of OTO
timing never become hard targets.

Output rows use the train_poc manual-gold schema (so ``train_poc`` consumes them
unchanged) but carry ``label_origin='oto_pseudo_snap'`` for provenance.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from typing import Iterable

import numpy as np

warnings.filterwarnings("ignore")

from core.oto_file_utils import read_text_with_fallback, parse_oto_line
from core.mfa_free_oto.oto_adapter import _alias_type_for_row
from core.mfa_free_oto.runtime_inference import predict_wav

VOWELS = set("aiueo") | {"N", "n"}


def _track(post, key):
    v = post.acoustic_scores.get(key) or post.event_scores.get(key)
    return np.asarray(v, dtype=float) if v is not None else None


def _snap(times, score, t_ms, radius_ms, *, polarity="max"):
    """Snap a time to the nearest strong landmark in ``score`` within radius."""
    if score is None or score.size != times.size:
        return float(t_ms)
    lo = float(t_ms) - radius_ms
    hi = float(t_ms) + radius_ms
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return float(t_ms)
    idx = np.flatnonzero(mask)
    seg = score[idx]
    pick = idx[int(np.argmax(seg) if polarity == "max" else np.argmin(seg))]
    return float(times[pick])


def _vowel_of(alias: str) -> str:
    toks = [tok for tok in str(alias or "").replace("-", " ").split() if tok]
    for tok in reversed(toks):
        t = tok.strip().lower()
        if t and t[-1] in VOWELS:
            return t[-1]
    return ""


def _contiguous_regions(mask, times, min_ms, *, trim_ms=0.0):
    """Yield (start_ms, end_ms) for contiguous True runs at least ``min_ms`` long."""
    out = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        start = float(times[i])
        end = float(times[j - 1])
        if end - start >= min_ms:
            out.append((start + trim_ms, end - trim_ms))
        i = j
    return [(a, b) for a, b in out if b - a >= max(20.0, min_ms - 2 * trim_ms)]


def build_rows_for_wav(wav_path, oto_rows, *, language="japanese", bank="", trusted=False):
    try:
        post = predict_wav(wav_path, language=language).posterior
    except Exception as exc:  # pragma: no cover - audio decode failures
        return None, f"predict_failed:{exc}"
    t = np.asarray(post.times_ms, dtype=float)
    if t.size < 8:
        return None, "too_short"
    rms = _track(post, "rms")
    if rms is None or rms.size != t.size:
        return None, "no_rms"
    rms_n = rms / max(1e-9, float(np.max(rms)))
    voicing = _track(post, "voicing")
    if voicing is None or voicing.size != t.size:
        voicing = np.zeros_like(t)
    vbl = _track(post, "vowel_boundary_likelihood")
    nucleus = _track(post, "nucleus_likelihood")
    flux = _track(post, "onset_strength")
    if flux is None or flux.size != t.size:
        flux = _track(post, "spectral_flux")
    dur = float(t[-1]) + 10.0

    frame_labels: list[dict] = []
    events: list[dict] = []

    # --- Dense acoustic frame labels (the reliable part of the supervision) ---
    # Confident silence: quiet AND unvoiced. Trim edges to avoid boundary frames.
    sil_mask = (rms_n < 0.06) & (voicing < 0.25)
    for a, b in _contiguous_regions(sil_mask, t, 90.0, trim_ms=15.0):
        frame_labels.append({"label": "silence", "phone": "sil", "start_ms": a, "end_ms": b, "confidence": 0.85})
    # Confident vowel: loud AND voiced. Trim edges so transitions stay IGNORE.
    vow_mask = (rms_n > 0.5) & (voicing > 0.55)
    for a, b in _contiguous_regions(vow_mask, t, 70.0, trim_ms=25.0):
        frame_labels.append({"label": "vowel", "phone": "v", "start_ms": a, "end_ms": b, "confidence": 0.75})

    # --- OTO-anchored events (the precise-localization signal) ---
    for r in sorted(oto_rows, key=lambda r: r["offset"]):
        alias = str(r["alias"])
        role = _alias_type_for_row(alias, "auto")
        anchor = float(r["offset"]) + float(r["pre"])
        if role in {"cv", "cv_head", "vcv", "vv", "v"}:
            # Vowel onset. For a hand-corrected (trusted) OTO the anchor IS the
            # boundary; only pseudo OTO needs snapping to an acoustic landmark.
            on = anchor if trusted else _snap(t, vbl, anchor, 45.0, polarity="max")
            i = int(np.argmin(np.abs(t - on)))
            if not trusted and voicing[i] < 0.35 and rms_n[i] < 0.2:
                continue  # snapped into silence -> distrust, skip
            events.append({"label": "cv_boundary", "phone": _vowel_of(alias) or "a", "time_ms": on, "confidence": 0.8})
            # nucleus = energy/nucleus peak shortly after onset
            win = (t >= on + 20.0) & (t <= on + 240.0)
            if np.any(win):
                src = nucleus if (nucleus is not None and nucleus.size == t.size) else rms_n
                j = np.flatnonzero(win)[int(np.argmax(src[win]))]
                events.append({"label": "vowel_nucleus", "phone": _vowel_of(alias) or "a", "time_ms": float(t[j]), "confidence": 0.7})
        elif role == "vc":
            # Consonant onset.
            if trusted:
                on = anchor
            else:
                on = _snap(t, flux, anchor, 50.0, polarity="max") if flux is not None else anchor
            events.append({"label": "phone_change", "phone": "c", "time_ms": on, "confidence": 0.6})

    if not frame_labels or not events:
        return None, "no_confident_labels"
    for slot_index, event in enumerate(events):
        event["slot_index"] = slot_index
    row = {
        "wav_path": os.path.abspath(wav_path),
        "wav_name": os.path.basename(wav_path),
        "bank": bank,
        "language": language,
        "label_source": "manual_gold",       # consumed by train_poc unchanged
        "label_origin": "gold_oto" if trusted else "oto_pseudo_snap",
        "duration_ms": dur,
        "frame_labels": frame_labels,
        "events": events,
        "slot_count": len(events),
    }
    return row, "ok"


def iter_oto_files(roots: Iterable[str]):
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "oto.ini"), recursive=True):
            yield path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="Voicebank root(s) to scan for oto.ini.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="japanese")
    ap.add_argument("--max-wavs-per-oto", type=int, default=8)
    ap.add_argument("--min-syllables", type=int, default=3, help="Prefer multi-syllable sequence wavs.")
    ap.add_argument("--trusted", action="store_true",
                    help="Treat the OTO as hand-corrected gold: use anchors directly, no acoustic snapping.")
    args = ap.parse_args()

    out_rows = []
    stats = {"oto": 0, "wavs": 0, "rows": 0, "skipped": 0}
    for oto_path in iter_oto_files(args.root):
        stats["oto"] += 1
        oto_dir = os.path.dirname(oto_path)
        bank = os.path.basename(oto_dir.rstrip("\\/")) or os.path.basename(os.path.dirname(oto_dir))
        # Top-level voicebank = first path segment under a cv/cvvc/vcv format root
        # (so val splits never leak the same singer across pitch sub-folders).
        parts = os.path.normpath(oto_path).split(os.sep)
        voicebank = bank
        for i, seg in enumerate(parts):
            if seg.lower() in {"cv", "cvvc", "vcv"} and i + 1 < len(parts):
                voicebank = parts[i + 1]
                break
        text = read_text_with_fallback(oto_path)
        bywav: dict[str, list] = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            r = parse_oto_line(line)
            if r:
                bywav.setdefault(r["wav"], []).append(r)
        # prefer multi-syllable wavs, then cap
        names = sorted(bywav, key=lambda w: (-(w.count("-") + w.count("_")), w))
        picked = names[: args.max_wavs_per_oto]
        for wn in picked:
            wpath = os.path.join(oto_dir, wn)
            if not os.path.isfile(wpath):
                stats["skipped"] += 1
                continue
            stats["wavs"] += 1
            row, why = build_rows_for_wav(wpath, bywav[wn], language=args.language, bank=bank, trusted=bool(args.trusted))
            if row is not None:
                row["voicebank"] = voicebank
            if row is None:
                stats["skipped"] += 1
                continue
            out_rows.append(row)
            stats["rows"] += 1
        print(f"[{stats['oto']}] {oto_path} -> rows={stats['rows']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("STATS", json.dumps(stats))
    print("WROTE", args.out, "rows", len(out_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
