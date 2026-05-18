"""Convert HTK-style .lab files (start_100ns end_100ns phone) plus matching .wav
files into a phoneme-boundary manifest JSONL that the trainer/evaluator can
consume directly. Each row carries `phones=[{phone,start_ms,end_ms}, ...]`,
which is the only field the trainer needs for boundary supervision; alias
columns are intentionally omitted because the lab is pronunciation-aligned
and the filename does not match the actual phones."""
from __future__ import annotations

import argparse
import json
import os
import wave
from contextlib import closing

HTK_TICK_MS = 1.0 / 10_000.0  # 1 tick = 100 ns = 1e-4 ms

# Default mapping policy (acoustic, not transcription).
#   - SIL: true silence regions.
#   - BREATH: unvoiced fricative breath noise (still energetic, but no voicing).
#   - DROP: phonetically voiced creak that doesn't fit any boundary class; the
#     interval is omitted entirely so the model is not pulled toward labeling
#     these frames as "silence" the way the original convention did.
DEFAULT_SIL_PHONES = ("AP", "SP", "sil", "pau", "cl")
DEFAULT_BREATH_PHONES = ("br", "exh", "axh")
DEFAULT_DROP_PHONES = ("vf",)


def parse_htk_lab(
    path: str,
    *,
    sil_phones: set[str],
    breath_phones: set[str],
    drop_phones: set[str],
    wav_duration_ms: float = 0.0,
) -> list[dict]:
    """Parse HTK lab. If wav_duration_ms > 0, clamp end_ms to the wav length and
    drop intervals that start past the wav end (many labelers emit a sentinel
    final SP extending to ~30s)."""
    sil_low = {p.lower() for p in sil_phones}
    breath_low = {p.lower() for p in breath_phones}
    drop_low = {p.lower() for p in drop_phones}
    out: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for raw in fh:
            parts = raw.strip().split()
            if len(parts) < 3:
                continue
            try:
                start_ms = float(parts[0]) * HTK_TICK_MS
                end_ms = float(parts[1]) * HTK_TICK_MS
            except ValueError:
                continue
            if end_ms <= start_ms:
                continue
            if wav_duration_ms > 0.0:
                if start_ms >= wav_duration_ms:
                    continue
                if end_ms > wav_duration_ms:
                    end_ms = wav_duration_ms
            phone_raw = parts[2].strip()
            lower = phone_raw.lower()
            if phone_raw in drop_phones or lower in drop_low:
                continue
            if phone_raw in sil_phones or lower in sil_low:
                phone = "sil"
            elif phone_raw in breath_phones or lower in breath_low:
                phone = "br"
            else:
                phone = phone_raw
            out.append({"phone": phone, "start_ms": start_ms, "end_ms": end_ms})
    return out


def wav_duration_ms(path: str) -> float:
    try:
        with closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return float(frames) * 1000.0 / float(rate)
    except Exception:
        return 0.0


def iter_pairs(lab_dir: str) -> list[tuple[str, str]]:
    pairs = []
    for fname in sorted(os.listdir(lab_dir)):
        if not fname.lower().endswith(".lab"):
            continue
        stem = os.path.splitext(fname)[0]
        wav = os.path.join(lab_dir, stem + ".wav")
        if not os.path.isfile(wav):
            continue
        lab = os.path.join(lab_dir, fname)
        pairs.append((wav, lab))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab-dir", action="append", required=True,
                    help="Directory containing matched <name>.wav + <name>.lab pairs. Repeatable.")
    ap.add_argument("--out", required=True, help="Output manifest JSONL path.")
    ap.add_argument("--language", default="", help="Language tag stored on each row (e.g. 'japanese').")
    ap.add_argument("--sil-phones", default=",".join(DEFAULT_SIL_PHONES),
                    help="Comma-separated phone tokens to remap to 'sil' (true silence).")
    ap.add_argument("--breath-phones", default=",".join(DEFAULT_BREATH_PHONES),
                    help="Comma-separated phone tokens to remap to 'br' (unvoiced breath/fricative noise).")
    ap.add_argument("--drop-phones", default=",".join(DEFAULT_DROP_PHONES),
                    help="Comma-separated phone tokens whose intervals are omitted entirely (e.g. vocal creak 'vf').")
    ap.add_argument("--speaker", default="", help="Optional speaker tag per row (kept in sidecar_meta).")
    args = ap.parse_args()

    sil_phones = {p.strip() for p in str(args.sil_phones).split(",") if p.strip()}
    breath_phones = {p.strip() for p in str(args.breath_phones).split(",") if p.strip()}
    drop_phones = {p.strip() for p in str(args.drop_phones).split(",") if p.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = 0
    skipped = 0
    per_dir = []
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for lab_dir in args.lab_dir:
            lab_dir_abs = os.path.abspath(lab_dir)
            pairs = iter_pairs(lab_dir_abs)
            kept = 0
            for wav, lab in pairs:
                dur_ms = wav_duration_ms(wav)
                phones = parse_htk_lab(
                    lab,
                    sil_phones=sil_phones,
                    breath_phones=breath_phones,
                    drop_phones=drop_phones,
                    wav_duration_ms=dur_ms,
                )
                if not phones:
                    skipped += 1
                    continue
                row = {
                    "wav_path": wav,
                    "wav_name": os.path.basename(wav),
                    "phones": phones,
                    "duration_ms": dur_ms,
                    "language": str(args.language),
                    "source": "htk_lab",
                    "lab_path": lab,
                }
                if args.speaker:
                    row["speaker"] = str(args.speaker)
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                rows += 1
            per_dir.append({"dir": lab_dir_abs, "pairs": len(pairs), "kept": kept})

    summary = {
        "out": os.path.abspath(args.out),
        "written_rows": rows,
        "skipped_empty_lab": skipped,
        "per_dir": per_dir,
        "sil_phones": sorted(sil_phones),
        "breath_phones": sorted(breath_phones),
        "drop_phones": sorted(drop_phones),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
