#!/usr/bin/env python
"""
Build SOFA full-label dataset from gold TextGrid phones tier.

Output layout:
  <out_root>/full_label/<subset_name>/
    wavs/
    transcriptions.csv  # columns: name, ph_seq, ph_dur
"""

import argparse
import csv
import json
import os
import shutil
import wave
from typing import Iterable, List, Sequence, Tuple

from textgrid import TextGrid


def get_wav_duration_sec(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
    if sr <= 0:
        return 0.0
    return float(frames) / float(sr)


def normalize_durations_to_wav_length(
    durations: Sequence[float],
    wav_length_sec: float,
    round_decimals: int = 5,
) -> List[float]:
    vals = [max(0.0, float(x)) for x in durations]
    total = sum(vals)
    if wav_length_sec <= 0:
        raise ValueError("wav_length_sec must be > 0")
    if total <= 0:
        raise ValueError("sum(durations) must be > 0")

    scaled = [x * (wav_length_sec / total) for x in vals]
    rounded = [round(x, round_decimals) for x in scaled]
    diff = round(wav_length_sec - sum(rounded), round_decimals)
    rounded[-1] = round(rounded[-1] + diff, round_decimals)
    if rounded[-1] <= 0:
        raise ValueError("normalized final duration became non-positive")
    return rounded


def find_phones_tier(tg: TextGrid, phones_tier_name: str):
    target = (phones_tier_name or "").strip().lower()
    for tier in tg.tiers:
        if getattr(tier, "name", "").strip().lower() == target:
            return tier
    # fallback: pick the last tier when exact name is not found
    if not tg.tiers:
        return None
    return tg.tiers[-1]


def extract_phone_seq_and_dur(
    tg_path: str,
    phones_tier_name: str,
    ignored_phonemes: Iterable[str],
) -> Tuple[List[str], List[float]]:
    ignored = {str(x).strip() for x in ignored_phonemes if str(x).strip()}
    tg = TextGrid.fromFile(tg_path)
    tier = find_phones_tier(tg, phones_tier_name=phones_tier_name)
    if tier is None:
        return [], []

    phones: List[str] = []
    durations: List[float] = []
    for interval in tier:
        mark = str(getattr(interval, "mark", "")).strip()
        if not mark or mark in ignored:
            continue
        start = float(getattr(interval, "minTime", 0.0))
        end = float(getattr(interval, "maxTime", 0.0))
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        phones.append(mark)
        durations.append(dur)
    return phones, durations


def write_rows_csv_json(rows: List[dict], csv_path: str, json_path: str) -> None:
    rows = list(rows or [])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, indent=2, ensure_ascii=False)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        if not fieldnames:
            cf.write("")
            return
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_full_labels(
    textgrid_dir: str,
    wav_dir: str,
    out_root: str,
    subset_name: str,
    phones_tier_name: str,
    ignored_phonemes: Iterable[str],
    round_decimals: int,
    copy_wavs: bool,
) -> dict:
    subset_dir = os.path.join(out_root, "full_label", subset_name)
    wav_out_dir = os.path.join(subset_dir, "wavs")
    os.makedirs(subset_dir, exist_ok=True)
    if copy_wavs:
        os.makedirs(wav_out_dir, exist_ok=True)

    tg_paths = {}
    for root, _, files in os.walk(textgrid_dir):
        for name in files:
            if not name.lower().endswith(".textgrid"):
                continue
            base = os.path.splitext(name)[0]
            tg_paths[base] = os.path.join(root, name)

    trans_rows = []
    skip_rows = []
    wav_files = sorted(
        [
            os.path.join(wav_dir, name)
            for name in os.listdir(wav_dir)
            if name.lower().endswith(".wav")
        ]
    )

    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        tg_path = tg_paths.get(base)
        if not tg_path:
            skip_rows.append(
                {
                    "name": base,
                    "reason": "missing_textgrid",
                }
            )
            continue

        phones, durations = extract_phone_seq_and_dur(
            tg_path=tg_path,
            phones_tier_name=phones_tier_name,
            ignored_phonemes=ignored_phonemes,
        )
        if not phones or not durations:
            skip_rows.append(
                {
                    "name": base,
                    "reason": "empty_phone_sequence",
                    "textgrid": tg_path,
                }
            )
            continue
        if len(phones) != len(durations):
            skip_rows.append(
                {
                    "name": base,
                    "reason": "phone_duration_length_mismatch",
                    "phones": len(phones),
                    "durations": len(durations),
                }
            )
            continue

        wav_len = get_wav_duration_sec(wav_path)
        try:
            norm_durations = normalize_durations_to_wav_length(
                durations=durations,
                wav_length_sec=wav_len,
                round_decimals=round_decimals,
            )
        except ValueError as e:
            skip_rows.append(
                {
                    "name": base,
                    "reason": "duration_normalization_failed",
                    "error": str(e),
                    "textgrid": tg_path,
                }
            )
            continue

        ph_seq = " ".join(phones)
        ph_dur = " ".join(f"{d:.{round_decimals}f}" for d in norm_durations)
        trans_rows.append(
            {
                "name": base,
                "ph_seq": ph_seq,
                "ph_dur": ph_dur,
            }
        )

        if copy_wavs:
            shutil.copy2(wav_path, os.path.join(wav_out_dir, os.path.basename(wav_path)))

    transcriptions_csv = os.path.join(subset_dir, "transcriptions.csv")
    with open(transcriptions_csv, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["name", "ph_seq", "ph_dur"])
        writer.writeheader()
        for row in trans_rows:
            writer.writerow(row)

    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    skip_csv = os.path.join(reports_dir, f"full_label_{subset_name}_skip_report.csv")
    skip_json = os.path.join(reports_dir, f"full_label_{subset_name}_skip_report.json")
    write_rows_csv_json(skip_rows, skip_csv, skip_json)

    summary = {
        "textgrid_dir": textgrid_dir,
        "wav_dir": wav_dir,
        "out_subset_dir": subset_dir,
        "transcriptions_csv": transcriptions_csv,
        "kept_rows": len(trans_rows),
        "skipped_rows": len(skip_rows),
        "copied_wavs": len(trans_rows) if copy_wavs else 0,
        "skip_report_csv": skip_csv,
        "skip_report_json": skip_json,
    }
    summary_path = os.path.join(reports_dir, f"full_label_{subset_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    summary["summary_json"] = summary_path
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--out-root", required=True, help="Output root folder (e.g. sofa_utau_kr/data)")
    parser.add_argument("--subset-name", default="gold")
    parser.add_argument("--phones-tier", default="phones")
    parser.add_argument("--ignored-phonemes", default="AP,SP,<AP>,<SP>,pau,cl")
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--no-copy-wavs", action="store_true")
    args = parser.parse_args()

    ignored = [x.strip() for x in args.ignored_phonemes.split(",") if x.strip()]
    summary = build_full_labels(
        textgrid_dir=args.textgrid_dir,
        wav_dir=args.wav_dir,
        out_root=args.out_root,
        subset_name=args.subset_name,
        phones_tier_name=args.phones_tier,
        ignored_phonemes=ignored,
        round_decimals=args.round_decimals,
        copy_wavs=not args.no_copy_wavs,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

