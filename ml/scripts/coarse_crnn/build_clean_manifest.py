from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from core.coarse_crnn.data_discovery import read_manifest, write_manifest
from core.coarse_crnn.lab_io import label_compatibility_ratio, load_aligned_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter and reweight coarse_crnn manifest rows for practical training.")
    parser.add_argument("--manifest", default=os.path.join("ml_workspace", "coarse_crnn", "manifest_full.jsonl"))
    parser.add_argument("--out", default=os.path.join("ml_workspace", "coarse_crnn", "manifest_clean.jsonl"))
    parser.add_argument("--summary", default=os.path.join("ml_workspace", "coarse_crnn", "manifest_clean_summary.json"))
    parser.add_argument("--min-compatible", type=float, default=0.70)
    parser.add_argument("--min-phone-segments", type=int, default=2)
    parser.add_argument("--max-dataset-duration", type=float, default=20.0)
    parser.add_argument("--max-public-duration", type=float, default=75.0)
    parser.add_argument("--max-edge-silence", type=float, default=8.0)
    parser.add_argument("--public-weight", type=float, default=0.35)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    kept: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        ok, reason, cleaned = _clean_row(
            row,
            min_compatible=float(args.min_compatible),
            min_phone_segments=int(args.min_phone_segments),
            max_dataset_duration=float(args.max_dataset_duration),
            max_public_duration=float(args.max_public_duration),
            max_edge_silence=float(args.max_edge_silence),
            public_weight=float(args.public_weight),
        )
        if ok:
            kept.append(cleaned)
            source_counts[str(cleaned.get("source", "") or "unknown")] += 1
        else:
            reason_counts[reason] += 1

    count = write_manifest(args.out, kept)
    summary = {
        "input": os.path.abspath(args.manifest),
        "output": os.path.abspath(args.out),
        "input_rows": len(rows),
        "kept_rows": count,
        "rejected_rows": len(rows) - count,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "kept_by_source": dict(sorted(source_counts.items())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _clean_row(
    row: dict[str, Any],
    *,
    min_compatible: float,
    min_phone_segments: int,
    max_dataset_duration: float,
    max_public_duration: float,
    max_edge_silence: float,
    public_weight: float,
) -> tuple[bool, str, dict[str, Any]]:
    label_path = str(row.get("label_path", "") or "")
    label_format = str(row.get("label_format", "") or "")
    language = str(row.get("language", "") or "")
    segments = load_aligned_segments(label_path, label_format)
    if not segments:
        return False, "empty_label", {}
    compat = label_compatibility_ratio(segments, language=language)
    if compat < float(min_compatible):
        return False, "low_label_compatibility", {}
    non_empty = [seg for seg in segments if str(seg.label or "").strip()]
    if len(non_empty) < int(min_phone_segments):
        return False, "too_few_phone_segments", {}
    duration = max(float(seg.end) for seg in segments)
    if duration <= 0.0:
        return False, "bad_duration", {}
    source = str(row.get("source", "") or "")
    max_duration = float(max_public_duration if source == "public_sinsy" else max_dataset_duration)
    if duration > max_duration:
        return False, "too_long", {}
    leading = max(0.0, float(non_empty[0].start))
    trailing = max(0.0, duration - float(non_empty[-1].end))
    if max(leading, trailing) > float(max_edge_silence):
        return False, "excessive_edge_silence", {}
    labeled_sec = sum(max(0.0, float(seg.end) - float(seg.start)) for seg in non_empty)
    active_ratio = labeled_sec / max(duration, 1e-6)
    if active_ratio < 0.02:
        return False, "too_little_labeled_audio", {}

    cleaned = dict(row)
    base_weight = float(cleaned.get("weight", 1.0) or 1.0)
    if source == "public_sinsy":
        base_weight *= float(public_weight)
    if duration > 8.0:
        base_weight *= 0.75
    if max(leading, trailing) > 2.0:
        base_weight *= 0.70
    cleaned["weight"] = round(max(0.05, min(1.0, base_weight)), 4)
    meta = dict(cleaned.get("meta") or {})
    meta.update(
        {
            "clean_compatibility": round(float(compat), 4),
            "clean_duration_sec": round(float(duration), 4),
            "clean_active_ratio": round(float(active_ratio), 4),
            "clean_leading_silence_sec": round(float(leading), 4),
            "clean_trailing_silence_sec": round(float(trailing), 4),
        }
    )
    cleaned["meta"] = meta
    return True, "kept", cleaned


if __name__ == "__main__":
    raise SystemExit(main())
