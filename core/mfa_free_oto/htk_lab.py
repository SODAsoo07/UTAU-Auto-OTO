from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .manifest import wav_duration_ms, write_manifest_jsonl
from .types import MANUAL_SOURCE

HTK_TO_MS = 1.0 / 10_000.0

VOWEL_LABELS = {"a", "i", "u", "e", "o", "ɯ", "ə", "ɑ"}
TRUE_SILENCE_LABELS = {"ap", "sp", "sil", "pau"}
OTHER_LABELS = {"br", "bre", "breath", "exh", "axh", "cl", "spn"}
DROP_LABELS = {"vf"}


@dataclass(frozen=True)
class HtkLabSegment:
    start_ms: float
    end_ms: float
    phone: str
    frame_label: str


@dataclass(frozen=True)
class HtkGoldBuildSummary:
    rows: int
    labelled_rows: int
    skipped_without_wav: int
    dropped_segments: int
    source_roots: list[str]
    out_path: str


def parse_htk_lab(
    path: str | Path,
    *,
    drop_labels: Iterable[str] = DROP_LABELS,
) -> tuple[list[HtkLabSegment], list[dict]]:
    dropped = {label.lower() for label in drop_labels}
    segments: list[HtkLabSegment] = []
    dropped_segments: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            parts = raw_line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start_ms = float(parts[0]) * HTK_TO_MS
                end_ms = float(parts[1]) * HTK_TO_MS
            except ValueError:
                continue
            if end_ms <= start_ms:
                continue
            phone = parts[2]
            low = phone.lower()
            if low in dropped:
                dropped_segments.append(
                    {
                        "line": line_no,
                        "phone": phone,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    }
                )
                continue
            segments.append(
                HtkLabSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    phone=phone,
                    frame_label=classify_htk_phone(phone),
                )
            )
    return segments, dropped_segments


def classify_htk_phone(phone: str) -> str:
    low = phone.strip().lower()
    if low in TRUE_SILENCE_LABELS:
        return "silence"
    if low in OTHER_LABELS:
        return "other"
    if low in VOWEL_LABELS:
        return "vowel"
    return "consonant"


def row_from_htk_lab(
    lab_path: str | Path,
    *,
    wav_path: str | Path | None = None,
    language: str,
    format_type: str,
    dataset_id: str | None = None,
    relative_to: str | Path | None = None,
) -> dict:
    lab = Path(lab_path)
    wav = Path(wav_path) if wav_path else lab.with_suffix(".wav")
    segments, dropped = parse_htk_lab(lab)
    base = Path(relative_to).resolve() if relative_to else None
    row = {
        "schema_version": 1,
        "row_id": lab.stem,
        "wav_name": wav.name,
        "wav_path": _rel_or_abs_path(wav.resolve(), base),
        "lab_path": _rel_or_abs_path(lab.resolve(), base),
        "duration_ms": wav_duration_ms(wav),
        "language": language,
        "format_type": format_type,
        "dataset_id": dataset_id or lab.parent.name,
        "label_source": MANUAL_SOURCE,
        "label_format": "htk_lab",
        "label_policy": {
            "true_silence": sorted(TRUE_SILENCE_LABELS),
            "other": sorted(OTHER_LABELS),
            "dropped": sorted(DROP_LABELS),
            "filename_is_not_label_source": True,
            "vowel_nucleus": "htk_vowel_segment_midpoint_pseudo_gold",
        },
        "expected_phones": expected_phones_from_segments(segments),
        "frame_labels": frame_labels_from_segments(segments),
        "events": events_from_segments(segments),
        "dropped_lab_segments": dropped,
        "needs_manual_labels": False,
    }
    return row


def build_gold_manifest_from_htk_lab_dirs(
    lab_roots: Sequence[str | Path],
    *,
    out_path: str | Path,
    language: str,
    format_type: str,
    dataset_id: str | None = None,
    relative_to: str | Path | None = None,
) -> tuple[list[dict], HtkGoldBuildSummary]:
    rows: list[dict] = []
    skipped_without_wav = 0
    dropped_segments = 0
    for root_like in lab_roots:
        root = Path(root_like)
        for lab_path in sorted(root.rglob("*.lab")):
            rel_parts = lab_path.relative_to(root).parts[:-1]
            if any(part.startswith("_") for part in rel_parts):
                continue
            wav_path = lab_path.with_suffix(".wav")
            if not wav_path.is_file():
                skipped_without_wav += 1
                continue
            row = row_from_htk_lab(
                lab_path,
                wav_path=wav_path,
                language=language,
                format_type=format_type,
                dataset_id=dataset_id,
                relative_to=relative_to,
            )
            dropped_segments += len(row.get("dropped_lab_segments") or [])
            rows.append(row)
    write_manifest_jsonl(out_path, rows)
    summary = HtkGoldBuildSummary(
        rows=len(rows),
        labelled_rows=len(rows),
        skipped_without_wav=skipped_without_wav,
        dropped_segments=dropped_segments,
        source_roots=[str(Path(root).resolve()) for root in lab_roots],
        out_path=str(Path(out_path)),
    )
    return rows, summary


def expected_phones_from_segments(segments: Sequence[HtkLabSegment]) -> list[str]:
    return [
        segment.phone
        for segment in segments
        if segment.frame_label not in {"silence"} and segment.phone.strip()
    ]


def frame_labels_from_segments(segments: Sequence[HtkLabSegment]) -> list[dict]:
    return [
        {
            "label": segment.frame_label,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "phone": segment.phone,
            "confidence": 1.0,
        }
        for segment in segments
    ]


def events_from_segments(segments: Sequence[HtkLabSegment]) -> list[dict]:
    events: list[dict] = []
    previous_trainable: HtkLabSegment | None = None
    for segment in segments:
        if segment.frame_label == "silence":
            continue
        if previous_trainable is not None:
            events.append(
                {
                    "label": "phone_change",
                    "time_ms": segment.start_ms,
                    "phone": segment.phone,
                    "confidence": 1.0,
                }
            )
        if previous_trainable is not None and previous_trainable.frame_label == "consonant" and segment.frame_label == "vowel":
            events.append(
                {
                    "label": "cv_boundary",
                    "time_ms": segment.start_ms,
                    "phone": segment.phone,
                    "confidence": 1.0,
                }
            )
        if segment.frame_label == "vowel":
            events.append(
                {
                    "label": "vowel_nucleus",
                    "time_ms": (segment.start_ms + segment.end_ms) * 0.5,
                    "phone": segment.phone,
                    "confidence": 0.75,
                    "source": "htk_vowel_segment_midpoint_pseudo_gold",
                    "vowel_start_ms": segment.start_ms,
                    "vowel_end_ms": segment.end_ms,
                }
            )
        previous_trainable = segment
    return events


def write_summary_json(path: str | Path, summary: HtkGoldBuildSummary) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel_or_abs_path(path: Path, base: Path | None) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
