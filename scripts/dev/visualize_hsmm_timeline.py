from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mfa_free_oto.review_overlay import write_review_html


def load_timeline_record(
    timeline_path: str | Path,
    *,
    wav: str = "",
    row_index: int | None = None,
) -> dict[str, object]:
    records = _load_jsonl(Path(timeline_path))
    if row_index is not None:
        if row_index < 0 or row_index >= len(records):
            raise ValueError(f"row index out of range: {row_index}")
        record = records[row_index]
        _require_hsmm_states(record)
        return record

    wav_key = str(wav or "").strip()
    if wav_key:
        normalized = os.path.normcase(os.path.abspath(wav_key))
        for record in records:
            record_wav = str(record.get("wav", "") or "")
            record_path = str(record.get("wav_path", "") or "")
            candidates = {
                record_wav,
                Path(record_wav).name,
                Path(record_wav).stem,
                record_path,
                Path(record_path).name,
                Path(record_path).stem,
            }
            path_match = record_path and os.path.normcase(os.path.abspath(record_path)) == normalized
            if wav_key in candidates or path_match:
                _require_hsmm_states(record)
                return record
        raise ValueError(f"wav not found in timeline: {wav_key}")

    for record in records:
        hsmm = record.get("hsmm")
        if isinstance(hsmm, Mapping) and list(hsmm.get("states", []) or []):
            return record
    raise ValueError("timeline contains no record with HSMM states")


def write_timeline_review_html(
    timeline_record: Mapping[str, object],
    out_path: str | Path,
    *,
    width: int = 1280,
) -> None:
    hsmm = timeline_record.get("hsmm")
    if not isinstance(hsmm, Mapping):
        raise ValueError("selected timeline record has no HSMM block")
    wav_path = str(timeline_record.get("wav_path", "") or "")
    if not Path(wav_path).is_file():
        raise FileNotFoundError(f"wav file not found: {wav_path}")
    states = [dict(item) for item in list(hsmm.get("states", []) or []) if isinstance(item, Mapping)]
    if not states:
        raise ValueError("selected timeline record has no HSMM states")
    events = [
        dict(item)
        for item in list(hsmm.get("events", []) or timeline_record.get("selected_events", []) or [])
        if isinstance(item, Mapping)
    ]
    adapted_rows = [
        dict(item)
        for item in list(timeline_record.get("adapted_rows", []) or [])
        if isinstance(item, Mapping)
    ]
    row = {
        "row_id": Path(wav_path).stem,
        "wav_name": str(timeline_record.get("wav", "") or Path(wav_path).name),
        "wav_path": wav_path,
        "expected_phones": list(timeline_record.get("expected_phones", []) or []),
        "label_source": "hsmm_timeline",
    }
    write_review_html(
        out_path,
        row,
        decoded_events=events,
        generated_oto_rows=adapted_rows,
        hsmm_states=states,
        width=width,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"timeline file not found: {path}")
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"timeline line {line_number} is not an object")
            records.append(payload)
    if not records:
        raise ValueError("timeline file is empty")
    return records


def _require_hsmm_states(record: Mapping[str, object]) -> None:
    hsmm = record.get("hsmm")
    if not isinstance(hsmm, Mapping) or not list(hsmm.get("states", []) or []):
        raise ValueError("selected timeline record has no HSMM states")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render HSMM state spans, selected boundaries, and generated OTO parameters from timeline.jsonl."
    )
    parser.add_argument("--timeline", required=True, help="HSMM generation *.timeline.jsonl path.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--wav", default="", help="Optional wav basename, stem, or absolute path.")
    parser.add_argument("--row-index", type=int, default=None, help="Optional zero-based timeline row index.")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    record = load_timeline_record(args.timeline, wav=args.wav, row_index=args.row_index)
    write_timeline_review_html(record, args.out, width=args.width)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).resolve()),
                "wav": str(record.get("wav", "") or ""),
                "wav_path": str(record.get("wav_path", "") or ""),
                "state_count": len(list(dict(record.get("hsmm") or {}).get("states", []) or [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
