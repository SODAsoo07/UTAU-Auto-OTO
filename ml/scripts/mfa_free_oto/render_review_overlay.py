from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.mfa_free_oto.manifest import load_manifest_jsonl
from core.mfa_free_oto.review_overlay import posterior_from_json, write_review_html


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a browser-friendly waveform/spectrogram/manual-label/prediction overlay."
    )
    parser.add_argument("--manifest", required=True, help="Manual-gold JSONL manifest.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--row-index", type=int, default=0, help="Zero-based row index when --row-id is omitted.")
    parser.add_argument("--row-id", help="Select row by row_id or wav_name.")
    parser.add_argument("--predictions", help="Optional JSON output from evaluate_ssl_poc.py.")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    rows = load_manifest_jsonl(args.manifest, require_manual=True, require_labels=False)
    row = _select_row(rows, args.row_id, args.row_index)
    posterior = None
    decoded_events = []
    if args.predictions:
        pred = _select_prediction(json.loads(Path(args.predictions).read_text(encoding="utf-8")), row)
        if pred:
            posterior = posterior_from_json(pred, row["wav_path"])
            decoded_events = pred.get("decoded_events") or []
    write_review_html(args.out, row, posterior=posterior, decoded_events=decoded_events, width=args.width)
    print(args.out)
    return 0


def _select_row(rows: list[dict], row_id: str | None, row_index: int) -> dict:
    if row_id:
        for row in rows:
            if row.get("row_id") == row_id or row.get("wav_name") == row_id:
                return row
        raise SystemExit(f"row not found: {row_id}")
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"row index out of range: {row_index}")
    return rows[row_index]


def _select_prediction(data: dict, row: dict) -> dict | None:
    predictions = data.get("predictions") or []
    row_key = row.get("row_id") or row.get("wav_name")
    for pred in predictions:
        if pred.get("row_id") == row_key:
            return pred
    return None


if __name__ == "__main__":
    raise SystemExit(main())
