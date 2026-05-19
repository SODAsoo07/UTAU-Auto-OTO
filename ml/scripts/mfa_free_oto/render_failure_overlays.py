from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.mfa_free_oto.manifest import load_manifest_jsonl
from core.mfa_free_oto.review_overlay import posterior_from_json, write_review_html


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render review overlays for failed or tagged rows from an MFA-free OTO evaluation JSON."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True, help="Evaluation JSON containing row_results and predictions.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--tag", action="append", help="Only render rows with this failure tag. Repeatable.")
    parser.add_argument("--include-non-catastrophic", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    rows = load_manifest_jsonl(args.manifest, require_manual=True, require_labels=False)
    rows_by_id = {str(row.get("row_id") or row.get("wav_name")): row for row in rows}
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    predictions = {str(pred.get("row_id")): pred for pred in metrics.get("predictions") or []}
    selected = _select_failures(
        metrics.get("row_results") or [],
        tags=set(args.tag or []),
        include_non_catastrophic=args.include_non_catastrophic,
        limit=args.limit,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for idx, result in enumerate(selected, start=1):
        row_id = str(result.get("row_id"))
        row = rows_by_id.get(row_id)
        pred = predictions.get(row_id)
        if row is None or pred is None:
            continue
        posterior = posterior_from_json(pred, row["wav_path"])
        out_path = out_dir / f"{idx:02d}_{_safe_name(row_id)}.html"
        write_review_html(
            out_path,
            row,
            posterior=posterior,
            decoded_events=pred.get("decoded_events") or [],
            width=args.width,
        )
        rendered.append(
            {
                "row_id": row_id,
                "path": str(out_path),
                "catastrophic": bool(result.get("catastrophic")),
                "hard_failure": bool(result.get("hard_failure")),
                "failure_tags": result.get("failure_tags") or [],
            }
        )
    summary = {"metrics": args.metrics, "rendered": rendered}
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


def _select_failures(row_results: list[dict], *, tags: set[str], include_non_catastrophic: bool, limit: int) -> list[dict]:
    selected: list[dict] = []
    for result in row_results:
        row_tags = set(result.get("failure_tags") or [])
        if tags and not (row_tags & tags):
            continue
        if not include_non_catastrophic and not result.get("catastrophic") and not result.get("hard_failure"):
            continue
        if not tags and include_non_catastrophic and not row_tags and not result.get("catastrophic") and not result.get("hard_failure"):
            continue
        selected.append(result)
    selected.sort(
        key=lambda item: (
            bool(item.get("catastrophic")),
            bool(item.get("hard_failure")),
            len(item.get("failure_tags") or []),
        ),
        reverse=True,
    )
    return selected[: max(0, limit)]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120]


if __name__ == "__main__":
    raise SystemExit(main())

