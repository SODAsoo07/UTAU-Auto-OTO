from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mfa_free_oto.review_overlay import write_review_html


DEFAULT_SUMMARY = r"ml_workspace\manual_oto_anchor\scorer\manual_oto_anchor_scorer_summary.json"
DEFAULT_OUT_DIR = r"ml_workspace\manual_oto_anchor\worst_overlays"


def _slug(value: object, *, fallback: str = "row") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text[:96].strip("._-") or fallback


def _load_examples(summary: dict[str, object], source: str) -> list[dict[str, object]]:
    eval_summary = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    if source == "joint":
        key = "worst_joint_lattice_examples"
    elif source == "filename":
        key = "worst_filename_slot_lattice_examples"
    elif source == "constrained":
        key = "worst_constrained_slot_lattice_examples"
    elif source == "gated":
        key = "worst_gated_constrained_accept_examples"
    elif source == "island":
        key = "worst_vowel_island_lattice_examples"
    elif source == "island_gated":
        key = "worst_vowel_island_gated_accept_examples"
    else:
        key = "worst_model_examples"
    values = eval_summary.get(key, []) if isinstance(eval_summary, dict) else []
    return [item for item in values if isinstance(item, dict)]


def _overlay_row(example: dict[str, object], source: str, idx: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    anchor = str(example.get("anchor", "anchor") or "anchor")
    alias = str(example.get("alias", "") or "")
    target_ms = float(example.get("target_ms", 0.0) or 0.0)
    pred_ms = float(example.get("pred_ms", 0.0) or 0.0)
    error_ms = float(example.get("error_ms", 0.0) or 0.0)
    row = {
        "row_id": f"{source}:{idx:04d}:{anchor}:{alias}",
        "wav_name": Path(str(example.get("wav_path", "") or "")).name,
        "wav_path": str(example.get("wav_path", "") or ""),
        "alias": alias,
        "alias_role": example.get("alias_role", ""),
        "alias_family": example.get("alias_family", ""),
        "anchor": anchor,
        "target_ms": target_ms,
        "pred_ms": pred_ms,
        "error_ms": error_ms,
        "events": [
            {
                "label": "manual_anchor",
                "time_ms": target_ms,
                "anchor": anchor,
                "alias": alias,
            }
        ],
        "auxiliary_events": [
            {
                "label": f"target_{anchor}",
                "time_ms": target_ms,
            },
            {
                "label": f"{source}_pred_{anchor}",
                "time_ms": pred_ms,
            },
            *(example.get("island_overlay_events") or []),
        ],
        "landmark_metadata": {
            "source": source,
            "anchor": anchor,
            "error_ms": error_ms,
            "raw_example": example,
        },
    }
    decoded_label = "joint_pred_anchor" if source in {"joint", "filename", "constrained", "gated", "island", "island_gated"} else "model_pred_anchor"
    decoded = [
        {
            "label": decoded_label,
            "selected_time_ms": pred_ms,
            "score": float(example.get("joint_score", example.get("score", 0.0)) or 0.0),
            "alias": f"{alias} {anchor}",
        }
    ]
    return row, decoded


def render_overlays(summary_path: Path, out_dir: Path, *, source: str, limit: int) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    examples = _load_examples(summary, source)[: max(0, int(limit))]
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    for idx, example in enumerate(examples, start=1):
        row, decoded = _overlay_row(example, source, idx)
        wav_path = str(row.get("wav_path", "") or "")
        if not wav_path or not Path(wav_path).exists():
            generated.append({"index": idx, "status": "missing_wav", "wav_path": wav_path, "example": example})
            continue
        name = "_".join(
            [
                f"{idx:04d}",
                _slug(source),
                _slug(example.get("anchor")),
                _slug(example.get("alias"), fallback="alias"),
                _slug(Path(wav_path).stem, fallback="wav"),
            ]
        )
        html_path = out_dir / f"{name}.html"
        write_review_html(html_path, row, decoded_events=decoded)
        generated.append(
            {
                "status": "ok",
                "index": idx,
                "path": str(html_path),
                "anchor": example.get("anchor"),
                "alias": example.get("alias"),
                "error_ms": example.get("error_ms"),
                "wav_path": wav_path,
            }
        )
    index_path = out_dir / f"index_{source}.html"
    rows = "\n".join(
        _index_row(item)
        for item in generated
    )
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Manual OTO anchor worst overlays - {html.escape(source)}</title></head>
<body>
<h1>Manual OTO anchor worst overlays - {html.escape(source)}</h1>
<p>summary: {html.escape(str(summary_path))}</p>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>#</th><th>status</th><th>anchor</th><th>alias</th><th>error_ms</th><th>html</th><th>wav</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return {
        "summary_path": str(summary_path),
        "out_dir": str(out_dir),
        "source": source,
        "limit": int(limit),
        "index_path": str(index_path),
        "generated": generated,
    }


def _index_row(item: dict[str, object]) -> str:
    path = str(item.get("path", "") or "")
    link = f'<a href="{html.escape(Path(path).name)}">open</a>' if path else ""
    return (
        "<tr>"
        f"<td>{html.escape(str(item.get('index', '')))}</td>"
        f"<td>{html.escape(str(item.get('status', '')))}</td>"
        f"<td>{html.escape(str(item.get('anchor', '')))}</td>"
        f"<td>{html.escape(str(item.get('alias', '')))}</td>"
        f"<td>{html.escape(str(item.get('error_ms', '')))}</td>"
        f"<td>{link}</td>"
        f"<td>{html.escape(str(item.get('wav_path', '')))}</td>"
        "</tr>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render worst manual OTO anchor scorer examples as HTML overlays.")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--source", choices=("model", "joint", "filename", "constrained", "gated", "island", "island_gated"), default="joint")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()
    result = render_overlays(Path(args.summary), Path(args.out_dir), source=str(args.source), limit=int(args.limit))
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
