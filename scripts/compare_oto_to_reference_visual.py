from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.compare_oto_pair_visual import FIELDS, _bar, _format_ms, _load_oto_rows, _row_cutoff_abs, _summary_card


def _mean_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return float(mean(vals)) if vals else 0.0


def _median_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return float(median(vals)) if vals else 0.0


def _compare_to_reference(
    reference_rows: List[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    *,
    label: str,
) -> Dict[str, object]:
    reference_by_key = {row["_match_key"]: row for row in reference_rows}
    candidate_by_key = {row["_match_key"]: row for row in candidate_rows}
    all_keys = sorted(set(reference_by_key) | set(candidate_by_key))
    rows: List[Dict[str, object]] = []
    missing_in_reference = 0
    missing_in_candidate = 0

    for key in all_keys:
        ref = reference_by_key.get(key)
        cand = candidate_by_key.get(key)
        if ref is None:
            missing_in_reference += 1
            continue
        if cand is None:
            missing_in_candidate += 1
            continue
        diffs = {field: float(cand.get(field, 0.0)) - float(ref.get(field, 0.0)) for field in FIELDS}
        diffs["cutoff_abs"] = _row_cutoff_abs(cand) - _row_cutoff_abs(ref)
        max_abs = max(abs(float(v)) for v in diffs.values()) if diffs else 0.0
        total_abs = sum(abs(float(v)) for v in diffs.values())
        rows.append(
            {
                "candidate": label,
                "wav": str(ref.get("wav", "")),
                "alias": str(ref.get("alias", "")),
                "occurrence": int(key[2]),
                "reference": {field: float(ref.get(field, 0.0)) for field in FIELDS},
                "candidate_values": {field: float(cand.get(field, 0.0)) for field in FIELDS},
                "diff": diffs,
                "max_abs_diff": float(max_abs),
                "total_abs_diff": float(total_abs),
            }
        )

    field_diffs = {field: [float(row["diff"][field]) for row in rows] for field in (*FIELDS, "cutoff_abs")}
    rows.sort(key=lambda row: float(row["max_abs_diff"]), reverse=True)
    return {
        "label": label,
        "summary": {
            "matched_rows": int(len(rows)),
            "reference_rows": int(len(reference_rows)),
            "candidate_rows": int(len(candidate_rows)),
            "missing_in_reference": int(missing_in_reference),
            "missing_in_candidate": int(missing_in_candidate),
            "mean_abs": {field: _mean_abs(values) for field, values in field_diffs.items()},
            "median_abs": {field: _median_abs(values) for field, values in field_diffs.items()},
            "max_abs": {field: max((abs(v) for v in values), default=0.0) for field, values in field_diffs.items()},
            "rows_over_20ms": int(sum(1 for row in rows if float(row["max_abs_diff"]) > 20.0)),
            "rows_over_50ms": int(sum(1 for row in rows if float(row["max_abs_diff"]) > 50.0)),
            "rows_over_100ms": int(sum(1 for row in rows if float(row["max_abs_diff"]) > 100.0)),
        },
        "rows": rows,
    }


def _write_csv(path: str, candidate_results: List[Dict[str, object]]) -> None:
    headers = [
        "candidate",
        "wav",
        "alias",
        "occurrence",
        "max_abs_diff",
        "total_abs_diff",
        *[f"reference_{field}" for field in FIELDS],
        *[f"candidate_{field}" for field in FIELDS],
        *[f"diff_{field}" for field in (*FIELDS, "cutoff_abs")],
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for result in candidate_results:
            for row in result["rows"]:
                flat = {
                    "candidate": row["candidate"],
                    "wav": row["wav"],
                    "alias": row["alias"],
                    "occurrence": row["occurrence"],
                    "max_abs_diff": f"{float(row['max_abs_diff']):.6f}",
                    "total_abs_diff": f"{float(row['total_abs_diff']):.6f}",
                }
                for field in FIELDS:
                    flat[f"reference_{field}"] = f"{float(row['reference'][field]):.6f}"
                    flat[f"candidate_{field}"] = f"{float(row['candidate_values'][field]):.6f}"
                    flat[f"diff_{field}"] = f"{float(row['diff'][field]):.6f}"
                flat["diff_cutoff_abs"] = f"{float(row['diff']['cutoff_abs']):.6f}"
                writer.writerow(flat)


def _write_html(
    path: str,
    *,
    title: str,
    reference_oto: str,
    candidates: List[Tuple[str, str]],
    candidate_results: List[Dict[str, object]],
    max_rows: int,
) -> None:
    cards = [_summary_card("reference rows", candidate_results[0]["summary"].get("reference_rows", 0) if candidate_results else 0)]
    comparison_rows = []
    max_scale = 100.0
    for result in candidate_results:
        summary = result["summary"]
        label = str(result["label"])
        mean_abs = summary.get("mean_abs", {})
        median_abs = summary.get("median_abs", {})
        max_abs = summary.get("max_abs", {})
        max_scale = max(max_scale, *(float(v) for v in max_abs.values()))
        cards.append(_summary_card(f"{label} matched", summary.get("matched_rows", 0)))
        cards.append(_summary_card(f"{label} offset MAE", f"{float(mean_abs.get('offset', 0.0)):.2f} ms"))
        cards.append(_summary_card(f"{label} pre MAE", f"{float(mean_abs.get('pre', 0.0)):.2f} ms"))
        cards.append(_summary_card(f"{label} >100ms", summary.get("rows_over_100ms", 0)))
        for field in (*FIELDS, "cutoff_abs"):
            comparison_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(field)}</td>"
                f"<td>{float(mean_abs.get(field, 0.0)):.2f}</td>"
                f"<td>{float(median_abs.get(field, 0.0)):.2f}</td>"
                f"<td>{float(max_abs.get(field, 0.0)):.2f}</td>"
                "</tr>"
            )

    sections = []
    for result in candidate_results:
        label = str(result["label"])
        row_html = []
        for row in result["rows"][: max(1, int(max_rows))]:
            diffs = row["diff"]
            row_html.append(
                "<tr>"
                f"<td>{html.escape(str(row['wav']))}</td>"
                f"<td>{html.escape(str(row['alias']))}</td>"
                f"<td>{int(row['occurrence'])}</td>"
                f"<td>{float(row['max_abs_diff']):.2f}</td>"
                + "".join(
                    f"<td><span class='num'>{_format_ms(float(diffs[field]))}</span>{_bar(float(diffs[field]), max_scale)}</td>"
                    for field in (*FIELDS, "cutoff_abs")
                )
                + "</tr>"
            )
        sections.append(
            f"""
<h2>{html.escape(label)} Largest Differences From Reference</h2>
<table>
<thead><tr><th>wav</th><th>alias</th><th>occ</th><th>max abs</th><th>offset</th><th>cons</th><th>cutoff</th><th>pre</th><th>ovl</th><th>cutoff_abs</th></tr></thead>
<tbody>{''.join(row_html)}</tbody>
</table>
"""
        )

    candidate_paths = "\n".join(
        f"<div class='path'>{html.escape(label)} OTO: {html.escape(path)}</div>" for label, path in candidates
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --line:#d8dde6; --ink:#1f2937; --muted:#687385; --pos:#1f9d6a; --neg:#c44d58; --bg:#f6f7f9; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
main {{ max-width:1420px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 10px; font-size:24px; font-weight:650; }}
h2 {{ margin:26px 0 10px; font-size:18px; }}
.path {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:18px 0; }}
.card {{ background:white; border:1px solid var(--line); border-radius:6px; padding:12px; }}
.label {{ color:var(--muted); font-size:12px; }}
.value {{ font-size:22px; font-weight:650; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); margin-bottom:18px; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; font-size:13px; vertical-align:middle; }}
th {{ background:#edf0f4; font-weight:650; position:sticky; top:0; }}
.num {{ display:inline-block; min-width:58px; font-variant-numeric:tabular-nums; }}
.bar {{ display:inline-block; width:96px; height:9px; background:#edf0f4; border-radius:4px; overflow:hidden; vertical-align:middle; }}
.bar span {{ display:block; height:100%; }}
.pos {{ background:var(--pos); }}
.neg {{ background:var(--neg); }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<div class="path">Reference OTO: {html.escape(reference_oto)}</div>
{candidate_paths}
<section class="cards">{''.join(cards)}</section>
<h2>Distance From Reference</h2>
<table>
<thead><tr><th>candidate</th><th>param</th><th>mean abs ms</th><th>median abs ms</th><th>max abs ms</th></tr></thead>
<tbody>{''.join(comparison_rows)}</tbody>
</table>
{''.join(sections)}
</main>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _parse_candidate_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        path = os.path.abspath(value)
        label = os.path.splitext(os.path.basename(path))[0] or "candidate"
        return label, path
    label, path = value.split("=", 1)
    return label.strip() or "candidate", os.path.abspath(path.strip())


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare generated OTO files against a manual/reference OTO.")
    parser.add_argument("--reference-oto", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="label=path. Repeat for each candidate.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="OTO comparison against manual reference")
    parser.add_argument("--language", default="korean")
    parser.add_argument("--max-rows", type=int, default=250)
    args = parser.parse_args(argv)

    reference_oto = os.path.abspath(args.reference_oto)
    if not os.path.isfile(reference_oto):
        raise FileNotFoundError(f"Reference OTO not found: {reference_oto}")
    candidates = [_parse_candidate_arg(value) for value in args.candidate]
    for label, path in candidates:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Candidate OTO not found ({label}): {path}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    language = str(args.language or "").strip().lower() or "korean"
    reference_rows = _load_oto_rows(reference_oto, language)
    candidate_results = [
        _compare_to_reference(reference_rows, _load_oto_rows(path, language), label=label)
        for label, path in candidates
    ]

    summary = {
        "reference_oto": reference_oto,
        "candidates": [
            {"label": str(result["label"]), "path": path, **dict(result["summary"])}
            for result, (_label, path) in zip(candidate_results, candidates)
        ],
    }
    html_path = os.path.join(out_dir, "visual_report.html")
    csv_path = os.path.join(out_dir, "details.csv")
    summary_path = os.path.join(out_dir, "summary.json")
    _write_csv(csv_path, candidate_results)
    _write_html(
        html_path,
        title=str(args.title),
        reference_oto=reference_oto,
        candidates=candidates,
        candidate_results=candidate_results,
        max_rows=int(args.max_rows),
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": {str(r["label"]): r["rows"][: int(args.max_rows)] for r in candidate_results}}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary, "html": html_path, "csv": csv_path}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
