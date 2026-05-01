from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from collections import defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_file_utils import parse_oto_line
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key

FIELDS = ("offset", "cons", "cutoff", "pre", "ovl")


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _format_ms(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):+.2f}"


def _mean_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return float(mean(vals)) if vals else 0.0


def _median_abs(values: Iterable[float]) -> float:
    vals = [abs(float(v)) for v in values]
    return float(median(vals)) if vals else 0.0


def _load_oto_rows(path: str, language: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            row = parse_oto_line(raw.strip())
            if not row:
                continue
            rows.append(dict(row))
    occurrence_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for idx, row in enumerate(rows):
        wav_norm = normalize_wav_key(str(row.get("wav", "")))
        alias_norm = canonicalize_alias_for_matching(language, str(row.get("alias", "")))
        key_base = (wav_norm, alias_norm)
        occurrence = int(occurrence_counts[key_base])
        occurrence_counts[key_base] += 1
        row["_row_index"] = int(idx)
        row["_wav_norm"] = wav_norm
        row["_alias_norm"] = alias_norm
        row["_occurrence"] = occurrence
        row["_match_key"] = (wav_norm, alias_norm, occurrence)
    return rows


def _row_cutoff_abs(row: Dict[str, object]) -> float:
    offset = _to_float(row.get("offset"), 0.0)
    cutoff = _to_float(row.get("cutoff"), 0.0)
    if cutoff < 0.0:
        return offset + abs(cutoff)
    return offset + abs(cutoff)


def _compare_rows(mfa_rows: List[Dict[str, object]], no_mfa_rows: List[Dict[str, object]]) -> Dict[str, object]:
    mfa_by_key = {row["_match_key"]: row for row in mfa_rows}
    no_mfa_by_key = {row["_match_key"]: row for row in no_mfa_rows}
    all_keys = sorted(set(mfa_by_key) | set(no_mfa_by_key))
    matched: List[Dict[str, object]] = []
    missing_in_mfa = 0
    missing_in_no_mfa = 0

    for key in all_keys:
        mfa = mfa_by_key.get(key)
        no_mfa = no_mfa_by_key.get(key)
        if mfa is None:
            missing_in_mfa += 1
            continue
        if no_mfa is None:
            missing_in_no_mfa += 1
            continue
        diffs = {field: _to_float(no_mfa.get(field), 0.0) - _to_float(mfa.get(field), 0.0) for field in FIELDS}
        diffs["cutoff_abs"] = _row_cutoff_abs(no_mfa) - _row_cutoff_abs(mfa)
        max_abs = max(abs(float(v)) for v in diffs.values()) if diffs else 0.0
        total_abs = sum(abs(float(v)) for v in diffs.values())
        matched.append(
            {
                "wav": str(mfa.get("wav", "")),
                "alias": str(mfa.get("alias", "")),
                "occurrence": int(key[2]),
                "mfa": {field: _to_float(mfa.get(field), 0.0) for field in FIELDS},
                "no_mfa": {field: _to_float(no_mfa.get(field), 0.0) for field in FIELDS},
                "diff": diffs,
                "max_abs_diff": float(max_abs),
                "total_abs_diff": float(total_abs),
            }
        )

    field_diffs = {field: [float(row["diff"][field]) for row in matched] for field in (*FIELDS, "cutoff_abs")}
    summary = {
        "matched_rows": int(len(matched)),
        "mfa_rows": int(len(mfa_rows)),
        "no_mfa_rows": int(len(no_mfa_rows)),
        "missing_in_mfa": int(missing_in_mfa),
        "missing_in_no_mfa": int(missing_in_no_mfa),
        "mean_abs": {field: _mean_abs(values) for field, values in field_diffs.items()},
        "median_abs": {field: _median_abs(values) for field, values in field_diffs.items()},
        "max_abs": {field: max((abs(v) for v in values), default=0.0) for field, values in field_diffs.items()},
        "rows_over_20ms": int(sum(1 for row in matched if float(row["max_abs_diff"]) > 20.0)),
        "rows_over_50ms": int(sum(1 for row in matched if float(row["max_abs_diff"]) > 50.0)),
        "rows_over_100ms": int(sum(1 for row in matched if float(row["max_abs_diff"]) > 100.0)),
    }
    matched.sort(key=lambda row: float(row["max_abs_diff"]), reverse=True)
    return {"summary": summary, "rows": matched}


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    headers = [
        "wav",
        "alias",
        "occurrence",
        "max_abs_diff",
        "total_abs_diff",
        *[f"mfa_{field}" for field in FIELDS],
        *[f"no_mfa_{field}" for field in FIELDS],
        *[f"diff_{field}" for field in (*FIELDS, "cutoff_abs")],
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            flat = {
                "wav": row["wav"],
                "alias": row["alias"],
                "occurrence": row["occurrence"],
                "max_abs_diff": f"{float(row['max_abs_diff']):.6f}",
                "total_abs_diff": f"{float(row['total_abs_diff']):.6f}",
            }
            for field in FIELDS:
                flat[f"mfa_{field}"] = f"{float(row['mfa'][field]):.6f}"
                flat[f"no_mfa_{field}"] = f"{float(row['no_mfa'][field]):.6f}"
                flat[f"diff_{field}"] = f"{float(row['diff'][field]):.6f}"
            flat["diff_cutoff_abs"] = f"{float(row['diff']['cutoff_abs']):.6f}"
            writer.writerow(flat)


def _bar(value: float, scale: float) -> str:
    width = min(100.0, abs(float(value)) / max(1.0, float(scale)) * 100.0)
    cls = "pos" if value >= 0.0 else "neg"
    return f'<span class="bar"><span class="{cls}" style="width:{width:.1f}%"></span></span>'


def _summary_card(label: str, value: object) -> str:
    return f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(str(value))}</div></div>"


def _write_html(path: str, title: str, mfa_oto: str, no_mfa_oto: str, summary: Dict[str, object], rows: List[Dict[str, object]], max_rows: int) -> None:
    scale = max(20.0, float(summary.get("max_abs", {}).get("offset", 0.0) or 0.0), 100.0)
    row_html = []
    for row in rows[: max(1, int(max_rows))]:
        diffs = row["diff"]
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['wav']))}</td>"
            f"<td>{html.escape(str(row['alias']))}</td>"
            f"<td>{int(row['occurrence'])}</td>"
            f"<td>{float(row['max_abs_diff']):.2f}</td>"
            + "".join(
                f"<td><span class='num'>{_format_ms(float(diffs[field]))}</span>{_bar(float(diffs[field]), scale)}</td>"
                for field in (*FIELDS, "cutoff_abs")
            )
            + "</tr>"
        )

    mean_abs = summary.get("mean_abs", {})
    median_abs = summary.get("median_abs", {})
    max_abs = summary.get("max_abs", {})
    stats_rows = []
    for field in (*FIELDS, "cutoff_abs"):
        stats_rows.append(
            "<tr>"
            f"<td>{html.escape(field)}</td>"
            f"<td>{float(mean_abs.get(field, 0.0)):.2f}</td>"
            f"<td>{float(median_abs.get(field, 0.0)):.2f}</td>"
            f"<td>{float(max_abs.get(field, 0.0)):.2f}</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --line:#d8dde6; --ink:#1f2937; --muted:#687385; --pos:#1f9d6a; --neg:#c44d58; --bg:#f6f7f9; }}
body {{ margin:0; font-family:Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
main {{ max-width:1360px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 10px; font-size:24px; font-weight:650; }}
h2 {{ margin:26px 0 10px; font-size:18px; }}
.path {{ color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:18px 0; }}
.card {{ background:white; border:1px solid var(--line); border-radius:6px; padding:12px; }}
.label {{ color:var(--muted); font-size:12px; }}
.value {{ font-size:22px; font-weight:650; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); }}
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
<div class="path">MFA OTO: {html.escape(mfa_oto)}</div>
<div class="path">No-MFA OTO: {html.escape(no_mfa_oto)}</div>
<section class="cards">
{_summary_card("matched rows", summary.get("matched_rows", 0))}
{_summary_card("MFA rows", summary.get("mfa_rows", 0))}
{_summary_card("No-MFA rows", summary.get("no_mfa_rows", 0))}
{_summary_card(">20 ms rows", summary.get("rows_over_20ms", 0))}
{_summary_card(">50 ms rows", summary.get("rows_over_50ms", 0))}
{_summary_card(">100 ms rows", summary.get("rows_over_100ms", 0))}
</section>
<h2>Parameter Distance</h2>
<table>
<thead><tr><th>param</th><th>mean abs ms</th><th>median abs ms</th><th>max abs ms</th></tr></thead>
<tbody>
{''.join(stats_rows)}
</tbody>
</table>
<h2>Largest Row Differences</h2>
<table>
<thead><tr><th>wav</th><th>alias</th><th>occ</th><th>max abs</th><th>offset</th><th>cons</th><th>cutoff</th><th>pre</th><th>ovl</th><th>cutoff_abs</th></tr></thead>
<tbody>
{''.join(row_html)}
</tbody>
</table>
</main>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visual comparison for MFA-generated OTO and No-MFA-generated OTO.")
    parser.add_argument("--mfa-oto", required=True)
    parser.add_argument("--no-mfa-oto", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="MFA vs No-MFA OTO comparison")
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--language", default="korean")
    args = parser.parse_args(argv)

    mfa_oto = os.path.abspath(args.mfa_oto)
    no_mfa_oto = os.path.abspath(args.no_mfa_oto)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(mfa_oto):
        raise FileNotFoundError(f"MFA OTO not found: {mfa_oto}")
    if not os.path.isfile(no_mfa_oto):
        raise FileNotFoundError(f"No-MFA OTO not found: {no_mfa_oto}")
    os.makedirs(out_dir, exist_ok=True)

    language = str(args.language or "").strip().lower() or "korean"
    result = _compare_rows(_load_oto_rows(mfa_oto, language), _load_oto_rows(no_mfa_oto, language))
    summary = dict(result["summary"])
    rows = list(result["rows"])
    summary["mfa_oto"] = mfa_oto
    summary["no_mfa_oto"] = no_mfa_oto

    summary_path = os.path.join(out_dir, "summary.json")
    detail_path = os.path.join(out_dir, "details.csv")
    html_path = os.path.join(out_dir, "visual_report.html")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows[: args.max_rows]}, handle, ensure_ascii=False, indent=2)
    _write_csv(detail_path, rows)
    _write_html(html_path, args.title, mfa_oto, no_mfa_oto, summary, rows, args.max_rows)

    print(json.dumps({"summary": summary, "html": html_path, "csv": detail_path}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
