from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_name(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "eval"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "eval"


def _infer_format_label_from_score_dir(score_dir: str, fallback: str) -> str:
    parts = [
        part.lower()
        for part in os.path.abspath(str(score_dir or "")).replace("\\", "/").split("/")
        if part
    ]
    for part in reversed(parts):
        for fmt in ("cvvc", "vcv", "cvc"):
            if part == fmt or part.startswith(f"{fmt}_") or part.endswith(f"_{fmt}") or f"_{fmt}_" in part:
                return fmt
    return fallback


def _float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def _run(cmd: List[str], *, cwd: str, stop_on_error: bool) -> int:
    print("[RUN] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0 and stop_on_error:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def _load_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _batch_error_count(run_name: str) -> int:
    path = os.path.join(ROOT_DIR, "logs", "oto_batch", str(run_name), "summary.json")
    payload = _load_json(path)
    try:
        return int(payload.get("error_cases", 0) or 0)
    except Exception:
        return 0


def _load_details(path: str) -> List[Dict[str, object]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_error_score(row: Dict[str, object]) -> float:
    keys = (
        "pre_abs_err_ms",
        "offset_err_ms",
        "cons_abs_err_ms",
        "cutoff_abs_err_ms",
        "ovl_abs_err_ms",
    )
    return max(abs(_float(row, key)) for key in keys)


def _fmt_ms(value: object) -> str:
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "0.0"


def _svg_timeline(row: Dict[str, object]) -> str:
    width = 720
    height = 86
    pad = 34
    fields = ("offset", "ovl_abs", "pre_abs", "cons_abs", "cutoff_abs")
    vals = []
    for prefix in ("manual", "candidate"):
        for field in fields:
            vals.append(_float(row, f"{field}_{prefix}"))
    lo = min(vals + [0.0])
    hi = max(vals + [1.0])
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo

    def x(v: float) -> float:
        return pad + ((float(v) - lo) / span) * (width - (pad * 2))

    marks = [
        ("offset", "O"),
        ("ovl_abs", "V"),
        ("pre_abs", "P"),
        ("cons_abs", "C"),
        ("cutoff_abs", "X"),
    ]
    parts = [
        f'<svg class="timeline" viewBox="0 0 {width} {height}" role="img">',
        f'<line x1="{pad}" y1="22" x2="{width - pad}" y2="22" class="axis"/>',
        f'<line x1="{pad}" y1="58" x2="{width - pad}" y2="58" class="axis"/>',
        f'<text x="4" y="26" class="label">manual</text>',
        f'<text x="4" y="62" class="label">auto</text>',
    ]
    for prefix, y, cls in (("manual", 22, "manual"), ("candidate", 58, "candidate")):
        off = x(_float(row, f"offset_{prefix}"))
        cut = x(_float(row, f"cutoff_abs_{prefix}"))
        parts.append(f'<line x1="{off:.2f}" y1="{y}" x2="{cut:.2f}" y2="{y}" class="bar {cls}"/>')
        for field, label in marks:
            xv = x(_float(row, f"{field}_{prefix}"))
            parts.append(f'<line x1="{xv:.2f}" y1="{y - 10}" x2="{xv:.2f}" y2="{y + 10}" class="tick {cls}"/>')
            parts.append(f'<text x="{xv + 3:.2f}" y="{y - 13}" class="mark">{label}</text>')
    parts.append(f'<text x="{pad}" y="82" class="scale">{_fmt_ms(lo)}ms</text>')
    parts.append(f'<text x="{width - pad - 80}" y="82" class="scale">{_fmt_ms(hi)}ms</text>')
    parts.append("</svg>")
    return "".join(parts)


def _summary_table(summary_rows: List[Dict[str, object]]) -> str:
    headers = [
        "format",
        "cases",
        "matched",
        "manual 0 rows",
        "match",
        "pre median",
        "pre >50ms",
        "cutoff median",
        "generated 0-ish",
    ]
    rows = [
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
    ]
    for item in summary_rows:
        s = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
        fmt = str(item.get("format", ""))
        rows.append(
            "<tr>"
            f"<td>{html.escape(fmt)}</td>"
            f"<td>{int(float(s.get('case_count', 0) or 0))}</td>"
            f"<td>{int(float(s.get('matched_rows', 0) or 0))}</td>"
            f"<td>{int(float(s.get('manual_zeroish_skipped', 0) or 0))}</td>"
            f"<td>{float(s.get('match_rate', 0.0) or 0.0):.3f}</td>"
            f"<td>{float(s.get('pre_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
            f"<td>{float(s.get('pre_abs_gt_50ms_rate', 0.0) or 0.0):.1%}</td>"
            f"<td>{float(s.get('cutoff_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
            f"<td>{float(s.get('candidate_zeroish_rate', 0.0) or 0.0):.1%}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _alias_type_tables(summary_rows: List[Dict[str, object]]) -> str:
    blocks: List[str] = ["<h2>Alias Type Summary</h2>"]
    for item in summary_rows:
        fmt = str(item.get("format", ""))
        by_alias = item.get("by_alias_type", {})
        if not isinstance(by_alias, dict) or not by_alias:
            blocks.append(f"<h3>{html.escape(fmt)}</h3><p>No alias type summary.</p>")
            continue
        rows = sorted(
            by_alias.items(),
            key=lambda kv: float((kv[1] or {}).get("pre_abs_median_ms", 0.0) or 0.0),
            reverse=True,
        )
        blocks.append(f"<h3>{html.escape(fmt)}</h3>")
        blocks.append(
            "<table class=\"compact\"><thead><tr>"
            "<th>alias type</th><th>matched</th><th>pre med</th><th>pre mae</th>"
            "<th>pre &gt;50ms</th><th>cutoff med</th><th>offset med</th><th>zeroish</th>"
            "</tr></thead><tbody>"
        )
        for alias_type, raw_stats in rows:
            stats = raw_stats if isinstance(raw_stats, dict) else {}
            blocks.append(
                "<tr>"
                f"<td>{html.escape(str(alias_type))}</td>"
                f"<td>{int(float(stats.get('matched_rows', 0) or 0))}</td>"
                f"<td>{float(stats.get('pre_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('pre_abs_mae_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('pre_abs_gt_50ms_rate', 0.0) or 0.0):.1%}</td>"
                f"<td>{float(stats.get('cutoff_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('offset_median_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('candidate_zeroish_rate', 0.0) or 0.0):.1%}</td>"
                "</tr>"
            )
        blocks.append("</tbody></table>")
    return "\n".join(blocks)


def _group_summary_tables(
    summary_rows: List[Dict[str, object]],
    *,
    title: str,
    payload_key: str,
    label: str,
) -> str:
    blocks: List[str] = [f"<h2>{html.escape(title)}</h2>"]
    for item in summary_rows:
        fmt = str(item.get("format", ""))
        grouped = item.get(payload_key, {})
        if not isinstance(grouped, dict) or not grouped:
            blocks.append(f"<h3>{html.escape(fmt)}</h3><p>No grouped summary.</p>")
            continue
        blocks.append(f"<h3>{html.escape(fmt)}</h3>")
        blocks.append(
            "<table class=\"compact\"><thead><tr>"
            f"<th>{html.escape(label)}</th><th>matched</th><th>pre med</th><th>pre mae</th>"
            "<th>pre &gt;50ms</th><th>offset med</th><th>cutoff med</th>"
            "</tr></thead><tbody>"
        )
        for group_key, raw_stats in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            stats = raw_stats if isinstance(raw_stats, dict) else {}
            blocks.append(
                "<tr>"
                f"<td>{html.escape(str(group_key))}</td>"
                f"<td>{int(float(stats.get('matched_rows', 0) or 0))}</td>"
                f"<td>{float(stats.get('pre_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('pre_abs_mae_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('pre_abs_gt_50ms_rate', 0.0) or 0.0):.1%}</td>"
                f"<td>{float(stats.get('offset_median_ms', 0.0) or 0.0):.1f}ms</td>"
                f"<td>{float(stats.get('cutoff_abs_median_ms', 0.0) or 0.0):.1f}ms</td>"
                "</tr>"
            )
        blocks.append("</tbody></table>")
    return "\n".join(blocks)


def _detail_cards(format_label: str, details: List[Dict[str, object]], *, worst_rows: int) -> str:
    rows = sorted(details, key=_row_error_score, reverse=True)[: max(0, int(worst_rows))]
    out = [f"<h2>{html.escape(format_label)} worst rows</h2>"]
    if not rows:
        out.append("<p>No matched rows to visualize.</p>")
        return "\n".join(out)
    for row in rows:
        err = _row_error_score(row)
        meta = (
            f"{row.get('case_id', '')} / {row.get('wav', '')} = "
            f"{row.get('alias', '')} [{row.get('alias_type', '')}, "
            f"{row.get('file_position_bucket', '?')}, "
            f"row {row.get('file_order_no', '?')}/{row.get('file_order_count', '?')}, "
            f"occ {row.get('occurrence_no', '?')}]"
        )
        out.append('<section class="card">')
        out.append(
            f"<h3>{html.escape(str(meta))}</h3>"
            f"<p class=\"err\">max error {err:.1f}ms, "
            f"pre {abs(_float(row, 'pre_abs_err_ms')):.1f}ms, "
            f"offset {abs(_float(row, 'offset_err_ms')):.1f}ms, "
            f"cutoff {abs(_float(row, 'cutoff_abs_err_ms')):.1f}ms</p>"
        )
        out.append(_svg_timeline(row))
        out.append("</section>")
    return "\n".join(out)


def _write_visual_report(score_infos: List[Dict[str, object]], *, out_path: str, worst_rows: int) -> str:
    summary_rows = []
    detail_blocks = []
    for info in score_infos:
        fmt = str(info.get("format", ""))
        score_dir = str(info.get("score_dir", ""))
        summary_payload = _load_json(os.path.join(score_dir, "summary.json"))
        summary = summary_payload.get("summary", {}) if isinstance(summary_payload.get("summary"), dict) else {}
        by_alias_type = (
            summary_payload.get("by_alias_type", {})
            if isinstance(summary_payload.get("by_alias_type", {}), dict)
            else {}
        )
        by_file_position = (
            summary_payload.get("by_file_position", {})
            if isinstance(summary_payload.get("by_file_position", {}), dict)
            else {}
        )
        by_alias_type_position = (
            summary_payload.get("by_alias_type_position", {})
            if isinstance(summary_payload.get("by_alias_type_position", {}), dict)
            else {}
        )
        by_occurrence_index = (
            summary_payload.get("by_occurrence_index", {})
            if isinstance(summary_payload.get("by_occurrence_index", {}), dict)
            else {}
        )
        details = _load_details(os.path.join(score_dir, "details.csv"))
        summary_rows.append(
            {
                "format": fmt,
                "summary": summary,
                "by_alias_type": by_alias_type,
                "by_file_position": by_file_position,
                "by_alias_type_position": by_alias_type_position,
                "by_occurrence_index": by_occurrence_index,
            }
        )
        detail_blocks.append(_detail_cards(fmt, details, worst_rows=worst_rows))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Korean OTO Evaluation</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    h2 {{ margin-top: 30px; border-bottom: 1px solid #d8dde6; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; background: white; width: 100%; margin: 12px 0 22px; }}
    th, td {{ border: 1px solid #d8dde6; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f6; }}
    h3 {{ font-size: 16px; margin: 16px 0 6px; }}
    .compact th, .compact td {{ padding: 6px 8px; font-size: 12px; }}
    .card {{ background: white; border: 1px solid #d8dde6; border-radius: 6px; padding: 12px; margin: 12px 0; }}
    .card h3 {{ font-size: 14px; margin: 0 0 4px; }}
    .err {{ margin: 0 0 8px; color: #6b2737; font-size: 13px; }}
    .timeline {{ width: 100%; height: 96px; display: block; }}
    .axis {{ stroke: #c7ced8; stroke-width: 1; }}
    .bar {{ stroke-width: 6; stroke-linecap: round; }}
    .manual {{ stroke: #2f6fbb; }}
    .candidate {{ stroke: #d25f2b; }}
    .tick {{ stroke-width: 2; }}
    .label, .scale, .mark {{ fill: #52606d; font-size: 11px; }}
    .mark {{ font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Korean CVC/CVVC/VCV OTO Evaluation</h1>
  {_summary_table(summary_rows)}
  {_alias_type_tables(summary_rows)}
  {_group_summary_tables(summary_rows, title="File Position Summary", payload_key="by_file_position", label="position")}
  {_group_summary_tables(summary_rows, title="Alias Type x Position Summary", payload_key="by_alias_type_position", label="alias/position")}
  {_group_summary_tables(summary_rows, title="Occurrence Summary", payload_key="by_occurrence_index", label="occurrence")}
  {"".join(detail_blocks)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html_text)
    return out_path


def _format_score_info(run_root: str, fmt: str) -> Dict[str, object]:
    return {
        "format": fmt,
        "run_root": run_root,
        "manifest": os.path.join(run_root, "manifest.json"),
        "score_dir": os.path.join(run_root, "scores"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Korean CVC/CVVC/VCV sequence OTO evaluation and build an HTML visual report."
    )
    parser.add_argument("--formats", nargs="+", default=["cvc", "cvvc", "vcv"])
    parser.add_argument("--limit-per-format", type=int, default=3)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--aligner", default="sequence")
    parser.add_argument("--fallback-aligner", default="")
    parser.add_argument("--base-mode", choices=("zero", "manual"), default="zero")
    parser.add_argument("--out-root", default=os.path.join(ROOT_DIR, "eval_runs", "dataset_oto_eval"))
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--score-dir", action="append", default=[], help="Existing score dir for visual-only mode.")
    parser.add_argument("--worst-rows", type=int, default=40)
    parser.add_argument("--strip-alias-suffix", default="")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    run_base = _safe_name(args.run_name or f"kr_sequence_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = os.path.abspath(args.out_root)
    score_infos: List[Dict[str, object]] = []

    if args.visual_only:
        for idx, score_dir in enumerate(args.score_dir or []):
            fallback = f"score_{idx + 1}"
            score_infos.append(
                {
                    "format": _infer_format_label_from_score_dir(score_dir, fallback),
                    "score_dir": os.path.abspath(score_dir),
                }
            )
        if not score_infos:
            raise SystemExit("--visual-only requires at least one --score-dir")
    else:
        for fmt in args.formats:
            fmt_key = str(fmt or "").strip().lower()
            run_name = f"{run_base}_{fmt_key}"
            run_root = os.path.join(out_root, run_name)
            build_cmd = [
                sys.executable,
                os.path.join("scripts", "build_dataset_eval_cases.py"),
                "--language",
                "korean",
                "--format",
                fmt_key,
                "--limit",
                str(max(0, int(args.limit_per_format))),
                "--aligner",
                args.aligner,
                "--base-mode",
                args.base_mode,
                "--run-name",
                run_name,
                "--out-dir",
                out_root,
            ]
            if args.fallback_aligner:
                build_cmd.extend(["--fallback-aligner", args.fallback_aligner])
            rc = _run(build_cmd, cwd=ROOT_DIR, stop_on_error=args.stop_on_error)
            info = _format_score_info(run_root, fmt_key)
            if rc == 0 and not args.skip_generation:
                batch_cmd = [
                    sys.executable,
                    os.path.join("scripts", "run_oto_generation_batch.py"),
                    "--config",
                    os.path.join(run_root, f"batch_{_safe_name(args.aligner)}.yaml"),
                    "--run-tag",
                    run_name,
                ]
                rc = _run(batch_cmd, cwd=ROOT_DIR, stop_on_error=args.stop_on_error)
                batch_errors = _batch_error_count(run_name)
                if batch_errors > 0:
                    print(f"[WARN] batch completed with error_cases={batch_errors} run={run_name}", flush=True)
                    if args.stop_on_error:
                        raise SystemExit(1)
            if rc == 0:
                score_cmd = [
                    sys.executable,
                    os.path.join("scripts", "score_oto_eval.py"),
                    "--manifest",
                    info["manifest"],
                    "--strip-alias-suffix",
                    args.strip_alias_suffix,
                ]
                rc = _run(score_cmd, cwd=ROOT_DIR, stop_on_error=args.stop_on_error)
            if rc == 0:
                score_infos.append(info)

    report_path = os.path.join(out_root, run_base, "visual_report.html")
    _write_visual_report(score_infos, out_path=report_path, worst_rows=args.worst_rows)
    print(f"[DONE] visual_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
