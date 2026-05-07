from __future__ import annotations

import argparse
import json
import os
from typing import Any


PARAM_KEYS = ("offset", "preutterance", "consonant", "cutoff_abs", "overlap")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OTO CRNN diagnostics across voicebanks.")
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("LABEL", "DIAGNOSTIC_JSON", "EVAL_JSON"),
        required=True,
        help="Voicebank label, analyze_oto_voicebank_diagnostics.py JSON, and evaluate_oto.py JSON.",
    )
    parser.add_argument("--out", required=True, help="Comparison JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report path.")
    parser.add_argument("--top-contexts", type=int, default=5)
    args = parser.parse_args()

    cases = []
    for label, diagnostic_path, eval_path in args.case:
        cases.append(
            {
                "label": label,
                "diagnostic_path": os.path.abspath(diagnostic_path),
                "eval_path": os.path.abspath(eval_path),
                "diagnostic": read_json(diagnostic_path),
                "eval": read_json(eval_path),
            }
        )

    result = compare_voicebanks(cases, top_contexts=int(args.top_contexts))
    write_json(args.out, result)
    if args.markdown_out:
        write_text(args.markdown_out, render_markdown_report(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def compare_voicebanks(cases: list[dict[str, Any]], *, top_contexts: int = 5) -> dict[str, Any]:
    rows = [_summarize_case(case, top_contexts=top_contexts) for case in cases]
    baseline = rows[0] if rows else None
    if baseline:
        for row in rows[1:]:
            row["delta_vs_first"] = _delta(row, baseline)
    return {
        "case_count": len(rows),
        "baseline_label": baseline.get("label") if baseline else None,
        "rows": rows,
        "notes": _build_notes(rows),
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# OTO CRNN CVVC Voicebank Comparison",
        "",
        f"- Cases: {result.get('case_count', 0)}",
        f"- Baseline: `{result.get('baseline_label', '')}`",
        "",
        "## Summary",
        "",
        "| Voicebank | rows | blank % | row-order violation % | pre <=50ms % | pre MAE | offset MAE | cutoff MAE | overlap MAE | hard fail % | onset lag p90 | syllable shift % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("rows") or []:
        params = row.get("param_mae_ms") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("label", "")),
                    _fmt(row.get("evaluated_items"), digits=0),
                    _pct(row.get("blank_alias_rate")),
                    _pct(row.get("row_order_violation_rate")),
                    _pct(row.get("preutterance_acc_50ms")),
                    _fmt(params.get("preutterance")),
                    _fmt(params.get("offset")),
                    _fmt(params.get("cutoff_abs")),
                    _fmt(params.get("overlap")),
                    _pct(row.get("hard_failure_rate")),
                    _fmt(row.get("onset_lag_p90_ms")),
                    _pct(row.get("syllable_shift_rate")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Weak Contexts", ""])
    for row in result.get("rows") or []:
        lines.extend([f"### {row.get('label', '')}", ""])
        lines.append("| context | rows | pre <=50ms % | pre MAE | offset MAE | cutoff MAE | overlap MAE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for context in row.get("weak_contexts") or []:
            params = context.get("param_mae_ms") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _code_cell(context.get("key", "")),
                        _fmt(context.get("files"), digits=0),
                        _pct(context.get("preutterance_acc_50ms")),
                        _fmt(params.get("preutterance")),
                        _fmt(params.get("offset")),
                        _fmt(params.get("cutoff_abs")),
                        _fmt(params.get("overlap")),
                    ]
                )
                + " |"
            )
        lines.append("")
    notes = result.get("notes") or []
    if notes:
        lines.extend(["## Notes", ""])
        lines.extend([f"- {note}" for note in notes])
        lines.append("")
    return "\n".join(lines)


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _summarize_case(case: dict[str, Any], *, top_contexts: int) -> dict[str, Any]:
    diagnostic = case.get("diagnostic") or {}
    eval_payload = case.get("eval") or {}
    row_order = diagnostic.get("row_order_summary") or {}
    target = diagnostic.get("target_summary") or {}
    target_count = _summary_count(target, "target_offset_ms")
    eval_error = diagnostic.get("eval_error_summary") or {}
    row = {
        "label": case.get("label", ""),
        "diagnostic_path": case.get("diagnostic_path", ""),
        "eval_path": case.get("eval_path", ""),
        "manifest_rows": diagnostic.get("manifest_rows", target_count),
        "eval_matched_rows": diagnostic.get("eval_matched_rows", eval_error.get("eval_matched_rows")),
        "evaluated_items": eval_payload.get("evaluated_items"),
        "blank_alias_rate": target.get("blank_alias_rate"),
        "row_order_violation_rate": row_order.get("monotonic_violation_rate"),
        "row_order_abs_offset_vs_ratio_mean_ms": _summary_value(row_order, "abs_offset_vs_row_ratio_ms", "mean"),
        "row_order_abs_offset_vs_ratio_p90_ms": _summary_value(row_order, "abs_offset_vs_row_ratio_ms", "p90"),
        "preutterance_acc_50ms": eval_payload.get("preutterance_acc_50ms"),
        "hard_failure_rate": eval_payload.get("hard_failure_rate"),
        "fn_voiced_miss_rate": eval_payload.get("fn_voiced_miss_rate"),
        "onset_lag_p90_ms": eval_payload.get("onset_lag_p90_ms"),
        "syllable_shift_rate": eval_payload.get("syllable_shift_rate"),
        "param_mae_ms": {key: (eval_payload.get("param_mae_ms") or {}).get(key) for key in PARAM_KEYS},
        "weak_contexts": _weak_contexts(eval_payload.get("by_alias_context") or {}, limit=top_contexts),
    }
    return row


def _weak_contexts(contexts: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, context in contexts.items():
        files = int(context.get("files", 0) or 0)
        if files <= 0:
            continue
        rows.append(
            {
                "key": key,
                "files": files,
                "preutterance_acc_50ms": context.get("preutterance_acc_50ms"),
                "param_mae_ms": {
                    "preutterance": context.get("preutterance_mae_ms"),
                    "offset": context.get("offset_mae_ms"),
                    "consonant": context.get("consonant_mae_ms"),
                    "cutoff_abs": context.get("cutoff_abs_mae_ms"),
                    "overlap": context.get("overlap_mae_ms"),
                },
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("preutterance_acc_50ms", 0.0) or 0.0),
            -float((row.get("param_mae_ms") or {}).get("preutterance", 0.0) or 0.0),
            -int(row.get("files", 0) or 0),
        )
    )
    return rows[:limit]


def _delta(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "blank_alias_rate",
        "row_order_violation_rate",
        "preutterance_acc_50ms",
        "hard_failure_rate",
        "fn_voiced_miss_rate",
        "onset_lag_p90_ms",
        "syllable_shift_rate",
    ):
        result[key] = _safe_sub(row.get(key), baseline.get(key))
    result["param_mae_ms"] = {
        key: _safe_sub((row.get("param_mae_ms") or {}).get(key), (baseline.get("param_mae_ms") or {}).get(key))
        for key in PARAM_KEYS
    }
    return result


def _build_notes(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) < 2:
        return []
    best_pre = max(rows, key=lambda row: float(row.get("preutterance_acc_50ms", 0.0) or 0.0))
    worst_pre = min(rows, key=lambda row: float(row.get("preutterance_acc_50ms", 0.0) or 0.0))
    highest_blank = max(rows, key=lambda row: float(row.get("blank_alias_rate", 0.0) or 0.0))
    return [
        f"Best preutterance <=50ms accuracy: {best_pre.get('label')} ({_pct(best_pre.get('preutterance_acc_50ms'))}).",
        f"Worst preutterance <=50ms accuracy: {worst_pre.get('label')} ({_pct(worst_pre.get('preutterance_acc_50ms'))}).",
        f"Highest blank alias rate: {highest_blank.get('label')} ({_pct(highest_blank.get('blank_alias_rate'))}).",
    ]


def _summary_value(payload: dict[str, Any], key: str, field: str) -> float | None:
    value = payload.get(key)
    if not isinstance(value, dict):
        return None
    raw = value.get(field)
    return float(raw) if raw is not None else None


def _summary_count(payload: dict[str, Any], key: str) -> int | None:
    raw = _summary_value(payload, key, "count")
    return int(raw) if raw is not None else None


def _safe_sub(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _fmt(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100.0:.1f}"


def _code_cell(value: Any) -> str:
    text = str(value or "").replace("|", "\\|")
    return "`" + text.replace("`", "\\`") + "`"


if __name__ == "__main__":
    raise SystemExit(main())
