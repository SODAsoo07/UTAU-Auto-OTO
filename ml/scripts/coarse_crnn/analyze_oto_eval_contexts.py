from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any


PARAM_NAMES = ("preutterance", "offset", "consonant", "cutoff_abs", "overlap")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize weak OTO CRNN evaluation contexts.")
    parser.add_argument("--eval", required=True, help="Evaluation JSON produced by evaluate_oto.py.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report output path.")
    parser.add_argument("--format-type", default="", help="Optional format_type filter, e.g. cvvc.")
    parser.add_argument("--min-files", type=int, default=5)
    parser.add_argument("--top-contexts", type=int, default=12)
    parser.add_argument("--top-examples", type=int, default=20)
    args = parser.parse_args()

    with open(args.eval, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = analyze_eval_contexts(
        payload,
        format_type=str(args.format_type or ""),
        min_files=int(args.min_files),
        top_contexts=int(args.top_contexts),
        top_examples=int(args.top_examples),
    )
    result["source_eval"] = os.path.abspath(args.eval)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if args.markdown_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown_out)), exist_ok=True)
        with open(args.markdown_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(render_markdown_report(result))

    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, ensure_ascii=False, indent=2))
    return 0


def analyze_eval_contexts(
    payload: dict[str, Any],
    *,
    format_type: str = "",
    min_files: int = 5,
    top_contexts: int = 12,
    top_examples: int = 20,
) -> dict[str, Any]:
    rows = [
        row
        for row in list(payload.get("files") or [])
        if not format_type or str(row.get("format_type", "") or "").lower() == format_type.lower()
    ]
    contexts = _group_rows(rows, _context_key)
    voicebanks = _group_rows(rows, lambda row: str(row.get("voicebank_id", "") or "unknown_voicebank"))
    roles = _group_rows(rows, _role_key)

    context_rows = [
        _summarize_group(key, group)
        for key, group in contexts.items()
        if len(group) >= int(min_files)
    ]
    voicebank_rows = [
        _summarize_group(key, group)
        for key, group in voicebanks.items()
        if len(group) >= int(min_files)
    ]
    role_rows = [
        _summarize_group(key, group)
        for key, group in roles.items()
        if len(group) >= int(min_files)
    ]
    context_rows.sort(key=_weakness_score, reverse=True)
    voicebank_rows.sort(key=_weakness_score, reverse=True)
    role_rows.sort(key=_weakness_score, reverse=True)

    examples = sorted(rows, key=_row_error_score, reverse=True)[: int(top_examples)]
    return {
        "format_type": format_type or "all",
        "evaluated_files": len(rows),
        "min_files": int(min_files),
        "top_contexts": context_rows[: int(top_contexts)],
        "top_voicebanks": voicebank_rows[: int(top_contexts)],
        "top_alias_roles": role_rows[: int(top_contexts)],
        "examples": [_compact_example(row) for row in examples],
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        f"# OTO CRNN Context Analysis: {result.get('format_type', 'all')}",
        "",
        f"- Source eval: `{result.get('source_eval', '')}`",
        f"- Evaluated files: {result.get('evaluated_files', 0)}",
        f"- Minimum files per group: {result.get('min_files', 0)}",
        "",
        "## Weak Contexts",
        "",
        _render_group_table(result.get("top_contexts") or []),
        "",
        "## Weak Voicebanks",
        "",
        _render_group_table(result.get("top_voicebanks") or []),
        "",
        "## Weak Alias Roles",
        "",
        _render_group_table(result.get("top_alias_roles") or []),
        "",
        "## Worst Examples",
        "",
        "| score | context | voicebank | alias | pre | offset | consonant | cutoff_abs | overlap | hard | low_conf |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in result.get("examples") or []:
        lines.append(
            "| {score:.2f} | `{context}` | `{voicebank_id}` | `{alias}` | {preutterance:.2f} | "
            "{offset:.2f} | {consonant:.2f} | {cutoff_abs:.2f} | {overlap:.2f} | {hard_failure} | {low_confidence} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_group_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| key | files | score | pre_acc50 | pre_mae | offset_mae | consonant_mae | cutoff_abs_mae | overlap_mae | hard_rate | low_conf_rate | worst_voicebank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `{key}` | {files} | {score:.2f} | {preutterance_acc_50ms:.4f} | {preutterance_mae_ms:.2f} | "
            "{offset_mae_ms:.2f} | {consonant_mae_ms:.2f} | {cutoff_abs_mae_ms:.2f} | {overlap_mae_ms:.2f} | "
            "{hard_failure_rate:.4f} | {low_confidence_rate:.4f} | `{worst_voicebank}` |".format(**row)
        )
    return "\n".join(lines)


def _group_rows(rows: list[dict[str, Any]], key_fn) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return dict(grouped)


def _context_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("format_type", "") or "other"),
            str(row.get("alias_type", "") or "other"),
            str(row.get("transition_type", "") or "other"),
        ]
    )


def _role_key(row: dict[str, Any]) -> str:
    role = str(row.get("alias_role", "") or "other")
    diph = "diph" if bool(row.get("is_diphthong", False)) else "mono"
    special = "special" if bool(row.get("is_special", False)) else "normal"
    return f"{role}|{diph}|{special}"


def _summarize_group(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = {name: [_param_error(row, name) for row in rows] for name in PARAM_NAMES}
    voicebank_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        voicebank_counts[str(row.get("voicebank_id", "") or "unknown_voicebank")] += 1
    worst_voicebank = max(voicebank_counts.items(), key=lambda item: item[1])[0] if voicebank_counts else ""
    pre_errors = errors["preutterance"]
    summary = {
        "key": key,
        "files": len(rows),
        "preutterance_acc_50ms": _ratio([value <= 50.0 for value in pre_errors]),
        "hard_failure_rate": _ratio([bool(row.get("hard_failure", False)) for row in rows]),
        "low_confidence_rate": _ratio([bool(row.get("low_confidence", False)) for row in rows]),
        "worst_voicebank": worst_voicebank,
    }
    for name, values in errors.items():
        summary[f"{name}_mae_ms"] = _mean(values)
    summary["score"] = _weakness_score(summary)
    return summary


def _weakness_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("cutoff_abs_mae_ms", 0.0) or 0.0) * 0.35
        + float(row.get("offset_mae_ms", 0.0) or 0.0) * 0.25
        + float(row.get("preutterance_mae_ms", 0.0) or 0.0) * 0.25
        + float(row.get("consonant_mae_ms", 0.0) or 0.0) * 0.15
        + (1.0 - float(row.get("preutterance_acc_50ms", 0.0) or 0.0)) * 100.0
    )


def _row_error_score(row: dict[str, Any]) -> float:
    return (
        _param_error(row, "cutoff_abs") * 0.35
        + _param_error(row, "offset") * 0.25
        + _param_error(row, "preutterance") * 0.25
        + _param_error(row, "consonant") * 0.15
    )


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": _row_error_score(row),
        "context": _context_key(row),
        "voicebank_id": str(row.get("voicebank_id", "") or "unknown_voicebank"),
        "alias": str(row.get("alias", "") or ""),
        "audio": str(row.get("audio", "") or ""),
        "preutterance": _param_error(row, "preutterance"),
        "offset": _param_error(row, "offset"),
        "consonant": _param_error(row, "consonant"),
        "cutoff_abs": _param_error(row, "cutoff_abs"),
        "overlap": _param_error(row, "overlap"),
        "hard_failure": bool(row.get("hard_failure", False)),
        "low_confidence": bool(row.get("low_confidence", False)),
    }


def _param_error(row: dict[str, Any], name: str) -> float:
    return float((row.get("param_abs_errors_ms") or {}).get(name, 0.0) or 0.0)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ratio(values: list[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
