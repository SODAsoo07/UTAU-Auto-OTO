from __future__ import annotations

import argparse
import json
import os
from typing import Any


PARAM_NAMES = ("preutterance", "offset", "consonant", "cutoff_abs", "overlap")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze OTO CRNN fallback trigger candidates from eval JSON.")
    parser.add_argument("--eval", required=True, help="Evaluation JSON produced by evaluate_oto.py.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report path.")
    parser.add_argument("--language", default="", help="Optional language filter, e.g. korean.")
    parser.add_argument("--format-type", default="", help="Optional format_type filter, e.g. cvvc.")
    parser.add_argument("--alias-role", default="", help="Optional alias_role filter, e.g. vc.")
    parser.add_argument("--bad-pre-ms", type=float, default=50.0)
    parser.add_argument("--bad-offset-ms", type=float, default=700.0)
    parser.add_argument("--bad-cutoff-ms", type=float, default=900.0)
    args = parser.parse_args()

    with open(args.eval, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = analyze_fallback_candidates(
        payload,
        language=str(args.language or ""),
        format_type=str(args.format_type or ""),
        alias_role=str(args.alias_role or ""),
        bad_pre_ms=float(args.bad_pre_ms),
        bad_offset_ms=float(args.bad_offset_ms),
        bad_cutoff_ms=float(args.bad_cutoff_ms),
    )
    result["source_eval"] = os.path.abspath(args.eval)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    if args.markdown_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown_out)), exist_ok=True)
        with open(args.markdown_out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown_report(result))

    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, ensure_ascii=False, indent=2))
    return 0


def analyze_fallback_candidates(
    payload: dict[str, Any],
    *,
    language: str = "",
    format_type: str = "",
    alias_role: str = "",
    bad_pre_ms: float = 50.0,
    bad_offset_ms: float = 700.0,
    bad_cutoff_ms: float = 900.0,
) -> dict[str, Any]:
    rows = [
        row
        for row in list(payload.get("files") or [])
        if _matches_filter(row, language=language, format_type=format_type, alias_role=alias_role)
    ]
    bad_flags = [_is_bad_row(row, bad_pre_ms=bad_pre_ms, bad_offset_ms=bad_offset_ms, bad_cutoff_ms=bad_cutoff_ms) for row in rows]
    candidates = _candidate_triggers(rows)
    candidate_rows = [
        _summarize_candidate(name, triggered, rows, bad_flags)
        for name, triggered in candidates
    ]
    candidate_rows.sort(key=lambda row: (row["bad_recall"], row["good_keep_rate"], -row["fallback_rate"]), reverse=True)
    examples = sorted(rows, key=_row_error_score, reverse=True)[:20]
    return {
        "filter": {
            "language": language,
            "format_type": format_type,
            "alias_role": alias_role,
        },
        "thresholds": {
            "bad_pre_ms": bad_pre_ms,
            "bad_offset_ms": bad_offset_ms,
            "bad_cutoff_ms": bad_cutoff_ms,
        },
        "rows": len(rows),
        "bad_rows": int(sum(1 for flag in bad_flags if flag)),
        "bad_rate": _ratio(bad_flags),
        "base_metrics": _metrics_with_fallback(rows, [False] * len(rows)),
        "candidates": candidate_rows,
        "examples": [_compact_example(row) for row in examples],
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    filt = result.get("filter") or {}
    lines = [
        "# OTO CRNN Fallback Candidate Analysis",
        "",
        f"- Source eval: `{result.get('source_eval', '')}`",
        f"- Filter: language=`{filt.get('language', '')}`, format=`{filt.get('format_type', '')}`, role=`{filt.get('alias_role', '')}`",
        f"- Rows: {result.get('rows', 0)}",
        f"- Bad rows: {result.get('bad_rows', 0)} ({float(result.get('bad_rate', 0.0)):.4f})",
        "",
        "## Candidate Triggers",
        "",
        "| trigger | fallback_rate | bad_recall | precision | good_keep | eff_pre_acc50 | eff_pre_mae | eff_offset_mae | eff_cutoff_mae |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("candidates") or []:
        lines.append(
            "| `{name}` | {fallback_rate:.4f} | {bad_recall:.4f} | {precision:.4f} | {good_keep_rate:.4f} | "
            "{effective_preutterance_acc_50ms:.4f} | {effective_preutterance_mae_ms:.2f} | "
            "{effective_offset_mae_ms:.2f} | {effective_cutoff_abs_mae_ms:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Worst Examples",
            "",
            "| score | role | voicebank | alias | conf | pred_err | pre | offset | cutoff_abs | hard | fn | shift | low_conf |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in result.get("examples") or []:
        lines.append(
            "| {score:.2f} | `{alias_role}` | `{voicebank_id}` | `{alias}` | {confidence:.4f} | {predicted_error_ms:.2f} | "
            "{preutterance:.2f} | {offset:.2f} | {cutoff_abs:.2f} | {hard_failure} | {fn_voiced_miss} | "
            "{syllable_shift_flag} | {low_confidence} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _matches_filter(row: dict[str, Any], *, language: str, format_type: str, alias_role: str) -> bool:
    if language and str(row.get("language", "") or "").lower() != language.lower():
        return False
    if format_type and str(row.get("format_type", "") or "").lower() != format_type.lower():
        return False
    if alias_role and str(row.get("alias_role", "") or "").lower() != alias_role.lower():
        return False
    return True


def _is_bad_row(row: dict[str, Any], *, bad_pre_ms: float, bad_offset_ms: float, bad_cutoff_ms: float) -> bool:
    return bool(
        _param_error(row, "preutterance") > float(bad_pre_ms)
        or _param_error(row, "offset") > float(bad_offset_ms)
        or _param_error(row, "cutoff_abs") > float(bad_cutoff_ms)
        or bool(row.get("hard_failure", False))
        or bool(row.get("fn_voiced_miss", False))
    )


def _candidate_triggers(rows: list[dict[str, Any]]) -> list[tuple[str, list[bool]]]:
    return [
        ("low_confidence", [bool(row.get("low_confidence", False)) for row in rows]),
        ("hard_failure", [bool(row.get("hard_failure", False)) for row in rows]),
        ("fn_voiced_miss", [bool(row.get("fn_voiced_miss", False)) for row in rows]),
        ("syllable_shift", [bool(row.get("syllable_shift_flag", False)) for row in rows]),
        ("confidence<=0.45", [_confidence(row) <= 0.45 for row in rows]),
        ("confidence<=0.50", [_confidence(row) <= 0.50 for row in rows]),
        ("confidence<=0.55", [_confidence(row) <= 0.55 for row in rows]),
        ("predicted_error>=80", [_predicted_error(row) >= 80.0 for row in rows]),
        ("predicted_error>=120", [_predicted_error(row) >= 120.0 for row in rows]),
        ("predicted_error>=160", [_predicted_error(row) >= 160.0 for row in rows]),
        ("heatmap<=0.15", [_heatmap_confidence(row) <= 0.15 for row in rows]),
        ("heatmap<=0.25", [_heatmap_confidence(row) <= 0.25 for row in rows]),
        (
            "runtime_low_conf_or_activity",
            [
                bool(row.get("low_confidence", False))
                or bool(row.get("hard_failure", False))
                or bool(row.get("fn_voiced_miss", False))
                for row in rows
            ],
        ),
    ]


def _summarize_candidate(name: str, triggered: list[bool], rows: list[dict[str, Any]], bad_flags: list[bool]) -> dict[str, Any]:
    total = len(rows)
    bad_total = sum(1 for flag in bad_flags if flag)
    good_total = max(0, total - bad_total)
    fallback_count = sum(1 for flag in triggered if flag)
    true_positive = sum(1 for trigger, bad in zip(triggered, bad_flags) if trigger and bad)
    false_positive = sum(1 for trigger, bad in zip(triggered, bad_flags) if trigger and not bad)
    metrics = _metrics_with_fallback(rows, triggered)
    return {
        "name": name,
        "rows": total,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / total) if total else 0.0,
        "bad_recall": float(true_positive / bad_total) if bad_total else 0.0,
        "precision": float(true_positive / fallback_count) if fallback_count else 0.0,
        "good_keep_rate": float((good_total - false_positive) / good_total) if good_total else 0.0,
        **{f"effective_{key}": value for key, value in metrics.items()},
    }


def _metrics_with_fallback(rows: list[dict[str, Any]], triggered: list[bool]) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in PARAM_NAMES}
    for row, fallback in zip(rows, triggered):
        for name in PARAM_NAMES:
            values[name].append(0.0 if fallback else _param_error(row, name))
    pre = values["preutterance"]
    return {
        "preutterance_acc_50ms": _ratio([value <= 50.0 for value in pre]),
        "preutterance_mae_ms": _mean(pre),
        "offset_mae_ms": _mean(values["offset"]),
        "consonant_mae_ms": _mean(values["consonant"]),
        "cutoff_abs_mae_ms": _mean(values["cutoff_abs"]),
        "overlap_mae_ms": _mean(values["overlap"]),
    }


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": _row_error_score(row),
        "alias_role": str(row.get("alias_role", "") or "other"),
        "voicebank_id": str(row.get("voicebank_id", "") or "unknown_voicebank"),
        "alias": str(row.get("alias", "") or ""),
        "audio": str(row.get("audio", "") or ""),
        "confidence": _confidence(row),
        "predicted_error_ms": _predicted_error(row),
        "preutterance": _param_error(row, "preutterance"),
        "offset": _param_error(row, "offset"),
        "cutoff_abs": _param_error(row, "cutoff_abs"),
        "hard_failure": bool(row.get("hard_failure", False)),
        "fn_voiced_miss": bool(row.get("fn_voiced_miss", False)),
        "syllable_shift_flag": bool(row.get("syllable_shift_flag", False)),
        "low_confidence": bool(row.get("low_confidence", False)),
    }


def _row_error_score(row: dict[str, Any]) -> float:
    return (
        _param_error(row, "cutoff_abs") * 0.35
        + _param_error(row, "offset") * 0.25
        + _param_error(row, "preutterance") * 0.25
        + _param_error(row, "consonant") * 0.15
    )


def _param_error(row: dict[str, Any], name: str) -> float:
    return float((row.get("param_abs_errors_ms") or {}).get(name, 0.0) or 0.0)


def _confidence(row: dict[str, Any]) -> float:
    return float(row.get("confidence", 0.0) or 0.0)


def _predicted_error(row: dict[str, Any]) -> float:
    value = row.get("predicted_error_ms")
    return float(value) if value is not None else 0.0


def _heatmap_confidence(row: dict[str, Any]) -> float:
    return float((row.get("confidence_components") or {}).get("heatmap", 0.0) or 0.0)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ratio(values: list[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
