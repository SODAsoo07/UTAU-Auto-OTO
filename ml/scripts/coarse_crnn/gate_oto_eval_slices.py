from __future__ import annotations

import argparse
import json
import os
from typing import Any


PARAM_NAMES = ("preutterance", "offset", "consonant", "cutoff_abs", "overlap")

DEFAULT_GATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "pattern": "korean/cvvc/vc",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.25,
        "max_preutterance_mae_ms": 220.0,
        "max_offset_mae_ms": 900.0,
        "max_cutoff_abs_mae_ms": 1800.0,
        "max_hard_failure_rate": 0.18,
        "max_fn_voiced_miss_rate": 0.60,
        "max_syllable_shift_rate": 0.55,
        "recommendation": "korean_cvvc_vc_activity_fallback",
    },
    {
        "pattern": "korean/cvvc/vv",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.20,
        "max_preutterance_mae_ms": 220.0,
        "max_offset_mae_ms": 900.0,
        "max_cutoff_abs_mae_ms": 1800.0,
        "max_hard_failure_rate": 0.18,
        "max_fn_voiced_miss_rate": 0.55,
        "max_syllable_shift_rate": 0.55,
        "recommendation": "korean_cvvc_vv_activity_fallback",
    },
    {
        "pattern": "korean/cvvc/other",
        "min_files": 10,
        "min_preutterance_acc_50ms": 0.20,
        "max_preutterance_mae_ms": 260.0,
        "max_offset_mae_ms": 1100.0,
        "max_cutoff_abs_mae_ms": 2200.0,
        "max_hard_failure_rate": 0.20,
        "max_fn_voiced_miss_rate": 0.65,
        "max_syllable_shift_rate": 0.65,
        "recommendation": "korean_cvvc_other_or_blank_review",
    },
    {
        "pattern": "korean/cvvc/-cv",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.25,
        "max_preutterance_mae_ms": 260.0,
        "max_offset_mae_ms": 900.0,
        "max_cutoff_abs_mae_ms": 2000.0,
        "max_hard_failure_rate": 0.18,
        "max_fn_voiced_miss_rate": 0.60,
        "max_syllable_shift_rate": 0.60,
        "recommendation": "korean_cvvc_head_cv_guard",
    },
    {
        "pattern": "korean/cvvc/cv",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.30,
        "max_preutterance_mae_ms": 230.0,
        "max_offset_mae_ms": 900.0,
        "max_cutoff_abs_mae_ms": 1700.0,
        "max_hard_failure_rate": 0.16,
        "max_fn_voiced_miss_rate": 0.60,
        "max_syllable_shift_rate": 0.60,
        "recommendation": "korean_cvvc_cv_anchor_guard",
    },
    {
        "pattern": "korean/cvvc/v-cv",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.25,
        "max_preutterance_mae_ms": 230.0,
        "max_offset_mae_ms": 900.0,
        "max_cutoff_abs_mae_ms": 1600.0,
        "max_hard_failure_rate": 0.16,
        "max_fn_voiced_miss_rate": 0.60,
        "max_syllable_shift_rate": 0.60,
        "recommendation": "korean_cvvc_v_cv_bridge_guard",
    },
    {
        "pattern": "korean/cvvc/*",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.35,
        "max_preutterance_mae_ms": 180.0,
        "max_offset_mae_ms": 700.0,
        "max_cutoff_abs_mae_ms": 900.0,
        "max_hard_failure_rate": 0.12,
        "max_fn_voiced_miss_rate": 0.55,
        "max_syllable_shift_rate": 0.55,
        "recommendation": "korean_cvvc_role_specific_gate",
    },
    {
        "pattern": "japanese/cvvc/*",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.40,
        "max_preutterance_mae_ms": 160.0,
        "max_offset_mae_ms": 650.0,
        "max_cutoff_abs_mae_ms": 750.0,
        "max_hard_failure_rate": 0.12,
        "max_fn_voiced_miss_rate": 0.45,
        "max_syllable_shift_rate": 0.45,
    },
    {
        "pattern": "korean/cvc/*",
        "min_files": 20,
        "min_preutterance_acc_50ms": 0.45,
        "max_preutterance_mae_ms": 150.0,
        "max_offset_mae_ms": 450.0,
        "max_cutoff_abs_mae_ms": 650.0,
        "max_hard_failure_rate": 0.10,
        "max_fn_voiced_miss_rate": 0.45,
        "max_syllable_shift_rate": 0.45,
        "recommendation": "korean_cvc_activity_fallback",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate OTO CRNN eval results by language/format/alias_role slices.")
    parser.add_argument("--eval", required=True, help="Evaluation JSON produced by evaluate_oto.py.")
    parser.add_argument("--spec", default="", help="Optional JSON gate spec list. Uses built-in diagnostic spec if omitted.")
    parser.add_argument("--out", required=True, help="Gate report JSON output path.")
    parser.add_argument("--markdown-out", default="", help="Optional Markdown report path.")
    parser.add_argument("--top-failures", type=int, default=20)
    args = parser.parse_args()

    payload = read_json(args.eval)
    specs = read_json(args.spec) if args.spec else list(DEFAULT_GATE_SPECS)
    if isinstance(specs, dict):
        specs = list(specs.get("specs") or [])
    report = gate_eval_slices(payload, specs=list(specs), top_failures=int(args.top_failures))
    report["source_eval"] = os.path.abspath(args.eval)
    report["source_spec"] = os.path.abspath(args.spec) if args.spec else "builtin:diagnostic-v2"
    write_json(args.out, report)
    if args.markdown_out:
        write_text(args.markdown_out, render_markdown_report(report))
    compact = {key: value for key, value in report.items() if key != "slices"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


def gate_eval_slices(
    payload: dict[str, Any],
    *,
    specs: list[dict[str, Any]],
    top_failures: int = 20,
) -> dict[str, Any]:
    slices = _language_format_role_slices(payload)
    checks: list[dict[str, Any]] = []
    for key, item in sorted(slices.items()):
        spec = _matching_spec(key, specs)
        if spec is None:
            checks.append(_slice_result(key, item, None, checks=[], status="unmatched"))
            continue
        files = int(item.get("files", 0) or 0)
        min_files = int(spec.get("min_files", 0) or 0)
        if files < min_files:
            checks.append(_slice_result(key, item, spec, checks=[], status="skipped:min_files"))
            continue
        slice_checks = _evaluate_slice_checks(item, spec)
        checks.append(_slice_result(key, item, spec, checks=slice_checks, status="passed" if all(c["passed"] for c in slice_checks) else "failed"))

    matched = [row for row in checks if row["status"] not in {"unmatched", "skipped:min_files"}]
    failed = [row for row in matched if row["status"] == "failed"]
    skipped = [row for row in checks if row["status"].startswith("skipped")]
    unmatched = [row for row in checks if row["status"] == "unmatched"]
    failed.sort(key=_failure_score, reverse=True)
    return {
        "spec_version": "diagnostic-v2",
        "slice_count": len(checks),
        "matched_count": len(matched),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "unmatched_count": len(unmatched),
        "passed": len(failed) == 0,
        "gate_specs": specs,
        "failed_slices": failed[: int(top_failures)],
        "slices": checks,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OTO CRNN Language/Format/Role Gate",
        "",
        f"- Source eval: `{report.get('source_eval', '')}`",
        f"- Spec: `{report.get('source_spec', '')}`",
        f"- Matched slices: {report.get('matched_count', 0)}",
        f"- Failed slices: {report.get('failed_count', 0)}",
        f"- Skipped slices: {report.get('skipped_count', 0)}",
        f"- Unmatched slices: {report.get('unmatched_count', 0)}",
        f"- Passed: {bool(report.get('passed', False))}",
        "",
        "## Failed Slices",
        "",
        _render_slice_table(report.get("failed_slices") or []),
        "",
        "## All Matched Slices",
        "",
        _render_slice_table([row for row in report.get("slices") or [] if row.get("status") in {"passed", "failed"}]),
        "",
    ]
    return "\n".join(lines)


def read_json(path: str) -> Any:
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


def _language_format_role_slices(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    existing = payload.get("by_language_format_role")
    if isinstance(existing, dict) and existing:
        return {str(key): dict(value or {}) for key, value in existing.items()}
    rows = list(payload.get("files") or [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "/".join(
            [
                str(row.get("language", "") or "unknown"),
                str(row.get("format_type", "") or "unknown"),
                str(row.get("alias_role", "") or "other"),
            ]
        )
        grouped.setdefault(key, []).append(row)
    return {key: _summarize_rows(key, group) for key, group in sorted(grouped.items())}


def _summarize_rows(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    language, format_type, alias_role = key.split("/", 2)
    pre_errors = [_param_error(row, "preutterance") for row in rows]
    return {
        "files": len(rows),
        "language": language,
        "format_type": format_type,
        "alias_role": alias_role,
        "preutterance_mae_ms": _mean(pre_errors),
        "preutterance_acc_50ms": _hit_rate(pre_errors, 50.0),
        "offset_mae_ms": _mean([_param_error(row, "offset") for row in rows]),
        "consonant_mae_ms": _mean([_param_error(row, "consonant") for row in rows]),
        "cutoff_abs_mae_ms": _mean([_param_error(row, "cutoff_abs") for row in rows]),
        "overlap_mae_ms": _mean([_param_error(row, "overlap") for row in rows]),
        "hard_failure_rate": _ratio([bool(row.get("hard_failure", False)) for row in rows]),
        "fn_voiced_miss_rate": _ratio([bool(row.get("fn_voiced_miss", False)) for row in rows]),
        "syllable_shift_rate": _ratio([bool(row.get("syllable_shift_flag", False)) for row in rows]),
    }


def _matching_spec(key: str, specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    slash_key = key.replace("|", "/")
    for spec in specs:
        pattern = str((spec or {}).get("pattern", "") or "")
        if _match_pattern(slash_key, pattern):
            return dict(spec)
    return None


def _match_pattern(key: str, pattern: str) -> bool:
    key_parts = key.split("/")
    pattern_parts = pattern.split("/")
    if len(key_parts) != len(pattern_parts):
        return False
    return all(pattern_part == "*" or key_part == pattern_part for key_part, pattern_part in zip(key_parts, pattern_parts))


def _evaluate_slice_checks(item: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.extend(
        [
            _min_check(item, spec, "preutterance_acc_50ms", "min_preutterance_acc_50ms"),
            _max_check(item, spec, "preutterance_mae_ms", "max_preutterance_mae_ms"),
            _max_check(item, spec, "offset_mae_ms", "max_offset_mae_ms"),
            _max_check(item, spec, "cutoff_abs_mae_ms", "max_cutoff_abs_mae_ms"),
            _max_check(item, spec, "hard_failure_rate", "max_hard_failure_rate"),
            _max_check(item, spec, "fn_voiced_miss_rate", "max_fn_voiced_miss_rate"),
            _max_check(item, spec, "syllable_shift_rate", "max_syllable_shift_rate"),
        ]
    )
    return [check for check in checks if check]


def _slice_result(
    key: str,
    item: dict[str, Any],
    spec: dict[str, Any] | None,
    *,
    checks: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    slash_key = key.replace("|", "/")
    result = {
        "key": slash_key,
        "status": status,
        "pattern": str((spec or {}).get("pattern", "")),
        "recommendation": str((spec or {}).get("recommendation", "")),
        "files": int(item.get("files", 0) or 0),
        "preutterance_acc_50ms": _float_or_none(item.get("preutterance_acc_50ms")),
        "preutterance_mae_ms": _float_or_none(item.get("preutterance_mae_ms")),
        "offset_mae_ms": _float_or_none(item.get("offset_mae_ms")),
        "consonant_mae_ms": _float_or_none(item.get("consonant_mae_ms")),
        "cutoff_abs_mae_ms": _float_or_none(item.get("cutoff_abs_mae_ms")),
        "overlap_mae_ms": _float_or_none(item.get("overlap_mae_ms")),
        "hard_failure_rate": _float_or_none(item.get("hard_failure_rate")),
        "fn_voiced_miss_rate": _float_or_none(item.get("fn_voiced_miss_rate")),
        "syllable_shift_rate": _float_or_none(item.get("syllable_shift_rate")),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not bool(check.get("passed", False))],
    }
    result["fallback_tags"] = _fallback_tags(result)
    return result


def _min_check(item: dict[str, Any], spec: dict[str, Any], metric: str, threshold_key: str) -> dict[str, Any] | None:
    if threshold_key not in spec:
        return None
    actual = _float_or_none(item.get(metric))
    threshold = float(spec[threshold_key])
    return {
        "name": metric,
        "operator": ">=",
        "actual": actual,
        "threshold": threshold,
        "passed": actual is not None and actual >= threshold,
    }


def _max_check(item: dict[str, Any], spec: dict[str, Any], metric: str, threshold_key: str) -> dict[str, Any] | None:
    if threshold_key not in spec:
        return None
    actual = _float_or_none(item.get(metric))
    threshold = float(spec[threshold_key])
    return {
        "name": metric,
        "operator": "<=",
        "actual": actual,
        "threshold": threshold,
        "passed": actual is not None and actual <= threshold,
    }


def _render_slice_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| slice | status | files | pre <=50ms | pre MAE | offset MAE | cutoff MAE | hard fail | voiced miss | shift | fallback tags | failed checks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _code_cell(row.get("key", "")),
                    str(row.get("status", "")),
                    str(int(row.get("files", 0) or 0)),
                    _pct(row.get("preutterance_acc_50ms")),
                    _fmt(row.get("preutterance_mae_ms")),
                    _fmt(row.get("offset_mae_ms")),
                    _fmt(row.get("cutoff_abs_mae_ms")),
                    _pct(row.get("hard_failure_rate")),
                    _pct(row.get("fn_voiced_miss_rate")),
                    _pct(row.get("syllable_shift_rate")),
                    ", ".join(row.get("fallback_tags") or []),
                    ", ".join(row.get("failed_checks") or []),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _failure_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("cutoff_abs_mae_ms", 0.0) or 0.0) * 0.35
        + float(row.get("offset_mae_ms", 0.0) or 0.0) * 0.25
        + float(row.get("preutterance_mae_ms", 0.0) or 0.0) * 0.25
        + float(row.get("fn_voiced_miss_rate", 0.0) or 0.0) * 90.0
        + float(row.get("syllable_shift_rate", 0.0) or 0.0) * 110.0
        + (1.0 - float(row.get("preutterance_acc_50ms", 0.0) or 0.0)) * 100.0
    )


def _fallback_tags(row: dict[str, Any]) -> list[str]:
    if str(row.get("status", "")) != "failed":
        return []
    failed = set(row.get("failed_checks") or [])
    tags: list[str] = []
    recommendation = str(row.get("recommendation", "") or "")
    if recommendation:
        tags.append(recommendation)
    if "fn_voiced_miss_rate" in failed:
        tags.append("activity_window_fallback")
    if "syllable_shift_rate" in failed:
        tags.append("syllable_shift_guard")
    if "offset_mae_ms" in failed or "cutoff_abs_mae_ms" in failed:
        tags.append("right_boundary_prior_review")
    if "preutterance_acc_50ms" in failed or "preutterance_mae_ms" in failed:
        tags.append("preutterance_prior_review")
    if "hard_failure_rate" in failed:
        tags.append("hard_failure_quarantine")
    return list(dict.fromkeys(tags))


def _param_error(row: dict[str, Any], name: str) -> float:
    return float((row.get("param_abs_errors_ms") or {}).get(name, 0.0) or 0.0)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / float(len(values))) if values else None


def _ratio(values: list[bool]) -> float | None:
    return float(sum(1 for value in values if value) / float(len(values))) if values else None


def _hit_rate(values: list[float], threshold: float) -> float | None:
    return float(sum(1 for value in values if float(value) <= float(threshold))) / float(len(values)) if values else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


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
