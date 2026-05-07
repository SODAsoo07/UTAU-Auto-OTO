from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from typing import Any


TARGET_PARAM_KEYS = (
    "target_offset_ms",
    "target_preutterance_ms",
    "target_overlap_ms",
    "target_consonant_ms",
    "target_cutoff_abs_ms",
)
ERROR_PARAM_KEYS = ("offset", "preutterance", "overlap", "consonant", "cutoff_abs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build voicebank-specific OTO CRNN diagnostics.")
    parser.add_argument("--manifest", required=True, help="OTO manifest JSONL.")
    parser.add_argument("--voicebank-id", required=True)
    parser.add_argument("--format-type", default="")
    parser.add_argument("--eval", default="", help="Optional evaluate_oto.py JSON to attach prediction errors.")
    parser.add_argument("--out", required=True, help="Diagnostic JSON output path.")
    parser.add_argument("--manifest-out", default="", help="Filtered diagnostic manifest JSONL output path.")
    parser.add_argument("--eval-manifest-out", default="", help="Filtered manifest rows matched to eval rows.")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--top-rows", type=int, default=20)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    eval_rows = read_eval_rows(args.eval) if args.eval else []
    result = analyze_voicebank(
        manifest_rows,
        eval_rows=eval_rows,
        voicebank_id=str(args.voicebank_id),
        format_type=str(args.format_type or ""),
        top_rows=int(args.top_rows),
    )
    result["source_manifest"] = os.path.abspath(args.manifest)
    if args.eval:
        result["source_eval"] = os.path.abspath(args.eval)

    compact = {key: value for key, value in result.items() if not key.endswith("_rows")}
    write_json(args.out, compact)
    if args.manifest_out:
        write_jsonl(args.manifest_out, result["diagnostic_manifest_rows"])
    if args.eval_manifest_out:
        write_jsonl(args.eval_manifest_out, result["diagnostic_eval_manifest_rows"])
    if args.markdown_out:
        write_text(args.markdown_out, render_markdown_report(result))

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


def analyze_voicebank(
    manifest_rows: list[dict[str, Any]],
    *,
    eval_rows: list[dict[str, Any]] | None = None,
    voicebank_id: str,
    format_type: str = "",
    top_rows: int = 20,
) -> dict[str, Any]:
    selected = [
        row
        for row in manifest_rows
        if str(row.get("voicebank_id", "") or "") == voicebank_id
        and (not format_type or str(row.get("format_type", "") or "").lower() == format_type.lower())
    ]
    eval_buckets = _eval_buckets(eval_rows or [], voicebank_id=voicebank_id, format_type=format_type)
    attached: list[dict[str, Any]] = []
    matched_manifest: list[dict[str, Any]] = []
    for row in selected:
        diagnostic = _compact_manifest_row(row)
        bucket = eval_buckets.get(_row_match_key(row))
        if bucket:
            diagnostic["eval"] = bucket.popleft()
            matched_manifest.append(row)
        attached.append(diagnostic)

    contexts = _group_rows(attached, lambda row: _context_key(row))
    row_buckets = _group_rows(attached, lambda row: _row_ratio_bucket(row.get("row_ratio_in_wav")))
    audio_groups = _group_rows(attached, lambda row: str(row.get("audio", "") or ""))
    alias_roles = _group_rows(attached, lambda row: _role_key(row))

    abnormal = [row for row in attached if _is_abnormal_row(row)]
    top_error_rows = sorted([row for row in attached if row.get("eval")], key=_eval_error_score, reverse=True)[:top_rows]
    top_order_mismatch_rows = sorted(attached, key=lambda row: abs(float(row.get("offset_vs_row_ratio_ms", 0.0) or 0.0)), reverse=True)[:top_rows]

    return {
        "voicebank_id": voicebank_id,
        "format_type": format_type or "all",
        "manifest_rows": len(selected),
        "eval_matched_rows": sum(1 for row in attached if row.get("eval")),
        "diagnostic_manifest_rows": selected,
        "diagnostic_eval_manifest_rows": matched_manifest,
        "target_summary": _summarize_manifest_rows(attached),
        "eval_error_summary": _summarize_eval_errors(attached),
        "row_order_summary": _summarize_row_order(audio_groups),
        "by_context": _summarize_groups(contexts),
        "by_row_ratio_bucket": _summarize_groups(row_buckets),
        "by_alias_role": _summarize_groups(alias_roles),
        "abnormal_rows": [_example_row(row) for row in abnormal[:top_rows]],
        "top_eval_error_rows": [_example_row(row) for row in top_error_rows],
        "top_order_mismatch_rows": [_example_row(row) for row in top_order_mismatch_rows],
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    lines = [
        f"# OTO CRNN Voicebank Diagnostics: {result.get('voicebank_id', '')}",
        "",
        f"- Format: `{result.get('format_type', 'all')}`",
        f"- Manifest rows: {result.get('manifest_rows', 0)}",
        f"- Eval matched rows: {result.get('eval_matched_rows', 0)}",
        f"- Source manifest: `{result.get('source_manifest', '')}`",
    ]
    if result.get("source_eval"):
        lines.append(f"- Source eval: `{result.get('source_eval', '')}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            _render_summary_table(result.get("target_summary") or {}, result.get("eval_error_summary") or {}),
            "",
            "## Row Order",
            "",
            _render_order_summary(result.get("row_order_summary") or {}),
            "",
            "## Contexts",
            "",
            _render_group_table(result.get("by_context") or []),
            "",
            "## Row Ratio Buckets",
            "",
            _render_group_table(result.get("by_row_ratio_bucket") or []),
            "",
            "## Alias Roles",
            "",
            _render_group_table(result.get("by_alias_role") or []),
            "",
            "## Worst Eval Rows",
            "",
            _render_example_table(result.get("top_eval_error_rows") or []),
            "",
            "## Largest Row-Order Mismatch Rows",
            "",
            _render_example_table(result.get("top_order_mismatch_rows") or []),
            "",
            "## Abnormal Rows",
            "",
            _render_example_table(result.get("abnormal_rows") or []),
            "",
        ]
    )
    return "\n".join(lines)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_eval_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("files") or [])


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _compact_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    duration = float(row.get("duration_ms", 0.0) or 0.0)
    ratio = float(row.get("row_ratio_in_wav", 0.0) or 0.0)
    target_offset = float(row.get("target_offset_ms", row.get("anchor_offset_ms", 0.0)) or 0.0)
    expected_by_ratio = duration * ratio
    return {
        "audio": str(row.get("audio", "") or ""),
        "wav": str(row.get("wav", "") or ""),
        "oto_path": str(row.get("oto_path", "") or ""),
        "alias": str(row.get("alias", "") or ""),
        "voicebank_id": str(row.get("voicebank_id", "") or ""),
        "language": str(row.get("language", "") or ""),
        "format_type": str(row.get("format_type", "") or ""),
        "alias_type": str(row.get("alias_type", "") or "other"),
        "transition_type": str(row.get("transition_type", "") or "other"),
        "alias_role": str(row.get("alias_role", "") or "other"),
        "is_diphthong": bool(row.get("is_diphthong", False)),
        "is_special": bool(row.get("is_special", False)),
        "row_index_in_wav": int(row.get("row_index_in_wav", 0) or 0),
        "file_row_count": int(row.get("file_row_count", 0) or 0),
        "row_ratio_in_wav": ratio,
        "duration_ms": duration,
        "target_offset_ms": target_offset,
        "target_preutterance_ms": float(row.get("target_preutterance_ms", 0.0) or 0.0),
        "target_overlap_ms": float(row.get("target_overlap_ms", 0.0) or 0.0),
        "target_consonant_ms": float(row.get("target_consonant_ms", 0.0) or 0.0),
        "target_cutoff_abs_ms": float(row.get("target_cutoff_abs_ms", 0.0) or 0.0),
        "offset_vs_row_ratio_ms": target_offset - expected_by_ratio,
        "quality_reason": str(row.get("quality_reason", "") or ""),
    }


def _eval_buckets(
    eval_rows: list[dict[str, Any]],
    *,
    voicebank_id: str,
    format_type: str,
) -> dict[tuple[str, str], deque[dict[str, Any]]]:
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in eval_rows:
        if str(row.get("voicebank_id", "") or "") != voicebank_id:
            continue
        if format_type and str(row.get("format_type", "") or "").lower() != format_type.lower():
            continue
        buckets[_row_match_key(row)].append(_compact_eval_row(row))
    return dict(buckets)


def _compact_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "predicted_error_ms": (
            float(row.get("predicted_error_ms")) if row.get("predicted_error_ms") is not None else None
        ),
        "low_confidence": bool(row.get("low_confidence", False)),
        "hard_failure": bool(row.get("hard_failure", False)),
        "hard_failure_reason": str(row.get("hard_failure_reason", "") or ""),
        "onset_lag_ms": float(row.get("onset_lag_ms", 0.0) or 0.0),
        "syllable_shift_flag": bool(row.get("syllable_shift_flag", False)),
        "param_abs_errors_ms": dict(row.get("param_abs_errors_ms") or {}),
        "anchor_abs_errors_ms": dict(row.get("anchor_abs_errors_ms") or {}),
    }


def _row_match_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("audio", "") or ""), str(row.get("alias", "") or ""))


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


def _group_rows(rows: list[dict[str, Any]], key_fn) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return dict(grouped)


def _summarize_groups(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for key, rows in groups.items():
        summary = {"key": key, "files": len(rows)}
        summary.update(_summarize_manifest_rows(rows))
        summary.update(_summarize_eval_errors(rows))
        summary["score"] = _diagnostic_score(summary)
        out.append(summary)
    out.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    return out


def _summarize_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in TARGET_PARAM_KEYS:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        out[key] = _distribution(values)
    out["offset_vs_row_ratio_ms"] = _distribution(
        [float(row.get("offset_vs_row_ratio_ms", 0.0) or 0.0) for row in rows]
    )
    out["duration_ms"] = _distribution([float(row.get("duration_ms", 0.0) or 0.0) for row in rows])
    out["blank_alias_rate"] = _ratio([not str(row.get("alias", "") or "").strip() for row in rows])
    out["non_ok_quality_rate"] = _ratio([str(row.get("quality_reason", "") or "ok") != "ok" for row in rows])
    return out


def _summarize_eval_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [row for row in rows if row.get("eval")]
    out: dict[str, Any] = {"eval_matched_rows": len(matched)}
    for key in ERROR_PARAM_KEYS:
        values = [
            float(((row.get("eval") or {}).get("param_abs_errors_ms") or {}).get(key, 0.0) or 0.0)
            for row in matched
        ]
        out[f"eval_{key}_abs_error_ms"] = _distribution(values)
    out["eval_hard_failure_rate"] = _ratio([bool((row.get("eval") or {}).get("hard_failure", False)) for row in matched])
    out["eval_low_confidence_rate"] = _ratio([bool((row.get("eval") or {}).get("low_confidence", False)) for row in matched])
    out["eval_preutterance_acc_50ms"] = _ratio(
        [
            float(((row.get("eval") or {}).get("param_abs_errors_ms") or {}).get("preutterance", 0.0) or 0.0)
            <= 50.0
            for row in matched
        ]
    )
    return out


def _summarize_row_order(audio_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    files = 0
    rows = 0
    monotonic_violations = 0
    negative_steps: list[float] = []
    abs_ratio_offsets: list[float] = []
    for group in audio_groups.values():
        ordered = sorted(group, key=lambda row: int(row.get("row_index_in_wav", 0) or 0))
        if not ordered:
            continue
        files += 1
        rows += len(ordered)
        previous = None
        for row in ordered:
            abs_ratio_offsets.append(abs(float(row.get("offset_vs_row_ratio_ms", 0.0) or 0.0)))
            current = float(row.get("target_offset_ms", 0.0) or 0.0)
            if previous is not None and current + 1.0 < previous:
                monotonic_violations += 1
                negative_steps.append(current - previous)
            previous = current
    return {
        "files": files,
        "rows": rows,
        "monotonic_violations": monotonic_violations,
        "monotonic_violation_rate": float(monotonic_violations / max(1, rows)),
        "negative_step_ms": _distribution(negative_steps),
        "abs_offset_vs_row_ratio_ms": _distribution(abs_ratio_offsets),
    }


def _row_ratio_bucket(value: Any) -> str:
    ratio = max(0.0, min(0.999999, float(value or 0.0)))
    start = int(ratio * 10) * 10
    return f"{start:02d}-{start + 10:02d}%"


def _is_abnormal_row(row: dict[str, Any]) -> bool:
    duration = float(row.get("duration_ms", 0.0) or 0.0)
    alias = str(row.get("alias", "") or "").strip()
    return (
        not alias
        or str(row.get("quality_reason", "") or "ok") != "ok"
        or float(row.get("target_offset_ms", 0.0) or 0.0) < -1.0
        or float(row.get("target_offset_ms", 0.0) or 0.0) > duration + 1.0
        or float(row.get("target_cutoff_abs_ms", 0.0) or 0.0) < -1.0
        or float(row.get("target_cutoff_abs_ms", 0.0) or 0.0) > duration + 1.0
    )


def _diagnostic_score(row: dict[str, Any]) -> float:
    cutoff = float((row.get("eval_cutoff_abs_abs_error_ms") or {}).get("mean", 0.0) or 0.0)
    offset = float((row.get("eval_offset_abs_error_ms") or {}).get("mean", 0.0) or 0.0)
    pre = float((row.get("eval_preutterance_abs_error_ms") or {}).get("mean", 0.0) or 0.0)
    ratio = float((row.get("offset_vs_row_ratio_ms") or {}).get("abs_mean", 0.0) or 0.0)
    return cutoff * 0.35 + offset * 0.25 + pre * 0.20 + ratio * 0.20


def _eval_error_score(row: dict[str, Any]) -> float:
    errors = ((row.get("eval") or {}).get("param_abs_errors_ms") or {})
    return (
        float(errors.get("cutoff_abs", 0.0) or 0.0) * 0.35
        + float(errors.get("offset", 0.0) or 0.0) * 0.25
        + float(errors.get("preutterance", 0.0) or 0.0) * 0.20
        + float(errors.get("consonant", 0.0) or 0.0) * 0.20
    )


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    errors = ((row.get("eval") or {}).get("param_abs_errors_ms") or {})
    return {
        "audio": row.get("audio", ""),
        "wav": row.get("wav", ""),
        "alias": row.get("alias", ""),
        "context": _context_key(row),
        "alias_role": row.get("alias_role", ""),
        "row_index_in_wav": row.get("row_index_in_wav", 0),
        "file_row_count": row.get("file_row_count", 0),
        "row_ratio_in_wav": row.get("row_ratio_in_wav", 0.0),
        "target_offset_ms": row.get("target_offset_ms", 0.0),
        "target_cutoff_abs_ms": row.get("target_cutoff_abs_ms", 0.0),
        "offset_vs_row_ratio_ms": row.get("offset_vs_row_ratio_ms", 0.0),
        "quality_reason": row.get("quality_reason", ""),
        "eval_offset_error_ms": errors.get("offset"),
        "eval_preutterance_error_ms": errors.get("preutterance"),
        "eval_cutoff_abs_error_ms": errors.get("cutoff_abs"),
        "eval_hard_failure": (row.get("eval") or {}).get("hard_failure"),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {"count": 0, "mean": None, "abs_mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": float(sum(clean) / len(clean)),
        "abs_mean": float(sum(abs(value) for value in clean) / len(clean)),
        "p50": _percentile(clean, 50.0),
        "p90": _percentile(clean, 90.0),
        "min": float(clean[0]),
        "max": float(clean[-1]),
    }


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (pct / 100.0)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return float(sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac)


def _ratio(values: list[bool]) -> float:
    return float(sum(1 for value in values if value) / len(values)) if values else 0.0


def _render_summary_table(target: dict[str, Any], errors: dict[str, Any]) -> str:
    lines = ["| metric | count | mean | abs_mean | p50 | p90 | min | max |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in list(TARGET_PARAM_KEYS) + ["offset_vs_row_ratio_ms", "duration_ms"]:
        lines.append(_render_distribution_row(key, target.get(key) or {}))
    for key in [f"eval_{name}_abs_error_ms" for name in ERROR_PARAM_KEYS]:
        lines.append(_render_distribution_row(key, errors.get(key) or {}))
    lines.extend(
        [
            f"| eval_preutterance_acc_50ms | {errors.get('eval_matched_rows', 0)} | {errors.get('eval_preutterance_acc_50ms', 0.0):.4f} |  |  |  |  |  |",
            f"| eval_hard_failure_rate | {errors.get('eval_matched_rows', 0)} | {errors.get('eval_hard_failure_rate', 0.0):.4f} |  |  |  |  |  |",
            f"| eval_low_confidence_rate | {errors.get('eval_matched_rows', 0)} | {errors.get('eval_low_confidence_rate', 0.0):.4f} |  |  |  |  |  |",
        ]
    )
    return "\n".join(lines)


def _render_order_summary(order: dict[str, Any]) -> str:
    lines = [
        f"- Files: {order.get('files', 0)}",
        f"- Rows: {order.get('rows', 0)}",
        f"- Monotonic violations: {order.get('monotonic_violations', 0)}",
        f"- Monotonic violation rate: {float(order.get('monotonic_violation_rate', 0.0) or 0.0):.4f}",
        "",
        "| metric | count | mean | abs_mean | p50 | p90 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _render_distribution_row("negative_step_ms", order.get("negative_step_ms") or {}),
        _render_distribution_row("abs_offset_vs_row_ratio_ms", order.get("abs_offset_vs_row_ratio_ms") or {}),
    ]
    return "\n".join(lines)


def _render_group_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| key | files | matched | score | offset_ratio_abs_mean | offset_mean | cutoff_mean | pre_err | offset_err | cutoff_err | pre_acc50 | hard_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{key}` | {files} | {matched} | {score:.2f} | {ratio:.2f} | {offset:.2f} | {cutoff:.2f} | "
            "{preerr:.2f} | {offerr:.2f} | {cuterr:.2f} | {preacc:.4f} | {hard:.4f} |".format(
                key=row.get("key", ""),
                files=row.get("files", 0),
                matched=row.get("eval_matched_rows", 0),
                score=float(row.get("score", 0.0) or 0.0),
                ratio=_mean_from(row, "offset_vs_row_ratio_ms", "abs_mean"),
                offset=_mean_from(row, "target_offset_ms"),
                cutoff=_mean_from(row, "target_cutoff_abs_ms"),
                preerr=_mean_from(row, "eval_preutterance_abs_error_ms"),
                offerr=_mean_from(row, "eval_offset_abs_error_ms"),
                cuterr=_mean_from(row, "eval_cutoff_abs_abs_error_ms"),
                preacc=float(row.get("eval_preutterance_acc_50ms", 0.0) or 0.0),
                hard=float(row.get("eval_hard_failure_rate", 0.0) or 0.0),
            )
        )
    return "\n".join(lines)


def _render_example_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| alias | context | row | ratio | target_offset | target_cutoff | offset_vs_ratio | pre_err | offset_err | cutoff_err | hard | quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{alias}` | `{context}` | {row_index_in_wav}/{file_row_count} | {row_ratio_in_wav:.4f} | "
            "{target_offset_ms:.2f} | {target_cutoff_abs_ms:.2f} | {offset_vs_row_ratio_ms:.2f} | "
            "{pre} | {off} | {cut} | {hard} | `{quality_reason}` |".format(
                alias=row.get("alias", ""),
                context=row.get("context", ""),
                row_index_in_wav=row.get("row_index_in_wav", 0),
                file_row_count=row.get("file_row_count", 0),
                row_ratio_in_wav=float(row.get("row_ratio_in_wav", 0.0) or 0.0),
                target_offset_ms=float(row.get("target_offset_ms", 0.0) or 0.0),
                target_cutoff_abs_ms=float(row.get("target_cutoff_abs_ms", 0.0) or 0.0),
                offset_vs_row_ratio_ms=float(row.get("offset_vs_row_ratio_ms", 0.0) or 0.0),
                pre=_fmt_nullable(row.get("eval_preutterance_error_ms")),
                off=_fmt_nullable(row.get("eval_offset_error_ms")),
                cut=_fmt_nullable(row.get("eval_cutoff_abs_error_ms")),
                hard=row.get("eval_hard_failure"),
                quality_reason=row.get("quality_reason", ""),
            )
        )
    return "\n".join(lines)


def _render_distribution_row(name: str, dist: dict[str, Any]) -> str:
    return (
        f"| {name} | {dist.get('count', 0)} | {_fmt_nullable(dist.get('mean'))} | {_fmt_nullable(dist.get('abs_mean'))} | "
        f"{_fmt_nullable(dist.get('p50'))} | {_fmt_nullable(dist.get('p90'))} | {_fmt_nullable(dist.get('min'))} | {_fmt_nullable(dist.get('max'))} |"
    )


def _mean_from(row: dict[str, Any], key: str, stat: str = "mean") -> float:
    return float((row.get(key) or {}).get(stat, 0.0) or 0.0)


def _fmt_nullable(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
