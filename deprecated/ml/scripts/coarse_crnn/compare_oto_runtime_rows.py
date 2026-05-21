from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare per-row CRNN runtime eval JSONs and find practical regressions/improvements."
    )
    parser.add_argument("--baseline", required=True, help="Baseline evaluate_oto_runtime.py JSON.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate JSON. Repeatable. Use label=path or just path.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--pre-threshold-ms", type=float, default=50.0)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--focus-top",
        type=int,
        default=30,
        help="Top N rows for baseline/candidate miss focus breakdowns.",
    )
    args = parser.parse_args()

    baseline = _load_eval(args.baseline)
    baseline_rows = _rows_by_key(baseline)
    comparisons = []
    for spec in args.candidate:
        label, path = _split_candidate_spec(spec)
        candidate = _load_eval(path)
        comparison = _compare_one(
            baseline_rows,
            _rows_by_key(candidate),
            label=label,
            path=path,
            pre_threshold_ms=float(args.pre_threshold_ms),
            top_n=max(1, int(args.top)),
            focus_top_n=max(1, int(args.focus_top)),
        )
        comparisons.append(comparison)

    payload = {
        "baseline": os.path.abspath(args.baseline),
        "pre_threshold_ms": float(args.pre_threshold_ms),
        "comparisons": comparisons,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(_summary_for_console(payload), ensure_ascii=False, indent=2))
    return 0


def _load_eval(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("files"), list):
        raise ValueError(f"eval JSON has no files list: {path}")
    return payload


def _split_candidate_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip() or os.path.basename(path), path.strip()
    return os.path.splitext(os.path.basename(spec))[0], spec


def _row_key(row: dict[str, Any]) -> str:
    return "\t".join(
        [
            os.path.normcase(os.path.abspath(str(row.get("audio", "") or ""))),
            str(row.get("alias", "") or ""),
            str(int(row.get("row_index_in_wav", 0) or 0)),
            str(int(row.get("file_row_count", 1) or 1)),
        ]
    )


def _rows_by_key(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("files", []):
        rows[_row_key(row)] = row
    return rows


def _compare_one(
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    *,
    label: str,
    path: str,
    pre_threshold_ms: float,
    top_n: int,
    focus_top_n: int,
) -> dict[str, Any]:
    common_keys = sorted(set(baseline_rows) & set(candidate_rows))
    missing_in_candidate = sorted(set(baseline_rows) - set(candidate_rows))
    extra_in_candidate = sorted(set(candidate_rows) - set(baseline_rows))
    counts = defaultdict(int)
    by_group: dict[str, dict[str, Any]] = {}
    rows_out = {
        "baseline_hit_candidate_miss": [],
        "candidate_hit_baseline_miss": [],
        "hard_failure_regressions": [],
        "hard_failure_improvements": [],
        "worst_pre_regressions": [],
        "best_pre_improvements": [],
    }

    pre_deltas: list[float] = []
    offset_deltas: list[float] = []
    hard_delta = 0
    for key in common_keys:
        base = baseline_rows[key]
        cand = candidate_rows[key]
        b_pre = _runtime_error(base, "preutterance")
        c_pre = _runtime_error(cand, "preutterance")
        b_offset = _runtime_error(base, "offset")
        c_offset = _runtime_error(cand, "offset")
        b_hard = bool((base.get("runtime") or {}).get("hard_failure", False))
        c_hard = bool((cand.get("runtime") or {}).get("hard_failure", False))
        b_hit = b_pre <= pre_threshold_ms
        c_hit = c_pre <= pre_threshold_ms
        pre_delta = c_pre - b_pre
        offset_delta = c_offset - b_offset
        pre_deltas.append(pre_delta)
        offset_deltas.append(offset_delta)
        hard_delta += int(c_hard) - int(b_hard)

        if b_hit and c_hit:
            counts["both_hit"] += 1
        elif (not b_hit) and (not c_hit):
            counts["both_miss"] += 1
        elif b_hit and not c_hit:
            counts["baseline_hit_candidate_miss"] += 1
            rows_out["baseline_hit_candidate_miss"].append(_row_summary(base, cand, pre_delta, offset_delta))
        else:
            counts["candidate_hit_baseline_miss"] += 1
            rows_out["candidate_hit_baseline_miss"].append(_row_summary(base, cand, pre_delta, offset_delta))

        if not b_hard and c_hard:
            counts["hard_failure_regressions"] += 1
            rows_out["hard_failure_regressions"].append(_row_summary(base, cand, pre_delta, offset_delta))
        elif b_hard and not c_hard:
            counts["hard_failure_improvements"] += 1
            rows_out["hard_failure_improvements"].append(_row_summary(base, cand, pre_delta, offset_delta))

        rows_out["worst_pre_regressions"].append(_row_summary(base, cand, pre_delta, offset_delta))
        rows_out["best_pre_improvements"].append(_row_summary(base, cand, pre_delta, offset_delta))
        group = _group_key(base)
        group_stats = by_group.setdefault(
            group,
            {
                "rows": 0,
                "baseline_hits": 0,
                "candidate_hits": 0,
                "baseline_hard": 0,
                "candidate_hard": 0,
                "pre_delta_sum": 0.0,
                "offset_delta_sum": 0.0,
            },
        )
        group_stats["rows"] += 1
        group_stats["baseline_hits"] += int(b_hit)
        group_stats["candidate_hits"] += int(c_hit)
        group_stats["baseline_hard"] += int(b_hard)
        group_stats["candidate_hard"] += int(c_hard)
        group_stats["pre_delta_sum"] += float(pre_delta)
        group_stats["offset_delta_sum"] += float(offset_delta)

    rows_out["baseline_hit_candidate_miss"].sort(key=lambda row: row["pre_delta_ms"], reverse=True)
    rows_out["candidate_hit_baseline_miss"].sort(key=lambda row: row["pre_delta_ms"])
    rows_out["hard_failure_regressions"].sort(key=lambda row: row["candidate_pre_error_ms"], reverse=True)
    rows_out["hard_failure_improvements"].sort(key=lambda row: row["baseline_pre_error_ms"], reverse=True)
    rows_out["worst_pre_regressions"].sort(key=lambda row: row["pre_delta_ms"], reverse=True)
    rows_out["best_pre_improvements"].sort(key=lambda row: row["pre_delta_ms"])
    for name in list(rows_out):
        rows_out[name] = rows_out[name][:top_n]

    group_rows = []
    for group, stats in by_group.items():
        n = max(1, int(stats["rows"]))
        group_rows.append(
            {
                "group": group,
                "rows": int(stats["rows"]),
                "baseline_pre_acc50": float(stats["baseline_hits"]) / n,
                "candidate_pre_acc50": float(stats["candidate_hits"]) / n,
                "pre_acc50_delta": (float(stats["candidate_hits"]) - float(stats["baseline_hits"])) / n,
                "baseline_hard_failure_rate": float(stats["baseline_hard"]) / n,
                "candidate_hard_failure_rate": float(stats["candidate_hard"]) / n,
                "hard_failure_delta": (float(stats["candidate_hard"]) - float(stats["baseline_hard"])) / n,
                "mean_pre_error_delta_ms": float(stats["pre_delta_sum"]) / n,
                "mean_offset_error_delta_ms": float(stats["offset_delta_sum"]) / n,
            }
        )
    group_rows.sort(key=lambda row: (row["pre_acc50_delta"], -row["rows"]))

    total = max(1, len(common_keys))
    baseline_loss_focus = _focus_breakdown(rows_out["baseline_hit_candidate_miss"], top_n=focus_top_n)
    candidate_gain_focus = _focus_breakdown(rows_out["candidate_hit_baseline_miss"], top_n=focus_top_n)
    return {
        "label": label,
        "path": os.path.abspath(path),
        "rows_compared": len(common_keys),
        "missing_in_candidate": len(missing_in_candidate),
        "extra_in_candidate": len(extra_in_candidate),
        "counts": dict(counts),
        "baseline_pre_acc50": (counts["both_hit"] + counts["baseline_hit_candidate_miss"]) / total,
        "candidate_pre_acc50": (counts["both_hit"] + counts["candidate_hit_baseline_miss"]) / total,
        "pre_acc50_delta": (
            (counts["both_hit"] + counts["candidate_hit_baseline_miss"])
            - (counts["both_hit"] + counts["baseline_hit_candidate_miss"])
        )
        / total,
        "mean_pre_error_delta_ms": _mean(pre_deltas),
        "mean_offset_error_delta_ms": _mean(offset_deltas),
        "hard_failure_delta_count": int(hard_delta),
        "worst_groups_by_pre_acc_delta": group_rows[:20],
        "best_groups_by_pre_acc_delta": list(reversed(group_rows[-20:])),
        "focus_breakdown": {
            "baseline_hit_candidate_miss": baseline_loss_focus,
            "candidate_hit_baseline_miss": candidate_gain_focus,
        },
        "rows": rows_out,
    }


def _runtime_error(row: dict[str, Any], name: str) -> float:
    runtime = row.get("runtime") or {}
    errors = runtime.get("param_abs_errors_ms") or runtime.get("anchor_abs_errors_ms") or {}
    return float(errors.get(name, 0.0) or 0.0)


def _group_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("language", "") or "other"),
            str(row.get("format_type", "") or "other"),
            str(row.get("alias_role", "") or row.get("alias_type", "") or "other"),
        ]
    )


def _row_summary(
    base: dict[str, Any],
    cand: dict[str, Any],
    pre_delta: float,
    offset_delta: float,
) -> dict[str, Any]:
    return {
        "audio": str(base.get("audio", "") or ""),
        "alias": str(base.get("alias", "") or ""),
        "voicebank_id": str(base.get("voicebank_id", "") or ""),
        "group": _group_key(base),
        "row_index_in_wav": int(base.get("row_index_in_wav", 0) or 0),
        "file_row_count": int(base.get("file_row_count", 1) or 1),
        "duration_ms": float(base.get("duration_ms", 0.0) or 0.0),
        "baseline_pre_error_ms": _runtime_error(base, "preutterance"),
        "candidate_pre_error_ms": _runtime_error(cand, "preutterance"),
        "pre_delta_ms": float(pre_delta),
        "baseline_offset_error_ms": _runtime_error(base, "offset"),
        "candidate_offset_error_ms": _runtime_error(cand, "offset"),
        "offset_delta_ms": float(offset_delta),
        "baseline_hard_failure": bool((base.get("runtime") or {}).get("hard_failure", False)),
        "candidate_hard_failure": bool((cand.get("runtime") or {}).get("hard_failure", False)),
        "baseline_confidence": float(base.get("confidence", 0.0) or 0.0),
        "candidate_confidence": float(cand.get("confidence", 0.0) or 0.0),
        "baseline_skipped_audio_ms": float(((base.get("confidence_components") or {}).get("skipped_audio_ms", 0.0)) or 0.0),
        "candidate_skipped_audio_ms": float(((cand.get("confidence_components") or {}).get("skipped_audio_ms", 0.0)) or 0.0),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _focus_breakdown(rows: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "\t".join(
            [
                str(row.get("voicebank_id", "") or "unknown_voicebank"),
                str(row.get("group", "") or "other|other|other"),
            ]
        )
        slot = by_key.setdefault(
            key,
            {
                "voicebank_id": str(row.get("voicebank_id", "") or "unknown_voicebank"),
                "group": str(row.get("group", "") or "other|other|other"),
                "count": 0,
                "pre_delta_sum": 0.0,
                "offset_delta_sum": 0.0,
                "baseline_pre_error_sum": 0.0,
                "candidate_pre_error_sum": 0.0,
                "baseline_confidence_sum": 0.0,
                "candidate_confidence_sum": 0.0,
                "sample_aliases": [],
            },
        )
        slot["count"] += 1
        slot["pre_delta_sum"] += float(row.get("pre_delta_ms", 0.0) or 0.0)
        slot["offset_delta_sum"] += float(row.get("offset_delta_ms", 0.0) or 0.0)
        slot["baseline_pre_error_sum"] += float(row.get("baseline_pre_error_ms", 0.0) or 0.0)
        slot["candidate_pre_error_sum"] += float(row.get("candidate_pre_error_ms", 0.0) or 0.0)
        slot["baseline_confidence_sum"] += float(row.get("baseline_confidence", 0.0) or 0.0)
        slot["candidate_confidence_sum"] += float(row.get("candidate_confidence", 0.0) or 0.0)
        aliases: list[str] = slot["sample_aliases"]
        alias = str(row.get("alias", "") or "")
        if alias and alias not in aliases and len(aliases) < 5:
            aliases.append(alias)

    out = []
    for item in by_key.values():
        n = max(1, int(item["count"]))
        out.append(
            {
                "voicebank_id": item["voicebank_id"],
                "group": item["group"],
                "count": int(item["count"]),
                "mean_pre_delta_ms": float(item["pre_delta_sum"]) / n,
                "mean_offset_delta_ms": float(item["offset_delta_sum"]) / n,
                "mean_baseline_pre_error_ms": float(item["baseline_pre_error_sum"]) / n,
                "mean_candidate_pre_error_ms": float(item["candidate_pre_error_sum"]) / n,
                "mean_baseline_confidence": float(item["baseline_confidence_sum"]) / n,
                "mean_candidate_confidence": float(item["candidate_confidence_sum"]) / n,
                "sample_aliases": list(item["sample_aliases"]),
            }
        )
    out.sort(key=lambda row: (-int(row["count"]), -float(row["mean_pre_delta_ms"])))
    return out[:top_n]


def _summary_for_console(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": payload["baseline"],
        "comparisons": [
            {
                "label": item["label"],
                "rows_compared": item["rows_compared"],
                "baseline_pre_acc50": item["baseline_pre_acc50"],
                "candidate_pre_acc50": item["candidate_pre_acc50"],
                "pre_acc50_delta": item["pre_acc50_delta"],
                "mean_pre_error_delta_ms": item["mean_pre_error_delta_ms"],
                "hard_failure_delta_count": item["hard_failure_delta_count"],
                "counts": item["counts"],
                "worst_groups_by_pre_acc_delta": item["worst_groups_by_pre_acc_delta"][:5],
                "top_baseline_loss_focus": (item.get("focus_breakdown") or {}).get("baseline_hit_candidate_miss", [])[:5],
            }
            for item in payload["comparisons"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
