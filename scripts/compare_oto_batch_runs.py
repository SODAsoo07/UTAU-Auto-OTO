import argparse
import csv
import json
import os
from typing import Dict, List


def _load_summary(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid summary json: {path}")
    return data


def _normalize_output_key(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    return os.path.normcase(os.path.normpath(raw))


def _case_map(summary: Dict[str, object], summary_label: str = "summary") -> Dict[str, Dict[str, object]]:
    results = summary.get("results") or []
    out: Dict[str, Dict[str, object]] = {}
    seen_outputs: Dict[str, str] = {}
    if not isinstance(results, list):
        return out
    for idx, row in enumerate(results, start=1):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        if name in out:
            raise ValueError(f"{summary_label}: duplicate case name: {name}")
        output_key = _normalize_output_key(str(row.get("output_oto", "") or ""))
        if output_key:
            prev_name = seen_outputs.get(output_key)
            if prev_name and prev_name != name:
                raise ValueError(
                    f"{summary_label}: duplicate output_oto path: {row.get('output_oto', '')} "
                    f"({prev_name}, {name})"
                )
            seen_outputs[output_key] = name
        row = dict(row)
        row["_summary_index"] = idx
        out[name] = row
    return out


def _get_validation_metrics(case: Dict[str, object]) -> Dict[str, int]:
    validation = case.get("validation") or {}
    if not isinstance(validation, dict):
        validation = {}
    return {
        "warnings": int(validation.get("warnings", 0) or 0),
        "errors": int(validation.get("errors", 0) or 0),
        "checked_files": int(validation.get("checked_files", 0) or 0),
        "total_lines": int(validation.get("total_lines", 0) or 0),
    }


def _compare_cases(base_case: Dict[str, object], new_case: Dict[str, object]) -> Dict[str, object]:
    base_status = str(base_case.get("status", "missing"))
    new_status = str(new_case.get("status", "missing"))
    base_val = _get_validation_metrics(base_case)
    new_val = _get_validation_metrics(new_case)

    return {
        "status_base": base_status,
        "status_new": new_status,
        "status_changed": base_status != new_status,
        "warnings_base": base_val["warnings"],
        "warnings_new": new_val["warnings"],
        "warnings_delta": new_val["warnings"] - base_val["warnings"],
        "errors_base": base_val["errors"],
        "errors_new": new_val["errors"],
        "errors_delta": new_val["errors"] - base_val["errors"],
        "checked_files_base": base_val["checked_files"],
        "checked_files_new": new_val["checked_files"],
        "total_lines_base": base_val["total_lines"],
        "total_lines_new": new_val["total_lines"],
    }


def _aggregate(case_diffs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    names = sorted(case_diffs.keys())
    status_changed = sum(1 for name in names if case_diffs[name].get("status_changed"))
    warnings_delta_sum = sum(int(case_diffs[name].get("warnings_delta", 0) or 0) for name in names)
    errors_delta_sum = sum(int(case_diffs[name].get("errors_delta", 0) or 0) for name in names)
    regressions = [
        name for name in names
        if int(case_diffs[name].get("errors_delta", 0) or 0) > 0
        or str(case_diffs[name].get("status_base")) == "ok" and str(case_diffs[name].get("status_new")) != "ok"
    ]
    improvements = [
        name for name in names
        if int(case_diffs[name].get("errors_delta", 0) or 0) < 0
        or str(case_diffs[name].get("status_base")) != "ok" and str(case_diffs[name].get("status_new")) == "ok"
    ]
    return {
        "cases": len(names),
        "status_changed_cases": int(status_changed),
        "warnings_delta_sum": int(warnings_delta_sum),
        "errors_delta_sum": int(errors_delta_sum),
        "regression_cases": regressions,
        "improvement_cases": improvements,
    }


def _default_output_paths(base_summary: str, new_summary: str):
    new_dir = os.path.dirname(os.path.abspath(new_summary))
    base_tag = os.path.splitext(os.path.basename(base_summary))[0]
    new_tag = os.path.splitext(os.path.basename(new_summary))[0]
    tag = f"compare_{new_tag}_vs_{base_tag}"
    return (
        os.path.join(new_dir, f"{tag}.json"),
        os.path.join(new_dir, f"{tag}.csv"),
    )


def _write_csv(path: str, case_diffs: Dict[str, Dict[str, object]]) -> None:
    fieldnames = [
        "case_name",
        "status_base",
        "status_new",
        "status_changed",
        "warnings_base",
        "warnings_new",
        "warnings_delta",
        "errors_base",
        "errors_new",
        "errors_delta",
        "checked_files_base",
        "checked_files_new",
        "total_lines_base",
        "total_lines_new",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted(case_diffs.keys()):
            row = dict(case_diffs[name])
            row["case_name"] = name
            writer.writerow(row)


def compare_runs(base_summary: str, new_summary: str) -> Dict[str, object]:
    base_data = _load_summary(base_summary)
    new_data = _load_summary(new_summary)
    base_cases = _case_map(base_data, summary_label=os.path.basename(base_summary))
    new_cases = _case_map(new_data, summary_label=os.path.basename(new_summary))

    all_names = sorted(set(base_cases.keys()) | set(new_cases.keys()))
    case_diffs: Dict[str, Dict[str, object]] = {}
    for name in all_names:
        case_diffs[name] = _compare_cases(
            base_case=base_cases.get(name, {"status": "missing", "validation": {}}),
            new_case=new_cases.get(name, {"status": "missing", "validation": {}}),
        )

    aggregate = _aggregate(case_diffs)
    return {
        "base_summary": os.path.abspath(base_summary),
        "new_summary": os.path.abspath(new_summary),
        "aggregate": aggregate,
        "case_diffs": case_diffs,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare two oto_batch summary.json files.")
    ap.add_argument("--base-summary", required=True, help="Baseline summary.json path")
    ap.add_argument("--new-summary", required=True, help="New run summary.json path")
    ap.add_argument("--out-json", default="", help="Output json report path")
    ap.add_argument("--out-csv", default="", help="Output csv report path")
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 2 when regression is detected",
    )
    args = ap.parse_args()

    base_summary = os.path.abspath(args.base_summary)
    new_summary = os.path.abspath(args.new_summary)
    if not os.path.exists(base_summary):
        raise FileNotFoundError(base_summary)
    if not os.path.exists(new_summary):
        raise FileNotFoundError(new_summary)

    default_json, default_csv = _default_output_paths(base_summary, new_summary)
    out_json = os.path.abspath(args.out_json) if args.out_json else default_json
    out_csv = os.path.abspath(args.out_csv) if args.out_csv else default_csv

    report = compare_runs(base_summary=base_summary, new_summary=new_summary)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_csv(out_csv, report["case_diffs"])

    agg = report["aggregate"]
    print(
        "[CompareBatch] "
        f"cases={agg['cases']} "
        f"status_changed={agg['status_changed_cases']} "
        f"warnings_delta_sum={agg['warnings_delta_sum']} "
        f"errors_delta_sum={agg['errors_delta_sum']}"
    )
    print(f"[CompareBatch] json={out_json}")
    print(f"[CompareBatch] csv={out_csv}")

    regression_cases = list(agg.get("regression_cases") or [])
    if regression_cases:
        print(f"[CompareBatch] regressions={len(regression_cases)}")
        for name in regression_cases[:20]:
            print(f"  - {name}")
        if args.fail_on_regression:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

