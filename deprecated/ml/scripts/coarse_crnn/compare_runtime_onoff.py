from __future__ import annotations

import argparse
import json
import os
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize fallback ON/OFF runtime eval deltas for boundary-accuracy diagnosis."
    )
    parser.add_argument("--on", required=True, help="evaluate_oto_runtime result with --use-row-base-fallback")
    parser.add_argument("--off", required=True, help="evaluate_oto_runtime result with --no-use-row-base-fallback")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload_on = _load_json(args.on)
    payload_off = _load_json(args.off)
    label = str(args.label or "").strip() or _default_label(args.on, args.off)
    report = _build_report(payload_on, payload_off, label=label, on_path=args.on, off_path=args.off)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    return 0


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_label(on_path: str, off_path: str) -> str:
    on_name = os.path.splitext(os.path.basename(str(on_path)))[0]
    off_name = os.path.splitext(os.path.basename(str(off_path)))[0]
    return f"{on_name}__vs__{off_name}"


def _build_report(
    on_payload: dict[str, Any],
    off_payload: dict[str, Any],
    *,
    label: str,
    on_path: str,
    off_path: str,
) -> dict[str, Any]:
    runtime_on = dict(on_payload.get("runtime") or {})
    runtime_off = dict(off_payload.get("runtime") or {})
    counts_on = dict(on_payload.get("postprocess_counts") or {})
    counts_off = dict(off_payload.get("postprocess_counts") or {})

    metric_fields = (
        "preutterance_acc_20ms",
        "preutterance_acc_50ms",
        "hard_failure_rate",
    )
    mae_fields = ("offset", "consonant", "cutoff_abs", "preutterance", "overlap")

    delta_runtime = {
        key: _safe_float(runtime_off.get(key)) - _safe_float(runtime_on.get(key))
        for key in metric_fields
    }
    delta_mae = {
        key: _safe_float((runtime_off.get("param_mae_ms") or {}).get(key))
        - _safe_float((runtime_on.get("param_mae_ms") or {}).get(key))
        for key in mae_fields
    }
    delta_counts = {
        key: int(counts_off.get(key, 0) or 0) - int(counts_on.get(key, 0) or 0)
        for key in (
            "low_confidence_fallback",
            "runtime_moved",
            "candidate_snap",
            "vc_matcher",
            "activity_fallback",
            "right_guard_changed",
        )
    }

    role_keys = sorted(
        set((on_payload.get("by_language_format_role") or {}).keys())
        | set((off_payload.get("by_language_format_role") or {}).keys())
    )
    role_rows = []
    for role_key in role_keys:
        on_group = ((on_payload.get("by_language_format_role") or {}).get(role_key) or {})
        off_group = ((off_payload.get("by_language_format_role") or {}).get(role_key) or {})
        on_runtime = dict(on_group.get("runtime") or {})
        off_runtime = dict(off_group.get("runtime") or {})
        on_files = int(on_group.get("files", 0) or 0)
        off_files = int(off_group.get("files", 0) or 0)
        if max(on_files, off_files) <= 0:
            continue
        role_rows.append(
            {
                "group": role_key,
                "rows_on": on_files,
                "rows_off": off_files,
                "pre50_on": _safe_float(on_runtime.get("preutterance_acc_50ms")),
                "pre50_off": _safe_float(off_runtime.get("preutterance_acc_50ms")),
                "pre50_delta_off_minus_on": _safe_float(off_runtime.get("preutterance_acc_50ms"))
                - _safe_float(on_runtime.get("preutterance_acc_50ms")),
                "hard_on": _safe_float(on_runtime.get("hard_failure_rate")),
                "hard_off": _safe_float(off_runtime.get("hard_failure_rate")),
                "hard_delta_off_minus_on": _safe_float(off_runtime.get("hard_failure_rate"))
                - _safe_float(on_runtime.get("hard_failure_rate")),
                "pre_mae_on": _safe_float((on_runtime.get("param_mae_ms") or {}).get("preutterance")),
                "pre_mae_off": _safe_float((off_runtime.get("param_mae_ms") or {}).get("preutterance")),
                "pre_mae_delta_off_minus_on": _safe_float((off_runtime.get("param_mae_ms") or {}).get("preutterance"))
                - _safe_float((on_runtime.get("param_mae_ms") or {}).get("preutterance")),
            }
        )
    role_rows.sort(key=lambda row: (row["pre50_delta_off_minus_on"], -max(row["rows_on"], row["rows_off"])))

    fallback_dependency = {
        "pre50_gain_from_fallback": _safe_float(runtime_on.get("preutterance_acc_50ms"))
        - _safe_float(runtime_off.get("preutterance_acc_50ms")),
        "pre20_gain_from_fallback": _safe_float(runtime_on.get("preutterance_acc_20ms"))
        - _safe_float(runtime_off.get("preutterance_acc_20ms")),
        "hard_drop_from_fallback": _safe_float(runtime_on.get("hard_failure_rate"))
        - _safe_float(runtime_off.get("hard_failure_rate")),
    }
    return {
        "label": str(label),
        "on_path": os.path.abspath(str(on_path)),
        "off_path": os.path.abspath(str(off_path)),
        "evaluated_items_on": int(on_payload.get("evaluated_items", 0) or 0),
        "evaluated_items_off": int(off_payload.get("evaluated_items", 0) or 0),
        "runtime_on": {
            "preutterance_acc_20ms": _safe_float(runtime_on.get("preutterance_acc_20ms")),
            "preutterance_acc_50ms": _safe_float(runtime_on.get("preutterance_acc_50ms")),
            "hard_failure_rate": _safe_float(runtime_on.get("hard_failure_rate")),
            "param_mae_ms": {k: _safe_float((runtime_on.get("param_mae_ms") or {}).get(k)) for k in mae_fields},
        },
        "runtime_off": {
            "preutterance_acc_20ms": _safe_float(runtime_off.get("preutterance_acc_20ms")),
            "preutterance_acc_50ms": _safe_float(runtime_off.get("preutterance_acc_50ms")),
            "hard_failure_rate": _safe_float(runtime_off.get("hard_failure_rate")),
            "param_mae_ms": {k: _safe_float((runtime_off.get("param_mae_ms") or {}).get(k)) for k in mae_fields},
        },
        "delta_off_minus_on": {
            "metrics": delta_runtime,
            "param_mae_ms": delta_mae,
            "postprocess_counts": delta_counts,
        },
        "fallback_dependency": fallback_dependency,
        "roles_worst_off_minus_on": role_rows[:20],
        "roles_best_off_minus_on": list(reversed(role_rows[-20:])),
    }


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": report.get("label"),
        "evaluated_items_on": report.get("evaluated_items_on"),
        "evaluated_items_off": report.get("evaluated_items_off"),
        "runtime_on": report.get("runtime_on"),
        "runtime_off": report.get("runtime_off"),
        "delta_off_minus_on": report.get("delta_off_minus_on"),
        "fallback_dependency": report.get("fallback_dependency"),
        "roles_worst_off_minus_on_top5": (report.get("roles_worst_off_minus_on") or [])[:5],
    }


if __name__ == "__main__":
    raise SystemExit(main())
