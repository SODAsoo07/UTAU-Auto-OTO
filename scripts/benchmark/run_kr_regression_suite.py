import argparse
import datetime as dt
import json
import os
import subprocess as sp
import sys
from typing import Dict, List, Optional, Tuple


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BATCH_SCRIPT = os.path.join(ROOT_DIR, "scripts", "run_oto_generation_batch.py")


def _safe_print(text: str) -> None:
    print(str(text), flush=True)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _extract_summary_path(output_lines: List[str]) -> str:
    for line in output_lines:
        s = str(line or "").strip()
        if s.startswith("[BatchOTO] summary="):
            return s.split("=", 1)[1].strip()
    return ""


def _compute_metrics(summary: Dict[str, object]) -> Dict[str, float]:
    results = summary.get("results") or []
    if not isinstance(results, list):
        results = []

    total_cases = int(summary.get("total_cases", len(results)) or len(results))
    ok_cases = int(summary.get("ok_cases", 0) or 0)
    error_cases = int(summary.get("error_cases", 0) or 0)
    skipped_cases = int(summary.get("skipped_cases", 0) or 0)

    validation_errors = 0
    validation_warnings = 0
    validation_total_lines = 0
    ml_fallback_cases = 0
    ml_rows = 0
    blank_flag_rows = 0
    mel_unreliable_rows = 0
    abstain_rows = 0

    for row in results:
        if not isinstance(row, dict):
            continue
        validation = row.get("validation") or {}
        if isinstance(validation, dict):
            validation_errors += int(validation.get("errors", 0) or 0)
            validation_warnings += int(validation.get("warnings", 0) or 0)
            validation_total_lines += int(validation.get("total_lines", 0) or 0)

        ml = row.get("ml") or {}
        if isinstance(ml, dict):
            if bool(ml.get("fallback_used", False)):
                ml_fallback_cases += 1
            reliability = ml.get("reliability_metrics") or {}
            if not isinstance(reliability, dict) or not reliability:
                legacy_report = ml.get("legacy_report") or {}
                if isinstance(legacy_report, dict):
                    reliability = legacy_report.get("reliability_metrics") or {}
            if isinstance(reliability, dict):
                rows = int(reliability.get("rows", 0) or 0)
                ml_rows += rows
                blank_flag_rows += int(reliability.get("blank_flag_rows", 0) or 0)
                mel_unreliable_rows += int(reliability.get("mel_unreliable_rows", 0) or 0)
                abstain_rows += int(reliability.get("abstain_rows", 0) or 0)

    case_denom = max(1, total_cases)
    row_denom = max(1, ml_rows)
    val_line_denom = max(1, validation_total_lines)
    return {
        "total_cases": total_cases,
        "ok_cases": ok_cases,
        "error_cases": error_cases,
        "skipped_cases": skipped_cases,
        "validation_errors": validation_errors,
        "validation_total_lines": validation_total_lines,
        "validation_warnings": validation_warnings,
        "validation_warning_rate": float(validation_warnings) / float(val_line_denom),
        "ml_fallback_cases": ml_fallback_cases,
        "ml_fallback_rate": float(ml_fallback_cases) / float(case_denom),
        "ml_rows": ml_rows,
        "blank_flag_rows": blank_flag_rows,
        "blank_flag_rate": float(blank_flag_rows) / float(row_denom),
        "mel_unreliable_rows": mel_unreliable_rows,
        "mel_unreliable_rate": float(mel_unreliable_rows) / float(row_denom),
        "abstain_rows": abstain_rows,
        "abstain_rate": float(abstain_rows) / float(row_denom),
    }


def _evaluate_thresholds(metrics: Dict[str, float], args) -> List[str]:
    issues: List[str] = []
    if int(metrics.get("error_cases", 0)) > int(args.max_error_cases):
        issues.append(
            f"error_cases {int(metrics.get('error_cases', 0))} > max_error_cases {int(args.max_error_cases)}"
        )
    if int(metrics.get("validation_errors", 0)) > int(args.max_validation_errors):
        issues.append(
            f"validation_errors {int(metrics.get('validation_errors', 0))} > max_validation_errors {int(args.max_validation_errors)}"
        )
    if int(metrics.get("validation_warnings", 0)) > int(args.max_validation_warnings):
        issues.append(
            f"validation_warnings {int(metrics.get('validation_warnings', 0))} > max_validation_warnings {int(args.max_validation_warnings)}"
        )
    if float(metrics.get("validation_warning_rate", 0.0)) > float(args.max_validation_warning_rate):
        issues.append(
            f"validation_warning_rate {metrics.get('validation_warning_rate', 0.0):.4f} > max_validation_warning_rate {float(args.max_validation_warning_rate):.4f}"
        )
    if float(metrics.get("ml_fallback_rate", 0.0)) > float(args.max_ml_fallback_rate):
        issues.append(
            f"ml_fallback_rate {metrics.get('ml_fallback_rate', 0.0):.4f} > max_ml_fallback_rate {float(args.max_ml_fallback_rate):.4f}"
        )
    if float(metrics.get("blank_flag_rate", 0.0)) > float(args.max_blank_flag_rate):
        issues.append(
            f"blank_flag_rate {metrics.get('blank_flag_rate', 0.0):.4f} > max_blank_flag_rate {float(args.max_blank_flag_rate):.4f}"
        )
    if float(metrics.get("mel_unreliable_rate", 0.0)) > float(args.max_mel_unreliable_rate):
        issues.append(
            f"mel_unreliable_rate {metrics.get('mel_unreliable_rate', 0.0):.4f} > max_mel_unreliable_rate {float(args.max_mel_unreliable_rate):.4f}"
        )
    if float(metrics.get("abstain_rate", 0.0)) > float(args.max_abstain_rate):
        issues.append(
            f"abstain_rate {metrics.get('abstain_rate', 0.0):.4f} > max_abstain_rate {float(args.max_abstain_rate):.4f}"
        )
    return issues


def _evaluate_baseline_delta(
    current: Dict[str, float],
    baseline: Dict[str, float],
    args,
) -> Tuple[Dict[str, float], List[str]]:
    delta = {
        "ml_fallback_rate_delta": float(current.get("ml_fallback_rate", 0.0)) - float(baseline.get("ml_fallback_rate", 0.0)),
        "blank_flag_rate_delta": float(current.get("blank_flag_rate", 0.0)) - float(baseline.get("blank_flag_rate", 0.0)),
        "mel_unreliable_rate_delta": float(current.get("mel_unreliable_rate", 0.0)) - float(baseline.get("mel_unreliable_rate", 0.0)),
        "abstain_rate_delta": float(current.get("abstain_rate", 0.0)) - float(baseline.get("abstain_rate", 0.0)),
        "validation_warnings_delta": int(current.get("validation_warnings", 0)) - int(baseline.get("validation_warnings", 0)),
        "validation_warning_rate_delta": float(current.get("validation_warning_rate", 0.0)) - float(baseline.get("validation_warning_rate", 0.0)),
    }
    issues: List[str] = []
    if float(delta["ml_fallback_rate_delta"]) > float(args.max_ml_fallback_rate_delta):
        issues.append(
            f"ml_fallback_rate_delta {delta['ml_fallback_rate_delta']:.4f} > max_ml_fallback_rate_delta {float(args.max_ml_fallback_rate_delta):.4f}"
        )
    if float(delta["blank_flag_rate_delta"]) > float(args.max_blank_flag_rate_delta):
        issues.append(
            f"blank_flag_rate_delta {delta['blank_flag_rate_delta']:.4f} > max_blank_flag_rate_delta {float(args.max_blank_flag_rate_delta):.4f}"
        )
    if float(delta["mel_unreliable_rate_delta"]) > float(args.max_mel_unreliable_rate_delta):
        issues.append(
            f"mel_unreliable_rate_delta {delta['mel_unreliable_rate_delta']:.4f} > max_mel_unreliable_rate_delta {float(args.max_mel_unreliable_rate_delta):.4f}"
        )
    if float(delta["abstain_rate_delta"]) > float(args.max_abstain_rate_delta):
        issues.append(
            f"abstain_rate_delta {delta['abstain_rate_delta']:.4f} > max_abstain_rate_delta {float(args.max_abstain_rate_delta):.4f}"
        )
    if int(delta["validation_warnings_delta"]) > int(args.max_validation_warnings_delta):
        issues.append(
            f"validation_warnings_delta {int(delta['validation_warnings_delta'])} > max_validation_warnings_delta {int(args.max_validation_warnings_delta)}"
        )
    if float(delta["validation_warning_rate_delta"]) > float(args.max_validation_warning_rate_delta):
        issues.append(
            f"validation_warning_rate_delta {delta['validation_warning_rate_delta']:.4f} > max_validation_warning_rate_delta {float(args.max_validation_warning_rate_delta):.4f}"
        )
    return delta, issues


def _run_batch(config_path: str, run_tag: str, skip_validation: bool, stop_on_error: bool) -> Tuple[int, List[str]]:
    cmd = [sys.executable, BATCH_SCRIPT, "--config", config_path]
    if run_tag:
        cmd.extend(["--run-tag", run_tag])
    if skip_validation:
        cmd.append("--skip-validation")
    if stop_on_error:
        cmd.append("--stop-on-error")

    _safe_print(f"[Regression] exec: {' '.join(cmd)}")
    proc = sp.Popen(
        cmd,
        cwd=ROOT_DIR,
        text=True,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines: List[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            output_lines.append(line)
            _safe_print(line)
    return int(proc.wait()), output_lines


def _write_report(report_path: str, report: Dict[str, object]) -> str:
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def _write_report_text(path: str, report: Dict[str, object]) -> str:
    lines = [
        f"status={report.get('status', '')}",
        f"summary={report.get('summary_path', '')}",
        f"created_at={report.get('created_at', '')}",
        "",
        "[metrics]",
    ]
    metrics = report.get("metrics") or {}
    if isinstance(metrics, dict):
        for k in sorted(metrics.keys()):
            lines.append(f"{k}={metrics.get(k)}")
    baseline = report.get("baseline") or {}
    if isinstance(baseline, dict) and baseline:
        lines.append("")
        lines.append("[baseline.metrics]")
        for k in sorted((baseline.get("metrics") or {}).keys()):
            lines.append(f"{k}={baseline.get('metrics', {}).get(k)}")
        lines.append("")
        lines.append("[baseline.delta]")
        for k in sorted((baseline.get("delta") or {}).keys()):
            lines.append(f"{k}={baseline.get('delta', {}).get(k)}")
    lines.append("")
    lines.append("[issues]")
    issues = report.get("issues") or []
    if isinstance(issues, list) and issues:
        for item in issues:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


def _write_baseline_artifacts(
    *,
    run_dir: str,
    summary: Dict[str, object],
    metrics: Dict[str, float],
    prefix: str,
) -> Tuple[str, str]:
    token = str(prefix or "").strip() or "regression_baseline"
    summary_path = os.path.join(run_dir, f"{token}_summary.json")
    metrics_path = os.path.join(run_dir, f"{token}_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "metrics": dict(metrics or {}),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return summary_path, metrics_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run KR OTO batch regression and apply quality gates."
    )
    ap.add_argument("--batch-config", required=True, help="YAML path for run_oto_generation_batch.py")
    ap.add_argument("--run-tag", default="", help="Optional run tag for batch runner")
    ap.add_argument("--baseline-summary", default="", help="Optional baseline summary.json path")
    ap.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write current run summary/metrics as baseline artifacts in run dir.",
    )
    ap.add_argument(
        "--baseline-prefix",
        default="regression_baseline",
        help="Filename prefix for --save-baseline artifacts.",
    )
    ap.add_argument("--skip-validation", action="store_true", help="Pass through to batch runner")
    ap.add_argument("--stop-on-error", action="store_true", help="Pass through to batch runner")
    ap.add_argument("--max-error-cases", type=int, default=0)
    ap.add_argument("--max-validation-errors", type=int, default=0)
    ap.add_argument("--max-validation-warnings", type=int, default=180)
    ap.add_argument("--max-validation-warning-rate", type=float, default=0.30)
    ap.add_argument("--max-ml-fallback-rate", type=float, default=0.35)
    ap.add_argument("--max-blank-flag-rate", type=float, default=0.58)
    ap.add_argument("--max-mel-unreliable-rate", type=float, default=0.70)
    ap.add_argument("--max-abstain-rate", type=float, default=0.62)
    ap.add_argument("--max-ml-fallback-rate-delta", type=float, default=0.08)
    ap.add_argument("--max-blank-flag-rate-delta", type=float, default=0.08)
    ap.add_argument("--max-mel-unreliable-rate-delta", type=float, default=0.08)
    ap.add_argument("--max-abstain-rate-delta", type=float, default=0.08)
    ap.add_argument("--max-validation-warnings-delta", type=int, default=6)
    ap.add_argument("--max-validation-warning-rate-delta", type=float, default=0.03)
    args = ap.parse_args()

    config_path = os.path.abspath(args.batch_config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(config_path)
    if not os.path.isfile(BATCH_SCRIPT):
        raise FileNotFoundError(BATCH_SCRIPT)

    code, lines = _run_batch(
        config_path=config_path,
        run_tag=str(args.run_tag or "").strip(),
        skip_validation=bool(args.skip_validation),
        stop_on_error=bool(args.stop_on_error),
    )
    if code != 0:
        _safe_print(f"[Regression] batch runner failed with code={code}")
        return int(code)

    summary_path = _extract_summary_path(lines)
    if not summary_path:
        raise RuntimeError("Could not locate summary path from batch output.")
    summary_path = os.path.abspath(summary_path)
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)

    summary = _load_json(summary_path)
    metrics = _compute_metrics(summary)
    issues = _evaluate_thresholds(metrics, args)

    baseline_block = {}
    baseline_path = str(args.baseline_summary or "").strip()
    if baseline_path:
        baseline_abs = os.path.abspath(baseline_path)
        baseline_summary = _load_json(baseline_abs)
        baseline_metrics = _compute_metrics(baseline_summary)
        delta, delta_issues = _evaluate_baseline_delta(metrics, baseline_metrics, args)
        issues.extend(delta_issues)
        baseline_block = {
            "summary_path": baseline_abs,
            "metrics": baseline_metrics,
            "delta": delta,
        }

    status = "pass" if not issues else "fail"
    run_dir = os.path.dirname(summary_path)
    report = {
        "status": status,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "batch_config": config_path,
        "summary_path": summary_path,
        "metrics": metrics,
        "baseline": baseline_block,
        "issues": issues,
    }

    report_json = os.path.join(run_dir, "regression_gate.json")
    report_txt = os.path.join(run_dir, "regression_gate.txt")
    _write_report(report_json, report)
    _write_report_text(report_txt, report)
    if bool(args.save_baseline):
        baseline_summary_out, baseline_metrics_out = _write_baseline_artifacts(
            run_dir=run_dir,
            summary=summary,
            metrics=metrics,
            prefix=str(args.baseline_prefix or ""),
        )
        _safe_print(f"[Regression] baseline_summary={baseline_summary_out}")
        _safe_print(f"[Regression] baseline_metrics={baseline_metrics_out}")
    _safe_print(f"[Regression] status={status}")
    _safe_print(f"[Regression] report_json={report_json}")
    _safe_print(f"[Regression] report_txt={report_txt}")
    if issues:
        for item in issues:
            _safe_print(f"[Regression][FAIL] {item}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
