import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from itertools import product
from typing import Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.compare_oto_batch_runs import compare_runs


ENV_KEYS = (
    "UTOA_JA_MAPPING_CONF_THRESHOLD",
    "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD",
    "UTOA_KR_MAPPING_CONF_THRESHOLD",
    "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD",
)


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_float_grid(raw: str) -> List[Optional[float]]:
    text = str(raw or "").strip()
    if not text:
        return [None]
    out: List[Optional[float]] = []
    seen = set()
    for part in text.split(","):
        token = part.strip().lower()
        if not token or token in {"none", "null", "-"}:
            value = None
        else:
            value = float(token)
        key = "none" if value is None else f"{value:.8f}"
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out or [None]


def _build_env_overrides(
    ja_conf: Optional[float],
    ja_spn: Optional[float],
    kr_conf: Optional[float],
    kr_spn: Optional[float],
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if ja_conf is not None:
        env["UTOA_JA_MAPPING_CONF_THRESHOLD"] = f"{float(ja_conf):.4f}"
    if ja_spn is not None:
        env["UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD"] = f"{float(ja_spn):.4f}"
    if kr_conf is not None:
        env["UTOA_KR_MAPPING_CONF_THRESHOLD"] = f"{float(kr_conf):.4f}"
    if kr_spn is not None:
        env["UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD"] = f"{float(kr_spn):.4f}"
    return env


def _format_env_suffix(env_overrides: Dict[str, str]) -> str:
    if not env_overrides:
        return "default"
    parts: List[str] = []
    key_map = {
        "UTOA_JA_MAPPING_CONF_THRESHOLD": "ja_conf",
        "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD": "ja_spn",
        "UTOA_KR_MAPPING_CONF_THRESHOLD": "kr_conf",
        "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD": "kr_spn",
    }
    for env_key in sorted(env_overrides.keys()):
        tag = key_map.get(env_key, env_key.lower())
        val = str(env_overrides[env_key]).replace(".", "p")
        parts.append(f"{tag}_{val}")
    return "__".join(parts)


def _run_batch_once(
    python_exe: str,
    batch_script: str,
    config_path: str,
    run_tag: str,
    env_overrides: Dict[str, str],
    stop_on_error: bool,
    log_path: str,
) -> Dict[str, object]:
    env = os.environ.copy()
    for key in ENV_KEYS:
        env.pop(key, None)
    env.update(env_overrides)

    cmd = [
        python_exe,
        batch_script,
        "--config",
        config_path,
        "--run-tag",
        run_tag,
    ]
    if stop_on_error:
        cmd.append("--stop-on-error")

    proc = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        if env_overrides:
            f.write(f"env_overrides={json.dumps(env_overrides, ensure_ascii=False)}\n")
        else:
            f.write("env_overrides={}\n")
        f.write("\n[stdout]\n")
        f.write(proc.stdout or "")
        f.write("\n[stderr]\n")
        f.write(proc.stderr or "")

    summary_path = os.path.join(ROOT_DIR, "logs", "oto_batch", run_tag, "summary.json")
    return {
        "returncode": int(proc.returncode),
        "summary_path": os.path.abspath(summary_path),
        "command": cmd,
    }


def _rank_key(row: Dict[str, object]):
    if bool(row.get("run_failed", False)):
        return (1, 10**9, 10**9, 10**9, 10**9)
    regression_count = int(row.get("regression_count", 0))
    errors_delta_sum = int(row.get("errors_delta_sum", 0))
    warnings_delta_sum = int(row.get("warnings_delta_sum", 0))
    improvement_count = int(row.get("improvement_count", 0))
    status_changed_cases = int(row.get("status_changed_cases", 0))
    return (0, regression_count, errors_delta_sum, warnings_delta_sum, -improvement_count, status_changed_cases)


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "run_tag",
        "env_suffix",
        "run_failed",
        "returncode",
        "regression_count",
        "improvement_count",
        "errors_delta_sum",
        "warnings_delta_sum",
        "status_changed_cases",
        "summary_path",
        "log_path",
        "UTOA_JA_MAPPING_CONF_THRESHOLD",
        "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD",
        "UTOA_KR_MAPPING_CONF_THRESHOLD",
        "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description="Sweep KR/JA mapping confidence thresholds and compare batch summary deltas."
    )
    ap.add_argument("--config", required=True, help="Path to oto batch config YAML")
    ap.add_argument("--baseline-summary", default="", help="Baseline summary.json path. Empty: run baseline first.")
    ap.add_argument("--run-prefix", default="th_tune", help="Run tag prefix for generated batch runs")
    ap.add_argument("--out-dir", default="", help="Output folder for tuning reports (default: logs/threshold_tuning/<tag>)")
    ap.add_argument("--batch-script", default=os.path.join("scripts", "run_oto_generation_batch.py"))
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--stop-on-error", action="store_true", help="Pass --stop-on-error to each batch run")
    ap.add_argument("--max-candidates", type=int, default=0, help="Limit number of candidate combinations (0=all)")
    ap.add_argument("--dry-run", action="store_true", help="Print candidates only and exit")
    ap.add_argument("--ja-conf-grid", default="0.64,0.68,0.72", help="Comma-separated JA confidence values; use 'none' to skip override")
    ap.add_argument("--ja-spn-grid", default="0.30,0.35", help="Comma-separated JA spn-ratio values; use 'none' to skip override")
    ap.add_argument("--kr-conf-grid", default="", help="Comma-separated KR confidence values; empty keeps default only")
    ap.add_argument("--kr-spn-grid", default="", help="Comma-separated KR spn-ratio values; empty keeps default only")
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    batch_script = os.path.abspath(args.batch_script)
    if not os.path.exists(batch_script):
        raise FileNotFoundError(batch_script)

    run_root_tag = f"{args.run_prefix}_{_now_tag()}"
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(ROOT_DIR, "logs", "threshold_tuning", run_root_tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "run_logs"), exist_ok=True)

    ja_conf_grid = _parse_float_grid(args.ja_conf_grid)
    ja_spn_grid = _parse_float_grid(args.ja_spn_grid)
    kr_conf_grid = _parse_float_grid(args.kr_conf_grid)
    kr_spn_grid = _parse_float_grid(args.kr_spn_grid)

    raw_candidates = []
    for ja_conf, ja_spn, kr_conf, kr_spn in product(ja_conf_grid, ja_spn_grid, kr_conf_grid, kr_spn_grid):
        env_overrides = _build_env_overrides(ja_conf, ja_spn, kr_conf, kr_spn)
        raw_candidates.append(env_overrides)

    # Preserve order while removing duplicates.
    deduped_candidates = []
    seen = set()
    for env_overrides in raw_candidates:
        key = json.dumps(env_overrides, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(env_overrides)

    if args.max_candidates > 0:
        deduped_candidates = deduped_candidates[: int(args.max_candidates)]

    print(f"[ThresholdTune] config={config_path}")
    print(f"[ThresholdTune] candidates={len(deduped_candidates)}")
    for i, cand in enumerate(deduped_candidates, 1):
        print(f"  - [{i:03d}] {_format_env_suffix(cand)}")

    if args.dry_run:
        print("[ThresholdTune] dry-run only. exit.")
        return

    baseline_summary = os.path.abspath(args.baseline_summary) if args.baseline_summary else ""
    if baseline_summary and not os.path.exists(baseline_summary):
        raise FileNotFoundError(baseline_summary)

    if not baseline_summary:
        base_tag = f"{run_root_tag}_baseline"
        base_log = os.path.join(out_dir, "run_logs", "baseline.log")
        base_run = _run_batch_once(
            python_exe=args.python_exe,
            batch_script=batch_script,
            config_path=config_path,
            run_tag=base_tag,
            env_overrides={},
            stop_on_error=bool(args.stop_on_error),
            log_path=base_log,
        )
        if int(base_run["returncode"]) != 0:
            raise RuntimeError(f"Baseline batch failed. log={base_log}")
        baseline_summary = str(base_run["summary_path"])
        if not os.path.exists(baseline_summary):
            raise RuntimeError(f"Baseline summary not found: {baseline_summary}")
        print(f"[ThresholdTune] baseline={baseline_summary}")

    rows: List[Dict[str, object]] = []
    for idx, env_overrides in enumerate(deduped_candidates, 1):
        env_suffix = _format_env_suffix(env_overrides)
        run_tag = f"{run_root_tag}_{idx:03d}"
        run_log = os.path.join(out_dir, "run_logs", f"{idx:03d}_{env_suffix}.log")
        run_info = _run_batch_once(
            python_exe=args.python_exe,
            batch_script=batch_script,
            config_path=config_path,
            run_tag=run_tag,
            env_overrides=env_overrides,
            stop_on_error=bool(args.stop_on_error),
            log_path=run_log,
        )
        summary_path = str(run_info["summary_path"])
        run_failed = int(run_info["returncode"]) != 0 or (not os.path.exists(summary_path))

        row: Dict[str, object] = {
            "rank": 0,
            "run_tag": run_tag,
            "env_suffix": env_suffix,
            "run_failed": bool(run_failed),
            "returncode": int(run_info["returncode"]),
            "regression_count": 999999 if run_failed else 0,
            "improvement_count": 0,
            "errors_delta_sum": 999999 if run_failed else 0,
            "warnings_delta_sum": 999999 if run_failed else 0,
            "status_changed_cases": 999999 if run_failed else 0,
            "summary_path": summary_path,
            "log_path": run_log,
            "UTOA_JA_MAPPING_CONF_THRESHOLD": env_overrides.get("UTOA_JA_MAPPING_CONF_THRESHOLD", ""),
            "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD": env_overrides.get("UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD", ""),
            "UTOA_KR_MAPPING_CONF_THRESHOLD": env_overrides.get("UTOA_KR_MAPPING_CONF_THRESHOLD", ""),
            "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD": env_overrides.get("UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD", ""),
        }

        if not run_failed:
            comp = compare_runs(base_summary=baseline_summary, new_summary=summary_path)
            agg = comp["aggregate"]
            row["regression_count"] = len(list(agg.get("regression_cases") or []))
            row["improvement_count"] = len(list(agg.get("improvement_cases") or []))
            row["errors_delta_sum"] = int(agg.get("errors_delta_sum", 0) or 0)
            row["warnings_delta_sum"] = int(agg.get("warnings_delta_sum", 0) or 0)
            row["status_changed_cases"] = int(agg.get("status_changed_cases", 0) or 0)
            comp_path = os.path.join(out_dir, "run_logs", f"{idx:03d}_{env_suffix}.compare.json")
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, indent=2)
            row["compare_path"] = comp_path

        rows.append(row)
        print(
            "[ThresholdTune] "
            f"{idx}/{len(deduped_candidates)} "
            f"failed={row['run_failed']} "
            f"reg={row['regression_count']} "
            f"imp={row['improvement_count']} "
            f"err_delta={row['errors_delta_sum']} "
            f"{env_suffix}"
        )

    ranked_rows = sorted(rows, key=_rank_key)
    for rank, row in enumerate(ranked_rows, 1):
        row["rank"] = rank

    report = {
        "created_at": dt.datetime.now().isoformat(),
        "config": config_path,
        "baseline_summary": baseline_summary,
        "out_dir": out_dir,
        "candidates": ranked_rows,
    }
    report_json = os.path.join(out_dir, "threshold_tuning_report.json")
    report_csv = os.path.join(out_dir, "threshold_tuning_report.csv")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_csv(report_csv, ranked_rows)

    best = ranked_rows[0] if ranked_rows else None
    print(f"[ThresholdTune] report_json={report_json}")
    print(f"[ThresholdTune] report_csv={report_csv}")
    if best:
        print(
            "[ThresholdTune] best="
            f"rank={best['rank']} "
            f"failed={best['run_failed']} "
            f"reg={best['regression_count']} "
            f"imp={best['improvement_count']} "
            f"err_delta={best['errors_delta_sum']} "
            f"env={best['env_suffix']}"
        )


if __name__ == "__main__":
    main()
