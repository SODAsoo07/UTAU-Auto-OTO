#!/usr/bin/env python
"""
Run SOFA evaluate.py and accumulate CSV/JSON reports.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from typing import Dict


def extract_json_block(text: str) -> Dict[str, float]:
    if not text:
        return {}
    # pick the last JSON object block in stdout
    matches = re.findall(r"\{[\s\S]*\}", text)
    for block in reversed(matches):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def append_csv_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    keys = list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa-repo-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--ignore", default="AP,SP,<AP>,<SP>,pau,cl")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    eval_py = os.path.join(args.sofa_repo_dir, "evaluate.py")
    if not os.path.exists(eval_py):
        raise FileNotFoundError(f"evaluate.py not found: {eval_py}")

    cmd = [args.python, eval_py, args.pred_dir, args.target_dir, "--ignore", args.ignore]
    if args.recursive:
        cmd.append("--recursive")
    if args.strict:
        cmd.append("--strict")

    proc = subprocess.run(
        cmd,
        cwd=args.sofa_repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = proc.stdout or ""
    metrics = extract_json_block(output)
    if not metrics:
        raise RuntimeError(f"SOFA evaluate output did not contain JSON metrics.\n{output[-2000:]}")

    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"eval_{stamp}"
    run_report = {
        "timestamp": now.isoformat(timespec="seconds"),
        "run_name": run_name,
        "pred_dir": args.pred_dir,
        "target_dir": args.target_dir,
        "recursive": bool(args.recursive),
        "strict": bool(args.strict),
        "ignore": args.ignore,
        "returncode": proc.returncode,
    }
    run_report.update(metrics)

    os.makedirs(args.report_dir, exist_ok=True)
    run_json = os.path.join(args.report_dir, f"{run_name}.json")
    run_txt = os.path.join(args.report_dir, f"{run_name}.log.txt")
    runs_csv = os.path.join(args.report_dir, "evaluate_runs.csv")
    runs_jsonl = os.path.join(args.report_dir, "evaluate_runs.jsonl")

    with open(run_json, "w", encoding="utf-8") as jf:
        json.dump(run_report, jf, indent=2, ensure_ascii=False)
    with open(run_txt, "w", encoding="utf-8") as tf:
        tf.write(output)
    append_csv_row(runs_csv, run_report)
    with open(runs_jsonl, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(run_report, ensure_ascii=False) + "\n")

    print(json.dumps(run_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

