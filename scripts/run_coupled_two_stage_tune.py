import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess as sp
import sys
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:
    yaml = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

ONE_STAGE_SCRIPT = os.path.join(ROOT_DIR, "scripts", "run_coupled_experiment_matrix.py")


def _load_yaml(path: str):
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(path: str, data: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=False, sort_keys=False)
    return path


def _safe_name(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "run"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_bool(value, default=False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _normalize_arg_key(name: str) -> str:
    key = str(name or "").strip()
    while key.startswith("-"):
        key = key[1:]
    return key.replace("_", "-").strip().lower()


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _same_value(a: Any, b: Any) -> bool:
    fa = _float_or_none(a)
    fb = _float_or_none(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-12
    return str(a) == str(b)


def _find_value_index(values: List[Any], best_value: Any) -> int:
    if not values:
        return 0
    for i, v in enumerate(values):
        if _same_value(v, best_value):
            return i
    best_f = _float_or_none(best_value)
    if best_f is not None:
        min_idx = 0
        min_dist = float("inf")
        for i, v in enumerate(values):
            vf = _float_or_none(v)
            if vf is None:
                continue
            d = abs(vf - best_f)
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx
    return 0


def _slice_with_min_count(values: List[Any], center_idx: int, window: int, min_count: int) -> List[Any]:
    n = len(values)
    if n <= 0:
        return []
    center = max(0, min(n - 1, int(center_idx)))
    left = max(0, center - max(0, int(window)))
    right = min(n, center + max(0, int(window)) + 1)
    out = list(values[left:right])
    while len(out) < min_count and (left > 0 or right < n):
        if left > 0:
            left -= 1
            out.insert(0, values[left])
            if len(out) >= min_count:
                break
        if right < n:
            out.append(values[right])
            right += 1
    return out


def _extract_report_path(text: str) -> str:
    for line in str(text or "").splitlines():
        s = line.strip()
        if s.startswith("[CoupledTune] report_json="):
            return s.split("=", 1)[1].strip()
    return ""


def _find_latest_report(search_root: str) -> str:
    root = os.path.abspath(search_root)
    if not os.path.isdir(root):
        return ""
    best_path = ""
    best_time = -1.0
    for dirpath, _dirnames, filenames in os.walk(root):
        if "experiment_report.json" not in filenames:
            continue
        p = os.path.join(dirpath, "experiment_report.json")
        try:
            mt = os.path.getmtime(p)
        except Exception:
            mt = -1.0
        if mt > best_time:
            best_time = mt
            best_path = p
    return best_path


def _run_one_stage(
    *,
    stage_name: str,
    stage_config: Dict[str, Any],
    stage_work_dir: str,
    max_trials: int,
) -> Dict[str, Any]:
    os.makedirs(stage_work_dir, exist_ok=True)
    stage_yaml = os.path.join(stage_work_dir, f"{stage_name}.config.yaml")
    _dump_yaml(stage_yaml, stage_config)

    cmd = [sys.executable, ONE_STAGE_SCRIPT, "--config", stage_yaml]
    if int(max_trials) > 0:
        cmd.extend(["--max-trials", str(int(max_trials))])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    stdout_lines: List[str] = []
    stderr = ""
    trial_total = 0
    trial_done = 0
    progress_key = ""
    re_trials = re.compile(r"^\[CoupledTune\]\s+trials=(\d+)\s*$")
    re_trial_done = re.compile(r"^\[CoupledTune\]\s+trial\s+(\d+)/(\d+)\s+done\b")

    proc = sp.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        text=True,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            stdout_lines.append(raw_line)
            print(line, flush=True)

            m_total = re_trials.match(line)
            if m_total:
                try:
                    trial_total = max(0, int(m_total.group(1)))
                except Exception:
                    trial_total = 0

            m_done = re_trial_done.match(line)
            if m_done:
                try:
                    trial_done = max(0, int(m_done.group(1)))
                    parsed_total = max(0, int(m_done.group(2)))
                    if parsed_total > 0:
                        trial_total = parsed_total
                except Exception:
                    pass

            if trial_total > 0:
                pct = max(0.0, min(100.0, 100.0 * float(trial_done) / float(trial_total)))
                key = f"{trial_done}/{trial_total}"
                if key != progress_key:
                    progress_key = key
                    print(
                        f"[TwoStage] {stage_name} progress {trial_done}/{trial_total} ({pct:.1f}%)",
                        flush=True,
                    )

    return_code = int(proc.wait())
    stdout = "".join(stdout_lines)
    log_path = os.path.join(stage_work_dir, f"{stage_name}.run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout)
        if stderr:
            f.write("\n[STDERR]\n")
            f.write(stderr)

    report_path = _extract_report_path(stdout)
    if not report_path:
        output_root = str(stage_config.get("output_root", "") or "").strip()
        report_path = _find_latest_report(output_root)

    report = {}
    if report_path and os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f) or {}

    return {
        "stage_name": stage_name,
        "command": cmd,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": log_path,
        "config_path": stage_yaml,
        "report_path": report_path,
        "report": report,
    }


def _get_best_value_from_report(best: Dict[str, Any], kind: str, name: str) -> Any:
    kind_norm = str(kind or "env").strip().lower()
    if kind_norm in {"arg", "args", "cli", "option"}:
        key = _normalize_arg_key(name)
        raw_key = f"arg:{key}"
        if isinstance(best.get("raw_values"), dict) and raw_key in best["raw_values"]:
            return best["raw_values"].get(raw_key)
        args = best.get("args") or {}
        if isinstance(args, dict):
            return args.get(key)
        return None

    raw_key = f"env:{name}"
    if isinstance(best.get("raw_values"), dict) and raw_key in best["raw_values"]:
        return best["raw_values"].get(raw_key)
    env = best.get("env") or {}
    if isinstance(env, dict):
        return env.get(name)
    return None


def _derive_stage2_parameters(
    stage1_params: List[Dict[str, Any]],
    stage1_best: Dict[str, Any],
    *,
    window: int,
    min_values_per_parameter: int,
    lock_parameters: List[str],
) -> List[Dict[str, Any]]:
    lock_set = {str(v).strip() for v in list(lock_parameters or []) if str(v).strip()}
    out = []
    for p in list(stage1_params or []):
        if not isinstance(p, dict):
            continue
        row = dict(p)
        name = str(row.get("name", "")).strip()
        kind = str(row.get("kind", "env")).strip().lower()
        values = list(row.get("values") or [])
        if not name or not values:
            continue
        if len(values) == 1:
            row["values"] = values
            out.append(row)
            continue
        best_value = _get_best_value_from_report(stage1_best, kind, name)
        best_idx = _find_value_index(values, best_value) if best_value is not None else 0
        if name in lock_set:
            row["values"] = [values[best_idx]]
            out.append(row)
            continue
        row["values"] = _slice_with_min_count(
            values,
            center_idx=best_idx,
            window=window,
            min_count=max(1, int(min_values_per_parameter)),
        )
        out.append(row)
    return out


def _build_stage_config(
    base_cfg: Dict[str, Any],
    *,
    output_root: str,
    run_tag: str,
    search_cfg: Dict[str, Any],
    train_matrix_args_override: Optional[Dict[str, Any]] = None,
    fixed_env_override: Optional[Dict[str, Any]] = None,
    force_dry_run: bool = False,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["output_root"] = output_root
    cfg["run_tag"] = run_tag
    cfg["search"] = dict(search_cfg or {})

    tm = dict(cfg.get("train_matrix") or {})
    args = dict(tm.get("args") or {})
    if isinstance(train_matrix_args_override, dict):
        args.update(train_matrix_args_override)
    if force_dry_run:
        args["dry-run"] = True
    tm["args"] = args
    cfg["train_matrix"] = tm

    fixed_env = dict(cfg.get("fixed_env") or {})
    if isinstance(fixed_env_override, dict):
        fixed_env.update(fixed_env_override)
    cfg["fixed_env"] = fixed_env
    return cfg


def _one_stage_summary(stage_result: Dict[str, Any]) -> Dict[str, Any]:
    report = stage_result.get("report") or {}
    best = report.get("best") or {}
    return {
        "return_code": int(stage_result.get("return_code", 1)),
        "report_path": str(stage_result.get("report_path", "") or ""),
        "best_score": best.get("score"),
        "best_env": best.get("env", {}),
        "best_raw_values": best.get("raw_values", {}),
    }


def main():
    ap = argparse.ArgumentParser(description="Run two-stage coupled tuning (broad -> narrow).")
    ap.add_argument("--config", required=True, help="Base one-stage config YAML")
    ap.add_argument("--stage1-max-trials", type=int, default=0, help="Override stage1 max trials")
    ap.add_argument("--stage2-max-trials", type=int, default=0, help="Override stage2 max trials")
    ap.add_argument("--neighbor-window", type=int, default=-1, help="Override stage2 neighbor window")
    ap.add_argument("--force-dry-run", action="store_true", help="Force --dry-run for train_matrix")
    ap.add_argument("--stage1-only", action="store_true", help="Run stage1 only")
    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml")
    if not os.path.isfile(ONE_STAGE_SCRIPT):
        raise FileNotFoundError(ONE_STAGE_SCRIPT)

    config_path = os.path.abspath(args.config)
    base_cfg = _load_yaml(config_path)
    two_stage_cfg = dict(base_cfg.get("two_stage") or {})
    if not _to_bool(two_stage_cfg.get("enabled", True), True):
        raise SystemExit("two_stage.enabled is false in config.")

    top_output_root = os.path.abspath(str(base_cfg.get("output_root", "logs/coupled_experiments")))
    top_run_tag = _safe_name(str(base_cfg.get("run_tag", "coupled_tune")).strip() or "coupled_tune")
    controller_dir = os.path.join(top_output_root, f"{top_run_tag}.two_stage.{_now_tag()}")
    os.makedirs(controller_dir, exist_ok=True)

    stage1_search = dict(base_cfg.get("search") or {})
    if int(args.stage1_max_trials) > 0:
        stage1_search["max_trials"] = int(args.stage1_max_trials)
    stage1_cfg = _build_stage_config(
        base_cfg,
        output_root=os.path.join(controller_dir, "stage1_runs"),
        run_tag=f"{top_run_tag}_s1",
        search_cfg=stage1_search,
        force_dry_run=bool(args.force_dry_run),
    )

    print(f"[TwoStage] controller_dir={controller_dir}")
    print("[TwoStage] stage1 start")
    stage1_result = _run_one_stage(
        stage_name="stage1",
        stage_config=stage1_cfg,
        stage_work_dir=os.path.join(controller_dir, "stage1"),
        max_trials=int(stage1_search.get("max_trials", 0) or 0),
    )

    stage1_report = stage1_result.get("report") or {}
    stage1_best = stage1_report.get("best") or {}
    summary = {
        "config": config_path,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "controller_dir": os.path.abspath(controller_dir),
        "stage1": _one_stage_summary(stage1_result),
        "stage2": {},
    }

    if args.stage1_only:
        summary["stage2"]["skipped"] = True
        summary["stage2"]["reason"] = "stage1_only"
    elif not stage1_best:
        summary["stage2"]["skipped"] = True
        summary["stage2"]["reason"] = "stage1_best_missing"
    else:
        print("[TwoStage] stage2 derive")
        stage2_cfg_local = dict(two_stage_cfg.get("stage2") or {})
        neighbor_window = int(stage2_cfg_local.get("neighbor_window", 1) or 1)
        if int(args.neighbor_window) >= 0:
            neighbor_window = int(args.neighbor_window)
        min_values = int(stage2_cfg_local.get("min_values_per_parameter", 2) or 2)
        lock_params = list(stage2_cfg_local.get("lock_parameters") or [])

        explicit_params = list(stage2_cfg_local.get("parameters") or [])
        if explicit_params:
            stage2_params = explicit_params
        else:
            stage1_params = list(stage1_search.get("parameters") or [])
            stage2_params = _derive_stage2_parameters(
                stage1_params,
                stage1_best,
                window=max(0, neighbor_window),
                min_values_per_parameter=max(1, min_values),
                lock_parameters=lock_params,
            )

        stage2_search = dict(stage1_search)
        stage2_search["parameters"] = stage2_params
        if "top_k" in stage2_cfg_local:
            stage2_search["top_k"] = stage2_cfg_local.get("top_k")
        if "shuffle" in stage2_cfg_local:
            stage2_search["shuffle"] = bool(stage2_cfg_local.get("shuffle"))
        if "random_seed" in stage2_cfg_local:
            stage2_search["random_seed"] = int(stage2_cfg_local.get("random_seed") or 1234)
        if "max_trials" in stage2_cfg_local:
            stage2_search["max_trials"] = int(stage2_cfg_local.get("max_trials") or 0)
        if int(args.stage2_max_trials) > 0:
            stage2_search["max_trials"] = int(args.stage2_max_trials)

        stage2_tm_args_override = dict(stage2_cfg_local.get("train_matrix_args") or {})
        base_tm_args = dict((base_cfg.get("train_matrix") or {}).get("args") or {})
        if "skip-build" not in stage2_tm_args_override and "skip_build" not in stage2_tm_args_override:
            base_skip_build = _to_bool(base_tm_args.get("skip-build", base_tm_args.get("skip_build", False)), False)
            if not base_skip_build:
                stage2_tm_args_override["skip-build"] = True

        stage2_cfg = _build_stage_config(
            base_cfg,
            output_root=os.path.join(controller_dir, "stage2_runs"),
            run_tag=f"{top_run_tag}_s2",
            search_cfg=stage2_search,
            train_matrix_args_override=stage2_tm_args_override,
            fixed_env_override=dict(stage2_cfg_local.get("fixed_env") or {}),
            force_dry_run=bool(args.force_dry_run),
        )

        print("[TwoStage] stage2 start")
        stage2_result = _run_one_stage(
            stage_name="stage2",
            stage_config=stage2_cfg,
            stage_work_dir=os.path.join(controller_dir, "stage2"),
            max_trials=int(stage2_search.get("max_trials", 0) or 0),
        )
        summary["stage2"] = _one_stage_summary(stage2_result)
        summary["stage2"]["derived_parameter_count"] = len(stage2_params)
        summary["stage2"]["neighbor_window"] = int(neighbor_window)

    summary_path = os.path.join(controller_dir, "two_stage_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_txt = os.path.join(controller_dir, "two_stage_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"config={config_path}\n")
        f.write(f"controller_dir={os.path.abspath(controller_dir)}\n")
        f.write(f"stage1_report={summary.get('stage1', {}).get('report_path', '')}\n")
        f.write(f"stage1_best_score={summary.get('stage1', {}).get('best_score', '')}\n")
        f.write(f"stage2_report={summary.get('stage2', {}).get('report_path', '')}\n")
        f.write(f"stage2_best_score={summary.get('stage2', {}).get('best_score', '')}\n")

    print(f"[TwoStage] summary_json={summary_path}")
    print(f"[TwoStage] summary_txt={summary_txt}")


if __name__ == "__main__":
    main()
