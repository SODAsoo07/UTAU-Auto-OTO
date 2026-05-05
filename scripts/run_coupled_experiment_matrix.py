import argparse
import datetime as dt
import itertools
import json
import os
import random
import subprocess as sp
import sys
from typing import Any, Dict, List, Tuple

try:
    import yaml
except Exception:
    yaml = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


PATH_ARG_KEYS = {
    "dataset-root",
    "workspace-root",
    "model-root",
    "rawmel-cache",
}


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _resolve_path(base_dir: str, raw_path: str) -> str:
    p = str(raw_path or "").strip()
    if not p:
        return ""
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir, p))


def _resolve_repo_path(config_dir: str, raw_path: str) -> str:
    p = str(raw_path or "").strip()
    if not p:
        return ""
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    if p.startswith("./") or p.startswith(".\\") or p.startswith("../") or p.startswith("..\\"):
        return os.path.normpath(os.path.join(config_dir, p))
    return os.path.normpath(os.path.join(ROOT_DIR, p))


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


def _env_str(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _load_yaml(path: str):
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_arg_key(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        return ""
    while key.startswith("-"):
        key = key[1:]
    key = key.replace("_", "-").strip().lower()
    return key


def _resolve_train_matrix_spec(config_dir: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    tm = dict(cfg.get("train_matrix") or {})
    script_raw = str(tm.get("script", "ml/scripts/coupled/train_matrix.py"))
    script_path = _resolve_repo_path(config_dir, script_raw)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"train_matrix script not found: {script_path}")

    fixed_args = dict(tm.get("args") or {})
    legacy_map = {
        "dataset_root": "dataset-root",
        "workspace_root": "workspace-root",
        "model_root": "model-root",
        "languages": "languages",
        "formats": "formats",
        "backend": "backend",
        "rawmel_cache": "rawmel-cache",
        "device": "device",
        "epochs": "epochs",
        "batch_size": "batch-size",
        "learning_rate": "learning-rate",
        "min_confidence": "min-confidence",
        "min_mapping_confidence": "min-mapping-confidence",
        "group_column": "group-column",
        "alias_types": "alias-types",
        "alias_family": "alias-family",
        "auto_oto_policy": "auto-oto-policy",
        "skip_build": "skip-build",
        "skip_eval": "skip-eval",
        "skip_train": "skip-train",
        "require_all": "require-all",
        "dry_run": "dry-run",
    }
    for old_key, arg_key in legacy_map.items():
        if old_key in cfg and arg_key not in fixed_args:
            fixed_args[arg_key] = cfg.get(old_key)

    out_args = {}
    for k, v in fixed_args.items():
        key = _normalize_arg_key(k)
        if not key:
            continue
        if key in PATH_ARG_KEYS and isinstance(v, str):
            out_args[key] = _resolve_repo_path(config_dir, v)
        else:
            out_args[key] = v
    return script_path, out_args


def _expand_parameter_grid(search_cfg: dict) -> List[Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]]:
    params = list((search_cfg or {}).get("parameters") or [])
    if not params:
        return [({}, {}, {})]

    parsed_params: List[Dict[str, Any]] = []
    for p in params:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        if not name:
            continue
        kind = str(p.get("kind", "env")).strip().lower()
        values = p.get("values")
        if values is None:
            values = [p.get("value")]
        if not isinstance(values, list):
            values = [values]
        values = [v for v in values if v is not None]
        if not values:
            continue
        parsed_params.append(
            {
                "name": name,
                "kind": kind,
                "values": values,
            }
        )

    if not parsed_params:
        return [({}, {}, {})]

    all_values = [p["values"] for p in parsed_params]
    out: List[Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]] = []
    for combo in itertools.product(*all_values):
        env_vars: Dict[str, str] = {}
        arg_overrides: Dict[str, Any] = {}
        raw_values: Dict[str, Any] = {}
        for param, value in zip(parsed_params, combo):
            kind = str(param["kind"]).strip().lower()
            if kind in {"arg", "args", "cli", "option"}:
                key = _normalize_arg_key(param["name"])
                if key:
                    arg_overrides[key] = value
                    raw_values[f"arg:{key}"] = value
                continue
            env_name = str(param["name"]).strip()
            if env_name:
                env_vars[env_name] = _env_str(value)
                raw_values[f"env:{env_name}"] = value
        out.append((env_vars, arg_overrides, raw_values))
    return out


def _value_to_arg_tokens(name: str, value: Any) -> List[str]:
    key = _normalize_arg_key(name)
    if not key:
        return []
    flag = f"--{key}"
    if value is None:
        return []
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        joined = ",".join(str(v) for v in value if v is not None)
        if not joined:
            return []
        return [flag, joined]
    text = str(value).strip()
    if text == "":
        return []
    return [flag, text]


def _build_command(python_exe: str, script_path: str, args_map: Dict[str, Any]) -> List[str]:
    cmd = [python_exe, script_path]
    for k in sorted(args_map.keys()):
        cmd.extend(_value_to_arg_tokens(k, args_map.get(k)))
    return cmd


def _extract_tail_json(stdout_text: str) -> Dict[str, Any]:
    text = str(stdout_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "{":
            continue
        chunk = text[i:]
        try:
            obj, end = decoder.raw_decode(chunk)
        except Exception:
            continue
        rest = chunk[end:].strip()
        if rest:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def _target_metrics_from_eval(eval_summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if not isinstance(eval_summary, dict):
        return {}

    # Current format (training.evaluate_coupled_bundle): eval_summary["targets"][target]["model_mae"].
    targets = eval_summary.get("targets")
    if isinstance(targets, dict):
        sample_val = next(iter(targets.values()), None)
        if isinstance(sample_val, dict):
            out = {}
            for name, row in targets.items():
                if not isinstance(row, dict):
                    continue
                try:
                    out[str(name)] = {
                        "baseline_mae": float(row.get("baseline_mae", row.get("baseline", 0.0)) or 0.0),
                        "model_mae": float(row.get("model_mae", row.get("mae", 0.0)) or 0.0),
                    }
                except Exception:
                    continue
            if out:
                return out

    # Legacy format in older reports: eval_summary["metrics"][target]["model_mae"].
    metrics = eval_summary.get("metrics")
    if isinstance(metrics, dict):
        out = {}
        for name, row in metrics.items():
            if not isinstance(row, dict):
                continue
            try:
                out[str(name)] = {
                    "baseline_mae": float(row.get("baseline_mae", row.get("baseline", 0.0)) or 0.0),
                    "model_mae": float(row.get("model_mae", row.get("mae", 0.0)) or 0.0),
                }
            except Exception:
                continue
        return out
    return {}


def _job_weight(weight_cfg: Dict[str, Any], language: str, fmt: str) -> float:
    if not isinstance(weight_cfg, dict):
        return 1.0
    lang = str(language or "").strip().lower()
    f = str(fmt or "").strip().lower()
    keys = [
        f"{lang}/{f}",
        f"{lang}/*",
        lang,
        f"*/{f}",
        "default",
        "*",
    ]
    for key in keys:
        if key in weight_cfg:
            try:
                return max(0.0, float(weight_cfg.get(key)))
            except Exception:
                return 1.0
    return 1.0


def _score_job(job: Dict[str, Any], objective: Dict[str, Any]) -> Dict[str, Any]:
    lang = str(job.get("language", "")).strip().lower()
    fmt = str(job.get("format_type", "")).strip().lower()
    status = str(job.get("status", "")).strip().lower()
    weight_cfg = dict(objective.get("job_weights") or {})
    target_weight_cfg = dict(objective.get("target_weights") or {})

    job_weight = _job_weight(weight_cfg, lang, fmt)
    failed_job_penalty = float(objective.get("failed_job_penalty", 2200.0))
    missing_eval_penalty = float(objective.get("missing_eval_penalty", 1200.0))
    missing_target_penalty = float(objective.get("missing_target_penalty", 180.0))
    regression_penalty_weight = float(objective.get("regression_penalty_weight", 1.5))
    confidence_penalty_weight = float(objective.get("confidence_penalty_weight", 20.0))
    low_rows_threshold = int(objective.get("low_rows_threshold", 0) or 0)
    low_rows_penalty_per_row = float(objective.get("low_rows_penalty_per_row", 0.0))

    if status not in {"ok"}:
        base = failed_job_penalty
        return {
            "language": lang,
            "format_type": fmt,
            "status": status or "failed",
            "job_weight": job_weight,
            "base_score": base,
            "weighted_score": base * job_weight,
            "reason": str(job.get("error") or "job_not_ok"),
        }

    eval_summary = job.get("eval_summary")
    if not isinstance(eval_summary, dict):
        base = missing_eval_penalty
        return {
            "language": lang,
            "format_type": fmt,
            "status": "missing_eval",
            "job_weight": job_weight,
            "base_score": base,
            "weighted_score": base * job_weight,
            "reason": "eval_summary_missing",
        }

    target_metrics = _target_metrics_from_eval(eval_summary)
    if not target_metrics:
        base = missing_eval_penalty
        return {
            "language": lang,
            "format_type": fmt,
            "status": "missing_eval",
            "job_weight": job_weight,
            "base_score": base,
            "weighted_score": base * job_weight,
            "reason": "target_metrics_missing",
        }

    if target_weight_cfg:
        targets = [str(t) for t in target_weight_cfg.keys()]
    else:
        targets = sorted(target_metrics.keys())

    mae_sum = 0.0
    reg_sum = 0.0
    weight_sum = 0.0
    details = {}
    missing_targets = 0
    for target in targets:
        tm = target_metrics.get(target)
        target_w = float(target_weight_cfg.get(target, 1.0)) if target_weight_cfg else 1.0
        target_w = max(0.0, target_w)
        if target_w <= 0.0:
            continue
        if not isinstance(tm, dict):
            missing_targets += 1
            continue
        model_mae = float(tm.get("model_mae", 0.0))
        baseline_mae = float(tm.get("baseline_mae", model_mae))
        mae_sum += target_w * model_mae
        reg_sum += target_w * max(0.0, model_mae - baseline_mae)
        weight_sum += target_w
        details[target] = {
            "baseline_mae": baseline_mae,
            "model_mae": model_mae,
            "improved": model_mae <= baseline_mae,
        }

    if weight_sum <= 0.0:
        base = missing_eval_penalty
        return {
            "language": lang,
            "format_type": fmt,
            "status": "missing_eval",
            "job_weight": job_weight,
            "base_score": base,
            "weighted_score": base * job_weight,
            "reason": "no_weighted_targets",
        }

    mae_score = mae_sum / weight_sum
    reg_score = (reg_sum / weight_sum) * regression_penalty_weight
    confidence = float(eval_summary.get("confidence_mean", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    conf_score = (1.0 - confidence) * confidence_penalty_weight
    missing_score = float(missing_targets) * missing_target_penalty
    rows = int(eval_summary.get("rows", 0) or 0)
    low_rows_score = 0.0
    if low_rows_threshold > 0 and rows < low_rows_threshold:
        low_rows_score = float(low_rows_threshold - rows) * low_rows_penalty_per_row

    base_score = float(mae_score + reg_score + conf_score + missing_score + low_rows_score)
    weighted = base_score * job_weight
    return {
        "language": lang,
        "format_type": fmt,
        "status": "ok",
        "rows": rows,
        "job_weight": job_weight,
        "base_score": base_score,
        "weighted_score": weighted,
        "components": {
            "mae": mae_score,
            "regression": reg_score,
            "confidence": conf_score,
            "missing_targets": missing_score,
            "low_rows": low_rows_score,
        },
        "targets": details,
    }


def _score_trial(summary: Dict[str, Any], objective: Dict[str, Any]) -> Dict[str, Any]:
    jobs = list(summary.get("jobs") or [])
    if not jobs:
        failed_trial_penalty = float(objective.get("failed_trial_penalty", 12000.0))
        return {
            "score": failed_trial_penalty,
            "job_scores": [],
            "failed_jobs": 1,
            "reason": "summary_jobs_missing",
        }

    job_scores = []
    total_score = 0.0
    failed_jobs = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        row = _score_job(job, objective)
        job_scores.append(row)
        total_score += float(row.get("weighted_score", 0.0))
        if str(row.get("status", "")).lower() != "ok":
            failed_jobs += 1

    return {
        "score": float(total_score),
        "job_scores": job_scores,
        "failed_jobs": int(failed_jobs),
    }


def _resolve_top_k(search_cfg: Dict[str, Any]) -> int:
    try:
        return max(1, int(search_cfg.get("top_k", 5) or 5))
    except Exception:
        return 5


def main():
    ap = argparse.ArgumentParser(description="Run coupled train_matrix sweeps and rank trials.")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--max-trials", type=int, default=0, help="Override max trials (0=from config)")
    args = ap.parse_args()

    config_path = os.path.abspath(args.config)
    cfg = _load_yaml(config_path)
    config_dir = os.path.dirname(config_path)

    output_root = _resolve_repo_path(config_dir, str(cfg.get("output_root", "logs/coupled_experiments")))
    run_tag = _safe_name(str(cfg.get("run_tag", "coupled_exp")).strip() or "coupled_exp")
    run_dir = os.path.join(output_root, f"{run_tag}.{_now_tag()}")
    os.makedirs(run_dir, exist_ok=True)
    trials_dir = os.path.join(run_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    script_path, fixed_args = _resolve_train_matrix_spec(config_dir, cfg)
    fixed_env = dict(cfg.get("fixed_env") or {})
    search_cfg = dict(cfg.get("search") or {})
    objective = dict(cfg.get("objective") or {})

    top_k = _resolve_top_k(search_cfg)
    max_trials_cfg = int(search_cfg.get("max_trials", 0) or 0)
    max_trials = int(args.max_trials) if int(args.max_trials) > 0 else max_trials_cfg
    shuffle = _to_bool(search_cfg.get("shuffle", False), False)
    random_seed = int(search_cfg.get("random_seed", 1234) or 1234)
    stop_on_error = _to_bool(cfg.get("stop_on_error", False), False)
    timeout_sec = int(cfg.get("trial_timeout_sec", 0) or 0)
    per_trial_model_root = _to_bool(cfg.get("per_trial_model_root", True), True)
    trial_model_root_base = _resolve_repo_path(
        config_dir,
        str(cfg.get("trial_model_root_base", os.path.join(run_dir, "models"))),
    )
    os.makedirs(trial_model_root_base, exist_ok=True)

    grid = _expand_parameter_grid(search_cfg)
    if shuffle and len(grid) > 1:
        rnd = random.Random(random_seed)
        rnd.shuffle(grid)
    if max_trials > 0:
        grid = grid[:max_trials]

    print(f"[CoupledTune] config={config_path}")
    print(f"[CoupledTune] script={script_path}")
    print(f"[CoupledTune] run_dir={os.path.abspath(run_dir)}")
    print(f"[CoupledTune] trials={len(grid)}")

    trial_results: List[Dict[str, Any]] = []
    for trial_idx, (trial_env_map, trial_arg_map, raw_values) in enumerate(grid, start=1):
        trial_tag = f"{run_tag}.t{trial_idx:03d}"
        trial_log = os.path.join(trials_dir, f"{trial_idx:03d}.train_matrix.log")

        merged_args = dict(fixed_args)
        for k, v in trial_arg_map.items():
            key = _normalize_arg_key(k)
            if key in PATH_ARG_KEYS and isinstance(v, str):
                merged_args[key] = _resolve_repo_path(config_dir, v)
            else:
                merged_args[key] = v
        if per_trial_model_root:
            merged_args["model-root"] = os.path.join(trial_model_root_base, f"trial_{trial_idx:03d}")
        if "workspace-root" in merged_args and isinstance(merged_args.get("workspace-root"), str):
            os.makedirs(str(merged_args["workspace-root"]), exist_ok=True)
        if "model-root" in merged_args and isinstance(merged_args.get("model-root"), str):
            os.makedirs(str(merged_args["model-root"]), exist_ok=True)

        cmd = _build_command(sys.executable, script_path, merged_args)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        for k, v in fixed_env.items():
            env[str(k)] = _env_str(v)
        for k, v in trial_env_map.items():
            env[str(k)] = str(v)

        run_error = ""
        stdout = ""
        stderr = ""
        print(f"[CoupledTune] trial {trial_idx}/{len(grid)} start")
        try:
            completed = sp.run(
                cmd,
                cwd=ROOT_DIR,
                env=env,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=(timeout_sec if timeout_sec > 0 else None),
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            if completed.returncode != 0:
                run_error = f"return_code={completed.returncode}"
        except Exception as exc:
            run_error = str(exc)

        with open(trial_log, "w", encoding="utf-8") as f:
            f.write(stdout or "")
            if stderr:
                f.write("\n[STDERR]\n")
                f.write(stderr)
            if run_error:
                f.write("\n[RUN_ERROR]\n")
                f.write(str(run_error))

        parsed_summary = _extract_tail_json(stdout)
        if not parsed_summary and not run_error:
            run_error = "summary_json_parse_failed"

        if parsed_summary:
            score_info = _score_trial(parsed_summary, objective)
            trial_score = float(score_info.get("score", float(objective.get("failed_trial_penalty", 12000.0))))
            job_scores = list(score_info.get("job_scores") or [])
            failed_jobs = int(score_info.get("failed_jobs", 0))
        else:
            trial_score = float(objective.get("failed_trial_penalty", 12000.0))
            failed_jobs = 1
            job_scores = []

        trial_row = {
            "trial_index": trial_idx,
            "trial_tag": trial_tag,
            "command": cmd,
            "env": dict(trial_env_map),
            "raw_values": dict(raw_values),
            "fixed_env": dict(fixed_env),
            "args": dict(merged_args),
            "run_error": run_error,
            "log_file": trial_log,
            "summary": parsed_summary,
            "job_scores": job_scores,
            "failed_jobs": failed_jobs,
            "score": trial_score,
        }
        trial_results.append(trial_row)

        with open(os.path.join(trials_dir, f"{trial_idx:03d}.result.json"), "w", encoding="utf-8") as f:
            json.dump(trial_row, f, ensure_ascii=False, indent=2)

        print(f"[CoupledTune] trial {trial_idx}/{len(grid)} done score={trial_score:.4f}")
        if run_error:
            print(f"[CoupledTune] trial {trial_idx} error={run_error}")
            if stop_on_error:
                print("[CoupledTune] stop_on_error=true, terminating sweep.")
                break

    ranked = sorted(trial_results, key=lambda r: float(r.get("score", 1e18)))
    best = ranked[0] if ranked else None
    top = ranked[: max(1, top_k)]

    report = {
        "config": config_path,
        "script_path": script_path,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": os.path.abspath(run_dir),
        "trials": len(trial_results),
        "objective": objective,
        "best": best,
        "top": top,
        "results": trial_results,
    }

    report_json = os.path.join(run_dir, "experiment_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_txt = os.path.join(run_dir, "experiment_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"config={config_path}\n")
        f.write(f"script={script_path}\n")
        f.write(f"run_dir={os.path.abspath(run_dir)}\n")
        f.write(f"trials={len(trial_results)}\n\n")
        for rank_idx, row in enumerate(top, start=1):
            f.write(f"[{rank_idx}] score={float(row.get('score', 0.0)):.4f}\n")
            f.write(f"trial_tag={row.get('trial_tag', '')}\n")
            f.write(f"env={json.dumps(row.get('env', {}), ensure_ascii=False)}\n")
            f.write(f"args={json.dumps(row.get('args', {}), ensure_ascii=False)}\n")
            if row.get("run_error"):
                f.write(f"run_error={row.get('run_error')}\n")
            for js in list(row.get("job_scores") or []):
                f.write(
                    f"  - {js.get('language','')}/{js.get('format_type','')} "
                    f"status={js.get('status','')} weighted_score={float(js.get('weighted_score', 0.0)):.4f}\n"
                )
                components = js.get("components") or {}
                if components:
                    f.write(
                        "    "
                        f"mae={float(components.get('mae', 0.0)):.4f} "
                        f"reg={float(components.get('regression', 0.0)):.4f} "
                        f"conf={float(components.get('confidence', 0.0)):.4f} "
                        f"missing={float(components.get('missing_targets', 0.0)):.4f} "
                        f"low_rows={float(components.get('low_rows', 0.0)):.4f}\n"
                    )
            f.write("\n")

    print(f"[CoupledTune] report_json={report_json}")
    print(f"[CoupledTune] report_txt={report_txt}")
    if best:
        print(f"[CoupledTune] best_score={float(best.get('score', 0.0)):.4f} env={best.get('env', {})}")
    else:
        print("[CoupledTune] no trial results")


if __name__ == "__main__":
    main()
