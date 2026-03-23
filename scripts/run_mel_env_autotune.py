import argparse
import datetime as dt
import itertools
import json
import os
import random
import subprocess as sp
import sys
from typing import Dict, List, Tuple

try:
    import yaml
except Exception:
    yaml = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.oto_file_utils import build_occurrence_map, read_oto_rows_for_profile
from core.oto_normalization import canonicalize_alias_for_matching, normalize_wav_key
from core.kr_oto_rules import should_ignore_korean_alias


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "case"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "case"


def _resolve_path(base_dir: str, raw_path: str) -> str:
    p = str(raw_path or "").strip()
    if not p:
        return ""
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir, p))


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


def _load_batch_case_index(batch_config_path: str) -> Dict[str, Dict[str, str]]:
    data = _load_yaml(batch_config_path)
    defaults = data.get("defaults") or {}
    cases = data.get("cases") or []
    if not isinstance(defaults, dict) or not isinstance(cases, list):
        raise ValueError("batch_config must include mapping defaults and list cases")

    config_dir = os.path.dirname(batch_config_path)
    out: Dict[str, Dict[str, str]] = {}
    for idx, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            continue
        name = str(case.get("name", f"case_{idx:03d}")).strip() or f"case_{idx:03d}"
        voicebank_dir = _resolve_path(
            config_dir,
            str(case.get("voicebank_dir", case.get("wav_dir", defaults.get("voicebank_dir", "")))).strip(),
        )
        language = str(case.get("language", defaults.get("language", ""))).strip().lower()
        out[name] = {
            "voicebank_dir": voicebank_dir,
            "language": language,
        }
    return out


def _read_rows_for_compare(path: str, language: str):
    lang = str(language or "").strip().lower()
    alias_filter = should_ignore_korean_alias if lang == "korean" else None
    return read_oto_rows_for_profile(
        path,
        alias_filter=alias_filter,
        wav_normalizer=normalize_wav_key,
        alias_normalizer=lambda a: canonicalize_alias_for_matching(lang, a),
    )


def _compare_oto_pair(auto_oto: str, ref_oto: str, language: str) -> Dict[str, float]:
    auto_rows = _read_rows_for_compare(auto_oto, language)
    ref_rows = _read_rows_for_compare(ref_oto, language)
    auto_map = build_occurrence_map(auto_rows)
    ref_map = build_occurrence_map(ref_rows)

    if not ref_map:
        return {
            "matched": 0,
            "auto_rows": len(auto_rows),
            "ref_rows": len(ref_rows),
            "missing_auto": 0,
            "missing_ref": 0,
            "coverage": 0.0,
            "mae_offset": 9999.0,
            "mae_cons": 9999.0,
            "mae_cutoff": 9999.0,
            "mae_pre": 9999.0,
            "mae_ovl": 9999.0,
        }

    common_keys = sorted(set(auto_map.keys()) & set(ref_map.keys()))
    if not common_keys:
        return {
            "matched": 0,
            "auto_rows": len(auto_rows),
            "ref_rows": len(ref_rows),
            "missing_auto": len(ref_map),
            "missing_ref": len(auto_map),
            "coverage": 0.0,
            "mae_offset": 9999.0,
            "mae_cons": 9999.0,
            "mae_cutoff": 9999.0,
            "mae_pre": 9999.0,
            "mae_ovl": 9999.0,
        }

    sum_offset = 0.0
    sum_cons = 0.0
    sum_cutoff = 0.0
    sum_pre = 0.0
    sum_ovl = 0.0
    for k in common_keys:
        a = auto_map[k]
        r = ref_map[k]
        sum_offset += abs(float(a["offset"]) - float(r["offset"]))
        sum_cons += abs(float(a["cons"]) - float(r["cons"]))
        sum_cutoff += abs(abs(float(a["cutoff"])) - abs(float(r["cutoff"])))
        sum_pre += abs(float(a["pre"]) - float(r["pre"]))
        sum_ovl += abs(float(a["ovl"]) - float(r["ovl"]))

    matched = len(common_keys)
    ref_count = len(ref_map)
    auto_count = len(auto_map)
    missing_auto = max(0, ref_count - matched)
    missing_ref = max(0, auto_count - matched)
    coverage = float(matched) / float(ref_count) if ref_count > 0 else 0.0
    return {
        "matched": matched,
        "auto_rows": auto_count,
        "ref_rows": ref_count,
        "missing_auto": missing_auto,
        "missing_ref": missing_ref,
        "coverage": coverage,
        "mae_offset": sum_offset / matched,
        "mae_cons": sum_cons / matched,
        "mae_cutoff": sum_cutoff / matched,
        "mae_pre": sum_pre / matched,
        "mae_ovl": sum_ovl / matched,
    }


def _score_case(
    metrics: Dict[str, float],
    *,
    field_weights: Dict[str, float],
    missing_pair_penalty: float,
    coverage_penalty: float,
) -> float:
    w_offset = float(field_weights.get("offset", 1.0))
    w_cons = float(field_weights.get("cons", 1.0))
    w_cutoff = float(field_weights.get("cutoff", 1.0))
    w_pre = float(field_weights.get("pre", 1.0))
    w_ovl = float(field_weights.get("ovl", 0.7))
    denom = max(1e-9, w_offset + w_cons + w_cutoff + w_pre + w_ovl)

    mae_weighted = (
        w_offset * float(metrics.get("mae_offset", 9999.0))
        + w_cons * float(metrics.get("mae_cons", 9999.0))
        + w_cutoff * float(metrics.get("mae_cutoff", 9999.0))
        + w_pre * float(metrics.get("mae_pre", 9999.0))
        + w_ovl * float(metrics.get("mae_ovl", 9999.0))
    ) / denom

    missing_count = float(metrics.get("missing_auto", 0) + metrics.get("missing_ref", 0))
    coverage = max(0.0, min(1.0, float(metrics.get("coverage", 0.0))))
    return float(mae_weighted) + missing_count * float(missing_pair_penalty) + (1.0 - coverage) * float(coverage_penalty)


def _expand_parameter_grid(search_cfg: dict) -> List[Tuple[Dict[str, str], Dict[str, object]]]:
    params = list((search_cfg or {}).get("parameters") or [])
    if not params:
        return [({}, {})]

    names: List[str] = []
    values_list: List[List[object]] = []
    for p in params:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        vals = list(p.get("values") or [])
        if not name:
            continue
        if not vals:
            raise ValueError(f"parameter '{name}' has empty values")
        names.append(name)
        values_list.append(vals)

    if not names:
        return [({}, {})]

    out: List[Tuple[Dict[str, str], Dict[str, object]]] = []
    for combo in itertools.product(*values_list):
        env_map: Dict[str, str] = {}
        raw_map: Dict[str, object] = {}
        for k, v in zip(names, combo):
            env_map[k] = _env_str(v)
            raw_map[k] = v
        out.append((env_map, raw_map))
    return out


def _prepare_targets(config_dir: str, tune_cfg: dict, batch_case_idx: Dict[str, Dict[str, str]]) -> List[dict]:
    targets = list(tune_cfg.get("cases") or [])
    if not targets:
        raise ValueError("tuning config 'cases' is empty")

    out: List[dict] = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        if name not in batch_case_idx:
            raise ValueError(f"case '{name}' is not found in batch_config")
        batch_case = batch_case_idx[name]
        language = str(t.get("language", batch_case.get("language", "korean"))).strip().lower() or "korean"
        ref_raw = str(t.get("reference_oto", "")).strip()
        if not ref_raw:
            raise ValueError(f"case '{name}' missing reference_oto")

        voicebank_dir = str(batch_case.get("voicebank_dir", "")).strip()
        ref_path = _resolve_path(config_dir, ref_raw)
        if voicebank_dir and not os.path.isabs(ref_raw):
            vb_candidate = os.path.normpath(os.path.join(voicebank_dir, ref_raw))
            if os.path.exists(vb_candidate):
                ref_path = vb_candidate
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"reference_oto not found for case '{name}': {ref_path}")

        out.append(
            {
                "name": name,
                "language": language,
                "weight": float(t.get("weight", 1.0) or 1.0),
                "reference_oto": ref_path,
            }
        )
    if not out:
        raise ValueError("no valid tuning cases resolved")
    return out


def main():
    parser = argparse.ArgumentParser(description="Grid-tune MEL-related runtime env vars against manual OTO golden set.")
    parser.add_argument("--config", required=True, help="Path to tuning YAML config.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    config_dir = os.path.dirname(config_path)
    cfg = _load_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ValueError("tuning config root must be a mapping")

    batch_config = _resolve_path(config_dir, str(cfg.get("batch_config", "")).strip())
    if not batch_config or not os.path.exists(batch_config):
        raise FileNotFoundError(f"batch_config not found: {batch_config}")

    output_root = _resolve_path(config_dir, str(cfg.get("output_root", "logs/oto_tune")).strip())
    run_tag = str(cfg.get("run_tag", "")).strip() or _now_tag()
    run_dir = os.path.join(output_root, _safe_name(run_tag))
    trials_dir = os.path.join(run_dir, "trials")
    os.makedirs(trials_dir, exist_ok=True)

    fixed_env = dict(cfg.get("fixed_env") or {})
    search_cfg = dict(cfg.get("search") or {})
    objective = dict(cfg.get("objective") or {})
    field_weights = dict(objective.get("field_weights") or {})
    missing_pair_penalty = float(objective.get("missing_pair_penalty", 40.0))
    coverage_penalty = float(objective.get("coverage_penalty", 220.0))
    failed_case_penalty = float(objective.get("failed_case_penalty", 6000.0))
    top_k = int(search_cfg.get("top_k", 5) or 5)
    max_trials = int(search_cfg.get("max_trials", 0) or 0)
    shuffle = _to_bool(search_cfg.get("shuffle", False), False)
    random_seed = int(search_cfg.get("random_seed", 1234) or 1234)
    timeout_sec = int(cfg.get("trial_timeout_sec", 0) or 0)
    stop_on_error = _to_bool(cfg.get("stop_on_error", False), False)
    skip_validation = _to_bool(cfg.get("skip_validation", True), True)

    grid = _expand_parameter_grid(search_cfg)
    if shuffle and len(grid) > 1:
        rnd = random.Random(random_seed)
        rnd.shuffle(grid)
    if max_trials > 0:
        grid = grid[:max_trials]

    batch_case_idx = _load_batch_case_index(batch_config)
    targets = _prepare_targets(config_dir, cfg, batch_case_idx)

    print(f"[Tune] config={config_path}")
    print(f"[Tune] batch_config={batch_config}")
    print(f"[Tune] run_dir={os.path.abspath(run_dir)}")
    print(f"[Tune] targets={len(targets)}")
    print(f"[Tune] trials={len(grid)}")

    trial_results = []
    for trial_idx, (trial_env_raw, trial_raw_values) in enumerate(grid, start=1):
        trial_tag = f"{_safe_name(run_tag)}.t{trial_idx:03d}"
        trial_log = os.path.join(trials_dir, f"{trial_idx:03d}.batch.log")
        summary_json = os.path.join(ROOT_DIR, "logs", "oto_batch", trial_tag, "summary.json")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        for k, v in fixed_env.items():
            env[str(k)] = _env_str(v)
        for k, v in trial_env_raw.items():
            env[str(k)] = str(v)

        cmd = [
            sys.executable,
            os.path.join(ROOT_DIR, "scripts", "run_oto_generation_batch.py"),
            "--config",
            batch_config,
            "--run-tag",
            trial_tag,
        ]
        if skip_validation:
            cmd.append("--skip-validation")

        print(f"[Tune] trial {trial_idx}/{len(grid)} start env={trial_env_raw}")
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
            with open(trial_log, "w", encoding="utf-8") as f:
                f.write(completed.stdout or "")
                if completed.stderr:
                    f.write("\n[STDERR]\n")
                    f.write(completed.stderr)
            run_error = "" if completed.returncode == 0 else f"batch_return_code={completed.returncode}"
        except Exception as e:
            run_error = str(e)
            with open(trial_log, "w", encoding="utf-8") as f:
                f.write(f"[EXCEPTION] {run_error}\n")
            completed = None

        case_metrics = []
        total_score = 0.0
        if run_error or not os.path.exists(summary_json):
            run_error = run_error or f"summary_missing:{summary_json}"
            for t in targets:
                case_score = failed_case_penalty * float(t.get("weight", 1.0))
                case_metrics.append(
                    {
                        "name": t["name"],
                        "language": t["language"],
                        "reference_oto": t["reference_oto"],
                        "status": "error",
                        "error": run_error,
                        "score": case_score,
                    }
                )
                total_score += case_score
        else:
            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
            by_name = {}
            for row in list(summary.get("results") or []):
                if isinstance(row, dict):
                    by_name[str(row.get("name", "")).strip()] = row

            for t in targets:
                name = str(t["name"])
                lang = str(t["language"])
                weight = float(t.get("weight", 1.0))
                ref_oto = str(t["reference_oto"])
                row = by_name.get(name)
                if not row:
                    case_score = failed_case_penalty * weight
                    case_metrics.append(
                        {
                            "name": name,
                            "language": lang,
                            "reference_oto": ref_oto,
                            "status": "error",
                            "error": "case_missing_in_summary",
                            "score": case_score,
                        }
                    )
                    total_score += case_score
                    continue

                status = str(row.get("status", "")).strip().lower()
                out_oto = str(row.get("output_oto", "")).strip()
                if status != "ok" or not out_oto or not os.path.exists(out_oto):
                    case_score = failed_case_penalty * weight
                    case_metrics.append(
                        {
                            "name": name,
                            "language": lang,
                            "reference_oto": ref_oto,
                            "output_oto": out_oto,
                            "status": "error",
                            "error": f"batch_status={status or 'unknown'}",
                            "score": case_score,
                        }
                    )
                    total_score += case_score
                    continue

                m = _compare_oto_pair(out_oto, ref_oto, lang)
                s = _score_case(
                    m,
                    field_weights=field_weights,
                    missing_pair_penalty=missing_pair_penalty,
                    coverage_penalty=coverage_penalty,
                )
                ws = s * weight
                total_score += ws
                case_metrics.append(
                    {
                        "name": name,
                        "language": lang,
                        "reference_oto": ref_oto,
                        "output_oto": out_oto,
                        "status": "ok",
                        "metrics": m,
                        "case_score": s,
                        "weight": weight,
                        "score": ws,
                    }
                )

        trial_row = {
            "trial_index": trial_idx,
            "trial_tag": trial_tag,
            "env": dict(trial_env_raw),
            "raw_values": dict(trial_raw_values),
            "fixed_env": dict(fixed_env),
            "batch_log": trial_log,
            "summary_json": summary_json,
            "run_error": run_error,
            "score": total_score,
            "cases": case_metrics,
        }
        trial_results.append(trial_row)

        with open(os.path.join(trials_dir, f"{trial_idx:03d}.result.json"), "w", encoding="utf-8") as f:
            json.dump(trial_row, f, ensure_ascii=False, indent=2)

        print(f"[Tune] trial {trial_idx}/{len(grid)} done score={total_score:.4f}")
        if run_error:
            print(f"[Tune] trial {trial_idx} error={run_error}")
            if stop_on_error:
                print("[Tune] stop_on_error=true, terminating sweep.")
                break

    ranked = sorted(trial_results, key=lambda r: float(r.get("score", 1e18)))
    best = ranked[0] if ranked else None
    top = ranked[: max(1, top_k)]

    report = {
        "config": config_path,
        "batch_config": batch_config,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": os.path.abspath(run_dir),
        "trials": len(trial_results),
        "targets": targets,
        "objective": {
            "field_weights": field_weights,
            "missing_pair_penalty": missing_pair_penalty,
            "coverage_penalty": coverage_penalty,
            "failed_case_penalty": failed_case_penalty,
        },
        "best": best,
        "top": top,
        "results": trial_results,
    }

    report_json = os.path.join(run_dir, "tune_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_txt = os.path.join(run_dir, "tune_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"config={config_path}\n")
        f.write(f"batch_config={batch_config}\n")
        f.write(f"run_dir={os.path.abspath(run_dir)}\n")
        f.write(f"trials={len(trial_results)}\n\n")
        for idx, row in enumerate(top, start=1):
            f.write(f"[{idx}] score={float(row.get('score', 0.0)):.4f}\n")
            f.write(f"trial_tag={row.get('trial_tag', '')}\n")
            f.write(f"env={json.dumps(row.get('env', {}), ensure_ascii=False)}\n")
            if row.get("run_error"):
                f.write(f"run_error={row.get('run_error')}\n")
            for c in list(row.get("cases") or []):
                f.write(
                    f"  - {c.get('name')} status={c.get('status')} score={float(c.get('score', 0.0)):.4f}\n"
                )
                m = c.get("metrics") or {}
                if m:
                    f.write(
                        "    "
                        f"matched={int(m.get('matched', 0))} "
                        f"coverage={float(m.get('coverage', 0.0)):.3f} "
                        f"mae(offset/cons/cut/pre/ovl)="
                        f"{float(m.get('mae_offset', 0.0)):.2f}/"
                        f"{float(m.get('mae_cons', 0.0)):.2f}/"
                        f"{float(m.get('mae_cutoff', 0.0)):.2f}/"
                        f"{float(m.get('mae_pre', 0.0)):.2f}/"
                        f"{float(m.get('mae_ovl', 0.0)):.2f}\n"
                    )
            f.write("\n")

    print(f"[Tune] report_json={report_json}")
    print(f"[Tune] report_txt={report_txt}")
    if best:
        print(f"[Tune] best_score={float(best.get('score', 0.0)):.4f} env={best.get('env', {})}")
    else:
        print("[Tune] no trial results")


if __name__ == "__main__":
    main()

