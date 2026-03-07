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

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from scripts.analyze_oto_diff import analyze as analyze_oto_diff
from scripts.compare_oto_batch_runs import compare_runs


ENV_KEYS = (
    "UTOA_JA_MAPPING_CONF_THRESHOLD",
    "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD",
    "UTOA_KR_MAPPING_CONF_THRESHOLD",
    "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD",
)


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "case"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "case"


def _to_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _resolve_path(config_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(config_dir, p))


def _resolve_case_path(config_dir: str, voicebank_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))
    if os.path.isabs(p):
        return os.path.normpath(p)
    vb_candidate = os.path.normpath(os.path.join(voicebank_dir, p))
    config_candidate = os.path.normpath(os.path.join(config_dir, p))
    if os.path.exists(vb_candidate):
        return vb_candidate
    if os.path.exists(config_candidate):
        return config_candidate
    return vb_candidate


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


def _build_runtime_batch_config(
    config_path: str,
    runtime_dir: str,
    run_tag: str,
) -> Dict[str, object]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to build runtime tuning configs.")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid batch config: {config_path}")

    defaults = data.get("defaults") or {}
    cases = data.get("cases") or []
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list.")

    config_dir = os.path.dirname(os.path.abspath(config_path))
    generated_root = os.path.join(runtime_dir, "generated_oto")
    os.makedirs(generated_root, exist_ok=True)

    runtime_defaults = dict(defaults)
    runtime_defaults["replace_oto_ini"] = False
    runtime_defaults["out_dir"] = generated_root
    runtime_defaults["output_name_template"] = "oto.{run_tag}.{name}.ini"
    runtime_defaults["base_oto"] = ""
    runtime_defaults["custom_phonemes_path"] = ""

    runtime_cases = []
    case_meta: Dict[str, Dict[str, object]] = {}
    seen_case_names = set()
    seen_safe_case_names: Dict[str, str] = {}

    for idx, raw_case in enumerate(cases, start=1):
        case = dict(raw_case) if isinstance(raw_case, dict) else {}
        name = str(case.get("name", f"case_{idx:03d}")).strip() or f"case_{idx:03d}"
        if name in seen_case_names:
            raise ValueError(f"duplicate case name in batch config: {name}")
        seen_case_names.add(name)
        safe_case_name = _safe_name(name)
        prev_safe_name = seen_safe_case_names.get(safe_case_name)
        if prev_safe_name and prev_safe_name != name:
            raise ValueError(
                f"generated output collision after sanitizing case names: "
                f"{prev_safe_name} vs {name} -> {safe_case_name}"
            )
        seen_safe_case_names[safe_case_name] = name
        language = str(case.get("language", defaults.get("language", "japanese"))).strip().lower()
        voicebank_dir = _resolve_path(
            config_dir,
            str(case.get("voicebank_dir", case.get("wav_dir", defaults.get("voicebank_dir", "")))).strip(),
        )
        textgrid_dir = _resolve_case_path(
            config_dir,
            voicebank_dir,
            str(case.get("textgrid_dir", case.get("tg_folder", defaults.get("textgrid_dir", "textgrids")))).strip(),
        )
        no_base_oto = _to_bool(case.get("no_base_oto", defaults.get("no_base_oto", False)), False)
        base_oto_raw = str(case.get("base_oto", defaults.get("base_oto", ""))).strip()
        base_oto_path = ""
        if (not no_base_oto) and base_oto_raw:
            base_oto_path = _resolve_case_path(config_dir, voicebank_dir, base_oto_raw)

        custom_phonemes_raw = str(
            case.get("custom_phonemes_path", defaults.get("custom_phonemes_path", ""))
        ).strip()
        custom_phonemes_path = _resolve_path(config_dir, custom_phonemes_raw) if custom_phonemes_raw else ""

        runtime_case = dict(case)
        runtime_case["voicebank_dir"] = voicebank_dir
        runtime_case["textgrid_dir"] = textgrid_dir
        runtime_case["replace_oto_ini"] = False
        runtime_case["out_dir"] = os.path.join(generated_root, safe_case_name)
        runtime_case["output_name"] = f"oto.{run_tag}.{safe_case_name}.ini"
        runtime_case.pop("output_name_template", None)
        if base_oto_path:
            runtime_case["base_oto"] = base_oto_path
        elif "base_oto" in runtime_case and not base_oto_raw:
            runtime_case["base_oto"] = ""
        if custom_phonemes_path:
            runtime_case["custom_phonemes_path"] = custom_phonemes_path

        runtime_cases.append(runtime_case)
        case_meta[name] = {
            "language": language,
            "base_oto": base_oto_path,
            "no_base_oto": bool(no_base_oto),
            "generated_out_dir": runtime_case["out_dir"],
            "generated_output_name": runtime_case["output_name"],
        }

    runtime_config = {
        "defaults": runtime_defaults,
        "cases": runtime_cases,
    }
    return {
        "config": runtime_config,
        "case_meta": case_meta,
    }


def _write_runtime_batch_config(
    config_path: str,
    runtime_dir: str,
    run_tag: str,
) -> Dict[str, object]:
    payload = _build_runtime_batch_config(config_path=config_path, runtime_dir=runtime_dir, run_tag=run_tag)
    config_doc = payload["config"]
    runtime_config_dir = os.path.join(runtime_dir, "runtime_configs")
    os.makedirs(runtime_config_dir, exist_ok=True)
    runtime_config_path = os.path.join(runtime_config_dir, f"{run_tag}.yaml")
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_doc, f, allow_unicode=True, sort_keys=False)
    return {
        "runtime_config_path": runtime_config_path,
        "case_meta": payload["case_meta"],
    }


def _compute_quality_metrics(
    summary_path: str,
    case_meta: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    total_case_count = 0
    quality_case_count = 0
    compared_case_count = 0
    shared_rows_total = 0
    suspicious_rows_total = 0
    weighted_mae_pre_abs = 0.0
    weighted_mae_offset = 0.0
    weighted_alias_mae_pre_abs = 0.0
    weighted_alias_mae_offset = 0.0
    weighted_p90_pre_abs = 0.0
    weighted_p90_offset = 0.0
    cv_like_rows_total = 0
    cv_like_suspicious_rows_total = 0
    cv_like_order_jump_rows_total = 0
    cv_like_order_jump_severity = 0.0
    cv_like_p90_pre_abs = 0.0
    cv_like_p90_offset = 0.0

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f) or {}
    results = summary.get("results") or []
    if not isinstance(results, list):
        results = []
    total_case_count = len(results)

    for row in results:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        meta = case_meta.get(name) or {}
        base_oto = str(meta.get("base_oto", "") or "").strip()
        if not base_oto:
            continue
        compared_case_count += 1
        auto_oto = str(row.get("output_oto", "") or "").strip()
        if not auto_oto or not os.path.exists(auto_oto) or not os.path.exists(base_oto):
            continue
        try:
            diff = analyze_oto_diff(
                base_oto,
                auto_oto,
                language=str(meta.get("language", "auto") or "auto"),
                topn=10,
            )
        except Exception:
            continue

        counts = diff.get("counts") or {}
        maes = diff.get("mae") or {}
        quality = diff.get("quality") or {}
        shared_rows = int(counts.get("shared_rows", 0) or 0)
        suspicious_rows = int(counts.get("suspicious_rows", 0) or 0)
        weight = max(shared_rows, 1)
        cv_like_rows = int(quality.get("cv_like_rows", 0) or 0)
        quality_case_count += 1
        shared_rows_total += shared_rows
        suspicious_rows_total += suspicious_rows
        weighted_mae_pre_abs += float(maes.get("mae_pre_abs", 0.0) or 0.0) * float(weight)
        weighted_mae_offset += float(maes.get("mae_offset", 0.0) or 0.0) * float(weight)
        weighted_alias_mae_pre_abs += float(quality.get("weighted_mae_pre_abs", maes.get("mae_pre_abs", 0.0)) or 0.0) * float(weight)
        weighted_alias_mae_offset += float(quality.get("weighted_mae_offset", maes.get("mae_offset", 0.0)) or 0.0) * float(weight)
        weighted_p90_pre_abs += float(quality.get("p90_pre_abs", 0.0) or 0.0) * float(weight)
        weighted_p90_offset += float(quality.get("p90_offset", 0.0) or 0.0) * float(weight)
        if cv_like_rows > 0:
            cv_like_rows_total += cv_like_rows
            cv_like_suspicious_rows_total += int(quality.get("cv_like_suspicious_rows", 0) or 0)
            cv_like_order_jump_rows_total += int(quality.get("cv_like_order_jump_rows", 0) or 0)
            cv_like_order_jump_severity += float(quality.get("cv_like_order_jump_severity", 0.0) or 0.0) * float(cv_like_rows)
            cv_like_p90_pre_abs += float(quality.get("cv_like_p90_pre_abs", 0.0) or 0.0) * float(cv_like_rows)
            cv_like_p90_offset += float(quality.get("cv_like_p90_offset", 0.0) or 0.0) * float(cv_like_rows)

    if quality_case_count <= 0:
        return {
            "quality_total_case_count": int(total_case_count),
            "quality_case_count": 0,
            "quality_compared_case_count": compared_case_count,
            "quality_reference_ratio": 0.0,
            "quality_success_ratio": 0.0,
            "quality_shared_rows": 0,
            "quality_suspicious_rows": 0,
            "quality_suspicious_ratio": 1.0,
            "quality_mae_pre_abs": 1.0e9,
            "quality_mae_offset": 1.0e9,
            "quality_weighted_mae_pre_abs": 1.0e9,
            "quality_weighted_mae_offset": 1.0e9,
            "quality_p90_pre_abs": 1.0e9,
            "quality_p90_offset": 1.0e9,
            "quality_cv_like_rows": 0,
            "quality_cv_like_suspicious_rows": 0,
            "quality_cv_like_suspicious_ratio": 1.0,
            "quality_cv_like_order_jump_rows": 0,
            "quality_cv_like_order_jump_ratio": 1.0,
            "quality_cv_like_order_jump_severity": 1.0e9,
            "quality_cv_like_p90_pre_abs": 1.0e9,
            "quality_cv_like_p90_offset": 1.0e9,
        }

    denom = float(max(shared_rows_total, 1))
    mae_weight = float(max(shared_rows_total, quality_case_count))
    cv_like_denom = float(max(cv_like_rows_total, 1))
    total_case_denom = float(max(total_case_count, 1))
    fallback_cv_ratio = float(suspicious_rows_total) / denom
    fallback_cv_p90_pre_abs = float(weighted_p90_pre_abs) / mae_weight
    fallback_cv_p90_offset = float(weighted_p90_offset) / mae_weight
    return {
        "quality_total_case_count": int(total_case_count),
        "quality_case_count": quality_case_count,
        "quality_compared_case_count": compared_case_count,
        "quality_reference_ratio": float(compared_case_count) / total_case_denom,
        "quality_success_ratio": float(quality_case_count) / total_case_denom,
        "quality_shared_rows": int(shared_rows_total),
        "quality_suspicious_rows": int(suspicious_rows_total),
        "quality_suspicious_ratio": float(suspicious_rows_total) / denom,
        "quality_mae_pre_abs": float(weighted_mae_pre_abs) / mae_weight,
        "quality_mae_offset": float(weighted_mae_offset) / mae_weight,
        "quality_weighted_mae_pre_abs": float(weighted_alias_mae_pre_abs) / mae_weight,
        "quality_weighted_mae_offset": float(weighted_alias_mae_offset) / mae_weight,
        "quality_p90_pre_abs": float(weighted_p90_pre_abs) / mae_weight,
        "quality_p90_offset": float(weighted_p90_offset) / mae_weight,
        "quality_cv_like_rows": int(cv_like_rows_total),
        "quality_cv_like_suspicious_rows": int(cv_like_suspicious_rows_total),
        "quality_cv_like_suspicious_ratio": (
            float(cv_like_suspicious_rows_total) / cv_like_denom if cv_like_rows_total > 0 else fallback_cv_ratio
        ),
        "quality_cv_like_order_jump_rows": int(cv_like_order_jump_rows_total),
        "quality_cv_like_order_jump_ratio": (
            float(cv_like_order_jump_rows_total) / cv_like_denom if cv_like_rows_total > 0 else fallback_cv_ratio
        ),
        "quality_cv_like_order_jump_severity": (
            float(cv_like_order_jump_severity) / cv_like_denom if cv_like_rows_total > 0 else fallback_cv_p90_pre_abs
        ),
        "quality_cv_like_p90_pre_abs": (
            float(cv_like_p90_pre_abs) / cv_like_denom if cv_like_rows_total > 0 else fallback_cv_p90_pre_abs
        ),
        "quality_cv_like_p90_offset": (
            float(cv_like_p90_offset) / cv_like_denom if cv_like_rows_total > 0 else fallback_cv_p90_offset
        ),
    }


def _run_batch_once(
    python_exe: str,
    batch_script: str,
    config_path: str,
    run_tag: str,
    env_overrides: Dict[str, str],
    stop_on_error: bool,
    log_path: str,
) -> Dict[str, object]:
    runtime_root = os.path.dirname(log_path)
    runtime_info = _write_runtime_batch_config(
        config_path=config_path,
        runtime_dir=runtime_root,
        run_tag=run_tag,
    )
    runtime_config_path = str(runtime_info["runtime_config_path"])
    env = os.environ.copy()
    for key in ENV_KEYS:
        env.pop(key, None)
    env.update(env_overrides)

    cmd = [
        python_exe,
        batch_script,
        "--config",
        runtime_config_path,
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
        f.write(f"runtime_config={runtime_config_path}\n")
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
        "runtime_config_path": runtime_config_path,
        "case_meta": runtime_info["case_meta"],
    }


def _rank_key(row: Dict[str, object]):
    if bool(row.get("run_failed", False)):
        return (1, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9, 10**9)
    regression_count = int(row.get("regression_count", 0))
    errors_delta_sum = int(row.get("errors_delta_sum", 0))
    quality_missing = 0 if int(row.get("quality_case_count", 0) or 0) > 0 else 1
    quality_total_case_count = int(row.get("quality_total_case_count", 0) or 0)
    quality_case_count = int(row.get("quality_case_count", 0) or 0)
    quality_compared_case_count = int(row.get("quality_compared_case_count", 0) or 0)
    quality_case_gap = max(quality_total_case_count - quality_case_count, 0)
    quality_reference_gap = max(quality_total_case_count - quality_compared_case_count, 0)
    cv_like_order_jump_ratio = float(row.get("quality_cv_like_order_jump_ratio", 1.0e9) or 1.0e9)
    cv_like_order_jump_severity = float(row.get("quality_cv_like_order_jump_severity", 1.0e9) or 1.0e9)
    cv_like_suspicious_ratio = float(row.get("quality_cv_like_suspicious_ratio", 1.0e9) or 1.0e9)
    suspicious_ratio = float(row.get("quality_suspicious_ratio", 1.0e9) or 1.0e9)
    weighted_mae_pre_abs = float(row.get("quality_weighted_mae_pre_abs", row.get("quality_mae_pre_abs", 1.0e9)) or 1.0e9)
    p90_pre_abs = float(row.get("quality_p90_pre_abs", 1.0e9) or 1.0e9)
    weighted_mae_offset = float(row.get("quality_weighted_mae_offset", row.get("quality_mae_offset", 1.0e9)) or 1.0e9)
    p90_offset = float(row.get("quality_p90_offset", 1.0e9) or 1.0e9)
    warnings_delta_sum = int(row.get("warnings_delta_sum", 0) or 0)
    improvement_count = int(row.get("improvement_count", 0))
    status_changed_cases = int(row.get("status_changed_cases", 0))
    return (
        0,
        regression_count,
        errors_delta_sum,
        quality_missing,
        quality_case_gap,
        quality_reference_gap,
        cv_like_order_jump_ratio,
        cv_like_order_jump_severity,
        cv_like_suspicious_ratio,
        suspicious_ratio,
        weighted_mae_pre_abs,
        p90_pre_abs,
        weighted_mae_offset,
        p90_offset,
        warnings_delta_sum,
        -improvement_count,
        status_changed_cases,
    )


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
        "runtime_config_path",
        "UTOA_JA_MAPPING_CONF_THRESHOLD",
        "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD",
        "UTOA_KR_MAPPING_CONF_THRESHOLD",
        "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD",
        "quality_total_case_count",
        "quality_case_count",
        "quality_compared_case_count",
        "quality_reference_ratio",
        "quality_success_ratio",
        "quality_shared_rows",
        "quality_suspicious_rows",
        "quality_suspicious_ratio",
        "quality_mae_pre_abs",
        "quality_mae_offset",
        "quality_weighted_mae_pre_abs",
        "quality_weighted_mae_offset",
        "quality_p90_pre_abs",
        "quality_p90_offset",
        "quality_cv_like_rows",
        "quality_cv_like_suspicious_rows",
        "quality_cv_like_suspicious_ratio",
        "quality_cv_like_order_jump_rows",
        "quality_cv_like_order_jump_ratio",
        "quality_cv_like_order_jump_severity",
        "quality_cv_like_p90_pre_abs",
        "quality_cv_like_p90_offset",
        "compare_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
            "runtime_config_path": str(run_info.get("runtime_config_path", "") or ""),
            "UTOA_JA_MAPPING_CONF_THRESHOLD": env_overrides.get("UTOA_JA_MAPPING_CONF_THRESHOLD", ""),
            "UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD": env_overrides.get("UTOA_JA_MAPPING_SPN_RATIO_THRESHOLD", ""),
            "UTOA_KR_MAPPING_CONF_THRESHOLD": env_overrides.get("UTOA_KR_MAPPING_CONF_THRESHOLD", ""),
            "UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD": env_overrides.get("UTOA_KR_MAPPING_SPN_RATIO_THRESHOLD", ""),
            "quality_total_case_count": 0,
            "quality_case_count": 0,
            "quality_compared_case_count": 0,
            "quality_reference_ratio": 0.0,
            "quality_success_ratio": 0.0,
            "quality_shared_rows": 0,
            "quality_suspicious_rows": 0,
            "quality_suspicious_ratio": 1.0,
            "quality_mae_pre_abs": 1.0e9,
            "quality_mae_offset": 1.0e9,
            "quality_weighted_mae_pre_abs": 1.0e9,
            "quality_weighted_mae_offset": 1.0e9,
            "quality_p90_pre_abs": 1.0e9,
            "quality_p90_offset": 1.0e9,
            "quality_cv_like_rows": 0,
            "quality_cv_like_suspicious_rows": 0,
            "quality_cv_like_suspicious_ratio": 1.0,
            "quality_cv_like_order_jump_rows": 0,
            "quality_cv_like_order_jump_ratio": 1.0,
            "quality_cv_like_order_jump_severity": 1.0e9,
            "quality_cv_like_p90_pre_abs": 1.0e9,
            "quality_cv_like_p90_offset": 1.0e9,
            "compare_path": "",
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
            row.update(
                _compute_quality_metrics(
                    summary_path=summary_path,
                    case_meta=dict(run_info.get("case_meta") or {}),
                )
            )

        rows.append(row)
        print(
            "[ThresholdTune] "
            f"{idx}/{len(deduped_candidates)} "
            f"failed={row['run_failed']} "
            f"reg={row['regression_count']} "
            f"cv_jump={float(row['quality_cv_like_order_jump_ratio']):.4f} "
            f"susp_ratio={float(row['quality_suspicious_ratio']):.4f} "
            f"p90_pre={float(row['quality_p90_pre_abs']):.2f} "
            f"mae_pre={float(row['quality_weighted_mae_pre_abs']):.2f} "
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
            f"cv_jump={float(best['quality_cv_like_order_jump_ratio']):.4f} "
            f"susp_ratio={float(best['quality_suspicious_ratio']):.4f} "
            f"p90_pre={float(best['quality_p90_pre_abs']):.2f} "
            f"mae_pre={float(best['quality_weighted_mae_pre_abs']):.2f} "
            f"imp={best['improvement_count']} "
            f"err_delta={best['errors_delta_sum']} "
            f"env={best['env_suffix']}"
        )


if __name__ == "__main__":
    main()
