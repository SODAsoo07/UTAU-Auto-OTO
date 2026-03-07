import argparse
import datetime as dt
import json
import os
import shutil
import sys
import traceback
from typing import Dict, List

try:
    import yaml
except Exception:
    yaml = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.ja_oto_generator import generate_ja_oto
from core.oto_generator import generate_oto
from core.oto_validator import validate_oto_timing


def _now_tag():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "case"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "case"


def _to_bool(v, default=False):
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


def _safe_console_print(text: str):
    s = str(text)
    try:
        print(s)
        return
    except UnicodeEncodeError:
        pass
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        safe = s.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        safe = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    print(safe)


def _resolve_path(config_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(config_dir, p))


def _resolve_case_path(config_dir: str, voicebank_dir: str, raw_path: str) -> str:
    """
    Case path resolver:
    1) absolute path 그대로 사용
    2) relative path는 voicebank_dir 기준 우선
    3) 없으면 config_dir 기준으로 fallback
    """
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


def _build_case_output_name(case: dict, defaults: dict, run_tag: str, index: int) -> str:
    explicit = str(case.get("output_name", "")).strip()
    if explicit:
        return explicit
    template = str(case.get("output_name_template", defaults.get("output_name_template", "oto.{run_tag}.ini"))).strip()
    if not template:
        template = "oto.{run_tag}.ini"
    case_name = _safe_name(str(case.get("name", f"case_{index:03d}")))
    try:
        return template.format(run_tag=run_tag, index=index, name=case_name)
    except Exception:
        return f"oto.{run_tag}.{case_name}.ini"


def _load_config(path: str):
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: python -m pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    defaults = data.get("defaults") or {}
    cases = data.get("cases") or []
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list.")
    return defaults, cases


def _render_exception_text(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return exc.__class__.__name__


def _resolve_case_settings(
    case: dict,
    defaults: dict,
    config_dir: str,
    run_tag: str,
    case_index: int,
    force_replace: bool = False,
    force_no_validation: bool = False,
) -> Dict[str, object]:
    name = str(case.get("name", f"case_{case_index:03d}")).strip() or f"case_{case_index:03d}"
    enabled = _to_bool(case.get("enabled", defaults.get("enabled", True)), True)
    language = str(case.get("language", defaults.get("language", "japanese"))).strip().lower()
    voicebank_dir = _resolve_path(
        config_dir,
        str(case.get("voicebank_dir", case.get("wav_dir", defaults.get("voicebank_dir", "")))).strip(),
    )
    tg_folder = _resolve_case_path(
        config_dir,
        voicebank_dir,
        str(case.get("textgrid_dir", case.get("tg_folder", defaults.get("textgrid_dir", "textgrids")))).strip(),
    )
    out_dir = _resolve_path(config_dir, str(case.get("out_dir", defaults.get("out_dir", voicebank_dir))).strip())
    if not out_dir:
        out_dir = voicebank_dir
    output_name = _build_case_output_name(case, defaults, run_tag, case_index)
    out_path = _resolve_path(out_dir, output_name) if not os.path.isabs(output_name) else _resolve_path(config_dir, output_name)
    no_base_oto = _to_bool(case.get("no_base_oto", defaults.get("no_base_oto", False)), False)
    base_oto_raw = str(case.get("base_oto", defaults.get("base_oto", ""))).strip()
    if no_base_oto:
        tpl_path = ""
    elif base_oto_raw:
        tpl_path = _resolve_case_path(config_dir, voicebank_dir, base_oto_raw)
    else:
        cand = os.path.join(voicebank_dir, "oto.ini")
        tpl_path = cand if os.path.exists(cand) else ""

    custom_phonemes_path = _resolve_path(
        config_dir,
        str(case.get("custom_phonemes_path", defaults.get("custom_phonemes_path", ""))).strip(),
    )
    replace_oto_ini = force_replace or _to_bool(case.get("replace_oto_ini", defaults.get("replace_oto_ini", False)), False)
    return {
        "name": name,
        "enabled": enabled,
        "language": language,
        "voicebank_dir": voicebank_dir,
        "textgrid_dir": tg_folder,
        "out_dir": out_dir,
        "output_oto": out_path,
        "tpl_path": tpl_path,
        "auto_format": str(case.get("auto_format", defaults.get("auto_format", ""))).strip(),
        "generate_openutau": _to_bool(case.get("generate_openutau", defaults.get("generate_openutau", False)), False),
        "gen_missing_vowels": _to_bool(case.get("gen_missing_vowels", defaults.get("gen_missing_vowels", False)), False),
        "enable_ml_correction": _to_bool(case.get("enable_ml_correction", defaults.get("enable_ml_correction", True)), True),
        "custom_phonemes_path": custom_phonemes_path,
        "alias_suffix": str(case.get("alias_suffix", defaults.get("alias_suffix", ""))).strip(),
        "alias_style": str(case.get("alias_style", defaults.get("alias_style", "original"))).strip().lower(),
        "do_validation": _to_bool(case.get("validation", defaults.get("validation", True)), True) and (not force_no_validation),
        "replace_oto_ini": replace_oto_ini,
        "replace_target_oto": os.path.join(voicebank_dir, "oto.ini") if voicebank_dir else "",
    }


def _validate_case_settings(case_info: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    if not bool(case_info.get("enabled", True)):
        return issues

    name = str(case_info.get("name", "") or "").strip() or "case"
    language = str(case_info.get("language", "") or "").strip().lower()
    voicebank_dir = str(case_info.get("voicebank_dir", "") or "").strip()
    textgrid_dir = str(case_info.get("textgrid_dir", "") or "").strip()
    tpl_path = str(case_info.get("tpl_path", "") or "").strip()
    custom_phonemes_path = str(case_info.get("custom_phonemes_path", "") or "").strip()
    out_path = str(case_info.get("output_oto", "") or "").strip()
    replace_target_oto = str(case_info.get("replace_target_oto", "") or "").strip()

    if language not in {"japanese", "korean"}:
        issues.append(f"case={name}: unsupported language: {language}")
    if not voicebank_dir or not os.path.isdir(voicebank_dir):
        issues.append(f"case={name}: voicebank_dir not found: {voicebank_dir}")
    if not textgrid_dir or not os.path.isdir(textgrid_dir):
        issues.append(f"case={name}: textgrid_dir not found: {textgrid_dir}")
    if tpl_path and not os.path.exists(tpl_path):
        issues.append(f"case={name}: base_oto not found: {tpl_path}")
    if custom_phonemes_path and not os.path.exists(custom_phonemes_path):
        issues.append(f"case={name}: custom_phonemes_path not found: {custom_phonemes_path}")
    if not out_path:
        issues.append(f"case={name}: output path could not be resolved")
    if bool(case_info.get("replace_oto_ini", False)) and not replace_target_oto:
        issues.append(f"case={name}: replace target oto.ini path could not be resolved")
    return issues


def _collect_preflight_issues(case_infos: List[Dict[str, object]]) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    seen_names = set()
    seen_output_paths: Dict[str, str] = {}
    seen_replace_targets: Dict[str, str] = {}

    for case_info in case_infos:
        name = str(case_info.get("name", "") or "").strip() or "case"
        if name in seen_names:
            errors.append(f"duplicate case name: {name}")
        else:
            seen_names.add(name)

        errors.extend(_validate_case_settings(case_info))
        if not bool(case_info.get("enabled", True)):
            continue

        out_path = str(case_info.get("output_oto", "") or "").strip()
        if out_path:
            prev_name = seen_output_paths.get(out_path)
            if prev_name and prev_name != name:
                errors.append(f"output collision: {name} and {prev_name} -> {out_path}")
            else:
                seen_output_paths[out_path] = name

        if bool(case_info.get("replace_oto_ini", False)):
            replace_target = str(case_info.get("replace_target_oto", "") or "").strip()
            if replace_target:
                prev_name = seen_replace_targets.get(replace_target)
                if prev_name and prev_name != name:
                    errors.append(f"replace collision: {name} and {prev_name} -> {replace_target}")
                else:
                    seen_replace_targets[replace_target] = name

    return {"errors": errors, "warnings": warnings}


def _write_preflight_report(run_dir: str, config_path: str, case_infos: List[Dict[str, object]], issues: Dict[str, List[str]]) -> str:
    report = {
        "config": os.path.abspath(config_path),
        "created_at": dt.datetime.now().isoformat(),
        "error_count": len(issues.get("errors") or []),
        "warning_count": len(issues.get("warnings") or []),
        "errors": list(issues.get("errors") or []),
        "warnings": list(issues.get("warnings") or []),
        "cases": case_infos,
    }
    out_path = os.path.join(run_dir, "preflight.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def _write_summary_text(summary_path: str, summary: Dict[str, object]) -> str:
    out_path = os.path.splitext(summary_path)[0] + ".txt"
    lines = [
        f"config={summary.get('config', '')}",
        f"run_tag={summary.get('run_tag', '')}",
        f"run_dir={summary.get('run_dir', '')}",
        f"total_cases={summary.get('total_cases', 0)}",
        f"ok_cases={summary.get('ok_cases', 0)}",
        f"error_cases={summary.get('error_cases', 0)}",
        f"skipped_cases={summary.get('skipped_cases', 0)}",
        "",
    ]
    for row in summary.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "") or "")
        status = str(row.get("status", "") or "")
        reason = str(row.get("reason", "") or "")
        lines.append(f"[{status}] {name}")
        if reason:
            lines.append(f"reason={reason}")
        validation = row.get("validation") or {}
        if isinstance(validation, dict) and validation:
            lines.append(
                "validation="
                f"checked_files={int(validation.get('checked_files', 0) or 0)} "
                f"warnings={int(validation.get('warnings', 0) or 0)} "
                f"errors={int(validation.get('errors', 0) or 0)}"
            )
        output_oto = str(row.get("output_oto", "") or "").strip()
        if output_oto:
            lines.append(f"output_oto={output_oto}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return out_path


def _run_one_case(
    case: dict,
    defaults: dict,
    config_dir: str,
    run_tag: str,
    case_index: int,
    run_dir: str,
    force_replace: bool = False,
    force_no_validation: bool = False,
):
    case_info = _resolve_case_settings(
        case=case,
        defaults=defaults,
        config_dir=config_dir,
        run_tag=run_tag,
        case_index=case_index,
        force_replace=force_replace,
        force_no_validation=force_no_validation,
    )
    name = str(case_info["name"])
    enabled = bool(case_info["enabled"])
    if not enabled:
        return {"name": name, "status": "skipped", "reason": "disabled"}

    language = str(case_info["language"]).strip().lower()
    if language not in {"japanese", "korean"}:
        return {"name": name, "status": "error", "reason": f"unsupported language: {language}"}

    voicebank_dir = str(case_info["voicebank_dir"])
    if not voicebank_dir or not os.path.isdir(voicebank_dir):
        return {"name": name, "status": "error", "reason": f"voicebank_dir not found: {voicebank_dir}"}

    tg_folder = str(case_info["textgrid_dir"])
    if not os.path.isdir(tg_folder):
        return {"name": name, "status": "error", "reason": f"textgrid_dir not found: {tg_folder}"}

    out_dir = str(case_info["out_dir"])
    if not out_dir:
        out_dir = voicebank_dir
    os.makedirs(out_dir, exist_ok=True)

    out_path = str(case_info["output_oto"])
    tpl_path = str(case_info["tpl_path"])
    auto_format = str(case_info["auto_format"])
    generate_openutau = bool(case_info["generate_openutau"])
    gen_missing_vowels = bool(case_info["gen_missing_vowels"])
    enable_ml_correction = bool(case_info["enable_ml_correction"])
    custom_phonemes_path = str(case_info["custom_phonemes_path"])
    alias_suffix = str(case_info["alias_suffix"])
    alias_style = str(case_info["alias_style"])
    do_validation = bool(case_info["do_validation"])
    replace_oto_ini = bool(case_info["replace_oto_ini"])

    case_log_name = f"{case_index:03d}_{_safe_name(name)}.log"
    case_log_path = os.path.join(run_dir, case_log_name)
    case_logs = []

    def log(msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}][{name}] {msg}"
        _safe_console_print(line)
        case_logs.append(line)

    log(f"start lang={language} format={auto_format or 'auto'}")
    log(f"voicebank={voicebank_dir}")
    log(f"textgrid_dir={tg_folder}")
    log(f"output_oto={out_path}")

    processed = 0
    total = 0
    errors = []
    validation = None
    status = "ok"
    reason = ""

    try:
        if language == "japanese":
            processed, total, errors = generate_ja_oto(
                tg_folder=tg_folder,
                tpl_path=tpl_path,
                out_path=out_path,
                params=None,
                generate_openutau=generate_openutau,
                gen_missing_vowels=gen_missing_vowels,
                enable_ml_correction=enable_ml_correction,
                fallback_format=auto_format or "cvvc",
                custom_phonemes_path=custom_phonemes_path,
                alias_suffix=alias_suffix,
                alias_style=alias_style,
                auto_format=auto_format or None,
                callback=log,
            )
        else:
            processed, total, errors = generate_oto(
                tg_folder=tg_folder,
                tpl_path=tpl_path,
                out_path=out_path,
                params=None,
                generate_openutau=generate_openutau,
                gen_missing_vowels=gen_missing_vowels,
                enable_ml_correction=enable_ml_correction,
                fallback_format=auto_format or "cvvc",
                custom_phonemes_path=custom_phonemes_path,
                alias_suffix=alias_suffix,
                auto_format=auto_format or None,
                callback=log,
            )

        if errors:
            status = "error"
            reason = f"generator_errors={len(errors)}"
            log(f"generator_errors={len(errors)}")

        if do_validation and os.path.exists(out_path):
            validation = validate_oto_timing(
                wav_dir=voicebank_dir,
                tg_folder=tg_folder,
                oto_path=out_path,
                language=language,
                callback=log,
            )

        if replace_oto_ini and os.path.exists(out_path):
            target_oto = os.path.join(voicebank_dir, "oto.ini")
            backup_oto = os.path.join(voicebank_dir, f"oto.backup.{run_tag}.ini")
            if os.path.exists(target_oto):
                shutil.copyfile(target_oto, backup_oto)
                log(f"backup_oto={backup_oto}")
            shutil.copyfile(out_path, target_oto)
            log(f"replaced_oto={target_oto}")

    except Exception as e:
        status = "error"
        reason = _render_exception_text(e)
        log(f"exception={reason}")
        case_logs.append(traceback.format_exc())

    with open(case_log_path, "w", encoding="utf-8") as f:
        for line in case_logs:
            f.write(line + "\n")

    result = {
        "name": name,
        "status": status,
        "reason": reason,
        "language": language,
        "voicebank_dir": voicebank_dir,
        "textgrid_dir": tg_folder,
        "output_oto": out_path,
        "processed_files": int(processed),
        "total_files": int(total),
        "generator_errors": errors,
        "validation": validation or {},
        "log_path": case_log_path,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch OTO generation for listening tests (per language/format/voicebank)."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-tag", default="", help="Run tag for output naming. Default: timestamp.")
    parser.add_argument("--replace", action="store_true", help="Replace each voicebank's oto.ini with generated output.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip auto validation step.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first error case.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    config_dir = os.path.dirname(config_path)
    defaults, cases = _load_config(config_path)

    if not cases:
        raise ValueError("No cases found in config.")

    run_tag = args.run_tag.strip() or _now_tag()
    run_dir = os.path.join("logs", "oto_batch", run_tag)
    os.makedirs(run_dir, exist_ok=True)

    case_infos = [
        _resolve_case_settings(
            case=case if isinstance(case, dict) else {},
            defaults=defaults,
            config_dir=config_dir,
            run_tag=run_tag,
            case_index=idx,
            force_replace=args.replace,
            force_no_validation=args.skip_validation,
        )
        for idx, case in enumerate(cases, start=1)
    ]
    preflight_issues = _collect_preflight_issues(case_infos)
    preflight_path = _write_preflight_report(
        run_dir=run_dir,
        config_path=config_path,
        case_infos=case_infos,
        issues=preflight_issues,
    )

    _safe_console_print(f"[BatchOTO] config={config_path}")
    _safe_console_print(f"[BatchOTO] run_tag={run_tag}")
    _safe_console_print(f"[BatchOTO] cases={len(cases)}")
    _safe_console_print(f"[BatchOTO] run_dir={os.path.abspath(run_dir)}")
    _safe_console_print(f"[BatchOTO] preflight={os.path.abspath(preflight_path)}")
    if preflight_issues["warnings"]:
        _safe_console_print(f"[BatchOTO][Preflight] warnings={len(preflight_issues['warnings'])}")
        for msg in preflight_issues["warnings"][:20]:
            _safe_console_print(f"[BatchOTO][Preflight][WARN] {msg}")
    if preflight_issues["errors"]:
        _safe_console_print(f"[BatchOTO][Preflight] errors={len(preflight_issues['errors'])}")
        for msg in preflight_issues["errors"][:20]:
            _safe_console_print(f"[BatchOTO][Preflight][ERROR] {msg}")
        raise ValueError(f"Preflight failed with {len(preflight_issues['errors'])} error(s).")

    results = []
    for idx, case in enumerate(cases, start=1):
        res = _run_one_case(
            case=case if isinstance(case, dict) else {},
            defaults=defaults,
            config_dir=config_dir,
            run_tag=run_tag,
            case_index=idx,
            run_dir=run_dir,
            force_replace=args.replace,
            force_no_validation=args.skip_validation,
        )
        results.append(res)
        _safe_console_print(f"[BatchOTO] done {idx}/{len(cases)} name={res.get('name')} status={res.get('status')}")
        if args.stop_on_error and res.get("status") == "error":
            _safe_console_print("[BatchOTO] stop_on_error triggered.")
            break

    summary = {
        "run_tag": run_tag,
        "config": config_path,
        "run_dir": os.path.abspath(run_dir),
        "total_cases": len(results),
        "ok_cases": sum(1 for r in results if r.get("status") == "ok"),
        "error_cases": sum(1 for r in results if r.get("status") == "error"),
        "skipped_cases": sum(1 for r in results if r.get("status") == "skipped"),
        "results": results,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_txt_path = _write_summary_text(summary_path, summary)
    _safe_console_print(f"[BatchOTO] summary={os.path.abspath(summary_path)}")
    _safe_console_print(f"[BatchOTO] summary_txt={os.path.abspath(summary_txt_path)}")


if __name__ == "__main__":
    main()
