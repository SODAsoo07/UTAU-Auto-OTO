import argparse
import datetime as dt
import json
import os
import shutil
import sys
import traceback

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


def _resolve_path(config_dir: str, raw_path: str) -> str:
    if not raw_path:
        return ""
    p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(config_dir, p))


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
    name = str(case.get("name", f"case_{case_index:03d}")).strip() or f"case_{case_index:03d}"
    enabled = _to_bool(case.get("enabled", defaults.get("enabled", True)), True)
    if not enabled:
        return {"name": name, "status": "skipped", "reason": "disabled"}

    language = str(case.get("language", defaults.get("language", "japanese"))).strip().lower()
    if language not in {"japanese", "korean"}:
        return {"name": name, "status": "error", "reason": f"unsupported language: {language}"}

    voicebank_dir = _resolve_path(config_dir, str(case.get("voicebank_dir", case.get("wav_dir", ""))).strip())
    if not voicebank_dir or not os.path.isdir(voicebank_dir):
        return {"name": name, "status": "error", "reason": f"voicebank_dir not found: {voicebank_dir}"}

    tg_folder_raw = str(case.get("textgrid_dir", case.get("tg_folder", "textgrids"))).strip()
    tg_folder = _resolve_path(voicebank_dir, tg_folder_raw) if not os.path.isabs(tg_folder_raw) else _resolve_path(config_dir, tg_folder_raw)
    if not os.path.isdir(tg_folder):
        return {"name": name, "status": "error", "reason": f"textgrid_dir not found: {tg_folder}"}

    out_dir = _resolve_path(config_dir, str(case.get("out_dir", defaults.get("out_dir", voicebank_dir))).strip())
    if not out_dir:
        out_dir = voicebank_dir
    os.makedirs(out_dir, exist_ok=True)

    output_name = _build_case_output_name(case, defaults, run_tag, case_index)
    out_path = _resolve_path(out_dir, output_name) if not os.path.isabs(output_name) else _resolve_path(config_dir, output_name)

    no_base_oto = _to_bool(case.get("no_base_oto", defaults.get("no_base_oto", False)), False)
    if no_base_oto:
        tpl_path = ""
    else:
        base_oto_raw = str(case.get("base_oto", defaults.get("base_oto", ""))).strip()
        if base_oto_raw:
            tpl_path = _resolve_path(config_dir, base_oto_raw)
        else:
            cand = os.path.join(voicebank_dir, "oto.ini")
            tpl_path = cand if os.path.exists(cand) else ""

    auto_format = str(case.get("auto_format", defaults.get("auto_format", ""))).strip()
    generate_openutau = _to_bool(case.get("generate_openutau", defaults.get("generate_openutau", False)), False)
    gen_missing_vowels = _to_bool(case.get("gen_missing_vowels", defaults.get("gen_missing_vowels", False)), False)
    enable_ml_correction = _to_bool(case.get("enable_ml_correction", defaults.get("enable_ml_correction", True)), True)
    enable_pytorch_bridge = _to_bool(case.get("enable_pytorch_bridge", defaults.get("enable_pytorch_bridge", False)), False)
    custom_phonemes_path = _resolve_path(config_dir, str(case.get("custom_phonemes_path", defaults.get("custom_phonemes_path", ""))).strip())
    alias_suffix = str(case.get("alias_suffix", defaults.get("alias_suffix", ""))).strip()
    alias_style = str(case.get("alias_style", defaults.get("alias_style", "original"))).strip().lower()
    do_validation = _to_bool(case.get("validation", defaults.get("validation", True)), True) and (not force_no_validation)
    replace_oto_ini = force_replace or _to_bool(case.get("replace_oto_ini", defaults.get("replace_oto_ini", False)), False)

    case_log_name = f"{case_index:03d}_{_safe_name(name)}.log"
    case_log_path = os.path.join(run_dir, case_log_name)
    case_logs = []

    def log(msg: str):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        case_logs.append(line)

    log(f"START name={name} lang={language} format={auto_format or 'auto'}")
    log(f"voicebank={voicebank_dir}")
    log(f"textgrids={tg_folder}")
    log(f"output={out_path}")

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
                enable_pytorch_bridge=enable_pytorch_bridge,
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
                enable_pytorch_bridge=enable_pytorch_bridge,
                fallback_format=auto_format or "cvvc",
                custom_phonemes_path=custom_phonemes_path,
                alias_suffix=alias_suffix,
                auto_format=auto_format or None,
                callback=log,
            )

        if errors:
            status = "error"
            reason = f"generator_errors={len(errors)}"
            log(f"ERROR generator_errors={len(errors)}")

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
                log(f"BACKUP {backup_oto}")
            shutil.copyfile(out_path, target_oto)
            log(f"REPLACED {target_oto}")

    except Exception as e:
        status = "error"
        reason = str(e)
        log(f"EXCEPTION {e}")
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

    print(f"[BatchOTO] config={config_path}")
    print(f"[BatchOTO] run_tag={run_tag}")
    print(f"[BatchOTO] cases={len(cases)}")
    print(f"[BatchOTO] run_dir={os.path.abspath(run_dir)}")

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
        print(f"[BatchOTO] done {idx}/{len(cases)} name={res.get('name')} status={res.get('status')}")
        if args.stop_on_error and res.get("status") == "error":
            print("[BatchOTO] stop_on_error triggered.")
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
    print(f"[BatchOTO] summary={os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
