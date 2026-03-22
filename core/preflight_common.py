from __future__ import annotations

import os
from typing import Dict, List

from core.error_codes import format_error_with_recovery, get_error_entry
from core.pipeline_status import has_textgrid_files, normalize_aligner_name


def _issue(code: str, message: str, level: str = "error") -> Dict[str, str]:
    normalized_code = str(code or "").strip().upper()
    entry = get_error_entry(normalized_code)
    return {
        "level": str(level or "error").strip().lower(),
        "code": normalized_code,
        "message": str(message or "").strip(),
        "display": format_error_with_recovery(normalized_code, message),
        "stage": str(entry.stage if entry else "unknown"),
        "cause": str(entry.cause if entry else ""),
        "recovery": str(entry.recovery if entry else ""),
    }


def issue_records_to_display_strings(issue_records: List[Dict[str, str]]) -> List[str]:
    return [
        str(entry.get("display") or "")
        for entry in list(issue_records or [])
        if str(entry.get("display") or "").strip()
    ]


def build_preflight_issue_payload(records: Dict[str, List[Dict[str, str]]]) -> Dict[str, object]:
    error_records = list((records or {}).get("errors") or [])
    warning_records = list((records or {}).get("warnings") or [])
    return {
        "errors": issue_records_to_display_strings(error_records),
        "warnings": issue_records_to_display_strings(warning_records),
        "error_records": error_records,
        "warning_records": warning_records,
    }


def validate_runtime_preflight(
    *,
    language: str,
    wav_dir: str,
    out_path: str,
    aligner: str,
    textgrid_dir: str,
    tpl_path: str = "",
    no_base_oto: bool = False,
    custom_phonemes_path: str = "",
    require_output: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    _ = str(language or "").strip().lower()  # reserved for language-specific rules
    wav_path = str(wav_dir or "").strip()
    output_path = str(out_path or "").strip()
    align_name = normalize_aligner_name(aligner, default="mfa")
    tg_dir = str(textgrid_dir or "").strip()
    template_path = str(tpl_path or "").strip()
    custom_path = str(custom_phonemes_path or "").strip()

    if (not wav_path) or (not os.path.isdir(wav_path)):
        errors.append(_issue("PRE_WAV_DIR_MISSING", f"voicebank/wav 폴더를 찾을 수 없습니다: {wav_path}"))

    if require_output and (not output_path):
        errors.append(_issue("PRE_OUTPUT_PATH_MISSING", "출력 OTO 경로가 비어 있습니다."))
    elif not output_path:
        warnings.append(_issue("PRE_OUTPUT_PATH_MISSING", "출력 경로가 없어 OTO 생성 단계는 건너뜁니다.", level="warning"))

    no_mfa_without_textgrid = align_name == "none" and (not has_textgrid_files(tg_dir))
    template_checked_in_no_mfa_path = False
    if no_mfa_without_textgrid:
        if no_base_oto:
            errors.append(
                _issue(
                    "PRE_TEMPLATE_MISSING",
                    "No-MFA 자동설정 모드에서는 TextGrid가 없을 때 베이스 OTO(템플릿 ini)가 필요합니다.",
                )
            )
        elif not template_path:
            errors.append(
                _issue(
                    "PRE_TEMPLATE_MISSING",
                    "No-MFA 자동설정 모드에서는 템플릿 OTO 경로를 지정해야 합니다.",
                )
            )
        elif not os.path.exists(template_path):
            template_checked_in_no_mfa_path = True
            errors.append(
                _issue(
                    "PRE_TEMPLATE_MISSING",
                    f"No-MFA 자동설정용 템플릿 OTO를 찾을 수 없습니다: {template_path}",
                )
            )
        else:
            warnings.append(
                _issue(
                    "PRE_TEXTGRID_REQUIRED",
                    "No-MFA 모드: TextGrid가 없어 베이스 OTO 재매핑 자동설정으로 진행합니다.",
                    level="warning",
                )
            )

    if (
        (not no_base_oto)
        and template_path
        and (not os.path.exists(template_path))
        and (not template_checked_in_no_mfa_path)
    ):
        errors.append(_issue("PRE_TEMPLATE_MISSING", f"템플릿 OTO 파일을 찾을 수 없습니다: {template_path}"))

    if custom_path and (not os.path.exists(custom_path)):
        errors.append(_issue("PRE_CUSTOM_PHONEME_MISSING", f"커스텀 phoneme 파일을 찾을 수 없습니다: {custom_path}"))

    return {"errors": errors, "warnings": warnings}


def collect_runtime_preflight_issue_records(
    *,
    language: str,
    wav_dir: str,
    out_path: str,
    aligner: str,
    textgrid_dir: str,
    tpl_path: str = "",
    no_base_oto: bool = False,
    custom_phonemes_path: str = "",
    require_output: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    return validate_runtime_preflight(
        language=language,
        wav_dir=wav_dir,
        out_path=out_path,
        aligner=aligner,
        textgrid_dir=textgrid_dir,
        tpl_path=tpl_path,
        no_base_oto=no_base_oto,
        custom_phonemes_path=custom_phonemes_path,
        require_output=require_output,
    )


def collect_runtime_preflight_issues(
    *,
    language: str,
    wav_dir: str,
    out_path: str,
    aligner: str,
    textgrid_dir: str,
    tpl_path: str = "",
    no_base_oto: bool = False,
    custom_phonemes_path: str = "",
    require_output: bool = False,
) -> Dict[str, object]:
    records = collect_runtime_preflight_issue_records(
        language=language,
        wav_dir=wav_dir,
        out_path=out_path,
        aligner=aligner,
        textgrid_dir=textgrid_dir,
        tpl_path=tpl_path,
        no_base_oto=no_base_oto,
        custom_phonemes_path=custom_phonemes_path,
        require_output=require_output,
    )
    return build_preflight_issue_payload(records)


def validate_batch_case_settings_records(case_info: Dict[str, object]) -> Dict[str, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    if not bool(case_info.get("enabled", True)):
        return {"errors": errors, "warnings": warnings}

    name = str(case_info.get("name", "") or "").strip() or "case"
    language = str(case_info.get("language", "") or "").strip().lower()
    voicebank_dir = str(case_info.get("voicebank_dir", "") or "").strip()
    textgrid_dir = str(case_info.get("textgrid_dir", "") or "").strip()
    tpl_path = str(case_info.get("tpl_path", "") or "").strip()
    custom_phonemes_path = str(case_info.get("custom_phonemes_path", "") or "").strip()
    out_path = str(case_info.get("output_oto", "") or "").strip()
    replace_target_oto = str(case_info.get("replace_target_oto", "") or "").strip()

    if language not in {"japanese", "korean"}:
        errors.append(_issue("UNSUPPORTED_LANGUAGE", f"case={name}: unsupported language: {language}"))
    if not voicebank_dir or not os.path.isdir(voicebank_dir):
        errors.append(_issue("PRE_WAV_DIR_MISSING", f"case={name}: voicebank_dir not found: {voicebank_dir}"))
    align_if_missing = bool(case_info.get("align_if_missing", False))
    if (not textgrid_dir or not os.path.isdir(textgrid_dir)) and not align_if_missing:
        errors.append(_issue("PRE_TEXTGRID_REQUIRED", f"case={name}: textgrid_dir not found: {textgrid_dir}"))
    if tpl_path and not os.path.exists(tpl_path):
        errors.append(_issue("PRE_TEMPLATE_MISSING", f"case={name}: base_oto not found: {tpl_path}"))
    if custom_phonemes_path and not os.path.exists(custom_phonemes_path):
        errors.append(
            _issue(
                "PRE_CUSTOM_PHONEME_MISSING",
                f"case={name}: custom_phonemes_path not found: {custom_phonemes_path}",
            )
        )
    if not out_path:
        errors.append(_issue("PRE_OUTPUT_PATH_MISSING", f"case={name}: output path could not be resolved"))
    if bool(case_info.get("replace_oto_ini", False)) and not replace_target_oto:
        errors.append(_issue("PRE_OUTPUT_PATH_MISSING", f"case={name}: replace target oto.ini path could not be resolved"))
    return {"errors": errors, "warnings": warnings}


def validate_batch_case_settings(case_info: Dict[str, object]) -> List[str]:
    records = validate_batch_case_settings_records(case_info)
    return [str(entry.get("display") or "") for entry in records.get("errors", []) if str(entry.get("display") or "").strip()]


def collect_batch_preflight_issue_records(case_infos: List[Dict[str, object]]) -> Dict[str, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    seen_names = set()
    seen_output_paths: Dict[str, str] = {}
    seen_replace_targets: Dict[str, str] = {}

    for case_info in case_infos:
        name = str(case_info.get("name", "") or "").strip() or "case"
        if name in seen_names:
            errors.append(_issue("PRE_DUPLICATE_CASE", f"duplicate case name: {name}"))
        else:
            seen_names.add(name)

        case_records = validate_batch_case_settings_records(case_info)
        errors.extend(case_records.get("errors", []))
        warnings.extend(case_records.get("warnings", []))
        if bool(case_info.get("enabled", True)) and bool(case_info.get("align_if_missing", False)):
            textgrid_dir = str(case_info.get("textgrid_dir", "") or "").strip()
            if not textgrid_dir or not os.path.isdir(textgrid_dir):
                warnings.append(
                    _issue(
                        "PRE_TEXTGRID_REQUIRED",
                        f"case={name}: textgrid_dir missing, align_if_missing=true -> runtime alignment will run",
                        level="warning",
                    )
                )
        if not bool(case_info.get("enabled", True)):
            continue

        out_path = str(case_info.get("output_oto", "") or "").strip()
        if out_path:
            prev_name = seen_output_paths.get(out_path)
            if prev_name and prev_name != name:
                errors.append(_issue("PRE_OUTPUT_COLLISION", f"output collision: {name} and {prev_name} -> {out_path}"))
            else:
                seen_output_paths[out_path] = name

        if bool(case_info.get("replace_oto_ini", False)):
            replace_target = str(case_info.get("replace_target_oto", "") or "").strip()
            if replace_target:
                prev_name = seen_replace_targets.get(replace_target)
                if prev_name and prev_name != name:
                    errors.append(_issue("PRE_REPLACE_COLLISION", f"replace collision: {name} and {prev_name} -> {replace_target}"))
                else:
                    seen_replace_targets[replace_target] = name

    return {"errors": errors, "warnings": warnings}


def collect_batch_preflight_issues(case_infos: List[Dict[str, object]]) -> Dict[str, List[str]]:
    records = collect_batch_preflight_issue_records(case_infos)
    payload = build_preflight_issue_payload(records)
    return {
        "errors": list(payload.get("errors") or []),
        "warnings": list(payload.get("warnings") or []),
    }
