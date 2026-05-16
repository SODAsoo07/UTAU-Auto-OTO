from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ErrorCodeEntry:
    code: str
    stage: str
    cause: str
    recovery: str


ERROR_CATALOG: Dict[str, ErrorCodeEntry] = {
    "OK": ErrorCodeEntry(
        "OK",
        "all",
        "The operation completed successfully.",
        "No action required.",
    ),
    "PRE_WAV_DIR_MISSING": ErrorCodeEntry(
        "PRE_WAV_DIR_MISSING",
        "preflight",
        "WAV/voicebank directory is missing or unreadable.",
        "Set a valid voicebank path and verify folder permissions.",
    ),
    "PRE_OUTPUT_PATH_MISSING": ErrorCodeEntry(
        "PRE_OUTPUT_PATH_MISSING",
        "preflight",
        "Output oto.ini path is missing.",
        "Set an explicit output path (or replace target path when replace mode is enabled).",
    ),
    "PRE_TEXTGRID_REQUIRED": ErrorCodeEntry(
        "PRE_TEXTGRID_REQUIRED",
        "preflight",
        "TextGrid input is required for the current settings.",
        "Provide TextGrid files or enable runtime alignment fallback.",
    ),
    "PRE_TEMPLATE_MISSING": ErrorCodeEntry(
        "PRE_TEMPLATE_MISSING",
        "preflight",
        "Template oto.ini path is configured but not found.",
        "Fix the template path or disable template usage.",
    ),
    "PRE_CUSTOM_PHONEME_MISSING": ErrorCodeEntry(
        "PRE_CUSTOM_PHONEME_MISSING",
        "preflight",
        "Custom phoneme map path is configured but not found.",
        "Fix the custom phoneme file path or disable the option.",
    ),
    "PRE_DUPLICATE_CASE": ErrorCodeEntry(
        "PRE_DUPLICATE_CASE",
        "preflight",
        "Batch config contains duplicated case names.",
        "Rename duplicated cases to unique names.",
    ),
    "PRE_OUTPUT_COLLISION": ErrorCodeEntry(
        "PRE_OUTPUT_COLLISION",
        "preflight",
        "Multiple enabled cases resolve to the same output oto path.",
        "Use unique output paths per case.",
    ),
    "PRE_REPLACE_COLLISION": ErrorCodeEntry(
        "PRE_REPLACE_COLLISION",
        "preflight",
        "Multiple enabled cases target the same replace_oto_ini file.",
        "Disable replace mode for conflicting cases or split targets.",
    ),
    "ALIGN_EXEC_MISSING": ErrorCodeEntry(
        "ALIGN_EXEC_MISSING",
        "align",
        "Alignment executable is missing.",
        "Install/repair the aligner runtime and verify executable path.",
    ),
    "ALIGN_MODEL_MISSING": ErrorCodeEntry(
        "ALIGN_MODEL_MISSING",
        "align",
        "Required alignment model is missing.",
        "Download or configure the proper aligner model.",
    ),
    "ALIGN_DICT_MISSING": ErrorCodeEntry(
        "ALIGN_DICT_MISSING",
        "align",
        "Required pronunciation dictionary is missing.",
        "Generate or provide a matching dictionary file.",
    ),
    "ALIGN_NOT_READY": ErrorCodeEntry(
        "ALIGN_NOT_READY",
        "align",
        "Aligner dependencies are not ready.",
        "Run aligner diagnose/repair and retry.",
    ),
    "ALIGN_OUTPUT_EMPTY": ErrorCodeEntry(
        "ALIGN_OUTPUT_EMPTY",
        "align",
        "Alignment completed without generating usable TextGrid outputs.",
        "Inspect aligner logs and input coverage, then rerun alignment.",
    ),
    "ALIGN_RUN_FAILED": ErrorCodeEntry(
        "ALIGN_RUN_FAILED",
        "align",
        "Alignment run failed.",
        "Check runtime logs, dependency state, and command arguments.",
    ),
    "ALIGN_USING_EXISTING": ErrorCodeEntry(
        "ALIGN_USING_EXISTING",
        "align",
        "Runtime alignment was skipped because existing TextGrid files were reused.",
        "No action required unless you want to regenerate alignment from scratch.",
    ),
    "ALIGN_SKIPPED": ErrorCodeEntry(
        "ALIGN_SKIPPED",
        "align",
        "Alignment stage was intentionally skipped by configuration.",
        "Ensure TextGrid input is already prepared for generation.",
    ),
    "ML_POLICY_OFF": ErrorCodeEntry(
        "ML_POLICY_OFF",
        "ml",
        "ML correction policy is disabled for this run.",
        "Enable ML policy if post-correction is required.",
    ),
    "ML_DISABLED_ENV": ErrorCodeEntry(
        "ML_DISABLED_ENV",
        "ml",
        "ML correction is disabled by environment/runtime flags.",
        "Clear the disable flag or set ML policy to auto/on.",
    ),
    "ML_INPUT_MISSING": ErrorCodeEntry(
        "ML_INPUT_MISSING",
        "ml",
        "Required ML input artifacts are missing.",
        "Generate required intermediate files and rerun ML correction.",
    ),
    "ML_MODEL_MISSING": ErrorCodeEntry(
        "ML_MODEL_MISSING",
        "ml",
        "ML model bundle could not be located.",
        "Install the ML model bundle or set the workspace/model path.",
    ),
    "ML_BUNDLE_INVALID": ErrorCodeEntry(
        "ML_BUNDLE_INVALID",
        "ml",
        "ML model bundle exists but is invalid or incomplete.",
        "Reinstall or rebuild the ML model bundle.",
    ),
    "ML_ROUTE_UNAVAILABLE": ErrorCodeEntry(
        "ML_ROUTE_UNAVAILABLE",
        "ml",
        "Requested ML route is unavailable in the current runtime.",
        "Use an available ML route or adjust runtime policy.",
    ),
    "ML_NO_FEATURES": ErrorCodeEntry(
        "ML_NO_FEATURES",
        "ml",
        "ML feature extraction produced no usable rows.",
        "Inspect input quality and feature extraction prerequisites.",
    ),
    "ML_INFER_FAILED": ErrorCodeEntry(
        "ML_INFER_FAILED",
        "ml",
        "ML inference failed during runtime.",
        "Check model compatibility/logs and retry with fallback policy.",
    ),
    "ML_APPLIED": ErrorCodeEntry(
        "ML_APPLIED",
        "ml",
        "ML correction was applied successfully.",
        "No action required.",
    ),
    "GENERATOR_ERROR": ErrorCodeEntry(
        "GENERATOR_ERROR",
        "generate",
        "OTO generation finished with one or more file-level errors.",
        "Inspect per-file logs and rerun after fixing problematic inputs.",
    ),
    "UNSUPPORTED_LANGUAGE": ErrorCodeEntry(
        "UNSUPPORTED_LANGUAGE",
        "input",
        "The selected language is not supported by the current generator.",
        "Set language to 'korean' or 'japanese' and rerun.",
    ),
    "VOICEBANK_MISSING": ErrorCodeEntry(
        "VOICEBANK_MISSING",
        "input",
        "Voicebank directory is missing or not readable.",
        "Verify the voicebank path and directory permissions.",
    ),
    "TEXTGRID_MISSING": ErrorCodeEntry(
        "TEXTGRID_MISSING",
        "input",
        "TextGrid directory is missing or has no TextGrid files.",
        "Provide valid TextGrid input or enable runtime alignment fallback.",
    ),
    "EXCEPTION": ErrorCodeEntry(
        "EXCEPTION",
        "all",
        "Unexpected unhandled exception occurred.",
        "Inspect traceback logs, then retry after addressing root cause.",
    ),
}


def get_error_entry(code: str) -> Optional[ErrorCodeEntry]:
    return ERROR_CATALOG.get(str(code or "").strip().upper())


def format_error_with_recovery(code: str, message: str = "") -> str:
    normalized = str(code or "").strip().upper()
    entry = get_error_entry(normalized)
    msg = str(message or "").strip()
    if not entry:
        if msg:
            return f"[{normalized}] {msg}"
        return f"[{normalized}]"
    if msg:
        return f"[{entry.code}] {msg} | recovery: {entry.recovery}"
    return f"[{entry.code}] cause: {entry.cause} | recovery: {entry.recovery}"
