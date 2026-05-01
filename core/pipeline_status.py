from __future__ import annotations

import os
from typing import Dict, List

OK = "OK"

# TICKET-001: Per-engine baseline quality tier (0.0 = unusable, 1.0 = best).
# A profile downgrade (e.g. accurate→fast) subtracts an additional 0.10.
ALIGNER_QUALITY_TIER: Dict[str, float] = {
    "mfa": 1.0,
    "sequence": 0.55,
    "domino": 0.80,
    "none": 0.0,
    "existing": 0.85,
}


def compute_alignment_quality_score(
    used_engine: str,
    fallback_notes: List[str],
) -> float:
    """Return an alignment quality score in [0.0, 1.0].

    Base score comes from ALIGNER_QUALITY_TIER.  Each fallback event
    (engine downgrade OR profile downgrade) reduces the score by 0.10,
    clamped to 0.0 at minimum.
    """
    eng = str(used_engine or "").strip().lower()
    base = ALIGNER_QUALITY_TIER.get(eng, 0.5)
    penalty = len(fallback_notes) * 0.10
    return max(0.0, round(base - penalty, 4))


ALIGN_USING_EXISTING = "ALIGN_USING_EXISTING"
ALIGN_EXEC_MISSING = "ALIGN_EXEC_MISSING"
ALIGN_MODEL_MISSING = "ALIGN_MODEL_MISSING"
ALIGN_DICT_MISSING = "ALIGN_DICT_MISSING"
ALIGN_NOT_READY = "ALIGN_NOT_READY"
ALIGN_RUN_FAILED = "ALIGN_RUN_FAILED"
ALIGN_OUTPUT_EMPTY = "ALIGN_OUTPUT_EMPTY"
ALIGN_SKIPPED = "ALIGN_SKIPPED"

ML_POLICY_OFF = "ML_POLICY_OFF"
ML_DISABLED_ENV = "ML_DISABLED_ENV"
ML_INPUT_MISSING = "ML_INPUT_MISSING"
ML_MODEL_MISSING = "ML_MODEL_MISSING"
ML_BUNDLE_INVALID = "ML_BUNDLE_INVALID"
ML_ROUTE_UNAVAILABLE = "ML_ROUTE_UNAVAILABLE"
ML_NO_FEATURES = "ML_NO_FEATURES"
ML_INFER_FAILED = "ML_INFER_FAILED"
ML_APPLIED = "ML_APPLIED"

GENERATOR_ERROR = "GENERATOR_ERROR"
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
VOICEBANK_MISSING = "VOICEBANK_MISSING"
TEXTGRID_MISSING = "TEXTGRID_MISSING"
EXCEPTION = "EXCEPTION"


def normalize_ml_policy(value, enabled_default: bool = True) -> str:
    if isinstance(value, bool):
        return "auto" if value else "off"
    text = str(value or "").strip().lower()
    if not text:
        return "auto" if enabled_default else "off"
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return "off"
    if text in {"1", "true", "yes", "on", "force", "required"}:
        return "on"
    if text in {"auto", "default", "if_ready", "optional"}:
        return "auto"
    return "auto" if enabled_default else "off"


def normalize_aligner_name(value, default: str = "mfa") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"none", "nomfa", "no-mfa", "no_mfa", "no mfa", "off", "skip"}:
        return "none"
    if text in {"mfa", "montreal"}:
        return "mfa"
    if text in {
        "sequence",
        "seq",
        "sequence_label",
        "sequence-label",
        "sequence_labeling",
        "시퀀스",
        "전용(시퀀스)",
        "전용시퀀스",
        "dedicated",
        "dedicated_aligner",
        "utau-sequence",
        "utau_sequence",
    }:
        return "sequence"
    if text in {"domino", "pydomino", "domino (jp)", "domino(jp)", "jp_domino", "jp-domino"}:
        return "domino"
    if "no-mfa" in text or "nomfa" in text:
        return "none"
    if "sequence" in text or "dedicated" in text or "시퀀스" in text:
        return "sequence"
    if "domino" in text:
        return "domino"
    return default


def resolve_aligner_chain(primary, fallback="") -> List[str]:
    chain: List[str] = []
    for raw in (primary, fallback):
        name = normalize_aligner_name(raw, default="")
        if not name or name in chain:
            continue
        chain.append(name)
    if not chain:
        chain.append("mfa")
    return chain


def has_textgrid_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for _dp, _dns, fns in os.walk(path):
        if any(fn.lower().endswith(".textgrid") for fn in fns):
            return True
    return False


def make_runtime_report(stage: str, code: str, message: str = "", **extra) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "stage": str(stage or "").strip().lower(),
        "code": str(code or OK).strip().upper(),
        "message": str(message or "").strip(),
    }
    payload.update(extra)
    return payload


def classify_alignment_error(engine: str, message: str) -> str:
    text = str(message or "").strip()
    lowered = text.lower()
    eng = normalize_aligner_name(engine, default="")
    if not text:
        return ALIGN_RUN_FAILED
    if "textgrid" in lowered and ("not found" in lowered or "missing" in lowered):
        return ALIGN_OUTPUT_EMPTY
    if "dictionary" in lowered:
        return ALIGN_DICT_MISSING
    if "checkpoint" in lowered or ".ckpt" in lowered:
        return ALIGN_MODEL_MISSING
    if ".onnx" in lowered or "onnx" in lowered:
        return ALIGN_MODEL_MISSING
    tokenizer_markers = (
        "missing korean tokenizer dependencies",
        "missing japanese tokenizer dependencies",
        "japanese tokenizer readiness is uncertain",
        "korean tokenizer dependencies (jamo + mecab backend) are missing",
        "please install korean support",
    )
    if any(marker in lowered for marker in tokenizer_markers):
        return ALIGN_NOT_READY
    if "japanese only" in lowered:
        return ALIGN_NOT_READY
    if "executable not found" in lowered or "infer.py" in lowered:
        return ALIGN_EXEC_MISSING
    if eng == "mfa" and "mfa executable" in lowered:
        return ALIGN_EXEC_MISSING
    if eng == "domino" and ("pydomino" in lowered or "domino executable" in lowered):
        return ALIGN_EXEC_MISSING
    if eng == "sequence" and ("python-textgrid" in lowered or "textgrid import" in lowered):
        return ALIGN_NOT_READY
    return ALIGN_RUN_FAILED


def first_non_ok_code(reports: List[Dict[str, object]], default: str = OK) -> str:
    for report in reports:
        code = str((report or {}).get("code", "") or "").strip().upper()
        if code and code != OK:
            return code
    return default


__all__ = [
    "ALIGN_DICT_MISSING",
    "ALIGN_EXEC_MISSING",
    "ALIGN_MODEL_MISSING",
    "ALIGN_NOT_READY",
    "ALIGN_OUTPUT_EMPTY",
    "ALIGN_RUN_FAILED",
    "ALIGN_SKIPPED",
    "ALIGN_USING_EXISTING",
    "ALIGNER_QUALITY_TIER",
    "EXCEPTION",
    "GENERATOR_ERROR",
    "ML_APPLIED",
    "ML_BUNDLE_INVALID",
    "ML_DISABLED_ENV",
    "ML_INFER_FAILED",
    "ML_INPUT_MISSING",
    "ML_MODEL_MISSING",
    "ML_NO_FEATURES",
    "ML_POLICY_OFF",
    "ML_ROUTE_UNAVAILABLE",
    "OK",
    "TEXTGRID_MISSING",
    "UNSUPPORTED_LANGUAGE",
    "VOICEBANK_MISSING",
    "classify_alignment_error",
    "compute_alignment_quality_score",
    "first_non_ok_code",
    "has_textgrid_files",
    "make_runtime_report",
    "normalize_aligner_name",
    "normalize_ml_policy",
    "resolve_aligner_chain",
]
