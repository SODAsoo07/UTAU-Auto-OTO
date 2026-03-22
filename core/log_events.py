import datetime
import json
import logging
import os
from typing import Dict


def classify_log_message(message: str) -> Dict[str, object]:
    text = str(message or "").strip()
    lowered = text.lower()
    if not text:
        return {
            "message": "",
            "severity": "debug",
            "category": "empty",
            "ui_visible": False,
        }

    severity = "info"
    category = "general"
    ui_visible = False

    error_prefixes = ("[error]", "error:", "❌", "오류:", "[오류]")
    warn_prefixes = ("[warn]", "warning:", "⚠", "경고:", "[경고]")
    success_prefixes = ("[ok]", "success:", "✅", "성공:", "완료:")
    progress_prefixes = ("ℹ", "⏳", "🔍", "🧪", "📝", "🧹")

    error_tokens = (
        "error",
        "failed",
        "failure",
        "exception",
        "traceback",
        "fatal",
        "오류",
        "실패",
        "예외",
        "치명",
    )
    warning_tokens = (
        "warning",
        "warn",
        "skipped",
        "skip",
        "degraded",
        "경고",
        "건너뜀",
        "건너뜁니다",
        "스킵",
        "누락",
    )
    success_tokens = (
        "success",
        "succeeded",
        "completed",
        "done",
        "완료",
        "성공",
        "정상 완료",
        "준비 완료",
    )
    progress_tokens = (
        "progress",
        "processing",
        "running",
        "in progress",
        "진행",
        "처리 중",
        "생성 중",
        "설치 중",
        "점검 중",
        "정렬 중",
        "검증 중",
        "다운로드 중",
    )

    starts_lower = lowered.startswith

    if "vc bridge a/b" in lowered:
        # Keep VC bridge diagnostics only in detail log.
        category = "bridge_debug"
        severity = "debug"
        ui_visible = False
    elif "token invariant" in lowered:
        # Token-invariant diagnostics are for detail log only.
        category = "bridge_debug"
        severity = "debug"
        ui_visible = False
    elif text.startswith("[OTO-ML]"):
        category = "ml"
        severity = "debug"
    elif text.startswith("[Prepare]"):
        category = "prepare"
        ui_visible = True
    elif any(starts_lower(prefix) for prefix in error_prefixes) or any(token in lowered for token in error_tokens):
        severity = "error"
        category = "error"
        ui_visible = True
    elif any(starts_lower(prefix) for prefix in warn_prefixes) or any(token in lowered for token in warning_tokens):
        severity = "warning"
        category = "warning"
        ui_visible = True
    elif any(starts_lower(prefix) for prefix in success_prefixes) or any(token in lowered for token in success_tokens):
        severity = "success"
        category = "success"
        ui_visible = True
    elif text.startswith(progress_prefixes) or any(key in lowered for key in progress_tokens):
        severity = "info"
        category = "progress"
        ui_visible = True
    elif text.startswith("[") and "]" in text:
        tag = text.split("]", 1)[0].strip().lower() + "]"
        if tag in {"[debug]", "[trace]"}:
            category = "internal"
            severity = "debug"
            ui_visible = False
        else:
            category = "tagged"
            severity = "info"
            ui_visible = True

    if "validation" in lowered or "검증" in text:
        category = "validation"
        ui_visible = True

    return {
        "message": text,
        "severity": severity,
        "category": category,
        "ui_visible": ui_visible,
    }


def append_structured_log(event_log_path: str, record: Dict[str, object]) -> None:
    if not event_log_path:
        return
    os.makedirs(os.path.dirname(event_log_path), exist_ok=True)
    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        **record,
    }
    with open(event_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_with_event(
    logger: logging.Logger,
    event_log_path: str,
    message: str,
    level: int = logging.INFO,
) -> Dict[str, object]:
    record = classify_log_message(message)
    if logger is not None:
        logger.log(level, record["message"])
    append_structured_log(event_log_path, record)
    return record
