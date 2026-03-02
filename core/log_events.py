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

    if text.startswith("[Prepare]"):
        category = "prepare"
        ui_visible = True
    elif text.startswith("[OTO-ML] 런타임 옵션"):
        category = "runtime_option"
        ui_visible = True
    elif text.startswith("[OTO-ML] 모델 로드"):
        category = "ml_model_load"
        severity = "debug"
    elif text.startswith("[OTO-ML]"):
        category = "ml"
        severity = "debug"
    elif text.startswith("[") and "]" in text:
        category = "internal"
        severity = "debug"
    elif text.startswith(("❌", "笶・")) or any(token in lowered for token in ("error", "failed", "exception", "traceback")):
        severity = "error"
        category = "error"
        ui_visible = True
    elif text.startswith(("⚠", "笞")) or "warning" in lowered or "skipped" in lowered or "skip" in lowered:
        severity = "warning"
        category = "warning"
        ui_visible = True
    elif text.startswith(("✅", "笨・")) or "완료" in text or "성공" in text:
        severity = "success"
        category = "success"
        ui_visible = True
    elif text.startswith(("🧪", "🔍", "📦", "🚀", "🔧", "📥", "🌐", "🔀", "ℹ", "📝", "🎉", "처리:", "OTO 생성 중", "Lab 생성", "사전 생성")):
        severity = "info"
        category = "progress"
        ui_visible = True

    if "validation" in lowered or "검증" in text:
        category = "validation"
        ui_visible = True
        if severity == "info":
            severity = "info"

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
