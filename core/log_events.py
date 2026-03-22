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
    elif text.startswith("[OTO-ML]"):
        category = "ml"
        severity = "debug"
    elif text.startswith("[") and "]" in text:
        category = "internal"
        severity = "debug"
    elif "vc bridge a/b" in lowered:
        # Keep VC bridge diagnostics only in detail log.
        category = "bridge_debug"
        severity = "debug"
        ui_visible = False
    elif "token invariant" in lowered:
        # Token-invariant diagnostics are for detail log only.
        category = "bridge_debug"
        severity = "debug"
        ui_visible = False
    elif text.startswith(("[ERROR]", "ERROR:")) or any(
        token in lowered for token in ("error", "failed", "exception", "traceback")
    ):
        severity = "error"
        category = "error"
        ui_visible = True
    elif text.startswith(("[WARN]", "WARNING:")) or any(
        token in lowered for token in ("warning", "skipped", "skip")
    ):
        severity = "warning"
        category = "warning"
        ui_visible = True
    elif text.startswith(("[OK]", "SUCCESS:")):
        severity = "success"
        category = "success"
        ui_visible = True
    elif text.startswith(("🧪", "-", "•")) or any(
        key in lowered for key in ("progress", "processing", "complete", "done")
    ):
        severity = "info"
        category = "progress"
        ui_visible = True

    if "validation" in lowered:
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
