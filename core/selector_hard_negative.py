from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _resolve_hard_negative_log_path() -> str:
    override = str(os.environ.get("UTOA_SELECTOR_HN_LOG", "") or "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "logs", "selector_hard_negatives.jsonl")


def _should_log_hard_negative(feature_row: Dict[str, object], reason: str) -> bool:
    if reason == "score_margin":
        return True
    mapping_conf = _to_float(feature_row.get("mapping_confidence"), 0.0)
    used_nuclei = _to_int(feature_row.get("used_nuclei_fallback"), 0) > 0
    jump_blocked = _to_int(feature_row.get("jump_blocked_flag"), 0) > 0
    silence_ratio = max(
        _to_float(feature_row.get("db_silence_ratio"), 0.0),
        _to_float(feature_row.get("mel_window_silence_ratio"), 0.0),
    )
    if mapping_conf < 0.55:
        return True
    if used_nuclei or jump_blocked:
        return True
    if silence_ratio >= 0.65:
        return True
    return False


def log_selector_hard_negative(
    feature_row: Dict[str, object],
    selected: Optional[Dict[str, object]],
    *,
    reason: str,
    margin: Optional[float] = None,
    log_path: str = "",
) -> bool:
    if not feature_row:
        return False
    if not _should_log_hard_negative(feature_row, reason):
        return False
    path = log_path or _resolve_hard_negative_log_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    candidate = selected.get("candidate") if isinstance(selected, dict) else None
    record = {
        "ts": time.time(),
        "reason": str(reason),
        "margin": None if margin is None else float(margin),
        "language": str(feature_row.get("language", "") or ""),
        "format_type": str(feature_row.get("format_type", "") or ""),
        "alias_type": str(feature_row.get("alias_type", "") or ""),
        "alias_group": str(feature_row.get("alias_group", "") or ""),
        "wav": str(feature_row.get("wav", "") or ""),
        "alias": str(feature_row.get("alias", "") or ""),
        "line_index": _to_int(feature_row.get("line_index"), -1),
        "mapping_confidence": _to_float(feature_row.get("mapping_confidence"), 0.0),
        "used_nuclei_fallback": _to_int(feature_row.get("used_nuclei_fallback"), 0),
        "jump_blocked_flag": _to_int(feature_row.get("jump_blocked_flag"), 0),
        "db_silence_ratio": _to_float(feature_row.get("db_silence_ratio"), 0.0),
        "mel_window_silence_ratio": _to_float(feature_row.get("mel_window_silence_ratio"), 0.0),
        "candidate_mode": str(candidate.get("candidate_mode", "") or "") if candidate else "",
        "candidate_index": _to_int(candidate.get("candidate_index"), -1) if candidate else -1,
        "scores": selected.get("scores") if isinstance(selected, dict) else [],
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


__all__ = ["log_selector_hard_negative"]
