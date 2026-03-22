from __future__ import annotations


def apply_row_confidence_penalty(confidence, amount, *, floor=0.0):
    try:
        conf = float(confidence)
    except Exception:
        conf = 0.0
    try:
        penalty = float(amount)
    except Exception:
        penalty = 0.0
    try:
        floor_value = float(floor)
    except Exception:
        floor_value = 0.0
    return max(floor_value, conf - penalty)


def should_trace_mapping_decision(*, mapping_tier="", expected_idx=0, mapped_idx=0, target_token="", mapped_token=""):
    tier = str(mapping_tier or "").strip().lower()
    if tier == "low":
        return True
    if int(mapped_idx) != int(expected_idx):
        return True
    return str(target_token or "").strip() != str(mapped_token or "").strip()


def build_mapping_trace_record(
    *,
    event,
    fname,
    alias,
    alias_type,
    format_type,
    target_token,
    expected_idx,
    mapped_idx,
    expected_token,
    mapped_token,
    mapping_tier,
    mapping_reason_code,
    mapping_confidence,
    extra=None,
):
    record = {
        "event": str(event or ""),
        "file": str(fname or ""),
        "alias": str(alias or ""),
        "alias_type": str(alias_type or ""),
        "format_type": str(format_type or ""),
        "target_token": str(target_token or ""),
        "expected_index": int(expected_idx),
        "mapped_index": int(mapped_idx),
        "delta": int(mapped_idx) - int(expected_idx),
        "expected_token": str(expected_token or ""),
        "mapped_token": str(mapped_token or ""),
        "mapping_tier": str(mapping_tier or ""),
        "mapping_reason_code": str(mapping_reason_code or ""),
        "mapping_confidence": float(mapping_confidence or 0.0),
    }
    if extra:
        record.update(dict(extra))
    return record


__all__ = [
    "apply_row_confidence_penalty",
    "build_mapping_trace_record",
    "should_trace_mapping_decision",
]
