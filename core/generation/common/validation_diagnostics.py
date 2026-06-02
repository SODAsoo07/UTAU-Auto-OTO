from __future__ import annotations

from typing import Mapping


def adapter_warning_validation_metrics(row: Mapping[str, object] | object) -> dict[str, object]:
    """Convert adapter row/anchor warnings into validation splitter metrics."""
    if not isinstance(row, Mapping):
        return {}
    out: dict[str, object] = {}
    out.update(_timing_clamp_validation_metrics(row))
    out.update(_row_warning_validation_metrics(row))
    out.update(_row_rule_validation_metrics(row))
    out.update(_local_refine_validation_metrics(row))
    return out


def runtime_metadata_validation_metrics(metadata: Mapping[str, object] | object) -> dict[str, object]:
    """Convert runtime posterior metadata into validation splitter metrics."""
    if not isinstance(metadata, Mapping) or not bool(metadata.get("rule_based")):
        return {}
    reason = str(metadata.get("rule_fallback_reason") or "rule_based")
    out: dict[str, object] = {"rule_based_inference_count": 1.0}
    if reason.startswith("checkpoint_missing_or_unset"):
        out["rule_based_checkpoint_missing_count"] = 1.0
    if reason.startswith("checkpoint_inference_failed"):
        out["rule_based_checkpoint_failed_count"] = 1.0
    return out


def _local_refine_validation_metrics(row: Mapping[str, object]) -> dict[str, object]:
    anchor = row.get("anchor")
    if not isinstance(anchor, Mapping):
        return {}
    warnings = [str(item) for item in list(anchor.get("warnings", []) or [])]
    delta = _float_or_none(anchor.get("local_refine_delta_ms"))
    margin = _float_or_none(anchor.get("local_refine_margin"))
    out: dict[str, object] = {}
    if delta is not None:
        out["local_refine_max_delta_ms"] = abs(float(delta))
    if margin is not None:
        out["local_refine_min_margin"] = float(margin)
    if "local_refine_changed_anchor" in warnings:
        out["local_refine_changed_anchor_count"] = 1.0
    if "local_refine_low_margin" in warnings:
        out["local_refine_low_margin_count"] = 1.0
    if "local_refine_rejected_slot_boundary" in warnings:
        out["local_refine_rejected_slot_boundary_count"] = 1.0
    return out


def _timing_clamp_validation_metrics(row: Mapping[str, object]) -> dict[str, object]:
    warnings = [str(item) for item in list(row.get("warnings", []) or [])]
    if not warnings:
        return {}
    out: dict[str, object] = {}
    clamp_fields = [
        warning.rsplit(".", 1)[-1]
        for warning in warnings
        if warning.startswith("timing_clamped.")
    ]
    if clamp_fields:
        out["timing_clamp_count"] = float(len(clamp_fields))
        for field in sorted(set(clamp_fields)):
            out[f"timing_clamp_{field}_count"] = float(clamp_fields.count(field))
    deltas: list[float] = []
    for warning in warnings:
        if not warning.startswith("timing_clamp_delta_ms:"):
            continue
        value = _float_or_none(warning.split(":", 1)[1])
        if value is not None:
            deltas.append(abs(float(value)))
    if deltas:
        out["timing_clamp_max_delta_ms"] = max(deltas)
    if "timing_clamp_large_delta" in warnings:
        out["timing_clamp_large_delta_count"] = 1.0
    return out


def _row_warning_validation_metrics(row: Mapping[str, object]) -> dict[str, object]:
    warnings = [str(item) for item in list(row.get("warnings", []) or [])]
    warnings.extend(str(item) for item in list(row.get("local_gate_warnings", []) or []))
    if not warnings:
        return {}
    out: dict[str, object] = {}
    if "island_count_mismatch" in warnings:
        out["row_warning_island_count_mismatch_count"] = 1.0
    if any(warning.startswith("preutterance_slot_shift") for warning in warnings):
        out["row_warning_preutterance_slot_shift_count"] = 1.0
    if any("local_gate_low" in warning for warning in warnings):
        out["row_warning_local_gate_low_count"] = 1.0
    if any("missing_local_gate" in warning for warning in warnings):
        out["row_warning_missing_local_gate_count"] = 1.0
    return out


def _row_rule_validation_metrics(row: Mapping[str, object]) -> dict[str, object]:
    rules = [str(item) for item in list(row.get("applied_rules", []) or [])]
    if not rules:
        return {}
    out: dict[str, object] = {}
    if "special_alias_source_timing_preserve" in rules:
        out["special_alias_source_timing_preserve_count"] = 1.0
    return out


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number == number else None


__all__ = ["adapter_warning_validation_metrics", "runtime_metadata_validation_metrics"]
