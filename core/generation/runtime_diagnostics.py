from __future__ import annotations

from typing import Any, Dict, Mapping


def build_mapping_failure_meta(
    *,
    mapping_confidence: float,
    mapping_reason_code: str,
    spn_ratio: float | None = None,
    include_reason_in_diag: bool = False,
    mapping_tier: str = "",
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    parts = []
    if spn_ratio is not None:
        parts.append(f"spn_ratio={float(spn_ratio):.2f}")
    parts.append(f"conf={float(mapping_confidence or 0.0):.2f}")
    reason = str(mapping_reason_code or "")
    if include_reason_in_diag and reason:
        parts.append(f"reason={reason}")
    out: Dict[str, Any] = {
        "diag_hint": "; ".join(parts),
        "mapping_confidence": float(mapping_confidence or 0.0),
        "mapping_reason_code": reason,
    }
    tier = str(mapping_tier or "")
    if tier:
        out["mapping_tier"] = tier
    if isinstance(extra, Mapping):
        out.update(dict(extra))
    return out


def build_mapping_abstain_meta(
    *,
    mapping_confidence: float,
    mapping_reason_code: str,
    plan_policy,
    mapping_tier: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mapping_confidence": float(mapping_confidence or 0.0),
        "mapping_reason_code": str(mapping_reason_code or ""),
        "plan_policy": dict(plan_policy or {}),
    }
    tier = str(mapping_tier or "")
    if tier:
        out["mapping_tier"] = tier
    return out
