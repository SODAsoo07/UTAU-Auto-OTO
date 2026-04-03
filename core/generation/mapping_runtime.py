from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from .mapping_reason_codes import (
    COMMON_REASON_FILENAME_TOKEN,
    KR_REASON_WORDS_KEEP,
    normalize_mapping_reason_code,
    resolve_mapping_reason_schema,
)


def _safe_report(runtime_report: Optional[MutableMapping[str, Any]]) -> Optional[MutableMapping[str, Any]]:
    if isinstance(runtime_report, MutableMapping):
        return runtime_report
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _sanitize_payload_value(value: Any):
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, tuple):
        out = []
        for item in value:
            safe_item = _sanitize_payload_value(item)
            if safe_item is not None:
                out.append(safe_item)
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            safe_item = _sanitize_payload_value(item)
            if safe_item is not None:
                out.append(safe_item)
        return out
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            safe_item = _sanitize_payload_value(item)
            if safe_item is not None:
                out[key_text] = safe_item
        return out
    return None


def _sanitize_runtime_mapping_extras(extra_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    extras = dict(extra_fields or {})
    if not extras:
        return {}
    out: Dict[str, Any] = {}
    for key, value in extras.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        safe_value = _sanitize_payload_value(value)
        if safe_value is not None:
            out[key_text] = safe_value
    return out


def _sanitize_vc_bridge_tuning_payload(bridge_tuning: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    tuning = dict(bridge_tuning or {})
    if not tuning:
        return {}
    return {
        "enabled": bool(tuning.get("enabled")),
        "source": str(tuning.get("source") or ""),
        "sample_count": _safe_int(tuning.get("sample_count"), 0),
        "tempo_scale": _safe_float(tuning.get("tempo_scale"), 1.0),
        "release_scale": _safe_float(tuning.get("release_scale"), 1.0),
        "noise_scale": _safe_float(tuning.get("noise_scale"), 0.0),
        "reason": str(tuning.get("reason") or ""),
    }


def _sanitize_vc_bridge_ab_payload(vc_bridge_ab_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = dict(vc_bridge_ab_summary or {})
    if not summary:
        return {}
    return {
        "rows": _safe_int(summary.get("rows"), 0),
        "pre_abs_shift_mean_ms": _safe_float(summary.get("pre_abs_shift_mean_ms"), 0.0),
        "pre_abs_shift_abs_mean_ms": _safe_float(summary.get("pre_abs_shift_abs_mean_ms"), 0.0),
        "cons_shift_mean_ms": _safe_float(summary.get("cons_shift_mean_ms"), 0.0),
        "cutoff_abs_shift_mean_ms": _safe_float(summary.get("cutoff_abs_shift_mean_ms"), 0.0),
        "ovl_shift_mean_ms": _safe_float(summary.get("ovl_shift_mean_ms"), 0.0),
    }


def build_common_mapping_runtime_payload(
    *,
    language: str,
    fallback_reason_code: str,
    format_value: str,
    mapping_confidence: float,
    mapping_margin: float,
    mapping_tier: str,
    trust_score: float,
    trust_tier: str,
    row_conf_floor: float,
    row_margin_floor: float,
    file_low_conf: bool,
    low_conf_reasons,
    blank_confidence_mean: float,
    plan_source: str,
    plan_margin: float,
    plan_coverage: float,
    mapping_reason_code: str,
) -> Dict[str, Any]:
    reason_code = normalize_mapping_reason_code(
        mapping_reason_code,
        language=str(language or "").strip().lower(),
        fallback_code=fallback_reason_code,
    )
    reason_schema = resolve_mapping_reason_schema(reason_code, language=language)
    return {
        "format": str(format_value or ""),
        "mapping_confidence": float(mapping_confidence or 0.0),
        "mapping_margin": float(mapping_margin or 0.0),
        "mapping_tier": str(mapping_tier or ""),
        "trust_score": float(trust_score or 0.0),
        "trust_tier": str(trust_tier or ""),
        "row_conf_floor": float(row_conf_floor or 0.0),
        "row_margin_floor": float(row_margin_floor or 0.0),
        "file_low_conf": bool(file_low_conf),
        "low_conf_reasons": list(low_conf_reasons or []),
        "blank_confidence_mean": float(blank_confidence_mean or 0.0),
        "plan_source": str(plan_source or ""),
        "plan_margin": float(plan_margin or 0.0),
        "plan_coverage": float(plan_coverage or 0.0),
        "mapping_reason_code": str(reason_code or ""),
        "mapping_reason_known": bool(reason_schema.get("known")),
        "mapping_reason_stage": str(reason_schema.get("stage") or "unknown"),
        "mapping_reason_cause": str(reason_schema.get("cause") or "unknown"),
        "mapping_reason_recovery": str(reason_schema.get("recovery") or "inspect_runtime_log"),
    }


def update_kr_mapping_runtime_report(
    runtime_report: Optional[MutableMapping[str, Any]],
    *,
    file_format: str,
    mapping_confidence: float,
    mapping_margin: float,
    mapping_tier: str,
    trust_score: float,
    trust_tier: str,
    file_conf_floor: float,
    row_conf_floor: float,
    row_margin_floor: float,
    file_low_conf: bool,
    low_conf_reasons,
    blank_confidence_mean: float,
    plan_source: str,
    plan_margin: float,
    plan_coverage: float,
    mapping_reason_code: str,
    row_blank_floor: Optional[float] = None,
    vc_bridge_tuning: Optional[Dict[str, Any]] = None,
    vc_bridge_ab_summary: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    report = _safe_report(runtime_report)
    if report is None:
        return
    payload = build_common_mapping_runtime_payload(
        language="kr",
        fallback_reason_code=KR_REASON_WORDS_KEEP,
        format_value=file_format,
        mapping_confidence=mapping_confidence,
        mapping_margin=mapping_margin,
        mapping_tier=mapping_tier,
        trust_score=trust_score,
        trust_tier=trust_tier,
        row_conf_floor=row_conf_floor,
        row_margin_floor=row_margin_floor,
        file_low_conf=file_low_conf,
        low_conf_reasons=low_conf_reasons,
        blank_confidence_mean=blank_confidence_mean,
        plan_source=plan_source,
        plan_margin=plan_margin,
        plan_coverage=plan_coverage,
        mapping_reason_code=mapping_reason_code,
    )
    payload["file_conf_floor"] = float(file_conf_floor or 0.0)
    if row_blank_floor is not None:
        payload["row_blank_floor"] = float(row_blank_floor)
    tuning_payload = _sanitize_vc_bridge_tuning_payload(vc_bridge_tuning)
    if tuning_payload:
        payload["vc_bridge_tuning"] = tuning_payload
        payload["vc_bridge_tuned"] = bool(tuning_payload.get("enabled"))
    ab_payload = _sanitize_vc_bridge_ab_payload(vc_bridge_ab_summary)
    if ab_payload:
        payload["vc_bridge_ab"] = ab_payload
    extra_payload = _sanitize_runtime_mapping_extras(extra_fields)
    if extra_payload:
        payload.update(extra_payload)
    report["mapping"] = payload


def update_ja_mapping_runtime_report(
    runtime_report: Optional[MutableMapping[str, Any]],
    *,
    format_type: str,
    mapping_confidence: float,
    mapping_margin: float,
    mapping_tier: str,
    trust_score: float,
    trust_tier: str,
    conf_threshold: float,
    row_conf_floor: float,
    row_margin_floor: float,
    file_low_conf: bool,
    low_conf_reasons,
    blank_confidence_mean: float,
    plan_source: str,
    plan_margin: float,
    plan_coverage: float,
    mapping_reason_code: str,
    filename_order_locked: bool,
    forced_words_mapping: bool,
    vc_bridge_tuning: Optional[Dict[str, Any]] = None,
    vc_bridge_ab_summary: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    report = _safe_report(runtime_report)
    if report is None:
        return
    payload = build_common_mapping_runtime_payload(
        language="ja",
        fallback_reason_code=COMMON_REASON_FILENAME_TOKEN,
        format_value=format_type,
        mapping_confidence=mapping_confidence,
        mapping_margin=mapping_margin,
        mapping_tier=mapping_tier,
        trust_score=trust_score,
        trust_tier=trust_tier,
        row_conf_floor=row_conf_floor,
        row_margin_floor=row_margin_floor,
        file_low_conf=file_low_conf,
        low_conf_reasons=low_conf_reasons,
        blank_confidence_mean=blank_confidence_mean,
        plan_source=plan_source,
        plan_margin=plan_margin,
        plan_coverage=plan_coverage,
        mapping_reason_code=mapping_reason_code,
    )
    payload["conf_threshold"] = float(conf_threshold or 0.0)
    payload["filename_order_locked"] = bool(filename_order_locked)
    payload["forced_words_mapping"] = bool(forced_words_mapping)
    tuning_payload = _sanitize_vc_bridge_tuning_payload(vc_bridge_tuning)
    if tuning_payload:
        payload["vc_bridge_tuning"] = tuning_payload
        payload["vc_bridge_tuned"] = bool(tuning_payload.get("enabled"))
    ab_payload = _sanitize_vc_bridge_ab_payload(vc_bridge_ab_summary)
    if ab_payload:
        payload["vc_bridge_ab"] = ab_payload
    extra_payload = _sanitize_runtime_mapping_extras(extra_fields)
    if extra_payload:
        payload.update(extra_payload)
    report["mapping"] = payload


def update_mapping_vc_bridge_runtime_report(
    runtime_report: Optional[MutableMapping[str, Any]],
    *,
    vc_bridge_tuning: Optional[Dict[str, Any]] = None,
    vc_bridge_ab_summary: Optional[Dict[str, Any]] = None,
) -> None:
    report = _safe_report(runtime_report)
    if report is None:
        return
    mapping = report.get("mapping")
    if not isinstance(mapping, MutableMapping):
        mapping = {}
        report["mapping"] = mapping
    tuning_payload = _sanitize_vc_bridge_tuning_payload(vc_bridge_tuning)
    if tuning_payload:
        mapping["vc_bridge_tuning"] = tuning_payload
        mapping["vc_bridge_tuned"] = bool(tuning_payload.get("enabled"))
    ab_payload = _sanitize_vc_bridge_ab_payload(vc_bridge_ab_summary)
    if ab_payload:
        mapping["vc_bridge_ab"] = ab_payload


def compute_runtime_low_conf_state(
    *,
    runtime_policy: Optional[MutableMapping[str, Any]],
    mapping_confidence: float,
    conf_floor: float,
    blank_confidence_mean: float,
    conf_below_reason: str,
    row_conf_floor_default: Optional[float] = None,
) -> Dict[str, Any]:
    policy = dict(runtime_policy or {})
    conf = float(mapping_confidence or 0.0)
    floor = float(conf_floor or 0.0)
    row_conf_default = float(row_conf_floor_default if row_conf_floor_default is not None else floor)
    row_conf_floor = float(policy.get("row_conf_floor", row_conf_default))
    row_margin_floor = float(policy.get("row_margin_floor", 6.0))
    file_low_conf = bool(policy.get("is_low_conf")) or (conf < floor)
    low_conf_reasons = list(policy.get("low_conf_reasons") or [])
    if blank_confidence_mean >= 0.55 and "blank_confidence_high" not in low_conf_reasons:
        low_conf_reasons.append("blank_confidence_high")
    reason = str(conf_below_reason or "").strip()
    if reason and conf < floor and reason not in low_conf_reasons:
        low_conf_reasons.append(reason)
    return {
        "row_conf_floor": float(row_conf_floor),
        "row_margin_floor": float(row_margin_floor),
        "file_low_conf": bool(file_low_conf),
        "low_conf_reasons": low_conf_reasons,
    }


def format_alignment_guard_summary(
    *,
    alignment_weight: float,
    blank_confidence_mean: float,
    anchor_lock_lite: bool,
) -> str:
    return (
        f"alignment_weight={float(alignment_weight or 0.0):.2f}, "
        f"blank_mean={float(blank_confidence_mean or 0.0):.2f}, "
        f"anchor_lock_lite={bool(anchor_lock_lite)}"
    )


def format_mapping_summary(mapping: Dict[str, Any]) -> str:
    if not isinstance(mapping, dict):
        return "format=- tier=- conf=0.00 reason=-"
    return (
        f"format={mapping.get('format', '-') or '-'} "
        f"tier={mapping.get('mapping_tier', '-') or '-'} "
        f"conf={float(mapping.get('mapping_confidence', 0.0) or 0.0):.2f} "
        f"reason={mapping.get('mapping_reason_code', '-') or '-'}"
    )


def format_mapping_reason_schema_summary(mapping: Dict[str, Any]) -> str:
    if not isinstance(mapping, dict):
        return "known=False stage=unknown cause=unknown recovery=inspect_runtime_log"

    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off", ""}:
                return False
        return bool(value)

    known = _coerce_bool(mapping.get("mapping_reason_known", False))
    stage = str(mapping.get("mapping_reason_stage") or "unknown")
    cause = str(mapping.get("mapping_reason_cause") or "unknown")
    recovery = str(mapping.get("mapping_reason_recovery") or "inspect_runtime_log")
    return (
        f"known={known} "
        f"stage={stage} "
        f"cause={cause} "
        f"recovery={recovery}"
    )
