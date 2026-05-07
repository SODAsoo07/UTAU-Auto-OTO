from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any, Dict, Optional


def _apply_mode_rank(mode: str) -> int:
    key = str(mode or "").strip().lower()
    order = {
        "full_apply": 0,
        "conservative_apply": 1,
        "template_preserve": 2,
        "review_required": 3,
    }
    return int(order.get(key, 0))


def _is_high_pitch_zone(pitch_zone: str) -> bool:
    return str(pitch_zone or "").strip().lower() in {"high", "ultra_high"}


def decide_row_application(
    *,
    routing_profile: str,
    pitch_zone: str,
    row_mapping_confidence: float,
    row_conf_floor: float,
    row_blank_confidence: Optional[float],
    row_blank_floor: Optional[float],
    row_jump_blocked: int,
    forced_selected: bool,
    file_mapping_low_conf: bool,
    row_abstain_skip: bool,
    row_abstain_reason: str = "",
) -> Dict[str, object]:
    reasons = []
    mode = "full_apply"

    if row_abstain_skip:
        reason = str(row_abstain_reason or "row_abstain").strip().lower() or "row_abstain"
        if reason in {
            "row_low_confidence",
            "row_low_margin_candidate",
            "row_high_blank_confidence",
        }:
            return {
                "mode": "conservative_apply",
                "reason_code": reason,
                "reasons": [reason, "low_conf_force_auto"],
            }
        return {
            "mode": "review_required",
            "reason_code": reason,
            "reasons": [reason],
        }

    if file_mapping_low_conf or float(row_mapping_confidence) < float(row_conf_floor):
        mode = "conservative_apply"
        reasons.append("low_model_conf")
    if (
        row_blank_floor is not None
        and row_blank_confidence is not None
        and float(row_blank_confidence) >= float(row_blank_floor)
    ):
        mode = "conservative_apply"
        reasons.append("blank_conf_high")
    if int(row_jump_blocked or 0) > 0:
        mode = "conservative_apply"
        reasons.append("jump_blocked")
    if bool(forced_selected) and mode in {"full_apply", "conservative_apply"}:
        mode = "conservative_apply"
        reasons.append("plan_mismatch")

    if _is_high_pitch_zone(pitch_zone) and mode != "review_required":
        if float(row_mapping_confidence) < max(0.0, float(row_conf_floor) - 0.06):
            mode = "conservative_apply"
            reasons.append("high_pitch_unstable")

    if str(routing_profile or "").strip().lower() == "high_pitch_safe" and mode == "full_apply":
        mode = "conservative_apply"
        reasons.append("high_pitch_unstable")

    reason_code = str(reasons[0]) if reasons else ""
    return {
        "mode": str(mode),
        "reason_code": reason_code,
        "reasons": list(reasons),
    }


def apply_zero_template_compute_policy(
    row_apply: MutableMapping[str, Any],
    *,
    line: str,
    use_template: bool,
    is_zero_placeholder_line_fn: Callable[[str], bool],
    fname: str,
    alias: str,
    debug_logging: bool,
    log_fn: Callable[[str], None],
) -> MutableMapping[str, Any]:
    mode = str(row_apply.get("mode") or "full_apply")
    if (
        not use_template
        or mode not in {"template_preserve", "review_required"}
        or not is_zero_placeholder_line_fn(line)
    ):
        return row_apply

    prev_mode = mode
    reason_code = str(row_apply.get("reason_code") or "")
    new_reason = f"{reason_code or prev_mode}_zero_template_compute"
    row_apply["mode"] = "conservative_apply"
    row_apply["reason_code"] = new_reason
    reasons = list(row_apply.get("reasons") or [])
    reasons.append("zero_template_compute")
    row_apply["reasons"] = reasons
    if debug_logging:
        log_fn(
            f"[WARN] {fname}: 0값 템플릿 행은 원본 보존 대신 계산값을 적용합니다 "
            f"({prev_mode} -> conservative_apply, {alias})"
        )
    return row_apply


def record_row_apply_decision(
    *,
    row_apply_mode_counts: MutableMapping[str, int],
    row_apply_decisions: list[Dict[str, Any]],
    row_apply: MutableMapping[str, Any],
    fname: str,
    alias: str,
    alias_type: str,
    file_format: str,
    selected_w_idx: Optional[int],
    row_mapping_confidence: float,
    row_blank_confidence: Optional[float],
    row_blank_floor: float,
    routing_profile: str,
    pitch_zone: str,
) -> tuple[str, str]:
    mode = str(row_apply.get("mode") or "full_apply")
    reason_code = str(row_apply.get("reason_code") or "")
    row_apply_mode_counts[mode] = int(row_apply_mode_counts.get(mode, 0)) + 1
    row_apply_decisions.append(
        {
            "file": str(fname),
            "alias": str(alias),
            "alias_type": str(alias_type),
            "format_type": str(file_format or ""),
            "selected_w_idx": int(selected_w_idx) if selected_w_idx is not None else -1,
            "mapping_confidence": float(row_mapping_confidence),
            "blank_confidence": float(row_blank_confidence) if row_blank_confidence is not None else None,
            "row_blank_floor": float(row_blank_floor),
            "apply_mode": str(mode),
            "reason_code": str(reason_code),
            "reasons": list(row_apply.get("reasons") or []),
            "routing_profile": str(routing_profile),
            "pitch_zone": str(pitch_zone),
        }
    )
    return mode, reason_code


def build_review_required_unset(
    *,
    row_apply: MutableMapping[str, Any],
    row_abstain: MutableMapping[str, Any],
    routing_profile: str,
) -> tuple[str, Dict[str, str]]:
    mode = str(row_apply.get("mode") or "review_required")
    reason = str(row_apply.get("reason_code") or row_abstain.get("reason") or "review_required")
    return reason, {
        "diag_hint": str(row_abstain.get("diag_hint", "") or ""),
        "apply_mode": str(mode),
        "routing_profile": str(routing_profile),
    }


def evaluate_row_apply_policy(
    *,
    row_apply_mode_counts: MutableMapping[str, int],
    row_apply_decisions: list[Dict[str, Any]],
    routing_profile: str,
    pitch_zone: str,
    row_mapping_confidence: float,
    row_conf_floor: float,
    row_blank_confidence: Optional[float],
    row_blank_floor: Optional[float],
    row_blank_floor_for_report: float,
    row_jump_blocked: int,
    forced_selected: bool,
    file_mapping_low_conf: bool,
    row_abstain: MutableMapping[str, Any],
    line: str,
    use_template: bool,
    is_zero_placeholder_line_fn: Callable[[str], bool],
    fname: str,
    alias: str,
    alias_type: str,
    file_format: str,
    selected_w_idx: Optional[int],
    debug_logging: bool,
    log_fn: Callable[[str], None],
    log_review_required_fn: Optional[Callable[[str], None]],
    record_unset_fn: Callable[..., None],
    final_lines: list[str],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    alias_suffix: str,
) -> Dict[str, Any]:
    row_apply = decide_row_application(
        routing_profile=routing_profile,
        pitch_zone=pitch_zone,
        row_mapping_confidence=row_mapping_confidence,
        row_conf_floor=row_conf_floor,
        row_blank_confidence=row_blank_confidence,
        row_blank_floor=row_blank_floor,
        row_jump_blocked=row_jump_blocked,
        forced_selected=forced_selected,
        file_mapping_low_conf=file_mapping_low_conf,
        row_abstain_skip=bool(row_abstain.get("should_skip")),
        row_abstain_reason=str(row_abstain.get("reason") or ""),
    )
    apply_zero_template_compute_policy(
        row_apply,
        line=line,
        use_template=use_template,
        is_zero_placeholder_line_fn=is_zero_placeholder_line_fn,
        fname=fname,
        alias=alias,
        debug_logging=debug_logging,
        log_fn=log_fn,
    )
    row_apply_mode, row_apply_reason_code = record_row_apply_decision(
        row_apply_mode_counts=row_apply_mode_counts,
        row_apply_decisions=row_apply_decisions,
        row_apply=row_apply,
        fname=fname,
        alias=alias,
        alias_type=alias_type,
        file_format=file_format,
        selected_w_idx=selected_w_idx,
        row_mapping_confidence=row_mapping_confidence,
        row_blank_confidence=row_blank_confidence,
        row_blank_floor=row_blank_floor_for_report,
        routing_profile=routing_profile,
        pitch_zone=pitch_zone,
    )

    if row_apply_mode == "review_required":
        review_reason, review_meta = build_review_required_unset(
            row_apply=row_apply,
            row_abstain=row_abstain,
            routing_profile=routing_profile,
        )
        if debug_logging and log_review_required_fn is not None:
            log_review_required_fn(review_reason)
        record_unset_fn(
            review_reason,
            fname,
            line,
            meta=review_meta,
        )
        if use_template:
            final_lines.append(
                apply_suffix_to_oto_line_fn(line, alias_suffix)
            )
        return {
            "should_continue": True,
            "row_apply": row_apply,
            "row_apply_mode": row_apply_mode,
            "row_apply_reason_code": row_apply_reason_code,
        }

    if row_apply_mode == "template_preserve" and use_template:
        final_lines.append(apply_suffix_to_oto_line_fn(line, alias_suffix))
        return {
            "should_continue": True,
            "row_apply": row_apply,
            "row_apply_mode": row_apply_mode,
            "row_apply_reason_code": row_apply_reason_code,
        }

    return {
        "should_continue": False,
        "row_apply": row_apply,
        "row_apply_mode": row_apply_mode,
        "row_apply_reason_code": row_apply_reason_code,
    }


__all__ = [
    "apply_zero_template_compute_policy",
    "build_review_required_unset",
    "decide_row_application",
    "evaluate_row_apply_policy",
    "record_row_apply_decision",
]
