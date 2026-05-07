from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from .file_stages import handle_mapping_failure_fallback, resolve_mapping_failure_reason
from .kr_file_loop import rollback_auto_placeholder_fallback
from .runtime_diagnostics import build_mapping_failure_meta


def has_kr_syllable_mapping_failure(syllables_info: Sequence[Mapping[str, Any]] | None) -> bool:
    if not syllables_info:
        return True
    for syllable in syllables_info:
        phones = (syllable or {}).get("phones") or []
        if len(phones) == 0:
            return True
    return False


def handle_kr_mapping_failure_with_guard(
    *,
    syllables_info: Sequence[Mapping[str, Any]] | None,
    low_quality_reasons,
    low_phone_quality: bool,
    has_word_intervals: bool,
    fname: str,
    lines: list[str],
    final_lines: list[str],
    alias_suffix: str,
    log_fn: Callable[[str], None],
    record_unset_lines_fn: Callable[..., None],
    apply_suffix_to_oto_line_fn: Callable[[str, str], str],
    use_template: bool,
    errors: list[str],
    placeholder_block_errors: list[str],
    spn_ratio: float,
    phone_quality: Mapping[str, Any],
    force_words_phone_fill: bool,
    mapping_confidence: float,
    mapping_reason_code: str,
) -> bool:
    if not has_kr_syllable_mapping_failure(syllables_info):
        return False

    fail_reason = resolve_mapping_failure_reason(
        low_quality_reasons=low_quality_reasons,
        low_phone_quality=low_phone_quality,
        has_word_intervals=has_word_intervals,
    )
    meta = build_mapping_failure_meta(
        mapping_confidence=mapping_confidence,
        mapping_reason_code=mapping_reason_code,
        spn_ratio=spn_ratio,
        extra={
            "phone_quality": dict(phone_quality or {}),
            "force_words_phone_fill": bool(force_words_phone_fill),
        },
    )
    prev_final_len = len(final_lines)
    handle_mapping_failure_fallback(
        fail_reason=fail_reason,
        fname=fname,
        lines=lines,
        final_lines=final_lines,
        alias_suffix=alias_suffix,
        log_fn=log_fn,
        failure_log_message=f"[ERROR] {fname}: 음절 경계 해석 실패로 자동 설정을 중단합니다.",
        record_unset_lines_fn=record_unset_lines_fn,
        meta=meta,
        apply_suffix_to_oto_line_fn=apply_suffix_to_oto_line_fn,
    )
    rollback_auto_placeholder_fallback(
        final_lines=final_lines,
        errors=errors,
        placeholder_block_errors=placeholder_block_errors,
        prev_final_len=prev_final_len,
        fname=fname,
        lines=lines,
        reason=fail_reason,
        use_template=use_template,
        log_fn=log_fn,
    )
    return True


def force_kr_runtime_abstain_to_sequence_lock(
    *,
    runtime_policy: MutableMapping[str, Any],
    fname: str,
    textgrid_trust_score: float,
    alignment_weight: float,
    plan_policy: Mapping[str, Any],
    log_fn: Callable[[str], None],
) -> bool:
    if not bool(runtime_policy.get("should_abstain")):
        return False

    log_fn(
        f"[WARN] {fname}: KR v2 planner 저신뢰 "
        f"(trust={float(textgrid_trust_score):.2f}, weight={float(alignment_weight):.2f}, "
        f"coverage={float(plan_policy.get('coverage', 0.0) or 0.0):.2f}, "
        f"margin={float(plan_policy.get('margin', 0.0) or 0.0):.1f}) -> 순서 잠금 보수 자동설정"
    )
    runtime_policy["should_abstain"] = False
    runtime_policy["force_sequence_lock"] = True
    runtime_policy["low_conf_force_auto"] = True
    runtime_policy["low_conf_reasons"] = sorted(
        set(list(runtime_policy.get("low_conf_reasons") or []) + ["planner_abstain_forced_auto"])
    )
    return True


__all__ = [
    "force_kr_runtime_abstain_to_sequence_lock",
    "handle_kr_mapping_failure_with_guard",
    "has_kr_syllable_mapping_failure",
]
